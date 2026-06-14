from __future__ import annotations

from freqtrade.mt5_trade.backtest import run_backtest
from freqtrade.mt5_trade.data import MT5Bar
from freqtrade.mt5_trade.models import MT5SymbolMapping
from freqtrade.mt5_trade.sizing import PositionSizer
from freqtrade.mt5_trade.strategies import HOLD, MT5Strategy, Signal, SmaCrossStrategy


class ScriptedStrategy(MT5Strategy):
    def __init__(self, signals: list[Signal]) -> None:
        self._signals = list(signals)
        self.closed: list[str] = []

    def on_bar(self, symbol, bars):
        return self._signals.pop(0) if self._signals else HOLD

    def on_position_closed(self, symbol: str) -> None:
        self.closed.append(symbol)


def _bars(closes: list[float]) -> list[MT5Bar]:
    return [MT5Bar(time=i, open=c, high=c, low=c, close=c) for i, c in enumerate(closes)]


def test_backtest_notifies_strategy_on_stop_loss_close() -> None:
    # Long with SL at 9; the next bar dips to 8 -> SL fires outside on_bar -> callback runs.
    strategy = ScriptedStrategy([Signal("enter_long", stop_loss=9.0), HOLD])
    result = run_backtest(
        strategy, {"EURUSD": _bars([10, 8])}, default_volume=1.0, warmup_bars=10
    )

    assert result.num_trades == 1
    assert result.trades[0].exit_price == 9.0  # closed at the stop, not the bar close
    assert strategy.closed == ["EURUSD"]


def test_backtest_notifies_strategy_on_exit_signal_close() -> None:
    strategy = ScriptedStrategy([Signal("enter_long"), Signal("exit")])
    run_backtest(strategy, {"EURUSD": _bars([1, 2, 3])}, default_volume=1.0, warmup_bars=10)

    assert strategy.closed == ["EURUSD"]


def test_backtest_computes_round_trip_pnl() -> None:
    # Enter long at close=1 (bar 0), exit at close=3 (bar 2): pnl = (3-1)*1*volume.
    strategy = ScriptedStrategy([Signal("enter_long"), HOLD, Signal("exit"), HOLD])
    result = run_backtest(
        strategy, {"EURUSD": _bars([1, 2, 3, 4])}, default_volume=1.0, warmup_bars=10
    )

    assert result.num_trades == 1
    trade = result.trades[0]
    assert trade.side == "buy"
    assert trade.entry_price == 1
    assert trade.exit_price == 3
    assert trade.pnl == 2
    assert result.win_rate == 1.0


def test_backtest_reports_profit_and_return_pct() -> None:
    strategy = ScriptedStrategy([Signal("enter_long"), Signal("exit")])
    result = run_backtest(
        strategy,
        {"XAUUSD": _bars([100, 102])},
        default_volume=0.01,
        warmup_bars=10,
        starting_balance=1000,
        contract_size=100,
    )

    assert result.total_pnl == 0.02
    assert result.total_profit == 2.0
    assert result.ending_balance == 1002.0
    assert result.return_pct == 0.2


def test_backtest_uses_risk_percent_sizing() -> None:
    strategy = ScriptedStrategy([Signal("enter_long", stop_loss=90), Signal("exit")])
    sizer = PositionSizer.from_config(
        {"position_sizing": {"mode": "risk_percent", "risk_per_trade": 1.0}},
        default_lot_size=0.99,
        contract_size=100,
    )
    result = run_backtest(
        strategy,
        {"XAUUSD": _bars([100, 110])},
        default_volume=0.99,
        warmup_bars=10,
        starting_balance=1000,
        contract_size=100,
        position_sizer=sizer,
        symbol_mappings={
            "XAUUSD": MT5SymbolMapping(
                base="XAU", quote="USD", mt5_symbol="XAUUSD", min_lot=0.01, lot_step=0.01
            )
        },
    )

    assert result.trades[0].volume == 0.01
    assert result.total_profit == 10.0
    assert result.return_pct == 1.0


def test_backtest_skips_risk_entry_when_min_lot_exceeds_risk() -> None:
    strategy = ScriptedStrategy([Signal("enter_long", stop_loss=80), Signal("exit")])
    sizer = PositionSizer.from_config(
        {"position_sizing": {"mode": "risk_percent", "risk_per_trade": 0.5}},
        default_lot_size=0.01,
        contract_size=100,
    )

    result = run_backtest(
        strategy,
        {"XAUUSD": _bars([100, 110])},
        warmup_bars=10,
        starting_balance=1000,
        contract_size=100,
        position_sizer=sizer,
    )

    assert result.num_trades == 0
    assert result.skipped_entries == 1


def test_backtest_closes_position_at_stop_loss() -> None:
    strategy = ScriptedStrategy([Signal("enter_long", stop_loss=95), HOLD])
    result = run_backtest(
        strategy,
        {"XAUUSD": [MT5Bar(0, 100, 100, 100, 100), MT5Bar(1, 98, 101, 94, 99)]},
        default_volume=1.0,
        warmup_bars=10,
    )

    assert result.num_trades == 1
    assert result.trades[0].exit_price == 95
    assert result.trades[0].pnl == -5


def test_backtest_short_trade_profits_when_price_falls() -> None:
    # Short at close=5 (bar 0), exit at close=2 (bar 2): pnl = (2-5)*-1*1 = 3.
    strategy = ScriptedStrategy([Signal("enter_short"), HOLD, Signal("exit")])
    result = run_backtest(
        strategy, {"EURUSD": _bars([5, 4, 2])}, default_volume=1.0, warmup_bars=10
    )

    assert result.num_trades == 1
    assert result.trades[0].side == "sell"
    assert result.trades[0].pnl == 3


def test_backtest_marks_open_position_out_at_end() -> None:
    # Enter long, never exit: closed at the final bar's close when close_at_end is set.
    strategy = ScriptedStrategy([Signal("enter_long")])
    result = run_backtest(
        strategy, {"EURUSD": _bars([1, 2, 5])}, default_volume=1.0, warmup_bars=10
    )

    assert result.num_trades == 1
    assert result.trades[0].exit_price == 5
    assert result.trades[0].pnl == 4


def test_backtest_limit_entry_fills_when_price_is_touched() -> None:
    # Buy-limit at 8 rests until a bar dips to it (bar index 2), then exits at the next close.
    strategy = ScriptedStrategy(
        [Signal("enter_long", order_kind="limit", price=8), HOLD, HOLD, Signal("exit")]
    )
    result = run_backtest(
        strategy, {"EURUSD": _bars([10, 9, 8, 7])}, default_volume=1.0, warmup_bars=10
    )

    assert result.num_trades == 1
    assert result.trades[0].entry_price == 8
    assert result.trades[0].exit_price == 7
    assert result.trades[0].pnl == -1


def test_backtest_limit_entry_cancelled_before_fill() -> None:
    # The strategy exits before price ever reaches the resting limit -> no trade.
    strategy = ScriptedStrategy(
        [Signal("enter_long", order_kind="limit", price=8), Signal("exit")]
    )
    result = run_backtest(
        strategy, {"EURUSD": _bars([10, 9])}, default_volume=1.0, warmup_bars=10
    )

    assert result.num_trades == 0


def test_backtest_buy_stop_fills_on_breakout() -> None:
    # Buy-stop at 12 fills when a bar trades up through it (bar index 2).
    strategy = ScriptedStrategy([Signal("enter_long", order_kind="stop", price=12)])
    result = run_backtest(
        strategy, {"EURUSD": _bars([10, 11, 13])}, default_volume=1.0, warmup_bars=10
    )

    assert result.num_trades == 1
    assert result.trades[0].entry_price == 12
    assert result.trades[0].pnl == 1


def test_backtest_runs_with_real_sma_strategy() -> None:
    closes = [10] * 5 + [11, 13, 15, 17] + [15, 12, 9, 7]
    result = run_backtest(
        SmaCrossStrategy(fast=2, slow=3),
        {"EURUSD": _bars(closes)},
        default_volume=0.1,
        warmup_bars=50,
    )

    # The up-then-down series should produce at least one completed trade.
    assert result.num_trades >= 1


def test_backtest_scale_out_takes_partial_then_runs_remainder() -> None:
    strategy = ScriptedStrategy(
        [Signal("enter_long", stop_loss=9.0, tp1=12.0, tp1_close_fraction=0.5,
                move_sl_to_breakeven=True)]
    )
    result = run_backtest(
        strategy, {"EURUSD": _bars([10, 13, 11])}, default_volume=1.0, warmup_bars=10
    )

    assert result.num_trades == 2
    # TP1 leg: half the position closed at TP1 (12).
    assert result.trades[0].volume == 0.5
    assert result.trades[0].exit_price == 12.0
    assert result.trades[0].pnl == 1.0
    # Runner leg: the remaining half marked out at the final close.
    assert result.trades[1].volume == 0.5
    assert result.trades[1].exit_price == 11.0
    # Only the full (runner) close notifies the strategy; the partial does not.
    assert strategy.closed == ["EURUSD"]


def test_backtest_scale_out_runner_stops_at_breakeven() -> None:
    strategy = ScriptedStrategy(
        [Signal("enter_long", stop_loss=9.0, tp1=12.0, tp1_close_fraction=0.5,
                move_sl_to_breakeven=True)]
    )
    # After TP1, price falls below the original 9 stop area — but the stop was moved to entry.
    result = run_backtest(
        strategy, {"EURUSD": _bars([10, 13, 9.5])}, default_volume=1.0, warmup_bars=10
    )

    assert result.num_trades == 2
    assert result.trades[0].exit_price == 12.0  # TP1 partial
    # Remainder exits at breakeven (entry = 10), not the original 9 stop.
    assert result.trades[1].exit_price == 10.0
    assert result.trades[1].pnl == 0.0


def test_backtest_scale_out_full_stop_before_tp1() -> None:
    # Price hits the original stop before ever reaching TP1 -> whole position closed at the stop.
    strategy = ScriptedStrategy(
        [Signal("enter_long", stop_loss=9.0, tp1=12.0, tp1_close_fraction=0.5,
                move_sl_to_breakeven=True)]
    )
    result = run_backtest(
        strategy, {"EURUSD": _bars([10, 8])}, default_volume=1.0, warmup_bars=10
    )

    assert result.num_trades == 1
    assert result.trades[0].volume == 1.0
    assert result.trades[0].exit_price == 9.0


def test_backtest_scale_out_snaps_off_grid_volume() -> None:
    strategy = ScriptedStrategy(
        [Signal("enter_long", stop_loss=9.0, tp1=12.0, tp1_close_fraction=0.5,
                move_sl_to_breakeven=True)]
    )
    result = run_backtest(
        strategy,
        {"XAUUSD": _bars([10, 13, 11])},
        default_volume=0.03,
        warmup_bars=10,
        symbol_mappings={
            "XAUUSD": MT5SymbolMapping(
                base="XAU", quote="USD", mt5_symbol="XAUUSD", min_lot=0.01, lot_step=0.01
            )
        },
    )

    assert result.num_trades == 2
    # 0.03 split 50% on a 0.01 grid -> close 0.01 (not 0.015), runner 0.02; both on-grid.
    assert result.trades[0].volume == 0.01
    assert result.trades[1].volume == 0.02


class _EnterExitStrategy(MT5Strategy):
    """Enters long when close == 100, exits when close == 110 (per symbol, stateless)."""

    def on_bar(self, symbol, bars):
        close = bars[-1].close
        if close == 100:
            return Signal("enter_long", stop_loss=95)
        if close == 110:
            return Signal("exit")
        return HOLD


def test_backtest_multi_symbol_sizes_on_time_ordered_balance() -> None:
    sizer = PositionSizer.from_config(
        {"position_sizing": {"mode": "risk_percent", "risk_per_trade": 10.0}},
        default_lot_size=0.01,
        contract_size=1.0,
    )
    # A wins (+200) but only closes at t=4 — after B enters at t=2. B must be sized on the
    # balance at t=2 (still 1000 -> vol 20), not A's end-of-period balance (1200 -> vol 24).
    data = {
        "A": [MT5Bar(0, 100, 100, 100, 100), MT5Bar(4, 110, 110, 110, 110)],
        "B": [MT5Bar(2, 100, 100, 100, 100), MT5Bar(6, 110, 110, 110, 110)],
    }
    result = run_backtest(
        _EnterExitStrategy(), data,
        starting_balance=1000, contract_size=1.0, position_sizer=sizer, warmup_bars=10,
    )

    b_trade = next(t for t in result.trades if t.symbol == "B")
    assert b_trade.volume == 20.0


def test_backtest_multi_symbol_closes_in_global_time_order() -> None:
    # B opens later but closes earlier (t=4) than A (t=6); trades must be in time order.
    data = {
        "A": [MT5Bar(0, 100, 100, 100, 100), MT5Bar(6, 110, 110, 110, 110)],
        "B": [MT5Bar(2, 100, 100, 100, 100), MT5Bar(4, 110, 110, 110, 110)],
    }
    result = run_backtest(_EnterExitStrategy(), data, default_volume=1.0, warmup_bars=10)

    assert [t.exit_time for t in result.trades] == [4, 6]


def test_backtest_multi_symbol_close_at_end_updates_balance_in_time_order() -> None:
    sizer = PositionSizer.from_config(
        {"position_sizing": {"mode": "risk_percent", "risk_per_trade": 10.0}},
        default_lot_size=0.01,
        contract_size=1.0,
    )
    # A is marked out (+100) at its last bar t=2; B enters later at t=4 and must be sized on the
    # post-A-close balance (1100 -> vol 22), not the pre-close balance (1000 -> vol 20).
    data = {
        "A": [MT5Bar(0, 100, 100, 100, 100), MT5Bar(2, 105, 105, 105, 105)],
        "B": [MT5Bar(4, 100, 100, 100, 100), MT5Bar(6, 110, 110, 110, 110)],
    }
    result = run_backtest(
        _EnterExitStrategy(), data,
        starting_balance=1000, contract_size=1.0, position_sizer=sizer, warmup_bars=10,
    )

    b_trade = next(t for t in result.trades if t.symbol == "B")
    assert b_trade.volume == 22.0


def test_backtest_normalizes_explicit_off_grid_volume() -> None:
    strategy = ScriptedStrategy([Signal("enter_long", volume=0.025), Signal("exit")])
    result = run_backtest(
        strategy, {"XAUUSD": _bars([100, 110])}, default_volume=0.01, warmup_bars=10,
        symbol_mappings={
            "XAUUSD": MT5SymbolMapping(
                base="XAU", quote="USD", mt5_symbol="XAUUSD", min_lot=0.01, lot_step=0.01
            )
        },
    )

    assert result.num_trades == 1
    # 0.025 floored to the 0.01 lot grid -> 0.02.
    assert result.trades[0].volume == 0.02


def test_backtest_pending_fill_resolves_sl_on_same_bar() -> None:
    strategy = ScriptedStrategy([Signal("enter_long", order_kind="limit", price=8, stop_loss=7)])
    # bar 1 dips to 6.9: the buy-limit (8) fills, then the stop (7) is hit in the same candle.
    data = {"EURUSD": [MT5Bar(0, 10, 10, 10, 10), MT5Bar(1, 9, 9, 6.9, 7.5)]}

    result = run_backtest(strategy, data, default_volume=1.0, warmup_bars=10)

    assert result.num_trades == 1
    assert result.trades[0].entry_price == 8
    assert result.trades[0].exit_price == 7   # stopped on the fill bar (not the bar close 7.5)
    assert result.trades[0].exit_time == 1


def test_backtest_expired_pending_does_not_fill() -> None:
    strategy = ScriptedStrategy([Signal("enter_long", order_kind="limit", price=8, expiration=30)])
    # The touch happens at t=60, past the order's expiry (30), so it must not fill.
    data = {"EURUSD": [MT5Bar(0, 10, 10, 10, 10), MT5Bar(60, 9, 9, 7, 8)]}

    result = run_backtest(strategy, data, default_volume=1.0, warmup_bars=10)

    assert result.num_trades == 0


def test_backtest_pending_fills_before_expiry() -> None:
    strategy = ScriptedStrategy([Signal("enter_long", order_kind="limit", price=8, expiration=100)])
    # The touch at t=60 is before the order's expiry (100), so it fills normally.
    data = {"EURUSD": [MT5Bar(0, 10, 10, 10, 10), MT5Bar(60, 9, 9, 7, 8)]}

    result = run_backtest(strategy, data, default_volume=1.0, warmup_bars=10)

    assert result.num_trades == 1
    assert result.trades[0].entry_price == 8
