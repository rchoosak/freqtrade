from __future__ import annotations

from freqtrade.mt5_trade.backtest import run_backtest
from freqtrade.mt5_trade.data import MT5Bar
from freqtrade.mt5_trade.models import MT5SymbolMapping
from freqtrade.mt5_trade.sizing import PositionSizer
from freqtrade.mt5_trade.strategy import HOLD, MT5Strategy, Signal, SmaCrossStrategy


class ScriptedStrategy(MT5Strategy):
    def __init__(self, signals: list[Signal]) -> None:
        self._signals = list(signals)

    def on_bar(self, symbol, bars):
        return self._signals.pop(0) if self._signals else HOLD


def _bars(closes: list[float]) -> list[MT5Bar]:
    return [MT5Bar(time=i, open=c, high=c, low=c, close=c) for i, c in enumerate(closes)]


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
