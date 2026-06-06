from __future__ import annotations

from freqtrade.mt5_trade.backtest import run_backtest
from freqtrade.mt5_trade.data import MT5Bar
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
