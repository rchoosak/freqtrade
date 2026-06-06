from __future__ import annotations

import pytest

from freqtrade.mt5_trade.data import MT5Bar
from freqtrade.mt5_trade.strategy import SmaCrossStrategy


def _bars(closes: list[float]) -> list[MT5Bar]:
    return [MT5Bar(time=i, open=c, high=c, low=c, close=c) for i, c in enumerate(closes)]


def test_sma_cross_rejects_bad_lengths() -> None:
    with pytest.raises(ValueError, match="shorter than slow"):
        SmaCrossStrategy(fast=30, slow=10)


def test_sma_cross_holds_without_enough_bars() -> None:
    strategy = SmaCrossStrategy(fast=2, slow=3)
    # slow + 1 = 4 bars required.
    assert strategy.on_bar("EURUSD", _bars([1, 2, 3])).action == "hold"


def test_sma_cross_enters_long_on_cross_up() -> None:
    strategy = SmaCrossStrategy(fast=2, slow=3)
    # Flat then a sharp rise so the fast SMA crosses above the slow SMA on the last bar.
    signal = strategy.on_bar("EURUSD", _bars([10, 10, 10, 20]))
    assert signal.action == "enter_long"


def test_sma_cross_enters_short_on_cross_down() -> None:
    strategy = SmaCrossStrategy(fast=2, slow=3)
    # Flat then a sharp drop so the fast SMA crosses below the slow SMA on the last bar.
    signal = strategy.on_bar("EURUSD", _bars([10, 10, 10, 1]))
    assert signal.action == "enter_short"


def test_sma_cross_holds_when_no_cross() -> None:
    strategy = SmaCrossStrategy(fast=2, slow=3)
    # Steady uptrend already established; no fresh cross on the last bar.
    signal = strategy.on_bar("EURUSD", _bars([1, 2, 3, 4, 5, 6]))
    assert signal.action == "hold"
