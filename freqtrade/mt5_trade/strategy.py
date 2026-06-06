from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

from freqtrade.mt5_trade.data import MT5Bar
from freqtrade.mt5_trade.models import OrderKind


SignalAction = Literal["enter_long", "enter_short", "exit", "hold"]


@dataclass(frozen=True)
class Signal:
    action: SignalAction = "hold"
    volume: float | None = None
    # Entry order type. A limit/stop entry requires an explicit price and rests as a pending
    # order until the market reaches it; market entries fill immediately.
    order_kind: OrderKind = "market"
    price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    comment: str | None = None
    # Epoch-seconds expiry for a pending entry (broker auto-cancels at that time).
    expiration: int | None = None

    def __post_init__(self) -> None:
        if self.order_kind != "market" and self.price is None:
            raise ValueError(f"{self.order_kind} entry signal requires an explicit price.")


# Shared singleton for "do nothing" to avoid allocating on every bar.
HOLD = Signal(action="hold")


class MT5Strategy(ABC):
    """Decides an action from the latest completed bars of a single symbol."""

    @abstractmethod
    def on_bar(self, symbol: str, bars: list[MT5Bar]) -> Signal:
        """Return a Signal given the most recent bars (oldest first)."""


def _sma(values: list[float], length: int) -> float:
    window = values[-length:]
    return sum(window) / len(window)


class SmaCrossStrategy(MT5Strategy):
    """
    Simple moving-average crossover demo strategy.

    Enters long when the fast SMA crosses above the slow SMA, short when it crosses below.
    Stateless: the crossover is detected from the last two bars so the bot's position manager
    owns entry/exit bookkeeping.
    """

    def __init__(self, fast: int = 10, slow: int = 30) -> None:
        if fast < 1 or slow < 1:
            raise ValueError("SMA lengths must be >= 1.")
        if fast >= slow:
            raise ValueError(f"fast ({fast}) must be shorter than slow ({slow}).")
        self.fast = fast
        self.slow = slow

    def on_bar(self, symbol: str, bars: list[MT5Bar]) -> Signal:
        # Need slow+1 bars to compare this bar's relationship against the previous bar's.
        if len(bars) < self.slow + 1:
            return HOLD

        closes = [bar.close for bar in bars]
        prev_fast = _sma(closes[:-1], self.fast)
        prev_slow = _sma(closes[:-1], self.slow)
        curr_fast = _sma(closes, self.fast)
        curr_slow = _sma(closes, self.slow)

        crossed_up = prev_fast <= prev_slow and curr_fast > curr_slow
        crossed_down = prev_fast >= prev_slow and curr_fast < curr_slow

        if crossed_up:
            return Signal(action="enter_long", comment=f"sma{self.fast}x{self.slow} up")
        if crossed_down:
            return Signal(action="enter_short", comment=f"sma{self.fast}x{self.slow} down")
        return HOLD
