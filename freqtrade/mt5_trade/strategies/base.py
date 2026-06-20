from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, get_args

from freqtrade.exceptions import OperationalException
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
    # Scale-out / breakeven plan. When ``tp1`` is set, the position manager closes
    # ``tp1_close_fraction`` of the position at ``tp1`` and (optionally) moves the stop to the
    # entry price, letting the remainder run to a trend-based exit. Independent of ``take_profit``
    # (which is a single full-position target).
    tp1: float | None = None
    tp1_close_fraction: float | None = None
    move_sl_to_breakeven: bool = False

    def __post_init__(self) -> None:
        # Literal annotations are not enforced at runtime; validate at the source (the strategy)
        # so a bad value fails loud here instead of as a KeyError in plan_transitions or as
        # divergent backtest/live behaviour for an unknown order_kind.
        if self.action not in get_args(SignalAction):
            allowed = ", ".join(get_args(SignalAction))
            raise ValueError(f"Invalid action {self.action!r}. Expected one of: {allowed}.")
        if self.order_kind not in get_args(OrderKind):
            allowed = ", ".join(get_args(OrderKind))
            raise ValueError(f"Invalid order_kind {self.order_kind!r}. Expected one of: {allowed}.")
        if self.volume is not None and self.volume <= 0:
            raise ValueError(f"Invalid volume {self.volume!r}. Expected a positive lot size.")
        if self.order_kind != "market" and self.price is None:
            raise ValueError(f"{self.order_kind} entry signal requires an explicit price.")
        if self.tp1_close_fraction is not None and not 0 < self.tp1_close_fraction < 1:
            raise ValueError("tp1_close_fraction must be between 0 and 1 (exclusive).")
        if self.tp1 is not None and self.tp1_close_fraction is None:
            raise ValueError("tp1 requires tp1_close_fraction.")
        if self.tp1 is not None and self.order_kind != "market":
            raise ValueError("tp1 scale-out is only supported for market entry signals.")


# Shared singleton for "do nothing" to avoid allocating on every bar.
HOLD = Signal(action="hold")


class MT5Strategy(ABC):
    """Decides an action from the latest completed bars of a single symbol."""

    @property
    def required_timeframe(self) -> str | None:
        """Timeframe the feed must provide, or ``None`` when any timeframe is supported."""
        return None

    @property
    def minimum_bars(self) -> int:
        """Minimum configured warmup required before this strategy can produce a signal."""
        return 1

    @abstractmethod
    def on_bar(self, symbol: str, bars: list[MT5Bar]) -> Signal:
        """Return a Signal given the most recent bars (oldest first)."""

    def on_position_closed(self, symbol: str) -> None:  # noqa: B027  # optional override
        """
        Notify the strategy that ``symbol``'s position was closed outside of ``on_bar`` — by a
        stop-loss/take-profit, the broker, or reconciliation. Stateful strategies override this
        to reset per-symbol tracking so the next setup isn't masked by stale internal state.
        No-op by default.
        """


def validate_strategy_runtime(
    strategy: MT5Strategy,
    *,
    timeframe: str | None,
    warmup_bars: int,
) -> None:
    """Fail fast when runtime settings cannot satisfy a strategy's data requirements."""
    required_timeframe = strategy.required_timeframe
    minimum_bars = strategy.minimum_bars
    if minimum_bars < 1:
        raise OperationalException(
            f"{strategy.__class__.__name__}.minimum_bars must be >= 1, got {minimum_bars}."
        )
    if (
        timeframe is not None
        and required_timeframe is not None
        and timeframe.upper() != required_timeframe.upper()
    ):
        raise OperationalException(
            f"{strategy.__class__.__name__} requires timeframe {required_timeframe}, "
            f"but the runtime is configured for {timeframe}."
        )
    if warmup_bars < minimum_bars:
        raise OperationalException(
            f"{strategy.__class__.__name__} requires warmup_bars >= {minimum_bars}, "
            f"but the runtime is configured for {warmup_bars}."
        )


def _validate_positive_int(name: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{name} must be >= 1.")
