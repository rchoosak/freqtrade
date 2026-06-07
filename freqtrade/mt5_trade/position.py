from __future__ import annotations

from dataclasses import dataclass

from freqtrade.mt5_trade.models import OrderKind, OrderSide
from freqtrade.mt5_trade.strategy import Signal


# Action -> the market order side that opens that position.
_ENTRY_SIDE: dict[str, OrderSide] = {"enter_long": "buy", "enter_short": "sell"}
# A position side -> the opposite market order side that closes it.
_CLOSE_SIDE: dict[str, OrderSide] = {"buy": "sell", "sell": "buy"}


@dataclass(frozen=True)
class OrderIntent:
    """A single order to send and the position it leaves behind."""

    side: OrderSide
    volume: float
    # Resulting position after the order fills: ("buy"|"sell", volume) or None when flat.
    result: tuple[OrderSide, float] | None
    reason: str
    order_kind: OrderKind = "market"
    price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    expiration: int | None = None


def plan_transitions(
    current: tuple[OrderSide, float] | None,
    signal: Signal,
    default_volume: float,
) -> list[OrderIntent]:
    """
    Translate a strategy signal + current position into the orders needed to reach the target.

    Single position per symbol: a same-direction entry is a no-op (no stacking), an opposite
    entry reverses (close then open), and an exit closes any open position. Shared by the live
    bot and the backtester so both apply identical position semantics.
    """
    if signal.action == "hold":
        return []

    if signal.action == "exit":
        if current is None:
            return []
        return [_close(current, "exit")]

    desired = _ENTRY_SIDE[signal.action]
    if current is not None and current[0] == desired:
        # Already in the desired direction; do not stack another position.
        return []

    intents: list[OrderIntent] = []
    if current is not None:
        intents.append(_close(current, "reverse"))

    volume = signal.volume if signal.volume is not None else default_volume
    intents.append(
        OrderIntent(
            side=desired,
            volume=volume,
            result=(desired, volume),
            reason="open",
            order_kind=signal.order_kind,
            price=signal.price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            expiration=signal.expiration,
        )
    )
    return intents


def _close(current: tuple[OrderSide, float], reason: str) -> OrderIntent:
    open_side, volume = current
    return OrderIntent(side=_CLOSE_SIDE[open_side], volume=volume, result=None, reason=reason)
