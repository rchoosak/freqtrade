from __future__ import annotations

from freqtrade.mt5_trade.bot import MT5ForexBot
from freqtrade.mt5_trade.data import MT5Bar, ReplayDataFeed
from freqtrade.mt5_trade.models import MT5BotConfig, MT5OrderResult
from freqtrade.mt5_trade.persistence import MT5TradeStore
from freqtrade.mt5_trade.strategies import HOLD, MT5Strategy, Signal


class RejectCloseBridge:
    """Rejects the closing leg (volume 1.0) but would accept the opposite open (volume 0.5)."""

    def __init__(self) -> None:
        self.orders: list = []
        self.cancelled: list = []
        self.closed = False

    def submit_order(self, order):
        self.orders.append(order)
        accepted = order.volume != 1.0
        return MT5OrderResult(accepted=accepted, order_id="ok" if accepted else None)

    def modify_sltp(self, symbol, stop_loss, take_profit):
        return MT5OrderResult(accepted=True, order_id=None)

    def cancel_order(self, ticket):
        self.cancelled.append(ticket)
        return MT5OrderResult(accepted=False, order_id=None, message="rejected")

    def broker_positions(self):
        return None

    def broker_orders(self):
        return None

    def close(self):
        self.closed = True


class ScriptedStrategy(MT5Strategy):
    def __init__(self, signals: list[Signal]) -> None:
        self._signals = list(signals)
        self.closed: list[str] = []

    def on_bar(self, symbol, bars):
        return self._signals.pop(0) if self._signals else HOLD

    def on_position_closed(self, symbol: str) -> None:
        self.closed.append(symbol)


def _bot(bridge, strategy, store, default_volume=0.5) -> MT5ForexBot:
    feed = ReplayDataFeed({"EURUSD": [MT5Bar(0, 1, 1, 1, 1)]}, warmup=1)
    cfg = MT5BotConfig(symbols=("EURUSD",), warmup_bars=1, poll_interval=1.0)
    return MT5ForexBot(bridge, feed, strategy, store, cfg, default_volume=default_volume)


def test_market_reversal_does_not_open_opposite_when_close_rejected() -> None:
    store = MT5TradeStore(":memory:")
    store.open_position("EURUSD", "buy", 1.0, 1.10)  # pre-existing long
    bridge = RejectCloseBridge()
    bot = _bot(bridge, ScriptedStrategy([Signal("enter_short")]), store)

    bot.run_once()

    # Only the (rejected) close was attempted; the opposite open was aborted.
    assert len(bridge.orders) == 1
    assert bridge.orders[0].volume == 1.0
    # Position is unchanged because the close failed.
    assert bot._positions["EURUSD"] == ("buy", 1.0)


def test_pending_reversal_does_not_open_opposite_when_cancel_rejected() -> None:
    store = MT5TradeStore(":memory:")
    bridge = RejectCloseBridge()
    bot = _bot(bridge, ScriptedStrategy([Signal("enter_short")]), store)
    # A resting buy-limit occupies the slot.
    bot._pendings["EURUSD"] = ("buy", 1.0, 123)
    bot._pending_placed["EURUSD"] = 0

    bot.run_once()

    # Cancel was attempted and rejected -> the opposite open must be aborted.
    assert bridge.cancelled == [123]
    assert len(bridge.orders) == 0
    # The pending order is still tracked (broker still has it live).
    assert "EURUSD" in bot._pendings
