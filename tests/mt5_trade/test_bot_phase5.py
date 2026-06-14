from __future__ import annotations

from freqtrade.mt5_trade.bot import MT5ForexBot
from freqtrade.mt5_trade.data import MT5Bar, ReplayDataFeed
from freqtrade.mt5_trade.models import MT5BotConfig, MT5OrderResult
from freqtrade.mt5_trade.persistence import MT5TradeStore
from freqtrade.mt5_trade.strategies import HOLD, MT5Strategy, Signal


class PendingBridge:
    """Bridge whose entries are accepted as resting pending orders; records cancels."""

    def __init__(self) -> None:
        self.orders: list = []
        self.cancelled: list = []
        self._ticket = 900
        self.closed = False

    def submit_order(self, order):
        self.orders.append(order)
        self._ticket += 1
        return MT5OrderResult(accepted=True, order_id=str(self._ticket), is_pending=True)

    def cancel_order(self, ticket):
        self.cancelled.append(ticket)
        return MT5OrderResult(accepted=True, order_id=str(ticket))

    def modify_sltp(self, symbol, stop_loss, take_profit):
        return MT5OrderResult(accepted=True, order_id=None)

    def broker_positions(self):
        return None

    def broker_orders(self):
        return None

    def close(self):
        self.closed = True


class ScriptedStrategy(MT5Strategy):
    def __init__(self, signals: list[Signal]) -> None:
        self._signals = list(signals)

    def on_bar(self, symbol, bars):
        return self._signals.pop(0) if self._signals else HOLD


def _feed(n: int = 5) -> ReplayDataFeed:
    bars = [MT5Bar(time=i, open=1.0, high=1.0, low=1.0, close=1.0 + i) for i in range(n)]
    return ReplayDataFeed({"EURUSD": bars}, warmup=n)


def _bot(bridge: PendingBridge, strategy: MT5Strategy, store: MT5TradeStore) -> MT5ForexBot:
    cfg = MT5BotConfig(symbols=("EURUSD",), warmup_bars=5, poll_interval=1.0)
    return MT5ForexBot(bridge, _feed(), strategy, store, cfg, default_volume=0.05)


def test_bot_cancels_resting_pending_on_exit() -> None:
    bridge = PendingBridge()
    store = MT5TradeStore(":memory:")
    bot = _bot(
        bridge,
        ScriptedStrategy([Signal("enter_long", order_kind="limit", price=1.2), Signal("exit")]),
        store,
    )

    bot.run_once()  # places a resting pending order
    assert bridge.orders and bridge.cancelled == []
    bot.run_once()  # exit -> cancel the resting order (no market close)

    assert bridge.cancelled == [901]
    assert len(bridge.orders) == 1  # exit did not send a new order


def test_bot_does_not_restack_pending_order() -> None:
    bridge = PendingBridge()
    store = MT5TradeStore(":memory:")
    signal = Signal("enter_long", order_kind="limit", price=1.2)
    bot = _bot(bridge, ScriptedStrategy([signal, signal, signal]), store)

    bot.run_once()
    bot.run_once()
    bot.run_once()

    # The resting order occupies the slot, so repeated same-side signals don't stack orders.
    assert len(bridge.orders) == 1


def test_bot_reverses_from_pending_by_cancelling_then_opening() -> None:
    bridge = PendingBridge()
    store = MT5TradeStore(":memory:")
    bot = _bot(
        bridge,
        ScriptedStrategy(
            [Signal("enter_long", order_kind="limit", price=1.2), Signal("enter_short")]
        ),
        store,
    )

    bot.run_once()  # resting buy-limit
    bot.run_once()  # reverse: cancel the buy-limit, then place a sell entry

    assert bridge.cancelled == [901]
    assert len(bridge.orders) == 2
    assert bridge.orders[1].side == "sell"
