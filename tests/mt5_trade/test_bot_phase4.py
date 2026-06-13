from __future__ import annotations

from freqtrade.mt5_trade.bot import MT5ForexBot
from freqtrade.mt5_trade.data import MT5Bar, ReplayDataFeed
from freqtrade.mt5_trade.models import MT5BotConfig, MT5OrderResult
from freqtrade.mt5_trade.persistence import MT5TradeStore
from freqtrade.mt5_trade.strategy import HOLD, MT5Strategy, Signal


class FakeBridge:
    def __init__(self, result: MT5OrderResult) -> None:
        self._result = result
        self.orders: list = []
        self.sltp: list = []
        self.closed = False

    def submit_order(self, order):
        self.orders.append(order)
        return self._result

    def modify_sltp(self, symbol, stop_loss, take_profit):
        self.sltp.append((symbol, stop_loss, take_profit))
        return MT5OrderResult(accepted=True, order_id=None)

    def cancel_order(self, ticket):
        return MT5OrderResult(accepted=True, order_id=str(ticket))

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


def _bot(bridge: FakeBridge, strategy: MT5Strategy, store: MT5TradeStore, vol: float = 0.05):
    cfg = MT5BotConfig(symbols=("EURUSD",), warmup_bars=5, poll_interval=1.0)
    return MT5ForexBot(bridge, _feed(), strategy, store, cfg, default_volume=vol)


def test_bot_does_not_record_pending_order_as_position() -> None:
    bridge = FakeBridge(MT5OrderResult(accepted=True, order_id="p1", is_pending=True))
    store = MT5TradeStore(":memory:")
    signal = Signal("enter_long", order_kind="limit", price=1.2)
    bot = _bot(bridge, ScriptedStrategy([signal]), store)

    bot.run_once()

    # The pending order was sent and recorded, but it is not a held position yet.
    assert store.order_count() == 1
    assert store.open_positions() == {}


def test_bot_tracks_partial_fill_volume() -> None:
    bridge = FakeBridge(
        MT5OrderResult(accepted=True, order_id="d1", filled_volume=0.03)
    )
    store = MT5TradeStore(":memory:")
    bot = _bot(bridge, ScriptedStrategy([Signal("enter_long")]), store, vol=0.05)

    bot.run_once()

    # Requested 0.05 but only 0.03 filled -> the tracked position reflects the fill.
    assert store.open_positions()["EURUSD"].volume == 0.03


def test_bot_builds_limit_order_with_price() -> None:
    bridge = FakeBridge(MT5OrderResult(accepted=True, order_id="d1", filled_volume=0.05))
    store = MT5TradeStore(":memory:")
    signal = Signal("enter_long", order_kind="limit", price=1.20)
    bot = _bot(bridge, ScriptedStrategy([signal]), store)

    bot.run_once()

    assert bridge.orders[0].order_kind == "limit"
    assert bridge.orders[0].price == 1.20


def test_bot_pending_entry_carries_sl_tp_on_order() -> None:
    bridge = FakeBridge(MT5OrderResult(accepted=True, order_id="p1", is_pending=True))
    store = MT5TradeStore(":memory:")
    signal = Signal("enter_long", order_kind="limit", price=1.2, stop_loss=1.1, take_profit=1.5)
    bot = _bot(bridge, ScriptedStrategy([signal]), store)

    bot.run_once()

    # The pending order carries SL/TP so the broker applies them when it fills.
    order = bridge.orders[0]
    assert order.order_kind == "limit"
    assert order.stop_loss == 1.1
    assert order.take_profit == 1.5


def test_bot_market_entry_sets_sl_tp_via_modify_not_on_order() -> None:
    bridge = FakeBridge(MT5OrderResult(accepted=True, order_id="m1"))
    store = MT5TradeStore(":memory:")
    signal = Signal("enter_long", stop_loss=0.9, take_profit=1.4)
    bot = _bot(bridge, ScriptedStrategy([signal]), store)

    bot.run_once()

    # Market entries keep SL/TP off the order and apply them after the fill via modify_sltp.
    assert bridge.orders[0].stop_loss is None
    assert bridge.orders[0].take_profit is None
    assert bridge.sltp == [("EURUSD", 0.9, 1.4)]
