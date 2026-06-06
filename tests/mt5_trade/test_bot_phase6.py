from __future__ import annotations

from freqtrade.mt5_trade.bot import MT5ForexBot
from freqtrade.mt5_trade.data import MT5Bar, ReplayDataFeed
from freqtrade.mt5_trade.models import MT5BotConfig, MT5OrderResult
from freqtrade.mt5_trade.persistence import MT5TradeStore
from freqtrade.mt5_trade.strategy import HOLD, MT5Strategy, Signal


class PendingBridge:
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


def test_bot_expires_stale_pending_order() -> None:
    bridge = PendingBridge()
    store = MT5TradeStore(":memory:")
    cfg = MT5BotConfig(symbols=("EURUSD",), warmup_bars=5, poll_interval=1.0, pending_expiry=2)
    strategy = ScriptedStrategy([Signal("enter_long", order_kind="limit", price=1.2)])
    bot = MT5ForexBot(bridge, _feed(), strategy, store, cfg, default_volume=0.05)

    bot.run_once()  # iter 1: places pending (age 0)
    assert bridge.cancelled == []
    bot.run_once()  # iter 2: age 1 < 2, still resting
    assert bridge.cancelled == []
    bot.run_once()  # iter 3: age 2 >= 2 -> expired and cancelled

    assert bridge.cancelled == [901]


def test_bot_keeps_pending_when_expiry_disabled() -> None:
    bridge = PendingBridge()
    store = MT5TradeStore(":memory:")
    cfg = MT5BotConfig(symbols=("EURUSD",), warmup_bars=5, poll_interval=1.0, pending_expiry=0)
    strategy = ScriptedStrategy([Signal("enter_long", order_kind="limit", price=1.2)])
    bot = MT5ForexBot(bridge, _feed(), strategy, store, cfg, default_volume=0.05)

    for _ in range(5):
        bot.run_once()

    assert bridge.cancelled == []


def test_store_records_order_metadata() -> None:
    store = MT5TradeStore(":memory:")
    from freqtrade.mt5_trade.models import MT5OrderRequest

    order = MT5OrderRequest(
        symbol="EURUSD", side="buy", volume=0.01,
        order_kind="limit", price=1.075, expiration=1700000000,
    )
    store.record_order(order, MT5OrderResult(accepted=True, order_id="1"))

    row = store._conn.execute(
        "SELECT order_kind, price, expiration FROM mt5_orders"
    ).fetchone()
    assert row["order_kind"] == "limit"
    assert row["price"] == 1.075
    assert row["expiration"] == 1700000000
