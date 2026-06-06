from __future__ import annotations

from freqtrade.mt5_trade.bot import MT5ForexBot
from freqtrade.mt5_trade.data import MT5Bar, ReplayDataFeed
from freqtrade.mt5_trade.models import BrokerPosition, MT5BotConfig, MT5OrderResult
from freqtrade.mt5_trade.persistence import MT5TradeStore
from freqtrade.mt5_trade.strategy import HOLD, MT5Strategy


class _NoSignalStrategy(MT5Strategy):
    def on_bar(self, symbol, bars):
        return HOLD


class FakeBridge:
    """Duck-typed stand-in for MT5ExecutionBridge with controllable broker positions."""

    def __init__(self, broker_positions: list[BrokerPosition] | None) -> None:
        self._broker_positions = broker_positions
        self.orders: list = []
        self.sltp: list = []
        self.closed = False

    def submit_order(self, order):
        self.orders.append(order)
        return MT5OrderResult(accepted=True, order_id="ok")

    def modify_sltp(self, symbol, stop_loss, take_profit):
        self.sltp.append((symbol, stop_loss, take_profit))
        return MT5OrderResult(accepted=True, order_id=None)

    def broker_positions(self):
        return self._broker_positions

    def close(self):
        self.closed = True


def _bot(bridge: FakeBridge, store: MT5TradeStore) -> MT5ForexBot:
    feed = ReplayDataFeed({"EURUSD": [MT5Bar(0, 1, 1, 1, 1)]}, warmup=1)
    cfg = MT5BotConfig(symbols=("EURUSD",), warmup_bars=1, poll_interval=1.0)
    return MT5ForexBot(bridge, feed, _NoSignalStrategy(), store, cfg, default_volume=0.01)


def test_reconcile_adopts_unknown_broker_position() -> None:
    store = MT5TradeStore(":memory:")
    bridge = FakeBridge([BrokerPosition("EURUSD", "buy", 0.30, price=1.08, ticket=1)])
    bot = _bot(bridge, store)

    bot.reconcile()

    assert store.open_positions()["EURUSD"].side == "buy"
    assert store.open_positions()["EURUSD"].volume == 0.30


def test_reconcile_drops_externally_closed_position() -> None:
    store = MT5TradeStore(":memory:")
    store.open_position("EURUSD", "buy", 0.10, 1.10)
    # Broker reports flat -> the stale local position must be dropped.
    bridge = FakeBridge([])
    bot = _bot(bridge, store)

    bot.reconcile()

    assert store.open_positions() == {}


def test_reconcile_noop_in_dry_run() -> None:
    store = MT5TradeStore(":memory:")
    store.open_position("EURUSD", "buy", 0.10, 1.10)
    # broker_positions() returns None in dry-run -> reconciliation does not apply.
    bridge = FakeBridge(None)
    bot = _bot(bridge, store)

    bot.reconcile()

    assert "EURUSD" in store.open_positions()
