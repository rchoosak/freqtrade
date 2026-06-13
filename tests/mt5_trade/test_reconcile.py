from __future__ import annotations

import pytest

from freqtrade.exceptions import OperationalException
from freqtrade.mt5_trade.bot import MT5ForexBot
from freqtrade.mt5_trade.data import MT5Bar, ReplayDataFeed
from freqtrade.mt5_trade.models import (
    BrokerOrder,
    BrokerPosition,
    MT5BotConfig,
    MT5OrderResult,
)
from freqtrade.mt5_trade.persistence import MT5TradeStore
from freqtrade.mt5_trade.strategy import HOLD, MT5Strategy


class _NoSignalStrategy(MT5Strategy):
    def __init__(self) -> None:
        self.closed: list[str] = []

    def on_bar(self, symbol, bars):
        return HOLD

    def on_position_closed(self, symbol: str) -> None:
        self.closed.append(symbol)


class FakeBridge:
    """Duck-typed stand-in for MT5ExecutionBridge with controllable broker state."""

    def __init__(self, broker_positions: list[BrokerPosition] | None, broker_orders=None) -> None:
        self._broker_positions = broker_positions
        self._broker_orders = broker_orders
        self.orders: list = []
        self.sltp: list = []
        self.cancelled: list = []
        self.closed = False

    def submit_order(self, order):
        self.orders.append(order)
        return MT5OrderResult(accepted=True, order_id="ok")

    def modify_sltp(self, symbol, stop_loss, take_profit):
        self.sltp.append((symbol, stop_loss, take_profit))
        return MT5OrderResult(accepted=True, order_id=None)

    def cancel_order(self, ticket):
        self.cancelled.append(ticket)
        return MT5OrderResult(accepted=True, order_id=str(ticket))

    def broker_positions(self):
        return self._broker_positions

    def broker_orders(self):
        return self._broker_orders

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
    assert store.open_positions()["EURUSD"].entry_price == 1.08
    assert store.open_positions()["EURUSD"].ticket == 1


def test_reconcile_rejects_duplicate_broker_positions_for_symbol() -> None:
    store = MT5TradeStore(":memory:")
    bridge = FakeBridge(
        [
            BrokerPosition("EURUSD", "buy", 0.10, price=1.08, ticket=1),
            BrokerPosition("EURUSD", "sell", 0.10, price=1.09, ticket=2),
        ],
        [],
    )
    bot = _bot(bridge, store)

    with pytest.raises(OperationalException, match="Multiple broker positions"):
        bot.reconcile()


def test_reconcile_drops_externally_closed_position() -> None:
    store = MT5TradeStore(":memory:")
    store.open_position("EURUSD", "buy", 0.10, 1.10)
    # Broker reports flat -> the stale local position must be dropped.
    bridge = FakeBridge([])
    bot = _bot(bridge, store)

    bot.reconcile()

    assert store.open_positions() == {}


def test_reconcile_drop_notifies_strategy() -> None:
    store = MT5TradeStore(":memory:")
    store.open_position("EURUSD", "buy", 0.10, 1.10)
    bridge = FakeBridge([])  # broker is flat -> external close
    strategy = _NoSignalStrategy()
    feed = ReplayDataFeed({"EURUSD": [MT5Bar(0, 1, 1, 1, 1)]}, warmup=1)
    cfg = MT5BotConfig(symbols=("EURUSD",), warmup_bars=1, poll_interval=1.0)
    bot = MT5ForexBot(bridge, feed, strategy, store, cfg, default_volume=0.01)

    bot.reconcile()

    assert strategy.closed == ["EURUSD"]


def test_reconcile_noop_in_dry_run() -> None:
    store = MT5TradeStore(":memory:")
    store.open_position("EURUSD", "buy", 0.10, 1.10)
    # broker_positions()/broker_orders() return None in dry-run -> reconciliation does not apply.
    bridge = FakeBridge(None, None)
    bot = _bot(bridge, store)

    bot.reconcile()

    assert "EURUSD" in store.open_positions()


def test_reconcile_adopts_broker_pending_order() -> None:
    store = MT5TradeStore(":memory:")
    bridge = FakeBridge([], [BrokerOrder("EURUSD", "buy", 0.10, price=1.07, ticket=5)])
    bot = _bot(bridge, store)

    bot.reconcile()

    assert bot._pendings["EURUSD"] == ("buy", 0.10, 5)


def test_reconcile_rejects_duplicate_broker_orders_for_symbol() -> None:
    store = MT5TradeStore(":memory:")
    bridge = FakeBridge(
        [],
        [
            BrokerOrder("EURUSD", "buy", 0.10, price=1.07, ticket=5),
            BrokerOrder("EURUSD", "sell", 0.10, price=1.06, ticket=6),
        ],
    )
    bot = _bot(bridge, store)

    with pytest.raises(OperationalException, match="Multiple broker orders"):
        bot.reconcile()


def test_reconcile_moves_filled_pending_to_position() -> None:
    store = MT5TradeStore(":memory:")
    # Broker now reports a position and no resting order: the pending filled.
    bridge = FakeBridge([BrokerPosition("EURUSD", "buy", 0.10, ticket=5)], [])
    bot = _bot(bridge, store)
    bot._pendings["EURUSD"] = ("buy", 0.10, 5)

    bot.reconcile()

    assert bot._positions["EURUSD"] == ("buy", 0.10)
    assert "EURUSD" not in bot._pendings


def test_reconcile_clears_managed_when_broker_position_identity_changes() -> None:
    store = MT5TradeStore(":memory:")
    store.open_position("EURUSD", "buy", 0.10, 1.10, ticket=1)
    store.set_managed_position("EURUSD", "buy", 1.10, 1.12, 0.5, True, False)
    bridge = FakeBridge([BrokerPosition("EURUSD", "buy", 0.10, price=1.20, ticket=2)], [])
    bot = _bot(bridge, store)

    assert "EURUSD" in bot._managed

    bot.reconcile()

    assert "EURUSD" not in bot._managed
    assert store.managed_positions() == {}
    position = store.open_positions()["EURUSD"]
    assert position.entry_price == 1.20
    assert position.ticket == 2


def test_restore_managed_rejects_position_mismatch() -> None:
    store = MT5TradeStore(":memory:")
    store.open_position("EURUSD", "sell", 0.10, 1.10, ticket=1)
    store.set_managed_position("EURUSD", "buy", 1.10, 1.12, 0.5, True, False)
    bridge = FakeBridge(None, None)

    bot = _bot(bridge, store)

    assert bot._managed == {}
    assert store.managed_positions() == {}


def test_reconcile_drops_cancelled_pending() -> None:
    store = MT5TradeStore(":memory:")
    bridge = FakeBridge([], [])
    bot = _bot(bridge, store)
    bot._pendings["EURUSD"] = ("buy", 0.10, 5)

    bot.reconcile()

    assert bot._pendings == {}
