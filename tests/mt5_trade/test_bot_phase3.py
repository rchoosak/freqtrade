from __future__ import annotations

from freqtrade.mt5_trade.bot import MT5ForexBot
from freqtrade.mt5_trade.data import MT5Bar, ReplayDataFeed
from freqtrade.mt5_trade.models import MT5BotConfig, MT5OrderResult
from freqtrade.mt5_trade.notifier import Notifier
from freqtrade.mt5_trade.persistence import MT5TradeStore
from freqtrade.mt5_trade.sizing import PositionSizer
from freqtrade.mt5_trade.strategies import HOLD, MT5Strategy, Signal


class FakeBridge:
    def __init__(self) -> None:
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
        return None

    def broker_orders(self):
        return None

    def account_balance(self):
        return None

    def close(self):
        self.closed = True


class RecordingNotifier(Notifier):
    def __init__(self) -> None:
        self.messages: list[str] = []

    def send(self, message: str) -> None:
        self.messages.append(message)


class ScriptedStrategy(MT5Strategy):
    def __init__(self, signals: list[Signal]) -> None:
        self._signals = list(signals)

    def on_bar(self, symbol, bars):
        return self._signals.pop(0) if self._signals else HOLD


class RaisingStrategy(MT5Strategy):
    def on_bar(self, symbol, bars):
        raise RuntimeError("boom")


def _feed(n: int = 20) -> ReplayDataFeed:
    bars = [MT5Bar(time=i, open=1.0, high=1.0, low=1.0, close=1.0 + i) for i in range(n)]
    return ReplayDataFeed({"EURUSD": bars}, warmup=1)


def test_bot_applies_sltp_on_open() -> None:
    bridge = FakeBridge()
    store = MT5TradeStore(":memory:")
    cfg = MT5BotConfig(symbols=("EURUSD",), warmup_bars=5, poll_interval=1.0)
    strategy = ScriptedStrategy([Signal("enter_long", stop_loss=1.07, take_profit=1.12)])
    bot = MT5ForexBot(bridge, _feed(), strategy, store, cfg, default_volume=0.05)

    bot.run_once()

    assert len(bridge.orders) == 1
    assert bridge.sltp == [("EURUSD", 1.07, 1.12)]


def test_bot_uses_risk_percent_position_sizing() -> None:
    bridge = FakeBridge()
    store = MT5TradeStore(":memory:")
    cfg = MT5BotConfig(symbols=("EURUSD",), warmup_bars=5, poll_interval=1.0)
    feed = ReplayDataFeed({"EURUSD": [MT5Bar(time=0, open=10, high=10, low=10, close=10)]})
    strategy = ScriptedStrategy([Signal("enter_long", stop_loss=5.0)])
    sizer = PositionSizer.from_config(
        {"position_sizing": {"mode": "risk_percent", "risk_per_trade": 1.0}},
        default_lot_size=0.05,
        contract_size=100,
    )
    bot = MT5ForexBot(
        bridge,
        feed,
        strategy,
        store,
        cfg,
        default_volume=0.05,
        position_sizer=sizer,
        account_balance=1000,
    )

    bot.run_once()

    assert bridge.orders[0].volume == 0.02


def test_bot_does_not_call_sltp_without_levels() -> None:
    bridge = FakeBridge()
    store = MT5TradeStore(":memory:")
    cfg = MT5BotConfig(symbols=("EURUSD",), warmup_bars=5, poll_interval=1.0)
    bot = MT5ForexBot(bridge, _feed(), ScriptedStrategy([Signal("enter_long")]), store, cfg)

    bot.run_once()

    assert bridge.sltp == []


def test_bot_recovers_from_iteration_errors_then_stops() -> None:
    bridge = FakeBridge()
    store = MT5TradeStore(":memory:")
    notifier = RecordingNotifier()
    cfg = MT5BotConfig(symbols=("EURUSD",), warmup_bars=1, poll_interval=1.0)
    bot = MT5ForexBot(
        bridge, _feed(), RaisingStrategy(), store, cfg, notifier=notifier, max_consecutive_errors=3
    )

    bot.run(sleep=lambda _seconds: None)

    assert bot.running is False
    assert any("Too many consecutive errors" in m for m in notifier.messages)
    assert any("Iteration error" in m for m in notifier.messages)
    # The loop kept going across failures rather than crashing.
    assert sum("Iteration error" in m for m in notifier.messages) == 3


class BrokerBalanceBridge(FakeBridge):
    def account_balance(self):
        return 2000.0


def test_bot_risk_percent_uses_live_broker_balance() -> None:
    bridge = BrokerBalanceBridge()
    store = MT5TradeStore(":memory:")
    cfg = MT5BotConfig(symbols=("EURUSD",), warmup_bars=5, poll_interval=1.0)
    feed = ReplayDataFeed({"EURUSD": [MT5Bar(time=0, open=10, high=10, low=10, close=10)]})
    strategy = ScriptedStrategy([Signal("enter_long", stop_loss=5.0)])
    sizer = PositionSizer.from_config(
        {"position_sizing": {"mode": "risk_percent", "risk_per_trade": 1.0}},
        default_lot_size=0.05,
        contract_size=100,
    )
    bot = MT5ForexBot(
        bridge, feed, strategy, store, cfg,
        default_volume=0.05, position_sizer=sizer, account_balance=1000,
    )

    bot.run_once()

    # Sized from the live broker balance 2000 (not the static config 1000):
    # 2000 * 1% / (5 * 100) = 0.04.
    assert bridge.orders[0].volume == 0.04
