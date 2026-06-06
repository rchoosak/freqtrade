from __future__ import annotations

from freqtrade.mt5_trade.bot import MT5ForexBot
from freqtrade.mt5_trade.data import MT5Bar, ReplayDataFeed
from freqtrade.mt5_trade.models import MT5BotConfig, MT5OrderResult
from freqtrade.mt5_trade.notifier import Notifier
from freqtrade.mt5_trade.persistence import MT5TradeStore
from freqtrade.mt5_trade.strategy import HOLD, MT5Strategy, Signal


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
