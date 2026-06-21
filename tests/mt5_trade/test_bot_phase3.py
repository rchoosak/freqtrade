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


class BrokerEquityBridge(BrokerBalanceBridge):
    def account_equity(self):
        return 1200.0


def test_bot_risk_percent_prefers_live_equity_over_balance() -> None:
    bridge = BrokerEquityBridge()
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

    # Equity 1200 includes floating losses and is safer than balance 2000:
    # 1200 * 1% / (5 * 100) = 0.024, floored to 0.02 lot.
    assert bridge.orders[0].volume == 0.02


def test_bot_rejects_pending_entry_with_risk_percent_sizing() -> None:
    bridge = FakeBridge()
    store = MT5TradeStore(":memory:")
    cfg = MT5BotConfig(symbols=("EURUSD",), warmup_bars=1, poll_interval=1.0)
    feed = ReplayDataFeed(
        {"EURUSD": [MT5Bar(time=0, open=100, high=100, low=100, close=100)]}
    )
    sizer = PositionSizer(
        mode="risk_percent",
        risk_per_trade=1.0,
        contract_size=100,
    )
    bot = MT5ForexBot(
        bridge,
        feed,
        ScriptedStrategy(
            [Signal("enter_long", order_kind="stop", price=110, stop_loss=90)]
        ),
        store,
        cfg,
        position_sizer=sizer,
        account_balance=10_000,
    )

    bot.run_once()

    assert bridge.orders == []
    assert store.open_positions() == {}


class SlippageRiskBridge(FakeBridge):
    def __init__(self, *, executable_price: float, fill_price: float) -> None:
        super().__init__()
        self._executable_price = executable_price
        self._fill_price = fill_price

    def executable_price(self, symbol, side):
        return self._executable_price

    def stop_loss_risk(self, symbol, side, volume, entry_price, stop_loss):
        distance = entry_price - stop_loss if side == "buy" else stop_loss - entry_price
        return max(0.0, distance) * volume * 100

    def submit_order(self, order):
        self.orders.append(order)
        if len(self.orders) == 1:
            return MT5OrderResult(
                accepted=True,
                order_id="entry",
                filled_volume=order.volume,
                fill_price=self._fill_price,
            )
        return MT5OrderResult(
            accepted=True,
            order_id="reduce",
            filled_volume=order.volume,
            fill_price=self._fill_price,
        )


def test_bot_sizes_market_entry_from_executable_quote() -> None:
    bridge = SlippageRiskBridge(executable_price=110, fill_price=110)
    store = MT5TradeStore(":memory:")
    cfg = MT5BotConfig(symbols=("EURUSD",), warmup_bars=1, poll_interval=1.0)
    feed = ReplayDataFeed(
        {"EURUSD": [MT5Bar(time=0, open=100, high=100, low=100, close=100)]}
    )
    sizer = PositionSizer(
        mode="risk_percent",
        risk_per_trade=1.0,
        contract_size=100,
        min_lot=0.01,
        lot_step=0.01,
    )
    bot = MT5ForexBot(
        bridge,
        feed,
        ScriptedStrategy([Signal("enter_long", stop_loss=90)]),
        store,
        cfg,
        position_sizer=sizer,
        account_balance=10_000,
    )

    bot.run_once()

    # $100 risk / (($110 executable - $90 stop) * 100) = 0.05 lot.
    assert bridge.orders[0].volume == 0.05
    assert len(bridge.orders) == 1


def test_bot_reduces_position_when_fill_exceeds_risk_cap() -> None:
    bridge = SlippageRiskBridge(executable_price=100, fill_price=110)
    store = MT5TradeStore(":memory:")
    cfg = MT5BotConfig(symbols=("EURUSD",), warmup_bars=1, poll_interval=1.0)
    feed = ReplayDataFeed(
        {"EURUSD": [MT5Bar(time=0, open=100, high=100, low=100, close=100)]}
    )
    sizer = PositionSizer(
        mode="risk_percent",
        risk_per_trade=0.75,
        capital_fraction=0.9,
        max_risk_amount=750,
        contract_size=100,
        min_lot=0.01,
        lot_step=0.01,
        max_lot=0.5,
    )
    bot = MT5ForexBot(
        bridge,
        feed,
        ScriptedStrategy([Signal("enter_long", stop_loss=90)]),
        store,
        cfg,
        position_sizer=sizer,
        account_balance=100_000,
    )

    bot.run_once()

    assert bridge.orders[0].volume == 0.5
    assert bridge.orders[1].side == "sell"
    assert bridge.orders[1].volume == 0.17
    assert store.open_positions()["EURUSD"].volume == 0.33
    assert bridge.stop_loss_risk("EURUSD", "buy", 0.33, 110, 90) == 660


def test_bot_closes_fill_when_broker_minimum_cannot_meet_risk_cap() -> None:
    bridge = SlippageRiskBridge(executable_price=100, fill_price=1000)
    store = MT5TradeStore(":memory:")
    cfg = MT5BotConfig(symbols=("EURUSD",), warmup_bars=1, poll_interval=1.0)
    feed = ReplayDataFeed(
        {"EURUSD": [MT5Bar(time=0, open=100, high=100, low=100, close=100)]}
    )
    sizer = PositionSizer(
        mode="risk_percent",
        risk_per_trade=0.75,
        capital_fraction=0.9,
        max_risk_amount=750,
        contract_size=100,
        min_lot=0.01,
        lot_step=0.01,
        max_lot=0.5,
    )
    bot = MT5ForexBot(
        bridge,
        feed,
        ScriptedStrategy([Signal("enter_long", stop_loss=90)]),
        store,
        cfg,
        position_sizer=sizer,
        account_balance=100_000,
    )

    bot.run_once()

    assert bridge.orders[1].side == "sell"
    assert bridge.orders[1].volume == 0.5
    assert store.open_positions() == {}


class UnverifiableFillRiskBridge(SlippageRiskBridge):
    def stop_loss_risk(self, symbol, side, volume, entry_price, stop_loss):
        if entry_price == self._fill_price:
            raise RuntimeError("order_calc_profit unavailable")
        return super().stop_loss_risk(symbol, side, volume, entry_price, stop_loss)


def test_bot_closes_fill_when_post_fill_risk_cannot_be_calculated() -> None:
    bridge = UnverifiableFillRiskBridge(executable_price=100, fill_price=110)
    store = MT5TradeStore(":memory:")
    cfg = MT5BotConfig(symbols=("EURUSD",), warmup_bars=1, poll_interval=1.0)
    feed = ReplayDataFeed(
        {"EURUSD": [MT5Bar(time=0, open=100, high=100, low=100, close=100)]}
    )
    sizer = PositionSizer(
        mode="risk_percent",
        risk_per_trade=0.75,
        capital_fraction=0.9,
        contract_size=100,
        min_lot=0.01,
        lot_step=0.01,
        max_lot=0.5,
    )
    bot = MT5ForexBot(
        bridge,
        feed,
        ScriptedStrategy([Signal("enter_long", stop_loss=90)]),
        store,
        cfg,
        position_sizer=sizer,
        account_balance=100_000,
    )

    bot.run_once()

    assert bridge.orders[1].side == "sell"
    assert bridge.orders[1].volume == 0.5
    assert store.open_positions() == {}
