from __future__ import annotations

import pytest

from freqtrade.exceptions import OperationalException
from freqtrade.mt5_trade.bot import MT5ForexBot
from freqtrade.mt5_trade.data import MT5Bar, ReplayDataFeed
from freqtrade.mt5_trade.execution import MT5ExecutionBridge
from freqtrade.mt5_trade.models import MT5BotConfig, MT5BridgeConfig, MT5SymbolMapping
from freqtrade.mt5_trade.persistence import MT5TradeStore
from freqtrade.mt5_trade.strategies import HOLD, MT5Strategy, Signal, SmaCrossStrategy


class ScriptedStrategy(MT5Strategy):
    """Emits a predetermined sequence of signals, one per on_bar call."""

    def __init__(self, signals: list[Signal]) -> None:
        self._signals = list(signals)

    def on_bar(self, symbol, bars):
        return self._signals.pop(0) if self._signals else HOLD


class M1WarmupStrategy(ScriptedStrategy):
    @property
    def required_timeframe(self) -> str:
        return "M1"

    @property
    def minimum_bars(self) -> int:
        return 20


def _bridge_config(dry_run: bool = True) -> MT5BridgeConfig:
    return MT5BridgeConfig(
        symbols=(MT5SymbolMapping(base="EUR", quote="USD", mt5_symbol="EURUSD"),),
        dry_run=dry_run,
    )


def _bot_config() -> MT5BotConfig:
    return MT5BotConfig(symbols=("EURUSD",), warmup_bars=5, poll_interval=1.0)


def _feed(n: int = 5) -> ReplayDataFeed:
    bars = [MT5Bar(time=i, open=1.0, high=1.0, low=1.0, close=1.0 + i) for i in range(n)]
    return ReplayDataFeed({"EURUSD": bars}, warmup=n)


def _make_bot(strategy: MT5Strategy, store: MT5TradeStore) -> MT5ForexBot:
    bridge = MT5ExecutionBridge(_bridge_config())
    return MT5ForexBot(bridge, _feed(), strategy, store, _bot_config(), default_volume=0.05)


def test_bot_rejects_strategy_timeframe_mismatch() -> None:
    strategy = M1WarmupStrategy([HOLD])
    config = MT5BotConfig(
        symbols=("EURUSD",), timeframe="M5", warmup_bars=20, poll_interval=1.0
    )

    with pytest.raises(OperationalException, match="requires timeframe M1"):
        MT5ForexBot(
            MT5ExecutionBridge(_bridge_config()),
            _feed(),
            strategy,
            MT5TradeStore(":memory:"),
            config,
        )


def test_bot_rejects_insufficient_strategy_warmup() -> None:
    strategy = M1WarmupStrategy([HOLD])
    config = MT5BotConfig(
        symbols=("EURUSD",), timeframe="M1", warmup_bars=19, poll_interval=1.0
    )

    with pytest.raises(OperationalException, match="requires warmup_bars >= 20"):
        MT5ForexBot(
            MT5ExecutionBridge(_bridge_config()),
            _feed(),
            strategy,
            MT5TradeStore(":memory:"),
            config,
        )


def test_bot_rejects_insufficient_actual_history() -> None:
    strategy = M1WarmupStrategy([HOLD])
    config = MT5BotConfig(
        symbols=("EURUSD",), timeframe="M1", warmup_bars=20, poll_interval=1.0
    )
    bot = MT5ForexBot(
        MT5ExecutionBridge(_bridge_config()),
        _feed(5),
        strategy,
        MT5TradeStore(":memory:"),
        config,
    )

    with pytest.raises(OperationalException, match=r"requires at least 20 bars.*only 5"):
        bot.run_once()


def test_bot_opens_position_once_without_restacking() -> None:
    store = MT5TradeStore(":memory:")
    bot = _make_bot(ScriptedStrategy([Signal("enter_long")] * 3), store)

    bot.run_once()
    bot.run_once()
    bot.run_once()

    # Only the first enter_long opens a position; the next two are ignored (already long).
    assert store.order_count() == 1
    assert store.open_positions()["EURUSD"].side == "buy"


def test_bot_reverses_position() -> None:
    store = MT5TradeStore(":memory:")
    bot = _make_bot(ScriptedStrategy([Signal("enter_long"), Signal("enter_short")]), store)

    bot.run_once()  # opens buy
    bot.run_once()  # closes buy, opens sell

    # buy-open, sell-to-close, sell-open = 3 orders.
    assert store.order_count() == 3
    assert store.open_positions()["EURUSD"].side == "sell"


def test_bot_exit_signal_closes_position() -> None:
    store = MT5TradeStore(":memory:")
    bot = _make_bot(ScriptedStrategy([Signal("enter_long"), Signal("exit")]), store)

    bot.run_once()  # opens buy
    bot.run_once()  # closes buy

    assert store.order_count() == 2
    assert store.open_positions() == {}


def test_bot_restores_open_positions_from_store() -> None:
    store = MT5TradeStore(":memory:")
    store.open_position("EURUSD", "buy", 0.05, 1.10)

    # A fresh bot sharing the store should already know it is long and not re-open.
    bot = _make_bot(ScriptedStrategy([Signal("enter_long")]), store)
    bot.run_once()

    assert store.order_count() == 0


def test_bot_run_loop_processes_replay_feed_to_exhaustion() -> None:
    # End-to-end with the real SMA strategy and a price series that crosses up then down.
    closes = [10] * 5 + [11, 12, 14, 16, 18] + [16, 13, 10, 8, 6]
    bars = [MT5Bar(time=i, open=c, high=c, low=c, close=c) for i, c in enumerate(closes)]
    feed = ReplayDataFeed({"EURUSD": bars}, warmup=1)
    store = MT5TradeStore(":memory:")
    bridge = MT5ExecutionBridge(_bridge_config())
    bot_config = MT5BotConfig(symbols=("EURUSD",), warmup_bars=50, poll_interval=1.0)
    bot = MT5ForexBot(
        bridge, feed, SmaCrossStrategy(fast=2, slow=3), store, bot_config, default_volume=0.05
    )

    bot.run(sleep=lambda _seconds: None)

    assert bot.running is False
    # The up-then-down series should have triggered at least one entry.
    assert store.order_count() >= 1
