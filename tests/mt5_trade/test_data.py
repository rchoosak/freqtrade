from __future__ import annotations

import json

import pytest

from freqtrade.exceptions import OperationalException
from freqtrade.mt5_trade.data import MT5Bar, ReplayDataFeed


def _bars(closes: list[float]) -> list[MT5Bar]:
    return [
        MT5Bar(time=i, open=c, high=c, low=c, close=c, volume=1.0)
        for i, c in enumerate(closes)
    ]


def test_replay_feed_reveals_one_bar_per_advance() -> None:
    feed = ReplayDataFeed({"EURUSD": _bars([1, 2, 3, 4])}, warmup=2)

    # warmup=2 → first two bars visible immediately.
    assert [b.close for b in feed.latest_bars("EURUSD", 10)] == [1, 2]
    assert feed.advance() is True
    assert [b.close for b in feed.latest_bars("EURUSD", 10)] == [1, 2, 3]
    assert feed.advance() is True
    assert [b.close for b in feed.latest_bars("EURUSD", 10)] == [1, 2, 3, 4]
    # Exhausted: no more bars to reveal.
    assert feed.advance() is False
    assert feed.exhausted is True


def test_replay_feed_latest_bars_respects_count() -> None:
    feed = ReplayDataFeed({"EURUSD": _bars([1, 2, 3, 4, 5])}, warmup=5)

    assert [b.close for b in feed.latest_bars("EURUSD", 2)] == [4, 5]
    assert feed.latest_bars("EURUSD", 0) == []


def test_replay_feed_unknown_symbol_raises() -> None:
    feed = ReplayDataFeed({"EURUSD": _bars([1, 2])}, warmup=1)

    with pytest.raises(KeyError, match="GBPUSD"):
        feed.latest_bars("GBPUSD", 1)


def test_replay_feed_empty_data_raises() -> None:
    with pytest.raises(OperationalException, match="at least one symbol"):
        ReplayDataFeed({})


def test_replay_feed_from_json(tmp_path) -> None:
    data_file = tmp_path / "bars.json"
    data_file.write_text(
        json.dumps({"EURUSD": [[0, 1.0, 1.1, 0.9, 1.05, 100], [1, 1.05, 1.2, 1.0, 1.15]]})
    )

    feed = ReplayDataFeed.from_json(str(data_file), warmup=2)
    bars = feed.latest_bars("EURUSD", 10)

    assert len(bars) == 2
    assert bars[0].close == 1.05
    assert bars[0].volume == 100
    # Volume defaults to 0 when omitted.
    assert bars[1].volume == 0.0


class _Row(dict):
    """Minimal stand-in for a numpy structured-array row (supports row["x"] and row.dtype.names)."""

    class _DType:
        names = ("time", "open", "high", "low", "close", "tick_volume")

    dtype = _DType()


class _RatesMT5:
    TIMEFRAME_M5 = 5

    def __init__(self) -> None:
        self.from_pos_calls: list = []
        self.healthy = True
        self.initialize_calls = 0
        self.shutdown_calls = 0

    def initialize(self, **kwargs):
        self.healthy = True
        self.initialize_calls += 1
        return True

    def terminal_info(self):
        return object() if self.healthy else None

    def symbol_select(self, symbol, enabled):
        return True

    def shutdown(self):
        self.shutdown_calls += 1
        return None

    def last_error(self):
        return (0, "ok")

    def copy_rates_from_pos(self, symbol, timeframe, start_pos, count):
        self.from_pos_calls.append((symbol, timeframe, start_pos, count))
        return [_Row(time=0, open=1.0, high=1.1, low=0.9, close=1.05, tick_volume=10)]


def test_live_feed_fetches_completed_bars_from_position_one() -> None:
    from freqtrade.mt5_trade.data import LiveMT5DataFeed
    from freqtrade.mt5_trade.gateway import LazyMT5Gateway
    from freqtrade.mt5_trade.models import MT5BridgeConfig, MT5SymbolMapping

    fake = _RatesMT5()
    config = MT5BridgeConfig(
        symbols=(MT5SymbolMapping(base="EUR", quote="USD", mt5_symbol="EURUSD"),),
        dry_run=False,
    )
    feed = LiveMT5DataFeed(LazyMT5Gateway(config, mt5_module=fake), timeframe="M5")

    bars = feed.latest_bars("EURUSD", 3)

    # start_pos must be 1 (skip the forming candle), not 0.
    assert fake.from_pos_calls == [("EURUSD", 5, 1, 3)]
    assert len(bars) == 1
    assert bars[0].close == 1.05


def test_live_feed_reconnects_before_fetch_when_terminal_is_stale() -> None:
    from freqtrade.mt5_trade.data import LiveMT5DataFeed
    from freqtrade.mt5_trade.gateway import LazyMT5Gateway
    from freqtrade.mt5_trade.models import MT5BridgeConfig, MT5SymbolMapping

    fake = _RatesMT5()
    config = MT5BridgeConfig(
        symbols=(MT5SymbolMapping(base="EUR", quote="USD", mt5_symbol="EURUSD"),),
        dry_run=False,
    )
    feed = LiveMT5DataFeed(LazyMT5Gateway(config, mt5_module=fake), timeframe="M5")

    feed.latest_bars("EURUSD", 3)
    fake.healthy = False
    feed.latest_bars("EURUSD", 3)

    assert fake.initialize_calls == 2
    assert fake.shutdown_calls == 1
