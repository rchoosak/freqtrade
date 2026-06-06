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
