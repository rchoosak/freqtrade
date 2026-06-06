from __future__ import annotations

import pytest

from freqtrade.mt5_trade.data import MT5Bar, MT5DataFeed, dump_bars_json, load_bars_json
from freqtrade.mt5_trade.history import MT5HistoryDownloader


class FakeFeed(MT5DataFeed):
    def __init__(self, data: dict[str, list[MT5Bar]]) -> None:
        self._data = data
        self.requests: list[tuple[str, int]] = []

    def latest_bars(self, symbol: str, count: int) -> list[MT5Bar]:
        self.requests.append((symbol, count))
        return self._data[symbol][-count:]


def _bars(closes: list[float]) -> list[MT5Bar]:
    return [
        MT5Bar(time=i, open=c, high=c, low=c, close=c, volume=1.0)
        for i, c in enumerate(closes)
    ]


def test_downloader_fetches_per_symbol() -> None:
    feed = FakeFeed({"EURUSD": _bars([1, 2, 3]), "GBPUSD": _bars([4, 5])})
    downloader = MT5HistoryDownloader(feed)

    data = downloader.download(["EURUSD", "GBPUSD"], count=2)

    assert [b.close for b in data["EURUSD"]] == [2, 3]
    assert [b.close for b in data["GBPUSD"]] == [4, 5]
    assert feed.requests == [("EURUSD", 2), ("GBPUSD", 2)]


def test_downloader_rejects_bad_count() -> None:
    downloader = MT5HistoryDownloader(FakeFeed({"EURUSD": _bars([1])}))
    with pytest.raises(ValueError, match="bar count"):
        downloader.download(["EURUSD"], count=0)


def test_downloader_writes_loadable_json(tmp_path) -> None:
    out = tmp_path / "bars.json"
    feed = FakeFeed({"EURUSD": _bars([1.1, 2.2, 3.3])})
    downloader = MT5HistoryDownloader(feed)

    downloader.download_to_json(["EURUSD"], count=3, path=str(out))
    reloaded = load_bars_json(str(out))

    assert [b.close for b in reloaded["EURUSD"]] == [1.1, 2.2, 3.3]


def test_dump_load_bars_round_trip(tmp_path) -> None:
    out = tmp_path / "bars.json"
    data = {"EURUSD": _bars([1.0, 2.0])}

    dump_bars_json(data, str(out))
    reloaded = load_bars_json(str(out))

    assert reloaded["EURUSD"][0].volume == 1.0
    assert reloaded["EURUSD"][1].close == 2.0
