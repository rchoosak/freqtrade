from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import datetime

from freqtrade.mt5_trade.data import MT5Bar, MT5DataFeed, dump_bars_json


logger = logging.getLogger(__name__)


class MT5HistoryDownloader:
    """
    Fetch historical bars from a data feed and cache them as JSON for backtesting / replay.

    Drives any ``MT5DataFeed``; in production this is a ``LiveMT5DataFeed`` over a connected
    gateway (live-only), but tests inject a fake feed so the download/cache flow is verified
    offline.
    """

    def __init__(self, feed: MT5DataFeed) -> None:
        self._feed = feed

    def download(self, symbols: Iterable[str], count: int) -> dict[str, list[MT5Bar]]:
        if count <= 0:
            raise ValueError(f"Invalid bar count {count!r}. Expected > 0.")
        data: dict[str, list[MT5Bar]] = {}
        for symbol in symbols:
            bars = self._feed.latest_bars(symbol, count)
            logger.info("Downloaded %d bars for %s.", len(bars), symbol)
            data[symbol] = bars
        return data

    def download_range(
        self, symbols: Iterable[str], date_from: datetime, date_to: datetime
    ) -> dict[str, list[MT5Bar]]:
        if date_from >= date_to:
            raise ValueError("date_from must be earlier than date_to.")
        data: dict[str, list[MT5Bar]] = {}
        for symbol in symbols:
            bars = self._feed.bars_range(symbol, date_from, date_to)
            logger.info(
                "Downloaded %d bars for %s (%s..%s).", len(bars), symbol, date_from, date_to
            )
            data[symbol] = bars
        return data

    def download_to_json(
        self, symbols: Iterable[str], count: int, path: str
    ) -> dict[str, list[MT5Bar]]:
        data = self.download(symbols, count)
        dump_bars_json(data, path)
        logger.info("Wrote bar cache to %s.", path)
        return data

    def download_range_to_json(
        self, symbols: Iterable[str], date_from: datetime, date_to: datetime, path: str
    ) -> dict[str, list[MT5Bar]]:
        data = self.download_range(symbols, date_from, date_to)
        dump_bars_json(data, path)
        logger.info("Wrote bar cache to %s.", path)
        return data
