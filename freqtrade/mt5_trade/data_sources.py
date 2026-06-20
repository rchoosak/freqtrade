from __future__ import annotations

import csv
import lzma
import struct
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

from freqtrade.exceptions import OperationalException
from freqtrade.mt5_trade.data import MT5Bar, MT5DataFeed, load_bars_json
from freqtrade.mt5_trade.history import MT5HistoryDownloader
from freqtrade.mt5_trade.models import MT5BotConfig, MT5BridgeConfig


class MT5HistoricalDataSource(ABC):
    """Historical OHLC data source that can populate the mt5-trade replay JSON cache."""

    @abstractmethod
    def load(self, symbols: Iterable[str]) -> dict[str, list[MT5Bar]]:
        """Load bars for the requested symbols."""


class MT5TerminalDataSource(MT5HistoricalDataSource):
    """Historical bars from a live MT5 terminal-backed feed."""

    def __init__(
        self,
        feed: MT5DataFeed,
        *,
        count: int = 1000,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> None:
        self._downloader = MT5HistoryDownloader(feed)
        self._count = count
        self._date_from = date_from
        self._date_to = date_to

    def load(self, symbols: Iterable[str]) -> dict[str, list[MT5Bar]]:
        if self._date_from is not None and self._date_to is not None:
            return self._downloader.download_range(symbols, self._date_from, self._date_to)
        return self._downloader.download(symbols, self._count)


class JsonDataSource(MT5HistoricalDataSource):
    """Historical bars from an existing mt5-trade replay/history JSON file."""

    def __init__(self, path: str) -> None:
        self._path = path

    def load(self, symbols: Iterable[str]) -> dict[str, list[MT5Bar]]:
        raw = load_bars_json(self._path, strict=False)
        requested = tuple(symbols)
        missing = [symbol for symbol in requested if symbol not in raw]
        if missing:
            raise OperationalException(
                f"JSON data source {self._path!r} is missing symbols: {', '.join(missing)}."
            )
        return {symbol: _sorted_unique_bars(symbol, raw[symbol]) for symbol in requested}


class CsvDataSource(MT5HistoricalDataSource):
    """Historical bars from one or more generic OHLCV CSV files."""

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._time_column = str(config.get("time_column", "time"))
        self._open_column = str(config.get("open_column", "open"))
        self._high_column = str(config.get("high_column", "high"))
        self._low_column = str(config.get("low_column", "low"))
        self._close_column = str(config.get("close_column", "close"))
        self._volume_column = config.get("volume_column", "volume")
        self._timezone = str(config.get("timezone", "UTC"))
        self._paths = self._resolve_paths(config)

    def load(self, symbols: Iterable[str]) -> dict[str, list[MT5Bar]]:
        data: dict[str, list[MT5Bar]] = {}
        for symbol in symbols:
            path = self._paths.get(symbol)
            if path is None:
                raise OperationalException(f"CSV data source has no path for symbol {symbol!r}.")
            data[symbol] = _sorted_unique_bars(symbol, self._load_symbol(path))
        return data

    def _load_symbol(self, path: str) -> list[MT5Bar]:
        rows: list[MT5Bar] = []
        with Path(path).open(newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            for row_number, row in enumerate(reader, start=2):
                try:
                    rows.append(
                        MT5Bar(
                            time=_parse_timestamp(row[self._time_column], self._timezone),
                            open=float(row[self._open_column]),
                            high=float(row[self._high_column]),
                            low=float(row[self._low_column]),
                            close=float(row[self._close_column]),
                            volume=_parse_volume(row, self._volume_column),
                        )
                    )
                except KeyError as exc:
                    raise OperationalException(
                        f"CSV data source {path!r} is missing column {exc.args[0]!r}."
                    ) from exc
                except ValueError as exc:
                    raise OperationalException(
                        f"CSV data source {path!r} has invalid row {row_number}: {exc}."
                    ) from exc
        return rows

    @staticmethod
    def _resolve_paths(config: dict[str, Any]) -> dict[str, str]:
        if "paths" in config:
            paths = config["paths"]
            if not isinstance(paths, dict):
                raise OperationalException("CSV data_source.paths must map symbol -> path.")
            return {str(symbol): str(path) for symbol, path in paths.items()}

        path = config.get("path")
        symbol = config.get("symbol")
        if path and symbol:
            return {str(symbol): str(path)}

        raise OperationalException(
            "CSV data_source requires either 'paths' or both 'path' and 'symbol'."
        )


@dataclass(frozen=True)
class DukascopyTick:
    time_ms: int
    ask: float
    bid: float
    ask_volume: float
    bid_volume: float


class DukascopyDataSource(MT5HistoricalDataSource):
    """Historical tick data from Dukascopy's public hourly .bi5 datafeed."""

    _BASE_URL = "https://datafeed.dukascopy.com/datafeed"
    _PRICE_MODES = {"bid", "ask", "mid"}

    def __init__(
        self,
        config: dict[str, Any],
        *,
        timeframe: str,
        date_from: datetime,
        date_to: datetime,
    ) -> None:
        if date_from >= date_to:
            raise OperationalException("Dukascopy data_source 'from' must be earlier than 'to'.")

        price = str(config.get("price", "bid")).lower()
        if price not in self._PRICE_MODES:
            raise OperationalException(
                f"Unsupported Dukascopy price {price!r}. Expected bid, ask, or mid."
            )

        self._config = config
        self._base_url = str(config.get("base_url", self._BASE_URL)).rstrip("/")
        self._timeframe = timeframe.upper()
        self._bucket_seconds = _timeframe_seconds(self._timeframe)
        self._date_from = _ensure_utc(date_from)
        self._date_to = _ensure_utc(date_to)
        self._price = price
        self._price_scale = float(config.get("price_scale", 100000))
        self._price_scales = {
            str(symbol).upper(): float(scale)
            for symbol, scale in config.get("price_scales", {}).items()
        }
        self._timeout = float(config.get("timeout", 60))
        self._retries = int(config.get("retries", 3))
        self._retry_sleep = float(config.get("retry_sleep", 2.0))
        self._instruments = {
            str(symbol): str(instrument)
            for symbol, instrument in config.get("instruments", {}).items()
        }

    def load(self, symbols: Iterable[str]) -> dict[str, list[MT5Bar]]:
        data: dict[str, list[MT5Bar]] = {}
        for symbol in symbols:
            instrument = self._instrument_for(symbol)
            ticks = self._load_ticks(instrument, self._price_scale_for(symbol, instrument))
            data[symbol] = self._aggregate_ticks(ticks)
        return data

    def _load_ticks(self, instrument: str, price_scale: float) -> list[DukascopyTick]:
        ticks: list[DukascopyTick] = []
        current = self._floor_hour(self._date_from)
        end_hour = self._floor_hour(self._date_to) + timedelta(hours=1)

        while current < end_hour:
            ticks.extend(self._download_hour(instrument, current, price_scale))
            current += timedelta(hours=1)

        start_ms = int(self._date_from.timestamp() * 1000)
        end_ms = int(self._date_to.timestamp() * 1000)
        return [tick for tick in ticks if start_ms <= tick.time_ms < end_ms]

    def _download_hour(
        self, instrument: str, hour: datetime, price_scale: float
    ) -> list[DukascopyTick]:
        url = self._hour_url(instrument, hour)
        scheme = urlparse(url).scheme
        if scheme not in {"http", "https"}:
            raise OperationalException(f"Unsupported Dukascopy URL scheme {scheme!r}.")
        request = urllib.request.Request(  # noqa: S310
            url, headers={"User-Agent": "freqtrade-mt5-trade/1.0"}
        )
        retryable = {429, 502, 503, 504}
        for attempt in range(1, self._retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self._timeout) as response:  # noqa: S310
                    compressed = response.read()
                return _parse_dukascopy_bi5(compressed, hour, price_scale=price_scale)
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    return []
                if exc.code not in retryable or attempt == self._retries:
                    raise OperationalException(
                        f"Dukascopy download failed for {url}: {exc}."
                    ) from exc
            except urllib.error.URLError as exc:
                if attempt == self._retries:
                    raise OperationalException(
                        f"Dukascopy download failed for {url}: {exc}."
                    ) from exc
            except OSError as exc:
                if attempt == self._retries:
                    raise OperationalException(
                        f"Dukascopy download failed for {url}: {exc}."
                    ) from exc
            except OperationalException as exc:
                if attempt == self._retries:
                    raise OperationalException(
                        f"Dukascopy download failed for {url}: {exc}"
                    ) from exc
            time.sleep(self._retry_sleep)

        raise AssertionError("unreachable")  # pragma: no cover

    def _hour_url(self, instrument: str, hour: datetime) -> str:
        # Dukascopy's path uses zero-based months (January = 00).
        month = hour.month - 1
        return (
            f"{self._base_url}/{instrument}/{hour.year:04d}/{month:02d}/"
            f"{hour.day:02d}/{hour.hour:02d}h_ticks.bi5"
        )

    def _instrument_for(self, symbol: str) -> str:
        if symbol in self._instruments:
            return self._instruments[symbol].upper()
        root = symbol.upper()
        if root.endswith(".MT5"):
            root = root[:-4]
        return root.replace("/", "").replace("_", "").replace("-", "").split(".", maxsplit=1)[0]

    def _price_scale_for(self, symbol: str, instrument: str) -> float:
        return self._price_scales.get(
            symbol.upper(),
            self._price_scales.get(instrument.upper(), self._price_scale),
        )

    def _aggregate_ticks(self, ticks: list[DukascopyTick]) -> list[MT5Bar]:
        buckets: dict[int, list[float]] = {}
        volumes: dict[int, float] = {}

        for tick in ticks:
            tick_second = tick.time_ms // 1000
            bucket = tick_second - (tick_second % self._bucket_seconds)
            price = self._tick_price(tick)
            buckets.setdefault(bucket, []).append(price)
            volumes[bucket] = volumes.get(bucket, 0.0) + tick.bid_volume + tick.ask_volume

        return [
            MT5Bar(
                time=bucket,
                open=prices[0],
                high=max(prices),
                low=min(prices),
                close=prices[-1],
                volume=volumes.get(bucket, 0.0),
            )
            for bucket, prices in sorted(buckets.items())
        ]

    def _tick_price(self, tick: DukascopyTick) -> float:
        if self._price == "ask":
            return tick.ask
        if self._price == "mid":
            return (tick.bid + tick.ask) / 2
        return tick.bid

    @staticmethod
    def _floor_hour(value: datetime) -> datetime:
        value = _ensure_utc(value)
        return value.replace(minute=0, second=0, microsecond=0)


def build_historical_data_source(
    bridge_config: MT5BridgeConfig,
    bot_config: MT5BotConfig,
) -> MT5HistoricalDataSource:
    """Build the configured historical data source for ``download-data-mt5``."""
    source_config = bridge_config.extra.get("data_source", {"type": "mt5"})
    if not isinstance(source_config, dict):
        raise OperationalException("mt5_trade.data_source must be an object.")

    source_type = str(source_config.get("type", "mt5")).lower()
    if source_type == "mt5":
        from freqtrade.mt5_trade.data import LiveMT5DataFeed
        from freqtrade.mt5_trade.gateway import LazyMT5Gateway

        feed = LiveMT5DataFeed(LazyMT5Gateway(bridge_config), timeframe=bot_config.timeframe)
        date_from = bridge_config.extra.get("history_from")
        date_to = bridge_config.extra.get("history_to")
        return MT5TerminalDataSource(
            feed,
            count=int(bridge_config.extra.get("history_bars", 1000)),
            date_from=_parse_iso_datetime(date_from) if date_from else None,
            date_to=_parse_iso_datetime(date_to) if date_to else None,
        )
    if source_type == "csv":
        return CsvDataSource(source_config)
    if source_type == "json":
        path = source_config.get("path") or bridge_config.extra.get("replay_data")
        if not path:
            raise OperationalException("JSON data_source requires a 'path'.")
        return JsonDataSource(str(path))
    if source_type == "dukascopy":
        date_from = (
            source_config.get("from")
            or source_config.get("date_from")
            or bridge_config.extra.get("history_from")
        )
        date_to = (
            source_config.get("to")
            or source_config.get("date_to")
            or bridge_config.extra.get("history_to")
        )
        if not date_from or not date_to:
            raise OperationalException(
                "Dukascopy data_source requires 'from'/'to' (or history_from/history_to)."
            )
        return DukascopyDataSource(
            source_config,
            timeframe=str(source_config.get("timeframe", bot_config.timeframe)),
            date_from=_parse_iso_datetime(date_from),
            date_to=_parse_iso_datetime(date_to),
        )

    raise OperationalException(
        "Unsupported mt5_trade data_source type "
        f"{source_type!r}. Expected mt5, csv, json, or dukascopy."
    )


def _parse_volume(row: dict[str, str], volume_column: Any) -> float:
    if volume_column is None:
        return 0.0
    value = row.get(str(volume_column))
    if value is None or value == "":
        return 0.0
    return float(value)


def _parse_timestamp(value: str, timezone_name: str) -> int:
    value = value.strip()
    if value.isdigit():
        return int(value)

    dt = _parse_iso_datetime(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo(timezone_name))
    return int(dt.astimezone(UTC).timestamp())


def _parse_iso_datetime(value: Any) -> datetime:
    text = str(value).strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    return datetime.fromisoformat(text)


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _timeframe_seconds(timeframe: str) -> int:
    mapping = {
        "M1": 60,
        "M5": 5 * 60,
        "M15": 15 * 60,
        "M30": 30 * 60,
        "H1": 60 * 60,
        "H4": 4 * 60 * 60,
        "D1": 24 * 60 * 60,
    }
    if timeframe not in mapping:
        raise OperationalException(
            f"Unsupported Dukascopy aggregation timeframe {timeframe!r}. "
            f"Expected one of: {', '.join(mapping)}."
        )
    return mapping[timeframe]


def _parse_dukascopy_bi5(
    compressed: bytes,
    hour: datetime,
    *,
    price_scale: float = 100000,
) -> list[DukascopyTick]:
    if not compressed:
        return []

    try:
        raw = lzma.decompress(compressed)
    except lzma.LZMAError as exc:
        raise OperationalException("Dukascopy .bi5 payload could not be decompressed.") from exc

    if len(raw) % 20 != 0:
        raise OperationalException(
            f"Dukascopy .bi5 payload has invalid length {len(raw)} (expected 20-byte records)."
        )

    hour_ms = int(_ensure_utc(hour).timestamp() * 1000)
    ticks: list[DukascopyTick] = []
    for offset in range(0, len(raw), 20):
        record = raw[offset : offset + 20]
        time_delta, ask, bid, ask_volume, bid_volume = struct.unpack(">IIIff", record)
        ticks.append(
            DukascopyTick(
                time_ms=hour_ms + int(time_delta),
                ask=ask / price_scale,
                bid=bid / price_scale,
                ask_volume=float(ask_volume),
                bid_volume=float(bid_volume),
            )
        )
    return ticks


def _sorted_unique_bars(symbol: str, bars: list[MT5Bar]) -> list[MT5Bar]:
    sorted_bars = sorted(bars, key=lambda bar: bar.time)
    seen: set[int] = set()
    for bar in sorted_bars:
        if bar.time in seen:
            raise OperationalException(f"Duplicate timestamp {bar.time} in bars for {symbol}.")
        seen.add(bar.time)
    return sorted_bars
