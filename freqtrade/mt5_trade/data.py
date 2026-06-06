from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from freqtrade.exceptions import OperationalException


if TYPE_CHECKING:
    from freqtrade.mt5_trade.gateway import LazyMT5Gateway


@dataclass(frozen=True)
class MT5Bar:
    time: int
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


class MT5DataFeed(ABC):
    """Source of completed OHLC bars per symbol."""

    @abstractmethod
    def latest_bars(self, symbol: str, count: int) -> list[MT5Bar]:
        """Return up to ``count`` most recent completed bars for ``symbol`` (oldest first)."""

    def advance(self) -> bool:
        """
        Move the feed forward one step before the next iteration.

        Live feeds always have more data (returns True). Bounded feeds (replay) return False
        once exhausted so the bot loop can stop.
        """
        return True

    def close(self) -> None:  # noqa: B027  # optional override; no-op by default
        """Release any resources held by the feed. Subclasses may override."""


class ReplayDataFeed(MT5DataFeed):
    """
    Deterministic, offline feed that replays pre-loaded bars one step at a time.

    Used for dry-run runs without an MT5 terminal and for end-to-end tests. Each ``advance()``
    reveals one more bar per symbol, simulating time passing.
    """

    def __init__(self, data: dict[str, list[MT5Bar]], *, warmup: int = 1) -> None:
        if not data:
            raise OperationalException("ReplayDataFeed requires at least one symbol of bars.")
        self._data = {symbol: list(bars) for symbol, bars in data.items()}
        # Reveal `warmup` bars up front so the first iteration already has history.
        self._cursor = {
            symbol: min(warmup, len(bars)) for symbol, bars in self._data.items()
        }

    @classmethod
    def from_json(cls, path: str, *, warmup: int = 1) -> ReplayDataFeed:
        """
        Build a replay feed from a JSON file mapping symbol -> list of bars.

        Each bar is ``[time, open, high, low, close, volume]`` (volume optional). Intended for
        offline dry-run runs and the CLI smoke test where no MT5 terminal is available.
        """
        import json
        from pathlib import Path

        raw = json.loads(Path(path).read_text())
        if not isinstance(raw, dict):
            raise OperationalException(
                f"Replay data file {path!r} must map symbol -> list of bars."
            )
        data: dict[str, list[MT5Bar]] = {}
        for symbol, rows in raw.items():
            data[symbol] = [
                MT5Bar(
                    time=int(row[0]),
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]) if len(row) > 5 else 0.0,
                )
                for row in rows
            ]
        return cls(data, warmup=warmup)

    def latest_bars(self, symbol: str, count: int) -> list[MT5Bar]:
        if symbol not in self._data:
            raise KeyError(f"ReplayDataFeed has no bars for symbol {symbol!r}.")
        revealed = self._data[symbol][: self._cursor[symbol]]
        return revealed[-count:] if count > 0 else []

    def advance(self) -> bool:
        advanced = False
        for symbol, bars in self._data.items():
            if self._cursor[symbol] < len(bars):
                self._cursor[symbol] += 1
                advanced = True
        return advanced

    @property
    def exhausted(self) -> bool:
        return all(self._cursor[s] >= len(b) for s, b in self._data.items())


# MT5 timeframe names accepted in config -> MetaTrader5 module constant attribute names.
_TIMEFRAME_ATTR = {
    "M1": "TIMEFRAME_M1",
    "M5": "TIMEFRAME_M5",
    "M15": "TIMEFRAME_M15",
    "M30": "TIMEFRAME_M30",
    "H1": "TIMEFRAME_H1",
    "H4": "TIMEFRAME_H4",
    "D1": "TIMEFRAME_D1",
    "W1": "TIMEFRAME_W1",
    "MN1": "TIMEFRAME_MN1",
}


class LiveMT5DataFeed(MT5DataFeed):
    """
    Live feed backed by ``MetaTrader5.copy_rates_from_pos`` via the shared gateway connection.

    Only exercised on a real MT5 terminal (Windows); unit tests inject a fake gateway/module.
    """

    def __init__(self, gateway: LazyMT5Gateway, timeframe: str = "M5") -> None:
        key = timeframe.upper()
        if key not in _TIMEFRAME_ATTR:
            raise OperationalException(
                f"Unsupported MT5 timeframe {timeframe!r}. Expected one of: "
                f"{', '.join(_TIMEFRAME_ATTR)}."
            )
        self._gateway = gateway
        self._timeframe_name = key

    def _timeframe(self) -> int:
        attr = _TIMEFRAME_ATTR[self._timeframe_name]
        timeframe = getattr(self._gateway.mt5, attr, None)
        if timeframe is None:
            raise OperationalException(f"MetaTrader5 module is missing constant {attr}.")
        return timeframe

    def latest_bars(self, symbol: str, count: int) -> list[MT5Bar]:
        self._gateway.connect()
        rates = self._gateway.mt5.copy_rates_from_pos(symbol, self._timeframe(), 0, count)
        if rates is None:
            raise OperationalException(
                f"MT5 copy_rates_from_pos returned no data for {symbol}: "
                f"{self._gateway.mt5.last_error()}"
            )
        bars = [
            MT5Bar(
                time=int(row["time"]),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["tick_volume"]) if "tick_volume" in row.dtype.names else 0.0,
            )
            for row in rates
        ]
        return bars

    def close(self) -> None:
        self._gateway.shutdown()
