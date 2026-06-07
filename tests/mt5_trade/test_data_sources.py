from __future__ import annotations

import json
import lzma
import struct
from datetime import UTC, datetime

import pytest

from freqtrade.commands.trade_commands import start_download_data_mt5
from freqtrade.exceptions import OperationalException
from freqtrade.mt5_trade.config import parse_mt5_config
from freqtrade.mt5_trade.data import MT5Bar, dump_bars_json, load_bars_json
from freqtrade.mt5_trade.data_sources import (
    CsvDataSource,
    DukascopyDataSource,
    DukascopyTick,
    JsonDataSource,
    build_historical_data_source,
)


def _section(tmp_path) -> dict:
    return {
        "dry_run": True,
        "history_file": str(tmp_path / "bars.json"),
        "trade_symbols": ["EURUSD"],
        "symbols": [{"base": "EUR", "quote": "USD", "mt5_symbol": "EURUSD"}],
    }


def test_csv_data_source_loads_and_sorts_bars(tmp_path) -> None:
    csv_path = tmp_path / "EURUSD.csv"
    csv_path.write_text(
        "\n".join(
            [
                "time,open,high,low,close,volume",
                "2024-01-01T00:05:00Z,1.1,1.2,1.0,1.15,11",
                "2024-01-01T00:00:00Z,1.0,1.1,0.9,1.05,10",
            ]
        )
    )

    source = CsvDataSource({"type": "csv", "paths": {"EURUSD": str(csv_path)}})
    data = source.load(["EURUSD"])

    assert [bar.close for bar in data["EURUSD"]] == [1.05, 1.15]
    assert data["EURUSD"][0].time == 1704067200
    assert data["EURUSD"][1].volume == 11.0


def test_csv_data_source_applies_timezone_to_naive_times(tmp_path) -> None:
    csv_path = tmp_path / "EURUSD.csv"
    csv_path.write_text("time,open,high,low,close\n2024-01-01T07:00:00,1,1,1,1\n")

    source = CsvDataSource(
        {
            "type": "csv",
            "path": str(csv_path),
            "symbol": "EURUSD",
            "timezone": "Asia/Bangkok",
            "volume_column": None,
        }
    )

    data = source.load(["EURUSD"])

    assert data["EURUSD"][0].time == 1704067200
    assert data["EURUSD"][0].volume == 0.0


def test_csv_data_source_rejects_duplicate_timestamps(tmp_path) -> None:
    csv_path = tmp_path / "EURUSD.csv"
    csv_path.write_text(
        "\n".join(
            [
                "time,open,high,low,close",
                "1704067200,1,1,1,1",
                "1704067200,2,2,2,2",
            ]
        )
    )
    source = CsvDataSource(
        {"type": "csv", "path": str(csv_path), "symbol": "EURUSD", "volume_column": None}
    )

    with pytest.raises(OperationalException, match="Duplicate timestamp"):
        source.load(["EURUSD"])


def test_json_data_source_filters_requested_symbols(tmp_path) -> None:
    json_path = tmp_path / "source.json"
    dump_bars_json(
        {
            "EURUSD": [MT5Bar(time=1, open=1, high=1, low=1, close=1)],
            "GBPUSD": [MT5Bar(time=2, open=2, high=2, low=2, close=2)],
        },
        str(json_path),
    )

    data = JsonDataSource(str(json_path)).load(["GBPUSD"])

    assert list(data) == ["GBPUSD"]
    assert data["GBPUSD"][0].close == 2


def test_build_historical_data_source_uses_csv_config(tmp_path) -> None:
    csv_path = tmp_path / "EURUSD.csv"
    csv_path.write_text("time,open,high,low,close\n1704067200,1,1,1,1\n")
    section = _section(tmp_path)
    section["data_source"] = {
        "type": "csv",
        "path": str(csv_path),
        "symbol": "EURUSD",
        "volume_column": None,
    }
    bridge, bot = parse_mt5_config(section)

    source = build_historical_data_source(bridge, bot)
    data = source.load(bot.symbols)

    assert data["EURUSD"][0].time == 1704067200


def test_download_data_mt5_command_imports_csv_offline(tmp_path) -> None:
    csv_path = tmp_path / "EURUSD.csv"
    out_path = tmp_path / "history.json"
    config_path = tmp_path / "mt5-config.json"
    csv_path.write_text("time,open,high,low,close,volume\n1704067200,1,2,0.5,1.5,99\n")
    section = _section(tmp_path)
    section["history_file"] = str(out_path)
    section["data_source"] = {
        "type": "csv",
        "path": str(csv_path),
        "symbol": "EURUSD",
    }
    config_path.write_text(json.dumps({"mt5_trade": section}))

    result = start_download_data_mt5({"config": [str(config_path)]})

    assert result == 0
    data = load_bars_json(str(out_path))
    assert data["EURUSD"][0].close == 1.5
    assert data["EURUSD"][0].volume == 99


def _bi5_payload(records: list[tuple[int, int, int, float, float]]) -> bytes:
    raw = b"".join(struct.pack(">IIIff", *record) for record in records)
    return lzma.compress(raw)


class FakeDukascopyDataSource(DukascopyDataSource):
    def __init__(self, *args, payloads: dict[datetime, bytes], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.payloads = payloads
        self.requests: list[tuple[str, datetime]] = []

    def _download_hour(self, instrument: str, hour: datetime, price_scale: float):
        self.requests.append((instrument, hour))
        payload = self.payloads.get(hour)
        return [] if payload is None else _parse_payload(payload, hour, price_scale=price_scale)


def _parse_payload(
    payload: bytes,
    hour: datetime,
    *,
    price_scale: float = 100000,
) -> list[DukascopyTick]:
    from freqtrade.mt5_trade.data_sources import _parse_dukascopy_bi5

    return _parse_dukascopy_bi5(payload, hour, price_scale=price_scale)


def test_dukascopy_source_parses_ticks_and_aggregates_to_timeframe() -> None:
    hour = datetime(2024, 1, 1, tzinfo=UTC)
    payload = _bi5_payload(
        [
            (0, 110010, 110000, 1.0, 2.0),
            (60_000, 110020, 110010, 1.5, 2.5),
            (300_000, 110120, 110100, 3.0, 4.0),
        ]
    )
    source = FakeDukascopyDataSource(
        {"type": "dukascopy", "price": "bid"},
        timeframe="M5",
        date_from=hour,
        date_to=datetime(2024, 1, 1, 0, 10, tzinfo=UTC),
        payloads={hour: payload},
    )

    data = source.load(["EURUSD"])

    assert source.requests == [("EURUSD", hour)]
    bars = data["EURUSD"]
    assert len(bars) == 2
    assert bars[0].time == 1704067200
    assert bars[0].open == 1.1
    assert bars[0].high == 1.1001
    assert bars[0].low == 1.1
    assert bars[0].close == 1.1001
    assert bars[0].volume == 7.0
    assert bars[1].open == 1.101


def test_dukascopy_source_can_aggregate_mid_price() -> None:
    hour = datetime(2024, 1, 1, tzinfo=UTC)
    payload = _bi5_payload([(0, 110020, 110000, 1.0, 1.0)])
    source = FakeDukascopyDataSource(
        {"type": "dukascopy", "price": "mid"},
        timeframe="M1",
        date_from=hour,
        date_to=datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
        payloads={hour: payload},
    )

    data = source.load(["EURUSD"])

    assert data["EURUSD"][0].close == 1.1001


def test_dukascopy_source_treats_empty_payload_as_no_ticks() -> None:
    assert _parse_payload(b"", datetime(2024, 1, 1, tzinfo=UTC)) == []


def test_dukascopy_source_applies_configured_price_scale() -> None:
    hour = datetime(2024, 1, 1, tzinfo=UTC)
    payload = _bi5_payload([(0, 4627005, 4626005, 1.0, 1.0)])
    source = FakeDukascopyDataSource(
        {"type": "dukascopy", "price": "bid", "price_scale": 1000},
        timeframe="M1",
        date_from=hour,
        date_to=datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
        payloads={hour: payload},
    )

    data = source.load(["XAUUSD"])

    assert data["XAUUSD"][0].close == 4626.005


def test_dukascopy_source_uses_zero_based_month_in_url() -> None:
    source = DukascopyDataSource(
        {"type": "dukascopy", "base_url": "https://example.test"},
        timeframe="M1",
        date_from=datetime(2024, 1, 1, tzinfo=UTC),
        date_to=datetime(2024, 1, 1, 1, tzinfo=UTC),
    )

    assert (
        source._hour_url("EURUSD", datetime(2024, 1, 2, 3, tzinfo=UTC))
        == "https://example.test/EURUSD/2024/00/02/03h_ticks.bi5"
    )


def test_build_historical_data_source_uses_dukascopy_config(tmp_path) -> None:
    section = _section(tmp_path)
    section["data_source"] = {
        "type": "dukascopy",
        "from": "2024-01-01T00:00:00Z",
        "to": "2024-01-01T01:00:00Z",
        "price": "bid",
    }
    bridge, bot = parse_mt5_config(section)

    source = build_historical_data_source(bridge, bot)

    assert isinstance(source, DukascopyDataSource)


def test_dukascopy_source_requires_date_range(tmp_path) -> None:
    section = _section(tmp_path)
    section["data_source"] = {"type": "dukascopy"}
    bridge, bot = parse_mt5_config(section)

    with pytest.raises(OperationalException, match="requires 'from'/'to'"):
        build_historical_data_source(bridge, bot)
