from __future__ import annotations

import json

import pytest

from freqtrade.exceptions import OperationalException
from freqtrade.mt5_trade.config import load_mt5_config, parse_mt5_config


def _valid_section() -> dict:
    return {
        "dry_run": False,
        "login": 123456,
        "password": "secret",
        "server": "Broker-Demo",
        "timeframe": "M15",
        "poll_interval": 3.0,
        "symbols": [
            {"base": "EUR", "quote": "USD", "mt5_symbol": "EURUSD"},
            {"base": "GBP", "quote": "USD", "mt5_symbol": "GBPUSD"},
        ],
    }


def test_parse_mt5_config_builds_typed_configs() -> None:
    bridge, bot = parse_mt5_config(_valid_section())

    assert bridge.dry_run is False
    assert bridge.login == 123456
    assert tuple(m.mt5_symbol for m in bridge.symbols) == ("EURUSD", "GBPUSD")
    assert bot.timeframe == "M15"
    assert bot.poll_interval == 3.0
    # trade symbols default to every mapped symbol.
    assert bot.symbols == ("EURUSD", "GBPUSD")


def test_parse_mt5_config_respects_explicit_trade_symbols() -> None:
    section = _valid_section()
    section["trade_symbols"] = ["EURUSD"]

    _, bot = parse_mt5_config(section)

    assert bot.symbols == ("EURUSD",)


def test_parse_mt5_config_reads_pending_expiry() -> None:
    section = _valid_section()
    section["pending_expiry"] = 12

    _, bot = parse_mt5_config(section)

    assert bot.pending_expiry == 12


def test_parse_mt5_config_accepts_backtest_account_settings() -> None:
    section = _valid_section()
    section["starting_balance"] = 1000
    section["contract_size"] = 100

    bridge, _ = parse_mt5_config(section)

    assert bridge.extra["starting_balance"] == 1000
    assert bridge.extra["contract_size"] == 100


def test_parse_mt5_config_accepts_position_sizing() -> None:
    section = _valid_section()
    section["position_sizing"] = {
        "mode": "risk_percent",
        "risk_per_trade": 1.0,
        "skip_if_min_lot_exceeds_risk": True,
    }

    bridge, _ = parse_mt5_config(section)

    assert bridge.extra["position_sizing"]["mode"] == "risk_percent"


def test_parse_mt5_config_accepts_csv_data_source() -> None:
    section = _valid_section()
    section["data_source"] = {
        "type": "csv",
        "path": "EURUSD.csv",
        "symbol": "EURUSD",
        "volume_column": None,
    }

    bridge, _ = parse_mt5_config(section)

    assert bridge.extra["data_source"]["type"] == "csv"


def test_parse_mt5_config_accepts_dukascopy_data_source() -> None:
    section = _valid_section()
    section["data_source"] = {
        "type": "dukascopy",
        "from": "2024-01-01T00:00:00Z",
        "to": "2024-01-02T00:00:00Z",
        "price": "mid",
    }

    bridge, _ = parse_mt5_config(section)

    assert bridge.extra["data_source"]["type"] == "dukascopy"


def test_parse_mt5_config_rejects_unknown_data_source() -> None:
    section = _valid_section()
    section["data_source"] = {"type": "websocket"}

    with pytest.raises(OperationalException, match="data_source"):
        parse_mt5_config(section)


def test_parse_mt5_config_rejects_unknown_position_sizing_mode() -> None:
    section = _valid_section()
    section["position_sizing"] = {"mode": "kelly"}

    with pytest.raises(OperationalException, match="position_sizing"):
        parse_mt5_config(section)


def test_parse_mt5_config_rejects_negative_pending_expiry() -> None:
    section = _valid_section()
    section["pending_expiry"] = -1

    with pytest.raises(OperationalException, match="pending_expiry"):
        parse_mt5_config(section)


def test_parse_mt5_config_rejects_zero_starting_balance() -> None:
    section = _valid_section()
    section["starting_balance"] = 0

    with pytest.raises(OperationalException, match="starting_balance"):
        parse_mt5_config(section)


def test_parse_mt5_config_rejects_missing_symbols() -> None:
    with pytest.raises(OperationalException, match="symbols"):
        parse_mt5_config({"dry_run": True})


def test_parse_mt5_config_rejects_bad_poll_interval() -> None:
    section = _valid_section()
    section["poll_interval"] = 0

    with pytest.raises(OperationalException, match="poll_interval"):
        parse_mt5_config(section)


def test_parse_mt5_config_rejects_symbol_missing_field() -> None:
    section = _valid_section()
    section["symbols"] = [{"base": "EUR", "quote": "USD"}]

    with pytest.raises(OperationalException, match="mt5_symbol"):
        parse_mt5_config(section)


def test_load_mt5_config_from_file_with_wrapped_section(tmp_path) -> None:
    config_file = tmp_path / "mt5.json"
    config_file.write_text(json.dumps({"mt5_trade": _valid_section()}))

    bridge, bot = load_mt5_config(str(config_file))

    assert bridge.server == "Broker-Demo"
    assert bot.symbols == ("EURUSD", "GBPUSD")


def test_load_mt5_config_rejects_file_without_section(tmp_path) -> None:
    config_file = tmp_path / "mt5.json"
    config_file.write_text(json.dumps({"something_else": 1}))

    with pytest.raises(OperationalException, match="No 'mt5_trade' section"):
        load_mt5_config(str(config_file))
