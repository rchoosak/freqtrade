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
