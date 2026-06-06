from __future__ import annotations

from typing import Any

from jsonschema import Draft4Validator

from freqtrade.exceptions import OperationalException
from freqtrade.mt5_trade.models import MT5BotConfig, MT5BridgeConfig


# Local JSON schema for the ``mt5_trade`` config section. Intentionally kept separate from
# freqtrade's main CONF_SCHEMA (which models the CCXT crypto bot) so the two stay decoupled.
MT5_CONFIG_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["symbols"],
    "properties": {
        "dry_run": {"type": "boolean"},
        "terminal_path": {"type": "string"},
        "login": {"type": "integer"},
        "password": {"type": "string"},
        "server": {"type": "string"},
        "default_lot_size": {"type": "number", "minimum": 0, "exclusiveMinimum": True},
        "deviation": {"type": "integer", "minimum": 0},
        "magic": {"type": "integer"},
        "comment": {"type": "string"},
        "timeframe": {"type": "string"},
        "poll_interval": {"type": "number", "minimum": 0, "exclusiveMinimum": True},
        "warmup_bars": {"type": "integer", "minimum": 1},
        "db_path": {"type": "string"},
        "trade_symbols": {"type": "array", "items": {"type": "string"}},
        "symbols": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["base", "quote", "mt5_symbol"],
                "properties": {
                    "base": {"type": "string"},
                    "quote": {"type": "string"},
                    "mt5_symbol": {"type": "string"},
                    "venue": {"type": "string"},
                    "price_precision": {"type": "integer", "minimum": 0},
                    "lot_precision": {"type": "integer", "minimum": 0},
                    "min_lot": {"type": "number", "minimum": 0, "exclusiveMinimum": True},
                    "lot_step": {"type": "number", "minimum": 0, "exclusiveMinimum": True},
                },
            },
        },
    },
}


def validate_mt5_section(section: dict[str, Any]) -> None:
    """Validate the raw ``mt5_trade`` section, raising OperationalException on the first error."""
    errors = sorted(Draft4Validator(MT5_CONFIG_SCHEMA).iter_errors(section), key=str)
    if errors:
        first = errors[0]
        location = ".".join(str(p) for p in first.path) or "mt5_trade"
        raise OperationalException(f"Invalid mt5_trade config at '{location}': {first.message}")


def parse_mt5_config(section: dict[str, Any]) -> tuple[MT5BridgeConfig, MT5BotConfig]:
    """Turn a validated ``mt5_trade`` section into typed bridge + bot configs."""
    validate_mt5_section(section)
    bridge = MT5BridgeConfig.from_dict(section)

    bot_data = dict(section)
    # ``trade_symbols`` selects which mapped symbols the bot trades; default to all of them.
    bot_data["symbols"] = section.get(
        "trade_symbols", [mapping.mt5_symbol for mapping in bridge.symbols]
    )
    bot = MT5BotConfig.from_dict(bot_data)
    return bridge, bot


def load_mt5_config(config: str | dict[str, Any]) -> tuple[MT5BridgeConfig, MT5BotConfig]:
    """
    Load MT5 bridge + bot config from a JSON file path or an already-parsed config dict.

    The ``mt5_trade`` section may sit at the top level of the file, or the file itself may be
    the section. Reuses freqtrade's config file loader for path inputs.
    """
    if isinstance(config, str):
        from freqtrade.configuration.load_config import load_config_file

        config = load_config_file(config)

    if "mt5_trade" in config:
        section = config["mt5_trade"]
    elif "symbols" in config:
        section = config
    else:
        raise OperationalException(
            "No 'mt5_trade' section found in config (expected a top-level 'mt5_trade' object "
            "or a config that is the section itself)."
        )

    if not isinstance(section, dict):
        raise OperationalException("The 'mt5_trade' config section must be an object.")

    return parse_mt5_config(section)
