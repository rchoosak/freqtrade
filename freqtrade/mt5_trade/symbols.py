from __future__ import annotations


def normalize_forex_symbol(symbol: str) -> str:
    cleaned = symbol.strip()
    venue_suffix = ".MT5"
    if cleaned.upper().endswith(venue_suffix):
        cleaned = cleaned[: -len(venue_suffix)]

    symbol_parts = cleaned.split(".", maxsplit=1)
    root = symbol_parts[0].replace("/", "").replace("_", "").replace("-", "").upper()
    if len(symbol_parts) == 2:
        return f"{root}.{symbol_parts[1]}"
    return root


def to_instrument_id(symbol: str, venue: str = "MT5") -> str:
    normalized = normalize_forex_symbol(symbol)
    root = normalized.split(".", maxsplit=1)[0]
    if len(root) != 6:
        raise ValueError(f"Expected a 6-letter forex symbol, got {symbol!r}.")
    return f"{normalized}.{venue.upper()}"


def instrument_id_to_mt5_symbol(instrument_id: str) -> str:
    return normalize_forex_symbol(instrument_id)
