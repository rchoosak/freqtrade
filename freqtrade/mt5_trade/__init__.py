"""
mt5-trade forex bridge and bot.

Connects to a MetaTrader5 terminal directly. The optional ``MetaTrader5`` import is kept lazy so
the main Freqtrade project remains importable on platforms where that runtime dependency is
unavailable (it is Windows-only).
"""

from freqtrade.mt5_trade.bot import MT5ForexBot
from freqtrade.mt5_trade.config import load_mt5_config, parse_mt5_config
from freqtrade.mt5_trade.data import (
    LiveMT5DataFeed,
    MT5Bar,
    MT5DataFeed,
    ReplayDataFeed,
)
from freqtrade.mt5_trade.execution import MT5ExecutionBridge
from freqtrade.mt5_trade.models import (
    MT5BotConfig,
    MT5BridgeConfig,
    MT5OrderRequest,
    MT5OrderResult,
    MT5SymbolMapping,
)
from freqtrade.mt5_trade.persistence import MT5TradeStore
from freqtrade.mt5_trade.runner import MT5TradeRuntime
from freqtrade.mt5_trade.strategy import MT5Strategy, Signal, SmaCrossStrategy
from freqtrade.mt5_trade.symbols import (
    instrument_id_to_mt5_symbol,
    normalize_forex_symbol,
    to_instrument_id,
)


__all__ = [
    "LiveMT5DataFeed",
    "MT5Bar",
    "MT5BotConfig",
    "MT5BridgeConfig",
    "MT5DataFeed",
    "MT5ExecutionBridge",
    "MT5ForexBot",
    "MT5OrderRequest",
    "MT5OrderResult",
    "MT5Strategy",
    "MT5SymbolMapping",
    "MT5TradeRuntime",
    "MT5TradeStore",
    "ReplayDataFeed",
    "Signal",
    "SmaCrossStrategy",
    "instrument_id_to_mt5_symbol",
    "load_mt5_config",
    "normalize_forex_symbol",
    "parse_mt5_config",
    "to_instrument_id",
]
