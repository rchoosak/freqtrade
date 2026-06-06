"""
mt5-trade forex bridge and bot.

Connects to a MetaTrader5 terminal directly. The optional ``MetaTrader5`` import is kept lazy so
the main Freqtrade project remains importable on platforms where that runtime dependency is
unavailable (it is Windows-only).
"""

from freqtrade.mt5_trade.backtest import BacktestResult, BacktestTrade, run_backtest
from freqtrade.mt5_trade.bot import MT5ForexBot
from freqtrade.mt5_trade.config import load_mt5_config, parse_mt5_config
from freqtrade.mt5_trade.data import (
    LiveMT5DataFeed,
    MT5Bar,
    MT5DataFeed,
    ReplayDataFeed,
    dump_bars_json,
    load_bars_json,
)
from freqtrade.mt5_trade.execution import MT5ExecutionBridge
from freqtrade.mt5_trade.history import MT5HistoryDownloader
from freqtrade.mt5_trade.models import (
    BrokerOrder,
    BrokerPosition,
    MT5BotConfig,
    MT5BridgeConfig,
    MT5OrderRequest,
    MT5OrderResult,
    MT5SymbolMapping,
)
from freqtrade.mt5_trade.notifier import (
    LoggingNotifier,
    Notifier,
    NullNotifier,
    RPCNotifier,
)
from freqtrade.mt5_trade.persistence import MT5TradeStore
from freqtrade.mt5_trade.position import OrderIntent, plan_transitions
from freqtrade.mt5_trade.runner import MT5TradeRuntime
from freqtrade.mt5_trade.strategy import MT5Strategy, Signal, SmaCrossStrategy
from freqtrade.mt5_trade.symbols import (
    instrument_id_to_mt5_symbol,
    normalize_forex_symbol,
    to_instrument_id,
)


__all__ = [
    "BacktestResult",
    "BacktestTrade",
    "BrokerOrder",
    "BrokerPosition",
    "LiveMT5DataFeed",
    "LoggingNotifier",
    "MT5Bar",
    "MT5BotConfig",
    "MT5BridgeConfig",
    "MT5DataFeed",
    "MT5ExecutionBridge",
    "MT5ForexBot",
    "MT5HistoryDownloader",
    "MT5OrderRequest",
    "MT5OrderResult",
    "MT5Strategy",
    "MT5SymbolMapping",
    "MT5TradeRuntime",
    "MT5TradeStore",
    "Notifier",
    "NullNotifier",
    "OrderIntent",
    "RPCNotifier",
    "ReplayDataFeed",
    "Signal",
    "SmaCrossStrategy",
    "dump_bars_json",
    "instrument_id_to_mt5_symbol",
    "load_bars_json",
    "load_mt5_config",
    "normalize_forex_symbol",
    "parse_mt5_config",
    "plan_transitions",
    "run_backtest",
    "to_instrument_id",
]
