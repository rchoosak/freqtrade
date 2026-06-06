import logging
import signal
from typing import Any

from freqtrade.exceptions import OperationalException


logger = logging.getLogger(__name__)


def start_trading(args: dict[str, Any]) -> int:
    """
    Main entry point for trading mode
    """
    # Import here to avoid loading worker module when it's not used
    from freqtrade.worker import Worker

    def term_handler(signum, frame):
        # Raise KeyboardInterrupt - so we can handle it in the same way as Ctrl-C
        raise KeyboardInterrupt()

    # Create and run worker
    worker = None
    try:
        signal.signal(signal.SIGTERM, term_handler)
        worker = Worker(args)
        worker.run()
    finally:
        if worker:
            logger.info("worker found ... calling exit")
            worker.exit()
    return 0


def start_trading_mt5(args: dict[str, Any]) -> int:
    """
    Entry point for the MT5 forex trading bot.

    Connects to a MetaTrader5 terminal directly (or replays bars offline in dry-run) and runs
    the forex bot loop. Separate from `start_trading`, which drives the CCXT crypto bot.
    """
    # The MT5 config file is standalone (an 'mt5_trade' section), not a full crypto-bot config,
    # so it is loaded directly rather than through Configuration/CONF_SCHEMA validation.
    from freqtrade.loggers import setup_logging_pre
    from freqtrade.mt5_trade.config import load_mt5_config
    from freqtrade.mt5_trade.runner import MT5TradeRuntime

    def term_handler(signum, frame):
        raise KeyboardInterrupt()

    setup_logging_pre()

    config_files = args.get("config") or []
    if not config_files:
        raise OperationalException(
            "trade-mt5 requires a --config file with an 'mt5_trade' section."
        )
    bridge_config, bot_config = load_mt5_config(config_files[0])

    signal.signal(signal.SIGTERM, term_handler)
    runtime = MT5TradeRuntime(bridge_config, bot_config)
    runtime.start()
    return 0


def _load_mt5_config_arg(args: dict[str, Any]):
    from freqtrade.mt5_trade.config import load_mt5_config

    config_files = args.get("config") or []
    if not config_files:
        raise OperationalException(
            "This command requires a --config file with an 'mt5_trade' section."
        )
    return load_mt5_config(config_files[0])


def start_download_data_mt5(args: dict[str, Any]) -> int:
    """Download historical MT5 bars and cache them as JSON for backtesting/replay."""
    from freqtrade.loggers import setup_logging_pre
    from freqtrade.mt5_trade.data import LiveMT5DataFeed
    from freqtrade.mt5_trade.gateway import LazyMT5Gateway
    from freqtrade.mt5_trade.history import MT5HistoryDownloader

    setup_logging_pre()
    bridge_config, bot_config = _load_mt5_config_arg(args)

    count = int(bridge_config.extra.get("history_bars", 1000))
    out_path = str(bridge_config.extra.get("history_file", "mt5_bars.json"))
    feed = LiveMT5DataFeed(LazyMT5Gateway(bridge_config), timeframe=bot_config.timeframe)
    MT5HistoryDownloader(feed).download_to_json(bot_config.symbols, count, out_path)
    logger.info("Saved %d-bar history for %s to %s.", count, list(bot_config.symbols), out_path)
    return 0


def start_backtest_mt5(args: dict[str, Any]) -> int:
    """Backtest the configured strategy over a cached JSON bar file and print a report."""
    from freqtrade.loggers import setup_logging_pre
    from freqtrade.mt5_trade.backtest import run_backtest
    from freqtrade.mt5_trade.data import load_bars_json
    from freqtrade.mt5_trade.runner import build_default_strategy

    setup_logging_pre()
    bridge_config, bot_config = _load_mt5_config_arg(args)

    data_path = bridge_config.extra.get("replay_data") or bridge_config.extra.get("history_file")
    if not data_path:
        raise OperationalException(
            "backtest-mt5 needs a 'replay_data' (or 'history_file') path in the config."
        )
    data = load_bars_json(str(data_path))
    result = run_backtest(
        build_default_strategy(bridge_config.extra),
        data,
        default_volume=bridge_config.default_lot_size,
        warmup_bars=bot_config.warmup_bars,
    )

    logger.info(
        "Backtest: trades=%d total_pnl=%.5f win_rate=%.1f%%",
        result.num_trades,
        result.total_pnl,
        result.win_rate * 100,
    )
    for trade in result.trades:
        logger.info(
            "  %s %s vol=%s entry=%s exit=%s pnl=%.5f",
            trade.symbol,
            trade.side,
            trade.volume,
            trade.entry_price,
            trade.exit_price,
            trade.pnl,
        )
    return 0
