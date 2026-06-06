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
