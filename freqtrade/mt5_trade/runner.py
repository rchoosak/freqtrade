from __future__ import annotations

import logging
from importlib.util import find_spec

from freqtrade.exceptions import OperationalException
from freqtrade.mt5_trade.bot import MT5ForexBot
from freqtrade.mt5_trade.data import LiveMT5DataFeed, MT5DataFeed, ReplayDataFeed
from freqtrade.mt5_trade.execution import MT5ExecutionBridge
from freqtrade.mt5_trade.gateway import LazyMT5Gateway
from freqtrade.mt5_trade.models import MT5BotConfig, MT5BridgeConfig
from freqtrade.mt5_trade.notifier import Notifier
from freqtrade.mt5_trade.persistence import MT5TradeStore
from freqtrade.mt5_trade.strategy import MT5Strategy, SmaCrossStrategy


logger = logging.getLogger(__name__)


def build_default_strategy(extra: dict) -> MT5Strategy:
    """Build the bot's strategy from the config's optional ``strategy`` parameters."""
    params = extra.get("strategy", {})
    return SmaCrossStrategy(fast=int(params.get("fast", 10)), slow=int(params.get("slow", 30)))


class MT5TradeRuntime:
    """
    Assembles and runs the MT5 forex bot from typed config.

    Connects to a MetaTrader5 terminal directly. When ``replay_data`` is configured
    (offline dry-run / tests) it replays bars from a file instead of the live feed.
    """

    def __init__(
        self,
        bridge_config: MT5BridgeConfig,
        bot_config: MT5BotConfig,
        *,
        strategy: MT5Strategy | None = None,
        feed: MT5DataFeed | None = None,
        store: MT5TradeStore | None = None,
        notifier: Notifier | None = None,
    ) -> None:
        self._bridge_config = bridge_config
        self._bot_config = bot_config
        self._strategy = strategy
        self._feed = feed
        self._store = store
        self._notifier = notifier

    def validate_environment(self) -> None:
        # The MetaTrader5 package (Windows-only) is required only for live trading; dry-run with
        # a replay feed needs no external runtime.
        if not self._bridge_config.dry_run and find_spec("MetaTrader5") is None:
            raise OperationalException(
                "MetaTrader5 is not installed. Install it on the Windows host running the MT5 "
                "terminal before starting live MT5 trading."
            )

    def build(self) -> MT5ForexBot:
        self.validate_environment()

        gateway = LazyMT5Gateway(self._bridge_config)
        bridge = MT5ExecutionBridge(self._bridge_config, gateway)
        store = self._store or MT5TradeStore(self._bot_config.db_path)
        strategy = self._strategy or self._build_strategy()
        feed = self._feed or self._build_feed(gateway)

        return MT5ForexBot(
            bridge=bridge,
            feed=feed,
            strategy=strategy,
            store=store,
            bot_config=self._bot_config,
            default_volume=self._bridge_config.default_lot_size,
            notifier=self._notifier,
        )

    def start(self) -> None:
        bot = self.build()
        bot.run()

    def _build_strategy(self) -> MT5Strategy:
        return build_default_strategy(self._bridge_config.extra)

    def _build_feed(self, gateway: LazyMT5Gateway) -> MT5DataFeed:
        replay_path = self._bridge_config.extra.get("replay_data")
        if replay_path:
            logger.info("Using replay data feed from %s.", replay_path)
            return ReplayDataFeed.from_json(
                str(replay_path), warmup=self._bot_config.warmup_bars
            )
        return LiveMT5DataFeed(gateway, timeframe=self._bot_config.timeframe)
