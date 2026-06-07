from __future__ import annotations

import importlib
import logging
import sys
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from freqtrade.exceptions import OperationalException
from freqtrade.mt5_trade.bot import MT5ForexBot
from freqtrade.mt5_trade.data import LiveMT5DataFeed, MT5DataFeed, ReplayDataFeed
from freqtrade.mt5_trade.execution import MT5ExecutionBridge
from freqtrade.mt5_trade.gateway import LazyMT5Gateway
from freqtrade.mt5_trade.models import MT5BotConfig, MT5BridgeConfig
from freqtrade.mt5_trade.notifier import Notifier
from freqtrade.mt5_trade.persistence import MT5TradeStore
from freqtrade.mt5_trade.sizing import PositionSizer
from freqtrade.mt5_trade.strategy import MT5Strategy, SmaCrossStrategy


logger = logging.getLogger(__name__)


def build_default_strategy(extra: dict) -> MT5Strategy:
    """
    Build the bot's strategy from the config's optional ``strategy`` section.

    With a ``class`` dotted path ("module.ClassName"), a custom ``MT5Strategy`` is imported and
    constructed with the remaining ``strategy`` keys as keyword arguments. An optional ``path``
    (a directory, or a ``.py`` file whose directory is used) is prepended to ``sys.path`` so
    user modules outside the package are importable. Without ``class`` the built-in
    ``SmaCrossStrategy`` is used.
    """
    raw = extra.get("strategy", {})
    if not isinstance(raw, dict):
        raise OperationalException("mt5_trade.strategy must be an object.")
    params = dict(raw)

    class_path = params.pop("class", None)
    search_path = params.pop("path", None)
    if class_path:
        return _load_custom_strategy(str(class_path), search_path, params)

    return SmaCrossStrategy(
        fast=int(params.get("fast", 10)),
        slow=int(params.get("slow", 30)),
        stop_loss_distance=_optional_float(params.get("stop_loss_distance")),
        take_profit_distance=_optional_float(params.get("take_profit_distance")),
    )


def _load_custom_strategy(
    class_path: str, search_path: Any, params: dict[str, Any]
) -> MT5Strategy:
    if search_path:
        _prepend_sys_path(str(search_path))
    strategy_cls = _load_strategy_class(class_path)
    # Constructor kwargs are the remaining strategy keys (class/path already removed).
    try:
        return strategy_cls(**params)
    except TypeError as exc:
        raise OperationalException(
            f"Failed to construct strategy {class_path!r} with {sorted(params)}: {exc}"
        ) from exc


def _prepend_sys_path(path_str: str) -> None:
    target = Path(path_str).expanduser()
    directory = target.parent if target.suffix == ".py" else target
    resolved = str(directory.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def _load_strategy_class(class_path: str) -> type[MT5Strategy]:
    module_name, _, class_name = class_path.rpartition(".")
    if not module_name or not class_name:
        raise OperationalException(
            f"strategy.class {class_path!r} must be a dotted path like 'module.ClassName'."
        )
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise OperationalException(
            f"Cannot import strategy module {module_name!r}: {exc}"
        ) from exc

    cls = getattr(module, class_name, None)
    if cls is None:
        raise OperationalException(
            f"Strategy class {class_name!r} not found in module {module_name!r}."
        )
    if not (isinstance(cls, type) and issubclass(cls, MT5Strategy)):
        raise OperationalException(
            f"strategy.class {class_path!r} is not an MT5Strategy subclass."
        )
    return cls


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
        account_balance = _optional_float(self._bridge_config.extra.get("starting_balance"))
        position_sizer = PositionSizer.from_config(
            self._bridge_config.extra,
            default_lot_size=self._bridge_config.default_lot_size,
            contract_size=float(self._bridge_config.extra.get("contract_size", 1.0)),
        )
        if position_sizer.requires_balance and account_balance is None:
            raise OperationalException("risk_percent position sizing requires starting_balance.")

        return MT5ForexBot(
            bridge=bridge,
            feed=feed,
            strategy=strategy,
            store=store,
            bot_config=self._bot_config,
            default_volume=self._bridge_config.default_lot_size,
            position_sizer=position_sizer,
            symbol_mappings=_symbol_mapping_lookup(self._bridge_config),
            account_balance=account_balance,
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


def _optional_float(value) -> float | None:
    return float(value) if value is not None else None


def _symbol_mapping_lookup(config: MT5BridgeConfig):
    mappings = {}
    for mapping in config.symbols:
        pair = f"{mapping.base}{mapping.quote}".upper()
        mappings[mapping.mt5_symbol] = mapping
        mappings[mapping.mt5_symbol.upper()] = mapping
        mappings[mapping.instrument_id] = mapping
        mappings[mapping.pair] = mapping
        mappings[pair] = mapping
    return mappings
