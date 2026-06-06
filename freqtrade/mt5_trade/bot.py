from __future__ import annotations

import logging
import time
from collections.abc import Callable

from freqtrade.mt5_trade.data import MT5DataFeed
from freqtrade.mt5_trade.execution import MT5ExecutionBridge
from freqtrade.mt5_trade.models import MT5BotConfig, MT5OrderRequest
from freqtrade.mt5_trade.persistence import MT5TradeStore
from freqtrade.mt5_trade.strategy import MT5Strategy, Signal


logger = logging.getLogger(__name__)

# Action -> the market order side that opens that position.
_ENTRY_SIDE = {"enter_long": "buy", "enter_short": "sell"}
# A position side -> the opposite market order side that closes it.
_CLOSE_SIDE = {"buy": "sell", "sell": "buy"}


class MT5ForexBot:
    """
    Forex trading loop: pull bars -> ask the strategy -> manage one position per symbol ->
    submit orders through the MT5 execution bridge -> persist the result.

    The loop is fully driven by injected collaborators so it runs and is tested offline with a
    ReplayDataFeed + dry-run bridge; live trading swaps in LiveMT5DataFeed + a connected gateway.
    """

    def __init__(
        self,
        bridge: MT5ExecutionBridge,
        feed: MT5DataFeed,
        strategy: MT5Strategy,
        store: MT5TradeStore,
        bot_config: MT5BotConfig,
        default_volume: float = 0.01,
    ) -> None:
        self._bridge = bridge
        self._feed = feed
        self._strategy = strategy
        self._store = store
        self._config = bot_config
        self._default_volume = default_volume
        self._running = False
        self._order_seq = 0
        # symbol -> (order side that is open, volume). Restored from the store on startup.
        self._positions: dict[str, tuple[str, float]] = {
            symbol: (pos.side, pos.volume)
            for symbol, pos in store.open_positions().items()
        }

    @property
    def running(self) -> bool:
        return self._running

    def run_once(self) -> None:
        """Evaluate every configured symbol exactly once."""
        for symbol in self._config.symbols:
            bars = self._feed.latest_bars(symbol, self._config.warmup_bars)
            if not bars:
                continue
            signal = self._strategy.on_bar(symbol, bars)
            self._handle_signal(symbol, signal, reference_price=bars[-1].close)

    def run(self, sleep: Callable[[float], None] = time.sleep) -> None:
        """
        Run the loop until the feed is exhausted (replay) or interrupted (live).

        ``sleep`` is injectable so tests don't actually wait.
        """
        self._running = True
        logger.info("MT5 forex bot started (symbols=%s).", list(self._config.symbols))
        try:
            while self._running:
                self.run_once()
                if not self._feed.advance():
                    logger.info("Data feed exhausted; stopping bot.")
                    break
                sleep(self._config.poll_interval)
        except KeyboardInterrupt:
            logger.info("MT5 forex bot interrupted; shutting down.")
        finally:
            self.stop()

    def stop(self) -> None:
        self._running = False
        self._bridge.close()
        self._feed.close()

    def _handle_signal(self, symbol: str, signal: Signal, reference_price: float) -> None:
        if signal.action == "hold":
            return

        current = self._positions.get(symbol)

        if signal.action == "exit":
            if current is not None:
                self._close(symbol, current, signal)
            return

        desired_side = _ENTRY_SIDE[signal.action]
        if current is not None and current[0] == desired_side:
            # Already in the desired direction; do not stack another position.
            return
        if current is not None:
            # Reverse: close the opposing position before opening the new one.
            self._close(symbol, current, signal)

        self._open(symbol, desired_side, signal, reference_price)

    def _open(self, symbol: str, side: str, signal: Signal, reference_price: float) -> None:
        volume = signal.volume if signal.volume is not None else self._default_volume
        order = self._build_order(symbol, side, volume, signal)
        result = self._bridge.submit_order(order)
        self._store.record_order(order, result)
        if result.accepted:
            self._positions[symbol] = (side, volume)
            self._store.open_position(symbol, side, volume, reference_price)
            logger.info("Opened %s %s %.2f (%s).", side, symbol, volume, result.order_id)
        else:
            logger.warning("Open %s %s rejected: %s", side, symbol, result.message)

    def _close(self, symbol: str, position: tuple[str, float], signal: Signal) -> None:
        open_side, volume = position
        close_side = _CLOSE_SIDE[open_side]
        order = self._build_order(symbol, close_side, volume, signal)
        result = self._bridge.submit_order(order)
        self._store.record_order(order, result)
        if result.accepted:
            self._positions.pop(symbol, None)
            self._store.close_position(symbol)
            logger.info("Closed %s %s %.2f (%s).", open_side, symbol, volume, result.order_id)
        else:
            logger.warning("Close %s %s rejected: %s", open_side, symbol, result.message)

    def _build_order(
        self, symbol: str, side: str, volume: float, signal: Signal
    ) -> MT5OrderRequest:
        self._order_seq += 1
        return MT5OrderRequest(
            symbol=symbol,
            side=side,  # type: ignore[arg-type]  # validated by MT5OrderRequest
            volume=volume,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            client_order_id=f"{symbol}-{self._order_seq}",
            comment=signal.comment,
        )
