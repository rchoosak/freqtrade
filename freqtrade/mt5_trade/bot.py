from __future__ import annotations

import logging
import time
from collections.abc import Callable

from freqtrade.mt5_trade.data import MT5DataFeed
from freqtrade.mt5_trade.execution import MT5ExecutionBridge
from freqtrade.mt5_trade.models import MT5BotConfig, MT5OrderRequest, OrderSide
from freqtrade.mt5_trade.notifier import LoggingNotifier, Notifier
from freqtrade.mt5_trade.persistence import MT5TradeStore
from freqtrade.mt5_trade.position import OrderIntent, plan_transitions
from freqtrade.mt5_trade.strategy import MT5Strategy, Signal


logger = logging.getLogger(__name__)


class MT5ForexBot:
    """
    Forex trading loop: pull bars -> ask the strategy -> manage one position per symbol ->
    submit orders through the MT5 execution bridge -> persist the result.

    The loop is fully driven by injected collaborators so it runs and is tested offline with a
    ReplayDataFeed + dry-run bridge; live trading swaps in LiveMT5DataFeed + a connected gateway.
    Position keys are the symbols the bot trades (mt5 symbols by default), matching the broker's
    reported position symbols for reconciliation.
    """

    def __init__(
        self,
        bridge: MT5ExecutionBridge,
        feed: MT5DataFeed,
        strategy: MT5Strategy,
        store: MT5TradeStore,
        bot_config: MT5BotConfig,
        default_volume: float = 0.01,
        notifier: Notifier | None = None,
        max_consecutive_errors: int = 5,
    ) -> None:
        self._bridge = bridge
        self._feed = feed
        self._strategy = strategy
        self._store = store
        self._config = bot_config
        self._default_volume = default_volume
        self._notifier = notifier or LoggingNotifier()
        self._max_consecutive_errors = max_consecutive_errors
        self._running = False
        self._order_seq = 0
        # symbol -> (open order side, volume). Restored from the store on startup.
        self._positions: dict[str, tuple[OrderSide, float]] = {
            symbol: (pos.side, pos.volume)  # type: ignore[misc]
            for symbol, pos in store.open_positions().items()
        }
        # symbol -> (side, volume, ticket) for resting pending orders the bot placed.
        # In-memory only; rebuilt from the broker via reconcile() (no broker in dry-run).
        self._pendings: dict[str, tuple[OrderSide, float, int | None]] = {}

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

        Reconciles broker state at startup and (optionally) on an interval, isolates per-
        iteration failures so a transient error does not crash the bot, and stops after too many
        consecutive errors. ``sleep`` is injectable so tests don't actually wait.
        """
        self._running = True
        self._notify(f"MT5 forex bot started (symbols={list(self._config.symbols)}).")
        self.reconcile()
        iteration = 0
        consecutive_errors = 0
        try:
            while self._running:
                iteration += 1
                try:
                    self.run_once()
                    consecutive_errors = 0
                except Exception as exc:  # keep the loop alive across transient failures
                    consecutive_errors += 1
                    logger.exception("MT5 bot iteration failed (%d in a row).", consecutive_errors)
                    self._notify(f"Iteration error: {exc}")
                    if consecutive_errors >= self._max_consecutive_errors:
                        self._notify("Too many consecutive errors; stopping bot.")
                        break

                if (
                    self._config.reconcile_interval > 0
                    and iteration % self._config.reconcile_interval == 0
                ):
                    self._safe_reconcile()

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

    def reconcile(self) -> None:
        """
        Align in-memory + stored positions and pending orders with the broker's reality
        (broker is authoritative).

        No-op in dry-run (no broker). Adopts broker positions/orders the bot didn't know about
        and drops ones that were closed or cancelled externally, so the bot never trades on a
        stale picture. A pending order that filled moves from the pending slot to a position.
        """
        positions = self._bridge.broker_positions()
        orders = self._bridge.broker_orders()
        if positions is None and orders is None:
            return

        changes: list[str] = []
        self._reconcile_positions(positions or [], changes)
        self._reconcile_pendings(orders or [], changes)

        if changes:
            self._notify("Reconciled: " + ", ".join(changes))

    def _reconcile_positions(self, positions: list, changes: list[str]) -> None:
        broker = {pos.symbol: (pos.side, pos.volume) for pos in positions}
        for symbol, state in broker.items():
            if self._positions.get(symbol) != state:
                self._positions[symbol] = state
                self._store.open_position(symbol, state[0], state[1], None)
                changes.append(f"{symbol}->{state[0]} {state[1]}")
        for symbol in list(self._positions):
            if symbol not in broker:
                self._positions.pop(symbol, None)
                self._store.close_position(symbol)
                changes.append(f"{symbol}->flat")

    def _reconcile_pendings(self, orders: list, changes: list[str]) -> None:
        broker = {o.symbol: (o.side, o.volume, o.ticket) for o in orders}
        for symbol, state in broker.items():
            # A filled order is a position now (handled above); don't double-occupy the slot.
            if symbol not in self._positions and self._pendings.get(symbol) != state:
                self._pendings[symbol] = state
                changes.append(f"{symbol} pending {state[0]}")
        for symbol in list(self._pendings):
            if symbol not in broker or symbol in self._positions:
                self._pendings.pop(symbol, None)
                changes.append(f"{symbol} pending cleared")

    def _safe_reconcile(self) -> None:
        try:
            self.reconcile()
        except Exception as exc:
            logger.exception("Reconciliation failed.")
            self._notify(f"Reconciliation error: {exc}")

    def _current_slot(self, symbol: str) -> tuple[OrderSide, float] | None:
        """The side/volume occupying a symbol, whether a filled position or a resting order."""
        if symbol in self._positions:
            return self._positions[symbol]
        if symbol in self._pendings:
            side, volume, _ticket = self._pendings[symbol]
            return (side, volume)
        return None

    def _handle_signal(self, symbol: str, signal: Signal, reference_price: float) -> None:
        current = self._current_slot(symbol)
        for intent in plan_transitions(current, signal, self._default_volume):
            self._execute(symbol, intent, signal, reference_price)

    def _execute(
        self, symbol: str, intent: OrderIntent, signal: Signal, reference_price: float
    ) -> None:
        is_open = intent.result is not None

        # Closing a symbol whose slot is a resting order means cancelling that order, not
        # sending a market close for a position the bot does not hold.
        if not is_open and symbol in self._pendings:
            self._cancel_pending(symbol)
            return

        order = self._build_order(symbol, intent, signal, is_open)
        result = self._bridge.submit_order(order)
        self._store.record_order(order, result)

        if not result.accepted:
            logger.warning(
                "%s %s %s rejected: %s", intent.reason, intent.side, symbol, result.message
            )
            self._notify(f"{intent.reason} {intent.side} {symbol} rejected: {result.message}")
            return

        if is_open and result.is_pending:
            # A resting limit/stop order is not a position yet; reconcile() adopts the fill later.
            oid = result.order_id
            ticket = int(oid) if oid is not None and oid.isdigit() else None
            self._pendings[symbol] = (intent.side, intent.volume, ticket)
            self._notify(f"pending {intent.side} {symbol} {intent.volume} ({result.order_id})")
            return

        if is_open:
            # A partial fill means the broker executed less than requested; track what we got.
            filled = result.filled_volume
            volume = filled if filled is not None and filled > 0 else intent.volume
            self._positions[symbol] = (intent.side, volume)
            self._store.open_position(symbol, intent.side, volume, reference_price)
            self._apply_sltp(symbol, signal)
        else:
            self._positions.pop(symbol, None)
            self._store.close_position(symbol)

        self._notify(
            f"{intent.reason} {intent.side} {symbol} {intent.volume} ({result.order_id})"
        )

    def _cancel_pending(self, symbol: str) -> None:
        side, _volume, ticket = self._pendings[symbol]
        if ticket is None:
            # Nothing to cancel at the broker; just forget the local slot.
            self._pendings.pop(symbol, None)
            return
        result = self._bridge.cancel_order(ticket)
        if result.accepted:
            self._pendings.pop(symbol, None)
            self._notify(f"cancelled pending {side} {symbol} ({ticket})")
        else:
            logger.warning("Cancel pending %s (%s) rejected: %s", symbol, ticket, result.message)
            self._notify(f"cancel pending {symbol} rejected: {result.message}")

    def _apply_sltp(self, symbol: str, signal: Signal) -> None:
        if signal.stop_loss is None and signal.take_profit is None:
            return
        result = self._bridge.modify_sltp(symbol, signal.stop_loss, signal.take_profit)
        if not result.accepted:
            logger.warning("SL/TP modify for %s rejected: %s", symbol, result.message)
            self._notify(f"SL/TP {symbol} rejected: {result.message}")

    def _build_order(
        self, symbol: str, intent: OrderIntent, signal: Signal, is_open: bool
    ) -> MT5OrderRequest:
        self._order_seq += 1
        return MT5OrderRequest(
            symbol=symbol,
            side=intent.side,
            volume=intent.volume,
            order_kind=intent.order_kind,
            price=intent.price,
            # SL/TP travel on the broker-side modify, not the entry order, so the same code
            # path works for market entries and later adjustments.
            client_order_id=f"{symbol}-{self._order_seq}",
            comment=signal.comment,
        )

    def _notify(self, message: str) -> None:
        try:
            self._notifier.send(message)
        except Exception:
            logger.exception("Notifier failed for message: %s", message)
