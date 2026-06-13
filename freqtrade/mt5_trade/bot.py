from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, replace

from freqtrade.exceptions import OperationalException
from freqtrade.mt5_trade.data import MT5Bar, MT5DataFeed
from freqtrade.mt5_trade.execution import MT5ExecutionBridge
from freqtrade.mt5_trade.models import (
    MT5BotConfig,
    MT5OrderRequest,
    MT5OrderResult,
    MT5SymbolMapping,
    OrderSide,
    split_lot,
)
from freqtrade.mt5_trade.notifier import LoggingNotifier, Notifier
from freqtrade.mt5_trade.persistence import MT5TradeStore
from freqtrade.mt5_trade.position import OrderIntent, plan_transitions
from freqtrade.mt5_trade.sizing import PositionSizer, entry_side_for_action
from freqtrade.mt5_trade.strategy import MT5Strategy, Signal


logger = logging.getLogger(__name__)


@dataclass
class _Managed:
    """Scale-out bookkeeping for an open position the bot is managing toward TP1/breakeven."""

    side: OrderSide
    entry_price: float
    tp1: float
    close_fraction: float
    move_be: bool
    scaled: bool = False


def _target_crossed(side: OrderSide, level: float, bar: MT5Bar) -> bool:
    # A favorable target: longs hit it on the bar high, shorts on the bar low.
    return bar.high >= level if side == "buy" else bar.low <= level


def _resolved_volume(result: MT5OrderResult, fallback: float) -> float:
    """
    Volume the broker actually acted on: the fill if reported, else the broker-normalized request
    volume (which can differ from the bot's config-normalized request), else the fallback.
    """
    if result.filled_volume is not None and result.filled_volume > 0:
        return result.filled_volume
    if result.requested_volume is not None and result.requested_volume > 0:
        return result.requested_volume
    return fallback


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
        position_sizer: PositionSizer | None = None,
        symbol_mappings: dict[str, MT5SymbolMapping] | None = None,
        account_balance: float | None = None,
        notifier: Notifier | None = None,
        max_consecutive_errors: int = 5,
    ) -> None:
        self._bridge = bridge
        self._feed = feed
        self._strategy = strategy
        self._store = store
        self._config = bot_config
        self._default_volume = default_volume
        self._position_sizer = position_sizer or PositionSizer(fixed_lot_size=default_volume)
        self._symbol_mappings = symbol_mappings or {}
        self._account_balance = account_balance
        self._notifier = notifier or LoggingNotifier()
        self._max_consecutive_errors = max_consecutive_errors
        self._running = False
        self._order_seq = 0
        self._iterations = 0
        # symbol -> (open order side, volume). Restored from the store on startup.
        self._positions: dict[str, tuple[OrderSide, float]] = {
            symbol: (pos.side, pos.volume)  # type: ignore[misc]
            for symbol, pos in store.open_positions().items()
        }
        # symbol -> (side, volume, ticket) for resting pending orders the bot placed.
        # In-memory only; rebuilt from the broker via reconcile() (no broker in dry-run).
        self._pendings: dict[str, tuple[OrderSide, float, int | None]] = {}
        # symbol -> the iteration a pending order was placed, for bot-side expiry.
        self._pending_placed: dict[str, int] = {}
        # symbol -> scale-out bookkeeping for positions opened with a TP1 plan.
        self._managed: dict[str, _Managed] = self._restore_managed(store)

    @staticmethod
    def _restore_managed(store: MT5TradeStore) -> dict[str, _Managed]:
        managed: dict[str, _Managed] = {}
        open_positions = store.open_positions()
        for symbol, state in store.managed_positions().items():
            if symbol not in open_positions:
                store.clear_managed_position(symbol)
                continue
            managed[symbol] = _Managed(
                side=state.side,  # type: ignore[arg-type]
                entry_price=state.entry_price,
                tp1=state.tp1,
                close_fraction=state.close_fraction,
                move_be=state.move_be,
                scaled=state.scaled,
            )
        return managed

    @property
    def running(self) -> bool:
        return self._running

    def run_once(self) -> None:
        """Evaluate every configured symbol exactly once."""
        self._iterations += 1
        for symbol in self._config.symbols:
            bars = self._feed.latest_bars(symbol, self._config.warmup_bars)
            if not bars:
                continue
            # Scale out before asking the strategy, mirroring the backtester's ordering.
            self._manage_scale_out(symbol, bars[-1])
            signal = self._strategy.on_bar(symbol, bars)
            self._handle_signal(symbol, signal, reference_price=bars[-1].close)
        self._expire_pendings()

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
                self._managed.pop(symbol, None)
                self._store.clear_managed_position(symbol)
                self._store.close_position(symbol)
                # The broker closed this position (e.g. SL/TP); let the strategy reset state.
                self._strategy.on_position_closed(symbol)
                changes.append(f"{symbol}->flat")

    def _reconcile_pendings(self, orders: list, changes: list[str]) -> None:
        broker = {o.symbol: (o.side, o.volume, o.ticket) for o in orders}
        for symbol, state in broker.items():
            # A filled order is a position now (handled above); don't double-occupy the slot.
            if symbol not in self._positions and self._pendings.get(symbol) != state:
                self._set_pending(symbol, state)
                changes.append(f"{symbol} pending {state[0]}")
        for symbol in list(self._pendings):
            if symbol not in broker or symbol in self._positions:
                self._clear_pending(symbol)
                changes.append(f"{symbol} pending cleared")

    def _set_pending(self, symbol: str, state: tuple[OrderSide, float, int | None]) -> None:
        self._pendings[symbol] = state
        self._pending_placed[symbol] = self._iterations

    def _clear_pending(self, symbol: str) -> None:
        self._pendings.pop(symbol, None)
        self._pending_placed.pop(symbol, None)

    def _expire_pendings(self) -> None:
        """Cancel resting pending orders that have lived past ``pending_expiry`` iterations."""
        limit = self._config.pending_expiry
        if limit <= 0:
            return
        for symbol in list(self._pendings):
            age = self._iterations - self._pending_placed.get(symbol, self._iterations)
            if age >= limit:
                self._cancel_pending(symbol, reason="expired")

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
        sized_signal = self._size_signal(symbol, signal, current, reference_price)
        if sized_signal is None:
            return
        for intent in plan_transitions(current, sized_signal, self._default_volume):
            # If a close/cancel leg fails, abort the rest so a reversal never opens the opposite
            # side while the old position/pending is still live at the broker.
            if not self._execute(symbol, intent, signal, reference_price):
                break

    def _size_signal(
        self,
        symbol: str,
        signal: Signal,
        current: tuple[OrderSide, float] | None,
        reference_price: float,
    ) -> Signal | None:
        side = entry_side_for_action(signal.action)
        if side is None:
            return signal
        if signal.volume is not None:
            # Explicit volume still has to obey the broker lot rules before we track it.
            decision = self._position_sizer.snap(
                signal.volume, symbol=symbol, mapping=self._symbol_mappings.get(symbol)
            )
            if decision.skipped:
                message = decision.reason or f"{symbol}: explicit volume rejected by lot rules."
                logger.warning(message)
                self._notify(message)
                return None
            return replace(signal, volume=decision.volume)
        if current is not None and current[0] == side:
            return signal

        entry_price = signal.price if signal.price is not None else reference_price
        decision = self._position_sizer.size_entry(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            stop_loss=signal.stop_loss,
            balance=self._current_balance(),
            mapping=self._symbol_mappings.get(symbol),
        )
        if decision.skipped:
            message = decision.reason or f"{symbol}: position sizing skipped entry."
            logger.warning(message)
            self._notify(message)
            return None
        return replace(signal, volume=decision.volume)

    def _current_balance(self) -> float | None:
        # For risk-percent sizing, prefer the broker's live balance so sizing compounds with
        # realized PnL (the configured starting balance is only a startup fallback). Dry-run and
        # fixed sizing have no broker balance, so the configured value is used.
        if self._position_sizer.requires_balance:
            broker_balance = self._bridge.account_balance()
            if broker_balance is not None:
                return broker_balance
        return self._account_balance

    def _execute(
        self, symbol: str, intent: OrderIntent, signal: Signal, reference_price: float
    ) -> bool:
        """
        Execute one transition intent. Returns True when the caller may proceed to the next
        intent, False when it must abort (a close/cancel leg failed, so opening the opposite
        side would leave the old position/pending live alongside a new one).
        """
        is_open = intent.result is not None

        # Closing a symbol whose slot is a resting order means cancelling that order, not
        # sending a market close for a position the bot does not hold.
        if not is_open and symbol in self._pendings:
            return self._cancel_pending(symbol)

        order = self._build_order(symbol, intent, signal, is_open)
        result = self._bridge.submit_order(order)
        self._store.record_order(order, result)

        if not result.accepted:
            logger.warning(
                "%s %s %s rejected: %s", intent.reason, intent.side, symbol, result.message
            )
            self._notify(f"{intent.reason} {intent.side} {symbol} rejected: {result.message}")
            return False

        if is_open and result.is_pending:
            # Invariant (enforced at the Signal level): scale-out (tp1) is market-only. A pending
            # order rests until filled and is adopted via reconcile(), which has no path to
            # register _managed metadata — so a tp1 here would be silently dropped. Fail loud if
            # the Signal-level guard ever regresses.
            if intent.tp1 is not None:
                raise OperationalException(
                    f"scale-out (tp1) is not supported for pending entries: {symbol}"
                )
            # A resting limit/stop order is not a position yet; reconcile() adopts the fill later.
            oid = result.order_id
            ticket = int(oid) if oid is not None and oid.isdigit() else None
            self._set_pending(symbol, (intent.side, intent.volume, ticket))
            self._notify(f"pending {intent.side} {symbol} {intent.volume} ({result.order_id})")
            return True

        if is_open:
            # Track what the broker actually acted on (fill, else its normalized request volume),
            # which can differ from intent.volume when broker lot rules differ from the config.
            volume = _resolved_volume(result, intent.volume)
            # Use the broker's actual fill price as the entry (the last-candle close only
            # approximates it); breakeven and bookkeeping then reference the true entry.
            entry_price = result.fill_price if result.fill_price is not None else reference_price
            self._positions[symbol] = (intent.side, volume)
            self._store.open_position(symbol, intent.side, volume, entry_price)
            self._apply_sltp(symbol, signal)
            self._register_scale_out(symbol, intent, entry_price)
        else:
            self._positions.pop(symbol, None)
            self._managed.pop(symbol, None)
            self._store.close_position(symbol)
            self._strategy.on_position_closed(symbol)

        self._notify(
            f"{intent.reason} {intent.side} {symbol} {intent.volume} ({result.order_id})"
        )
        return True

    def _cancel_pending(self, symbol: str, reason: str = "cancelled") -> bool:
        """Cancel a resting pending order. Returns True if the slot is now clear, else False."""
        side, _volume, ticket = self._pendings[symbol]
        if ticket is None:
            # Nothing to cancel at the broker; just forget the local slot.
            self._clear_pending(symbol)
            return True
        result = self._bridge.cancel_order(ticket)
        if result.accepted:
            self._clear_pending(symbol)
            self._notify(f"{reason} pending {side} {symbol} ({ticket})")
            return True
        logger.warning("Cancel pending %s (%s) rejected: %s", symbol, ticket, result.message)
        self._notify(f"cancel pending {symbol} rejected: {result.message}")
        return False

    def _apply_sltp(self, symbol: str, signal: Signal) -> None:
        if signal.stop_loss is None and signal.take_profit is None:
            return
        result = self._bridge.modify_sltp(symbol, signal.stop_loss, signal.take_profit)
        if not result.accepted:
            logger.warning("SL/TP modify for %s rejected: %s", symbol, result.message)
            self._notify(f"SL/TP {symbol} rejected: {result.message}")

    def _register_scale_out(
        self, symbol: str, intent: OrderIntent, entry_price: float
    ) -> None:
        if intent.tp1 is None or intent.tp1_close_fraction is None:
            self._managed.pop(symbol, None)
            self._store.clear_managed_position(symbol)
            return
        managed = _Managed(
            side=intent.side,
            entry_price=entry_price,
            tp1=intent.tp1,
            close_fraction=intent.tp1_close_fraction,
            move_be=intent.move_sl_to_breakeven,
        )
        self._managed[symbol] = managed
        self._store.set_managed_position(
            symbol,
            managed.side,
            managed.entry_price,
            managed.tp1,
            managed.close_fraction,
            managed.move_be,
            managed.scaled,
        )

    def _manage_scale_out(self, symbol: str, bar: MT5Bar) -> None:
        """Close the TP1 fraction and move the stop to breakeven once price reaches TP1."""
        managed = self._managed.get(symbol)
        if managed is None or managed.scaled:
            return
        if not _target_crossed(managed.side, managed.tp1, bar):
            return

        position = self._positions.get(symbol)
        if position is None:
            self._managed.pop(symbol, None)
            self._store.clear_managed_position(symbol)
            return

        side, volume = position
        mapping = self._symbol_mappings.get(symbol)
        split = split_lot(
            volume,
            managed.close_fraction,
            min_lot=mapping.min_lot if mapping is not None else 0.0,
            lot_step=mapping.lot_step if mapping is not None else 0.0,
        )
        if split is None:
            # Cannot divide into two lot-step-aligned legs that both clear min_lot; run whole.
            managed.scaled = True
            self._persist_managed(symbol, managed)
            self._notify(f"scale-out skipped for {symbol}: cannot split into valid lots")
            return
        close_volume, remaining = split

        close_side: OrderSide = "sell" if side == "buy" else "buy"
        self._order_seq += 1
        order = MT5OrderRequest(
            symbol=symbol,
            side=close_side,
            volume=close_volume,
            client_order_id=f"{symbol}-tp1-{self._order_seq}",
            comment="tp1 scale-out",
        )
        result = self._bridge.submit_order(order)
        self._store.record_order(order, result)
        if not result.accepted:
            logger.warning("TP1 scale-out for %s rejected: %s", symbol, result.message)
            self._notify(f"tp1 scale-out {symbol} rejected: {result.message}")
            return

        # Track what actually closed. Prefer the broker's reported fill, then its normalized
        # request volume; filled == 0 means nothing closed -> retry next bar. When the broker
        # reports neither, keep the grid-aligned split remainder.
        filled = result.filled_volume
        if filled is not None and filled <= 0:
            return
        if filled is not None and filled > 0:
            closed = filled
            remaining = volume - closed
        elif result.requested_volume is not None and result.requested_volume > 0:
            closed = result.requested_volume
            remaining = volume - closed
        else:
            closed = close_volume

        self._positions[symbol] = (side, remaining)
        self._store.open_position(symbol, side, remaining, managed.entry_price)
        managed.scaled = True
        self._persist_managed(symbol, managed)
        if managed.move_be:
            be = self._bridge.modify_sltp(symbol, managed.entry_price, None)
            if not be.accepted:
                logger.warning("Breakeven SL move for %s rejected: %s", symbol, be.message)
        self._notify(f"scaled out {symbol} {closed} @ {managed.tp1}; runner {remaining}")

    def _persist_managed(self, symbol: str, managed: _Managed) -> None:
        self._store.set_managed_position(
            symbol,
            managed.side,
            managed.entry_price,
            managed.tp1,
            managed.close_fraction,
            managed.move_be,
            managed.scaled,
        )

    def _build_order(
        self, symbol: str, intent: OrderIntent, signal: Signal, is_open: bool
    ) -> MT5OrderRequest:
        self._order_seq += 1
        # A pending entry carries its SL/TP on the order itself, so the broker applies them when
        # the order fills (the bot only learns of the fill later via reconcile). Market entries
        # set SL/TP via _apply_sltp after the fill; close orders carry none.
        pending_entry = is_open and intent.order_kind != "market"
        return MT5OrderRequest(
            symbol=symbol,
            side=intent.side,
            volume=intent.volume,
            order_kind=intent.order_kind,
            price=intent.price,
            expiration=intent.expiration,
            stop_loss=intent.stop_loss if pending_entry else None,
            take_profit=intent.take_profit if pending_entry else None,
            client_order_id=f"{symbol}-{self._order_seq}",
            comment=signal.comment,
        )

    def _notify(self, message: str) -> None:
        try:
            self._notifier.send(message)
        except Exception:
            logger.exception("Notifier failed for message: %s", message)
