from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from decimal import Decimal
from typing import Any, NoReturn

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
from freqtrade.mt5_trade.strategies import (
    MT5Strategy,
    Signal,
    validate_strategy_bars,
    validate_strategy_runtime,
)


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


@dataclass(frozen=True)
class _ScaleOutPlan:
    managed: _Managed
    side: OrderSide
    volume: float
    close_volume: float
    remaining: float
    position_ticket: int | None


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


def _ticket_from_order_id(order_id: str | None) -> int | None:
    """Pending order tickets are returned as order ids; market fills do not use this."""
    return int(order_id) if order_id is not None and order_id.isdigit() else None


def _same_price(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return True
    return abs(left - right) <= max(1e-9, abs(left) * 1e-9)


def _same_volume(left: float, right: float) -> bool:
    return abs(left - right) <= max(1e-9, abs(left) * 1e-9)


def _position_identity_changed(
    old: tuple[float | None, int | None] | None,
    new: tuple[float | None, int | None],
) -> bool:
    if old is None:
        return False
    old_price, old_ticket = old
    new_price, new_ticket = new
    if old_ticket is not None and new_ticket is not None and old_ticket != new_ticket:
        return True
    return not _same_price(old_price, new_price)


def _ticket_learned(
    old: tuple[float | None, int | None] | None,
    new: tuple[float | None, int | None],
) -> bool:
    return old is not None and old[1] is None and new[1] is not None


def _unique_by_symbol(items: Iterable[Any], label: str) -> dict[str, Any]:
    unique: dict[str, Any] = {}
    duplicates: set[str] = set()
    for item in items:
        symbol = item.symbol
        if symbol in unique:
            duplicates.add(symbol)
            continue
        unique[symbol] = item
    if duplicates:
        symbols = ", ".join(sorted(duplicates))
        raise OperationalException(
            f"Multiple broker {label} for symbol(s): {symbols}. mt5-trade supports one "
            f"{label[:-1]} per symbol; resolve the broker state manually before continuing."
        )
    return unique


def _open_side_for_close(close_side: OrderSide) -> OrderSide:
    return "buy" if close_side == "sell" else "sell"


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
        validate_strategy_runtime(
            strategy,
            timeframe=bot_config.timeframe,
            warmup_bars=bot_config.warmup_bars,
        )
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
        # symbol -> (entry_price, broker position ticket) used to detect when a broker-side
        # close/reopen leaves the same side and volume but represents a different live position.
        self._position_ids: dict[str, tuple[float | None, int | None]] = {
            symbol: (pos.entry_price, pos.ticket) for symbol, pos in store.open_positions().items()
        }
        # symbol -> (side, volume, ticket) for resting pending orders the bot placed.
        # In-memory only; rebuilt from the broker via reconcile() (no broker in dry-run).
        self._pendings: dict[str, tuple[OrderSide, float, int | None]] = {}
        # symbol -> the iteration a pending order was placed, for bot-side expiry.
        self._pending_placed: dict[str, int] = {}
        # symbol -> scale-out bookkeeping for positions opened with a TP1 plan.
        self._managed: dict[str, _Managed] = self._restore_managed(store)
        self._restore_strategy_position_states()

    @staticmethod
    def _restore_managed(store: MT5TradeStore) -> dict[str, _Managed]:
        managed: dict[str, _Managed] = {}
        open_positions = store.open_positions()
        for symbol, state in store.managed_positions().items():
            open_position = open_positions.get(symbol)
            if open_position is None:
                store.clear_managed_position(symbol)
                continue
            if state.side != open_position.side or not _same_price(
                state.entry_price, open_position.entry_price
            ):
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
    def _strategy_id(self) -> str:
        strategy_type = type(self._strategy)
        return f"{strategy_type.__module__}.{strategy_type.__qualname__}"

    def _restore_strategy_position_states(self) -> None:
        open_positions = self._store.open_positions()
        for symbol, state in self._store.strategy_position_states().items():
            position = open_positions.get(symbol)
            identity_matches = (
                position is not None
                and state.strategy == self._strategy_id
                and state.side == position.side
                and position.entry_price is not None
                and _same_price(state.entry_price, position.entry_price)
                and state.ticket == position.ticket
            )
            if not identity_matches:
                self._store.clear_strategy_position_state(symbol)
                continue
            self._strategy.restore_position_state(
                symbol,
                state.side,  # type: ignore[arg-type]
                state.state,
            )

    @property
    def running(self) -> bool:
        return self._running

    def run_once(self) -> None:
        """Evaluate every configured symbol exactly once."""
        self._iterations += 1
        for symbol in self._config.symbols:
            bars = self._feed.latest_bars(symbol, self._config.warmup_bars)
            validate_strategy_bars(self._strategy, symbol, bars)
            position = self._positions.get(symbol)
            self._strategy.on_position_state(
                symbol,
                position[0] if position is not None else None,
            )
            # Scale out before asking the strategy, mirroring the backtester's ordering.
            self._manage_scale_out(symbol, bars[-1])
            signal = self._strategy.on_bar(symbol, bars)
            self._persist_strategy_position_state(symbol)
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
        iteration = 0
        consecutive_errors = 0
        try:
            self.reconcile()
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
        self._refresh_from_broker()

    def _refresh_from_broker(self) -> bool:
        positions = self._bridge.broker_positions()
        orders = self._bridge.broker_orders()
        if positions is None and orders is None:
            return False

        changes: list[str] = []
        self._reconcile_positions(positions or [], changes)
        self._reconcile_pendings(orders or [], changes)

        if changes:
            self._notify("Reconciled: " + ", ".join(changes))
        return True

    def _reconcile_positions(self, positions: list, changes: list[str]) -> None:
        broker = _unique_by_symbol(positions, "positions")
        for symbol, pos in broker.items():
            state = (pos.side, pos.volume)
            identity = (pos.price, pos.ticket)
            previous_position = self._positions.get(symbol)
            slot_changed = previous_position != state
            side_changed = previous_position is not None and previous_position[0] != state[0]
            identity_changed = _position_identity_changed(self._position_ids.get(symbol), identity)
            ticket_learned = _ticket_learned(self._position_ids.get(symbol), identity)
            if slot_changed or identity_changed or ticket_learned:
                if side_changed or identity_changed:
                    self._clear_managed(symbol)
                    self._clear_strategy_position_state(symbol, reset_strategy=True)
                self._positions[symbol] = state
                self._position_ids[symbol] = identity
                self._store.open_position(symbol, state[0], state[1], pos.price, ticket=pos.ticket)
                if ticket_learned:
                    self._persist_strategy_position_state(symbol)
                changes.append(f"{symbol}->{state[0]} {state[1]}")
        for symbol in list(self._positions):
            if symbol not in broker:
                self._positions.pop(symbol, None)
                self._position_ids.pop(symbol, None)
                self._clear_managed(symbol)
                self._store.close_position(symbol)
                # The broker closed this position (e.g. SL/TP); let the strategy reset state.
                self._strategy.on_position_closed(symbol)
                changes.append(f"{symbol}->flat")

    def _reconcile_pendings(self, orders: list, changes: list[str]) -> None:
        broker = {
            symbol: (order.side, order.volume, order.ticket)
            for symbol, order in _unique_by_symbol(orders, "orders").items()
        }
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
        if current is not None and current[0] == side:
            return signal
        if self._position_sizer.requires_balance and signal.order_kind != "market":
            message = (
                f"{symbol}: risk_percent sizing supports market entries only; "
                f"{signal.order_kind} entry skipped."
            )
            logger.warning(message)
            self._notify(message)
            return None
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

        if signal.order_kind == "market":
            entry_price = self._executable_entry_price(symbol, side, reference_price)
        elif signal.price is not None:
            entry_price = signal.price
        else:  # Signal validates this invariant, but keep sizing fail-loud if it regresses.
            raise OperationalException(f"{symbol}: pending entry requires an explicit price.")
        loss_per_lot = None
        if self._position_sizer.requires_balance and signal.stop_loss is not None:
            loss_per_lot = self._stop_loss_risk(
                symbol,
                side,
                1.0,
                entry_price,
                signal.stop_loss,
            )
        decision = self._position_sizer.size_entry(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            stop_loss=signal.stop_loss,
            balance=self._current_balance(),
            mapping=self._symbol_mappings.get(symbol),
            loss_per_lot=loss_per_lot,
        )
        if decision.skipped:
            message = decision.reason or f"{symbol}: position sizing skipped entry."
            logger.warning(message)
            self._notify(message)
            return None
        return replace(signal, volume=decision.volume)

    def _current_balance(self) -> float | None:
        # Equity includes floating PnL and is safer than balance while other positions are open.
        # Fall back to live balance, then configured starting balance for dry-run/offline use.
        if self._position_sizer.requires_balance:
            equity_reader = getattr(self._bridge, "account_equity", None)
            if equity_reader is not None:
                broker_equity = equity_reader()
                if broker_equity is not None:
                    return broker_equity
            broker_balance = self._bridge.account_balance()
            if broker_balance is not None:
                return broker_balance
        return self._account_balance

    def _executable_entry_price(
        self,
        symbol: str,
        side: OrderSide,
        reference_price: float,
    ) -> float:
        price_reader = getattr(self._bridge, "executable_price", None)
        if price_reader is None:
            return reference_price
        price = price_reader(symbol, side)
        if price is None:
            return reference_price
        if price <= 0:
            raise OperationalException(f"{symbol}: executable {side} price must be positive.")
        return float(price)

    def _stop_loss_risk(
        self,
        symbol: str,
        side: OrderSide,
        volume: float,
        entry_price: float,
        stop_loss: float,
    ) -> float | None:
        distance = entry_price - stop_loss if side == "buy" else stop_loss - entry_price
        if distance <= 0:
            return None
        risk_reader = getattr(self._bridge, "stop_loss_risk", None)
        risk = (
            risk_reader(symbol, side, volume, entry_price, stop_loss)
            if risk_reader is not None
            else None
        )
        if risk is None:
            risk = self._position_sizer.stop_loss_risk(
                side=side,
                volume=volume,
                entry_price=entry_price,
                stop_loss=stop_loss,
            )
        if risk <= 0:
            raise OperationalException(
                f"{symbol}: broker returned invalid stop-loss risk for {side} {volume} lot."
            )
        return float(risk)

    def _entry_spread_ok(self, symbol: str) -> bool:
        # Live-only spread gate: skip new entries when the broker spread exceeds the configured
        # cap (0 = disabled). M1 bars carry no spread, so this never affects backtests, and the
        # bridge returns None in dry-run.
        limit = self._config.max_spread_points
        if limit <= 0:
            return True
        spread = self._bridge.current_spread_points(symbol)
        if spread is not None and spread > limit:
            message = f"{symbol} entry skipped: spread {spread} > max {limit} points."
            logger.info(message)
            self._notify(message)
            return False
        return True

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

        position_ticket = None
        if not is_open:
            ticket_ok, position_ticket = self._resolve_position_ticket(symbol, intent.reason)
            if not ticket_ok:
                return False
            if not self._close_intent_matches_current_position(symbol, intent):
                message = (
                    f"{intent.reason} {symbol} skipped: broker position changed during "
                    "ticket refresh; will re-plan on the next bar."
                )
                logger.warning(message)
                self._notify(message)
                return False

        if is_open and not self._entry_spread_ok(symbol):
            return False

        order = self._build_order(
            symbol, intent, signal, is_open, position_ticket=position_ticket
        )
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
            ticket = _ticket_from_order_id(result.order_id)
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
            self._clear_strategy_position_state(symbol)
            self._positions[symbol] = (intent.side, volume)
            self._position_ids[symbol] = (entry_price, None)
            self._store.open_position(symbol, intent.side, volume, entry_price, ticket=None)
            self._protect_entry_sltp(symbol, signal)
            entry_price = self._position_ids.get(symbol, (entry_price, None))[0] or entry_price
            if not self._enforce_filled_risk(
                symbol,
                intent.side,
                entry_price,
                signal.stop_loss,
            ):
                return True
            self._register_scale_out(symbol, intent, entry_price)
        else:
            filled = _resolved_volume(result, intent.volume)
            if filled < intent.volume and not _same_volume(filled, intent.volume):
                remaining = float(
                    Decimal(str(intent.volume)) - Decimal(str(max(0.0, filled)))
                )
                open_side = _open_side_for_close(intent.side)
                stored_entry_price, ticket = self._position_ids.get(
                    symbol, (None, position_ticket)
                )
                self._positions[symbol] = (open_side, remaining)
                self._position_ids[symbol] = (stored_entry_price, ticket)
                self._store.open_position(
                    symbol,
                    open_side,
                    remaining,
                    stored_entry_price,
                    ticket=ticket,
                )
                self._notify(
                    f"{intent.reason} partially closed {symbol} by {filled}; "
                    f"remaining {remaining}"
                )
                return False
            self._positions.pop(symbol, None)
            self._position_ids.pop(symbol, None)
            self._clear_managed(symbol)
            self._store.close_position(symbol)
            self._strategy.on_position_closed(symbol)

        final_volume = (
            self._positions.get(symbol, (intent.side, intent.volume))[1]
            if is_open
            else intent.volume
        )
        self._notify(f"{intent.reason} {intent.side} {symbol} {final_volume} ({result.order_id})")
        return True

    def _close_intent_matches_current_position(self, symbol: str, intent: OrderIntent) -> bool:
        position = self._positions.get(symbol)
        if position is None:
            return False
        expected_side = _open_side_for_close(intent.side)
        side, volume = position
        return side == expected_side and _same_volume(volume, intent.volume)

    def _protect_entry_sltp(self, symbol: str, signal: Signal) -> None:
        if signal.stop_loss is None and signal.take_profit is None:
            return
        ticket_ok, position_ticket = self._resolve_position_ticket(symbol, "SL/TP")
        if not ticket_ok:
            self._abort_unprotected_entry(
                symbol,
                "SL/TP ticket lookup failed after market entry",
            )
        if not self._apply_sltp(symbol, signal, position_ticket=position_ticket):
            self._abort_unprotected_entry(symbol, "SL/TP modify failed after market entry")

    def _enforce_filled_risk(
        self,
        symbol: str,
        side: OrderSide,
        entry_price: float,
        stop_loss: float | None,
    ) -> bool:
        """Reduce a filled entry when slippage makes its broker-calculated stop risk too large."""
        if not self._position_sizer.requires_balance or stop_loss is None:
            return True
        position = self._positions.get(symbol)
        balance = self._current_balance()
        if position is None or balance is None:
            return True

        current_side, volume = position
        if current_side != side:
            self._abort_overrisk_entry(
                symbol,
                "position side changed before post-fill risk validation",
            )
        actual_risk = self._filled_stop_loss_risk_or_close(
            symbol,
            side,
            volume,
            entry_price,
            stop_loss,
        )
        if actual_risk is None:
            return False
        budget = self._position_sizer.risk_budget(balance)
        if actual_risk <= budget + max(0.01, budget * 1e-9):
            return True

        close_volume = self._risk_reduction_volume(
            symbol,
            side,
            volume,
            entry_price,
            stop_loss,
            balance,
            actual_risk,
        )

        refreshed = self._resolved_position_for_risk_close(
            symbol,
            side,
            "risk cap reduction",
        )
        if refreshed is None:
            return False
        refreshed_volume, position_ticket = refreshed
        if not _same_volume(refreshed_volume, volume):
            refreshed_entry = self._position_ids.get(symbol, (entry_price, None))[0] or entry_price
            return self._enforce_filled_risk(
                symbol,
                side,
                refreshed_entry,
                stop_loss,
            )
        remaining = self._submit_risk_reduction(
            symbol,
            side,
            volume,
            close_volume,
            position_ticket,
        )
        if remaining <= 0:
            return False

        residual_risk = self._filled_stop_loss_risk_or_close(
            symbol,
            side,
            remaining,
            entry_price,
            stop_loss,
        )
        if residual_risk is None:
            return False
        if residual_risk <= budget + max(0.01, budget * 1e-9):
            return True

        # Broker normalization or a partial fill can leave the position above budget. Flatten it
        # rather than carrying exposure that contradicts the configured hard cap.
        remaining = self._submit_risk_reduction(
            symbol,
            side,
            remaining,
            remaining,
            position_ticket,
        )
        if remaining > 0:
            self._abort_overrisk_entry(
                symbol,
                "position remained above the risk cap after emergency close",
            )
        return False

    def _risk_reduction_volume(
        self,
        symbol: str,
        side: OrderSide,
        volume: float,
        entry_price: float,
        stop_loss: float,
        balance: float,
        actual_risk: float,
    ) -> float:
        safe = self._position_sizer.size_entry(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            stop_loss=stop_loss,
            balance=balance,
            mapping=self._symbol_mappings.get(symbol),
            loss_per_lot=actual_risk / volume,
        )
        safe_volume = safe.volume or 0.0
        close_volume = (
            volume
            if safe_volume <= 0
            else float(Decimal(str(volume)) - Decimal(str(safe_volume)))
        )
        mapping = self._symbol_mappings.get(symbol)
        return (
            volume
            if mapping is not None and close_volume < mapping.min_lot
            else close_volume
        )

    def _filled_stop_loss_risk_or_close(
        self,
        symbol: str,
        side: OrderSide,
        volume: float,
        entry_price: float,
        stop_loss: float,
    ) -> float | None:
        try:
            risk = self._stop_loss_risk(
                symbol,
                side,
                volume,
                entry_price,
                stop_loss,
            )
            if risk is None:
                raise OperationalException("stop is no longer beyond the filled entry price")
            return risk
        except Exception as exc:
            self._close_unverifiable_risk(symbol, side, volume, exc)
            return None

    def _close_unverifiable_risk(
        self,
        symbol: str,
        side: OrderSide,
        volume: float,
        error: Exception,
    ) -> bool:
        self._notify(f"risk validation failed for {symbol}: {error}; closing full position")
        refreshed = self._resolved_position_for_risk_close(
            symbol,
            side,
            f"unverifiable risk ({error})",
        )
        if refreshed is None:
            return False
        refreshed_volume, position_ticket = refreshed
        remaining = self._submit_risk_reduction(
            symbol,
            side,
            refreshed_volume,
            refreshed_volume,
            position_ticket,
        )
        if remaining > 0:
            self._abort_overrisk_entry(
                symbol,
                f"risk could not be calculated ({error}) and emergency close was partial",
            )
        return False

    def _resolved_position_for_risk_close(
        self,
        symbol: str,
        expected_side: OrderSide,
        reason: str,
    ) -> tuple[float, int | None] | None:
        ticket_ok, position_ticket = self._resolve_position_ticket(symbol, reason)
        if not ticket_ok:
            self._abort_overrisk_entry(
                symbol,
                f"position ticket lookup failed during {reason}",
            )
        position = self._positions.get(symbol)
        if position is None:
            return None
        side, volume = position
        if side != expected_side:
            self._abort_overrisk_entry(
                symbol,
                f"position side changed during {reason}",
            )
        return volume, position_ticket

    def _submit_risk_reduction(
        self,
        symbol: str,
        side: OrderSide,
        current_volume: float,
        close_volume: float,
        position_ticket: int | None,
    ) -> float:
        close_side: OrderSide = "sell" if side == "buy" else "buy"
        self._order_seq += 1
        order = MT5OrderRequest(
            symbol=symbol,
            side=close_side,
            volume=close_volume,
            position_ticket=position_ticket,
            client_order_id=f"{symbol}-risk-{self._order_seq}",
            comment="risk cap reduction",
        )
        try:
            result = self._bridge.submit_order(order)
        except Exception as exc:
            self._abort_overrisk_entry(
                symbol,
                f"emergency risk close raised an exception: {exc}",
            )
        self._store.record_order(order, result)
        if not result.accepted and close_volume < current_volume:
            self._notify(
                f"risk reduction {symbol} {close_volume} rejected; closing full position"
            )
            return self._submit_risk_reduction(
                symbol,
                side,
                current_volume,
                current_volume,
                position_ticket,
            )
        if not result.accepted:
            self._abort_overrisk_entry(
                symbol,
                f"emergency risk close rejected: {result.message}",
            )

        closed = min(current_volume, _resolved_volume(result, close_volume))
        if closed <= 0:
            self._abort_overrisk_entry(symbol, "emergency risk close filled zero volume")
        remaining = max(
            0.0,
            float(Decimal(str(current_volume)) - Decimal(str(closed))),
        )
        entry_price, ticket = self._position_ids.get(symbol, (None, position_ticket))
        if remaining <= 1e-12:
            self._positions.pop(symbol, None)
            self._position_ids.pop(symbol, None)
            self._clear_managed(symbol)
            self._store.close_position(symbol)
            self._strategy.on_position_closed(symbol)
            self._notify(f"closed {symbol}: filled entry exceeded risk cap")
            return 0.0

        self._positions[symbol] = (side, remaining)
        self._position_ids[symbol] = (entry_price, ticket)
        self._store.open_position(symbol, side, remaining, entry_price, ticket=ticket)
        self._notify(f"reduced {symbol} by {closed} lot to enforce risk cap; remaining {remaining}")
        return remaining

    def _abort_overrisk_entry(self, symbol: str, reason: str) -> NoReturn:
        self._running = False
        message = f"{reason}: {symbol}; bot stopped because live exposure may exceed its risk cap."
        logger.error(message)
        self._notify(message)
        raise OperationalException(message)

    def _abort_unprotected_entry(self, symbol: str, reason: str) -> NoReturn:
        self._clear_managed(symbol)
        self._running = False
        message = (
            f"{reason}: {symbol} position is tracked but the bot is stopping to avoid "
            "unmanaged live exposure."
        )
        logger.error(message)
        self._notify(message)
        raise OperationalException(message)

    def _resolve_position_ticket(
        self, symbol: str, reason: str
    ) -> tuple[bool, int | None]:
        ticket = self._position_ids.get(symbol, (None, None))[1]
        if ticket is not None:
            return True, ticket

        broker_available = self._refresh_from_broker()
        ticket = self._position_ids.get(symbol, (None, None))[1]
        if broker_available and ticket is None:
            message = (
                f"{reason} {symbol} rejected: broker position ticket is required to manage "
                "positions safely on MT5 hedging accounts."
            )
            logger.warning(message)
            self._notify(message)
            return False, None
        return True, ticket

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

    def _apply_sltp(
        self,
        symbol: str,
        signal: Signal,
        *,
        position_ticket: int | None = None,
    ) -> bool:
        if signal.stop_loss is None and signal.take_profit is None:
            return True
        result = self._modify_sltp(
            symbol, signal.stop_loss, signal.take_profit, position_ticket=position_ticket
        )
        if not result.accepted:
            logger.warning("SL/TP modify for %s rejected: %s", symbol, result.message)
            self._notify(f"SL/TP {symbol} rejected: {result.message}")
            return False
        return True

    def _modify_sltp(
        self,
        symbol: str,
        stop_loss: float | None,
        take_profit: float | None,
        *,
        position_ticket: int | None = None,
    ) -> MT5OrderResult:
        if position_ticket is None:
            return self._bridge.modify_sltp(symbol, stop_loss, take_profit)
        return self._bridge.modify_sltp(
            symbol, stop_loss, take_profit, position_ticket=position_ticket
        )

    def _register_scale_out(
        self, symbol: str, intent: OrderIntent, entry_price: float
    ) -> None:
        if intent.tp1 is None or intent.tp1_close_fraction is None:
            self._clear_managed(symbol)
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

    def _clear_managed(self, symbol: str) -> None:
        self._managed.pop(symbol, None)
        self._store.clear_managed_position(symbol)

    def _persist_strategy_position_state(self, symbol: str) -> None:
        position = self._positions.get(symbol)
        identity = self._position_ids.get(symbol)
        state = self._strategy.persistent_position_state(symbol)
        if position is None or identity is None or state is None:
            self._store.clear_strategy_position_state(symbol)
            return
        entry_price, ticket = identity
        if entry_price is None:
            self._store.clear_strategy_position_state(symbol)
            return
        self._store.set_strategy_position_state(
            symbol,
            self._strategy_id,
            position[0],
            entry_price,
            ticket,
            state,
        )

    def _clear_strategy_position_state(
        self,
        symbol: str,
        *,
        reset_strategy: bool = False,
    ) -> None:
        self._store.clear_strategy_position_state(symbol)
        if reset_strategy:
            self._strategy.on_position_closed(symbol)

    def _manage_scale_out(self, symbol: str, bar: MT5Bar) -> None:
        """Close the TP1 fraction and move the stop to breakeven once price reaches TP1."""
        plan = self._scale_out_plan(symbol, bar)
        if plan is None:
            return

        close_side: OrderSide = "sell" if plan.side == "buy" else "buy"
        self._order_seq += 1
        order = MT5OrderRequest(
            symbol=symbol,
            side=close_side,
            volume=plan.close_volume,
            position_ticket=plan.position_ticket,
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
            remaining = plan.volume - closed
        elif result.requested_volume is not None and result.requested_volume > 0:
            closed = result.requested_volume
            remaining = plan.volume - closed
        else:
            closed = plan.close_volume
            remaining = plan.remaining

        self._positions[symbol] = (plan.side, remaining)
        self._position_ids[symbol] = (
            plan.managed.entry_price,
            self._position_ids.get(symbol, (None, None))[1],
        )
        self._store.open_position(
            symbol,
            plan.side,
            remaining,
            plan.managed.entry_price,
            ticket=self._position_ids[symbol][1],
        )
        plan.managed.scaled = True
        self._persist_managed(symbol, plan.managed)
        if plan.managed.move_be:
            be = self._modify_sltp(
                symbol,
                plan.managed.entry_price,
                None,
                position_ticket=plan.position_ticket,
            )
            if not be.accepted:
                logger.warning("Breakeven SL move for %s rejected: %s", symbol, be.message)
        self._notify(f"scaled out {symbol} {closed} @ {plan.managed.tp1}; runner {remaining}")

    def _scale_out_plan(self, symbol: str, bar: MT5Bar) -> _ScaleOutPlan | None:
        managed = self._managed.get(symbol)
        if managed is None or managed.scaled:
            return None
        if not _target_crossed(managed.side, managed.tp1, bar):
            return None

        ticket_ok, position_ticket = self._resolve_position_ticket(symbol, "tp1 scale-out")
        if not ticket_ok:
            return None

        return self._refreshed_scale_out_plan(symbol, bar, position_ticket)

    def _refreshed_scale_out_plan(
        self,
        symbol: str,
        bar: MT5Bar,
        position_ticket: int | None,
    ) -> _ScaleOutPlan | None:
        managed = self._managed.get(symbol)
        if managed is None or managed.scaled:
            return None
        if not _target_crossed(managed.side, managed.tp1, bar):
            return None

        position = self._positions.get(symbol)
        if position is None:
            self._clear_managed(symbol)
            return None

        side, volume = position
        if side != managed.side:
            self._clear_managed(symbol)
            self._notify(
                f"scale-out skipped for {symbol}: broker position side changed "
                "during ticket refresh"
            )
            return None

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
            return None
        close_volume, remaining = split
        return _ScaleOutPlan(
            managed=managed,
            side=side,
            volume=volume,
            close_volume=close_volume,
            remaining=remaining,
            position_ticket=position_ticket,
        )

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
        self,
        symbol: str,
        intent: OrderIntent,
        signal: Signal,
        is_open: bool,
        *,
        position_ticket: int | None = None,
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
            position_ticket=position_ticket,
            client_order_id=f"{symbol}-{self._order_seq}",
            comment=signal.comment,
        )

    def _notify(self, message: str) -> None:
        try:
            self._notifier.send(message)
        except Exception:
            logger.exception("Notifier failed for message: %s", message)
