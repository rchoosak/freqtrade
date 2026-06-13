from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace

from freqtrade.mt5_trade.data import MT5Bar
from freqtrade.mt5_trade.models import MT5SymbolMapping, OrderKind, OrderSide, split_lot
from freqtrade.mt5_trade.position import plan_transitions
from freqtrade.mt5_trade.sizing import PositionSizer, entry_side_for_action
from freqtrade.mt5_trade.strategy import MT5Strategy, Signal


@dataclass(frozen=True)
class BacktestTrade:
    symbol: str
    side: OrderSide
    volume: float
    entry_time: int
    exit_time: int
    entry_price: float
    exit_price: float
    pnl: float


@dataclass
class BacktestResult:
    trades: list[BacktestTrade] = field(default_factory=list)
    starting_balance: float | None = None
    contract_size: float = 1.0
    skipped_entries: int = 0

    @property
    def num_trades(self) -> int:
        return len(self.trades)

    @property
    def total_pnl(self) -> float:
        return sum(trade.pnl for trade in self.trades)

    @property
    def total_profit(self) -> float:
        return self.total_pnl * self.contract_size

    @property
    def ending_balance(self) -> float | None:
        if self.starting_balance is None:
            return None
        return self.starting_balance + self.total_profit

    @property
    def return_pct(self) -> float | None:
        if self.starting_balance is None:
            return None
        return self.total_profit / self.starting_balance * 100

    @property
    def wins(self) -> int:
        return sum(1 for trade in self.trades if trade.pnl > 0)

    @property
    def win_rate(self) -> float:
        return self.wins / self.num_trades if self.trades else 0.0


@dataclass
class _OpenState:
    side: OrderSide
    volume: float
    entry_price: float
    entry_time: int
    stop_loss: float | None = None
    take_profit: float | None = None
    tp1: float | None = None
    tp1_close_fraction: float | None = None
    move_be: bool = False
    scaled: bool = False


@dataclass
class _Pending:
    side: OrderSide
    volume: float
    price: float
    kind: OrderKind
    stop_loss: float | None = None
    take_profit: float | None = None
    tp1: float | None = None
    tp1_close_fraction: float | None = None
    move_be: bool = False


def _pnl_for(side: OrderSide, entry_price: float, exit_price: float, volume: float) -> float:
    # Profit in price units * volume; long gains when price rises, short when it falls.
    direction = 1.0 if side == "buy" else -1.0
    return (exit_price - entry_price) * direction * volume


def _is_filled(pending: _Pending, bar: MT5Bar) -> bool:
    """Whether a resting limit/stop order is touched by this bar's range."""
    if pending.kind == "limit":
        # Buy limit rests below the market and fills on a dip; sell limit above, on a rally.
        return bar.low <= pending.price if pending.side == "buy" else bar.high >= pending.price
    if pending.kind == "stop":
        # Buy stop fills on a breakout up; sell stop on a breakdown.
        return bar.high >= pending.price if pending.side == "buy" else bar.low <= pending.price
    return True


def run_backtest(
    strategy: MT5Strategy,
    data: dict[str, list[MT5Bar]],
    *,
    default_volume: float = 0.01,
    warmup_bars: int = 200,
    close_at_end: bool = True,
    starting_balance: float | None = None,
    contract_size: float = 1.0,
    position_sizer: PositionSizer | None = None,
    symbol_mappings: Mapping[str, MT5SymbolMapping] | None = None,
) -> BacktestResult:
    """
    Replay ``strategy`` over historical bars and report realized round-trip P&L.

    Reuses ``plan_transitions`` so backtest position semantics match the live bot exactly:
    one position per symbol, no stacking, entries/reversals/exits handled identically. Market
    entries fill at the bar close; limit/stop entries rest until a later bar's range touches the
    price (and are cancelled if the strategy reverses/exits first). Any position still open at the
    end is marked out at the final close when ``close_at_end`` is set.
    """
    sizer = position_sizer or PositionSizer(
        fixed_lot_size=default_volume,
        contract_size=contract_size,
    )
    _validate_backtest_account(starting_balance, contract_size, sizer)
    result = BacktestResult(starting_balance=starting_balance, contract_size=contract_size)
    current_balance = starting_balance

    # Evolve every symbol as one chronological stream so a shared (risk-percent) balance is the
    # true account balance at each timestamp, not the end-of-period balance of an earlier symbol.
    positions: dict[str, _OpenState | None] = {}
    pendings: dict[str, _Pending | None] = {}

    # Merge all symbols' bars into one time-ordered event stream; ties break by symbol name.
    events = sorted(
        (
            (bar.time, symbol, index)
            for symbol, bars in data.items()
            for index, bar in enumerate(bars)
        ),
        key=lambda event: (event[0], event[1]),
    )

    for _bar_time, symbol, index in events:
        bars = data[symbol]
        bar = bars[index]
        position = positions.get(symbol)
        pending = pendings.get(symbol)
        mapping = symbol_mappings.get(symbol) if symbol_mappings is not None else None

        if pending is not None and _is_filled(pending, bar):
            position = _open_from_pending(pending, bar.time)
            pending = None
            # The fill is intrabar at the order price; the rest of this bar can still hit SL/TP
            # (live carries broker-side SL/TP on the filled order). _manage_open_position skips
            # the fill bar via its entry-bar guard, so resolve it here, conservatively stop-first.
            position, current_balance = _resolve_exits(
                strategy, result, symbol, position, bar, current_balance, mapping
            )

        position, current_balance = _manage_open_position(
            strategy, result, symbol, position, bar, current_balance, mapping
        )

        window = bars[: index + 1][-warmup_bars:]
        signal = strategy.on_bar(symbol, window)
        current = _current_state(position, pending)

        sized_signal = _size_signal(
            sizer,
            symbol,
            signal,
            current,
            reference_price=bar.close,
            balance=current_balance,
            mapping=mapping,
        )
        if sized_signal is None:
            result.skipped_entries += 1
        else:
            position, pending, current_balance = _apply_signal_intents(
                strategy,
                result,
                symbol,
                position,
                pending,
                current,
                sized_signal,
                bar,
                default_volume,
                current_balance,
            )

        # Mark a symbol's still-open position out at its own last bar, in time order, so its
        # final P&L is in current_balance before any later bar of another symbol is sized.
        if close_at_end and index == len(bars) - 1 and position is not None:
            current_balance = _record_close(
                strategy, result, symbol, position, bar.close, bar.time, current_balance
            )
            position = None

        positions[symbol] = position
        pendings[symbol] = pending

    return result


def _validate_backtest_account(
    starting_balance: float | None,
    contract_size: float,
    position_sizer: PositionSizer,
) -> None:
    if starting_balance is not None and starting_balance <= 0:
        raise ValueError(
            f"Invalid starting_balance {starting_balance!r}. Expected a positive number."
        )
    if contract_size <= 0:
        raise ValueError(f"Invalid contract_size {contract_size!r}. Expected a positive number.")
    if position_sizer.requires_balance and starting_balance is None:
        raise ValueError("risk_percent position sizing requires starting_balance.")


def _current_state(
    position: _OpenState | None,
    pending: _Pending | None,
) -> tuple[OrderSide, float] | None:
    if position is not None:
        return (position.side, position.volume)
    if pending is not None:
        return (pending.side, pending.volume)
    return None


def _manage_open_position(
    strategy: MT5Strategy,
    result: BacktestResult,
    symbol: str,
    position: _OpenState | None,
    bar: MT5Bar,
    current_balance: float | None,
    mapping: MT5SymbolMapping | None,
) -> tuple[_OpenState | None, float | None]:
    # A market entry's own bar is never seen here (it opens later in the loop than this call), so
    # the entry-bar skip only ever applies to a pending fill — and that case is resolved
    # separately right after the fill (see run_backtest), where the rest of the bar can still hit
    # SL/TP, matching the broker-side SL/TP that a live pending order carries.
    if position is None or position.entry_time == bar.time:
        return position, current_balance
    return _resolve_exits(strategy, result, symbol, position, bar, current_balance, mapping)


def _resolve_exits(
    strategy: MT5Strategy,
    result: BacktestResult,
    symbol: str,
    position: _OpenState,
    bar: MT5Bar,
    current_balance: float | None,
    mapping: MT5SymbolMapping | None,
) -> tuple[_OpenState | None, float | None]:
    # Conservative ordering: a stop is resolved before any take-profit on the same bar.
    stop = position.stop_loss
    if stop is not None and _stop_crossed(position.side, stop, bar):
        return None, _record_close(
            strategy, result, symbol, position, stop, bar.time, current_balance
        )

    # Scale-out: close a fraction at TP1, move the stop to breakeven, let the rest run.
    if (
        not position.scaled
        and position.tp1 is not None
        and _target_crossed(position.side, position.tp1, bar)
    ):
        split = split_lot(
            position.volume,
            position.tp1_close_fraction or 0.0,
            min_lot=mapping.min_lot if mapping is not None else 0.0,
            lot_step=mapping.lot_step if mapping is not None else 0.0,
        )
        if split is None:
            # Can't split into two valid legs; run the whole position to the trend exit.
            return replace(position, scaled=True, tp1=None), current_balance
        closed, remaining = split
        current_balance = _record_partial(
            result, symbol, position, position.tp1, bar.time, closed, current_balance
        )
        position = replace(
            position,
            volume=remaining,
            stop_loss=position.entry_price if position.move_be else position.stop_loss,
            scaled=True,
            tp1=None,
        )
        return position, current_balance

    # Single full-position target (risk_reward mode).
    tp = position.take_profit
    if tp is not None and _target_crossed(position.side, tp, bar):
        return None, _record_close(
            strategy, result, symbol, position, tp, bar.time, current_balance
        )

    return position, current_balance


def _stop_crossed(side: OrderSide, level: float, bar: MT5Bar) -> bool:
    return bar.low <= level if side == "buy" else bar.high >= level


def _target_crossed(side: OrderSide, level: float, bar: MT5Bar) -> bool:
    return bar.high >= level if side == "buy" else bar.low <= level


def _apply_signal_intents(
    strategy: MT5Strategy,
    result: BacktestResult,
    symbol: str,
    position: _OpenState | None,
    pending: _Pending | None,
    current: tuple[OrderSide, float] | None,
    signal: Signal,
    bar: MT5Bar,
    default_volume: float,
    current_balance: float | None,
) -> tuple[_OpenState | None, _Pending | None, float | None]:
    for intent in plan_transitions(current, signal, default_volume):
        if intent.result is None:
            if position is not None:
                current_balance = _record_close(
                    strategy, result, symbol, position, bar.close, bar.time, current_balance
                )
                position = None
            else:
                pending = None
        elif intent.order_kind == "market":
            position = _open_from_intent(intent, bar.close, bar.time)
            pending = None
        else:
            pending = _pending_from_intent(intent)
            position = None
    return position, pending, current_balance


def _size_signal(
    position_sizer: PositionSizer,
    symbol: str,
    signal: Signal,
    current: tuple[OrderSide, float] | None,
    *,
    reference_price: float,
    balance: float | None,
    mapping: MT5SymbolMapping | None,
) -> Signal | None:
    side = entry_side_for_action(signal.action)
    if side is None:
        return signal
    if signal.volume is not None:
        # Explicit volume still has to obey the broker lot rules (same as the live bot).
        decision = position_sizer.snap(signal.volume, symbol=symbol, mapping=mapping)
        if decision.skipped:
            return None
        return replace(signal, volume=decision.volume)
    if current is not None and current[0] == side:
        return signal

    entry_price = signal.price if signal.price is not None else reference_price
    decision = position_sizer.size_entry(
        symbol=symbol,
        side=side,
        entry_price=entry_price,
        stop_loss=signal.stop_loss,
        balance=balance,
        mapping=mapping,
    )
    if decision.skipped:
        return None
    return replace(signal, volume=decision.volume)


def _open_from_pending(pending: _Pending, entry_time: int) -> _OpenState:
    return _OpenState(
        pending.side,
        pending.volume,
        pending.price,
        entry_time,
        pending.stop_loss,
        pending.take_profit,
        pending.tp1,
        pending.tp1_close_fraction,
        pending.move_be,
    )


def _open_from_intent(intent, entry_price: float, entry_time: int) -> _OpenState:
    return _OpenState(
        intent.side,
        intent.volume,
        entry_price,
        entry_time,
        intent.stop_loss,
        intent.take_profit,
        intent.tp1,
        intent.tp1_close_fraction,
        intent.move_sl_to_breakeven,
    )


def _pending_from_intent(intent) -> _Pending:
    return _Pending(
        intent.side,
        intent.volume,
        intent.price or 0.0,
        intent.order_kind,
        intent.stop_loss,
        intent.take_profit,
        intent.tp1,
        intent.tp1_close_fraction,
        intent.move_sl_to_breakeven,
    )


def _record_close(
    strategy: MT5Strategy,
    result: BacktestResult,
    symbol: str,
    state: _OpenState,
    exit_price: float,
    exit_time: int,
    current_balance: float | None,
) -> float | None:
    new_balance = _book_trade(
        result, symbol, state, exit_price, exit_time, state.volume, current_balance
    )
    # A full close tells the strategy so stateful strategies reset per-symbol tracking.
    strategy.on_position_closed(symbol)
    return new_balance


def _record_partial(
    result: BacktestResult,
    symbol: str,
    state: _OpenState,
    exit_price: float,
    exit_time: int,
    volume: float,
    current_balance: float | None,
) -> float | None:
    # A scale-out leg closes only part of the position; it is still open, so the strategy is
    # NOT told of a close here.
    return _book_trade(result, symbol, state, exit_price, exit_time, volume, current_balance)


def _book_trade(
    result: BacktestResult,
    symbol: str,
    state: _OpenState,
    exit_price: float,
    exit_time: int,
    volume: float,
    current_balance: float | None,
) -> float | None:
    pnl = _pnl_for(state.side, state.entry_price, exit_price, volume)
    result.trades.append(
        BacktestTrade(
            symbol=symbol,
            side=state.side,
            volume=volume,
            entry_time=state.entry_time,
            exit_time=exit_time,
            entry_price=state.entry_price,
            exit_price=exit_price,
            pnl=pnl,
        )
    )
    if current_balance is None:
        return None
    return current_balance + pnl * result.contract_size
