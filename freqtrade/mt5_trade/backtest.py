from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace

from freqtrade.mt5_trade.data import MT5Bar
from freqtrade.mt5_trade.models import MT5SymbolMapping, OrderKind, OrderSide
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


@dataclass
class _Pending:
    side: OrderSide
    volume: float
    price: float
    kind: OrderKind
    stop_loss: float | None = None
    take_profit: float | None = None


def _pnl(state: _OpenState, exit_price: float) -> float:
    # Profit in price units * volume; long gains when price rises, short when it falls.
    direction = 1.0 if state.side == "buy" else -1.0
    return (exit_price - state.entry_price) * direction * state.volume


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

    for symbol, bars in data.items():
        position: _OpenState | None = None
        pending: _Pending | None = None
        mapping = symbol_mappings.get(symbol) if symbol_mappings is not None else None

        for index, bar in enumerate(bars):
            if pending is not None and _is_filled(pending, bar):
                position = _open_from_pending(pending, bar.time)
                pending = None

            position, current_balance = _stop_position_if_hit(
                result, symbol, position, bar, current_balance
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
                continue

            position, pending, current_balance = _apply_signal_intents(
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

        if close_at_end and position is not None and bars:
            current_balance = _record_close(
                result, symbol, position, bars[-1].close, bars[-1].time, current_balance
            )

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


def _stop_position_if_hit(
    result: BacktestResult,
    symbol: str,
    position: _OpenState | None,
    bar: MT5Bar,
    current_balance: float | None,
) -> tuple[_OpenState | None, float | None]:
    if position is None or position.entry_time == bar.time:
        return position, current_balance

    stop_exit = _stop_exit_price(position, bar)
    if stop_exit is None:
        return position, current_balance

    new_balance = _record_close(result, symbol, position, stop_exit, bar.time, current_balance)
    return None, new_balance


def _apply_signal_intents(
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
                    result, symbol, position, bar.close, bar.time, current_balance
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
    if side is None or signal.volume is not None:
        return signal
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
    )


def _open_from_intent(intent, entry_price: float, entry_time: int) -> _OpenState:
    return _OpenState(
        intent.side,
        intent.volume,
        entry_price,
        entry_time,
        intent.stop_loss,
        intent.take_profit,
    )


def _pending_from_intent(intent) -> _Pending:
    return _Pending(
        intent.side,
        intent.volume,
        intent.price or 0.0,
        intent.order_kind,
        intent.stop_loss,
        intent.take_profit,
    )


def _stop_exit_price(position: _OpenState, bar: MT5Bar) -> float | None:
    if position.side == "buy":
        if position.stop_loss is not None and bar.low <= position.stop_loss:
            return position.stop_loss
        if position.take_profit is not None and bar.high >= position.take_profit:
            return position.take_profit
    else:
        if position.stop_loss is not None and bar.high >= position.stop_loss:
            return position.stop_loss
        if position.take_profit is not None and bar.low <= position.take_profit:
            return position.take_profit
    return None


def _record_close(
    result: BacktestResult,
    symbol: str,
    state: _OpenState,
    exit_price: float,
    exit_time: int,
    current_balance: float | None,
) -> float | None:
    trade = _close_trade(symbol, state, exit_price, exit_time)
    result.trades.append(trade)
    if current_balance is None:
        return None
    return current_balance + trade.pnl * result.contract_size


def _close_trade(
    symbol: str, state: _OpenState, exit_price: float, exit_time: int
) -> BacktestTrade:
    return BacktestTrade(
        symbol=symbol,
        side=state.side,
        volume=state.volume,
        entry_time=state.entry_time,
        exit_time=exit_time,
        entry_price=state.entry_price,
        exit_price=exit_price,
        pnl=_pnl(state, exit_price),
    )
