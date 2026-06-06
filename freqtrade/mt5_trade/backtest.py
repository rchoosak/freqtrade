from __future__ import annotations

from dataclasses import dataclass, field

from freqtrade.mt5_trade.data import MT5Bar
from freqtrade.mt5_trade.models import OrderKind, OrderSide
from freqtrade.mt5_trade.position import plan_transitions
from freqtrade.mt5_trade.strategy import MT5Strategy


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

    @property
    def num_trades(self) -> int:
        return len(self.trades)

    @property
    def total_pnl(self) -> float:
        return sum(trade.pnl for trade in self.trades)

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


@dataclass
class _Pending:
    side: OrderSide
    volume: float
    price: float
    kind: OrderKind


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
) -> BacktestResult:
    """
    Replay ``strategy`` over historical bars and report realized round-trip P&L.

    Reuses ``plan_transitions`` so backtest position semantics match the live bot exactly:
    one position per symbol, no stacking, entries/reversals/exits handled identically. Market
    entries fill at the bar close; limit/stop entries rest until a later bar's range touches the
    price (and are cancelled if the strategy reverses/exits first). Any position still open at the
    end is marked out at the final close when ``close_at_end`` is set.
    """
    result = BacktestResult()

    for symbol, bars in data.items():
        position: _OpenState | None = None
        pending: _Pending | None = None

        for index, bar in enumerate(bars):
            # 1. A resting order fills first if this bar's range reaches its price.
            if pending is not None and _is_filled(pending, bar):
                position = _OpenState(pending.side, pending.volume, pending.price, bar.time)
                pending = None

            window = bars[: index + 1][-warmup_bars:]
            signal = strategy.on_bar(symbol, window)
            if position is not None:
                current = (position.side, position.volume)
            elif pending is not None:
                current = (pending.side, pending.volume)
            else:
                current = None

            for intent in plan_transitions(current, signal, default_volume):
                if intent.result is None:
                    if position is not None:
                        result.trades.append(
                            _close_trade(symbol, position, bar.close, bar.time)
                        )
                        position = None
                    else:
                        # Cancel a resting order the strategy no longer wants.
                        pending = None
                elif intent.order_kind == "market":
                    position = _OpenState(intent.side, intent.volume, bar.close, bar.time)
                    pending = None
                else:
                    pending = _Pending(
                        intent.side, intent.volume, intent.price or 0.0, intent.order_kind
                    )
                    position = None

        if close_at_end and position is not None and bars:
            result.trades.append(_close_trade(symbol, position, bars[-1].close, bars[-1].time))

    return result


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
