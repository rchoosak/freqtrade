from __future__ import annotations

from dataclasses import dataclass, field

from freqtrade.mt5_trade.data import MT5Bar
from freqtrade.mt5_trade.models import OrderSide
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


def _pnl(state: _OpenState, exit_price: float) -> float:
    # Profit in price units * volume; long gains when price rises, short when it falls.
    direction = 1.0 if state.side == "buy" else -1.0
    return (exit_price - state.entry_price) * direction * state.volume


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
    one position per symbol, no stacking, entries/reversals/exits handled identically. Fills are
    simulated at each bar's close. Any position still open at the end is marked out at the final
    close when ``close_at_end`` is set.
    """
    result = BacktestResult()

    for symbol, bars in data.items():
        open_state: _OpenState | None = None

        for index, bar in enumerate(bars):
            window = bars[: index + 1][-warmup_bars:]
            signal = strategy.on_bar(symbol, window)
            current = (open_state.side, open_state.volume) if open_state else None

            for intent in plan_transitions(current, signal, default_volume):
                if intent.result is None:
                    # A close always follows an open in plan_transitions.
                    if open_state is not None:
                        result.trades.append(
                            _close_trade(symbol, open_state, bar.close, bar.time)
                        )
                        open_state = None
                else:
                    open_state = _OpenState(intent.side, intent.volume, bar.close, bar.time)

        if close_at_end and open_state is not None and bars:
            result.trades.append(
                _close_trade(symbol, open_state, bars[-1].close, bars[-1].time)
            )

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
