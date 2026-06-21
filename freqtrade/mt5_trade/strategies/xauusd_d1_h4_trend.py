from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

from freqtrade.mt5_trade.data import MT5Bar
from freqtrade.mt5_trade.models import OrderSide
from freqtrade.mt5_trade.strategies.base import (
    HOLD,
    MT5Strategy,
    Signal,
    _validate_positive_int,
)
from freqtrade.mt5_trade.strategies.indicators import (
    _adx_series,
    _aggregate_completed_bars,
    _atr_series,
    _ema_series,
)


class XauusdD1H4TrendStrategy(MT5Strategy):
    """
    Long-horizon XAUUSD trend strategy using H1 input, H4 breakouts, and a D1 regime.

    Entries require a directional D1 EMA/ADX regime and an H4 Donchian breakout aligned with
    H4 EMA10/30. Positions have an ATR initial stop and no fixed take-profit. They exit on an
    H4-close Chandelier reversal or when the latest completed D1 candle crosses the exit EMA.
    """

    def __init__(
        self,
        *,
        d1_fast: int = 30,
        d1_slow: int = 150,
        d1_slope_lookback: int = 3,
        d1_adx_length: int = 14,
        d1_long_adx_min: float = 15.0,
        d1_short_adx_min: float = 20.0,
        d1_exit_ema: int = 50,
        h4_fast: int = 10,
        h4_slow: int = 30,
        breakout_lookback: int = 15,
        atr_length: int = 14,
        initial_stop_atr: float = 2.5,
        max_breakout_atr: float = 2.5,
        max_channel_breakout_atr: float | None = None,
        chandelier_lookback: int = 22,
        chandelier_atr: float = 3.0,
        allow_short: bool = True,
        daily_anchor_hour: int = 0,
        session_break_hours: list[int] | tuple[int, ...] | None = None,
        sunday_open_hour: int = 23,
        friday_close_hour: int = 21,
    ) -> None:
        for name, value in (
            ("d1_fast", d1_fast),
            ("d1_slow", d1_slow),
            ("d1_slope_lookback", d1_slope_lookback),
            ("d1_adx_length", d1_adx_length),
            ("d1_exit_ema", d1_exit_ema),
            ("h4_fast", h4_fast),
            ("h4_slow", h4_slow),
            ("breakout_lookback", breakout_lookback),
            ("atr_length", atr_length),
            ("chandelier_lookback", chandelier_lookback),
        ):
            _validate_positive_int(name, value)
        if d1_fast >= d1_slow:
            raise ValueError("d1_fast must be shorter than d1_slow.")
        if h4_fast >= h4_slow:
            raise ValueError("h4_fast must be shorter than h4_slow.")
        if d1_long_adx_min < 0 or d1_short_adx_min < 0:
            raise ValueError("D1 ADX thresholds must be >= 0.")
        if initial_stop_atr <= 0:
            raise ValueError("initial_stop_atr must be > 0.")
        if max_breakout_atr <= 0:
            raise ValueError("max_breakout_atr must be > 0.")
        if max_channel_breakout_atr is not None and max_channel_breakout_atr <= 0:
            raise ValueError("max_channel_breakout_atr must be > 0.")
        if chandelier_atr <= 0:
            raise ValueError("chandelier_atr must be > 0.")
        break_hours = _validate_session_hours(
            daily_anchor_hour,
            session_break_hours,
            sunday_open_hour,
            friday_close_hour,
        )

        self.d1_fast = d1_fast
        self.d1_slow = d1_slow
        self.d1_slope_lookback = d1_slope_lookback
        self.d1_adx_length = d1_adx_length
        self.d1_long_adx_min = d1_long_adx_min
        self.d1_short_adx_min = d1_short_adx_min
        self.d1_exit_ema = d1_exit_ema
        self.h4_fast = h4_fast
        self.h4_slow = h4_slow
        self.breakout_lookback = breakout_lookback
        self.atr_length = atr_length
        self.initial_stop_atr = initial_stop_atr
        self.max_breakout_atr = max_breakout_atr
        self.max_channel_breakout_atr = (
            max_channel_breakout_atr
            if max_channel_breakout_atr is not None
            else max_breakout_atr
        )
        self.chandelier_lookback = chandelier_lookback
        self.chandelier_atr = chandelier_atr
        self.allow_short = allow_short
        self.daily_anchor_hour = daily_anchor_hour
        self.session_break_hours = frozenset(break_hours)
        self.sunday_open_hour = sunday_open_hour
        self.friday_close_hour = friday_close_hour
        self._position_sides: dict[str, OrderSide] = {}
        self._last_h4_time: dict[str, int] = {}
        self._trailing_stops: dict[str, float] = {}
        self._pending_exits: dict[str, Signal] = {}

    @property
    def required_timeframe(self) -> str:
        return "H1"

    @property
    def minimum_bars(self) -> int:
        d1_need = max(
            self.d1_slow + self.d1_slope_lookback,
            2 * self.d1_adx_length,
            self.d1_exit_ema,
        )
        h4_need = max(
            self.h4_slow,
            self.breakout_lookback + 1,
            self.atr_length + 1,
            self.chandelier_lookback,
        )
        # Two extra buckets absorb partial leading/trailing periods and scheduled market pauses.
        return max((d1_need + 2) * 24, (h4_need + 2) * 4)

    def on_position_state(self, symbol: str, side: OrderSide | None) -> None:
        if side is None:
            self._position_sides.pop(symbol, None)
            self._trailing_stops.pop(symbol, None)
            self._pending_exits.pop(symbol, None)
        else:
            if self._position_sides.get(symbol) != side:
                self._trailing_stops.pop(symbol, None)
                self._pending_exits.pop(symbol, None)
            self._position_sides[symbol] = side

    def on_position_closed(self, symbol: str) -> None:
        self._position_sides.pop(symbol, None)
        self._trailing_stops.pop(symbol, None)
        self._pending_exits.pop(symbol, None)

    def persistent_position_state(self, symbol: str) -> dict[str, Any] | None:
        stop = self._trailing_stops.get(symbol)
        if stop is None:
            return None
        return {"trailing_stop": stop}

    def restore_position_state(
        self,
        symbol: str,
        side: OrderSide,
        state: dict[str, Any],
    ) -> None:
        raw_stop = state.get("trailing_stop")
        if not isinstance(raw_stop, int | float):
            raise ValueError(f"{symbol}: persisted trailing_stop must be numeric.")
        stop = float(raw_stop)
        if not math.isfinite(stop) or stop <= 0:
            raise ValueError(f"{symbol}: persisted trailing_stop must be finite and positive.")
        self._position_sides[symbol] = side
        self._trailing_stops[symbol] = stop

    def on_bar(self, symbol: str, bars: list[MT5Bar]) -> Signal:
        if len(bars) < self.minimum_bars:
            return HOLD

        h4_bars = _aggregate_completed_bars(
            bars,
            4 * 3600,
            3600,
            expected_slot=self._is_expected_h1_slot,
        )
        d1_bars = _aggregate_completed_bars(
            bars,
            24 * 3600,
            3600,
            anchor_seconds=self.daily_anchor_hour * 3600,
            expected_slot=self._is_expected_h1_slot,
            merge_sunday_into_monday=True,
        )
        if len(h4_bars) < self._minimum_h4_bars or len(d1_bars) < self._minimum_d1_bars:
            return HOLD

        side = self._position_sides.get(symbol)
        pending_exit = self._pending_exits.get(symbol)
        if side is not None and pending_exit is not None:
            return pending_exit

        h4_time = h4_bars[-1].time
        if self._last_h4_time.get(symbol) == h4_time:
            return HOLD
        self._last_h4_time[symbol] = h4_time

        if side is not None:
            exit_signal = self._exit_signal(symbol, side, h4_bars, d1_bars)
            if exit_signal is None:
                return HOLD
            self._pending_exits[symbol] = exit_signal
            return exit_signal

        regime = self._d1_regime(d1_bars)
        if regime == "long" and self._h4_breakout(h4_bars, "buy"):
            return self._entry_signal(h4_bars, "buy")
        if regime == "short" and self.allow_short and self._h4_breakout(h4_bars, "sell"):
            return self._entry_signal(h4_bars, "sell")
        return HOLD

    @property
    def _minimum_d1_bars(self) -> int:
        return max(
            self.d1_slow + self.d1_slope_lookback,
            2 * self.d1_adx_length,
            self.d1_exit_ema,
        )

    @property
    def _minimum_h4_bars(self) -> int:
        return max(
            self.h4_slow,
            self.breakout_lookback + 1,
            self.atr_length + 1,
            self.chandelier_lookback,
        )

    def _d1_regime(self, bars: list[MT5Bar]) -> str:
        closes = [bar.close for bar in bars]
        fast = _ema_series(closes, self.d1_fast)
        slow = _ema_series(closes, self.d1_slow)
        adx = _adx_series(bars, self.d1_adx_length)[-1]
        if adx is None:
            return "neutral"

        close = closes[-1]
        slow_now = slow[-1]
        slow_then = slow[-1 - self.d1_slope_lookback]
        if (
            close > slow_now
            and fast[-1] > slow_now
            and slow_now > slow_then
            and adx >= self.d1_long_adx_min
        ):
            return "long"
        if (
            close < slow_now
            and fast[-1] < slow_now
            and slow_now < slow_then
            and adx >= self.d1_short_adx_min
        ):
            return "short"
        return "neutral"

    def _h4_breakout(self, bars: list[MT5Bar], side: OrderSide) -> bool:
        closes = [bar.close for bar in bars]
        fast = _ema_series(closes, self.h4_fast)[-1]
        slow = _ema_series(closes, self.h4_slow)[-1]
        previous_atr = _atr_series(bars[:-1], self.atr_length)[-1]
        if previous_atr is None or previous_atr <= 0:
            return False

        current = bars[-1]
        previous_close = bars[-2].close
        true_range = max(
            current.high - current.low,
            abs(current.high - previous_close),
            abs(current.low - previous_close),
        )
        if true_range > self.max_breakout_atr * previous_atr:
            return False
        channel = bars[-self.breakout_lookback - 1 : -1]
        if side == "buy":
            boundary = max(bar.high for bar in channel)
            breakout_distance = current.close - boundary
            return (
                fast > slow
                and breakout_distance > 0
                and breakout_distance <= self.max_channel_breakout_atr * previous_atr
            )
        boundary = min(bar.low for bar in channel)
        breakout_distance = boundary - current.close
        return (
            fast < slow
            and breakout_distance > 0
            and breakout_distance <= self.max_channel_breakout_atr * previous_atr
        )

    def _is_expected_h1_slot(self, timestamp: int) -> bool:
        current = datetime.fromtimestamp(timestamp, UTC)
        weekday = current.weekday()
        hour = current.hour
        if weekday == 5:
            return False
        if weekday == 6:
            return hour >= self.sunday_open_hour
        if hour in self.session_break_hours:
            return False
        if weekday == 4:
            return hour < self.friday_close_hour
        return True

    def _entry_signal(self, bars: list[MT5Bar], side: OrderSide) -> Signal:
        atr = _atr_series(bars, self.atr_length)[-1]
        if atr is None:
            return HOLD
        entry = bars[-1].close
        stop_loss = (
            entry - self.initial_stop_atr * atr
            if side == "buy"
            else entry + self.initial_stop_atr * atr
        )
        return Signal(
            action="enter_long" if side == "buy" else "enter_short",
            stop_loss=stop_loss,
            comment=f"d1 {side} regime + h4 donchian breakout",
        )

    def _exit_signal(
        self,
        symbol: str,
        side: OrderSide,
        h4_bars: list[MT5Bar],
        d1_bars: list[MT5Bar],
    ) -> Signal | None:
        d1_closes = [bar.close for bar in d1_bars]
        exit_ema = _ema_series(d1_closes, self.d1_exit_ema)[-1]
        current_h4 = h4_bars[-1]
        if side == "buy" and d1_closes[-1] < exit_ema:
            return Signal(action="exit", comment="d1 close below exit ema")
        if side == "sell" and d1_closes[-1] > exit_ema:
            return Signal(action="exit", comment="d1 close above exit ema")

        atr = _atr_series(h4_bars, self.atr_length)[-1]
        if atr is None:
            return None
        window = h4_bars[-self.chandelier_lookback :]
        if side == "buy":
            candidate = max(bar.high for bar in window) - self.chandelier_atr * atr
            stop = max(candidate, self._trailing_stops.get(symbol, candidate))
            self._trailing_stops[symbol] = stop
            if current_h4.close < stop:
                return Signal(action="exit", comment="h4 chandelier long exit")
        else:
            candidate = min(bar.low for bar in window) + self.chandelier_atr * atr
            stop = min(candidate, self._trailing_stops.get(symbol, candidate))
            self._trailing_stops[symbol] = stop
            if current_h4.close > stop:
                return Signal(action="exit", comment="h4 chandelier short exit")
        return None


def _validate_session_hours(
    daily_anchor_hour: int,
    session_break_hours: list[int] | tuple[int, ...] | None,
    sunday_open_hour: int,
    friday_close_hour: int,
) -> tuple[int, ...]:
    if not 0 <= daily_anchor_hour <= 23:
        raise ValueError("daily_anchor_hour must be between 0 and 23.")
    if not 0 <= sunday_open_hour <= 23:
        raise ValueError("sunday_open_hour must be between 0 and 23.")
    if not 0 <= friday_close_hour <= 24:
        raise ValueError("friday_close_hour must be between 0 and 24.")
    break_hours = tuple((21, 22) if session_break_hours is None else session_break_hours)
    if any(not isinstance(hour, int) or not 0 <= hour <= 23 for hour in break_hours):
        raise ValueError("session_break_hours must contain UTC hours between 0 and 23.")
    return break_hours
