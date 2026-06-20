from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from freqtrade.mt5_trade.data import MT5Bar
from freqtrade.mt5_trade.strategies.base import (
    HOLD,
    MT5Strategy,
    Signal,
    _validate_positive_int,
)
from freqtrade.mt5_trade.strategies.indicators import (
    _aggregate_m1,
    _aggregate_m1_to_m5,
    _atr_series,
    _bb_percent_b,
    _crosses_down_from_zone,
    _crosses_up_from_zone,
    _ema_series,
    _rsi_series,
    _sma_optional_series,
    _stoch_rsi,
)
from freqtrade.mt5_trade.strategies.time_filters import (
    _parse_blackout_windows,
    _parse_sessions,
    _time_in_range,
)


def _validate_h1_config(fast: int, slow: int, slope_lookback: int, mode: str) -> None:
    _validate_positive_int("h1_fast", fast)
    _validate_positive_int("h1_slow", slow)
    _validate_positive_int("h1_slope_lookback", slope_lookback)
    if fast >= slow:
        raise ValueError("h1_fast must be shorter than h1_slow.")
    if mode not in {"strict", "block_opposite"}:
        raise ValueError("h1_filter_mode must be 'strict' or 'block_opposite'.")


def _validate_trend_strength_config(
    atr_length: int,
    min_spread_atr: float,
    spread_lookback: int,
    slope_lookback: int,
) -> None:
    _validate_positive_int("trend_atr_length", atr_length)
    _validate_positive_int("trend_spread_lookback", spread_lookback)
    _validate_positive_int("trend_slope_lookback", slope_lookback)
    if min_spread_atr < 0:
        raise ValueError("trend_min_spread_atr must be >= 0.")


class M5TrendM1EntryStrategy(MT5Strategy):
    """
    M1 entry strategy gated by an M5 trend and an optional H1 directional filter.

    The feed must provide M1 bars. Completed groups of five M1 bars are aggregated to M5 for the
    trend rules, while the latest M1 close drives Stoch RSI entry triggers.
    """

    def __init__(
        self,
        *,
        trend_fast: int = 9,
        trend_slow: int = 26,
        trend_rsi_length: int = 14,
        trend_bb_length: int = 20,
        trend_bb_stddev: float = 2.0,
        trend_min_ma_spread: float = 0.0,
        trend_atr_length: int = 14,
        trend_min_spread_atr: float = 0.0,
        trend_require_spread_expansion: bool = False,
        trend_spread_lookback: int = 3,
        trend_require_slow_slope: bool = False,
        trend_slope_lookback: int = 3,
        use_h1_filter: bool = False,
        h1_filter_mode: str = "strict",
        h1_fast: int = 9,
        h1_slow: int = 26,
        h1_slope_lookback: int = 3,
        sideways_rsi_low: float = 45.0,
        sideways_rsi_high: float = 55.0,
        stoch_rsi_length: int = 14,
        stoch_k_smooth: int = 3,
        stoch_d_smooth: int = 3,
        oversold: float = 20.0,
        overbought: float = 80.0,
        require_m1_confirmation: bool = False,
        stop_mode: str = "m1_swing",
        swing_lookback: int = 10,
        stop_buffer: float = 10.0,
        trend_exit_mode: str = "ma26_cross",
        take_profit_mode: str = "none",
        risk_reward: float = 1.0,
        tp1_rr: float = 1.0,
        tp1_close_fraction: float = 0.5,
        use_session_filter: bool = True,
        timezone: str = "Asia/Bangkok",
        sessions: list[str] | None = None,
        blackout_windows: list[dict[str, str]] | None = None,
    ) -> None:
        _validate_positive_int("trend_fast", trend_fast)
        _validate_positive_int("trend_slow", trend_slow)
        _validate_positive_int("trend_rsi_length", trend_rsi_length)
        _validate_positive_int("trend_bb_length", trend_bb_length)
        _validate_trend_strength_config(
            trend_atr_length,
            trend_min_spread_atr,
            trend_spread_lookback,
            trend_slope_lookback,
        )
        _validate_h1_config(h1_fast, h1_slow, h1_slope_lookback, h1_filter_mode)
        _validate_positive_int("stoch_rsi_length", stoch_rsi_length)
        _validate_positive_int("stoch_k_smooth", stoch_k_smooth)
        _validate_positive_int("stoch_d_smooth", stoch_d_smooth)
        _validate_positive_int("swing_lookback", swing_lookback)
        if trend_fast >= trend_slow:
            raise ValueError("trend_fast must be shorter than trend_slow.")
        if trend_bb_stddev <= 0:
            raise ValueError("trend_bb_stddev must be > 0.")
        if trend_min_ma_spread < 0:
            raise ValueError("trend_min_ma_spread must be >= 0.")
        if stop_buffer < 0:
            raise ValueError("stop_buffer must be >= 0.")
        if stop_mode not in {"m1_swing", "m5_previous"}:
            raise ValueError("stop_mode must be 'm1_swing' or 'm5_previous'.")
        if trend_exit_mode not in {"ma26_cross", "strict_filter"}:
            raise ValueError("trend_exit_mode must be 'ma26_cross' or 'strict_filter'.")
        if take_profit_mode not in {"none", "risk_reward", "scale_out"}:
            raise ValueError("take_profit_mode must be 'none', 'risk_reward', or 'scale_out'.")
        if risk_reward <= 0:
            raise ValueError("risk_reward must be > 0.")
        if tp1_rr <= 0:
            raise ValueError("tp1_rr must be > 0.")
        if not 0 < tp1_close_fraction < 1:
            raise ValueError("tp1_close_fraction must be between 0 and 1 (exclusive).")

        self.trend_fast = trend_fast
        self.trend_slow = trend_slow
        self.trend_rsi_length = trend_rsi_length
        self.trend_bb_length = trend_bb_length
        self.trend_bb_stddev = trend_bb_stddev
        self.trend_min_ma_spread = trend_min_ma_spread
        self.trend_atr_length = trend_atr_length
        self.trend_min_spread_atr = trend_min_spread_atr
        self.trend_require_spread_expansion = trend_require_spread_expansion
        self.trend_spread_lookback = trend_spread_lookback
        self.trend_require_slow_slope = trend_require_slow_slope
        self.trend_slope_lookback = trend_slope_lookback
        self.use_h1_filter = use_h1_filter
        self.h1_filter_mode = h1_filter_mode
        self.h1_fast = h1_fast
        self.h1_slow = h1_slow
        self.h1_slope_lookback = h1_slope_lookback
        self.sideways_rsi_low = sideways_rsi_low
        self.sideways_rsi_high = sideways_rsi_high
        self.stoch_rsi_length = stoch_rsi_length
        self.stoch_k_smooth = stoch_k_smooth
        self.stoch_d_smooth = stoch_d_smooth
        self.oversold = oversold
        self.overbought = overbought
        self.require_m1_confirmation = require_m1_confirmation
        self.stop_mode = stop_mode
        self.swing_lookback = swing_lookback
        self.stop_buffer = stop_buffer
        self.trend_exit_mode = trend_exit_mode
        self.take_profit_mode = take_profit_mode
        self.risk_reward = risk_reward
        self.tp1_rr = tp1_rr
        self.tp1_close_fraction = tp1_close_fraction
        self.use_session_filter = use_session_filter
        self.timezone = ZoneInfo(timezone)
        self.sessions = _parse_sessions(
            sessions or ["14:00-17:00", "19:00-22:30"]
        )
        self.blackout_windows = _parse_blackout_windows(blackout_windows or [])
        self._last_trend: dict[str, str] = {}
        self._last_m5_time: dict[str, int] = {}
        self._h1_cache: dict[str, tuple[int, str]] = {}

    def on_bar(self, symbol: str, bars: list[MT5Bar]) -> Signal:
        if len(bars) < self._minimum_m1_bars:
            return HOLD

        current_bar = bars[-1]
        if self._is_blackout(current_bar.time):
            return Signal(action="exit", comment="news blackout")

        # The prior strategy used a 300-bar feed window. Keep M5/M1 signal calculations on that
        # same recent window when H1 filtering increases the feed warmup to ~2,000 M1 bars.
        signal_bars = bars[-max(300, self._minimum_signal_m1_bars) :]
        m5_bars = _aggregate_m1_to_m5(signal_bars)
        if len(m5_bars) < self._minimum_m5_bars:
            return HOLD

        trend = self._m5_trend(m5_bars)
        trend_exit = self._trend_exit_signal(symbol, m5_bars, trend)
        if trend_exit is not None:
            return trend_exit

        if self.use_session_filter and not self._in_session(current_bar.time):
            return HOLD

        h1_bias = self._h1_bias(symbol, bars) if self.use_h1_filter else "neutral"

        if trend == "long" and self._h1_allows("long", h1_bias) and self._m1_buy_trigger(
            signal_bars
        ):
            stop_loss = self._stop_loss("buy", signal_bars, m5_bars)
            tp1, fraction, move_be = self._scale_out("buy", current_bar.close, stop_loss)
            return Signal(
                action="enter_long",
                stop_loss=stop_loss,
                take_profit=self._take_profit("buy", current_bar.close, stop_loss),
                tp1=tp1,
                tp1_close_fraction=fraction,
                move_sl_to_breakeven=move_be,
                comment=self._entry_comment("long"),
            )
        if trend == "short" and self._h1_allows("short", h1_bias) and self._m1_sell_trigger(
            signal_bars
        ):
            stop_loss = self._stop_loss("sell", signal_bars, m5_bars)
            tp1, fraction, move_be = self._scale_out("sell", current_bar.close, stop_loss)
            return Signal(
                action="enter_short",
                stop_loss=stop_loss,
                take_profit=self._take_profit("sell", current_bar.close, stop_loss),
                tp1=tp1,
                tp1_close_fraction=fraction,
                move_sl_to_breakeven=move_be,
                comment=self._entry_comment("short"),
            )
        return HOLD

    def on_position_closed(self, symbol: str) -> None:
        # A real close (SL/TP/external/reconcile) ends the tracked trend position; forget it so
        # the next opposite M5 setup is not masked by a stale exit on this symbol.
        self._last_trend.pop(symbol, None)

    @property
    def _minimum_m5_bars(self) -> int:
        needs = [self.trend_slow, self.trend_rsi_length + 1, self.trend_bb_length]
        if self.trend_min_spread_atr > 0:
            needs.append(self.trend_atr_length + 1)
        if self.trend_require_spread_expansion:
            needs.append(self.trend_slow + self.trend_spread_lookback)
        if self.trend_require_slow_slope:
            needs.append(self.trend_slow + self.trend_slope_lookback)
        return max(needs) + 1

    @property
    def _minimum_m1_bars(self) -> int:
        minimum = self._minimum_signal_m1_bars
        if self.use_h1_filter:
            minimum = max(minimum, self._minimum_h1_m1_bars)
        return minimum

    @property
    def _minimum_signal_m1_bars(self) -> int:
        # Stoch RSI chains RSI(trend_rsi_length) -> stoch(stoch_rsi_length) -> K -> D, then needs
        # two D values to detect a cross. The inner RSI uses trend_rsi_length (see _stoch_rsi),
        # not trend_fast, so warm-up must count trend_rsi_length.
        stoch_need = self.trend_rsi_length + self.stoch_rsi_length + self.stoch_k_smooth
        stoch_need += self.stoch_d_smooth + 2
        return max(self._minimum_m5_bars * 5, stoch_need, self.swing_lookback)

    @property
    def _minimum_h1_m1_bars(self) -> int:
        # Two extra H1 buckets cover partial candles at either edge of a sliding M1 window.
        complete_h1_bars = self.h1_slow + self.h1_slope_lookback
        return (complete_h1_bars + 2) * 60

    def _h1_allows(self, direction: str, bias: str) -> bool:
        if not self.use_h1_filter:
            return True
        if self.h1_filter_mode == "block_opposite":
            return bias == "neutral" or bias == direction
        return bias == direction

    def _entry_comment(self, direction: str) -> str:
        trend_label = "h1/m5" if self.use_h1_filter else "m5"
        return f"{trend_label} trend {direction} + m1 stoch rsi trigger"

    def _h1_bias(self, symbol: str, bars: list[MT5Bar]) -> str:
        # H1 changes only when the minute-59 bar completes. Cache by the latest completed H1
        # bucket so M1 backtests do not aggregate ~2,000 bars on every minute.
        current_hour = bars[-1].time - (bars[-1].time % 3600)
        latest_complete_hour = (
            current_hour if bars[-1].time % 3600 == 59 * 60 else current_hour - 3600
        )
        cached = self._h1_cache.get(symbol)
        if cached is not None and cached[0] == latest_complete_hour:
            return cached[1]

        h1_bars = _aggregate_m1(bars, 3600)
        required = self.h1_slow + self.h1_slope_lookback
        if len(h1_bars) < required:
            bias = "neutral"
        else:
            closes = [bar.close for bar in h1_bars]
            fast = _ema_series(closes, self.h1_fast)
            slow = _ema_series(closes, self.h1_slow)
            slow_now = slow[-1]
            slow_then = slow[-1 - self.h1_slope_lookback]
            if fast[-1] > slow_now and closes[-1] > slow_now and slow_now > slow_then:
                bias = "long"
            elif fast[-1] < slow_now and closes[-1] < slow_now and slow_now < slow_then:
                bias = "short"
            else:
                bias = "neutral"

        self._h1_cache[symbol] = (latest_complete_hour, bias)
        return bias

    def _m5_trend(self, bars: list[MT5Bar]) -> str:
        closes = [bar.close for bar in bars]
        ema_fast = _ema_series(closes, self.trend_fast)
        ema_slow = _ema_series(closes, self.trend_slow)
        rsi = _rsi_series(closes, self.trend_rsi_length)
        bb_percent = _bb_percent_b(closes, self.trend_bb_length, self.trend_bb_stddev)

        close = closes[-1]
        fast = ema_fast[-1]
        slow = ema_slow[-1]
        prev_rsi = rsi[-2]
        curr_rsi = rsi[-1]
        curr_bb = bb_percent[-1]
        if prev_rsi is None or curr_rsi is None or curr_bb is None:
            return "neutral"

        if self.sideways_rsi_low <= curr_rsi <= self.sideways_rsi_high:
            return "neutral"

        spread = abs(fast - slow)
        if spread < self.trend_min_ma_spread:
            return "neutral"
        if self.trend_min_spread_atr > 0:
            atr = _atr_series(bars, self.trend_atr_length)[-1]
            if atr is None or atr <= 0 or spread / atr < self.trend_min_spread_atr:
                return "neutral"

        if self.trend_require_spread_expansion:
            previous_spread = abs(
                ema_fast[-1 - self.trend_spread_lookback]
                - ema_slow[-1 - self.trend_spread_lookback]
            )
            if spread <= previous_spread:
                return "neutral"

        if close > fast and fast > slow and curr_rsi > 50 and curr_rsi > prev_rsi:
            if (
                self.trend_require_slow_slope
                and slow <= ema_slow[-1 - self.trend_slope_lookback]
            ):
                return "neutral"
            return "long" if curr_bb > 0.5 else "neutral"
        if close < fast and fast < slow and curr_rsi < 50 and curr_rsi < prev_rsi:
            if (
                self.trend_require_slow_slope
                and slow >= ema_slow[-1 - self.trend_slope_lookback]
            ):
                return "neutral"
            return "short" if curr_bb < 0.5 else "neutral"
        return "neutral"

    def _trend_exit_signal(
        self, symbol: str, m5_bars: list[MT5Bar], trend: str
    ) -> Signal | None:
        m5_time = m5_bars[-1].time
        if self._last_m5_time.get(symbol) == m5_time:
            return None

        active = self._last_trend.get(symbol)
        self._last_m5_time[symbol] = m5_time

        if active == "long" and self._m5_exit_hit(m5_bars, "long", trend):
            self._last_trend.pop(symbol, None)
            return Signal(action="exit", comment="m5 long exit rule")
        if active == "short" and self._m5_exit_hit(m5_bars, "short", trend):
            self._last_trend.pop(symbol, None)
            return Signal(action="exit", comment="m5 short exit rule")

        if trend in {"long", "short"} and active is None:
            self._last_trend[symbol] = trend
        return None

    def _m5_exit_hit(self, bars: list[MT5Bar], active: str, trend: str) -> bool:
        if self.trend_exit_mode == "strict_filter":
            return trend != active

        closes = [bar.close for bar in bars]
        fast = _ema_series(closes, self.trend_fast)
        slow = _ema_series(closes, self.trend_slow)
        close = closes[-1]
        if active == "long":
            crossed_down = fast[-2] >= slow[-2] and fast[-1] < slow[-1]
            return close < slow[-1] or crossed_down

        crossed_up = fast[-2] <= slow[-2] and fast[-1] > slow[-1]
        return close > slow[-1] or crossed_up

    def _m1_buy_trigger(self, bars: list[MT5Bar]) -> bool:
        closes = [bar.close for bar in bars]
        stoch = _stoch_rsi(closes, self.trend_rsi_length, self.stoch_rsi_length)
        k = _sma_optional_series(stoch, self.stoch_k_smooth)
        d = _sma_optional_series(k, self.stoch_d_smooth)
        if not _crosses_up_from_zone(k, d, self.oversold):
            return False
        return not self.require_m1_confirmation or self._m1_buy_confirmation(bars)

    def _m1_sell_trigger(self, bars: list[MT5Bar]) -> bool:
        closes = [bar.close for bar in bars]
        stoch = _stoch_rsi(closes, self.trend_rsi_length, self.stoch_rsi_length)
        k = _sma_optional_series(stoch, self.stoch_k_smooth)
        d = _sma_optional_series(k, self.stoch_d_smooth)
        if not _crosses_down_from_zone(k, d, self.overbought):
            return False
        return not self.require_m1_confirmation or self._m1_sell_confirmation(bars)

    def _m1_buy_confirmation(self, bars: list[MT5Bar]) -> bool:
        closes = [bar.close for bar in bars]
        fast = _ema_series(closes, self.trend_fast)
        slow = _ema_series(closes, self.trend_slow)
        bb_percent = _bb_percent_b(closes, self.trend_bb_length, self.trend_bb_stddev)
        return fast[-2] <= slow[-2] and fast[-1] > slow[-1] and (bb_percent[-1] or 0) > 0.5

    def _m1_sell_confirmation(self, bars: list[MT5Bar]) -> bool:
        closes = [bar.close for bar in bars]
        fast = _ema_series(closes, self.trend_fast)
        slow = _ema_series(closes, self.trend_slow)
        bb_percent = _bb_percent_b(closes, self.trend_bb_length, self.trend_bb_stddev)
        return fast[-2] >= slow[-2] and fast[-1] < slow[-1] and (bb_percent[-1] or 1) < 0.5

    def _stop_loss(self, side: str, m1_bars: list[MT5Bar], m5_bars: list[MT5Bar]) -> float:
        if self.stop_mode == "m5_previous":
            reference = self._previous_m5(m1_bars, m5_bars)
            return (
                reference.low - self.stop_buffer
                if side == "buy"
                else reference.high + self.stop_buffer
            )

        window = m1_bars[-self.swing_lookback :]
        return (
            min(bar.low for bar in window) - self.stop_buffer
            if side == "buy"
            else max(bar.high for bar in window) + self.stop_buffer
        )

    def _previous_m5(self, m1_bars: list[MT5Bar], m5_bars: list[MT5Bar]) -> MT5Bar:
        # The "previous M5 candle" is the completed bucket before the one holding the entry M1
        # bar. When the trigger fires on the 5th M1 minute, that bucket is already complete and is
        # m5_bars[-1] (the entry's own candle), so step back to m5_bars[-2]. When the trigger
        # fires mid-bucket, m5_bars[-1] is already the prior completed candle.
        current_bucket = m1_bars[-1].time - (m1_bars[-1].time % 300)
        if m5_bars[-1].time == current_bucket and len(m5_bars) >= 2:
            return m5_bars[-2]
        return m5_bars[-1]

    def _take_profit(self, side: str, entry_price: float, stop_loss: float) -> float | None:
        # Only the single full-position target mode sets a fixed TP. In scale-out mode the
        # remainder runs to the trend exit, so there is no fixed full TP.
        if self.take_profit_mode != "risk_reward":
            return None

        risk_distance = abs(entry_price - stop_loss)
        if side == "buy":
            return entry_price + risk_distance * self.risk_reward
        return entry_price - risk_distance * self.risk_reward

    def _scale_out(
        self, side: str, entry_price: float, stop_loss: float
    ) -> tuple[float | None, float | None, bool]:
        # TP1 at tp1_rr x risk; close tp1_close_fraction there and move the stop to breakeven,
        # letting the rest run to the M5 trend exit (TP2).
        if self.take_profit_mode != "scale_out":
            return None, None, False
        risk_distance = abs(entry_price - stop_loss)
        tp1 = (
            entry_price + risk_distance * self.tp1_rr
            if side == "buy"
            else entry_price - risk_distance * self.tp1_rr
        )
        return tp1, self.tp1_close_fraction, True

    def _in_session(self, timestamp: int) -> bool:
        local_time = datetime.fromtimestamp(timestamp, UTC).astimezone(self.timezone).time()
        return any(_time_in_range(local_time, start, end) for start, end in self.sessions)

    def _is_blackout(self, timestamp: int) -> bool:
        value = datetime.fromtimestamp(timestamp, UTC)
        return any(start <= value <= end for start, end in self.blackout_windows)
