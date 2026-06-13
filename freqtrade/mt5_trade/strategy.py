from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime, time
from typing import Literal
from zoneinfo import ZoneInfo

from freqtrade.mt5_trade.data import MT5Bar
from freqtrade.mt5_trade.models import OrderKind


SignalAction = Literal["enter_long", "enter_short", "exit", "hold"]


@dataclass(frozen=True)
class Signal:
    action: SignalAction = "hold"
    volume: float | None = None
    # Entry order type. A limit/stop entry requires an explicit price and rests as a pending
    # order until the market reaches it; market entries fill immediately.
    order_kind: OrderKind = "market"
    price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    comment: str | None = None
    # Epoch-seconds expiry for a pending entry (broker auto-cancels at that time).
    expiration: int | None = None
    # Scale-out / breakeven plan. When ``tp1`` is set, the position manager closes
    # ``tp1_close_fraction`` of the position at ``tp1`` and (optionally) moves the stop to the
    # entry price, letting the remainder run to a trend-based exit. Independent of ``take_profit``
    # (which is a single full-position target).
    tp1: float | None = None
    tp1_close_fraction: float | None = None
    move_sl_to_breakeven: bool = False

    def __post_init__(self) -> None:
        if self.order_kind != "market" and self.price is None:
            raise ValueError(f"{self.order_kind} entry signal requires an explicit price.")
        if self.tp1_close_fraction is not None and not 0 < self.tp1_close_fraction < 1:
            raise ValueError("tp1_close_fraction must be between 0 and 1 (exclusive).")
        if self.tp1 is not None and self.tp1_close_fraction is None:
            raise ValueError("tp1 requires tp1_close_fraction.")


# Shared singleton for "do nothing" to avoid allocating on every bar.
HOLD = Signal(action="hold")


class MT5Strategy(ABC):
    """Decides an action from the latest completed bars of a single symbol."""

    @abstractmethod
    def on_bar(self, symbol: str, bars: list[MT5Bar]) -> Signal:
        """Return a Signal given the most recent bars (oldest first)."""

    def on_position_closed(self, symbol: str) -> None:  # noqa: B027  # optional override
        """
        Notify the strategy that ``symbol``'s position was closed outside of ``on_bar`` — by a
        stop-loss/take-profit, the broker, or reconciliation. Stateful strategies override this
        to reset per-symbol tracking so the next setup isn't masked by stale internal state.
        No-op by default.
        """


def _sma(values: list[float], length: int) -> float:
    window = values[-length:]
    return sum(window) / len(window)


class SmaCrossStrategy(MT5Strategy):
    """
    Simple moving-average crossover demo strategy.

    Enters long when the fast SMA crosses above the slow SMA, short when it crosses below.
    Stateless: the crossover is detected from the last two bars so the bot's position manager
    owns entry/exit bookkeeping.
    """

    def __init__(
        self,
        fast: int = 10,
        slow: int = 30,
        *,
        stop_loss_distance: float | None = None,
        take_profit_distance: float | None = None,
    ) -> None:
        if fast < 1 or slow < 1:
            raise ValueError("SMA lengths must be >= 1.")
        if fast >= slow:
            raise ValueError(f"fast ({fast}) must be shorter than slow ({slow}).")
        if stop_loss_distance is not None and stop_loss_distance <= 0:
            raise ValueError("stop_loss_distance must be > 0 when configured.")
        if take_profit_distance is not None and take_profit_distance <= 0:
            raise ValueError("take_profit_distance must be > 0 when configured.")
        self.fast = fast
        self.slow = slow
        self.stop_loss_distance = stop_loss_distance
        self.take_profit_distance = take_profit_distance

    def on_bar(self, symbol: str, bars: list[MT5Bar]) -> Signal:
        # Need slow+1 bars to compare this bar's relationship against the previous bar's.
        if len(bars) < self.slow + 1:
            return HOLD

        closes = [bar.close for bar in bars]
        prev_fast = _sma(closes[:-1], self.fast)
        prev_slow = _sma(closes[:-1], self.slow)
        curr_fast = _sma(closes, self.fast)
        curr_slow = _sma(closes, self.slow)

        crossed_up = prev_fast <= prev_slow and curr_fast > curr_slow
        crossed_down = prev_fast >= prev_slow and curr_fast < curr_slow

        if crossed_up:
            return self._entry_signal(
                "enter_long", bars[-1].close, f"sma{self.fast}x{self.slow} up"
            )
        if crossed_down:
            return self._entry_signal(
                "enter_short", bars[-1].close, f"sma{self.fast}x{self.slow} down"
            )
        return HOLD

    def _entry_signal(self, action: SignalAction, entry_price: float, comment: str) -> Signal:
        is_long = action == "enter_long"
        stop_loss = None
        take_profit = None
        if self.stop_loss_distance is not None:
            stop_loss = (
                entry_price - self.stop_loss_distance
                if is_long
                else entry_price + self.stop_loss_distance
            )
        if self.take_profit_distance is not None:
            take_profit = (
                entry_price + self.take_profit_distance
                if is_long
                else entry_price - self.take_profit_distance
            )
        return Signal(
            action=action,
            stop_loss=stop_loss,
            take_profit=take_profit,
            comment=comment,
        )


class M5TrendM1EntryStrategy(MT5Strategy):
    """
    M1 entry strategy gated by an internally aggregated M5 trend filter.

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

    def on_bar(self, symbol: str, bars: list[MT5Bar]) -> Signal:
        if len(bars) < self._minimum_m1_bars:
            return HOLD

        current_bar = bars[-1]
        if self._is_blackout(current_bar.time):
            return Signal(action="exit", comment="news blackout")

        m5_bars = _aggregate_m1_to_m5(bars)
        if len(m5_bars) < self._minimum_m5_bars:
            return HOLD

        trend = self._m5_trend(m5_bars)
        trend_exit = self._trend_exit_signal(symbol, m5_bars, trend)
        if trend_exit is not None:
            return trend_exit

        if self.use_session_filter and not self._in_session(current_bar.time):
            return HOLD

        if trend == "long" and self._m1_buy_trigger(bars):
            stop_loss = self._stop_loss("buy", bars, m5_bars)
            tp1, fraction, move_be = self._scale_out("buy", current_bar.close, stop_loss)
            return Signal(
                action="enter_long",
                stop_loss=stop_loss,
                take_profit=self._take_profit("buy", current_bar.close, stop_loss),
                tp1=tp1,
                tp1_close_fraction=fraction,
                move_sl_to_breakeven=move_be,
                comment="m5 trend long + m1 stoch rsi trigger",
            )
        if trend == "short" and self._m1_sell_trigger(bars):
            stop_loss = self._stop_loss("sell", bars, m5_bars)
            tp1, fraction, move_be = self._scale_out("sell", current_bar.close, stop_loss)
            return Signal(
                action="enter_short",
                stop_loss=stop_loss,
                take_profit=self._take_profit("sell", current_bar.close, stop_loss),
                tp1=tp1,
                tp1_close_fraction=fraction,
                move_sl_to_breakeven=move_be,
                comment="m5 trend short + m1 stoch rsi trigger",
            )
        return HOLD

    def on_position_closed(self, symbol: str) -> None:
        # A real close (SL/TP/external/reconcile) ends the tracked trend position; forget it so
        # the next opposite M5 setup is not masked by a stale exit on this symbol.
        self._last_trend.pop(symbol, None)

    @property
    def _minimum_m5_bars(self) -> int:
        return max(self.trend_slow, self.trend_rsi_length + 1, self.trend_bb_length) + 1

    @property
    def _minimum_m1_bars(self) -> int:
        stoch_need = self.trend_fast + self.stoch_rsi_length + self.stoch_k_smooth
        stoch_need += self.stoch_d_smooth + 2
        return max(self._minimum_m5_bars * 5, stoch_need, self.swing_lookback)

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

        if close > fast and fast > slow and curr_rsi > 50 and curr_rsi > prev_rsi:
            return "long" if curr_bb > 0.5 else "neutral"
        if close < fast and fast < slow and curr_rsi < 50 and curr_rsi < prev_rsi:
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


def _aggregate_m1_to_m5(bars: list[MT5Bar]) -> list[MT5Bar]:
    buckets: dict[int, list[MT5Bar]] = {}
    for bar in bars:
        bucket = bar.time - (bar.time % 300)
        buckets.setdefault(bucket, []).append(bar)

    aggregated: list[MT5Bar] = []
    for bucket, items in sorted(buckets.items()):
        if len(items) < 5:
            continue
        ordered = sorted(items, key=lambda item: item.time)
        aggregated.append(
            MT5Bar(
                time=bucket,
                open=ordered[0].open,
                high=max(item.high for item in ordered),
                low=min(item.low for item in ordered),
                close=ordered[-1].close,
                volume=sum(item.volume for item in ordered),
            )
        )
    return aggregated


def _ema_series(values: list[float], length: int) -> list[float]:
    alpha = 2 / (length + 1)
    ema = values[0]
    series = [ema]
    for value in values[1:]:
        ema = value * alpha + ema * (1 - alpha)
        series.append(ema)
    return series


def _rsi_series(values: list[float], length: int) -> list[float | None]:
    result: list[float | None] = [None] * len(values)
    if len(values) <= length:
        return result

    gains: list[float] = []
    losses: list[float] = []
    for index in range(1, length + 1):
        change = values[index] - values[index - 1]
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))

    avg_gain = sum(gains) / length
    avg_loss = sum(losses) / length
    result[length] = _rsi_from_averages(avg_gain, avg_loss)
    for index in range(length + 1, len(values)):
        change = values[index] - values[index - 1]
        gain = max(change, 0.0)
        loss = max(-change, 0.0)
        avg_gain = (avg_gain * (length - 1) + gain) / length
        avg_loss = (avg_loss * (length - 1) + loss) / length
        result[index] = _rsi_from_averages(avg_gain, avg_loss)
    return result


def _rsi_from_averages(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0:
        return 100.0
    relative_strength = avg_gain / avg_loss
    return 100 - (100 / (1 + relative_strength))


def _bb_percent_b(
    values: list[float],
    length: int,
    stddev_multiplier: float,
) -> list[float | None]:
    result: list[float | None] = [None] * len(values)
    for index in range(length - 1, len(values)):
        window = values[index - length + 1 : index + 1]
        mean = sum(window) / length
        variance = sum((value - mean) ** 2 for value in window) / length
        stddev = math.sqrt(variance)
        upper = mean + stddev_multiplier * stddev
        lower = mean - stddev_multiplier * stddev
        result[index] = 0.5 if upper == lower else (values[index] - lower) / (upper - lower)
    return result


def _stoch_rsi(
    values: list[float],
    rsi_length: int,
    stoch_length: int,
) -> list[float | None]:
    rsi = _rsi_series(values, rsi_length)
    result: list[float | None] = [None] * len(values)
    for index in range(len(values)):
        window = rsi[index - stoch_length + 1 : index + 1]
        if len(window) < stoch_length or any(value is None for value in window):
            continue
        concrete = [float(value) for value in window if value is not None]
        low = min(concrete)
        high = max(concrete)
        current_rsi = rsi[index]
        if current_rsi is None:
            continue
        result[index] = 50.0 if high == low else (current_rsi - low) / (high - low) * 100
    return result


def _sma_optional_series(values: list[float | None], length: int) -> list[float | None]:
    result: list[float | None] = [None] * len(values)
    for index in range(length - 1, len(values)):
        window = values[index - length + 1 : index + 1]
        if any(value is None for value in window):
            continue
        result[index] = sum(float(value) for value in window if value is not None) / length
    return result


def _crosses_up_from_zone(
    k: list[float | None],
    d: list[float | None],
    zone: float,
) -> bool:
    values = _last_two_pairs(k, d)
    if values is None:
        return False
    prev_k, curr_k, prev_d, curr_d = values
    return bool(
        prev_k <= prev_d
        and min(prev_k, prev_d) < zone
        and curr_k > curr_d
        and curr_k > zone
    )


def _crosses_down_from_zone(
    k: list[float | None],
    d: list[float | None],
    zone: float,
) -> bool:
    values = _last_two_pairs(k, d)
    if values is None:
        return False
    prev_k, curr_k, prev_d, curr_d = values
    return bool(
        prev_k >= prev_d
        and max(prev_k, prev_d) > zone
        and curr_k < curr_d
        and curr_k < zone
    )


def _last_two_pairs(
    k: list[float | None],
    d: list[float | None],
) -> tuple[float, float, float, float] | None:
    if len(k) < 2 or len(d) < 2:
        return None
    prev_k = k[-2]
    curr_k = k[-1]
    prev_d = d[-2]
    curr_d = d[-1]
    if prev_k is None or curr_k is None or prev_d is None or curr_d is None:
        return None
    return prev_k, curr_k, prev_d, curr_d


def _parse_sessions(values: list[str]) -> list[tuple[time, time]]:
    sessions: list[tuple[time, time]] = []
    for value in values:
        start, separator, end = value.partition("-")
        if not separator:
            raise ValueError(f"Invalid session {value!r}. Expected 'HH:MM-HH:MM'.")
        sessions.append((_parse_time(start), _parse_time(end)))
    return sessions


def _parse_time(value: str) -> time:
    hour, separator, minute = value.strip().partition(":")
    if not separator:
        raise ValueError(f"Invalid time {value!r}. Expected 'HH:MM'.")
    return time(hour=int(hour), minute=int(minute))


def _time_in_range(value: time, start: time, end: time) -> bool:
    if start <= end:
        return start <= value <= end
    return value >= start or value <= end


def _parse_blackout_windows(values: list[dict[str, str]]) -> list[tuple[datetime, datetime]]:
    windows: list[tuple[datetime, datetime]] = []
    for value in values:
        start = _parse_datetime(value["from"])
        end = _parse_datetime(value["to"])
        if start > end:
            raise ValueError("blackout_windows 'from' must be earlier than 'to'.")
        windows.append((start, end))
    return windows


def _parse_datetime(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _validate_positive_int(name: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{name} must be >= 1.")
