from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from freqtrade.mt5_trade.data import MT5Bar
from freqtrade.mt5_trade.strategies.base import (
    HOLD,
    MT5Strategy,
    Signal,
    SignalAction,
    _validate_positive_int,
)
from freqtrade.mt5_trade.strategies.indicators import (
    _aggregate_m1,
    _atr_series,
    _ema_series,
    _sma,
)
from freqtrade.mt5_trade.strategies.time_filters import _parse_sessions, _time_in_range


@dataclass(frozen=True)
class _OrderBlock:
    """A detected order-block zone. ``side`` is the trade direction it enables."""

    side: str  # "bullish" -> long setup, "bearish" -> short setup
    low: float
    high: float
    formed_index: int  # index of the structure break that confirmed the OB


class SmcOrderBlockStrategy(MT5Strategy):
    """
    Smart-Money-Concepts strategy for XAUUSD M1: Order Block detection + Liquidity Sweep reaction.

    Combined sequential rule, evaluated on the close of each M1 candle (``on_bar``):

    1. Track the most recent valid Order Block. A *bullish* OB is the last bearish candle before an
       up-impulse that closes above the prior swing high (a bullish break of structure); a *bearish*
       OB is the last bullish candle before a down-impulse that closes below the prior swing low.
    2. A setup activates only when the current candle mitigates that OB and performs a liquidity
       sweep: for a bullish OB the wick pierces below the OB low (sweeps sell-side liquidity) but
       the candle closes back inside the zone; symmetric above the OB high for a bearish OB.
    3. Entry fires when that same candle is a reversal-confirmation candle (engulfing or long-wick
       pinbar in the trade direction).

    Filters: a high-liquidity session window (default London/NY overlap 12:00-21:00 UTC) and a
    volatility gate (ATR floor and/or volume above its average) to avoid dead-hour fakeouts.

    The strategy only emits a ``Signal``; connection, the candle loop, position sizing, SL/TP and
    the breakeven scale-out are owned by ``MT5ForexBot`` / the backtester, so the exact same code
    runs live and in backtest. Stateless: a same-direction re-entry is a no-op via the bot's
    single-position-per-symbol rule, so no per-symbol tracking is needed.
    """

    def __init__(
        self,
        *,
        structure_lookback: int = 50,
        swing_strength: int = 2,
        atr_length: int = 14,
        min_atr: float = 0.0,
        volume_length: int = 20,
        volume_factor: float = 1.0,
        point: float = 0.01,
        stop_buffer_points: float = 10.0,
        risk_reward: float = 2.0,
        breakeven_points: float = 200.0,
        tp1_close_fraction: float = 0.5,
        pinbar_wick_ratio: float = 2.0,
        use_session_filter: bool = True,
        timezone: str = "UTC",
        sessions: list[str] | None = None,
        use_htf_filter: bool = False,
        htf_minutes: int = 15,
        htf_ema_fast: int = 21,
        htf_ema_slow: int = 50,
        htf_invert: bool = False,
    ) -> None:
        _validate_positive_int("structure_lookback", structure_lookback)
        _validate_positive_int("swing_strength", swing_strength)
        _validate_positive_int("atr_length", atr_length)
        _validate_positive_int("volume_length", volume_length)
        if point <= 0:
            raise ValueError("point must be > 0.")
        if risk_reward <= 0:
            raise ValueError("risk_reward must be > 0.")
        if pinbar_wick_ratio <= 0:
            raise ValueError("pinbar_wick_ratio must be > 0.")
        if min_atr < 0:
            raise ValueError("min_atr must be >= 0.")
        if volume_factor < 0:
            raise ValueError("volume_factor must be >= 0.")
        if stop_buffer_points < 0:
            raise ValueError("stop_buffer_points must be >= 0.")
        if breakeven_points < 0:
            raise ValueError("breakeven_points must be >= 0.")
        if not 0 < tp1_close_fraction < 1:
            raise ValueError("tp1_close_fraction must be between 0 and 1 (exclusive).")
        _validate_positive_int("htf_minutes", htf_minutes)
        _validate_positive_int("htf_ema_fast", htf_ema_fast)
        _validate_positive_int("htf_ema_slow", htf_ema_slow)
        if htf_ema_fast >= htf_ema_slow:
            raise ValueError("htf_ema_fast must be shorter than htf_ema_slow.")

        self.structure_lookback = structure_lookback
        self.swing_strength = swing_strength
        self.atr_length = atr_length
        self.min_atr = min_atr
        self.volume_length = volume_length
        self.volume_factor = volume_factor
        self.point = point
        self.stop_buffer = stop_buffer_points * point
        self.risk_reward = risk_reward
        self.breakeven_distance = breakeven_points * point
        self.tp1_close_fraction = tp1_close_fraction
        self.pinbar_wick_ratio = pinbar_wick_ratio
        self.use_session_filter = use_session_filter
        self.timezone = ZoneInfo(timezone)
        self.sessions = _parse_sessions(sessions or ["12:00-21:00"])
        self.use_htf_filter = use_htf_filter
        self.htf_minutes = htf_minutes
        self.htf_ema_fast = htf_ema_fast
        self.htf_ema_slow = htf_ema_slow
        self.htf_invert = htf_invert

    @property
    def required_timeframe(self) -> str:
        return "M1"

    @property
    def minimum_bars(self) -> int:
        # Enough history to confirm swings within the lookback, warm up ATR, and average volume.
        structure_need = self.structure_lookback + 2 * self.swing_strength + 1
        base = max(structure_need, self.atr_length + 1, self.volume_length)
        if self.use_htf_filter:
            # Also need enough M1 bars to build htf_ema_slow + 1 completed HTF candles.
            base = max(base, (self.htf_ema_slow + 1) * self.htf_minutes)
        return base + 1

    def on_bar(self, symbol: str, bars: list[MT5Bar]) -> Signal:
        if len(bars) < self.minimum_bars:
            return HOLD

        current = bars[-1]
        if self.use_session_filter and not self._in_session(current.time):
            return HOLD

        block = self._recent_order_block(bars)
        if block is None:
            return HOLD

        previous = bars[-2]
        if block.side == "bullish" and self._bullish_trigger(block, previous, current):
            if not self._htf_allows("long", bars):
                return HOLD
            if not self._volatility_ok(bars, current):
                return HOLD
            return self._entry_signal("enter_long", current.close, current.low - self.stop_buffer)
        if block.side == "bearish" and self._bearish_trigger(block, previous, current):
            if not self._htf_allows("short", bars):
                return HOLD
            if not self._volatility_ok(bars, current):
                return HOLD
            return self._entry_signal(
                "enter_short", current.close, current.high + self.stop_buffer
            )
        return HOLD

    # --- setup detection -------------------------------------------------------------------

    def _bullish_trigger(self, block: _OrderBlock, previous: MT5Bar, current: MT5Bar) -> bool:
        # Sweep: wick pierces below the OB low (mitigates the zone) but closes back inside it.
        swept = current.low < block.low and current.close > block.low
        if not swept:
            return False
        return _bullish_engulfing(previous, current) or _bullish_pinbar(
            current, self.pinbar_wick_ratio
        )

    def _bearish_trigger(self, block: _OrderBlock, previous: MT5Bar, current: MT5Bar) -> bool:
        swept = current.high > block.high and current.close < block.high
        if not swept:
            return False
        return _bearish_engulfing(previous, current) or _bearish_pinbar(
            current, self.pinbar_wick_ratio
        )

    def _recent_order_block(self, bars: list[MT5Bar]) -> _OrderBlock | None:
        n = len(bars)
        k = self.swing_strength
        start = max(0, n - self.structure_lookback)
        # Confirmed swings inside the lookback. The trigger candle (n-1) is never a swing itself.
        swing_highs = [i for i in range(start, n - 1) if _is_swing_high(bars, i, k)]
        swing_lows = [i for i in range(start, n - 1) if _is_swing_low(bars, i, k)]

        blocks: list[_OrderBlock] = []
        bullish = self._bullish_block(bars, swing_highs, start)
        if bullish is not None:
            blocks.append(bullish)
        bearish = self._bearish_block(bars, swing_lows, start)
        if bearish is not None:
            blocks.append(bearish)
        if not blocks:
            return None
        # Prefer the most recently formed structure break when both sides exist.
        return max(blocks, key=lambda block: block.formed_index)

    def _bullish_block(
        self, bars: list[MT5Bar], swing_highs: list[int], start: int
    ) -> _OrderBlock | None:
        k = self.swing_strength
        # Most recent bullish break of structure: a close above a swing high confirmed before it.
        # Keyed off the *broken* swing (not merely the newest one), so a higher unbroken impulse
        # peak does not mask the structure break that formed the order block.
        swing: int | None = None
        break_index: int | None = None
        for candidate in range(len(bars) - 1, start, -1):
            broken = [
                s
                for s in swing_highs
                if s + k <= candidate - 1 and bars[candidate].close > bars[s].high
            ]
            if broken:
                swing = max(broken)
                break_index = candidate
                break
        if swing is None or break_index is None:
            return None
        ob_index = _last_bearish(bars, swing, break_index)
        if ob_index is None:
            return None
        low, high = bars[ob_index].low, bars[ob_index].high
        # Invalidated if an already-closed candle closed below the zone before the trigger candle.
        if any(bars[j].close < low for j in range(ob_index + 1, len(bars) - 1)):
            return None
        return _OrderBlock("bullish", low, high, break_index)

    def _bearish_block(
        self, bars: list[MT5Bar], swing_lows: list[int], start: int
    ) -> _OrderBlock | None:
        k = self.swing_strength
        swing: int | None = None
        break_index: int | None = None
        for candidate in range(len(bars) - 1, start, -1):
            broken = [
                s
                for s in swing_lows
                if s + k <= candidate - 1 and bars[candidate].close < bars[s].low
            ]
            if broken:
                swing = max(broken)
                break_index = candidate
                break
        if swing is None or break_index is None:
            return None
        ob_index = _last_bullish(bars, swing, break_index)
        if ob_index is None:
            return None
        low, high = bars[ob_index].low, bars[ob_index].high
        if any(bars[j].close > high for j in range(ob_index + 1, len(bars) - 1)):
            return None
        return _OrderBlock("bearish", low, high, break_index)

    # --- filters ---------------------------------------------------------------------------

    def _volatility_ok(self, bars: list[MT5Bar], current: MT5Bar) -> bool:
        if self.min_atr > 0:
            atr = _atr_series(bars, self.atr_length)[-1]
            if atr is None or atr < self.min_atr:
                return False
        if self.volume_factor > 0:
            average = _sma([float(bar.volume) for bar in bars], self.volume_length)
            if average > 0 and current.volume < self.volume_factor * average:
                return False
        return True

    def _htf_allows(self, side: str, bars: list[MT5Bar]) -> bool:
        # Confluence gate. Default (htf_invert=False): trade with the HTF trend — longs need an up
        # bias, shorts a down bias. Inverted: fade the HTF trend (long into a down bias, short into
        # an up bias). A neutral bias blocks either way. Off when use_htf_filter is False.
        if not self.use_htf_filter:
            return True
        bias = self._htf_bias(bars)
        if side == "long":
            required = "down" if self.htf_invert else "up"
        else:
            required = "up" if self.htf_invert else "down"
        return bias == required

    def _htf_bias(self, bars: list[MT5Bar]) -> str:
        # Higher-timeframe confluence: aggregate M1 -> htf_minutes candles and read an EMA trend.
        # Longs require an up bias, shorts a down bias; otherwise the M1 setup is skipped.
        htf = _aggregate_m1(bars, self.htf_minutes * 60)
        if len(htf) < self.htf_ema_slow + 1:
            return "neutral"
        closes = [bar.close for bar in htf]
        fast = _ema_series(closes, self.htf_ema_fast)
        slow = _ema_series(closes, self.htf_ema_slow)
        last = closes[-1]
        if fast[-1] > slow[-1] and last > slow[-1]:
            return "up"
        if fast[-1] < slow[-1] and last < slow[-1]:
            return "down"
        return "neutral"

    def _in_session(self, timestamp: int) -> bool:
        local_time = datetime.fromtimestamp(timestamp, UTC).astimezone(self.timezone).time()
        return any(_time_in_range(local_time, start, end) for start, end in self.sessions)

    # --- signal construction ---------------------------------------------------------------

    def _entry_signal(self, action: SignalAction, entry_price: float, stop_loss: float) -> Signal:
        risk = abs(entry_price - stop_loss)
        is_long = action == "enter_long"
        take_profit = (
            entry_price + self.risk_reward * risk
            if is_long
            else entry_price - self.risk_reward * risk
        )
        tp1, fraction, move_be = self._scale_out(is_long, entry_price, risk)
        return Signal(
            action=action,
            stop_loss=stop_loss,
            take_profit=take_profit,
            tp1=tp1,
            tp1_close_fraction=fraction,
            move_sl_to_breakeven=move_be,
            comment=f"smc {_block_side(is_long)} OB sweep + reversal",
        )

    def _scale_out(
        self, is_long: bool, entry_price: float, risk: float
    ) -> tuple[float | None, float | None, bool]:
        # Move SL to breakeven by closing a fraction once price runs `breakeven_points` in favour.
        # Only when that TP1 sits between entry and the full TP, so it resolves before the full TP.
        if not (0 < self.breakeven_distance < self.risk_reward * risk):
            return None, None, False
        tp1 = (
            entry_price + self.breakeven_distance
            if is_long
            else entry_price - self.breakeven_distance
        )
        return tp1, self.tp1_close_fraction, True


def _block_side(is_long: bool) -> str:
    return "bullish" if is_long else "bearish"


def _is_swing_high(bars: list[MT5Bar], index: int, k: int) -> bool:
    if index - k < 0 or index + k >= len(bars):
        return False
    left = all(bars[index].high > bars[j].high for j in range(index - k, index))
    right = all(bars[index].high > bars[j].high for j in range(index + 1, index + k + 1))
    return left and right


def _is_swing_low(bars: list[MT5Bar], index: int, k: int) -> bool:
    if index - k < 0 or index + k >= len(bars):
        return False
    left = all(bars[index].low < bars[j].low for j in range(index - k, index))
    right = all(bars[index].low < bars[j].low for j in range(index + 1, index + k + 1))
    return left and right


def _last_bearish(bars: list[MT5Bar], start: int, before: int) -> int | None:
    # Last down-close candle in [start, before) — the bullish order block's origin candle.
    for index in range(before - 1, start - 1, -1):
        if bars[index].close < bars[index].open:
            return index
    return None


def _last_bullish(bars: list[MT5Bar], start: int, before: int) -> int | None:
    for index in range(before - 1, start - 1, -1):
        if bars[index].close > bars[index].open:
            return index
    return None


def _bullish_engulfing(previous: MT5Bar, current: MT5Bar) -> bool:
    return (
        current.close > current.open
        and previous.close < previous.open
        and current.close >= previous.open
        and current.open <= previous.close
    )


def _bearish_engulfing(previous: MT5Bar, current: MT5Bar) -> bool:
    return (
        current.close < current.open
        and previous.close > previous.open
        and current.close <= previous.open
        and current.open >= previous.close
    )


def _bullish_pinbar(bar: MT5Bar, ratio: float) -> bool:
    candle_range = bar.high - bar.low
    if candle_range <= 0:
        return False
    body = abs(bar.close - bar.open)
    lower_wick = min(bar.open, bar.close) - bar.low
    closes_high = (bar.high - bar.close) <= candle_range / 3
    return lower_wick >= ratio * body and closes_high


def _bearish_pinbar(bar: MT5Bar, ratio: float) -> bool:
    candle_range = bar.high - bar.low
    if candle_range <= 0:
        return False
    body = abs(bar.close - bar.open)
    upper_wick = bar.high - max(bar.open, bar.close)
    closes_low = (bar.close - bar.low) <= candle_range / 3
    return upper_wick >= ratio * body and closes_low
