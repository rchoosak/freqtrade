from __future__ import annotations

import math
from collections.abc import Callable
from datetime import UTC, datetime

from freqtrade.mt5_trade.data import MT5Bar


def _sma(values: list[float], length: int) -> float:
    window = values[-length:]
    return sum(window) / len(window)


def _aggregate_m1(bars: list[MT5Bar], bucket_seconds: int) -> list[MT5Bar]:
    # Aggregate M1 bars into ``bucket_seconds`` candles, keeping only completed buckets (those with
    # the full count of constituent minutes), so an in-progress HTF candle never biases a filter.
    expected = max(1, bucket_seconds // 60)
    buckets: dict[int, list[MT5Bar]] = {}
    for bar in bars:
        bucket = bar.time - (bar.time % bucket_seconds)
        buckets.setdefault(bucket, []).append(bar)

    aggregated: list[MT5Bar] = []
    for bucket, items in sorted(buckets.items()):
        if len(items) < expected:
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


def _aggregate_m1_to_m5(bars: list[MT5Bar]) -> list[MT5Bar]:
    return _aggregate_m1(bars, 300)


def _aggregate_completed_bars(
    bars: list[MT5Bar],
    bucket_seconds: int,
    source_seconds: int,
    *,
    anchor_seconds: int = 0,
    expected_slot: Callable[[int], bool] | None = None,
    merge_sunday_into_monday: bool = False,
) -> list[MT5Bar]:
    """
    Aggregate lower-timeframe bars only when every expected source slot is present.

    ``expected_slot`` makes completeness session-aware: scheduled closures are excluded from the
    required slots, while an unexpected missing bar rejects the bucket. For markets with a short
    Sunday session, ``merge_sunday_into_monday`` folds that fragment into the completed Monday D1
    candle instead of treating Sunday as a standalone trading day.
    """
    if bucket_seconds < source_seconds or bucket_seconds % source_seconds != 0:
        raise ValueError("bucket_seconds must be a multiple of source_seconds.")
    if not bars:
        return []

    buckets: dict[int, list[MT5Bar]] = {}
    for bar in bars:
        bucket = ((bar.time - anchor_seconds) // bucket_seconds) * bucket_seconds
        bucket += anchor_seconds
        buckets.setdefault(bucket, []).append(bar)

    latest_bucket = max(buckets)
    final_slot_offset = bucket_seconds - source_seconds
    aggregated: dict[int, MT5Bar] = {}
    sunday_fragments: set[int] = set()
    for bucket, items in sorted(buckets.items()):
        ordered = sorted(items, key=lambda item: item.time)
        actual_slots = {item.time for item in ordered}
        expected_slots = [
            timestamp
            for timestamp in range(bucket, bucket + bucket_seconds, source_seconds)
            if expected_slot is None or expected_slot(timestamp)
        ]
        if not expected_slots or any(timestamp not in actual_slots for timestamp in expected_slots):
            continue
        # A session may end before the bucket's nominal final slot (Friday close, holidays).
        # Do not finalize such a latest bucket until a later bucket proves the session is over;
        # otherwise a late DST-dependent bar can arrive after the strategy already processed it.
        if (
            bucket == latest_bucket
            and expected_slots[-1] < bucket + final_slot_offset
        ):
            continue
        aggregated[bucket] = MT5Bar(
            time=bucket,
            open=ordered[0].open,
            high=max(item.high for item in ordered),
            low=min(item.low for item in ordered),
            close=ordered[-1].close,
            volume=sum(item.volume for item in ordered),
        )
        if (
            datetime.fromtimestamp(bucket, UTC).weekday() == 6
            and len(expected_slots) < bucket_seconds // source_seconds / 2
        ):
            sunday_fragments.add(bucket)

    if merge_sunday_into_monday:
        for bucket in sunday_fragments:
            monday_bucket = bucket + bucket_seconds
            sunday = aggregated.pop(bucket)
            monday = aggregated.get(monday_bucket)
            if monday is None:
                continue
            aggregated[monday_bucket] = MT5Bar(
                time=monday.time,
                open=sunday.open,
                high=max(sunday.high, monday.high),
                low=min(sunday.low, monday.low),
                close=monday.close,
                volume=sunday.volume + monday.volume,
            )

    return [aggregated[bucket] for bucket in sorted(aggregated)]


def _atr_series(bars: list[MT5Bar], length: int) -> list[float | None]:
    # Wilder's ATR. True range uses the previous bar's close, so result[i] is None until
    # there are `length` completed true-range values (i.e. from index `length` onward).
    result: list[float | None] = [None] * len(bars)
    if len(bars) <= length:
        return result

    true_ranges: list[float] = []
    for index in range(1, len(bars)):
        high = bars[index].high
        low = bars[index].low
        prev_close = bars[index - 1].close
        true_ranges.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))

    # true_ranges[j] is the TR of bars[j + 1]; the first ATR (at bar index `length`) is the
    # simple mean of the first `length` true ranges, then smoothed Wilder-style after that.
    atr = sum(true_ranges[:length]) / length
    result[length] = atr
    for index in range(length + 1, len(bars)):
        atr = (atr * (length - 1) + true_ranges[index - 1]) / length
        result[index] = atr
    return result


def _adx_series(bars: list[MT5Bar], length: int) -> list[float | None]:
    """Wilder ADX series aligned to ``bars``; the first value appears at ``2*length - 1``."""
    result: list[float | None] = [None] * len(bars)
    if len(bars) < 2 * length:
        return result

    true_ranges: list[float] = []
    plus_dm: list[float] = []
    minus_dm: list[float] = []
    for index in range(1, len(bars)):
        current = bars[index]
        previous = bars[index - 1]
        up_move = current.high - previous.high
        down_move = previous.low - current.low
        true_ranges.append(
            max(
                current.high - current.low,
                abs(current.high - previous.close),
                abs(current.low - previous.close),
            )
        )
        plus_dm.append(up_move if up_move > down_move and up_move > 0 else 0.0)
        minus_dm.append(down_move if down_move > up_move and down_move > 0 else 0.0)

    smoothed_tr = sum(true_ranges[:length])
    smoothed_plus = sum(plus_dm[:length])
    smoothed_minus = sum(minus_dm[:length])
    dx: list[float | None] = [None] * len(bars)
    for index in range(length, len(bars)):
        if index > length:
            source_index = index - 1
            smoothed_tr = smoothed_tr - smoothed_tr / length + true_ranges[source_index]
            smoothed_plus = (
                smoothed_plus - smoothed_plus / length + plus_dm[source_index]
            )
            smoothed_minus = (
                smoothed_minus - smoothed_minus / length + minus_dm[source_index]
            )
        plus_di = 100 * smoothed_plus / smoothed_tr if smoothed_tr > 0 else 0.0
        minus_di = 100 * smoothed_minus / smoothed_tr if smoothed_tr > 0 else 0.0
        denominator = plus_di + minus_di
        dx[index] = 100 * abs(plus_di - minus_di) / denominator if denominator > 0 else 0.0

    first_adx = 2 * length - 1
    initial_dx = [float(value) for value in dx[length : first_adx + 1] if value is not None]
    result[first_adx] = sum(initial_dx) / length
    for index in range(first_adx + 1, len(bars)):
        current_dx = dx[index]
        previous_adx = result[index - 1]
        if current_dx is None or previous_adx is None:
            continue
        result[index] = (previous_adx * (length - 1) + current_dx) / length
    return result


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
