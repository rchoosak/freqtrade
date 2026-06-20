from __future__ import annotations

import pytest

from freqtrade.mt5_trade.data import MT5Bar
from freqtrade.mt5_trade.strategies import M5TrendM1EntryStrategy, Signal, SmaCrossStrategy


def _bars(closes: list[float]) -> list[MT5Bar]:
    return [MT5Bar(time=i, open=c, high=c, low=c, close=c) for i, c in enumerate(closes)]


def _m1_bars_from_m5_closes(closes: list[float]) -> list[MT5Bar]:
    bars: list[MT5Bar] = []
    previous = closes[0]
    for bucket_index, close in enumerate(closes):
        bucket_start = bucket_index * 300
        open_price = previous if bucket_index else close
        values = [
            open_price + (close - open_price) * (index + 1) / 5 for index in range(5)
        ]
        for index, value in enumerate(values):
            bars.append(
                MT5Bar(
                    time=bucket_start + index * 60,
                    open=value,
                    high=value + 0.1,
                    low=value - 0.1,
                    close=value,
                )
            )
        previous = close
    return bars


def _append_m1_closes(bars: list[MT5Bar], closes: list[float]) -> list[MT5Bar]:
    result = list(bars)
    start = len(result) * 60
    for index, close in enumerate(closes):
        result.append(
            MT5Bar(
                time=start + index * 60,
                open=close,
                high=close + 0.1,
                low=close - 0.1,
                close=close,
            )
        )
    return result


def _m1_bars_from_h1_closes(closes: list[float]) -> list[MT5Bar]:
    bars: list[MT5Bar] = []
    previous = closes[0]
    for hour_index, close in enumerate(closes):
        hour_start = hour_index * 3600
        open_price = previous if hour_index else close
        for minute in range(60):
            value = open_price + (close - open_price) * (minute + 1) / 60
            bars.append(
                MT5Bar(
                    time=hour_start + minute * 60,
                    open=value,
                    high=value + 0.1,
                    low=value - 0.1,
                    close=value,
                )
            )
        previous = close
    return bars


def _shift_bars(bars: list[MT5Bar], start: int) -> list[MT5Bar]:
    return [
        MT5Bar(
            time=start + index * 60,
            open=bar.open,
            high=bar.high,
            low=bar.low,
            close=bar.close,
            volume=bar.volume,
        )
        for index, bar in enumerate(bars)
    ]


def _m5_m1_strategy() -> M5TrendM1EntryStrategy:
    return M5TrendM1EntryStrategy(
        trend_fast=2,
        trend_slow=4,
        trend_rsi_length=3,
        trend_bb_length=4,
        sideways_rsi_low=49,
        sideways_rsi_high=51,
        stoch_rsi_length=3,
        stoch_k_smooth=1,
        stoch_d_smooth=2,
        swing_lookback=5,
        stop_buffer=0.5,
        use_session_filter=False,
    )


def test_sma_cross_rejects_bad_lengths() -> None:
    with pytest.raises(ValueError, match="shorter than slow"):
        SmaCrossStrategy(fast=30, slow=10)


def test_signal_rejects_scale_out_pending_entry() -> None:
    with pytest.raises(ValueError, match="tp1 scale-out is only supported for market"):
        Signal(
            action="enter_long",
            order_kind="limit",
            price=1.0,
            tp1=2.0,
            tp1_close_fraction=0.5,
        )


def test_sma_cross_holds_without_enough_bars() -> None:
    strategy = SmaCrossStrategy(fast=2, slow=3)
    # slow + 1 = 4 bars required.
    assert strategy.on_bar("EURUSD", _bars([1, 2, 3])).action == "hold"


def test_sma_cross_enters_long_on_cross_up() -> None:
    strategy = SmaCrossStrategy(fast=2, slow=3)
    # Flat then a sharp rise so the fast SMA crosses above the slow SMA on the last bar.
    signal = strategy.on_bar("EURUSD", _bars([10, 10, 10, 20]))
    assert signal.action == "enter_long"


def test_sma_cross_enters_short_on_cross_down() -> None:
    strategy = SmaCrossStrategy(fast=2, slow=3)
    # Flat then a sharp drop so the fast SMA crosses below the slow SMA on the last bar.
    signal = strategy.on_bar("EURUSD", _bars([10, 10, 10, 1]))
    assert signal.action == "enter_short"


def test_sma_cross_holds_when_no_cross() -> None:
    strategy = SmaCrossStrategy(fast=2, slow=3)
    # Steady uptrend already established; no fresh cross on the last bar.
    signal = strategy.on_bar("EURUSD", _bars([1, 2, 3, 4, 5, 6]))
    assert signal.action == "hold"


def test_m5_trend_m1_entry_enters_long_on_pullback_recovery() -> None:
    bars = _m1_bars_from_m5_closes([100, 101, 102, 103, 104, 103, 105, 106])
    bars = _append_m1_closes(bars, [110, 107, 104, 105])

    signal = _m5_m1_strategy().on_bar("XAUUSD", bars)

    assert signal.action == "enter_long"
    assert signal.stop_loss == 103.4


def test_m5_trend_m1_entry_scale_out_mode_emits_tp1() -> None:
    bars = _m1_bars_from_m5_closes([100, 101, 102, 103, 104, 103, 105, 106])
    bars = _append_m1_closes(bars, [110, 107, 104, 105])
    strategy = M5TrendM1EntryStrategy(
        trend_fast=2,
        trend_slow=4,
        trend_rsi_length=3,
        trend_bb_length=4,
        stoch_rsi_length=3,
        stoch_k_smooth=1,
        stoch_d_smooth=2,
        swing_lookback=5,
        stop_buffer=0.5,
        use_session_filter=False,
        take_profit_mode="scale_out",
        tp1_rr=1.0,
        tp1_close_fraction=0.5,
    )

    signal = strategy.on_bar("XAUUSD", bars)

    # entry=105, stop=103.4 -> risk 1.6 -> TP1 = 105 + 1.6*1.0 = 106.6
    assert signal.action == "enter_long"
    assert signal.stop_loss == 103.4
    assert signal.tp1 == 106.6
    assert signal.tp1_close_fraction == 0.5
    assert signal.move_sl_to_breakeven is True
    assert signal.take_profit is None  # runner has no fixed full TP


def test_m5_trend_m1_entry_enters_short_on_rebound_rejection() -> None:
    bars = _m1_bars_from_m5_closes([110, 109, 108, 107, 106, 107, 105, 104])
    bars = _append_m1_closes(bars, [100, 103, 106, 105])

    signal = _m5_m1_strategy().on_bar("XAUUSD", bars)

    assert signal.action == "enter_short"
    assert signal.stop_loss == 106.6


def test_m5_trend_m1_h1_bias_requires_ema_alignment_and_slow_slope() -> None:
    strategy = M5TrendM1EntryStrategy(
        trend_fast=2,
        trend_slow=4,
        trend_rsi_length=3,
        trend_bb_length=4,
        h1_fast=2,
        h1_slow=4,
        h1_slope_lookback=1,
        use_h1_filter=True,
        use_session_filter=False,
    )

    up = _m1_bars_from_h1_closes([100, 101, 102, 103, 104])
    down = _m1_bars_from_h1_closes([104, 103, 102, 101, 100])
    flat = _m1_bars_from_h1_closes([100, 100, 100, 100, 100])

    assert strategy._h1_bias("UP", up) == "long"
    assert strategy._h1_bias("DOWN", down) == "short"
    assert strategy._h1_bias("FLAT", flat) == "neutral"


def test_m5_trend_m1_h1_filter_modes_handle_neutral_bias() -> None:
    strict = M5TrendM1EntryStrategy(use_h1_filter=True, h1_filter_mode="strict")
    block_opposite = M5TrendM1EntryStrategy(
        use_h1_filter=True,
        h1_filter_mode="block_opposite",
    )

    assert strict._h1_allows("long", "neutral") is False
    assert block_opposite._h1_allows("long", "neutral") is True
    assert block_opposite._h1_allows("short", "long") is False
    assert block_opposite._h1_allows("short", "short") is True


def test_m5_trend_m1_rejects_unknown_h1_filter_mode() -> None:
    with pytest.raises(ValueError, match="h1_filter_mode"):
        M5TrendM1EntryStrategy(h1_filter_mode="loose")


def test_m5_trend_m1_h1_filter_blocks_short_during_h1_uptrend() -> None:
    h1_context = _m1_bars_from_h1_closes([100, 105, 110, 115])
    short_setup = _m1_bars_from_m5_closes([110, 109, 108, 107, 106, 107, 105, 104])
    short_setup = _append_m1_closes(short_setup, [100, 103, 106, 105])
    bars = h1_context + _shift_bars(short_setup, len(h1_context) * 60)
    strategy = M5TrendM1EntryStrategy(
        trend_fast=2,
        trend_slow=4,
        trend_rsi_length=3,
        trend_bb_length=4,
        sideways_rsi_low=49,
        sideways_rsi_high=51,
        stoch_rsi_length=3,
        stoch_k_smooth=1,
        stoch_d_smooth=2,
        swing_lookback=5,
        stop_buffer=0.5,
        use_h1_filter=True,
        h1_fast=2,
        h1_slow=3,
        h1_slope_lookback=1,
        use_session_filter=False,
    )

    assert strategy._h1_bias("XAUUSD", bars) == "long"
    assert strategy._m5_trend(_m1_bars_from_m5_closes(
        [110, 109, 108, 107, 106, 107, 105, 104]
    )) == "short"
    assert strategy.on_bar("XAUUSD", bars).action == "hold"


def test_m5_trend_m1_entry_holds_outside_configured_session() -> None:
    bars = _m1_bars_from_m5_closes([100, 101, 102, 103, 104, 103, 105, 106])
    bars = _append_m1_closes(bars, [110, 107, 104, 105])
    strategy = M5TrendM1EntryStrategy(
        trend_fast=2,
        trend_slow=4,
        trend_rsi_length=3,
        trend_bb_length=4,
        sideways_rsi_low=49,
        sideways_rsi_high=51,
        stoch_rsi_length=3,
        stoch_k_smooth=1,
        stoch_d_smooth=2,
        swing_lookback=5,
        stop_buffer=0.5,
        use_session_filter=True,
        timezone="UTC",
        sessions=["23:00-23:59"],
    )

    assert strategy.on_bar("XAUUSD", bars).action == "hold"


def test_m5_trend_m1_entry_exits_during_blackout_window() -> None:
    bars = _m1_bars_from_m5_closes([100, 101, 102, 103, 104, 103, 105, 106])
    strategy = M5TrendM1EntryStrategy(
        trend_fast=2,
        trend_slow=4,
        trend_rsi_length=3,
        trend_bb_length=4,
        blackout_windows=[{"from": "1970-01-01T00:00:00Z", "to": "1970-01-01T01:00:00Z"}],
    )

    assert strategy.on_bar("XAUUSD", bars).action == "exit"


def test_m5_trend_m1_entry_exits_when_m5_trend_invalidates() -> None:
    strategy = _m5_m1_strategy()
    long_bars = _m1_bars_from_m5_closes([100, 101, 102, 103, 104, 103, 105, 106])
    neutral_bars = _m1_bars_from_m5_closes([100, 101, 102, 103, 104, 103, 105, 106, 104])

    assert strategy.on_bar("XAUUSD", long_bars).action == "hold"
    assert strategy.on_bar("XAUUSD", neutral_bars).action == "exit"


def test_m5_trend_m1_entry_resets_state_on_position_closed() -> None:
    strategy = _m5_m1_strategy()
    long_bars = _m1_bars_from_m5_closes([100, 101, 102, 103, 104, 103, 105, 106])
    neutral_bars = _m1_bars_from_m5_closes([100, 101, 102, 103, 104, 103, 105, 106, 104])

    # on_bar records the M5 trend as the active position direction.
    assert strategy.on_bar("XAUUSD", long_bars).action == "hold"
    assert strategy._last_trend.get("XAUUSD") == "long"

    # An external close (SL/TP/broker/reconcile) clears the tracked state...
    strategy.on_position_closed("XAUUSD")
    assert "XAUUSD" not in strategy._last_trend

    # ...so the trend-invalidation no longer emits a stale exit for a position that is gone
    # (contrast test_m5_trend_m1_entry_exits_when_m5_trend_invalidates, which sees "exit").
    assert strategy.on_bar("XAUUSD", neutral_bars).action != "exit"


def _m5_previous_strategy() -> M5TrendM1EntryStrategy:
    return M5TrendM1EntryStrategy(
        trend_fast=2,
        trend_slow=4,
        trend_rsi_length=3,
        trend_bb_length=4,
        stoch_rsi_length=3,
        stoch_k_smooth=1,
        stoch_d_smooth=2,
        stop_mode="m5_previous",
        stop_buffer=0.5,
        use_session_filter=False,
    )


def _m5(time: int, low: float, high: float) -> MT5Bar:
    return MT5Bar(time=time, open=low, high=high, low=low, close=low)


def test_m5_previous_stop_steps_back_when_trigger_on_fifth_minute() -> None:
    strategy = _m5_previous_strategy()
    m5_bars = [_m5(0, 10, 11), _m5(300, 20, 21), _m5(600, 30, 31)]
    # Entry M1 bar at the 5th minute (offset 240) of bucket 600 -> that bucket is m5_bars[-1]
    # (the entry's own candle), so the "previous M5 candle" is m5_bars[-2] (bucket 300).
    m1_bars = [MT5Bar(time=840, open=30, high=31, low=30, close=30)]

    assert strategy._stop_loss("buy", m1_bars, m5_bars) == 20 - 0.5
    assert strategy._stop_loss("sell", m1_bars, m5_bars) == 21 + 0.5


def test_m5_previous_stop_uses_last_completed_bucket_mid_bucket() -> None:
    strategy = _m5_previous_strategy()
    m5_bars = [_m5(0, 10, 11), _m5(300, 20, 21), _m5(600, 30, 31)]
    # Entry M1 bar at the 2nd minute (offset 60) of the in-progress bucket 900 (not in m5_bars),
    # so m5_bars[-1] (bucket 600) is already the previous completed M5 candle.
    m1_bars = [MT5Bar(time=960, open=30, high=31, low=30, close=30)]

    assert strategy._stop_loss("buy", m1_bars, m5_bars) == 30 - 0.5
    assert strategy._stop_loss("sell", m1_bars, m5_bars) == 31 + 0.5


def test_signal_rejects_unknown_action() -> None:
    with pytest.raises(ValueError, match="Invalid action"):
        Signal(action="enter_longg")  # type: ignore[arg-type]


def test_signal_rejects_unknown_order_kind() -> None:
    with pytest.raises(ValueError, match="Invalid order_kind"):
        Signal(action="enter_long", order_kind="stoploss", price=1.0)  # type: ignore[arg-type]


def test_signal_rejects_non_positive_volume() -> None:
    with pytest.raises(ValueError, match="Invalid volume"):
        Signal(action="enter_long", volume=0)
    with pytest.raises(ValueError, match="Invalid volume"):
        Signal(action="enter_long", volume=-0.5)


def test_m5_trend_m1_minimum_bars_counts_rsi_length_not_fast() -> None:
    strategy = M5TrendM1EntryStrategy(
        trend_fast=1, trend_slow=3, trend_rsi_length=2, trend_bb_length=3,
        stoch_rsi_length=20, stoch_k_smooth=1, stoch_d_smooth=1, swing_lookback=5,
        use_session_filter=False,
    )
    # Stoch warm-up = trend_rsi_length(2) + stoch_rsi_length(20) + k(1) + d(1) + 2 = 26, the
    # binding constraint here. Using trend_fast(1) instead would have under-counted to 25.
    assert strategy._minimum_m1_bars == 26


def test_m5_trend_m1_h1_filter_expands_minimum_warmup() -> None:
    strategy = M5TrendM1EntryStrategy(
        trend_fast=2,
        trend_slow=4,
        trend_rsi_length=3,
        trend_bb_length=4,
        use_h1_filter=True,
        h1_fast=9,
        h1_slow=26,
        h1_slope_lookback=3,
        use_session_filter=False,
    )

    assert strategy._minimum_h1_m1_bars == 1860
    assert strategy._minimum_m1_bars == 1860


def test_m5_trend_m1_atr_spread_filter_rejects_weak_trend() -> None:
    bars = _bars([100, 101, 102, 103, 104, 103, 105, 106])
    base = _m5_m1_strategy()
    filtered = M5TrendM1EntryStrategy(
        trend_fast=2,
        trend_slow=4,
        trend_rsi_length=3,
        trend_bb_length=4,
        sideways_rsi_low=49,
        sideways_rsi_high=51,
        trend_atr_length=3,
        trend_min_spread_atr=100.0,
        use_session_filter=False,
    )

    assert base._m5_trend(bars) == "long"
    assert filtered._m5_trend(bars) == "neutral"


def test_m5_trend_m1_spread_expansion_filter_rejects_fading_trend() -> None:
    bars = _bars([100, 97, 97.5, 100.5, 97.5, 96.5, 94.5, 95, 97, 99, 97, 95])
    base = _m5_m1_strategy()
    filtered = M5TrendM1EntryStrategy(
        trend_fast=2,
        trend_slow=4,
        trend_rsi_length=3,
        trend_bb_length=4,
        sideways_rsi_low=49,
        sideways_rsi_high=51,
        trend_require_spread_expansion=True,
        trend_spread_lookback=2,
        use_session_filter=False,
    )

    assert base._m5_trend(bars) == "short"
    assert filtered._m5_trend(bars) == "neutral"


def test_m5_trend_m1_slow_slope_filter_rejects_unconfirmed_trend() -> None:
    bars = _bars([100, 99.5, 99, 98, 95, 92, 91.5, 94.5, 95.5, 96.5, 97.5, 94.5])
    base = _m5_m1_strategy()
    filtered = M5TrendM1EntryStrategy(
        trend_fast=2,
        trend_slow=4,
        trend_rsi_length=3,
        trend_bb_length=4,
        sideways_rsi_low=49,
        sideways_rsi_high=51,
        trend_require_slow_slope=True,
        trend_slope_lookback=2,
        use_session_filter=False,
    )

    assert base._m5_trend(bars) == "short"
    assert filtered._m5_trend(bars) == "neutral"


def test_m5_trend_m1_rejects_invalid_trend_strength_config() -> None:
    with pytest.raises(ValueError, match="trend_min_spread_atr"):
        M5TrendM1EntryStrategy(trend_min_spread_atr=-0.1)
    with pytest.raises(ValueError, match="trend_spread_lookback"):
        M5TrendM1EntryStrategy(trend_spread_lookback=0)
