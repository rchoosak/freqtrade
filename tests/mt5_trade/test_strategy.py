from __future__ import annotations

import pytest

from freqtrade.mt5_trade.data import MT5Bar
from freqtrade.mt5_trade.strategy import M5TrendM1EntryStrategy, SmaCrossStrategy


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
