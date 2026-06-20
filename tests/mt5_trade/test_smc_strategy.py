from __future__ import annotations

import pytest

from freqtrade.exceptions import OperationalException
from freqtrade.mt5_trade.bot import MT5ForexBot
from freqtrade.mt5_trade.data import MT5Bar, ReplayDataFeed
from freqtrade.mt5_trade.models import MT5BotConfig, MT5OrderResult
from freqtrade.mt5_trade.persistence import MT5TradeStore
from freqtrade.mt5_trade.position import OrderIntent
from freqtrade.mt5_trade.strategies import (
    HOLD,
    MT5Strategy,
    Signal,
    SmcOrderBlockStrategy,
    validate_strategy_runtime,
)
from freqtrade.mt5_trade.strategies.indicators import _atr_series


# (open, high, low, close) for a bullish setup: prior swing high at idx6, bearish order block at
# idx8 (zone 100.5-101.8), an up-impulse at idx9 that breaks the swing high, a retrace, then a
# liquidity sweep below the OB low with a bullish-engulfing close at the final (trigger) candle.
_BULLISH_ROWS = [
    (100.0, 100.0, 100.0, 100.0),  # 0 flat filler
    (100.0, 100.0, 100.0, 100.0),  # 1
    (100.0, 100.0, 100.0, 100.0),  # 2
    (100.0, 100.0, 100.0, 100.0),  # 3
    (100.0, 100.0, 100.0, 100.0),  # 4 (lookback start)
    (100.0, 101.0, 99.0, 100.5),  # 5 swing low (99)
    (100.5, 103.0, 100.0, 102.0),  # 6 swing high (103)
    (102.0, 102.5, 101.0, 101.5),  # 7 pullback
    (101.5, 101.8, 100.5, 100.8),  # 8 bearish order block, zone [100.5, 101.8]
    (101.0, 104.0, 101.0, 103.5),  # 9 impulse: close 103.5 breaks swing high 103
    (103.0, 103.2, 102.0, 102.2),  # 10 retrace
    (102.2, 102.3, 101.2, 101.4),  # 11
    (101.4, 101.5, 100.9, 101.0),  # 12
    (101.0, 101.1, 100.7, 100.9),  # 13
    (100.9, 101.0, 100.6, 100.7),  # 14 prior candle (bearish) for the engulfing
    (100.6, 101.6, 100.2, 101.2),  # 15 trigger: sweep < 100.5 then bullish engulfing close
]

_SMALL = dict(
    structure_lookback=12,
    swing_strength=1,
    atr_length=2,
    volume_length=2,
    min_atr=0.0,
    volume_factor=0.0,
    use_session_filter=False,
)


def _bars(rows: list[tuple[float, float, float, float]]) -> list[MT5Bar]:
    return [
        MT5Bar(time=index * 60, open=o, high=h, low=low, close=c, volume=100.0)
        for index, (o, h, low, c) in enumerate(rows)
    ]


def _mirror(rows: list[tuple[float, float, float, float]], pivot: float = 100.0) -> list:
    # Reflect prices around `pivot` so a bullish setup becomes the exact bearish mirror image
    # (high<->low swap keeps OHLC consistent).
    return [(2 * pivot - o, 2 * pivot - low, 2 * pivot - h, 2 * pivot - c) for o, h, low, c in rows]


def test_bullish_order_block_sweep_triggers_long() -> None:
    strategy = SmcOrderBlockStrategy(**_SMALL)
    signal = strategy.on_bar("XAUUSD", _bars(_BULLISH_ROWS))

    assert signal.action == "enter_long"
    # SL at the sweep candle low (100.2) minus the 10-point buffer (10 * 0.01).
    assert signal.stop_loss == pytest.approx(100.1)
    # TP at risk_reward (2.0) x risk; risk = 101.2 - 100.1 = 1.1 -> TP = 101.2 + 2.2.
    assert signal.take_profit == pytest.approx(103.4)
    # Breakeven scale-out: TP1 at +200 points (2.0), partial close, SL -> entry.
    assert signal.tp1 == pytest.approx(103.2)
    assert signal.tp1_close_fraction == pytest.approx(0.5)
    assert signal.move_sl_to_breakeven is True


def test_bearish_order_block_sweep_triggers_short() -> None:
    strategy = SmcOrderBlockStrategy(**_SMALL)
    signal = strategy.on_bar("XAUUSD", _bars(_mirror(_BULLISH_ROWS)))

    assert signal.action == "enter_short"
    assert signal.stop_loss == pytest.approx(99.9)
    assert signal.take_profit == pytest.approx(96.6)
    assert signal.tp1 == pytest.approx(96.8)
    assert signal.move_sl_to_breakeven is True


def test_no_sweep_holds() -> None:
    rows = list(_BULLISH_ROWS)
    # Trigger candle stays inside the zone (low 100.6 never pierces the 100.5 OB low) -> no sweep.
    rows[-1] = (100.6, 101.6, 100.6, 101.2)
    strategy = SmcOrderBlockStrategy(**_SMALL)

    assert strategy.on_bar("XAUUSD", _bars(rows)) is HOLD


def test_insufficient_bars_holds() -> None:
    strategy = SmcOrderBlockStrategy(**_SMALL)
    # One short of the warmup minimum (16) -> HOLD.
    assert strategy.required_timeframe == "M1"
    assert strategy.minimum_bars == 16
    assert strategy.on_bar("XAUUSD", _bars(_BULLISH_ROWS)[:15]) is HOLD


def test_runtime_rejects_non_m1_timeframe() -> None:
    strategy = SmcOrderBlockStrategy(**_SMALL)

    with pytest.raises(OperationalException, match="requires timeframe M1"):
        validate_strategy_runtime(
            strategy,
            timeframe="M5",
            warmup_bars=strategy.minimum_bars,
        )


def test_session_filter_blocks_out_of_window() -> None:
    # Bars sit near the 1970 epoch (00:xx UTC), outside the default 12:00-21:00 window.
    strategy = SmcOrderBlockStrategy(
        structure_lookback=12,
        swing_strength=1,
        atr_length=2,
        volume_length=2,
        use_session_filter=True,
    )
    assert strategy.on_bar("XAUUSD", _bars(_BULLISH_ROWS)) is HOLD


def test_volatility_floor_blocks_low_atr() -> None:
    strategy = SmcOrderBlockStrategy(
        structure_lookback=12,
        swing_strength=1,
        atr_length=2,
        volume_length=2,
        min_atr=1e9,
        volume_factor=0.0,
        use_session_filter=False,
    )
    assert strategy.on_bar("XAUUSD", _bars(_BULLISH_ROWS)) is HOLD


@pytest.mark.parametrize(
    "kwargs",
    [
        {"risk_reward": 0.0},
        {"point": 0.0},
        {"tp1_close_fraction": 1.0},
        {"swing_strength": 0},
        {"pinbar_wick_ratio": 0.0},
    ],
)
def test_invalid_parameters_raise(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        SmcOrderBlockStrategy(**kwargs)


def test_atr_series_matches_manual_wilder() -> None:
    bars = [
        MT5Bar(time=0, open=8.0, high=10.0, low=8.0, close=9.0, volume=1.0),
        MT5Bar(time=60, open=9.0, high=12.0, low=9.0, close=11.0, volume=1.0),  # TR=3
        MT5Bar(time=120, open=11.0, high=13.0, low=11.0, close=12.0, volume=1.0),  # TR=2
        MT5Bar(time=180, open=12.0, high=14.0, low=12.0, close=13.0, volume=1.0),  # TR=2
    ]
    series = _atr_series(bars, length=2)
    assert series[0] is None
    assert series[1] is None
    assert series[2] == pytest.approx(2.5)  # mean(3, 2)
    assert series[3] == pytest.approx(2.25)  # (2.5 * 1 + 2) / 2


# --- live-only spread gate (bot level) -----------------------------------------------------


class _NoStrategy(MT5Strategy):
    def on_bar(self, symbol, bars):
        return HOLD


class _SpreadBridge:
    def __init__(self, spread: int | None) -> None:
        self._spread = spread
        self.orders: list = []

    def submit_order(self, order):
        self.orders.append(order)
        return MT5OrderResult(
            accepted=True, order_id="ok", fill_price=1.10, filled_volume=order.volume
        )

    def modify_sltp(self, symbol, stop_loss, take_profit, *, position_ticket=None):
        return MT5OrderResult(accepted=True, order_id=None)

    def current_spread_points(self, symbol):
        return self._spread

    def broker_positions(self):
        return None

    def broker_orders(self):
        return None

    def close(self) -> None:
        pass


def _spread_bot(bridge: _SpreadBridge, *, max_spread_points: int) -> MT5ForexBot:
    store = MT5TradeStore(":memory:")
    feed = ReplayDataFeed({"EURUSD": [MT5Bar(0, 1, 1, 1, 1)]}, warmup=1)
    cfg = MT5BotConfig(
        symbols=("EURUSD",),
        warmup_bars=1,
        poll_interval=1.0,
        max_spread_points=max_spread_points,
    )
    return MT5ForexBot(bridge, feed, _NoStrategy(), store, cfg, default_volume=0.01)


def _open_intent() -> OrderIntent:
    return OrderIntent(side="buy", volume=0.01, result=("buy", 0.01), reason="open")


def test_spread_gate_blocks_entry_when_too_wide() -> None:
    bridge = _SpreadBridge(spread=30)
    bot = _spread_bot(bridge, max_spread_points=20)

    proceeded = bot._execute("EURUSD", _open_intent(), Signal(action="enter_long"), 1.10)

    assert proceeded is False
    assert bridge.orders == []


def test_spread_gate_allows_entry_within_limit() -> None:
    bridge = _SpreadBridge(spread=10)
    bot = _spread_bot(bridge, max_spread_points=20)

    bot._execute("EURUSD", _open_intent(), Signal(action="enter_long"), 1.10)

    assert len(bridge.orders) == 1


def test_spread_gate_disabled_by_default() -> None:
    bridge = _SpreadBridge(spread=999)
    bot = _spread_bot(bridge, max_spread_points=0)

    bot._execute("EURUSD", _open_intent(), Signal(action="enter_long"), 1.10)

    assert len(bridge.orders) == 1


# --- HTF bias / confluence -----------------------------------------------------------------


def _trend_bars(start: float, step: float, n: int) -> list[MT5Bar]:
    bars: list[MT5Bar] = []
    price = start
    for i in range(n):
        o, c = price, price + step
        bars.append(
            MT5Bar(
                time=i * 60,
                open=o,
                high=max(o, c) + 0.1,
                low=min(o, c) - 0.1,
                close=c,
                volume=100.0,
            )
        )
        price = c
    return bars


def _htf_strategy(**overrides) -> SmcOrderBlockStrategy:
    params = dict(
        structure_lookback=12,
        swing_strength=1,
        atr_length=2,
        volume_length=2,
        min_atr=0.0,
        volume_factor=0.0,
        use_session_filter=False,
        use_htf_filter=True,
        htf_minutes=2,
        htf_ema_fast=2,
        htf_ema_slow=3,
    )
    params.update(overrides)
    return SmcOrderBlockStrategy(**params)


def test_htf_bias_detects_uptrend() -> None:
    assert _htf_strategy()._htf_bias(_trend_bars(100.0, 0.5, 16)) == "up"


def test_htf_bias_detects_downtrend() -> None:
    assert _htf_strategy()._htf_bias(_trend_bars(120.0, -0.5, 16)) == "down"


def test_htf_filter_gates_counter_trend_long() -> None:
    bars = _bars(_BULLISH_ROWS)
    # The same bullish OB setup fires when the HTF filter is off...
    assert _htf_strategy(use_htf_filter=False).on_bar("XAUUSD", bars).action == "enter_long"
    # ...but is skipped when the HTF bias is not up (the M1 sweep entry follows a pullback).
    gated = _htf_strategy()
    assert gated._htf_bias(bars) != "up"
    assert gated.on_bar("XAUUSD", bars) is HOLD


def test_htf_invalid_ema_order_raises() -> None:
    with pytest.raises(ValueError):
        SmcOrderBlockStrategy(htf_ema_fast=50, htf_ema_slow=20)


def test_htf_invert_flips_required_bias() -> None:
    up_bars = _trend_bars(100.0, 0.5, 16)  # clear up bias
    trend = _htf_strategy()  # follow HTF trend
    invert = _htf_strategy(htf_invert=True)  # fade HTF trend

    # Trend-following: up bias allows longs, blocks shorts.
    assert trend._htf_allows("long", up_bars) is True
    assert trend._htf_allows("short", up_bars) is False
    # Inverted: up bias blocks longs, allows shorts (fade).
    assert invert._htf_allows("long", up_bars) is False
    assert invert._htf_allows("short", up_bars) is True
