from __future__ import annotations

from datetime import UTC, datetime

import pytest

from freqtrade.mt5_trade.backtest import run_backtest
from freqtrade.mt5_trade.bot import MT5ForexBot
from freqtrade.mt5_trade.data import MT5Bar, ReplayDataFeed
from freqtrade.mt5_trade.execution import MT5ExecutionBridge
from freqtrade.mt5_trade.models import (
    MT5BotConfig,
    MT5BridgeConfig,
    MT5SymbolMapping,
)
from freqtrade.mt5_trade.persistence import MT5TradeStore
from freqtrade.mt5_trade.sizing import PositionSizer
from freqtrade.mt5_trade.strategies import XauusdD1H4TrendStrategy
from freqtrade.mt5_trade.strategies.indicators import (
    _adx_series,
    _aggregate_completed_bars,
)


def _strategy(**overrides) -> XauusdD1H4TrendStrategy:
    params = {
        "d1_fast": 2,
        "d1_slow": 3,
        "d1_slope_lookback": 1,
        "d1_adx_length": 2,
        "d1_long_adx_min": 10.0,
        "d1_short_adx_min": 10.0,
        "d1_exit_ema": 2,
        "h4_fast": 2,
        "h4_slow": 3,
        "breakout_lookback": 3,
        "atr_length": 2,
        "initial_stop_atr": 2.5,
        "max_breakout_atr": 2.5,
        "chandelier_lookback": 3,
        "chandelier_atr": 2.0,
    }
    params.update(overrides)
    return XauusdD1H4TrendStrategy(**params)


def _h1_trend(step: float, hours: int = 7 * 24) -> list[MT5Bar]:
    bars: list[MT5Bar] = []
    price = 100.0
    for index in range(hours):
        open_price = price
        close = price + step
        bars.append(
            MT5Bar(
                time=index * 3600,
                open=open_price,
                high=max(open_price, close) + 0.1,
                low=min(open_price, close) - 0.1,
                close=close,
                volume=100.0,
            )
        )
        price = close
    return bars


def _append_h1(bars: list[MT5Bar], steps: list[float]) -> list[MT5Bar]:
    result = list(bars)
    price = result[-1].close
    for step in steps:
        open_price = price
        close = price + step
        result.append(
            MT5Bar(
                time=len(result) * 3600,
                open=open_price,
                high=max(open_price, close) + 0.1,
                low=min(open_price, close) - 0.1,
                close=close,
                volume=100.0,
            )
        )
        price = close
    return result


def test_contract_requires_h1_and_daily_warmup() -> None:
    strategy = XauusdD1H4TrendStrategy()

    assert strategy.required_timeframe == "H1"
    assert strategy.minimum_bars == 3720


def test_enters_long_on_d1_trend_and_h4_breakout() -> None:
    strategy = _strategy()

    signal = strategy.on_bar("XAUUSD", _h1_trend(0.5))

    assert signal.action == "enter_long"
    assert signal.stop_loss is not None
    assert signal.stop_loss < 184.0
    assert signal.take_profit is None


def test_enters_short_on_d1_downtrend_and_h4_breakout() -> None:
    strategy = _strategy()

    signal = strategy.on_bar("XAUUSD", _h1_trend(-0.5))

    assert signal.action == "enter_short"
    assert signal.stop_loss is not None
    assert signal.stop_loss > 16.0


def test_allow_short_false_blocks_short_entry() -> None:
    strategy = _strategy(allow_short=False)

    assert strategy.on_bar("XAUUSD", _h1_trend(-0.5)).action == "hold"


def test_processes_each_completed_h4_candle_once() -> None:
    strategy = _strategy()
    bars = _h1_trend(0.5)

    assert strategy.on_bar("XAUUSD", bars).action == "enter_long"
    assert strategy.on_bar("XAUUSD", bars).action == "hold"


def test_chandelier_stop_ratchets_and_exits_on_reversal() -> None:
    strategy = _strategy()
    bars = _h1_trend(0.5)
    assert strategy.on_bar("XAUUSD", bars).action == "enter_long"
    strategy.on_position_state("XAUUSD", "buy")

    bars = _append_h1(bars, [0.5] * 4)
    assert strategy.on_bar("XAUUSD", bars).action == "hold"
    ratcheted_stop = strategy._trailing_stops["XAUUSD"]

    bars = _append_h1(bars, [-3.0] * 4)
    signal = strategy.on_bar("XAUUSD", bars)

    assert signal.action == "exit"
    assert signal.comment == "h4 chandelier long exit"
    assert strategy._trailing_stops["XAUUSD"] == ratcheted_stop


def test_restored_chandelier_stop_cannot_loosen() -> None:
    strategy = _strategy()
    strategy.restore_position_state("XAUUSD", "buy", {"trailing_stop": 118.0})
    h4_bars = [
        MT5Bar(time=index * 14400, open=115, high=120, low=110, close=115)
        for index in range(3)
    ]
    d1_bars = [
        MT5Bar(time=index * 86400, open=100, high=121, low=99, close=100 + index * 10)
        for index in range(3)
    ]

    signal = strategy._exit_signal("XAUUSD", "buy", h4_bars, d1_bars)

    assert signal is not None
    assert signal.action == "exit"
    assert strategy._trailing_stops["XAUUSD"] == 118.0


def test_d1_exit_ema_closes_long_before_chandelier() -> None:
    strategy = _strategy()
    h4_bars = _aggregate_completed_bars(_h1_trend(0.5), 4 * 3600, 3600)
    d1_bars = [
        MT5Bar(time=0, open=100, high=101, low=99, close=100),
        MT5Bar(time=86400, open=100, high=102, low=99, close=101),
        MT5Bar(time=172800, open=101, high=101, low=79, close=80),
    ]

    signal = strategy._exit_signal("XAUUSD", "buy", h4_bars, d1_bars)

    assert signal is not None
    assert signal.action == "exit"
    assert signal.comment == "d1 close below exit ema"


def test_oversized_breakout_is_rejected() -> None:
    strategy = _strategy()
    h4_bars = _aggregate_completed_bars(_h1_trend(0.5), 4 * 3600, 3600)
    last = h4_bars[-1]
    h4_bars[-1] = MT5Bar(
        time=last.time,
        open=last.open,
        high=last.high + 10,
        low=last.low - 10,
        close=last.close,
        volume=last.volume,
    )

    assert strategy._h4_breakout(h4_bars, "buy") is False


def test_gap_breakout_is_rejected_by_true_range() -> None:
    strategy = _strategy()
    h4_bars = [
        MT5Bar(time=index * 14400, open=99, high=101, low=99, close=100)
        for index in range(4)
    ]
    h4_bars.append(
        MT5Bar(time=4 * 14400, open=150, high=152, low=149, close=151)
    )

    assert strategy._h4_breakout(h4_bars, "buy") is False


def test_breakout_close_too_far_beyond_channel_is_rejected() -> None:
    strategy = _strategy(max_channel_breakout_atr=0.25)
    h4_bars = [
        MT5Bar(time=index * 14400, open=99, high=101, low=99, close=100 + index * 0.1)
        for index in range(4)
    ]
    h4_bars.append(
        MT5Bar(time=4 * 14400, open=101, high=102, low=100.5, close=102)
    )

    assert strategy._h4_breakout(h4_bars, "buy") is False


def test_adx_reaches_100_in_one_directional_market() -> None:
    d1_bars = _aggregate_completed_bars(_h1_trend(0.5), 24 * 3600, 3600)

    assert _adx_series(d1_bars, 2)[-1] == pytest.approx(100.0)


def test_aggregation_excludes_partial_latest_bucket() -> None:
    bars = _h1_trend(0.5, hours=26)

    aggregated = _aggregate_completed_bars(bars, 24 * 3600, 3600)

    assert len(aggregated) == 1
    assert aggregated[0].time == 0


def test_aggregation_keeps_older_bucket_with_scheduled_pause() -> None:
    strategy = _strategy()
    bars = [
        MT5Bar(time=hour * 3600, open=100, high=101, low=99, close=100, volume=100)
        for hour in [*range(21), 23]
    ]
    next_day = bars[-1].close
    bars.append(
        MT5Bar(
            time=24 * 3600,
            open=next_day,
            high=next_day + 0.6,
            low=next_day - 0.1,
            close=next_day + 0.5,
        )
    )

    aggregated = _aggregate_completed_bars(
        bars,
        24 * 3600,
        3600,
        expected_slot=strategy._is_expected_h1_slot,
    )

    assert len(aggregated) == 1
    assert aggregated[0].time == 0
    assert aggregated[0].volume == 2200


def test_aggregation_rejects_unexpected_missing_h1_slot() -> None:
    strategy = _strategy()
    bars = [
        MT5Bar(time=hour * 3600, open=100, high=101, low=99, close=100)
        for hour in [*range(21), 23]
        if hour != 10
    ]
    bars.extend(
        MT5Bar(time=(24 + hour) * 3600, open=100, high=101, low=99, close=100)
        for hour in range(21)
    )

    aggregated = _aggregate_completed_bars(
        bars,
        24 * 3600,
        3600,
        expected_slot=strategy._is_expected_h1_slot,
    )

    assert all(bar.time != 0 for bar in aggregated)


def test_sunday_fragment_is_merged_into_completed_monday() -> None:
    strategy = _strategy()
    sunday = int(datetime(2025, 1, 5, tzinfo=UTC).timestamp())
    monday = sunday + 86400
    bars = [
        MT5Bar(time=sunday + 23 * 3600, open=99, high=101, low=98, close=100, volume=1)
    ]
    bars.extend(
        MT5Bar(
            time=monday + hour * 3600,
            open=100,
            high=102,
            low=99,
            close=101,
            volume=1,
        )
        for hour in [*range(21), 23]
    )

    aggregated = _aggregate_completed_bars(
        bars,
        24 * 3600,
        3600,
        expected_slot=strategy._is_expected_h1_slot,
        merge_sunday_into_monday=True,
    )

    assert len(aggregated) == 1
    assert aggregated[0].time == monday
    assert aggregated[0].open == 99
    assert aggregated[0].close == 101
    assert aggregated[0].volume == 23


def test_bot_restores_chandelier_stop_only_for_same_position_identity() -> None:
    strategy = _strategy()
    bars = _append_h1(_h1_trend(0.5), [0.5] * 4)
    store = MT5TradeStore(":memory:")
    store.open_position("XAUUSD", "buy", 0.1, 100.0, ticket=42)
    config = MT5BridgeConfig(
        symbols=(MT5SymbolMapping(base="XAU", quote="USD", mt5_symbol="XAUUSD"),),
        dry_run=True,
    )
    bot_config = MT5BotConfig(
        symbols=("XAUUSD",),
        timeframe="H1",
        warmup_bars=len(bars),
        poll_interval=1,
    )
    bot = MT5ForexBot(
        MT5ExecutionBridge(config),
        ReplayDataFeed({"XAUUSD": bars}, warmup=len(bars)),
        strategy,
        store,
        bot_config,
    )

    bot.run_once()

    persisted = store.strategy_position_states()["XAUUSD"]
    ratcheted_stop = persisted.state["trailing_stop"]
    restarted_strategy = _strategy()
    MT5ForexBot(
        MT5ExecutionBridge(config),
        ReplayDataFeed({"XAUUSD": bars}, warmup=len(bars)),
        restarted_strategy,
        store,
        bot_config,
    )
    assert restarted_strategy._trailing_stops["XAUUSD"] == ratcheted_stop

    store.open_position("XAUUSD", "buy", 0.1, 100.0, ticket=99)
    replacement_strategy = _strategy()
    MT5ForexBot(
        MT5ExecutionBridge(config),
        ReplayDataFeed({"XAUUSD": bars}, warmup=len(bars)),
        replacement_strategy,
        store,
        bot_config,
    )
    assert "XAUUSD" not in replacement_strategy._trailing_stops
    assert store.strategy_position_states() == {}


def test_backtest_runs_long_horizon_strategy_end_to_end() -> None:
    strategy = _strategy()

    result = run_backtest(
        strategy,
        {"XAUUSD": _h1_trend(0.5)},
        default_volume=0.01,
        warmup_bars=strategy.minimum_bars,
    )

    assert result.num_trades == 1
    assert result.trades[0].side == "buy"
    assert result.trades[0].pnl > 0


def test_backtest_calculates_lot_from_portfolio_risk() -> None:
    strategy = _strategy()
    sizer = PositionSizer.from_config(
        {
            "position_sizing": {
                "mode": "risk_percent",
                "risk_per_trade": 0.75,
                "capital_fraction": 0.9,
                "max_risk_amount": 750,
                "max_lot": 0.5,
            }
        },
        default_lot_size=0.01,
        contract_size=100,
    )

    result = run_backtest(
        strategy,
        {"XAUUSD": _h1_trend(0.5)},
        warmup_bars=strategy.minimum_bars,
        starting_balance=100000,
        contract_size=100,
        position_sizer=sizer,
        symbol_mappings={
            "XAUUSD": MT5SymbolMapping(
                base="XAU",
                quote="USD",
                mt5_symbol="XAUUSD",
                price_precision=2,
                lot_precision=2,
                min_lot=0.01,
                lot_step=0.01,
            )
        },
    )

    assert result.trades[0].volume == 0.5
    assert result.trades[0].volume != 0.01


@pytest.mark.parametrize(
    "kwargs",
    [
        {"d1_fast": 3, "d1_slow": 2},
        {"h4_fast": 3, "h4_slow": 2},
        {"initial_stop_atr": 0},
        {"max_breakout_atr": 0},
        {"max_channel_breakout_atr": 0},
        {"chandelier_atr": 0},
        {"daily_anchor_hour": 24},
        {"session_break_hours": [21, 24]},
        {"sunday_open_hour": 24},
        {"friday_close_hour": 25},
    ],
)
def test_invalid_parameters_raise(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        XauusdD1H4TrendStrategy(**kwargs)
