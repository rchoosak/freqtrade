from __future__ import annotations

from freqtrade.mt5_trade.bot import MT5ForexBot
from freqtrade.mt5_trade.data import MT5Bar, ReplayDataFeed
from freqtrade.mt5_trade.models import MT5BotConfig, MT5OrderResult, MT5SymbolMapping
from freqtrade.mt5_trade.persistence import MT5TradeStore
from freqtrade.mt5_trade.strategy import HOLD, MT5Strategy, Signal


class FakeBridge:
    def __init__(self) -> None:
        self.orders: list = []
        self.sltp: list = []
        self.closed = False

    def submit_order(self, order):
        self.orders.append(order)
        return MT5OrderResult(accepted=True, order_id="ok")

    def modify_sltp(self, symbol, stop_loss, take_profit):
        self.sltp.append((symbol, stop_loss, take_profit))
        return MT5OrderResult(accepted=True, order_id=None)

    def cancel_order(self, ticket):
        return MT5OrderResult(accepted=True, order_id=str(ticket))

    def broker_positions(self):
        return None

    def broker_orders(self):
        return None

    def close(self):
        self.closed = True


class ScriptedStrategy(MT5Strategy):
    def __init__(self, signals: list[Signal]) -> None:
        self._signals = list(signals)
        self.closed: list[str] = []

    def on_bar(self, symbol, bars):
        return self._signals.pop(0) if self._signals else HOLD

    def on_position_closed(self, symbol: str) -> None:
        self.closed.append(symbol)


def _bot(bridge, strategy, store, *, default_volume=1.0, mappings=None) -> MT5ForexBot:
    cfg = MT5BotConfig(symbols=("EURUSD",), warmup_bars=50, poll_interval=1.0)
    feed = ReplayDataFeed(
        {
            "EURUSD": [
                MT5Bar(time=0, open=10, high=10, low=10, close=10),  # entry bar
                MT5Bar(time=1, open=11, high=13, low=11, close=12),  # high 13 reaches TP1=12
            ]
        },
        warmup=1,
    )
    return MT5ForexBot(
        bridge, feed, strategy, store, cfg,
        default_volume=default_volume, symbol_mappings=mappings or {},
    )


def _entry_signal() -> Signal:
    return Signal("enter_long", tp1=12.0, tp1_close_fraction=0.5, move_sl_to_breakeven=True)


def test_bot_scales_out_half_and_moves_stop_to_breakeven() -> None:
    bridge = FakeBridge()
    store = MT5TradeStore(":memory:")
    bot = _bot(bridge, ScriptedStrategy([_entry_signal(), HOLD]), store, default_volume=1.0)

    bot.run_once()           # opens the full position (no managed trigger yet)
    bot._feed.advance()      # reveal the bar whose high reaches TP1
    bot.run_once()           # TP1 reached -> close half + move stop to breakeven

    # Entry order + a half-size opposite close order.
    assert len(bridge.orders) == 2
    assert bridge.orders[1].side == "sell"
    assert bridge.orders[1].volume == 0.5
    assert bridge.orders[1].comment == "tp1 scale-out"
    # Stop moved to entry (10) on the remainder.
    assert bridge.sltp == [("EURUSD", 10.0, None)]
    # Remaining runner is half the position; managed slot marked scaled.
    assert bot._positions["EURUSD"] == ("buy", 0.5)
    assert bot._managed["EURUSD"].scaled is True
    # Partial scale-out is not a full close -> strategy not notified.
    assert store.open_positions()["EURUSD"].volume == 0.5


def test_bot_scale_out_skipped_when_position_too_small_to_split() -> None:
    bridge = FakeBridge()
    store = MT5TradeStore(":memory:")
    mappings = {
        "EURUSD": MT5SymbolMapping(
            base="EUR", quote="USD", mt5_symbol="EURUSD", min_lot=0.01, lot_step=0.01
        )
    }
    bot = _bot(
        bridge, ScriptedStrategy([_entry_signal(), HOLD]), store,
        default_volume=0.01, mappings=mappings,
    )

    bot.run_once()
    bot._feed.advance()
    bot.run_once()

    # 50% of 0.01 = 0.005 < min_lot -> no partial order, position kept whole.
    assert len(bridge.orders) == 1
    assert bot._positions["EURUSD"] == ("buy", 0.01)
    assert bot._managed["EURUSD"].scaled is True


def test_bot_runner_full_close_notifies_strategy() -> None:
    bridge = FakeBridge()
    store = MT5TradeStore(":memory:")
    strategy = ScriptedStrategy([_entry_signal(), HOLD, Signal("exit")])
    bot = _bot(bridge, strategy, store, default_volume=1.0)

    bot.run_once()           # open
    bot._feed.advance()
    bot.run_once()           # scale out half (partial -> no notify)
    bot.run_once()           # exit signal closes the runner -> notify

    assert strategy.closed == ["EURUSD"]
    assert "EURUSD" not in bot._managed


def test_bot_scale_out_snaps_off_grid_close_volume() -> None:
    bridge = FakeBridge()
    store = MT5TradeStore(":memory:")
    mappings = {
        "EURUSD": MT5SymbolMapping(
            base="EUR", quote="USD", mt5_symbol="EURUSD", min_lot=0.01, lot_step=0.01
        )
    }
    bot = _bot(
        bridge, ScriptedStrategy([_entry_signal(), HOLD]), store,
        default_volume=0.03, mappings=mappings,
    )

    bot.run_once()
    bot._feed.advance()
    bot.run_once()

    # 0.03 * 0.5 = 0.015 is off-grid -> snapped to a 0.01 close, leaving a 0.02 runner.
    assert bridge.orders[1].volume == 0.01
    assert bot._positions["EURUSD"] == ("buy", 0.02)


class PartialFillBridge:
    """Fills the TP1 scale-out close only partially (broker-reported filled_volume)."""

    def __init__(self, tp1_filled: float) -> None:
        self._tp1_filled = tp1_filled
        self.orders: list = []
        self.sltp: list = []
        self.closed = False

    def submit_order(self, order):
        self.orders.append(order)
        if order.comment == "tp1 scale-out":
            return MT5OrderResult(accepted=True, order_id="ok", filled_volume=self._tp1_filled)
        return MT5OrderResult(accepted=True, order_id="ok")

    def modify_sltp(self, symbol, stop_loss, take_profit):
        self.sltp.append((symbol, stop_loss, take_profit))
        return MT5OrderResult(accepted=True, order_id=None)

    def cancel_order(self, ticket):
        return MT5OrderResult(accepted=True, order_id=str(ticket))

    def broker_positions(self):
        return None

    def broker_orders(self):
        return None

    def close(self):
        self.closed = True


def _scale_mapping():
    return {
        "EURUSD": MT5SymbolMapping(
            base="EUR", quote="USD", mt5_symbol="EURUSD", min_lot=0.01, lot_step=0.01
        )
    }


def test_bot_scale_out_tracks_actual_partial_fill() -> None:
    import pytest

    bridge = PartialFillBridge(tp1_filled=0.01)  # requested 0.02, only 0.01 filled
    store = MT5TradeStore(":memory:")
    bot = _bot(
        bridge, ScriptedStrategy([_entry_signal(), HOLD]), store,
        default_volume=0.04, mappings=_scale_mapping(),
    )

    bot.run_once()
    bot._feed.advance()
    bot.run_once()

    # Order requested 0.02, but only 0.01 filled -> runner reflects the true remaining 0.03.
    assert bridge.orders[1].volume == 0.02
    assert bot._positions["EURUSD"][1] == pytest.approx(0.03)
    assert bot._managed["EURUSD"].volume == pytest.approx(0.03)


def test_bot_scale_out_retries_when_nothing_filled() -> None:
    bridge = PartialFillBridge(tp1_filled=0.0)  # accepted but filled nothing
    store = MT5TradeStore(":memory:")
    bot = _bot(
        bridge, ScriptedStrategy([_entry_signal(), HOLD]), store,
        default_volume=0.04, mappings=_scale_mapping(),
    )

    bot.run_once()
    bot._feed.advance()
    bot.run_once()

    # Nothing filled -> position untouched, not marked scaled, no breakeven move yet.
    assert bot._positions["EURUSD"] == ("buy", 0.04)
    assert bot._managed["EURUSD"].scaled is False
    assert bridge.sltp == []
