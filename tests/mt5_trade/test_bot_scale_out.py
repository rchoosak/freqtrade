from __future__ import annotations

import pytest

from freqtrade.exceptions import OperationalException
from freqtrade.mt5_trade.bot import MT5ForexBot
from freqtrade.mt5_trade.data import MT5Bar, ReplayDataFeed
from freqtrade.mt5_trade.models import (
    BrokerPosition,
    MT5BotConfig,
    MT5OrderResult,
    MT5SymbolMapping,
)
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

    def modify_sltp(self, symbol, stop_loss, take_profit, *, position_ticket=None):
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


class NumericOrderBridge(FakeBridge):
    def submit_order(self, order):
        self.orders.append(order)
        return MT5OrderResult(accepted=True, order_id="12345")


class BrokerPositionBridge(FakeBridge):
    def __init__(self, positions) -> None:
        super().__init__()
        self._positions = positions

    def broker_positions(self):
        return self._positions

    def broker_orders(self):
        return []


class SltpTicketBridge(BrokerPositionBridge):
    def modify_sltp(self, symbol, stop_loss, take_profit, *, position_ticket=None):
        self.sltp.append((symbol, stop_loss, take_profit, position_ticket))
        return MT5OrderResult(accepted=True, order_id=None)


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


def _store_scale_out_position(
    store: MT5TradeStore,
    *,
    side: str = "buy",
    volume: float = 1.0,
    entry_price: float = 10.0,
    ticket: int | None = None,
) -> None:
    store.open_position("EURUSD", side, volume, entry_price, ticket=ticket)
    store.set_managed_position("EURUSD", side, entry_price, 12.0, 0.5, True, False)


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


def test_bot_does_not_store_market_order_id_as_position_ticket() -> None:
    bridge = NumericOrderBridge()
    store = MT5TradeStore(":memory:")
    bot = _bot(bridge, ScriptedStrategy([Signal("enter_long")]), store, default_volume=1.0)

    bot.run_once()

    assert store.open_positions()["EURUSD"].ticket is None
    assert bot._position_ids["EURUSD"][1] is None


def test_bot_exit_order_carries_position_ticket() -> None:
    bridge = FakeBridge()
    store = MT5TradeStore(":memory:")
    store.open_position("EURUSD", "buy", 1.0, 10.0, ticket=42)
    bot = _bot(bridge, ScriptedStrategy([Signal("exit")]), store, default_volume=1.0)

    bot.run_once()

    assert bridge.orders[0].position_ticket == 42


def test_bot_close_reconciles_to_learn_missing_position_ticket() -> None:
    bridge = BrokerPositionBridge([BrokerPosition("EURUSD", "buy", 1.0, price=10.0, ticket=77)])
    store = MT5TradeStore(":memory:")
    store.open_position("EURUSD", "buy", 1.0, 10.0)
    bot = _bot(bridge, ScriptedStrategy([Signal("exit")]), store, default_volume=1.0)

    bot.run_once()

    assert bridge.orders[0].position_ticket == 77


def test_bot_close_aborts_when_ticket_refresh_changes_position() -> None:
    bridge = BrokerPositionBridge([BrokerPosition("EURUSD", "sell", 0.5, price=11.0, ticket=77)])
    store = MT5TradeStore(":memory:")
    store.open_position("EURUSD", "buy", 1.0, 10.0)
    bot = _bot(bridge, ScriptedStrategy([Signal("exit")]), store, default_volume=1.0)

    bot.run_once()

    assert bridge.orders == []
    position = store.open_positions()["EURUSD"]
    assert position.side == "sell"
    assert position.volume == 0.5


def test_bot_sltp_uses_reconciled_position_ticket() -> None:
    bridge = SltpTicketBridge([BrokerPosition("EURUSD", "buy", 1.0, price=10.0, ticket=77)])
    store = MT5TradeStore(":memory:")
    signal = Signal("enter_long", stop_loss=9.0, take_profit=12.0)
    bot = _bot(bridge, ScriptedStrategy([signal]), store, default_volume=1.0)

    bot.run_once()

    assert bridge.sltp == [("EURUSD", 9.0, 12.0, 77)]


def test_bot_market_entry_with_sltp_stops_when_ticket_missing() -> None:
    bridge = BrokerPositionBridge([BrokerPosition("EURUSD", "buy", 1.0, price=10.0, ticket=None)])
    store = MT5TradeStore(":memory:")
    signal = Signal(
        "enter_long",
        stop_loss=9.0,
        tp1=12.0,
        tp1_close_fraction=0.5,
        move_sl_to_breakeven=True,
    )
    bot = _bot(bridge, ScriptedStrategy([signal]), store, default_volume=1.0)

    with pytest.raises(OperationalException, match="SL/TP ticket lookup failed"):
        bot.run_once()

    assert len(bridge.orders) == 1
    assert bridge.sltp == []
    assert store.open_positions()["EURUSD"].side == "buy"
    assert store.managed_positions() == {}
    assert "EURUSD" not in bot._managed


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


def test_bot_scale_out_order_carries_position_ticket() -> None:
    bridge = FakeBridge()
    store = MT5TradeStore(":memory:")
    bot = _bot(bridge, ScriptedStrategy([_entry_signal(), HOLD]), store, default_volume=1.0)

    bot.run_once()
    bot._position_ids["EURUSD"] = (10.0, 42)
    bot._feed.advance()
    bot.run_once()

    assert bridge.orders[1].position_ticket == 42


def test_bot_scale_out_recomputes_after_ticket_refresh_changes_volume() -> None:
    bridge = BrokerPositionBridge([BrokerPosition("EURUSD", "buy", 0.4, price=10.0, ticket=77)])
    store = MT5TradeStore(":memory:")
    _store_scale_out_position(store, volume=1.0)
    bot = _bot(bridge, ScriptedStrategy([HOLD]), store, default_volume=1.0)

    bot._feed.advance()
    bot.run_once()

    assert len(bridge.orders) == 1
    assert bridge.orders[0].volume == 0.2
    assert bridge.orders[0].position_ticket == 77
    assert store.open_positions()["EURUSD"].volume == 0.2


def test_bot_scale_out_aborts_when_ticket_refresh_changes_side() -> None:
    bridge = BrokerPositionBridge([BrokerPosition("EURUSD", "sell", 0.5, price=10.0, ticket=77)])
    store = MT5TradeStore(":memory:")
    _store_scale_out_position(store, side="buy", volume=1.0)
    bot = _bot(bridge, ScriptedStrategy([HOLD]), store, default_volume=1.0)

    bot._feed.advance()
    bot.run_once()

    assert bridge.orders == []
    assert store.managed_positions() == {}
    position = store.open_positions()["EURUSD"]
    assert position.side == "sell"
    assert position.volume == 0.5


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

    def modify_sltp(self, symbol, stop_loss, take_profit, *, position_ticket=None):
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


def test_bot_normalizes_explicit_off_grid_volume() -> None:
    bridge = FakeBridge()
    store = MT5TradeStore(":memory:")
    bot = _bot(
        bridge, ScriptedStrategy([Signal("enter_long", volume=0.025), HOLD]), store,
        default_volume=0.01, mappings=_scale_mapping(),
    )

    bot.run_once()

    # Explicit 0.025 snapped to the 0.01 grid -> 0.02 in both the order and tracked state.
    assert bridge.orders[0].volume == 0.02
    assert bot._positions["EURUSD"] == ("buy", 0.02)


class FillPriceBridge:
    """Entry fills at a configured price (the scale-out close uses no fill price)."""

    def __init__(self, entry_fill_price: float) -> None:
        self._entry_fill_price = entry_fill_price
        self.orders: list = []
        self.sltp: list = []
        self.closed = False

    def submit_order(self, order):
        self.orders.append(order)
        if order.comment == "tp1 scale-out":
            return MT5OrderResult(accepted=True, order_id="c")
        return MT5OrderResult(accepted=True, order_id="e", fill_price=self._entry_fill_price)

    def modify_sltp(self, symbol, stop_loss, take_profit, *, position_ticket=None):
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


def test_bot_scale_out_breakeven_uses_actual_fill_price() -> None:
    # Entry fills at 10.5 even though the last candle closed at 10.0.
    bridge = FillPriceBridge(entry_fill_price=10.5)
    store = MT5TradeStore(":memory:")
    bot = _bot(bridge, ScriptedStrategy([_entry_signal(), HOLD]), store, default_volume=1.0)

    bot.run_once()
    bot._feed.advance()
    bot.run_once()

    # Breakeven stop moves to the true fill (10.5), not the candle close (10.0).
    assert bridge.sltp == [("EURUSD", 10.5, None)]


def test_bot_restores_scale_out_plan_after_restart() -> None:
    store = MT5TradeStore(":memory:")
    first_bridge = FakeBridge()
    first = _bot(
        first_bridge,
        ScriptedStrategy([_entry_signal()]),
        store,
        default_volume=1.0,
    )
    first.run_once()
    assert "EURUSD" in first._managed

    restarted_bridge = FakeBridge()
    restarted = _bot(
        restarted_bridge,
        ScriptedStrategy([HOLD]),
        store,
        default_volume=1.0,
    )
    assert "EURUSD" in restarted._managed

    restarted._feed.advance()
    restarted.run_once()

    assert len(restarted_bridge.orders) == 1
    assert restarted_bridge.orders[0].comment == "tp1 scale-out"
    assert restarted_bridge.sltp == [("EURUSD", 10.0, None)]
    assert store.open_positions()["EURUSD"].volume == 0.5
    assert store.managed_positions()["EURUSD"].scaled is True
