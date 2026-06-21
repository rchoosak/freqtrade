from __future__ import annotations

from freqtrade.mt5_trade.models import MT5OrderRequest, MT5OrderResult
from freqtrade.mt5_trade.persistence import MT5TradeStore


def test_store_records_orders() -> None:
    store = MT5TradeStore(":memory:")
    order = MT5OrderRequest(symbol="EURUSD", side="buy", volume=0.01, client_order_id="c1")
    result = MT5OrderResult(accepted=True, order_id="999", retcode=10009, message="done")

    store.record_order(order, result)

    assert store.order_count() == 1


def test_store_open_and_close_position() -> None:
    store = MT5TradeStore(":memory:")

    store.open_position("EURUSD", "buy", 0.02, 1.085, ticket=7)
    positions = store.open_positions()
    assert positions["EURUSD"].side == "buy"
    assert positions["EURUSD"].volume == 0.02
    assert positions["EURUSD"].entry_price == 1.085
    assert positions["EURUSD"].ticket == 7

    # Re-opening the same symbol replaces the row rather than duplicating it.
    store.open_position("EURUSD", "sell", 0.03, 1.090, ticket=8)
    position = store.open_positions()["EURUSD"]
    assert position.side == "sell"
    assert position.ticket == 8

    store.close_position("EURUSD")
    assert store.open_positions() == {}


def test_store_persists_across_connections(tmp_path) -> None:
    db_path = str(tmp_path / "trades.sqlite")
    store = MT5TradeStore(db_path)
    store.open_position("EURUSD", "buy", 0.01, 1.10)
    store.close()

    reopened = MT5TradeStore(db_path)
    assert "EURUSD" in reopened.open_positions()


def test_store_persists_managed_position_metadata(tmp_path) -> None:
    db_path = str(tmp_path / "trades.sqlite")
    store = MT5TradeStore(db_path)
    store.open_position("EURUSD", "buy", 0.04, 1.10)
    store.set_managed_position(
        "EURUSD",
        "buy",
        entry_price=1.10,
        tp1=1.12,
        close_fraction=0.5,
        move_be=True,
        scaled=False,
    )
    store.close()

    reopened = MT5TradeStore(db_path)
    managed = reopened.managed_positions()["EURUSD"]
    assert managed.side == "buy"
    assert managed.entry_price == 1.10
    assert managed.tp1 == 1.12
    assert managed.close_fraction == 0.5
    assert managed.move_be is True
    assert managed.scaled is False


def test_store_clears_managed_position_with_position() -> None:
    store = MT5TradeStore(":memory:")
    store.open_position("EURUSD", "buy", 0.04, 1.10)
    store.set_managed_position("EURUSD", "buy", 1.10, 1.12, 0.5, True, False)

    store.close_position("EURUSD")

    assert store.open_positions() == {}
    assert store.managed_positions() == {}


def test_store_persists_strategy_position_state_with_identity(tmp_path) -> None:
    db_path = str(tmp_path / "trades.sqlite")
    store = MT5TradeStore(db_path)
    store.open_position("XAUUSD", "buy", 0.1, 2350.0, ticket=42)
    store.set_strategy_position_state(
        "XAUUSD",
        "example.Strategy",
        "buy",
        2350.0,
        42,
        {"trailing_stop": 2400.0},
    )
    store.close()

    reopened = MT5TradeStore(db_path)
    state = reopened.strategy_position_states()["XAUUSD"]
    assert state.strategy == "example.Strategy"
    assert state.side == "buy"
    assert state.entry_price == 2350.0
    assert state.ticket == 42
    assert state.state == {"trailing_stop": 2400.0}

    reopened.close_position("XAUUSD")
    assert reopened.strategy_position_states() == {}
