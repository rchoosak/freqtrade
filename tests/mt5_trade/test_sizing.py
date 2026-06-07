from __future__ import annotations

from freqtrade.mt5_trade.models import MT5SymbolMapping
from freqtrade.mt5_trade.sizing import PositionSizer


def test_fixed_position_sizer_uses_configured_lot_size() -> None:
    sizer = PositionSizer.from_config(
        {"position_sizing": {"mode": "fixed", "lot_size": 0.03}},
        default_lot_size=0.01,
    )

    decision = sizer.size_entry(
        symbol="EURUSD",
        side="buy",
        entry_price=1.1,
        stop_loss=None,
        balance=None,
    )

    assert decision.volume == 0.03


def test_risk_percent_position_sizer_floors_to_lot_step() -> None:
    sizer = PositionSizer.from_config(
        {"position_sizing": {"mode": "risk_percent", "risk_per_trade": 1.0}},
        default_lot_size=0.01,
        contract_size=100,
    )
    mapping = MT5SymbolMapping(
        base="XAU",
        quote="USD",
        mt5_symbol="XAUUSD",
        min_lot=0.01,
        lot_step=0.01,
    )

    decision = sizer.size_entry(
        symbol="XAUUSD",
        side="buy",
        entry_price=4500,
        stop_loss=4490,
        balance=2500,
        mapping=mapping,
    )

    assert decision.volume == 0.02


def test_risk_percent_position_sizer_skips_when_min_lot_exceeds_risk() -> None:
    sizer = PositionSizer.from_config(
        {"position_sizing": {"mode": "risk_percent", "risk_per_trade": 0.5}},
        default_lot_size=0.01,
        contract_size=100,
    )

    decision = sizer.size_entry(
        symbol="XAUUSD",
        side="buy",
        entry_price=4500,
        stop_loss=4480,
        balance=1000,
    )

    assert decision.skipped is True
    assert "above target" in str(decision.reason)
