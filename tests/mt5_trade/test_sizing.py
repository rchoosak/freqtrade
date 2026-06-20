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


def test_risk_percent_uses_only_configured_capital_fraction() -> None:
    sizer = PositionSizer.from_config(
        {
            "position_sizing": {
                "mode": "risk_percent",
                "risk_per_trade": 0.25,
                "capital_fraction": 0.8,
            }
        },
        default_lot_size=0.01,
        contract_size=100,
    )

    decision = sizer.size_entry(
        symbol="XAUUSD",
        side="buy",
        entry_price=4500,
        stop_loss=4490,
        balance=100000,
    )

    # 100,000 * 80% * 0.25% = 200 risk; 200 / (10 * 100) = 0.20 lot.
    assert decision.volume == 0.2


def test_risk_percent_honors_cash_risk_and_lot_caps() -> None:
    sizer = PositionSizer.from_config(
        {
            "position_sizing": {
                "mode": "risk_percent",
                "risk_per_trade": 1.0,
                "max_risk_amount": 150,
                "max_lot": 0.1,
            }
        },
        default_lot_size=0.01,
        contract_size=100,
    )

    decision = sizer.size_entry(
        symbol="XAUUSD",
        side="buy",
        entry_price=4500,
        stop_loss=4490,
        balance=100000,
    )

    # Cash cap would produce 0.15 lot, then the hard lot cap reduces it to 0.10.
    assert decision.volume == 0.1


def test_position_sizer_rejects_invalid_capital_guards() -> None:
    for config in (
        {"capital_fraction": 0},
        {"capital_fraction": 1.1},
        {"max_risk_amount": 0},
    ):
        try:
            PositionSizer.from_config(
                {"position_sizing": {"mode": "risk_percent", **config}},
                default_lot_size=0.01,
            )
        except ValueError:
            continue
        raise AssertionError(f"Expected invalid position sizing config to fail: {config}")
