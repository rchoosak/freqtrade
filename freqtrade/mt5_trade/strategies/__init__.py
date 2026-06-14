from __future__ import annotations

from freqtrade.mt5_trade.strategies.base import (
    HOLD,
    MT5Strategy,
    Signal,
    SignalAction,
)
from freqtrade.mt5_trade.strategies.m5_trend_m1_entry import M5TrendM1EntryStrategy
from freqtrade.mt5_trade.strategies.sma_cross import SmaCrossStrategy


__all__ = [
    "HOLD",
    "M5TrendM1EntryStrategy",
    "MT5Strategy",
    "Signal",
    "SignalAction",
    "SmaCrossStrategy",
]
