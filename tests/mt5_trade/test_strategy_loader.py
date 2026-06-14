from __future__ import annotations

import pytest

from freqtrade.exceptions import OperationalException
from freqtrade.mt5_trade.runner import build_default_strategy
from freqtrade.mt5_trade.strategies import MT5Strategy, SmaCrossStrategy


def test_build_strategy_defaults_to_sma_without_class() -> None:
    strategy = build_default_strategy({"strategy": {"fast": 3, "slow": 8}})

    assert isinstance(strategy, SmaCrossStrategy)
    assert strategy.fast == 3
    assert strategy.slow == 8


def test_build_strategy_loads_class_from_dotted_path() -> None:
    strategy = build_default_strategy(
        {
            "strategy": {
                "class": "freqtrade.mt5_trade.strategies.SmaCrossStrategy",
                "fast": 4,
                "slow": 9,
            }
        }
    )

    assert isinstance(strategy, SmaCrossStrategy)
    assert strategy.fast == 4
    assert strategy.slow == 9


def test_build_strategy_loads_from_path(tmp_path) -> None:
    module_file = tmp_path / "mystrat.py"
    module_file.write_text(
        "from freqtrade.mt5_trade.strategies import HOLD, MT5Strategy\n"
        "\n"
        "class MyStrat(MT5Strategy):\n"
        "    def __init__(self, threshold=1.0):\n"
        "        self.threshold = threshold\n"
        "    def on_bar(self, symbol, bars):\n"
        "        return HOLD\n"
    )

    strategy = build_default_strategy(
        {"strategy": {"class": "mystrat.MyStrat", "path": str(tmp_path), "threshold": 2.5}}
    )

    assert isinstance(strategy, MT5Strategy)
    assert strategy.__class__.__name__ == "MyStrat"
    assert strategy.threshold == 2.5


def test_build_strategy_rejects_non_strategy_class() -> None:
    with pytest.raises(OperationalException, match="not an MT5Strategy"):
        build_default_strategy({"strategy": {"class": "builtins.dict"}})


def test_build_strategy_rejects_missing_module() -> None:
    with pytest.raises(OperationalException, match="Cannot import strategy module"):
        build_default_strategy({"strategy": {"class": "no_such_module_xyz.Foo"}})


def test_build_strategy_rejects_missing_class() -> None:
    with pytest.raises(OperationalException, match="not found in module"):
        build_default_strategy(
            {"strategy": {"class": "freqtrade.mt5_trade.strategies.NopeStrategy"}}
        )


def test_build_strategy_rejects_bare_class_name() -> None:
    with pytest.raises(OperationalException, match="dotted path"):
        build_default_strategy({"strategy": {"class": "JustAName"}})


def test_build_strategy_reports_constructor_errors() -> None:
    with pytest.raises(OperationalException, match="Failed to construct strategy"):
        build_default_strategy(
            {
                "strategy": {
                    "class": "freqtrade.mt5_trade.strategies.SmaCrossStrategy",
                    "unknown_param": 5,
                }
            }
        )


def test_build_strategy_rejects_non_object_strategy() -> None:
    with pytest.raises(OperationalException, match="must be an object"):
        build_default_strategy({"strategy": "hello"})
