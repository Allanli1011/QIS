# -*- coding: utf-8 -*-
"""成本映射与配置加载测试。"""
from qis.backtest.costs import cost_bps_by_name, load_settings


def test_cost_bps_by_name_uses_class_and_default():
    ac = {"SP500": "equity_index", "WTI": "energy", "FOO": "unknown_class"}
    table = {"default": 1.0, "equity_index": 0.5, "energy": 1.5}
    out = cost_bps_by_name(ac, table)
    assert out == {"SP500": 0.5, "WTI": 1.5, "FOO": 1.0}


def test_load_settings_file():
    cfg = load_settings()
    assert "backtest" in cfg and "cost_bps" in cfg
    assert cfg["backtest"]["ann_factor"] == 252
    assert cfg["cost_bps"]["default"] > 0
