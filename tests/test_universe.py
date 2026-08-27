# -*- coding: utf-8 -*-
"""标的池测试。"""
from qis.data.universe import Universe


def test_load_default_yaml():
    u = Universe.from_yaml()
    assert len(u) > 0
    assert "SP500" in u.names()
    assert "ESc1" in u.rics()


def test_all_rics_includes_carry_legs():
    u = Universe.from_yaml()
    all_rics = u.all_rics()
    assert "ESc1" in all_rics and "ESc2" in all_rics
    assert len(all_rics) == len(set(all_rics))  # 无重复


def test_asset_classes_and_carry_legs():
    u = Universe.from_yaml()
    ac = u.asset_classes()
    assert ac["SP500"] == "equity_index"
    legs = u.carry_legs()
    assert legs["SP500"] == "ESc2"
    assert "EURUSD" not in legs  # 外汇现货未配 carry 腿


def test_ric_name_roundtrip(tmp_path):
    p = tmp_path / "u.yaml"
    p.write_text(
        "instruments:\n  - { name: X, ric: Xc1, asset_class: bond }\n",
        encoding="utf-8",
    )
    u = Universe.from_yaml(p)
    assert u.ric_to_name() == {"Xc1": "X"}
    assert u.by_class("bond")[0]["name"] == "X"
