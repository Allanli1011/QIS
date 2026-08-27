# -*- coding: utf-8 -*-
"""换月调整测试。"""
import numpy as np
import pandas as pd
import pytest

from qis.data.roll import (
    adjusted_price_index, adjusted_returns, roll_starts,
    v1_roll_window, volume_roll_window,
)


def _s(idx, vals):
    return pd.Series(vals, index=idx, dtype=float)


def test_volume_roll_window():
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    fv = _s(idx, [100, 100, 90, 50, 40])
    dv = _s(idx, [50, 50, 95, 200, 210])
    roll = volume_roll_window(fv, dv)
    assert roll.tolist() == [False, False, True, True, True]


def test_v1_roll_window():
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    # day3 起 v1 已切新合约（收益与 c1 不同），c1 尚未切
    c1 = _s(idx, [100, 101, 102, 103, 104])
    v1 = _s(idx, [100, 101, 102, 103.5, 104.2])
    w = v1_roll_window(c1, v1, tol=1e-4)
    assert w.tolist() == [False, False, False, True, True]


def test_roll_starts_marks_window_beginnings():
    idx = pd.date_range("2024-01-01", periods=7, freq="D")
    window = pd.Series([False, True, True, False, False, True, False], index=idx)
    starts = roll_starts(window)
    assert starts.tolist() == [False, True, False, False, False, True, False]


def test_roll_day_uses_deferred_return():
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    # day2 换月：近月 100→50（拼接 gap），远月真实波动 60→61.2（+2%）
    fc = _s(idx, [100, 100, 50, 51])
    dc = _s(idx, [60, 60, 61.2, 62.424])
    fv = _s(idx, [100, 100, 10, 10])
    dv = _s(idx, [50, 50, 200, 200])
    r = adjusted_returns(fc, dc, fv, dv)
    assert r.iloc[1] == pytest.approx(0.0)          # 正常日：近月收益
    assert r.iloc[2] == pytest.approx(0.02)         # 换月日：远月收益（无 -50% 幻影）
    assert r.iloc[3] == pytest.approx(0.02)


def test_v1_takes_priority_over_volume():
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    fc = _s(idx, [100, 100, 50, 51])
    dc = _s(idx, [60, 60, 61.2, 62.424])
    # 成交量规则说 day3 换月，但 v1 说 day2 已切 → 用 v1
    fv = _s(idx, [100, 100, 100, 10])
    dv = _s(idx, [50, 50, 50, 200])
    v1 = _s(idx, [100, 100, 50.5, 51.3])
    r = adjusted_returns(fc, dc, fv, dv, v1_close=v1)
    assert r.iloc[2] == pytest.approx(0.02)  # day2 用远月收益
    assert r.iloc[3] == pytest.approx(0.02)  # day3 v1/c1 仍背离（c1 未切），仍用远月


def test_no_volume_falls_back_to_plain_returns():
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    fc = _s(idx, [100, 102, 101])
    r = adjusted_returns(fc, fc)  # 无 volume、无 v1
    expected = fc.pct_change(fill_method=None)
    pd.testing.assert_series_equal(r, expected)


def test_adjusted_price_index():
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    r = pd.Series([np.nan, 0.01, -0.01], index=idx)
    px = adjusted_price_index(r, base=100.0)
    assert np.isnan(px.iloc[0])
    assert px.iloc[1] == pytest.approx(101.0)
    assert px.iloc[2] == pytest.approx(99.99)
