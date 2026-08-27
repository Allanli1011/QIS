# -*- coding: utf-8 -*-
"""换月识别与调整收益测试。"""
import numpy as np
import pandas as pd
import pytest

from qis.data.roll import (
    adjust,
    adjusted_price_index,
    adjusted_returns,
    oi_roll_days,
    roll_days,
    roll_diagnostics,
    volume_roll_days,
)


def _idx(n):
    return pd.date_range("2024-01-01", periods=n, freq="B")


def _oi_series(n, roll_at, low=1_000.0, high=50_000.0):
    """持仓量：换月日跳升到新合约的量级，之后逐日萎缩。"""
    idx = _idx(n)
    vals = []
    cur = high
    for i in range(n):
        if i in roll_at:
            cur = high
        else:
            cur = cur * 0.97 if i else cur
        vals.append(cur if i not in roll_at else high)
    # 换月前一天压到很低，制造跳升
    s = pd.Series(vals, index=idx)
    for i in roll_at:
        if i > 0:
            s.iloc[i - 1] = low
    return s


def test_oi_jump_marks_roll_day():
    n = 60
    oi = _oi_series(n, roll_at={20, 45})
    mask = oi_roll_days(oi)
    assert list(np.flatnonzero(mask.to_numpy())) == [20, 45]


def test_min_gap_collapses_adjacent_hits():
    """一次换月的连续跳升只算一天。"""
    idx = _idx(30)
    oi = pd.Series(1000.0, index=idx)
    oi.iloc[10] = 50_000.0
    oi.iloc[11] = 3_000_000.0   # 紧邻的第二次跳升属于同一次换月
    mask = oi_roll_days(oi, min_gap=10)
    assert int(mask.sum()) == 1
    assert mask.to_numpy()[10]


def test_volume_used_when_oi_missing():
    n = 60
    vol = _oi_series(n, roll_at={30})
    oi = pd.Series(np.nan, index=_idx(n))
    mask, method = roll_days(pd.Series(100.0, index=_idx(n)), oi, vol)
    assert method == "volume"
    assert int(mask.sum()) == 1


def test_no_signal_reports_none_instead_of_guessing():
    """识别不出来要显式返回 none，而不是悄悄用一串错误收益。"""
    n = 40
    flat = pd.Series(1000.0, index=_idx(n))
    mask, method = roll_days(pd.Series(100.0, index=_idx(n)), flat, flat)
    assert method == "none"
    assert not mask.any()
    d = roll_diagnostics(pd.Series(100.0, index=_idx(n)), flat, flat)
    assert d["method"] == "none" and d["plausible"] is False


def test_roll_day_return_uses_front_over_lagged_deferred():
    """换月日收益 = c1(T)/c2(T-1) − 1（同一张合约），不是 c1 或 c2 的自身收益。"""
    n = 30
    idx = _idx(n)
    # 平日 c1、c2 各自平移；第 15 天 c1 跳到 c2 的合约上（contango，价格跳升）
    c1 = pd.Series(100.0, index=idx)
    c2 = pd.Series(105.0, index=idx)
    c1.iloc[15:] = 105.5      # 换月后 c1 指向原 c2 合约，且当天涨了 0.5
    c2.iloc[15:] = 110.0
    oi = _oi_series(n, roll_at={15})

    r, mask, method = adjust(c1, c2, front_oi=oi)
    assert method == "oi" and bool(mask.iloc[15])
    # naive 会记成 +5.5%（幻影），正确值是 105.5/105 − 1
    assert r.iloc[15] == pytest.approx(105.5 / 105.0 - 1.0, rel=1e-12)
    assert r.iloc[15] < 0.01
    # 非换月日仍取 c1 自身收益
    assert r.iloc[16] == pytest.approx(0.0, abs=1e-12)


def test_falls_back_to_naive_when_deferred_missing():
    n = 30
    idx = _idx(n)
    c1 = pd.Series(100.0, index=idx); c1.iloc[15:] = 105.0
    c2 = pd.Series(np.nan, index=idx)
    oi = _oi_series(n, roll_at={15})
    r = adjusted_returns(c1, c2, front_oi=oi)
    assert r.iloc[15] == pytest.approx(0.05, rel=1e-12)   # 无从修正就保持原样，不造 NaN


def test_diagnostics_flags_implausible_frequency():
    """每年换几十次换月是不可信的，必须报出来。"""
    n = 500
    idx = _idx(n)
    oi = pd.Series(1000.0, index=idx)
    oi.iloc[::12] = 500_000.0          # 每 12 天一次 → 约 21 次/年
    d = roll_diagnostics(pd.Series(100.0, index=idx), oi, None)
    assert d["method"] == "oi"
    assert d["per_year"] > 14
    assert d["plausible"] is False


def test_adjusted_price_index():
    r = pd.Series([np.nan, 0.01, -0.02, 0.03], index=_idx(4))
    idx = adjusted_price_index(r, base=100.0)
    # 基准落在首个有效收益的前一天，第一天的收益才不会丢
    assert idx.iloc[0] == pytest.approx(100.0)
    assert idx.iloc[1] == pytest.approx(101.0)
    assert idx.iloc[3] == pytest.approx(100 * 1.01 * 0.98 * 1.03)
    # 对指数再做 pct_change 应完整还原原收益序列（含第一天）
    np.testing.assert_allclose(idx.pct_change().values[1:], r.values[1:], atol=1e-12)


def test_price_index_keeps_leading_nans_out_of_range():
    r = pd.Series([np.nan, np.nan, 0.01, 0.02], index=_idx(4))
    idx = adjusted_price_index(r)
    assert np.isnan(idx.iloc[0])              # 上市前
    assert idx.iloc[1] == pytest.approx(100.0)  # 基准日
    np.testing.assert_allclose(idx.pct_change().values[2:], r.values[2:], atol=1e-12)
