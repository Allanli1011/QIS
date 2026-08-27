# -*- coding: utf-8 -*-
"""指标数值测试。"""
import numpy as np
import pandas as pd
import pytest

from qis.analytics import metrics


def test_ann_return_compound():
    r = pd.Series([0.001] * 252)
    assert metrics.ann_return(r) == pytest.approx(1.001**252 - 1, rel=1e-12)


def test_ann_vol():
    r = pd.Series([0.01, -0.01] * 100)
    assert metrics.ann_vol(r) == pytest.approx(r.std(ddof=1) * np.sqrt(252), rel=1e-12)


def test_sharpe():
    rng = np.random.default_rng(42)
    r = pd.Series(rng.normal(0.0005, 0.01, 500))
    expected = r.mean() / r.std(ddof=1) * np.sqrt(252)
    assert metrics.sharpe(r) == pytest.approx(expected, rel=1e-12)


def test_max_drawdown():
    equity = pd.Series([1.0, 1.1, 0.99, 1.2, 1.0])
    # 峰值 1.2，谷底 1.0 → mdd = 1/1.2 - 1
    assert metrics.max_drawdown(equity) == pytest.approx(1.0 / 1.2 - 1, rel=1e-12)


def test_sortino_only_penalizes_downside():
    r = pd.Series([0.02, 0.03, -0.01, 0.01, 0.04])
    downside = r[r < 0]
    dd = np.sqrt((downside**2).mean())
    assert metrics.sortino(r) == pytest.approx(r.mean() / dd * np.sqrt(252), rel=1e-12)


def test_hit_ratio_ignores_zero_days():
    r = pd.Series([0.01, -0.01, 0.0, 0.02, -0.03, 0.0])
    assert metrics.hit_ratio(r) == pytest.approx(0.5)


def test_calmar():
    r = pd.Series([0.001] * 126 + [-0.002] * 63 + [0.001] * 63)
    eq = (1 + r).cumprod()
    expected = metrics.ann_return(r) / abs(metrics.max_drawdown(eq))
    assert metrics.calmar(r) == pytest.approx(expected, rel=1e-12)


def test_summary_keys():
    r = pd.Series(np.random.default_rng(1).normal(0, 0.01, 300))
    t = pd.Series(0.01, index=r.index)
    s = metrics.summary(r, t)
    for k in ["ann_return", "ann_vol", "sharpe", "sortino", "max_drawdown",
              "calmar", "hit_ratio", "ann_turnover", "n_days"]:
        assert k in s.index
    assert s["n_days"] == 300


def test_ann_return_uses_calendar_span_not_row_count():
    """
    回归：并集日历每年约 263 行，用 len/252 当年数会把 16.6 年当成 17.4 年，
    年化收益被系统性压低。有时间索引时应按日历跨度折算。
    """
    idx = pd.date_range("2020-01-01", periods=263 * 2, freq="D")[: 263 * 2]
    r = pd.Series(0.0004, index=idx)
    total = (1 + r).prod()
    years = (idx[-1] - idx[0]).days / 365.25
    assert metrics.ann_return(r) == pytest.approx(total ** (1 / years) - 1, rel=1e-9)
    # 无时间索引时退回 len/ann_factor（老行为）
    plain = pd.Series(0.0004, index=range(len(r)))
    assert metrics.ann_return(plain) == pytest.approx(total ** (252 / len(r)) - 1, rel=1e-9)


def test_infer_ann_factor_from_index():
    idx = pd.bdate_range("2015-01-01", "2025-01-01")
    r = pd.Series(0.0, index=idx)
    af = metrics.infer_ann_factor(r)
    assert 255 < af < 265                      # 工作日约 261/年
    # 样本不足一年或非时间索引 → 回退默认值
    assert metrics.infer_ann_factor(pd.Series(0.0, index=idx[:50])) == 252
    assert metrics.infer_ann_factor(pd.Series([0.0, 0.1])) == 252


def test_summary_infers_ann_factor_when_not_given():
    idx = pd.bdate_range("2015-01-01", "2025-01-01")
    rng = np.random.default_rng(2)
    r = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)
    s = metrics.summary(r)
    assert 255 < s["ann_factor"] < 265
    # 显式指定时以调用方为准
    assert metrics.summary(r, ann_factor=252)["ann_factor"] == 252
