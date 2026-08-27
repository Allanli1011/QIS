# -*- coding: utf-8 -*-
"""并集日历下的价格→收益语义，以及价格清洗。"""
import numpy as np
import pandas as pd
import pytest

from qis.data.panel import (
    alive_mask,
    bad_tick_mask,
    clean_prices,
    coverage,
    ffill_prices,
    to_returns,
)


def _idx(n):
    return pd.date_range("2020-01-01", periods=n, freq="B")


def test_holiday_gap_defers_move_instead_of_dropping_it():
    """标的假期当日收益为 0，跨假期的涨跌落到下一个交易日——不能整段丢掉。"""
    px = pd.DataFrame({"A": [100.0, 101.0, np.nan, 103.0, 104.0]}, index=_idx(5))
    r = to_returns(px)["A"]
    assert r.iloc[1] == pytest.approx(0.01)
    assert r.iloc[2] == pytest.approx(0.0)                 # 假期
    assert r.iloc[3] == pytest.approx(103.0 / 101.0 - 1)   # 跨假期的真实涨幅
    # 复利还原：整段累计收益必须等于首尾价格之比
    assert float((1 + r.fillna(0)).prod()) == pytest.approx(104.0 / 100.0)


def test_pre_listing_and_post_delisting_stay_nan():
    px = pd.DataFrame({"A": [np.nan, np.nan, 50.0, 51.0, np.nan]}, index=_idx(5))
    alive = alive_mask(px)["A"]
    assert list(alive) == [False, False, True, True, False]
    r = to_returns(px)["A"]
    assert np.isnan(r.iloc[0]) and np.isnan(r.iloc[1])
    assert np.isnan(r.iloc[4])          # 退市后不可交易，不能当成 0 收益继续持有
    assert r.iloc[3] == pytest.approx(51.0 / 50.0 - 1)


def test_ffill_only_inside_alive_range():
    px = pd.DataFrame({"A": [np.nan, 10.0, np.nan, 12.0, np.nan]}, index=_idx(5))
    f = ffill_prices(px)["A"]
    assert np.isnan(f.iloc[0])
    assert f.iloc[2] == pytest.approx(10.0)   # 区间内沿用上一收盘
    assert np.isnan(f.iloc[4])                # 区间外不填充


def test_coverage_reports_real_quote_density():
    px = pd.DataFrame({"A": [np.nan, 1.0, np.nan, 1.0, 1.0]}, index=_idx(5))
    assert coverage(px)["A"] == pytest.approx(3 / 4)


def test_bad_tick_caught_but_real_spike_survives():
    """判据要求尖刺前后两天彼此自洽，真实暴涨不会被误伤。"""
    # 100 倍单位错误（ROUGHRICE 2011-01-04 的形态）
    bad = pd.Series([13.6, 14.1, 1390.0, 13.7, 13.4], index=_idx(5))
    assert bool(bad_tick_mask(bad).iloc[2])
    assert np.isnan(clean_prices(bad).iloc[2])
    # 真实暴涨后回落但不回到原位（VIX 2018-02-05 的形态）
    real = pd.Series([13.28, 15.63, 33.2, 23.9, 23.45], index=_idx(5))
    assert not bad_tick_mask(real).any()
    pd.testing.assert_series_equal(clean_prices(real), real)


def test_non_positive_prices_are_dropped():
    """负结算价（2020-04-20 WTI −13.1）在比率收益口径下没有意义。"""
    px = pd.Series([20.0, 18.0, -13.1, 9.0], index=_idx(4))
    c = clean_prices(px)
    assert np.isnan(c.iloc[2])
    r = to_returns(c.to_frame("A"))["A"]
    assert np.isfinite(r.iloc[3])
    assert r.iloc[3] == pytest.approx(9.0 / 18.0 - 1)   # 跨过坏点的真实跌幅仍保留


def test_works_on_series_and_frame_alike():
    s = pd.Series([1.0, np.nan, 2.0], index=_idx(3))
    assert to_returns(s).iloc[2] == pytest.approx(1.0)
    assert to_returns(s.to_frame("A"))["A"].iloc[2] == pytest.approx(1.0)
