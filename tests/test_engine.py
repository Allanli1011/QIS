# -*- coding: utf-8 -*-
"""引擎数值测试：手工构造 4 日 2 标的案例，逐步核对收益/换手/成本。"""
import numpy as np
import pandas as pd
import pytest

from qis.backtest.engine import run_backtest


@pytest.fixture
def case():
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    prices = pd.DataFrame(
        {"A": [100.0, 110.0, 110.0, 121.0], "B": [100.0, 100.0, 90.0, 99.0]},
        index=idx,
    )
    weights = pd.DataFrame(
        {"A": [0.5, 0.5, 1.0, 1.0], "B": [0.5, 0.5, 0.0, 0.0]},
        index=idx,
    )
    return prices, weights


def test_gross_returns(case):
    prices, weights = case
    res = run_backtest(prices, weights, cost_bps=0.0)
    # t 日权重作用于 t→t+1：day1 用 day0 权重
    expected = [0.0, 0.05, -0.05, 0.10]
    np.testing.assert_allclose(res.gross.values, expected, atol=1e-12)


def test_turnover_accounts_for_drift(case):
    prices, weights = case
    res = run_backtest(prices, weights, cost_bps=0.0)
    # day0 建仓 1.0；day1 漂移后 (0.5*1.1, 0.5*1.0)/1.05 → 换手 2*|0.5-0.5238...|
    # day2 漂移到 (0.526316, 0.473684)，调到 (1,0) → 换手 0.947368
    expected = [1.0, 0.047619048, 0.947368421, 0.0]
    np.testing.assert_allclose(res.turnover.values, expected, atol=1e-8)


def test_cost_and_net(case):
    prices, weights = case
    res = run_backtest(prices, weights, cost_bps=10.0)  # 10 bps = 0.001
    np.testing.assert_allclose(res.cost.values, res.turnover.values * 0.001, atol=1e-12)
    np.testing.assert_allclose(res.net.values, res.gross.values - res.cost.values, atol=1e-12)
    np.testing.assert_allclose(
        res.equity.values, np.cumprod(1.0 + res.net.values), atol=1e-12
    )


def test_per_instrument_cost(case):
    prices, weights = case
    res = run_backtest(prices, weights, cost_bps={"A": 10.0, "B": 20.0})
    # day0 建仓：|ΔA|=0.5*10bps + |ΔB|=0.5*20bps
    assert res.cost.iloc[0] == pytest.approx(0.5 * 0.001 + 0.5 * 0.002, abs=1e-12)


def test_no_lookahead(case):
    """当天权重不影响当天收益。"""
    prices, weights = case
    w2 = weights.copy()
    w2.iloc[-1] = [0.0, 1.0]  # 改最后一天权重
    res1 = run_backtest(prices, weights)
    res2 = run_backtest(prices, w2)
    # 最后一天的 gross 不变（用的是前一天权重）
    assert res1.gross.iloc[-1] == res2.gross.iloc[-1]


def test_nan_prices_are_safe():
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    prices = pd.DataFrame({"A": [100.0, 101.0, np.nan]}, index=idx)
    weights = pd.DataFrame({"A": [1.0, 1.0, 1.0]}, index=idx)
    res = run_backtest(prices, weights, cost_bps=0.0)
    assert np.isfinite(res.net).all()
    assert res.net.iloc[-1] == pytest.approx(0.0)


def test_roll_cost_charged_on_roll_days():
    """换月日按 |持仓| × 2 倍单边费率计成本。"""
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    prices = pd.DataFrame({"A": [100.0, 100.0, 100.0, 100.0]}, index=idx)
    weights = pd.DataFrame({"A": [1.0, 1.0, 1.0, 1.0]}, index=idx)
    roll_mask = pd.DataFrame({"A": [False, True, False, True]}, index=idx)
    res = run_backtest(prices, weights, cost_bps=10.0, roll_mask=roll_mask)
    # day1、day3 换月：各 1.0 × 0.001 × 2 = 0.002；day0 建仓 0.001
    expected = [0.001, 0.002, 0.0, 0.002]
    np.testing.assert_allclose(res.cost.values, expected, atol=1e-12)
    # 无 roll_mask 时只有建仓成本
    res2 = run_backtest(prices, weights, cost_bps=10.0)
    np.testing.assert_allclose(res2.cost.values, [0.001, 0.0, 0.0, 0.0], atol=1e-12)
