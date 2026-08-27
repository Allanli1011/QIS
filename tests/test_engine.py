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


def test_data_gap_does_not_fabricate_turnover():
    """
    回归：目标权重恒定的买入持有，遇到某标的的假期不应产生换手。

    旧引擎只把持仓在缺口日清零、目标权重不动，|目标−漂移| 把整个仓位
    记成一轮平仓+建仓，恒定 0.5 的持仓会报出 2.00 的总换手。
    """
    idx = pd.date_range("2020-01-01", periods=6, freq="B")
    prices = pd.DataFrame({
        "A": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
        "B": [50.0, 50.5, np.nan, 51.5, 52.0, 52.5],   # 第 3 天是 B 的假期
    }, index=idx)
    weights = pd.DataFrame(0.5, index=idx, columns=["A", "B"])
    res = run_backtest(prices, weights, cost_bps=10.0)

    assert res.weights["B"].iloc[2] == pytest.approx(0.5)   # 假期照样持仓
    # 只有首日建仓 1.0 + 少量真实漂移再平衡，远小于旧引擎的 2.0
    assert res.turnover.sum() < 1.05
    # 跨假期的涨幅必须落到下一个交易日，不能丢
    assert res.returns["B"].iloc[3] == pytest.approx(51.5 / 50.5 - 1)


def test_delisted_instrument_stops_trading_without_cost():
    """退市后既不持仓也不该继续计换手。"""
    idx = pd.date_range("2020-01-01", periods=4, freq="B")
    prices = pd.DataFrame({"A": [100.0, 101.0, np.nan, np.nan]}, index=idx)
    weights = pd.DataFrame({"A": [1.0, 1.0, 1.0, 1.0]}, index=idx)
    res = run_backtest(prices, weights, cost_bps=10.0)
    assert res.turnover.iloc[2] == pytest.approx(0.0)
    assert res.turnover.iloc[3] == pytest.approx(0.0)
    assert res.weights["A"].iloc[2] == pytest.approx(0.0)


def test_returns_exposed_for_attribution():
    """归因必须复用引擎的收益，避免调用方重算出不同口径。"""
    idx = pd.date_range("2020-01-01", periods=4, freq="B")
    prices = pd.DataFrame({"A": [100.0, 110.0, np.nan, 121.0]}, index=idx)
    weights = pd.DataFrame({"A": [1.0, 1.0, 1.0, 1.0]}, index=idx)
    res = run_backtest(prices, weights)
    assert res.returns.shape == prices.shape
    assert res.returns["A"].iloc[1] == pytest.approx(0.10)
    assert res.returns["A"].iloc[2] == pytest.approx(0.0)     # 假期
    assert res.returns["A"].iloc[3] == pytest.approx(0.10)    # 跨假期
    # 分标的贡献之和 = 组合费前收益
    np.testing.assert_allclose((res.weights * res.returns).sum(axis=1).values,
                               res.gross.values, atol=1e-12)
