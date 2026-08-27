# -*- coding: utf-8 -*-
"""策略与组合构建测试（toy 数据，验证方向与归一化）。"""
import numpy as np
import pandas as pd
import pytest

from qis.portfolio.construction import normalize_gross, vol_target_scale
from qis.strategy.carry import carry_signals, carry_weights
from qis.strategy.trend import trend_signals, trend_weights
from qis.strategy.xsmom import xsmom_signals, xsmom_weights


def _trendy_prices(n=300):
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    rng = np.random.default_rng(7)
    up = np.cumprod(1 + 0.001 + rng.normal(0, 0.005, n))
    down = np.cumprod(1 - 0.001 + rng.normal(0, 0.005, n))
    flat = np.cumprod(1 + rng.normal(0, 0.005, n))
    return pd.DataFrame({"UP": up, "DOWN": down, "FLAT": flat}, index=idx)


def test_trend_signal_direction():
    prices = _trendy_prices()
    sig = trend_signals(prices, lookbacks=(21, 63))
    last = sig.iloc[-1]
    assert last["UP"] > 0 and last["DOWN"] < 0
    assert sig.abs().max().max() <= 1.0


def test_trend_weights_gross_normalized():
    prices = _trendy_prices()
    w = trend_weights(prices, lookbacks=(21, 63), gross=1.0)
    active = w.abs().sum(axis=1)
    active = active[active > 0]
    np.testing.assert_allclose(active.values, 1.0, atol=1e-10)
    assert w["UP"].iloc[-1] > 0 > w["DOWN"].iloc[-1]


def test_xsmom_long_strong_short_weak():
    prices = _trendy_prices()
    sig = xsmom_signals(prices, lookback=63)
    last = sig.iloc[-1]
    assert last["UP"] > last["FLAT"] > last["DOWN"]
    assert sig.mean(axis=1).abs().max() < 0.35  # 截面大致去均值


def test_xsmom_groups_isolated():
    prices = _trendy_prices()
    sig = xsmom_signals(prices, lookback=63,
                        groups={"UP": "g1", "DOWN": "g1", "FLAT": "g2"})
    assert sig["FLAT"].iloc[-1] == 0.0  # 单标的组信号为 0（组内 <3 个）


def test_carry_cross_sectional_long_high_short_low():
    idx = pd.date_range("2023-01-01", periods=300, freq="B")
    rng = np.random.default_rng(3)
    base = np.cumprod(1 + rng.normal(0, 0.01, 300))
    front = pd.DataFrame({"A": base * 100, "B": base * 50, "C": base * 10}, index=idx)
    # A 大幅贴水(高 carry)，B 小幅贴水，C 升水(低 carry)
    deferred = pd.DataFrame({"A": base * 90, "B": base * 49, "C": base * 11}, index=idx)
    w = carry_weights(front, deferred, groups={"A": "g", "B": "g", "C": "g"}, mode="xs")
    assert w["A"].iloc[-1] > 0 > w["C"].iloc[-1]
    # 组内多空对冲：权重和≈0
    assert w.iloc[-1].sum() == pytest.approx(0.0, abs=1e-10)


def test_carry_ts_mode_uses_own_history():
    idx = pd.date_range("2022-01-01", periods=400, freq="B")
    # A：价差长期 -2%（结构性 contango），近期升到 +2% → 信号应为正
    front = pd.Series(100.0, index=idx)
    defer_a = pd.Series(102.0, index=idx)
    defer_a.iloc[-50:] = 98.0
    f = pd.DataFrame({"A": front})
    d = pd.DataFrame({"A": defer_a})
    sig = carry_signals(f, d, mode="ts", lookback=126)
    assert sig["A"].iloc[-1] > 0
    # 结构性 contango 未变时信号应≈0（不会永久做空）
    sig_flat = carry_signals(f, pd.DataFrame({"A": pd.Series(102.0, index=idx)}),
                             mode="ts", lookback=126)
    assert abs(sig_flat["A"].iloc[-1]) < 1e-12


def test_carry_no_leg_instrument_zero():
    idx = pd.date_range("2023-01-01", periods=300, freq="B")
    rng = np.random.default_rng(3)
    base = np.cumprod(1 + rng.normal(0, 0.01, 300))
    front = pd.DataFrame({"A": base * 100, "B": base * 50}, index=idx)
    deferred = pd.DataFrame({"A": base * 95}, index=idx)  # B 无腿
    w = carry_weights(front, deferred)
    assert (w["B"] == 0.0).all()


def test_normalize_gross_zero_rows_stay_zero():
    w = pd.DataFrame({"A": [0.0, 1.0], "B": [0.0, -1.0]})
    out = normalize_gross(w)
    assert out.iloc[0].sum() == 0.0
    assert out.iloc[1].abs().sum() == pytest.approx(1.0)


def test_weight_band():
    from qis.portfolio.construction import weight_band
    target = pd.DataFrame({
        "A": [0.30, 0.31, 0.50, 0.51],   # 0.31→0.30 偏离 0.01 被带挡下；0.50 放行
        "B": [0.20, 0.20, 0.20, 0.20],
    })
    out = weight_band(target, 0.03)
    assert out["A"].tolist() == [0.30, 0.30, 0.50, 0.50]
    assert out["B"].tolist() == [0.20] * 4
    # thresh=0 原样返回
    pd.testing.assert_frame_equal(weight_band(target, 0.0), target)
    # 无未来函数：改最后一行不影响历史
    t2 = target.copy(); t2.iloc[-1] = [9.9, 9.9]
    pd.testing.assert_frame_equal(weight_band(t2, 0.03).iloc[:-1], out.iloc[:-1])


def test_vol_target_scale_respects_cap_and_no_nans():
    prices = _trendy_prices()
    rets = prices.pct_change(fill_method=None).fillna(0.0)
    w = trend_weights(prices, lookbacks=(21, 63))
    scaled = vol_target_scale(w, rets, target=0.10, max_leverage=2.0)
    assert np.isfinite(scaled.values).all()
    gross_ratio = scaled.abs().sum(axis=1) / w.abs().sum(axis=1).replace(0, np.nan)
    assert gross_ratio.max() <= 2.0 + 1e-9


def test_xsmom_signal_is_exactly_zero_mean():
    """
    回归：rank(pct=True) 取值 1/n..1，均值是 (n+1)/(2n) 不是 0.5，
    直接减 0.5 会给这个"截面中性"策略留下 +1/n 的系统性多头偏移。
    """
    rng = np.random.default_rng(3)
    for n in (3, 5, 12):
        cols = [f"I{i}" for i in range(n)]
        px = pd.DataFrame(
            100 * np.cumprod(1 + rng.normal(0, 0.01, (300, n)), axis=0),
            index=pd.date_range("2022-01-01", periods=300, freq="B"), columns=cols)
        sig = xsmom_signals(px, lookback=126)
        warm = sig.iloc[200:]
        assert warm.mean(axis=1).abs().max() < 1e-12
        assert warm.abs().to_numpy().max() <= 1.0 + 1e-12


def test_weight_band_does_not_mutate_input():
    """
    回归：旧实现用 target.iloc[i] 取到 DataFrame 的视图并写回，
    会把调用方的权重矩阵一起改掉，令"带 vs 不带"的对照实验拿到同一份数据。
    """
    from qis.portfolio.construction import weight_band
    target = pd.DataFrame({"A": [0.10, 0.101, 0.30], "B": [-0.20, -0.201, -0.50]})
    before = target.copy()
    out = weight_band(target, 0.03)
    pd.testing.assert_frame_equal(target, before)
    assert not out.equals(before)


def test_weight_band_handles_empty_frame():
    from qis.portfolio.construction import weight_band
    empty = pd.DataFrame(columns=["A", "B"], dtype=float)
    assert weight_band(empty, 0.03).empty          # 旧实现在这里抛 IndexError


def test_cap_binding_share_flags_inert_vol_target():
    """杠杆上限长期绑定时，波动目标形同虚设——这件事必须能被量化出来。"""
    from qis.portfolio.construction import cap_binding_share, vol_target_factor
    idx = pd.date_range("2020-01-01", periods=400, freq="B")
    rng = np.random.default_rng(11)
    # 极低波动的账本：target/vol_est 远超上限
    rets = pd.DataFrame({"A": rng.normal(0, 0.00002, 400)}, index=idx)
    w = pd.DataFrame({"A": 1.0}, index=idx)
    _, raw = vol_target_factor(w, rets, target=0.10, span=40, ann_factor=252,
                               max_leverage=3.0)
    assert cap_binding_share(raw, 3.0) > 0.9
    # 高波动账本上限不绑定
    rets2 = pd.DataFrame({"A": rng.normal(0, 0.02, 400)}, index=idx)
    _, raw2 = vol_target_factor(w, rets2, target=0.10, span=40, ann_factor=252,
                                max_leverage=3.0)
    assert cap_binding_share(raw2, 3.0) < 0.1


def test_strategies_survive_calendar_gaps():
    """并集日历的假期洞不该把信号打成 0（旧实现 shift(lb) 一碰到 NaN 就整段失效）。"""
    px = _trendy_prices(400)
    holed = px.copy()
    holed.iloc[::9, 0] = np.nan          # UP 每 9 天缺一天
    for fn in (trend_weights, xsmom_weights):
        w = fn(holed)
        assert np.isfinite(w.values).all()
        assert w.iloc[-1].abs().sum() == pytest.approx(1.0)
        assert abs(w["UP"].iloc[-1]) > 1e-6   # 有洞的标的仍然拿得到权重


def test_carry_raw_requires_synchronous_quotes():
    """
    回归：两腿的停牌日未必重合，各自 ffill 再相除会把"旧 c1 / 新 c2"凑成价差，
    凭空造出期限结构变动。carry 信号对这种噪声极其敏感
    （实测 corr(信噪比, IC) = −0.53）。
    """
    from qis.strategy.carry import carry_raw
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    front = pd.DataFrame({"A": [100.0, 101.0, 102.0, 103.0, 104.0]}, index=idx)
    defer = pd.DataFrame({"A": [105.0, np.nan, np.nan, 108.0, 109.0]}, index=idx)
    raw = carry_raw(front, defer)["A"]
    first = 100.0 / 105.0 - 1.0
    # 远月缺报价的两天应沿用上一个**同步**价差，而不是拿 101/105、102/105 现算
    assert raw.iloc[1] == pytest.approx(first)
    assert raw.iloc[2] == pytest.approx(first)
    assert raw.iloc[3] == pytest.approx(103.0 / 108.0 - 1.0)


def test_carry_smoothing_reduces_signal_noise():
    """价差先平滑再取信号，是为了压掉 c1/c2 的日度测量噪声。"""
    from qis.strategy.carry import carry_signals
    idx = pd.date_range("2022-01-01", periods=600, freq="B")
    rng = np.random.default_rng(4)
    lvl = pd.Series(np.linspace(0, 0.02, 600), index=idx)          # 缓慢的真实期限结构
    noise = pd.Series(rng.normal(0, 0.01, 600), index=idx)          # 日度测量噪声
    f = pd.DataFrame({"A": 100 * (1 + lvl + noise)}, index=idx)
    d = pd.DataFrame({"A": 100.0}, index=idx)
    s1 = carry_signals(f, d, smooth=1)["A"].iloc[300:]
    s21 = carry_signals(f, d, smooth=21)["A"].iloc[300:]
    assert s21.diff().abs().mean() < s1.diff().abs().mean() / 3


def test_xsmom_defaults_rank_across_whole_universe():
    """
    截面动量靠广度：本池按资产类别分组时 rates/crypto 只有 2 个成员会被整组置零。
    默认必须是全体一起排名。
    """
    import inspect
    from qis.strategy.xsmom import xsmom_weights
    sig = inspect.signature(xsmom_weights)
    assert sig.parameters["groups"].default is None
    assert sig.parameters["risk_adjusted"].default is True

    rng = np.random.default_rng(9)
    idx = pd.date_range("2020-01-01", periods=400, freq="B")
    cols = [f"I{i}" for i in range(8)]
    px = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.01, (400, 8)), axis=0),
                      index=idx, columns=cols)
    # 两个成员的小组会被整组置零 → 分组排名等于放弃这些标的
    tiny = {c: ("small" if i < 2 else "big") for i, c in enumerate(cols)}
    w_grouped = xsmom_weights(px, groups=tiny)
    assert w_grouped[cols[:2]].abs().sum().sum() == pytest.approx(0.0)
    # 全体排名时它们照样参与
    assert xsmom_weights(px)[cols[:2]].abs().sum().sum() > 0


def test_xsmom_risk_adjusted_ranking_does_not_let_vol_dominate():
    """跨资产类别比原始收益，两端会被高波动品种长期霸占；先除波动才可比。"""
    from qis.strategy.xsmom import xsmom_signals
    idx = pd.date_range("2020-01-01", periods=500, freq="B")
    rng = np.random.default_rng(13)
    # LOUD 波动是其他标的的 10 倍，但漂移相同
    data = {f"Q{i}": 100 * np.cumprod(1 + rng.normal(0.0002, 0.005, 500)) for i in range(5)}
    data["LOUD"] = 100 * np.cumprod(1 + rng.normal(0.0002, 0.05, 500))
    px = pd.DataFrame(data, index=idx)
    raw = xsmom_signals(px, risk_adjusted=False).iloc[300:]["LOUD"].abs().mean()
    adj = xsmom_signals(px, risk_adjusted=True).iloc[300:]["LOUD"].abs().mean()
    assert adj < raw
