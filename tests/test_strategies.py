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
    """两种构造都必须给出正确方向；边界各按各自的定义。"""
    prices = _trendy_prices()
    ew = trend_signals(prices)                                   # 默认 ewmac
    assert ew.iloc[-1]["UP"] > 0 > ew.iloc[-1]["DOWN"]
    assert ew.abs().max().max() <= 2.0 + 1e-9                    # clip=2.0

    sv = trend_signals(prices, method="sign", lookbacks=(21, 63))
    assert sv.iloc[-1]["UP"] > 0 > sv.iloc[-1]["DOWN"]
    assert sv.abs().max().max() <= 1.0                           # 符号投票落在 [-1,1]


def test_ewmac_keeps_conviction_that_sign_vote_throws_away():
    """
    符号投票只看涨跌方向：微弱上涨与强劲上涨记同一票。
    EWMAC 保留强度，强趋势应当给出明显更大的信号。
    """
    idx = pd.date_range("2022-01-01", periods=400, freq="B")
    rng = np.random.default_rng(21)
    noise = rng.normal(0, 0.003, 400)         # 两者共用同一份噪声，只有漂移不同
    weak = 100 * np.cumprod(1 + 0.0006 + noise)     # +23%
    strong = 100 * np.cumprod(1 + 0.0030 + noise)   # +221%
    px = pd.DataFrame({"WEAK": weak, "STRONG": strong}, index=idx)
    sv = trend_signals(px, method="sign").iloc[-1]
    ew = trend_signals(px).iloc[-1]
    # 符号投票：两者都是满票，10 倍的涨幅差异被完全抹平
    assert sv["WEAK"] == pytest.approx(1.0) and sv["STRONG"] == pytest.approx(1.0)
    # EWMAC：强趋势拿到约 2 倍的信号
    assert ew["STRONG"] > 1.8 * ew["WEAK"] > 0


def test_ewmac_clip_bounds_single_instrument_conviction():
    from qis.strategy.trend import ewmac_signals
    idx = pd.date_range("2022-01-01", periods=400, freq="B")
    px = pd.DataFrame({"A": 100 * np.cumprod(np.full(400, 1.01))}, index=idx)  # 极端趋势
    assert ewmac_signals(px, clip=1.0).abs().max().max() <= 1.0 + 1e-9


def test_trend_weights_gross_normalized():
    prices = _trendy_prices()
    w = trend_weights(prices, method="sign", lookbacks=(21, 63), gross=1.0)
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


def test_carry_annualized_makes_cycles_comparable():
    """
    月度合约的 c1/c2 只跨 1 个月、季度合约跨 3 个月，不年化就没法在截面上比。
    同样的年化 carry，两者的信号应当一致。
    """
    from qis.strategy.carry import carry_raw
    idx = pd.date_range("2024-01-01", periods=10, freq="B")
    # MONTHLY 每月 -1%（年化 -12%）；QUARTERLY 每季 -3%（年化也是 -12%）
    front = pd.DataFrame({"MONTHLY": 99.0, "QUARTERLY": 97.0}, index=idx)
    defer = pd.DataFrame({"MONTHLY": 100.0, "QUARTERLY": 100.0}, index=idx)
    horizon = pd.Series({"MONTHLY": 1 / 12, "QUARTERLY": 0.25})
    plain = carry_raw(front, defer).iloc[-1]
    assert abs(plain["MONTHLY"]) < abs(plain["QUARTERLY"]) / 2   # 未年化：差 3 倍
    ann = carry_raw(front, defer, horizon=horizon).iloc[-1]
    assert ann["MONTHLY"] == pytest.approx(ann["QUARTERLY"], rel=0.02)


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
    # 显式用 ts 模式：这里测的是平滑机制，单标的撑不起截面去均值
    s1 = carry_signals(f, d, smooth=1, mode="ts")["A"].iloc[300:]
    s21 = carry_signals(f, d, smooth=21, mode="ts")["A"].iloc[300:]
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


def test_curve_carry_recovers_known_slope():
    """从多月曲线拟合的年化 carry 必须与理论斜率一致。"""
    from qis.strategy.carry import curve_carry
    idx = pd.date_range("2024-01-01", periods=4, freq="B")
    horizon = pd.Series({"A": 1 / 12})          # 月度合约
    # contango：每月 log 价 +1% → 年化 carry = −12%
    up = {n: pd.DataFrame({"A": 100 * np.exp(0.01 * (n - 1))}, index=idx) for n in (1, 2, 3, 4)}
    assert curve_carry(up, horizon)["A"].iloc[-1] == pytest.approx(-0.12, rel=1e-9)
    # backwardation：每月 −2% → 年化 carry = +24%
    dn = {n: pd.DataFrame({"A": 100 * np.exp(-0.02 * (n - 1))}, index=idx) for n in (1, 2, 3, 4)}
    assert curve_carry(dn, horizon)["A"].iloc[-1] == pytest.approx(0.24, rel=1e-9)


def test_curve_carry_beats_two_point_spread_on_noise():
    """
    N 点回归斜率的方差约为两点差分的 1/N——这是 c1/c2 信噪比问题的根治办法。
    同一条真实曲线加同等噪声，4 点估计必须明显比 2 点稳。
    """
    from qis.strategy.carry import curve_carry
    rng = np.random.default_rng(7)
    idx = pd.date_range("2024-01-01", periods=300, freq="B")
    horizon = pd.Series({"A": 1 / 12})
    true_slope = -0.004
    legs = {}
    for n in (1, 2, 3, 4):
        lp = np.log(100) + true_slope * (n - 1) + rng.normal(0, 0.002, len(idx))
        legs[n] = pd.DataFrame({"A": np.exp(lp)}, index=idx)
    c4 = curve_carry(legs, horizon, use_legs=(1, 2, 3, 4))["A"]
    c2 = curve_carry(legs, horizon, use_legs=(1, 2), min_legs=2)["A"]
    # 两者都是无偏估计，差别在方差：4 点应至少稳一倍（理论 sqrt(1/N) 关系）
    assert c4.std() < c2.std() / 2
    assert c4.mean() == pytest.approx(-true_slope * 12, abs=0.005)   # 无偏


def test_curve_carry_requires_synchronous_quotes():
    """某月缺报价那天不能凑数：可用月份不足 min_legs 时应给 NaN。"""
    from qis.strategy.carry import curve_carry
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    legs = {n: pd.DataFrame({"A": [100.0 * n, 100.0 * n, 100.0 * n]}, index=idx)
            for n in (1, 2, 3)}
    legs[3].iloc[1] = np.nan
    legs[2].iloc[1] = np.nan
    out = curve_carry(legs, pd.Series({"A": 0.25}), use_legs=(1, 2, 3), min_legs=3)["A"]
    assert np.isnan(out.iloc[1])
    assert np.isfinite(out.iloc[0])


def test_curve_carry_prefers_deferred_legs():
    """
    c1 临近到期有收敛与流动性效应，是曲线上噪声最大的一点。
    有足够月份时应当去掉近月——去掉后估计明显更稳。
    """
    from qis.strategy.carry import curve_carry
    rng = np.random.default_rng(31)
    idx = pd.date_range("2024-01-01", periods=400, freq="B")
    horizon = pd.Series({"A": 1 / 12})
    slope = -0.004
    legs = {}
    for n in (1, 2, 3, 4):
        noise = 0.008 if n == 1 else 0.002          # 近月噪声大 4 倍
        lp = np.log(100) + slope * (n - 1) + rng.normal(0, noise, len(idx))
        legs[n] = pd.DataFrame({"A": np.exp(lp)}, index=idx)
    with_front = curve_carry(legs, horizon, use_legs=(1, 2, 3, 4))["A"]
    no_front = curve_carry(legs, horizon, use_legs=(2, 3, 4))["A"]
    assert no_front.std() < with_front.std()


def test_factor_exposure_and_neutralization():
    """
    因子暴露诊断：发现无意中背上的押注；中性化把它投影掉。
    （本池实测 trend/xsmom 都长期持正 carry 暴露，而它们没有表达过对 carry 的看法。）
    """
    from qis.portfolio.construction import (cross_sectional_z, factor_exposure,
                                            neutralize_factor)
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    cols = ["A", "B", "C", "D"]
    # 因子：A 最高（3.0）、D 最低（0.0）
    fac = pd.DataFrame({c: float(len(cols) - 1 - i) for i, c in enumerate(cols)}, index=idx)
    z = cross_sectional_z(fac)
    assert z.mean(axis=1).abs().max() < 1e-12          # 截面零均值
    # 权重偏向高因子标的 → 正暴露
    w = pd.DataFrame({"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1}, index=idx)
    assert factor_exposure(w, z).mean() > 0
    # 完全中性化后暴露归零
    wn = neutralize_factor(w, z, alpha=1.0)
    assert factor_exposure(wn, z).abs().max() < 1e-10
    # 部分中性化按比例缩减暴露
    half = factor_exposure(neutralize_factor(w, z, alpha=0.5), z).mean()
    assert half == pytest.approx(factor_exposure(w, z).mean() * 0.5, rel=1e-9)


def test_neutralization_preserves_unrelated_component():
    """中性化只投影掉因子方向，与因子正交的持仓不应被改动。"""
    from qis.portfolio.construction import cross_sectional_z, neutralize_factor
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    fac = pd.DataFrame({"A": 1.0, "B": 2.0, "C": 3.0}, index=idx)
    z = cross_sectional_z(fac)
    orth = pd.DataFrame({"A": 1.0, "B": -2.0, "C": 1.0}, index=idx)   # 与 z 正交
    assert float((orth * z).sum(axis=1).abs().max()) < 1e-12
    pd.testing.assert_frame_equal(neutralize_factor(orth, z), orth, atol=1e-12)
