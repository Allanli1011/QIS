# -*- coding: utf-8 -*-
"""
期限结构 carry。

原始 carry_i = front_i / deferred_i - 1（backwardation 为正）。
不同资产的结构性升贴水差异大（股指/债券长期 contango），部分标的
（天然气、农产品）价差的季节性强，因此默认用时序模式：
信号 = 自身价差 − 其滚动均值（见 carry_signals 的 mode 参数）。

输入为两个价格矩阵：
  front    : date × instrument 近月连续价（列名 = 标的名）
  deferred : date × instrument 远月连续价（与 front 同列名，缺腿标的可缺列）
  groups   : instrument → 组名；None 表示全体一组（仅 mode="xs" 使用）
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from qis.data.panel import ffill_prices, to_returns
from qis.portfolio.construction import inverse_vol, normalize_gross
from qis.portfolio.risk import ewma_vol


def carry_raw(front: pd.DataFrame, deferred: pd.DataFrame,
              horizon: pd.Series | None = None) -> pd.DataFrame:
    """
    carry = front/deferred - 1（无腿的标的列保持 NaN）。

    horizon 给出每个标的两腿到期日的间隔（年）时，把 carry **年化**：
    月度合约的 c1/c2 只跨 1 个月、季度合约跨 3 个月，不年化就没法在截面上比，
    时序上合约周期一变信号水平也会跟着跳。见 qis.data.roll.contract_horizon。
    """
    common = [c for c in front.columns if c in deferred.columns]
    f, d = front[common], deferred[common]
    # 只在**两腿同日都有真实报价**时计价差。两腿各自 ffill 再相除是错的：
    # 它们的停牌日未必重合（SILVER 有 30.6% 的日子不一致），
    # 那样会把"旧的 c1 / 新的 c2"凑成一个价差，凭空造出期限结构变动
    # ——实测价差的日变动被放大 32%，而 carry 信号对这种噪声极其敏感。
    spread = (f / d - 1.0).where(f.notna() & d.notna())
    spread = ffill_prices(spread)
    if horizon is not None:
        h = horizon.reindex(spread.columns)
        spread = spread.div(h.where(h > 0), axis=1)
    return spread


def curve_carry(
    curves: "dict[int, pd.DataFrame]",
    horizon: pd.Series,
    use_legs: tuple[int, ...] = (1, 2, 3, 4),
    min_legs: int = 3,
) -> pd.DataFrame:
    """
    从多月合约曲线用**回归斜率**估年化 carry。

        log F_i = a + b·t_i ,   t_i = (i−1)·h    （h = 相邻合约到期间隔，年）
        carry   = −b

    曲线下倾（backwardation）时 b<0、carry>0——持有近月、沿曲线往上滚可赚 roll yield。

    为什么用回归而不是两点价差：c1/c2 是两个带噪报价的差分，信噪比极差
    （本池实测 corr(信噪比, IC) = −0.53，噪声大的品种 carry 系统性做反）。
    N 点回归斜率的方差约为两点差分的 1/N，这是该问题的根治办法而不是缓解。

    只用**当日有真实报价**的月份，且至少 min_legs 个——否则不同日期的报价
    会被凑成一条曲线、造出虚假斜率。
    """
    legs = [n for n in use_legs if n in curves]
    if len(legs) < min_legs:
        return pd.DataFrame()

    idx = curves[legs[0]].index
    cols = curves[legs[0]].columns
    for n in legs[1:]:
        idx = idx.union(curves[n].index)
        cols = cols.union(curves[n].columns)

    n_obs = None
    sx = sy = sxx = sxy = None
    for n in legs:
        y = np.log(curves[n].reindex(index=idx, columns=cols))
        m = y.notna()
        x = float(n - 1)                       # 合约序号（相对），乘 h 才是年
        y0 = y.fillna(0.0)
        mi = m.astype(float)
        n_obs = mi if n_obs is None else n_obs + mi
        sx = x * mi if sx is None else sx + x * mi
        sy = y0 if sy is None else sy + y0
        sxx = (x * x) * mi if sxx is None else sxx + (x * x) * mi
        sxy = x * y0 if sxy is None else sxy + x * y0

    denom = n_obs * sxx - sx * sx
    slope = (n_obs * sxy - sx * sy) / denom.where(denom > 0)
    slope = slope.where(n_obs >= min_legs)

    h = horizon.reindex(slope.columns)
    return (-slope).div(h.where(h > 0), axis=1)


def carry_signals(
    front: pd.DataFrame,
    deferred: pd.DataFrame,
    groups: dict[str, str] | None = None,
    mode: str = "xs",
    lookback: int = 252,
    smooth: int = 21,
    horizon: pd.Series | None = None,
) -> pd.DataFrame:
    """
    carry 信号。

      mode="xs": 截面模式（默认）——组内去均值（多 carry 高者、空 carry 低者）。
                 **前提是先年化**（传 horizon）：不年化时月度合约的 c1/c2 只跨
                 1 个月、季度合约跨 3 个月，截面上比的是合约周期而不是 carry。
                 本池实测 ts −0.97 / xs −0.35。
      mode="ts": 时序模式——自身价差减其 lookback 日滚动均值。
                 剔除结构性升贴水，但也丢掉了截面上的相对信息。
    """
    raw = carry_raw(front, deferred, horizon=horizon)
    return signals_from_raw(raw, groups=groups, mode=mode, lookback=lookback,
                            smooth=smooth, columns=front.columns)


def signals_from_raw(
    raw: pd.DataFrame,
    groups: dict[str, str] | None = None,
    mode: str = "xs",
    lookback: int = 252,
    smooth: int = 21,
    columns: pd.Index | None = None,
) -> pd.DataFrame:
    """
    已经算好的**年化 carry** → 交易信号（平滑 + 去均值）。

    raw 可以来自两点价差（carry_raw）也可以来自整条曲线的回归斜率（curve_carry）；
    后者信噪比高得多，见 curve_carry 的 docstring。
    """
    if smooth > 1:
        # 先平滑价差再取信号。c1/c2 的日度测量噪声（非同步收盘、买卖价跳动）
        # 会让"carry 高"往往只是 c1 当天被推高了、次日回落，
        # 策略于是系统性追高：实测 corr(信噪比, IC) = −0.53，
        # 高噪声组 IC −0.040 而低噪声组 −0.008。平滑直接打掉这个机制。
        raw = raw.rolling(smooth, min_periods=max(2, smooth // 2)).mean()

    if mode == "ts":
        demeaned = raw - raw.rolling(lookback, min_periods=lookback // 2).mean()
    elif mode == "xs":
        def _demean(df: pd.DataFrame) -> pd.DataFrame:
            n = df.notna().sum(axis=1)
            sig = df.sub(df.mean(axis=1), axis=0)
            return sig.where(n >= 2, 0.0)

        if groups is None:
            demeaned = _demean(raw)
        else:
            demeaned = pd.DataFrame(index=raw.index, columns=raw.columns, dtype=float)
            members: dict[str, list[str]] = {}
            for inst, g in groups.items():
                if inst in raw.columns:
                    members.setdefault(g, []).append(inst)
            for cols in members.values():
                demeaned[cols] = _demean(raw[cols])
    else:
        raise ValueError(f"unknown mode: {mode}")

    cols = columns if columns is not None else raw.columns
    out = pd.DataFrame(0.0, index=raw.index, columns=cols)
    common = [c for c in demeaned.columns if c in out.columns]
    out[common] = demeaned[common].fillna(0.0)
    return out


def carry_weights(
    front: pd.DataFrame,
    deferred: pd.DataFrame,
    groups: dict[str, str] | None = None,
    mode: str = "ts",
    lookback: int = 252,
    smooth: int = 21,
    horizon: pd.Series | None = None,
    vol_span: int = 40,
    gross: float = 1.0,
) -> pd.DataFrame:
    sig = carry_signals(front, deferred, groups=groups, mode=mode,
                        lookback=lookback, smooth=smooth, horizon=horizon)
    vol = ewma_vol(to_returns(front), span=vol_span)
    w = inverse_vol(sig, vol, gross=gross)
    # 无 carry 腿的标的权重保持 0
    no_leg = [c for c in front.columns if c not in deferred.columns]
    if no_leg:
        w[no_leg] = 0.0
        w = normalize_gross(w, gross)
    return w
