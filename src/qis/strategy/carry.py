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

    out = pd.DataFrame(0.0, index=front.index, columns=front.columns)
    out[demeaned.columns] = demeaned.fillna(0.0)
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
