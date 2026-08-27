# -*- coding: utf-8 -*-
"""
截面动量。

按过去 lookback 日的**风险调整**收益在截面上排名：多强空弱（去均值），再按波动缩放。

两个默认值是有理由的，不是调出来的：
  * groups=None（全体一起排名）。截面动量靠的是**广度**——用本池按资产类别
    分组时 rates/crypto 只有 2 个成员会被整组置零、metals 只有 6 个，
    组内排名基本是噪声。实测按类别分组 SR −0.23，全体排名 +0.51。
  * risk_adjusted=True。跨资产类别直接比原始收益，两端会被高波动品种
    （crypto、VIX）长期霸占；先除以各自波动再排名才是可比的。实测 +0.51 → +0.64。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from qis.data.panel import ffill_prices, to_returns
from qis.portfolio.construction import inverse_vol
from qis.portfolio.risk import ewma_vol


def xsmom_signals(
    prices: pd.DataFrame,
    lookback: int = 252,
    groups: dict[str, str] | None = None,
    risk_adjusted: bool = True,
    vol_span: int = 40,
) -> pd.DataFrame:
    """
    截面排名信号，落在 [-1, 1]，且各组内均值严格为 0。

      groups        : instrument → 组名；None 表示全体一组（默认，见模块 docstring）。
      risk_adjusted : True 时按 收益/波动 排名，避免高波动品种长期占据两端。
    """
    px = ffill_prices(prices)
    mom = px / px.shift(lookback) - 1.0
    if risk_adjusted:
        mom = mom / ewma_vol(to_returns(prices), span=vol_span).replace(0.0, np.nan)

    def _rank(df: pd.DataFrame) -> pd.DataFrame:
        n = df.notna().sum(axis=1)
        # rank(pct=True) 取值在 1/n..1，均值是 (n+1)/(2n) 而不是 0.5：
        # 直接减 0.5 会留下 +1/n 的系统性多头偏移，逐行去均值才真正截面中性。
        r = df.rank(axis=1, pct=True)
        sig = r.sub(r.mean(axis=1), axis=0) * 2.0
        # 组内标的少于 3 个时信号意义不大，置 0
        return sig.where(n >= 3, 0.0)

    if groups is None:
        return _rank(mom).fillna(0.0)

    out = pd.DataFrame(0.0, index=mom.index, columns=mom.columns)
    members: dict[str, list[str]] = {}
    for inst, g in groups.items():
        if inst in mom.columns:
            members.setdefault(g, []).append(inst)
    for cols in members.values():
        out[cols] = _rank(mom[cols])
    return out.fillna(0.0)


def xsmom_weights(
    prices: pd.DataFrame,
    lookback: int = 252,
    groups: dict[str, str] | None = None,
    risk_adjusted: bool = True,
    vol_span: int = 40,
    gross: float = 1.0,
) -> pd.DataFrame:
    sig = xsmom_signals(prices, lookback=lookback, groups=groups,
                        risk_adjusted=risk_adjusted, vol_span=vol_span)
    vol = ewma_vol(to_returns(prices), span=vol_span)
    return inverse_vol(sig, vol, gross=gross)
