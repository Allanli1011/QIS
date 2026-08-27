# -*- coding: utf-8 -*-
"""
截面动量。

按过去 lookback 日收益在组内排名：多强空弱（组内去均值），再按波动缩放。
可整体排名（groups=None）或按资产类别分组排名。
"""
from __future__ import annotations

import pandas as pd

from qis.data.panel import ffill_prices, to_returns
from qis.portfolio.construction import inverse_vol
from qis.portfolio.risk import ewma_vol


def xsmom_signals(
    prices: pd.DataFrame,
    lookback: int = 126,
    groups: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    截面排名信号，落在 [-1, 1]，且各组内均值严格为 0。

      groups: instrument → 组名；None 表示全体一组。
    """
    px = ffill_prices(prices)
    mom = px / px.shift(lookback) - 1.0

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
    lookback: int = 126,
    groups: dict[str, str] | None = None,
    vol_span: int = 40,
    gross: float = 1.0,
) -> pd.DataFrame:
    sig = xsmom_signals(prices, lookback=lookback, groups=groups)
    vol = ewma_vol(to_returns(prices), span=vol_span)
    return inverse_vol(sig, vol, gross=gross)
