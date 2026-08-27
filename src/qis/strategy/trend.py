# -*- coding: utf-8 -*-
"""
时序动量 / 趋势。

多窗口动量符号投票（-1..1），逐标的按 EWMA 波动缩放后归一化毛敞口。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from qis.portfolio.construction import inverse_vol
from qis.portfolio.risk import ewma_vol


def trend_signals(
    prices: pd.DataFrame,
    lookbacks: tuple[int, ...] = (21, 63, 126, 252),
    smooth: int = 5,
) -> pd.DataFrame:
    """每个窗口取价格涨跌符号，等权平均后做 smooth 日均值（降换手），落在 [-1, 1]。"""
    sig = None
    for lb in lookbacks:
        s = np.sign(prices / prices.shift(lb) - 1.0)
        sig = s if sig is None else sig + s
    sig = sig / len(lookbacks)
    if smooth > 1:
        sig = sig.rolling(smooth, min_periods=1).mean()
    return sig.fillna(0.0)


def trend_weights(
    prices: pd.DataFrame,
    lookbacks: tuple[int, ...] = (21, 63, 126, 252),
    vol_span: int = 40,
    smooth: int = 5,
    gross: float = 1.0,
) -> pd.DataFrame:
    """趋势信号 ÷ EWMA 波动，归一化到 sum|w| = gross。"""
    sig = trend_signals(prices, lookbacks, smooth=smooth)
    vol = ewma_vol(prices.pct_change(fill_method=None), span=vol_span)
    return inverse_vol(sig, vol, gross=gross)
