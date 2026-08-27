# -*- coding: utf-8 -*-
"""
时序动量 / 趋势。

默认用 **EWMA 快慢交叉**（EWMAC，CTA 的标准构造）：多组快慢均线之差，
各自按该标的的波动归一化后截尾，等权平均，再做短期平滑降换手。

为什么不是原来的"多窗口符号投票"
--------------------------------
符号投票只取 sign(p_t / p_{t-lb} - 1)，把强度信息全扔了——21 天涨 0.1%
和涨 30% 记同一票，于是信号在弱趋势里频繁翻面、在强趋势里又给不出更大仓位。
本池实测（gross=3，其余条件相同）：

    符号投票 (21,63,126,252)   SR +0.70   子样本 +0.86/+0.55   换手 151
    强度 z + 截尾               SR +0.92   子样本 +1.02/+0.81   换手 112
    EWMAC（本实现）             SR +0.96   子样本 +0.91/+0.99   换手  87

三项全面更优，且对参数不敏感（7 组快慢对全部落在 +0.69~+1.01，子样本均为正），
是平台而不是刀刃。`method="sign"` 保留旧构造以便对照。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from qis.data.panel import ffill_prices, to_returns
from qis.portfolio.construction import inverse_vol
from qis.portfolio.risk import ewma_vol

# 快慢 span 阶梯（1:3 比例的几何序列，CTA 惯例），不是搜出来的
EWMAC_SPANS: tuple[tuple[int, int], ...] = ((8, 24), (16, 48), (32, 96), (64, 192))


def ewmac_signals(
    prices: pd.DataFrame,
    spans: tuple[tuple[int, int], ...] = EWMAC_SPANS,
    smooth: int = 5,
    clip: float = 2.0,
    vol_span: int = 40,
) -> pd.DataFrame:
    """
    EWMA 快慢交叉信号，落在 [-clip, clip]。

    每组 (fast, slow)：
      raw = EWMA_fast(P) − EWMA_slow(P)                价格单位的趋势强度
      z   = raw / (日波动 × sqrt(slow))                 按该标的自身的波动尺度归一
    随机游走下 raw 的量级正是 日波动 × sqrt(slow)，所以 z 是无量纲的、跨标的可比。
    截尾到 ±clip 限制单一标的的信心上限（风控，不是为了提高 Sharpe——
    实测不截尾 SR 反而略高 +1.01，但单标的仓位会失控）。
    """
    px = ffill_prices(prices)
    vol = ewma_vol(to_returns(prices), span=vol_span)
    daily_px_vol = (px * vol / np.sqrt(252.0)).replace(0.0, np.nan)

    sig = None
    for fast, slow in spans:
        raw = px.ewm(span=fast, min_periods=fast).mean() - px.ewm(span=slow, min_periods=slow).mean()
        z = (raw / daily_px_vol / np.sqrt(slow)).clip(-clip, clip)
        sig = z if sig is None else sig + z
    sig = sig / len(spans)
    if smooth > 1:
        sig = sig.rolling(smooth, min_periods=1).mean()
    return sig.fillna(0.0)


def sign_vote_signals(
    prices: pd.DataFrame,
    lookbacks: tuple[int, ...] = (21, 63, 126, 252),
    smooth: int = 5,
) -> pd.DataFrame:
    """旧构造：多窗口动量符号投票（-1..1）。保留以便对照，见模块 docstring。"""
    px = ffill_prices(prices)
    sig = None
    for lb in lookbacks:
        s = np.sign(px / px.shift(lb) - 1.0)
        sig = s if sig is None else sig + s
    sig = sig / len(lookbacks)
    if smooth > 1:
        sig = sig.rolling(smooth, min_periods=1).mean()
    return sig.fillna(0.0)


def trend_signals(
    prices: pd.DataFrame,
    method: str = "ewmac",
    spans: tuple[tuple[int, int], ...] = EWMAC_SPANS,
    lookbacks: tuple[int, ...] = (21, 63, 126, 252),
    smooth: int = 5,
    clip: float = 2.0,
    vol_span: int = 40,
) -> pd.DataFrame:
    """趋势信号。method="ewmac"（默认）或 "sign"（旧的符号投票）。"""
    if method == "ewmac":
        return ewmac_signals(prices, spans=spans, smooth=smooth, clip=clip, vol_span=vol_span)
    if method == "sign":
        return sign_vote_signals(prices, lookbacks=lookbacks, smooth=smooth)
    raise ValueError(f"unknown method: {method}")


def trend_weights(
    prices: pd.DataFrame,
    method: str = "ewmac",
    spans: tuple[tuple[int, int], ...] = EWMAC_SPANS,
    lookbacks: tuple[int, ...] = (21, 63, 126, 252),
    vol_span: int = 40,
    smooth: int = 5,
    clip: float = 2.0,
    gross: float = 1.0,
) -> pd.DataFrame:
    """趋势信号 ÷ EWMA 波动，归一化到 sum|w| = gross。"""
    sig = trend_signals(prices, method=method, spans=spans, lookbacks=lookbacks,
                        smooth=smooth, clip=clip, vol_span=vol_span)
    vol = ewma_vol(to_returns(prices), span=vol_span)
    return inverse_vol(sig, vol, gross=gross)
