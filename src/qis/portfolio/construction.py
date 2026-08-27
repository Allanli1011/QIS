# -*- coding: utf-8 -*-
"""
权重构建。

  * normalize_gross : 归一化使 sum|w| = gross（默认 1）
  * inverse_vol     : 信号 ÷ 波动率（零相关假设下的风险平价近似）
  * vol_target_scale: 组合级波动目标缩放（含杠杆上限）
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from qis.portfolio.risk import ewma_vol_series


def normalize_gross(weights: pd.DataFrame, gross: float = 1.0) -> pd.DataFrame:
    """逐日归一化：sum|w_i| = gross。全零行保持全零。"""
    total = weights.abs().sum(axis=1)
    scale = (gross / total).where(total > 0)
    return weights.mul(scale, axis=0).fillna(0.0)


def inverse_vol(signals: pd.DataFrame, vol: pd.DataFrame, gross: float = 1.0) -> pd.DataFrame:
    """w_i ∝ signal_i / vol_i，再按 gross 归一化。"""
    raw = signals / vol.replace(0.0, np.nan)
    return normalize_gross(raw.replace([np.inf, -np.inf], np.nan).fillna(0.0), gross)


def vol_target_scale(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    target: float = 0.10,
    span: int = 40,
    ann_factor: int = 252,
    max_leverage: float = 3.0,
) -> pd.DataFrame:
    """
    组合级波动目标：按组合历史收益（昨日权重 × 收益）的 EWMA 波动缩放整体权重。

    仅用 t 日及之前的信息，无前视。
    """
    port_ret = (weights.shift(1) * returns.fillna(0.0)).sum(axis=1)
    vol_est = ewma_vol_series(port_ret, span=span, ann_factor=ann_factor)
    scale = (target / vol_est.replace(0.0, np.nan)).clip(upper=max_leverage)
    scaled = weights.mul(scale, axis=0)
    # 波动估计未就绪（前期 NaN）时不缩放
    return scaled.where(scale.notna(), weights)


def weight_band(target: pd.DataFrame, thresh: float) -> pd.DataFrame:
    """
    无交易带（no-trade band）：目标权重与现行权重偏离 < thresh 的标的沿用旧权重。

    日频逆波动/波动目标会让权重每天微调，这种高频微调既贡献换手，
    又系统性地对短期反转做反向交易（费前也亏）。实证（本池 2010 起，trend）：
    band=0 → 费前 Sharpe +0.17；band=0.03 → +0.63；band=0.05 → +0.83。
    thresh=0 时原样返回。
    """
    if thresh <= 0:
        return target
    out = target.copy()
    prev = target.iloc[0].copy()
    for i in range(1, len(target)):
        cur = target.iloc[i]
        keep = (cur - prev).abs() < thresh
        cur[keep] = prev[keep]
        out.iloc[i] = cur
        prev = cur
    return out
