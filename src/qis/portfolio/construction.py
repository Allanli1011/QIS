# -*- coding: utf-8 -*-
"""
权重构建。

  * normalize_gross : 归一化使 sum|w| = gross（默认 1）
  * inverse_vol     : 信号 ÷ 波动率（零相关假设下的风险平价近似）
  * vol_target_scale: 组合级波动目标缩放（含杠杆上限）
  * weight_band     : 无交易带
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


def vol_target_factor(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    target: float = 0.10,
    span: int = 40,
    ann_factor: int = 252,
    max_leverage: float = 3.0,
) -> tuple[pd.Series, pd.Series]:
    """
    返回 (截断后的缩放系数, 未截断的原始系数)。

    分开返回是为了让调用方能看出杠杆上限有没有一直绑定——上限一旦长期绑定，
    波动目标就名存实亡（策略实际跑的是固定杠杆，调 target 不会有任何反应）。
    """
    port_ret = (weights.shift(1) * returns.fillna(0.0)).sum(axis=1)
    vol_est = ewma_vol_series(port_ret, span=span, ann_factor=ann_factor)
    raw = target / vol_est.replace(0.0, np.nan)
    return raw.clip(upper=max_leverage), raw


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
    scale, _ = vol_target_factor(weights, returns, target, span, ann_factor, max_leverage)
    scaled = weights.mul(scale, axis=0)
    # 波动估计未就绪（前期 NaN）时不缩放
    return scaled.where(scale.notna(), weights)


def cap_binding_share(raw_factor: pd.Series, max_leverage: float,
                      index: pd.Index | None = None) -> float:
    """
    杠杆上限被触发的天数占比（接近 1 表示波动目标形同虚设，调 target 不会有反应）。

    raw_factor 取自 `vol_target_factor` 的第二个返回值。index 用于只统计回测窗口——
    权重是在完整历史上算的，把 start 之前的预热期也统计进去会让这个比例失真。
    """
    raw = raw_factor if index is None else raw_factor.reindex(index)
    raw = raw.dropna()
    if raw.empty:
        return float("nan")
    return float((raw > max_leverage).mean())


def weight_band(target: pd.DataFrame, thresh: float) -> pd.DataFrame:
    """
    无交易带（no-trade band）：目标权重与现行权重偏离 < thresh 的标的沿用旧权重。

    日频逆波动/波动目标会让权重每天微调，这种高频微调既贡献换手，
    又系统性地对短期反转做反向交易（费前也亏）。
    thresh=0 或空表时原样返回。

    注意：全程在 numpy 副本上做，绝不改写入参——早先版本用 `target.iloc[i]`
    取到的是 DataFrame 的视图，写回去会把调用方的权重矩阵一并改掉，
    导致"带 vs 不带"这类对照实验拿到同一份数据。
    """
    if thresh <= 0 or len(target) == 0:
        return target
    vals = target.to_numpy(dtype=float, copy=True)
    prev = vals[0].copy()
    for i in range(1, len(vals)):
        cur = vals[i].copy()
        keep = np.abs(cur - prev) < thresh
        cur[keep] = prev[keep]
        vals[i] = cur
        prev = cur
    return pd.DataFrame(vals, index=target.index, columns=target.columns)
