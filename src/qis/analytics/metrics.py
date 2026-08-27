# -*- coding: utf-8 -*-
"""绩效指标。输入均为日收益序列（小数）。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ann_return(returns: pd.Series, ann_factor: int = 252) -> float:
    """年化复合收益。"""
    r = returns.dropna()
    if len(r) == 0:
        return float("nan")
    total = (1.0 + r).prod()
    years = len(r) / ann_factor
    if years <= 0 or total <= 0:
        return float("nan")
    return float(total ** (1.0 / years) - 1.0)


def ann_vol(returns: pd.Series, ann_factor: int = 252) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    return float(r.std(ddof=1) * np.sqrt(ann_factor))


def sharpe(returns: pd.Series, ann_factor: int = 252) -> float:
    """年化 Sharpe（rf=0，用算术均值/波动）。"""
    r = returns.dropna()
    if len(r) < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * np.sqrt(ann_factor))


def sortino(returns: pd.Series, ann_factor: int = 252) -> float:
    """年化 Sortino（下行波动，阈值 0）。"""
    r = returns.dropna()
    downside = r[r < 0]
    if len(r) < 2 or len(downside) == 0:
        return float("nan")
    dd = np.sqrt((downside**2).mean())
    if dd == 0:
        return float("nan")
    return float(r.mean() / dd * np.sqrt(ann_factor))


def drawdown(equity: pd.Series) -> pd.Series:
    """回撤序列（≤0）。"""
    peak = equity.cummax()
    return equity / peak - 1.0


def max_drawdown(equity: pd.Series) -> float:
    if len(equity) == 0:
        return float("nan")
    return float(drawdown(equity).min())


def calmar(returns: pd.Series, equity: pd.Series | None = None, ann_factor: int = 252) -> float:
    """Calmar = 年化收益 / |最大回撤|。"""
    eq = equity if equity is not None else (1.0 + returns.fillna(0.0)).cumprod()
    mdd = max_drawdown(eq)
    if mdd == 0 or np.isnan(mdd):
        return float("nan")
    return float(ann_return(returns, ann_factor) / abs(mdd))


def hit_ratio(returns: pd.Series) -> float:
    """胜率：日收益 > 0 的比例（剔除无持仓的 0 收益日）。"""
    r = returns.dropna()
    r = r[r != 0]
    if len(r) == 0:
        return float("nan")
    return float((r > 0).mean())


def ann_turnover(turnover: pd.Series, ann_factor: int = 252) -> float:
    """年化单边换手（日均 × 年化因子）。"""
    t = turnover.dropna()
    if len(t) == 0:
        return float("nan")
    return float(t.mean() * ann_factor)


def summary(
    returns: pd.Series,
    turnover: pd.Series | None = None,
    ann_factor: int = 252,
) -> pd.Series:
    """指标汇总。"""
    r = returns.fillna(0.0)
    equity = (1.0 + r).cumprod()
    out = {
        "ann_return": ann_return(r, ann_factor),
        "ann_vol": ann_vol(r, ann_factor),
        "sharpe": sharpe(r, ann_factor),
        "sortino": sortino(r, ann_factor),
        "max_drawdown": max_drawdown(equity),
        "calmar": calmar(r, equity, ann_factor),
        "hit_ratio": hit_ratio(r),
        "skew": float(r.skew()) if len(r) > 2 else float("nan"),
        "kurtosis": float(r.kurt()) if len(r) > 3 else float("nan"),
        "n_days": int(r.shape[0]),
    }
    if turnover is not None:
        out["ann_turnover"] = ann_turnover(turnover, ann_factor)
    return pd.Series(out)
