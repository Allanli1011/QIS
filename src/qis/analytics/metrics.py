# -*- coding: utf-8 -*-
"""绩效指标。输入均为日收益序列（小数）。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def infer_ann_factor(returns: pd.Series, default: float = 252) -> float:
    """
    从时间索引推断年化因子（观测数 / 日历年）。

    多市场并集日历每年约 260+ 个观测，硬写 252 会让波动与 Sharpe 系统性偏低。
    非时间索引、或样本不足一年时退回 default。
    """
    idx = returns.dropna().index
    if not isinstance(idx, pd.DatetimeIndex) or len(idx) < 2:
        return float(default)
    years = (idx[-1] - idx[0]).days / 365.25
    if years < 1.0:
        return float(default)
    return float(len(idx) / years)


def _years(r: pd.Series, ann_factor: float) -> float:
    """样本长度（年）：有时间索引就用日历跨度，否则退回 len / ann_factor。"""
    idx = r.index
    if isinstance(idx, pd.DatetimeIndex) and len(idx) > 1:
        days = (idx[-1] - idx[0]).days
        if days > 0:
            return days / 365.25
    return len(r) / ann_factor


def ann_return(returns: pd.Series, ann_factor: float = 252) -> float:
    """
    年化复合收益。

    有时间索引时按**日历跨度**折算：并集日历的行数多于 252/年，
    用 len/252 当年数会把 16.6 年的样本当成 17.4 年，年化收益被系统性压低。
    """
    r = returns.dropna()
    if len(r) == 0:
        return float("nan")
    total = (1.0 + r).prod()
    years = _years(r, ann_factor)
    if years <= 0 or total <= 0:
        return float("nan")
    return float(total ** (1.0 / years) - 1.0)


def ann_vol(returns: pd.Series, ann_factor: float = 252) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    return float(r.std(ddof=1) * np.sqrt(ann_factor))


def sharpe(returns: pd.Series, ann_factor: float = 252) -> float:
    """年化 Sharpe（rf=0，用算术均值/波动）。"""
    r = returns.dropna()
    if len(r) < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * np.sqrt(ann_factor))


def sortino(returns: pd.Series, ann_factor: float = 252) -> float:
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


def calmar(returns: pd.Series, equity: pd.Series | None = None, ann_factor: float = 252) -> float:
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


def ann_turnover(turnover: pd.Series, ann_factor: float = 252) -> float:
    """年化单边换手（日均 × 年化因子）。"""
    t = turnover.dropna()
    if len(t) == 0:
        return float("nan")
    return float(t.mean() * ann_factor)


def summary(
    returns: pd.Series,
    turnover: pd.Series | None = None,
    ann_factor: float | None = None,
) -> pd.Series:
    """指标汇总。ann_factor=None 时从时间索引推断实际观测频率。"""
    r = returns.fillna(0.0)
    if ann_factor is None:
        ann_factor = infer_ann_factor(r)
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
        "ann_factor": float(ann_factor),
    }
    if turnover is not None:
        out["ann_turnover"] = ann_turnover(turnover, ann_factor)
    return pd.Series(out)
