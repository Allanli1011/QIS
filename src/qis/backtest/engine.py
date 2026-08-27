# -*- coding: utf-8 -*-
"""
向量化日频回测引擎。

约定（防前视）：
  * weights 为 t 日收盘确定的目标权重；
  * t 日权重作用于 t→t+1 的收益（引擎内 shift(1)）；
  * 换手 = |t 日目标权重 − 昨日持仓经今日价格漂移后的权重|；
  * 成本 = 换手 × 成本率，从当日收益中扣减。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import pandas as pd


@dataclass
class BacktestResult:
    """回测输出。所有序列以交易日为索引。"""

    gross: pd.Series        # 费前日收益
    net: pd.Series          # 费后日收益
    equity: pd.Series       # 费后净值（起始 1.0）
    gross_equity: pd.Series # 费前净值
    turnover: pd.Series     # 每日单边换手（sum |Δw|）
    cost: pd.Series         # 每日成本（收益口径）
    weights: pd.DataFrame   # 每日实际持仓权重（t 日持仓 = 输入权重的 t-1 值）


def run_backtest(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    cost_bps: float | Mapping[str, float] = 0.0,
    roll_mask: pd.DataFrame | None = None,
) -> BacktestResult:
    """
    参数
      prices   : date × instrument 收盘价矩阵
      weights  : date × instrument 目标权重（t 日收盘生成；NaN 视为 0）
      cost_bps : 单边成本率（bps）。标量为全标的统一值；Mapping 为 instrument → bps
      roll_mask: date × instrument 布尔，True 表示该日为该标的的换月交易日：
                 持仓需平旧开新，按 |持仓| × 2 倍单边费率计换月成本
    """
    prices = prices.sort_index()
    rets = prices.pct_change(fill_method=None)
    weights = weights.reindex(index=rets.index, columns=rets.columns).fillna(0.0)

    # t 日持仓 = t-1 日目标权重
    w_held = weights.shift(1).fillna(0.0)

    # 收益中的 NaN（标的尚未上市/缺数据）按 0 处理，并清零对应持仓
    valid = rets.notna()
    w_held = w_held.where(valid, 0.0)
    rets = rets.fillna(0.0)

    gross = (w_held * rets).sum(axis=1)

    # 持仓经当日价格漂移后的收盘权重：w_held*(1+r)/(1+gross)
    drift = w_held * (1.0 + rets)
    denom = (1.0 + gross).where((1.0 + gross) != 0.0)
    drift = drift.div(denom, axis=0).fillna(0.0)

    # t 日收盘调仓：目标权重 - 漂移权重（首日 drift=0，建仓计入换手）
    delta = (weights - drift).abs()
    turnover = delta.sum(axis=1)

    rates = _rate_frame(cost_bps, rets.columns, rets.index)
    cost = (delta * rates).sum(axis=1)

    # 换月成本：平旧 + 开新 = 2 倍单边
    if roll_mask is not None:
        rm = roll_mask.reindex(index=rets.index, columns=rets.columns).fillna(False)
        roll_cost = (w_held.abs() * rates * 2.0).where(rm, 0.0).sum(axis=1)
        cost = cost + roll_cost

    net = gross - cost
    equity = (1.0 + net).cumprod()
    gross_equity = (1.0 + gross).cumprod()

    return BacktestResult(
        gross=gross, net=net, equity=equity, gross_equity=gross_equity,
        turnover=turnover, cost=cost, weights=w_held,
    )


def _rate_frame(
    cost_bps: float | Mapping[str, float],
    columns: pd.Index,
    index: pd.Index,
) -> pd.DataFrame:
    """把成本率参数展开成 date × instrument 的比率矩阵（小数）。"""
    if isinstance(cost_bps, Mapping):
        rates = pd.Series({c: float(cost_bps.get(c, 0.0)) for c in columns})
    else:
        rates = pd.Series(float(cost_bps), index=columns)
    rates = rates.fillna(0.0) / 1e4
    return pd.DataFrame(
        {c: rates[c] for c in columns}, index=index,
    )
