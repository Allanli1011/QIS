# -*- coding: utf-8 -*-
"""
向量化日频回测引擎。

约定（防前视）：
  * weights 为 t 日收盘确定的目标权重；
  * t 日权重作用于 t→t+1 的收益（引擎内 shift(1)）；
  * 换手 = |t 日目标权重 − 昨日持仓经今日价格漂移后的权重|；
  * 成本 = 换手 × 成本率，从当日收益中扣减。

价格矩阵按并集交易日历给出，收益一律走 `qis.data.panel.to_returns`：
标的自身假期当日收益为 0（跨假期涨跌落到下一交易日），
仅上市前 / 退市后视为不可交易——那些日子持仓与目标权重同时清零，
不产生换手（详见 panel 模块 docstring）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from qis.data.panel import ffill_prices


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
    returns: pd.DataFrame   # 引擎实际使用的标的日收益（供归因复用，勿重算）


def run_backtest(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    cost_bps: float | Mapping[str, float] = 0.0,
    roll_mask: pd.DataFrame | None = None,
) -> BacktestResult:
    """
    参数
      prices   : date × instrument 收盘价矩阵（并集日历，允许假期缺口）
      weights  : date × instrument 目标权重（t 日收盘生成；NaN 视为 0）
      cost_bps : 单边成本率（bps）。标量为全标的统一值；Mapping 为 instrument → bps
      roll_mask: date × instrument 布尔，True 表示该日为该标的的换月交易日：
                 持仓需平旧开新，按 |持仓| × 2 倍单边费率计换月成本
    """
    prices = prices.sort_index()
    # 存续区间内前值填充：假期沿用上一收盘（当日收益 0，跨假期涨跌落到下一交易日）
    filled = ffill_prices(prices)
    rets = filled.pct_change(fill_method=None)
    weights = weights.reindex(index=rets.index, columns=rets.columns).fillna(0.0)

    # t 日持仓 = t-1 日目标权重
    w_held = weights.shift(1).fillna(0.0)

    # 可交易 = 当日有价（含建仓首日）。上市前 / 退市后既不能持仓，也不该产生换手：
    # 两者必须同时清零——只清持仓会让 |目标 − 漂移| 把整个仓位记成一轮虚构换手。
    tradable = filled.notna()
    w_held = w_held.where(tradable, 0.0)
    weights = weights.where(tradable, 0.0)
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
        rm = roll_mask.reindex(index=rets.index, columns=rets.columns).fillna(False).astype(bool)
        roll_cost = (w_held.abs() * rates * 2.0).where(rm, 0.0).sum(axis=1)
        cost = cost + roll_cost

    net = gross - cost
    equity = (1.0 + net).cumprod()
    gross_equity = (1.0 + gross).cumprod()

    return BacktestResult(
        gross=gross, net=net, equity=equity, gross_equity=gross_equity,
        turnover=turnover, cost=cost, weights=w_held, returns=rets,
    )


def _rate_frame(
    cost_bps: float | Mapping[str, float],
    columns: pd.Index,
    index: pd.Index,
) -> pd.DataFrame:
    """把成本率参数展开成 date × instrument 的比率矩阵（小数）。"""
    if isinstance(cost_bps, Mapping):
        rates = pd.Series({c: float(cost_bps.get(c, 0.0)) for c in columns}, dtype=float)
    else:
        rates = pd.Series(float(cost_bps), index=columns, dtype=float)
    rates = rates.fillna(0.0) / 1e4
    return pd.DataFrame(
        {c: rates[c] for c in columns}, index=index, dtype=float,
    )
