# -*- coding: utf-8 -*-
"""
连续合约换月处理。

LSEG 的 c1/c2 连续价是换月日直接拼接的，未做复权：换月日 c1 的收益是
新旧合约的价差（contango 时为幻影正收益、backwardation 时为幻影负收益），
并非持仓者的真实损益。直接用 c1 算收益会让回测系统性失真。

换月窗口识别（分两层，按数据可用性）：
  1. v1 背离法（优先）：LSEG 的 v1 序列按成交量切换拼接。c1 与 v1 引用同一
     合约时二者日收益完全相等；v1 先切到新合约后到 c1 机械切换前的窗口内，
     二者背离。该窗口即 LSEG 官方成交量换月点（仅 CME 集团/JGB 等有 v1）。
  2. 成交量规则（回退）：远月成交量 > 近月成交量的日子。

窗口内的收益用远月（c2）收益替代——换月后持有的已是远月；
窗口起点日为换月交易日（平旧开新），供引擎计换月成本。
"""
from __future__ import annotations

import pandas as pd


def v1_roll_window(c1_close: pd.Series, v1_close: pd.Series, tol: float = 1e-4) -> pd.Series:
    """v1 背离窗口：|c1 日收益 − v1 日收益| > tol 的日子（二者引用不同合约）。"""
    c, v = c1_close.align(v1_close, join="inner")
    r1, rv = c.pct_change(fill_method=None), v.pct_change(fill_method=None)
    div = ((r1 - rv).abs() > tol) & r1.notna() & rv.notna()
    return div.reindex(c1_close.index, fill_value=False)


def volume_roll_window(front_vol: pd.Series, defer_vol: pd.Series) -> pd.Series:
    """成交量规则窗口：远月成交量 > 近月成交量的日子。"""
    f, d = front_vol.align(defer_vol, join="left")
    return (d > f).fillna(False)


def roll_starts(window: pd.Series) -> pd.Series:
    """换月交易日：每个连续窗口的第一天（平旧开新发生日）。"""
    return (window & ~window.shift(1, fill_value=False)).fillna(False)


def adjusted_returns(
    front_close: pd.Series,
    defer_close: pd.Series,
    front_vol: pd.Series | None = None,
    defer_vol: pd.Series | None = None,
    v1_close: pd.Series | None = None,
) -> pd.Series:
    """
    单标的的换月调整日收益：窗口外取近月收益，窗口内取远月收益。
    窗口优先用 v1 背离法，其次成交量规则；都没有则退化为普通收益。
    """
    r1 = front_close.pct_change(fill_method=None)
    window = roll_window(front_close, front_vol, defer_vol, v1_close)
    if window is None or not window.any():
        return r1
    fc, dc = front_close.align(defer_close, join="left")
    r2 = dc.pct_change(fill_method=None)
    out = r1.copy()
    out[window] = r2[window]
    return out


def roll_window(
    front_close: pd.Series,
    front_vol: pd.Series | None = None,
    defer_vol: pd.Series | None = None,
    v1_close: pd.Series | None = None,
) -> pd.Series | None:
    """换月窗口布尔序列（优先 v1 背离法，其次成交量规则）；无数据返回 None。"""
    if v1_close is not None and not v1_close.isna().all():
        w = v1_roll_window(front_close, v1_close)
        if w.any():
            return w
    if (front_vol is not None and defer_vol is not None
            and not front_vol.isna().all() and not defer_vol.isna().all()):
        return volume_roll_window(front_vol, defer_vol)
    return None


def adjusted_price_index(returns: pd.Series, base: float = 100.0) -> pd.Series:
    """把日收益序列转成类价格指数（首个有效收益日起 base=100）。"""
    r = returns.fillna(0.0)
    idx = (1.0 + r).cumprod() * base
    idx[returns.isna()] = float("nan")
    return idx
