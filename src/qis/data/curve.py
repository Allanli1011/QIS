# -*- coding: utf-8 -*-
"""
多月合约曲线（c1, c2, c3, …）的加载与整理。

两点价差（c1/c2）估 carry 的信噪比很差：它是两个带噪报价的差分，
实测本池 corr(信噪比, IC) = −0.53，噪声大的品种 carry 信号系统性做反。
拉出整条曲线后可以用回归斜率估 carry——N 点回归斜率的方差约为两点差分的 1/N。
"""
from __future__ import annotations

import re
from typing import Mapping

import pandas as pd

from qis.data.panel import clean_prices

_CN = re.compile(r"c(\d+)$")


def curve_ric(ric: str, n: int) -> str | None:
    """c1 连续合约的第 n 月 RIC（ESc1 → ESc3）；非 cN 形式返回 None。"""
    m = _CN.search(ric)
    return f"{ric[: m.start()]}c{n}" if m else None


def load_curve(store, ric: str, depth: int = 4) -> dict[int, pd.Series]:
    """
    读取 c1..c{depth} 的收盘价（已过坏价清洗）。缺失的月份直接不在返回值里。
    """
    out: dict[int, pd.Series] = {}
    for n in range(1, depth + 1):
        r = curve_ric(ric, n)
        if r is None:
            continue
        df = store.load(r)
        if len(df) and "close" in df.columns:
            s = clean_prices(df["close"]).dropna()
            if len(s):
                out[n] = s
    return out


def curve_matrix(
    store,
    instruments: list[dict],
    depth: int = 4,
) -> dict[int, pd.DataFrame]:
    """
    整个标的池的曲线，按月份号组织：{n: date × name 的收盘价矩阵}。

    只保留能凑齐 >= 2 个月份的标的（少于 2 个月无法估斜率）。
    """
    per_leg: dict[int, dict[str, pd.Series]] = {}
    for inst in instruments:
        name, ric = inst["name"], inst["ric"]
        legs = load_curve(store, ric, depth=depth)
        if len(legs) < 2:
            continue
        for n, s in legs.items():
            per_leg.setdefault(n, {})[name] = s
    return {n: pd.DataFrame(cols).sort_index() for n, cols in sorted(per_leg.items())}


def curve_depth(curves: Mapping[int, pd.DataFrame]) -> pd.Series:
    """每个标的实际可用的曲线月份数。"""
    counts: dict[str, int] = {}
    for df in curves.values():
        for c in df.columns:
            counts[c] = counts.get(c, 0) + 1
    return pd.Series(counts, dtype=int)
