# -*- coding: utf-8 -*-
"""风险估计：EWMA 波动率。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ewma_vol(returns: pd.DataFrame, span: int = 40, ann_factor: int = 252) -> pd.DataFrame:
    """逐标的 EWMA 年化波动率（date × instrument）。"""
    return returns.ewm(span=span, min_periods=max(10, span // 2)).std() * np.sqrt(ann_factor)


def ewma_vol_series(returns: pd.Series, span: int = 40, ann_factor: int = 252) -> pd.Series:
    """单序列 EWMA 年化波动率（如组合收益）。"""
    return returns.ewm(span=span, min_periods=max(10, span // 2)).std() * np.sqrt(ann_factor)
