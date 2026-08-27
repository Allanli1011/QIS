# -*- coding: utf-8 -*-
"""
策略约定。

策略 = 纯函数：prices (date × instrument 收盘价矩阵) → weights (date × instrument 目标权重)。
  * weights 在 t 日收盘用截至 t 日的数据生成，由回测引擎 shift(1) 后作用于 t+1 收益；
  * 输出前须把毛敞口归一化（normalize_gross），波动目标等组合级缩放由调用方叠加。
"""
from __future__ import annotations

from typing import Callable

import pandas as pd

# 策略函数签名：价格矩阵 → 目标权重矩阵
StrategyFn = Callable[[pd.DataFrame], pd.DataFrame]
