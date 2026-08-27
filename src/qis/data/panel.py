# -*- coding: utf-8 -*-
"""
并集交易日历下的价格面板工具。

多市场标的拼成的 date × instrument 价格矩阵，索引是各标的交易日历的**并集**：
某标的自身的假期在矩阵里就是一个 NaN 洞。对这样的矩阵直接 pct_change 有两个后果：

  1. 洞的当天与次日都得到 NaN，跨洞的真实涨跌被 fillna(0) 整段丢弃
     （不是顺延到下一交易日，是永久消失）；
  2. 引擎按"该日不可交易"清零持仓、而目标权重仍在，虚构出一轮平仓 + 建仓的换手。

正确做法：在标的**自身存续区间内**前值填充——假期当日收益为 0，跨假期的涨跌
落到下一个交易日；首个有效价之前与末个有效价之后保持 NaN（真正不可交易）。

Series 与 DataFrame 都适用。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def alive_mask(prices):
    """存续区间掩码：首个有效价当日 → 末个有效价当日（含两端）。"""
    notna = prices.notna()
    after_first = notna.cumsum() > 0
    before_last = notna[::-1].cumsum()[::-1] > 0
    return after_first & before_last


def ffill_prices(prices):
    """存续区间内前值填充（假期沿用上一收盘），区间外保持 NaN。"""
    return prices.ffill().where(alive_mask(prices))


def to_returns(prices):
    """
    并集日历下的日收益。

    假期当日为 0；跨假期的涨跌落到下一个交易日；存续区间外为 NaN（不可交易）。
    """
    return ffill_prices(prices).pct_change(fill_method=None)


def coverage(prices) -> pd.Series:
    """每列的原始报价覆盖率（有真实收盘的天数 / 存续区间天数），用于数据质量体检。"""
    alive = alive_mask(prices)
    n_alive = alive.sum()
    return (prices.notna().sum() / n_alive.where(n_alive > 0)).astype(float)


# 单日往返尖刺的判定阈值（对数收益）：0.4 ≈ ±50%
SPIKE_LOG = 0.4


def bad_tick_mask(prices, thresh: float = SPIKE_LOG):
    """
    单日"往返尖刺"掩码：某日价格相对前后两天都出现巨幅背离，而前后两天彼此一致。

    这类点是错价/单位错误，不是行情：
      ROUGHRICE 2011-01-04  13.6 → **1390.0** → 13.7   （100 倍单位错误）
      HSCEI     2025-06-02  8502 → **5380**  → 8300    （坏报价）
    真实的暴涨暴跌不会当天原路返回，所以"前后两天彼此一致"这个条件很关键——
    VIX 2018-02-05 的 +112%（15.63 → 33.2 → 23.9）是真行情，不会被误伤。
    """
    p = prices.where(prices > 0)
    prev, nxt = p.shift(1), p.shift(-1)
    up = np.log(p / prev)
    down = np.log(nxt / p)
    across = np.log(nxt / prev)          # 跨过当日：前后两天是否自洽
    return ((up.abs() > thresh) & (down.abs() > thresh)
            & (np.sign(up) != np.sign(down)) & (across.abs() < thresh)).fillna(False)


def clean_prices(prices, thresh: float = SPIKE_LOG):
    """
    价格清洗：剔除非正价格与单日往返尖刺，置为 NaN 交给 ffill 处理。

    非正价格（如 2020-04-20 WTI 结算 −13.1）在比率收益口径下没有意义——
    p_t/p_{t-1} 会给出 −172% 这种数，10 倍杠杆下直接把净值打穿。
    该日标记为不可交易，跨过它的真实跌幅仍会落到下一个交易日。
    """
    p = prices.where(prices > 0)
    return p.where(~bad_tick_mask(p, thresh))


def traded_mask(prices):
    """真实有报价的日子（区别于前值填充出来的假期）。"""
    return prices.notna() & alive_mask(prices)


def liquid_mask(prices, window: int = 63, min_coverage: float = 0.5):
    """
    可交易掩码：滚动窗口内**真实报价占比**达标才算这个标的当天可交易。

    报价过稀的序列不能当日频标的用：COAL(ATWMc1) 在 2012 全年只有 1 个报价，
    于是几个月的涨跌被压成一天的"日收益"（−12.7%），波动估计完全失真，
    逆波动加权反而给了它 115% NAV 的仓位——单日亏掉组合 14.7%。
    正常日线的覆盖率在 90% 以上，取 50% 作门槛只会挡掉真正断档的区段。
    """
    traded = traded_mask(prices)
    cov = traded.rolling(window, min_periods=max(5, window // 4)).mean()
    return (cov >= min_coverage).fillna(False) & alive_mask(prices)
