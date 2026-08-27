# -*- coding: utf-8 -*-
"""
连续合约换月处理。

LSEG 的 c1/c2 连续价是换月日直接拼接的，未做复权：换月日 c1 的"收益"里含
新旧合约的价差（contango 时是幻影正收益、backwardation 时是幻影负收益），
并非持仓者的真实损益。直接用 c1 算收益会让回测系统性失真。

换月日识别——用持仓量（open interest）的不连续跳升
--------------------------------------------------
c1 是一条**拼接**序列：换月那天它指向的合约变了，于是挂在 c1 上的持仓量也
从"即将到期、持仓已萎缩的旧合约"跳到"新的主力合约"，量级往往翻数倍。
实测这个信号双峰分离得非常干净（ES/NQ：换月日 log 跳升 +0.8~1.5，
平日 +0.01~0.06，差 20 倍以上），且抓到的日期正好落在各品种的到期月。

优先用 oi，没有 oi 时退回 volume（同样的跳升逻辑，阈值更高因为成交量本身更吵）。
两者都没有则不做调整——但会在 `roll_diagnostics` 里报出来，不静默放过。

换月日的收益口径
----------------
设换月日为 T，c1 在 T 日已切到新合约（记 N+1），T-1 日 c1 还是旧合约 N。
  * c1(T)/c1(T-1) − 1  ——  跨了两张合约，是幻影收益；
  * c2(T)/c2(T-1) − 1  ——  c2 当天同样滚动了（N+1 → N+2），一样被污染；
  * c1(T)/c2(T-1) − 1  ——  c1(T) 与 c2(T-1) 指的是**同一张合约 N+1**，干净。
最后一个才是持仓者在 T-1 收盘换到新合约后、T 日的真实损益，本模块采用它。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from qis.data.panel import ffill_prices

# oi/volume 跳升的对数阈值：oi 翻 1.65 倍以上视为换了合约
OI_JUMP = 0.5
VOL_JUMP = 1.5
# 两次换月的最小间隔（交易日）：月度合约约 21 天一次，10 天足以去重又不会合并真实换月
MIN_GAP = 10


def _enforce_min_gap(hits: pd.Series, min_gap: int) -> pd.Series:
    """同一簇里只保留第一天，避免一次换月被记成连续几天。"""
    out = pd.Series(False, index=hits.index)
    last = -10**9
    for i, flag in enumerate(hits.to_numpy()):
        if flag and i - last >= min_gap:
            out.iloc[i] = True
            last = i
    return out


def _jump_days(series: pd.Series | None, thresh: float, min_gap: int) -> pd.Series | None:
    """序列出现 log 跳升 > thresh 的日子（去簇后）。数据不可用返回 None。"""
    if series is None or series.isna().all():
        return None
    s = pd.to_numeric(series, errors="coerce").replace(0.0, np.nan)
    if s.notna().sum() < 20:
        return None
    jump = np.log(s / s.shift(1))
    hits = (jump > thresh).fillna(False)
    if not hits.any():
        return None
    return _enforce_min_gap(hits, min_gap)


def oi_roll_days(front_oi: pd.Series, thresh: float = OI_JUMP,
                 min_gap: int = MIN_GAP) -> pd.Series | None:
    """持仓量跳升法（首选）。"""
    return _jump_days(front_oi, thresh, min_gap)


def volume_roll_days(front_vol: pd.Series, thresh: float = VOL_JUMP,
                     min_gap: int = MIN_GAP) -> pd.Series | None:
    """成交量跳升法（无 oi 时的回退）。"""
    return _jump_days(front_vol, thresh, min_gap)


def roll_days(
    front_close: pd.Series,
    front_oi: pd.Series | None = None,
    front_vol: pd.Series | None = None,
) -> tuple[pd.Series, str]:
    """
    换月交易日布尔序列 + 所用方法（"oi" / "volume" / "none"）。

    识别不出来时返回全 False 与 "none"——调用方据此可知该标的未做换月调整。
    """
    for name, mask in (("oi", oi_roll_days(front_oi) if front_oi is not None else None),
                       ("volume", volume_roll_days(front_vol) if front_vol is not None else None)):
        if mask is not None and mask.any():
            return mask.reindex(front_close.index, fill_value=False), name
    return pd.Series(False, index=front_close.index), "none"


def adjust(
    front_close: pd.Series,
    defer_close: pd.Series,
    front_oi: pd.Series | None = None,
    front_vol: pd.Series | None = None,
) -> tuple[pd.Series, pd.Series, str]:
    """
    单标的换月调整，一次返回 (调整后日收益, 换月日掩码, 识别方法)。

    平日取 c1 自身收益；换月日 T 改用 c1(T)/c2(T-1) − 1（同一张合约，见模块 docstring）。
    c2 在 T-1 缺报价时该日退回 c1 naive 收益（无从修正，保持原样而不是造 NaN）。
    """
    f = ffill_prices(front_close)
    r1 = f.pct_change(fill_method=None)
    mask, method = roll_days(front_close, front_oi, front_vol)
    if not mask.any():
        return r1, mask, method
    d = ffill_prices(defer_close.reindex(front_close.index))
    r_roll = f / d.shift(1) - 1.0
    return r1.where(~mask | r_roll.isna(), r_roll), mask, method


def adjusted_returns(
    front_close: pd.Series,
    defer_close: pd.Series,
    front_oi: pd.Series | None = None,
    front_vol: pd.Series | None = None,
) -> pd.Series:
    """`adjust` 的便捷包装：只要调整后的日收益。"""
    return adjust(front_close, defer_close, front_oi, front_vol)[0]


def roll_diagnostics(
    front_close: pd.Series,
    front_oi: pd.Series | None = None,
    front_vol: pd.Series | None = None,
    lo: float = 1.0,
    hi: float = 14.0,
) -> dict:
    """
    换月识别体检：方法、每年换月次数、是否落在合理区间（月度合约 ~12，季度 ~4）。

    次数离谱（或方法为 none）说明该标的的换月调整不可信，应当在报告里显式暴露，
    而不是让回测悄悄用一串错误收益。
    """
    mask, method = roll_days(front_close, front_oi, front_vol)
    idx = front_close.dropna().index
    years = (idx[-1] - idx[0]).days / 365.25 if len(idx) > 1 else 0.0
    per_year = float(mask.sum() / years) if years > 0 else float("nan")
    ok = method != "none" and lo <= per_year <= hi
    return {"method": method, "n_rolls": int(mask.sum()),
            "per_year": per_year, "plausible": bool(ok)}


def contract_horizon(mask: pd.Series, ann_days: float = 252.0,
                      min_rolls: int = 4) -> float:
    """
    从换月日的间隔估计 c1 与 c2 两腿到期日的间隔（年）。

    c1 与 c2 同步滚动，所以相邻两次换月的间隔就是相邻合约的到期间隔。
    这个数用来把 c1/c2 价差**年化**——本池的合约周期从 18 到 476 个交易日
    相差 27 倍，不年化的话截面上比的是"合约周期长短"而不是 carry。
    换月次数不足以估计时返回 nan（由调用方兜底）。
    """
    idx = np.flatnonzero(mask.to_numpy())
    if len(idx) < min_rolls:
        return float("nan")
    return float(np.median(np.diff(idx))) / ann_days


def adjusted_price_index(returns: pd.Series, base: float = 100.0) -> pd.Series:
    """
    把日收益序列转成类价格指数。

    基准 base 落在**首个有效收益的前一天**：下游是对这条指数再做 pct_change 的，
    把基准放在首个有效收益当天会让第一天的收益无处安放、白白丢掉。
    （首行即为有效收益时前面没有基准位可用，只能牺牲那一天。）
    """
    valid = returns.notna()
    if not valid.any():
        return pd.Series(np.nan, index=returns.index, dtype=float)
    idx = (1.0 + returns.fillna(0.0)).cumprod() * base
    keep = valid.copy()
    first = int(np.argmax(valid.to_numpy()))
    if first > 0:
        keep.iloc[first - 1] = True
    return idx.where(keep)
