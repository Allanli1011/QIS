# -*- coding: utf-8 -*-
"""
LSEG / Refinitiv Workspace 行情适配器。

要点：
  * session 管理：复用已开 session，掉线自动重开（lseg-data 的 default session 是进程单例）。
  * 值字段归一化：不同 RIC 的"收盘/结算/评估"列名各异
    (TRDPRC_1 / SETTLE / FAIR_VALUE / VAL_0430 / MID_PRICE...)，统一取成 `close`。

依赖：LSEG/Refinitiv Workspace 桌面端在线；可选环境变量 LSEG_APP_KEY。
"""
from __future__ import annotations

import os
from typing import Optional, Sequence

import lseg.data as ld
import pandas as pd

# 收盘/结算/评估值的候选列，按优先级归一化到 close
_VALUE_PRIORITY = [
    "TRDPRC_1", "SETTLE", "FAIR_VALUE", "OFFCL_VAL", "MID_PRICE",
    "VAL_0430", "VAL_1630", "VAL_0000", "CLOSE", "CF_CLOSE", "CF_LAST", "VALUE",
]
_OHLCV_MAP = {
    "OPEN_PRC": "open", "HIGH_1": "high", "LOW_1": "low",
    "ACVOL_UNS": "volume", "OPINT_1": "oi",
}


class LSEGSource:
    """LSEG 行情适配器（懒开 session、可复用）。"""

    _opened = False  # 类级别：default session 是进程单例

    def __init__(self, app_key: Optional[str] = None):
        # app_key 可选：不给就用内置默认 key 连本机 Workspace 桌面会话
        self.app_key = (app_key or os.environ.get("LSEG_APP_KEY") or "").strip() or None

    # ---- session ----
    def ensure_session(self) -> None:
        """复用已开 session；未开或已掉线则重开。"""
        if LSEGSource._opened and self._is_open():
            return
        if self.app_key:
            ld.open_session(app_key=self.app_key)
        else:
            ld.open_session()
        LSEGSource._opened = True

    @staticmethod
    def _is_open() -> bool:
        try:
            s = ld.session.get_default()
            return s is not None and str(getattr(s, "open_state", "")) == "OpenState.Opened"
        except Exception:
            return False

    def close(self) -> None:
        try:
            ld.close_session()
        finally:
            LSEGSource._opened = False

    # ---- 历史行情 ----
    def history(
        self,
        ric: str,
        count: Optional[int] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        value_field: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        返回归一化日线：index=date(DatetimeIndex)，列含 `close`（+ open/high/low/volume/oi 若有）。
        空数据返回空 DataFrame。count 缺省时：给了 start 取大值（拉全长历史），否则 500 条。
        """
        self.ensure_session()
        if count is None:
            count = 100_000 if start else 500
        kwargs: dict = dict(universe=ric, count=count)
        if start:
            kwargs["start"] = start
        if end:
            kwargs["end"] = end
        raw = ld.get_history(**kwargs)
        if raw is None or len(raw) == 0:
            return pd.DataFrame()
        return self.normalize(raw, ric=ric, value_field=value_field)

    # ---- 字段归一化（独立于 session，便于单测） ----
    @staticmethod
    def normalize(raw: pd.DataFrame, ric: str = "", value_field: Optional[str] = None) -> pd.DataFrame:
        """把 ld.get_history 的原始结果归一化为 close(+OHLCV) 日线。"""
        df = raw.copy()
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"

        close_col = value_field if (value_field and value_field in df.columns) else None
        if close_col is None:
            for c in _VALUE_PRIORITY:
                if c in df.columns:
                    close_col = c
                    break
        out = pd.DataFrame(index=df.index)
        if close_col is not None:
            out["close"] = pd.to_numeric(df[close_col], errors="coerce")
        else:
            # 兜底：第一列数值
            num = df.apply(pd.to_numeric, errors="coerce")
            out["close"] = num.iloc[:, 0]
        for src_col, dst in _OHLCV_MAP.items():
            if src_col in df.columns:
                out[dst] = pd.to_numeric(df[src_col], errors="coerce")
        if ric:
            out["ric"] = ric
        return out.dropna(subset=["close"])

    # ---- 快照 ----
    def snapshot(self, rics: Sequence[str], fields: Sequence[str]) -> pd.DataFrame:
        self.ensure_session()
        df = ld.get_data(universe=list(rics), fields=list(fields))
        return df if df is not None else pd.DataFrame()

    # ---- 发现/搜索 ----
    def search(self, query: str, top: int = 20, view: str = "SEARCH_ALL") -> pd.DataFrame:
        self.ensure_session()
        view_obj = getattr(ld.discovery.Views, view, ld.discovery.Views.SEARCH_ALL)
        df = ld.discovery.search(
            query=query, view=view_obj,
            select="RIC,DocumentTitle,ExchangeName", top=top,
        )
        return df if df is not None else pd.DataFrame()


# 模块级单例，便于复用
_default: Optional[LSEGSource] = None


def get_source(app_key: Optional[str] = None) -> LSEGSource:
    global _default
    if _default is None:
        _default = LSEGSource(app_key=app_key)
    return _default
