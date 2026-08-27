# -*- coding: utf-8 -*-
"""
parquet 本地缓存：按 RIC 存日线，支持增量更新。

缓存文件：{data_dir}/{ric 安全文件名}.parquet
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from qis.data.lseg import LSEGSource, get_source

_DEFAULT_DATA_DIR = Path(__file__).resolve().parents[3] / "data"


def _safe_name(ric: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", ric)


class DataStore:
    """日线 parquet 缓存（增量更新）。"""

    def __init__(self, data_dir: Optional[str | Path] = None):
        self.data_dir = Path(data_dir) if data_dir else _DEFAULT_DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def path(self, ric: str) -> Path:
        return self.data_dir / f"{_safe_name(ric)}.parquet"

    # ---- 读写 ----
    def load(self, ric: str) -> pd.DataFrame:
        p = self.path(ric)
        if not p.exists():
            return pd.DataFrame()
        df = pd.read_parquet(p)
        df.index = pd.to_datetime(df.index)
        return df.sort_index()

    def save(self, ric: str, df: pd.DataFrame) -> Path:
        p = self.path(ric)
        df.sort_index().to_parquet(p)
        return p

    def last_date(self, ric: str) -> Optional[pd.Timestamp]:
        df = self.load(ric)
        return df.index.max() if len(df) else None

    # ---- 更新 ----
    def update(
        self,
        ric: str,
        start: str = "2000-01-01",
        end: Optional[str] = None,
        source: Optional[LSEGSource] = None,
        value_field: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        增量更新：已有缓存则从最后日期的下一日拉取并拼接；否则全量拉取。
        返回更新后的完整缓存。
        """
        src = source or get_source()
        cached = self.load(ric)

        fetch_start = start
        if len(cached):
            next_day = cached.index.max() + pd.offsets.Day(1)
            fetch_start = max(pd.Timestamp(start), next_day).strftime("%Y-%m-%d")

        new = src.history(ric, start=fetch_start, end=end, value_field=value_field)
        if len(new):
            combined = pd.concat([cached, new])
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        else:
            combined = cached
        if len(combined):
            self.save(ric, combined)
        return combined

    def update_many(
        self,
        rics: Iterable[str],
        start: str = "2000-01-01",
        end: Optional[str] = None,
        source: Optional[LSEGSource] = None,
        verbose: bool = True,
    ) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        src = source or get_source()
        for ric in rics:
            try:
                df = self.update(ric, start=start, end=end, source=src)
                out[ric] = df
                if verbose:
                    last = df.index.max().date() if len(df) else "-"
                    print(f"  {ric:<14} rows={len(df):>5}  last={last}")
            except Exception as e:  # 单个失败不阻断整批
                print(f"  {ric:<14} FAILED: {e}")
                out[ric] = pd.DataFrame()
        return out

    # ---- 组合读取 ----
    def load_close_matrix(self, rics: Iterable[str]) -> pd.DataFrame:
        """读取多个 RIC 的 close，拼成 date × ric 的价格矩阵（列顺序同输入）。"""
        cols = {}
        for ric in rics:
            df = self.load(ric)
            if len(df):
                cols[ric] = df["close"]
        if not cols:
            return pd.DataFrame()
        return pd.DataFrame(cols).sort_index()
