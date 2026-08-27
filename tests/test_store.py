# -*- coding: utf-8 -*-
"""缓存层测试：parquet 读写、增量更新（用假数据源，不依赖 LSEG）。"""
import pandas as pd

from qis.data.store import DataStore


class FakeSource:
    """模拟 LSEGSource.history：返回 start 起的合成日线。"""

    def history(self, ric, count=500, start=None, end=None, value_field=None):
        idx = pd.date_range(start or "2024-01-01", end or "2024-01-10", freq="D")
        return pd.DataFrame({"close": range(1, len(idx) + 1), "ric": ric}, index=idx)


def test_save_load_roundtrip(tmp_path):
    store = DataStore(tmp_path)
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    df = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=idx)
    store.save("ESc1", df)
    loaded = store.load("ESc1")
    pd.testing.assert_frame_equal(loaded, df.sort_index(), check_freq=False)


def test_load_missing_returns_empty(tmp_path):
    store = DataStore(tmp_path)
    assert store.load("NOPE").empty


def test_incremental_update(tmp_path):
    store = DataStore(tmp_path)
    src = FakeSource()
    # 首次全量
    df1 = store.update("X", start="2024-01-01", end="2024-01-05", source=src)
    assert len(df1) == 5
    # 二次更新：从缓存最后日期的下一日拉取
    df2 = store.update("X", start="2024-01-01", end="2024-01-10", source=src)
    assert len(df2) == 10
    assert not df2.index.duplicated().any()
    assert df2.index.max() == pd.Timestamp("2024-01-10")


def test_update_is_idempotent(tmp_path):
    store = DataStore(tmp_path)
    src = FakeSource()
    store.update("X", start="2024-01-01", end="2024-01-05", source=src)
    again = store.update("X", start="2024-01-01", end="2024-01-05", source=src)
    assert len(again) == 5  # 不重复追加


def test_load_close_matrix(tmp_path):
    store = DataStore(tmp_path)
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    store.save("A", pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=idx))
    store.save("B", pd.DataFrame({"close": [4.0, 5.0, 6.0]}, index=idx))
    m = store.load_close_matrix(["A", "B", "MISSING"])
    assert list(m.columns) == ["A", "B"]  # 缺失标的被跳过，列序保持
    assert m.loc[idx[1], "B"] == 5.0
