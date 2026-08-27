# -*- coding: utf-8 -*-
"""LSEG 字段归一化测试（不依赖 session，纯本地数据）。"""
import numpy as np
import pandas as pd

from qis.data.lseg import LSEGSource


def _raw(cols: dict) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    return pd.DataFrame(cols, index=idx)


def test_normalize_prefers_trdprc_over_settle():
    raw = _raw({"SETTLE": [1.0, 2.0, 3.0], "TRDPRC_1": [10.0, 20.0, 30.0]})
    out = LSEGSource.normalize(raw, ric="X")
    assert out["close"].tolist() == [10.0, 20.0, 30.0]
    assert out["ric"].iloc[0] == "X"


def test_normalize_settle_when_no_trdprc():
    raw = _raw({"SETTLE": [1.0, 2.0, 3.0]})
    out = LSEGSource.normalize(raw)
    assert out["close"].tolist() == [1.0, 2.0, 3.0]


def test_normalize_explicit_value_field():
    raw = _raw({"TRDPRC_1": [1.0, 2.0, 3.0], "FAIR_VALUE": [7.0, 8.0, 9.0]})
    out = LSEGSource.normalize(raw, value_field="FAIR_VALUE")
    assert out["close"].tolist() == [7.0, 8.0, 9.0]


def test_normalize_fallback_first_numeric():
    raw = _raw({"SOME_COL": [5.0, 6.0, 7.0]})
    out = LSEGSource.normalize(raw)
    assert out["close"].tolist() == [5.0, 6.0, 7.0]


def test_normalize_ohlcv_and_dropna():
    raw = _raw({
        "TRDPRC_1": [1.0, np.nan, 3.0],
        "OPEN_PRC": [0.9, 1.9, 2.9],
        "ACVOL_UNS": [100, 200, 300],
    })
    out = LSEGSource.normalize(raw)
    assert len(out) == 2  # 中间 NaN close 被剔除
    assert "open" in out.columns and "volume" in out.columns
    assert out["open"].tolist() == [0.9, 2.9]
