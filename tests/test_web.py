# -*- coding: utf-8 -*-
"""Web API 冒烟测试（临时小数据集，不依赖 LSEG）。"""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from qis.data.store import DataStore
from qis.web.api import create_app
from qis.web.service import QISService


@pytest.fixture
def client(tmp_path):
    # 两个标的（一个有远月腿）的 toy 缓存
    u_yaml = tmp_path / "u.yaml"
    u_yaml.write_text(
        "instruments:\n"
        "  - { name: AAA, ric: AAc1, asset_class: energy, carry_leg: AAc2 }\n"
        "  - { name: BBB, ric: 'BB=', asset_class: fx }\n",
        encoding="utf-8",
    )
    store = DataStore(tmp_path / "data")
    idx = pd.date_range("2020-01-01", periods=400, freq="B")
    rng = np.random.default_rng(5)
    for ric, base in [("AAc1", 100.0), ("AAc2", 99.0), ("BB=", 1.2)]:
        px = base * np.cumprod(1 + rng.normal(0.0003, 0.01, 400))
        store.save(ric, pd.DataFrame({
            "close": px, "volume": rng.integers(1000, 2000, 400).astype(float),
        }, index=idx))

    svc = QISService(universe_path=str(u_yaml))
    svc.store = store
    return TestClient(create_app(service=svc))


def test_overview(client):
    r = client.get("/api/overview")
    assert r.status_code == 200
    d = r.json()
    assert d["n_instruments"] == 2
    assert "energy" in d["asset_classes"]


def test_instruments(client):
    r = client.get("/api/instruments")
    assert r.status_code == 200
    names = {i["name"] for i in r.json()}
    assert names == {"AAA", "BBB"}


def test_instrument_series(client):
    r = client.get("/api/instruments/AAA/series?years=1")
    assert r.status_code == 200
    assert len(r.json()["points"]) > 100
    assert client.get("/api/instruments/NOPE/series").status_code == 404


def test_run(client):
    r = client.get("/api/run?strategy=trend&start=2020-06-01")
    assert r.status_code == 200
    d = r.json()
    assert d["n_instruments"] == 2
    for k in ["equity", "drawdown", "monthly", "attribution", "metrics"]:
        assert k in d
    assert len(d["equity"]) > 100


def test_run_bad_strategy(client):
    assert client.get("/api/run?strategy=nope").status_code == 400


def test_data_status(client):
    r = client.get("/api/data/status")
    assert r.status_code == 200
    assert {x["ric"] for x in r.json()} == {"AAc1", "AAc2", "BB="}
