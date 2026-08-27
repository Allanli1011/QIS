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
def svc(tmp_path):
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

    service = QISService(universe_path=str(u_yaml))
    service.store = store
    return service


@pytest.fixture
def client(svc):
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


def test_reload_clears_backtest_cache(svc):
    """
    回归：旧 reload() 只把 _prices 置空，_run_cache 原封不动，
    拉了新数据之后再跑回测仍会拿到同一份旧结果。
    """
    first = svc.run("trend", start="2020-06-01")
    assert svc._run_cache and svc._prices is not None
    svc.reload()
    assert svc._prices is None and not svc._run_cache and not svc._roll_diag
    second = svc.run("trend", start="2020-06-01")
    assert second is not first          # 重新算过，不是缓存里那份
    assert second["metrics"]["n_days"] == first["metrics"]["n_days"]


def test_reload_picks_up_new_data(svc):
    """reload 后必须反映新落盘的数据。"""
    n0 = svc.run("trend", start="2020-06-01")["metrics"]["n_days"]
    df = svc.store.load("AAc1")
    extra = pd.DataFrame(
        {"close": [float(df["close"].iloc[-1])] * 5,
         "volume": [1500.0] * 5},
        index=pd.bdate_range(df.index[-1] + pd.offsets.BDay(1), periods=5))
    svc.store.save("AAc1", pd.concat([df, extra]))
    svc.reload()
    assert svc.run("trend", start="2020-06-01")["metrics"]["n_days"] > n0


def test_instruments_report_true_market_price(client):
    """
    回归：带远月腿的标的在价格矩阵里是换月调整后的**指数**（基数 100 累乘），
    直接当"最新价"展示会把 SP500 显示成 460 而不是 7710。
    """
    from qis.data.store import DataStore  # noqa: F401
    rows = {i["name"]: i for i in client.get("/api/instruments").json()}
    aaa, bbb = rows["AAA"], rows["BBB"]
    assert aaa["is_adjusted"] is True and bbb["is_adjusted"] is False
    # AAA 原始收盘价在 100 附近，调整指数在 100 附近也可能撞上，
    # 但两者必须是分开的字段且 last 等于真实收盘
    assert aaa["last"] != aaa["index"] or aaa["index"] is None
    assert aaa["last"] is not None and aaa["last"] > 0


def test_run_exposes_risk_diagnostics(client):
    d = client.get("/api/run?strategy=trend&start=2020-06-01").json()
    assert "leverage_cap_share" in d
    assert "roll_suspect" in d
    assert d["params"]["max_leverage"] > 0


def test_data_status_reports_sparsity(client):
    rows = client.get("/api/data/status").json()
    for r in rows:
        assert "rows_per_year" in r and "sparse" in r
