# -*- coding: utf-8 -*-
"""FastAPI 路由：QIS 研究终端的 API 与静态页。"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from qis.web.service import STRATEGIES, QISService, get_service

_STATIC = Path(__file__).resolve().parent / "static"


def create_app(service: Optional[QISService] = None) -> FastAPI:
    svc = service or get_service()
    app = FastAPI(title="QIS Terminal", version="0.1.0")

    @app.exception_handler(ValueError)
    def _value_error(_, exc: ValueError):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.get("/api/overview")
    def overview():
        return {
            "n_instruments": len(svc.universe),
            "n_rics": len(svc.universe.all_rics()),
            "asset_classes": svc.asset_classes(),
            "strategies": STRATEGIES,
            "settings": svc.settings["backtest"],
        }

    @app.get("/api/instruments")
    def instruments():
        return svc.instruments()

    @app.get("/api/instruments/{name}/series")
    def instrument_series(name: str, years: int = Query(3, ge=0, le=30)):
        try:
            return svc.instrument_series(name.upper(), years=years)
        except KeyError:
            raise HTTPException(404, f"unknown instrument: {name}")

    @app.get("/api/run")
    def run(strategy: str,
            start: Optional[str] = None, end: Optional[str] = None,
            vol_target: Optional[float] = None, gross: float = 1.0,
            classes: Optional[str] = None, with_cost: bool = True,
            band: Optional[float] = None):
        cls = tuple(sorted(classes.split(","))) if classes else None
        return svc.run(strategy, start=start, end=end, vol_target=vol_target,
                       gross=gross, classes=cls, with_cost=with_cost, band=band)

    @app.get("/api/data/status")
    def data_status():
        return svc.data_status()

    @app.get("/")
    def index():
        return FileResponse(_STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=_STATIC), name="static")
    return app


def main(host: str = "127.0.0.1", port: int = 8600) -> None:
    import uvicorn
    uvicorn.run(create_app(), host=host, port=port)
