# -*- coding: utf-8 -*-
"""Web 服务层：数据加载、回测运行、结果序列化（供 FastAPI 路由调用）。"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd

from qis.analytics.metrics import drawdown, infer_ann_factor, summary
from qis.backtest.costs import cost_bps_by_name, load_settings
from qis.backtest.engine import run_backtest
from qis.cli import _adjusted_price_matrix, _strategy_weights
from qis.data.store import DataStore
from qis.data.universe import Universe
from qis.data.panel import coverage, to_returns
from qis.portfolio.construction import (cap_binding_share, cross_sectional_z,
                                        factor_exposure, vol_target_factor,
                                        weight_band)

STRATEGIES = ["trend", "xsmom", "carry"]

_CLASS_LABELS = {
    "equity_index": "股指", "bond": "债券", "fx": "外汇", "energy": "能源",
    "metal": "金属", "ags": "农产品", "rates": "利率", "crypto": "加密",
    "default": "其他",
}


def class_label(ac: str) -> str:
    return _CLASS_LABELS.get(ac, ac)


def _jsonable(v):
    """numpy/pandas 标量 → JSON 原生类型；NaN/Inf/NA → None。"""
    if isinstance(v, (np.floating, float)):
        return None if not math.isfinite(v) else float(v)
    if isinstance(v, (np.integer, int)) and not isinstance(v, bool):
        return int(v)
    if isinstance(v, (str, bool)) or v is None:
        return v
    try:
        return None if pd.isna(v) else v
    except (TypeError, ValueError):
        return v


def _series_points(s: pd.Series, precision: int = 6) -> list:
    return [[d.strftime("%Y-%m-%d"), _jsonable(round(v, precision))]
            for d, v in s.items()]


class QISService:
    """应用级服务：持有一份 universe/store，内存缓存价格矩阵与回测结果。"""

    def __init__(self, universe_path: Optional[str] = None):
        self.universe = Universe.from_yaml(universe_path)
        self.store = DataStore()
        self.settings = load_settings()
        self._prices: Optional[pd.DataFrame] = None
        self._roll_mask: Optional[pd.DataFrame] = None
        self._roll_diag: dict = {}
        self._raw_last: dict = {}
        self._run_cache: dict = {}

    # ---------------- 数据 ----------------
    def prices(self) -> pd.DataFrame:
        """全标的换月调整价格矩阵（首次构建后缓存）。"""
        if self._prices is None:
            self._prices, self._roll_mask, self._roll_diag = _adjusted_price_matrix(
                self.store, self.universe, with_roll_mask=True, with_diagnostics=True)
            # 真实市场收盘价：价格矩阵里带远月腿的标的是换月调整后的**指数**
            # （基数 100 累乘），不是行情价，不能拿去当"最新价"展示。
            self._raw_last = {}
            for inst in self.universe.instruments:
                df = self.store.load(inst["ric"])
                if len(df) and "close" in df.columns:
                    sr = df["close"].dropna()
                    if len(sr):
                        self._raw_last[inst["name"]] = (float(sr.iloc[-1]),
                                                        sr.index[-1].strftime("%Y-%m-%d"))
        return self._prices

    def roll_mask(self) -> pd.DataFrame:
        self.prices()  # 确保已构建
        return self._roll_mask

    def roll_diagnostics(self) -> dict:
        self.prices()
        return self._roll_diag

    def reload(self) -> None:
        """丢弃所有缓存。回测结果缓存也必须清——否则拉了新数据仍返回旧结果。"""
        self._prices = None
        self._roll_mask = None
        self._roll_diag = {}
        self._raw_last = {}
        self._run_cache.clear()

    def asset_classes(self) -> list[str]:
        classes = list(dict.fromkeys(self.universe.asset_classes().values()))
        return sorted(classes)

    def instruments(self) -> list[dict]:
        """标的清单 + 最新价与区间涨跌。"""
        px = self.prices()
        out = []
        for inst in self.universe.instruments:
            name = inst["name"]
            if name not in px.columns:
                continue
            s = px[name].dropna()
            if s.empty:
                continue
            last = s.iloc[-1]
            def chg(days: int) -> Optional[float]:
                if len(s) > days and s.iloc[-days - 1]:
                    return _jsonable(last / s.iloc[-days - 1] - 1.0)
                return None
            raw_px, raw_date = self._raw_last.get(name, (None, None))
            out.append({
                "name": name, "ric": inst["ric"],
                "asset_class": inst["asset_class"],
                "class_label": class_label(inst["asset_class"]),
                "has_carry_leg": bool(inst.get("carry_leg")),
                # last = 真实市场收盘价；index = 换月调整后的价格指数（涨跌幅的来源）
                "last": _jsonable(raw_px if raw_px is not None else last),
                "last_date": raw_date or s.index[-1].strftime("%Y-%m-%d"),
                "index": _jsonable(last),
                "is_adjusted": bool(inst.get("carry_leg")),
                "chg_1d": chg(1), "chg_1m": chg(21), "chg_1y": chg(252),
                "n_days": int(len(s)),
            })
        return out

    def instrument_series(self, name: str, years: int = 3) -> dict:
        px = self.prices()
        if name not in px.columns:
            raise KeyError(name)
        s = px[name].dropna()
        if years > 0:
            s = s.loc[s.index >= s.index[-1] - pd.offsets.Day(365 * years)]
        return {"name": name, "points": _series_points(s, precision=4)}

    def data_status(self) -> list[dict]:
        out = []
        for ric in self.universe.all_rics():
            df = self.store.load(ric)
            rows_per_year = None
            if len(df) > 1:
                yrs = (df.index[-1] - df.index[0]).days / 365.25
                rows_per_year = round(len(df) / yrs, 1) if yrs > 0 else None
            out.append({
                "ric": ric,
                "rows": int(len(df)),
                "first": df.index[0].strftime("%Y-%m-%d") if len(df) else None,
                "last": df.index[-1].strftime("%Y-%m-%d") if len(df) else None,
                # 正常日线约 250 行/年；明显偏低说明这条序列本身缺数（如 NIYc1 ~94）
                "rows_per_year": rows_per_year,
                "sparse": bool(rows_per_year is not None and rows_per_year < 200),
            })
        return out

    def roll_quality(self) -> list[dict]:
        """换月识别体检：识别不出或次数离谱的标的，其收益未被可靠调整。"""
        return [{"name": n, **d} for n, d in sorted(self.roll_diagnostics().items())]

    # ---------------- 回测 ----------------
    def run(self, strategy: str, start: Optional[str] = None,
            end: Optional[str] = None, vol_target: Optional[float] = None,
            gross: Optional[float] = None, classes: Optional[tuple[str, ...]] = None,
            with_cost: bool = True, band: Optional[float] = None) -> dict:
        if strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy: {strategy}")
        key = (strategy, start, end, vol_target, gross, classes, with_cost, band)
        if key in self._run_cache:
            return self._run_cache[key]
        bt = self.settings["backtest"]
        start = start or bt["start"]
        vol_target = vol_target if vol_target is not None else bt["vol_target"]
        gross = gross if gross is not None else bt.get("gross", 1.0)

        full = self.prices()
        mask_full = self.roll_mask()
        if classes:
            keep = [i["name"] for i in self.universe.instruments
                    if i["asset_class"] in classes and i["name"] in full.columns]
            if not keep:
                raise ValueError("no data for given filters")
            full = full[keep]
            mask_full = mask_full[keep]

        # 信号 / 波动目标 / 无交易带都在完整历史上算，最后才切窗口：
        # 先切会让 start 之后第一年被 lookback 预热吃掉（trend 需 252 日）。
        w_raw = _strategy_weights(strategy, full, self.universe, self.store, gross)
        band = band if band is not None else bt.get("rebal_band", 0.0)
        rets_full = to_returns(full)
        max_lev = float(bt.get("max_leverage", 3.0))
        af = infer_ann_factor(rets_full.iloc[:, 0], bt["ann_factor"]) if full.shape[1] else bt["ann_factor"]
        scale_f, raw_f = vol_target_factor(w_raw, rets_full, vol_target, bt["vol_lookback"],
                                           af, max_lev)
        w = w_raw.mul(scale_f, axis=0).where(scale_f.notna(), w_raw)
        # 无交易带作用在杠杆之后的实际持仓上（见 cli.cmd_run 注释：与 max_leverage 有耦合）
        if band > 0:
            w = weight_band(w, band)

        prices = full.loc[start:end].dropna(how="all")
        if prices.empty:
            raise ValueError("no data for given filters")
        w = w.reindex(index=prices.index, columns=prices.columns).fillna(0.0)
        mask = mask_full.reindex(index=prices.index, columns=prices.columns).fillna(False)
        # 只统计回测窗口内的上限绑定比例（权重是在完整历史上算的）
        cap_share = cap_binding_share(raw_f, max_lev, prices.index)
        # 无意中背上的 carry 因子押注：trend/xsmom 都长期持正暴露，
        # 而它们没有任何一处显式表达过对 carry 的看法
        try:
            from qis.cli import _annualized_carry, carry_horizons
            cz = cross_sectional_z(
                _annualized_carry(self.store, self.universe,
                                  carry_horizons(self.store, self.universe))
                .rolling(21, min_periods=10).mean())
            carry_expo = float(factor_exposure(w, cz).reindex(prices.index).dropna().mean())
        except Exception:
            carry_expo = float("nan")

        cost = (cost_bps_by_name(self.universe.asset_classes(), self.settings["cost_bps"])
                if with_cost else 0.0)
        res = run_backtest(prices, w, cost_bps=cost, roll_mask=mask)
        stats = summary(res.net, res.turnover)
        rets = res.returns

        # 月度收益表（热力图）；转 float64 避免 pivot 产出 pd.NA
        monthly = (1.0 + res.net.fillna(0.0)).resample("ME").prod() - 1.0
        mtab = pd.DataFrame({"y": monthly.index.year, "m": monthly.index.month,
                             "r": monthly.values}).pivot(index="y", columns="m", values="r").astype(float)

        # 分标的年化贡献
        contrib = ((res.weights * rets).mean() * stats["ann_factor"]).sort_values()
        ac_map = self.universe.asset_classes()
        attribution = [{"name": n, "asset_class": ac_map.get(n, "default"),
                        "contrib": _jsonable(v)} for n, v in contrib.items()]

        # 各类别毛敞口时间序列（堆叠面积图；净额可正可负，堆叠会误导）
        # 按周均值降采样（显示粒度，避免日频调仓造成的视觉噪声）
        wbc = res.weights.abs().T.groupby(pd.Series(ac_map)).sum().T if len(res.weights.columns) else res.weights
        wbc = wbc.resample("W").mean().dropna(how="all")
        wbc.columns = [class_label(c) for c in wbc.columns]

        # 最新权重（绝对值前 15）
        last_w = res.weights.iloc[-1].sort_values(key=abs, ascending=False).head(15)

        win = int(round(stats["ann_factor"]))
        roll_sharpe = (res.net.rolling(win).mean()
                       / res.net.rolling(win).std() * math.sqrt(win))

        out = {
            "strategy": strategy,
            "params": {"start": start, "end": end, "vol_target": vol_target,
                       "gross": gross, "classes": list(classes or []),
                       "with_cost": with_cost, "band": band,
                       "max_leverage": max_lev},
            # 上限长期绑定 = 波动目标没生效，调 vol_target 不会有反应，前端要能看到
            "leverage_cap_share": _jsonable(cap_share),
            "carry_factor_exposure": _jsonable(carry_expo),
            "roll_suspect": sorted(
                n for n, d in self.roll_diagnostics().items()
                if n in prices.columns and not d["plausible"]),
            "metrics": {k: _jsonable(v) for k, v in stats.items()},
            "equity": _series_points(res.equity),
            "gross_equity": _series_points(res.gross_equity),
            "drawdown": _series_points(drawdown(res.equity), precision=5),
            "rolling_sharpe": _series_points(roll_sharpe.dropna(), precision=3),
            "monthly": {
                "years": [int(y) for y in mtab.index],
                "months": [int(m) for m in mtab.columns],
                "values": [[_jsonable(v) for v in row] for row in mtab.values],
            },
            "attribution": attribution,
            "weights_by_class": {
                "dates": [d.strftime("%Y-%m-%d") for d in wbc.index],
                "series": [{"name": c, "data": [_jsonable(v) for v in wbc[c]]}
                           for c in wbc.columns],
            },
            "latest_weights": [{"name": n, "weight": _jsonable(v)} for n, v in last_w.items()],
            "n_instruments": int(prices.shape[1]),
        }
        self._run_cache[key] = out
        return out


@lru_cache(maxsize=1)
def get_service() -> QISService:
    return QISService()
