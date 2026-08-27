# -*- coding: utf-8 -*-
"""
Walk-forward 验证：每年末只用当时已有的数据选配置，交易下一年，拼接样本外收益。

用法：uv run python scripts/walkforward.py [--start-oos 2014]

为什么需要：回测里报出的 Sharpe 是在**看过全样本之后**选定的配置上算的，
无论选择过程多克制，都带着选择偏差。这个脚本给出诚实的样本外预期：
把每一个做过的选择（trend 的信号构造与快慢对、xsmom 的回看期与排名口径）
都放进候选集，逐年重选，只统计次年的收益。

本池 2015-2026 的结果（见 README）：
    trend  全样本 +0.99 → 样本外 +0.83   （偏差 0.15）
    xsmom  全样本 +0.79 → 样本外 +0.53   （偏差 0.26）
    合成   全样本 +0.94 → 样本外 +0.74   （偏差 0.20）
xsmom 的偏差约为 trend 的两倍——因为它的回看期是看着数据定的，
而 trend 的快慢阶梯用的是 CTA 惯例（1:3 几何序列），没有拿数据挑。
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from qis.analytics.metrics import infer_ann_factor
from qis.backtest.costs import cost_bps_by_name
from qis.backtest.engine import run_backtest
from qis.data.panel import liquid_mask, to_returns
from qis.portfolio.construction import (inverse_vol, normalize_gross,
                                        vol_target_factor, weight_band)
from qis.portfolio.risk import ewma_vol
from qis.strategy.trend import EWMAC_SPANS, trend_signals
from qis.strategy.xsmom import xsmom_weights
from qis.web.service import QISService

SHIPPED = {"trend": "trend:ewmac_std", "xsmom": "xsmom:ra252"}


def build_candidates(svc: QISService, gross: float, vol: pd.DataFrame) -> dict:
    """候选集 = 我在研究过程中做过的每一个选择。"""
    full = svc.prices()
    out: dict[str, pd.DataFrame] = {}
    spans = {"ewmac_std": EWMAC_SPANS,
             "ewmac_slow": ((16, 48), (32, 96), (64, 192)),
             "ewmac_fast": ((8, 24), (16, 48), (32, 96)),
             "ewmac_1to4": ((16, 64), (32, 128), (64, 256)),
             "ewmac_single": ((32, 96),)}
    for k, sp in spans.items():
        out[f"trend:{k}"] = inverse_vol(trend_signals(full, spans=sp), vol, gross=gross)
    for lb in [(21, 63, 126, 252), (63, 126, 252), (21, 63), (126, 252)]:
        out[f"trend:sign{lb}"] = inverse_vol(
            trend_signals(full, method="sign", lookbacks=lb), vol, gross=gross)
    for lb in (63, 126, 189, 252, 378, 504):
        out[f"xsmom:ra{lb}"] = xsmom_weights(full, lookback=lb, gross=gross)
        out[f"xsmom:raw{lb}"] = xsmom_weights(full, lookback=lb,
                                              risk_adjusted=False, gross=gross)
    out["xsmom:byclass252"] = xsmom_weights(
        full, lookback=252, groups=svc.universe.asset_classes(), gross=gross)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-oos", type=int, default=2014, help="第一个用于选配置的年末")
    ap.add_argument("--start", default="2010-01-01")
    args = ap.parse_args()

    svc = QISService()
    u, bt_cfg = svc.universe, svc.settings["backtest"]
    full = svc.prices()
    rets = to_returns(full)
    mask = svc.roll_mask()
    cost = cost_bps_by_name(u.asset_classes(), svc.settings["cost_bps"])
    liq = liquid_mask(full)
    vol = ewma_vol(rets)
    gross = float(bt_cfg.get("gross", 1.0))
    max_lev = float(bt_cfg.get("max_leverage", 3.0))
    af = infer_ann_factor(rets.iloc[:, 0], bt_cfg["ann_factor"])

    def net_return(w_raw: pd.DataFrame) -> pd.Series:
        w = normalize_gross(w_raw.where(liq.reindex_like(w_raw).fillna(False), 0.0), gross)
        sc, _ = vol_target_factor(w, rets, bt_cfg["vol_target"], bt_cfg["vol_lookback"],
                                  af, max_lev)
        w = w.mul(sc, axis=0).where(sc.notna(), w)
        w = weight_band(w, bt_cfg.get("rebal_band", 0.0))
        p = full.loc[args.start:].dropna(how="all")
        ww = w.reindex(index=p.index, columns=p.columns).fillna(0.0)
        m = mask.reindex(index=p.index, columns=p.columns).fillna(False)
        return run_backtest(p, ww, cost_bps=cost, roll_mask=m).net

    def sharpe(x: pd.Series) -> float:
        x = x.dropna()
        return float(x.mean() / x.std() * np.sqrt(af)) if len(x) > 20 and x.std() > 0 else np.nan

    cands = build_candidates(svc, gross, vol)
    print(f"候选配置 {len(cands)} 个，计算各自收益…", flush=True)
    R = pd.DataFrame({k: net_return(w) for k, w in cands.items()}).dropna(how="all")

    fams = {f: [c for c in R.columns if c.startswith(f + ":")] for f in ("trend", "xsmom")}
    years = sorted(set(R.index.year))
    oos: dict[str, list] = {f: [] for f in fams}
    picks: dict[str, list] = {f: [] for f in fams}

    print(f"\n{'交易年':<8}{'trend 选中':<24}{'SR':>7}   {'xsmom 选中':<18}{'SR':>7}")
    for y in years:
        if y < args.start_oos or y + 1 not in years:
            continue
        row = [str(y + 1)]
        for f in fams:
            best = R.loc[:f"{y}-12-31", fams[f]].apply(sharpe).idxmax()
            nxt = R.loc[f"{y+1}-01-01":f"{y+1}-12-31", best]
            oos[f].append(nxt)
            picks[f].append(best)
            row += [best.split(":")[1], f"{sharpe(nxt):+.2f}"]
        print(f"{row[0]:<8}{row[1]:<24}{row[2]:>7}   {row[3]:<18}{row[4]:>7}")

    wf = {f: pd.concat(oos[f]) for f in fams}
    per = wf["trend"].index
    blend_wf = (wf["trend"].reindex(per) + wf["xsmom"].reindex(per)) / 2
    blend_ship = (R.loc[per, SHIPPED["trend"]] + R.loc[per, SHIPPED["xsmom"]]) / 2

    print(f"\n{'':<10}{'全样本':>12}{'样本外(WF)':>14}{'选择偏差':>10}")
    for f in fams:
        print(f"{f:<10}{sharpe(R[SHIPPED[f]]):>+12.3f}{sharpe(wf[f]):>+14.3f}"
              f"{sharpe(R[SHIPPED[f]]) - sharpe(wf[f]):>10.3f}")
    print(f"{'合成':<10}{sharpe((R[SHIPPED['trend']] + R[SHIPPED['xsmom']]) / 2):>+12.3f}"
          f"{sharpe(blend_wf):>+14.3f}"
          f"{sharpe((R[SHIPPED['trend']]+R[SHIPPED['xsmom']])/2) - sharpe(blend_wf):>10.3f}")

    eq = (1 + blend_wf).cumprod()
    dd = float((eq / eq.cummax() - 1).min())
    yb = blend_wf.groupby(blend_wf.index.year).apply(sharpe)
    print(f"\n合成样本外：年化 {eq.iloc[-1] ** (1 / ((per[-1]-per[0]).days/365.25)) - 1:+.2%}"
          f"  最大回撤 {dd:.1%}  为正年份 {int((yb > 0).sum())}/{len(yb)}")
    print(f"（同期交付的固定配置 SR {sharpe(blend_ship):+.3f}）")
    for f in fams:
        c = Counter(p.split(":")[1] for p in picks[f])
        print(f"  {f} 各年选中: {dict(c)}")


if __name__ == "__main__":
    main()
