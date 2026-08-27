# -*- coding: utf-8 -*-
"""
命令行入口。

  qis fetch-data            从 LSEG 拉取标的池日线，增量更新本地 parquet 缓存
  qis run --strategy trend  用缓存数据跑回测，输出指标与 tearsheet
  qis report                从已保存的收益 CSV 重新生成 tearsheet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from qis.analytics.metrics import infer_ann_factor, summary
from qis.analytics.report import tearsheet
from qis.backtest.costs import cost_bps_by_name, load_settings
from qis.backtest.engine import run_backtest
from qis.data.lseg import get_source
from qis.data.panel import clean_prices, liquid_mask, to_returns
from qis.data.roll import (adjust, adjusted_price_index, contract_horizon,
                           roll_diagnostics, roll_days)
from qis.data.store import DataStore
from qis.data.universe import Universe
from qis.portfolio.construction import (cap_binding_share, inverse_vol, normalize_gross,
                                        vol_target_factor, weight_band)
from qis.portfolio.risk import ewma_vol
from qis.data.curve import curve_matrix, curve_depth
from qis.strategy.carry import carry_raw, curve_carry, signals_from_raw
from qis.strategy.trend import trend_weights
from qis.strategy.xsmom import xsmom_weights

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="qis", description="QIS 策略研究回测平台")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch-data", help="从 LSEG 拉取标的池日线到本地缓存")
    f.add_argument("--universe", default=None, help="universe.yaml 路径")
    f.add_argument("--start", default="2000-01-01")
    f.add_argument("--end", default=None)

    r = sub.add_parser("run", help="跑策略回测")
    r.add_argument("--strategy", required=True, choices=["trend", "xsmom", "carry"])
    r.add_argument("--universe", default=None)
    r.add_argument("--start", default=None, help="回测起始日（默认取 settings）")
    r.add_argument("--end", default=None)
    r.add_argument("--vol-target", type=float, default=None, help="组合波动目标（默认取 settings）")
    r.add_argument("--gross", type=float, default=None, help="策略毛敞口（默认取 settings）")
    r.add_argument("--no-cost", action="store_true", help="不计交易成本")
    r.add_argument("--band", type=float, default=None, help="无交易带阈值（默认取 settings 的 rebal_band）")
    r.add_argument("--attrib", action="store_true", help="输出分标的年化盈亏贡献")
    r.add_argument("--tag", default=None, help="输出文件名后缀")
    r.add_argument("--out", default=str(REPORTS_DIR))

    t = sub.add_parser("report", help="从收益 CSV 生成 tearsheet")
    t.add_argument("--returns", required=True, help="run 输出的 *_returns.csv")
    t.add_argument("--title", default="QIS Strategy")
    t.add_argument("--out", default=None)

    s = sub.add_parser("serve", help="启动 Web 研究终端")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8600)
    return p


def cmd_serve(args: argparse.Namespace) -> int:
    from qis.web.api import main as web_main
    print(f"QIS Terminal: http://{args.host}:{args.port}")
    web_main(host=args.host, port=args.port)
    return 0


# ---------------------------------------------------------------- fetch-data
def cmd_fetch_data(args: argparse.Namespace) -> int:
    u = Universe.from_yaml(args.universe)
    store = DataStore()
    rics = u.all_rics()
    print(f"标的池 {len(u)} 个，共 {len(rics)} 条 RIC（含 carry 腿），开始增量更新…")
    store.update_many(rics, start=args.start, end=args.end)
    print("完成。")
    return 0


# ---------------------------------------------------------------- run
def _adjusted_price_matrix(store: DataStore, u: Universe,
                           with_roll_mask: bool = False,
                           with_diagnostics: bool = False):
    """
    换月调整后的价格矩阵（date × name）：
    有远月腿的标的按持仓量跳升识别换月日、用 c1(T)/c2(T-1) 修正当日收益再积成
    价格指数（见 qis.data.roll）；无腿标的（如 FX 现货）用原始收盘。

    with_roll_mask=True   额外返回换月交易日矩阵（date × name，bool），供引擎计换月成本；
    with_diagnostics=True 再额外返回换月识别体检字典（name → 方法/次数/是否合理）。
    """
    legs = u.carry_legs()
    cols: dict[str, pd.Series] = {}
    masks: dict[str, pd.Series] = {}
    diag: dict[str, dict] = {}
    for inst in u.instruments:
        name, ric = inst["name"], inst["ric"]
        f = store.load(ric)
        if f.empty:
            continue
        leg_ric = legs.get(name)
        d = store.load(leg_ric) if leg_ric else pd.DataFrame()
        # 先洗掉非正价格与单日往返尖刺，否则一个错价就能在杠杆下打穿净值
        fc = clean_prices(f["close"])
        if len(d) and "close" in d.columns:
            r, mask, method = adjust(fc, clean_prices(d["close"]), f.get("oi"), f.get("volume"))
            cols[name] = adjusted_price_index(r)
            masks[name] = mask
            if with_diagnostics:
                diag[name] = roll_diagnostics(fc, f.get("oi"), f.get("volume"))
        else:
            cols[name] = fc
    prices = pd.DataFrame(cols).sort_index()
    out = [prices]
    if with_roll_mask:
        # 逐列赋值而不是 reindex+fillna：后者在空/缺列时会落到 object dtype，
        # 触发 pandas 的 downcasting FutureWarning
        rm = pd.DataFrame(False, index=prices.index, columns=prices.columns)
        for name, m in masks.items():
            if name in rm.columns:
                rm[name] = m.reindex(prices.index, fill_value=False).astype(bool)
        out.append(rm)
    if with_diagnostics:
        out.append(diag)
    return out[0] if len(out) == 1 else tuple(out)


def _strategy_weights(name: str, prices: pd.DataFrame, u: Universe,
                      store: DataStore, gross: float,
                      raw_quotes: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    策略权重，并按报价流动性掩码剔除断档期的标的。

    raw_quotes 是"哪天有真实报价"的依据（缺省时退回 prices 自身）——
    prices 是已经前值填充过的价格指数，看不出哪些是假期填出来的。
    """
    w = _raw_strategy_weights(name, prices, u, store, gross)
    liq = liquid_mask(raw_quotes if raw_quotes is not None else prices)
    liq = liq.reindex(index=w.index, columns=w.columns).fillna(False)
    return normalize_gross(w.where(liq, 0.0), gross)


def carry_horizons(store: DataStore, u: Universe) -> pd.Series:
    """
    每个标的 c1/c2 两腿到期日的间隔（年），用于把 carry 年化。

    换月识别不可靠的标的（次数离谱 → 间隔估计也不可信）用**同资产类别中位数**兜底，
    整类都没有时退回 0.25 年（季度合约）。
    """
    ac = u.asset_classes()
    est: dict[str, float] = {}
    for i in u.instruments:
        name, ric = i["name"], i["ric"]
        if not i.get("carry_leg"):
            continue
        f = store.load(ric)
        if f.empty:
            continue
        d = roll_diagnostics(f["close"], f.get("oi"), f.get("volume"))
        if not d["plausible"]:
            continue                      # 识别不可信 → 不用它的间隔估计
        mask, _ = roll_days(f["close"], f.get("oi"), f.get("volume"))
        h = contract_horizon(mask)
        if h == h and h > 0:
            est[name] = h
    by_ac: dict[str, list[float]] = {}
    for n, h in est.items():
        by_ac.setdefault(ac.get(n, "default"), []).append(h)
    med = {k: float(np.median(v)) for k, v in by_ac.items()}
    out = {}
    for i in u.instruments:
        if not i.get("carry_leg"):
            continue
        n = i["name"]
        out[n] = est.get(n, med.get(ac.get(n, "default"), 0.25))
    return pd.Series(out, dtype=float)


def _annualized_carry(store: DataStore, u: Universe, horizon: pd.Series,
                      depth: int = 4, min_legs: int = 3) -> pd.DataFrame:
    """
    年化 carry 矩阵（date × name）。

    优先用**整条曲线的回归斜率**：N 点回归斜率的方差约为两点差分的 1/N，
    这是 c1/c2 信噪比问题（corr(信噪比, IC) = −0.53）的根治办法。
    实测本池年化 carry 的绝对日噪声降低 2~4 倍（SILVER 0.060 → 0.017）。

    月份取舍：**能去掉近月就去掉**。c1 临近到期时有收敛与流动性效应，
    是曲线上噪声最大的一点（SILVER 用 (2,3,4) 比 (1,2,3,4) 再干净 1.6 倍）。
    有 4 个月份的标的用 (2,3,4)，只有 3 个月份的用 (1,2,3)，
    都不够的回退到 c1/c2 两点价差。
    """
    futs = [i for i in u.instruments if i.get("carry_leg")]
    curves = curve_matrix(store, futs, depth=depth)

    legs = u.carry_legs()
    leg_rics = {v: k for k, v in legs.items()}
    front = clean_prices(store.load_close_matrix(u.rics()).rename(columns=u.ric_to_name()))
    defer = clean_prices(store.load_close_matrix(list(leg_rics)).rename(columns=leg_rics))
    two_pt = carry_raw(front, defer, horizon=horizon)
    if not curves:
        return two_pt

    deep = curve_depth(curves)
    near = curve_carry(curves, horizon, use_legs=tuple(range(1, min_legs + 1)),
                       min_legs=min_legs)                      # 含近月，覆盖广
    far = curve_carry(curves, horizon, use_legs=tuple(range(2, depth + 1)),
                      min_legs=min_legs)                       # 去掉近月，更干净
    out = two_pt.reindex(index=two_pt.index.union(near.index)).copy()
    for c in out.columns:
        d = int(deep.get(c, 0))
        if d > min_legs and c in far.columns:
            out[c] = far[c].reindex(out.index)
        elif d >= min_legs and c in near.columns:
            out[c] = near[c].reindex(out.index)
    return out


def _raw_strategy_weights(name: str, prices: pd.DataFrame, u: Universe,
                          store: DataStore, gross: float) -> pd.DataFrame:
    if name == "trend":
        return trend_weights(prices, gross=gross)
    if name == "xsmom":
        # 不按资产类别分组：本池的类别太小（rates/crypto 各 2 个会被整组置零），
        # 组内排名支撑不起截面动量，靠的是全池广度。见 strategy/xsmom.py。
        return xsmom_weights(prices, gross=gross)
    if name == "carry":
        # 年化 carry 优先用整条曲线的回归斜率（信噪比远高于两点价差），
        # 曲线不够深的标的回退到 c1/c2。波动估计用换月调整后的收益。
        H = carry_horizons(store, u)
        raw = _annualized_carry(store, u, H)
        sig = signals_from_raw(raw, columns=prices.columns)
        vol = ewma_vol(to_returns(prices))
        # 必须连 index 一起对齐：raw_front 是未切窗口的全历史，只对齐列会让权重
        # 带着回测窗口之前的几千行进入后续的波动目标缩放，污染 EWMA 预热。
        sig = sig.reindex(index=prices.index, columns=prices.columns).fillna(0.0)
        w = inverse_vol(sig, vol, gross=gross)
        # 无远月腿的标的（FX 现货等）没有期限结构，权重保持 0
        with_leg = set(u.carry_legs())
        no_leg = [c for c in prices.columns if c not in with_leg]
        if no_leg:
            w[no_leg] = 0.0
            w = normalize_gross(w, gross)
        return w
    raise ValueError(f"unknown strategy: {name}")


def cmd_run(args: argparse.Namespace) -> int:
    settings = load_settings()
    bt = settings["backtest"]
    start = args.start or bt["start"]
    end = args.end or bt.get("end")
    vol_target = args.vol_target if args.vol_target is not None else bt["vol_target"]
    gross = args.gross if args.gross is not None else bt.get("gross", 1.0)

    u = Universe.from_yaml(args.universe)
    store = DataStore()
    full, roll_mask_full, diag = _adjusted_price_matrix(
        store, u, with_roll_mask=True, with_diagnostics=True)
    if full.empty:
        print("本地缓存为空，请先运行：qis fetch-data")
        return 1

    # 信号、波动目标、无交易带全部在**完整历史**上计算，最后才切到回测窗口。
    # 先切窗口会让 start 之后的第一年被 lookback 预热吃掉
    # （trend 需 252 日，从 2010 起的回测里 2010 全年空仓）。
    w_raw = _strategy_weights(args.strategy, full, u, store, gross)
    band = args.band if args.band is not None else bt.get("rebal_band", 0.0)
    rets_full = to_returns(full)
    max_lev = float(bt.get("max_leverage", 3.0))
    af = infer_ann_factor(rets_full.iloc[:, 0] if full.shape[1] else pd.Series(dtype=float),
                          bt["ann_factor"])
    scale_f, raw_f = vol_target_factor(w_raw, rets_full, vol_target, bt["vol_lookback"],
                                       af, max_lev)
    w = w_raw.mul(scale_f, axis=0).where(scale_f.notna(), w_raw)
    # 无交易带作用在杠杆之后的实际持仓上（与历史标定一致）。
    # 注意：阈值是绝对权重单位，会随杠杆水平改变其相对强度——
    # 若把 max_leverage 调高到上限不再绑定，带会开始阻挡波动目标的降杠杆动作
    # （实测 max_leverage=10 时 trend 波动从 12% 被顶到 22%）。改杠杆时要一并重标定。
    if band > 0:
        w = weight_band(w, band)

    front = full.loc[start:end].dropna(how="all")
    w = w.reindex(index=front.index, columns=front.columns).fillna(0.0)
    roll_mask = roll_mask_full.reindex(index=front.index, columns=front.columns).fillna(False)
    # 只统计回测窗口内的上限绑定比例（权重是在完整历史上算的）
    cap_share = cap_binding_share(raw_f, max_lev, front.index)

    # 成本
    if args.no_cost:
        cost: float | dict[str, float] = 0.0
    else:
        cost = cost_bps_by_name(u.asset_classes(), settings["cost_bps"])

    res = run_backtest(front, w, cost_bps=cost, roll_mask=roll_mask)
    stats = summary(res.net, res.turnover)

    tag = f"_{args.tag}" if args.tag else ""
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.strategy}{tag}"
    csv_path = out_dir / f"{stem}_returns.csv"
    png_path = out_dir / f"{stem}_tearsheet.png"

    pd.DataFrame({
        "net": res.net, "gross": res.gross,
        "turnover": res.turnover, "cost": res.cost,
        "equity": res.equity,
    }).to_csv(csv_path)
    tearsheet(res.net, res.turnover, title=f"{args.strategy} (vol target {vol_target:.0%})",
              ann_factor=bt["ann_factor"], out_path=png_path)

    print(f"\n== {args.strategy} 回测 {front.index[0].date()} → {front.index[-1].date()} ==")
    for k, v in stats.items():
        if k in ("ann_return", "ann_vol", "max_drawdown", "hit_ratio"):
            print(f"  {k:<14} {v:>10.2%}")
        elif k == "n_days":
            print(f"  {k:<14} {int(v):>10d}")
        else:
            print(f"  {k:<14} {v:>10.2f}")
    if cap_share == cap_share and cap_share > 0.5:
        print(f"\n  ! 杠杆上限({max_lev:g}×)在 {cap_share:.0%} 的交易日绑定，"
              f"波动目标 {vol_target:.0%} 实际够不着——此时调 --vol-target 不会有反应")

    print(f"\n收益序列: {csv_path}\ntearsheet: {png_path}")

    suspect = {n: d for n, d in diag.items() if not d["plausible"]}
    if suspect:
        print(f"\n== 换月识别存疑 {len(suspect)}/{len(diag)} 个标的（其收益未被可靠换月调整）==")
        for n, d in sorted(suspect.items(), key=lambda kv: -kv[1]["per_year"])[:10]:
            print(f"  {n:<13}方法={d['method']:<8}{d['per_year']:>6.1f} 次/年")

    if args.attrib:
        contrib = (res.weights * res.returns).mean() * stats["ann_factor"]
        contrib = contrib.sort_values()
        print("\n== 分标的年化贡献（首尾各 5） ==")
        for name in list(contrib.index[:5]) + list(contrib.index[-5:]):
            print(f"  {name:<10} {contrib[name]:>+8.2%}")
    return 0


# ---------------------------------------------------------------- report
def cmd_report(args: argparse.Namespace) -> int:
    df = pd.read_csv(args.returns, index_col=0, parse_dates=True)
    out = args.out or str(Path(args.returns).with_name(
        Path(args.returns).name.replace("_returns.csv", "_tearsheet.png")))
    turnover = df["turnover"] if "turnover" in df.columns else None
    tearsheet(df["net"], turnover, title=args.title, out_path=out)
    print(f"tearsheet: {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.cmd == "fetch-data":
        return cmd_fetch_data(args)
    if args.cmd == "run":
        return cmd_run(args)
    if args.cmd == "report":
        return cmd_report(args)
    if args.cmd == "serve":
        return cmd_serve(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
