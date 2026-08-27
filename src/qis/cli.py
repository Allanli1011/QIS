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

import pandas as pd

from qis.analytics.metrics import summary
from qis.analytics.report import tearsheet
from qis.backtest.costs import cost_bps_by_name, load_settings
from qis.backtest.engine import run_backtest
from qis.data.lseg import get_source
from qis.data.roll import adjusted_price_index, adjusted_returns, roll_starts, roll_window
from qis.data.store import DataStore
from qis.data.universe import Universe
from qis.portfolio.construction import inverse_vol, normalize_gross, vol_target_scale, weight_band
from qis.portfolio.risk import ewma_vol
from qis.strategy.carry import carry_signals
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
    r.add_argument("--gross", type=float, default=1.0, help="策略毛敞口")
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
def v1_ric(ric: str) -> str | None:
    """c1 连续合约对应的 v1（成交量拼接）RIC；非 c1 返回 None。"""
    return ric[:-2] + "v1" if ric.endswith("c1") else None


def cmd_fetch_data(args: argparse.Namespace) -> int:
    u = Universe.from_yaml(args.universe)
    store = DataStore()
    rics = u.all_rics()
    print(f"标的池 {len(u)} 个，共 {len(rics)} 条 RIC（含 carry 腿），开始增量更新…")
    store.update_many(rics, start=args.start, end=args.end)
    # 附带更新 v1 序列（成交量拼接，供换月窗口识别；无权限/无此序列的自动跳过）
    v1s = [v for r in u.rics() if (v := v1_ric(r))]
    print(f"尝试更新 {len(v1s)} 条 v1 序列…")
    src = get_source()
    ok = 0
    for ric in v1s:
        try:
            df = store.update(ric, start=args.start, end=args.end, source=src)
            ok += bool(len(df))
        except Exception:
            pass  # 无 v1 的标的静默跳过
    print(f"完成（v1 可用 {ok}/{len(v1s)}）。")
    return 0


# ---------------------------------------------------------------- run
def _adjusted_price_matrix(store: DataStore, u: Universe,
                           with_roll_mask: bool = False):
    """
    换月调整后的价格矩阵（date × name）：
    有远月腿的标的用换月窗口（优先 v1 背离法，回退成交量规则）修正收益再积成
    价格指数；无腿标的（如 FX 现货）用原始收盘。

    with_roll_mask=True 时额外返回换月交易日矩阵（date × name，bool），
    供引擎计换月成本。
    """
    legs = u.carry_legs()
    cols: dict[str, pd.Series] = {}
    masks: dict[str, pd.Series] = {}
    for inst in u.instruments:
        name, ric = inst["name"], inst["ric"]
        f = store.load(ric)
        if f.empty:
            continue
        leg_ric = legs.get(name)
        d = store.load(leg_ric) if leg_ric else pd.DataFrame()
        vr = v1_ric(ric)
        v1 = store.load(vr) if vr else pd.DataFrame()
        v1_close = v1["close"] if len(v1) and "close" in v1.columns else None
        if len(d):
            window = roll_window(
                f["close"],
                f.get("volume"), d.get("volume"),
                v1_close=v1_close,
            )
            r = adjusted_returns(f["close"], d["close"], f.get("volume"), d.get("volume"),
                                 v1_close=v1_close)
            cols[name] = adjusted_price_index(r)
            if with_roll_mask and window is not None:
                masks[name] = roll_starts(window)
        else:
            cols[name] = f["close"]
    prices = pd.DataFrame(cols).sort_index()
    if not with_roll_mask:
        return prices
    roll_mask = pd.DataFrame(masks, index=prices.index).reindex(columns=prices.columns).fillna(False)
    return prices, roll_mask
    return pd.DataFrame(cols).sort_index()


def _strategy_weights(name: str, prices: pd.DataFrame, u: Universe,
                      store: DataStore, gross: float) -> pd.DataFrame:
    if name == "trend":
        return trend_weights(prices, gross=gross)
    if name == "xsmom":
        return xsmom_weights(prices, groups=u.asset_classes(), gross=gross)
    if name == "carry":
        # 信号用原始近/远月价差（水平关系），波动估计用调整后收益
        legs = u.carry_legs()
        leg_rics = {v: k for k, v in legs.items()}  # leg ric -> name
        raw_front = store.load_close_matrix(u.rics()).rename(columns=u.ric_to_name())
        deferred = store.load_close_matrix(list(leg_rics)).rename(columns=leg_rics)
        sig = carry_signals(raw_front, deferred)  # 默认时序模式
        vol = ewma_vol(prices.pct_change(fill_method=None))
        w = inverse_vol(sig.reindex(columns=prices.columns, fill_value=0.0), vol, gross=gross)
        no_leg = [c for c in prices.columns if c not in deferred.columns]
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

    u = Universe.from_yaml(args.universe)
    store = DataStore()
    front, roll_mask = _adjusted_price_matrix(store, u, with_roll_mask=True)
    if front.empty:
        print("本地缓存为空，请先运行：qis fetch-data")
        return 1
    roll_mask = roll_mask.loc[start:end]
    front = front.loc[start:end].dropna(how="all")

    # 策略权重 → 波动目标缩放 → 无交易带
    w = _strategy_weights(args.strategy, front, u, store, args.gross)
    rets = front.pct_change(fill_method=None)
    w = vol_target_scale(w, rets, target=vol_target, span=bt["vol_lookback"],
                         ann_factor=bt["ann_factor"])
    band = args.band if args.band is not None else bt.get("rebal_band", 0.0)
    if band > 0:
        w = weight_band(w, band)

    # 成本
    if args.no_cost:
        cost: float | dict[str, float] = 0.0
    else:
        cost = cost_bps_by_name(u.asset_classes(), settings["cost_bps"])

    res = run_backtest(front, w, cost_bps=cost, roll_mask=roll_mask)
    stats = summary(res.net, res.turnover, bt["ann_factor"])

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
    print(f"\n收益序列: {csv_path}\ntearsheet: {png_path}")

    if args.attrib:
        rets = front.pct_change(fill_method=None)
        contrib = (res.weights * rets.fillna(0.0)).mean() * bt["ann_factor"]
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
