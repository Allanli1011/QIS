# -*- coding: utf-8 -*-
"""
拉取多月合约曲线（c3 及以后）到本地缓存。

用法：uv run python scripts/fetch_curve.py [--depth 4] [--start 2000-01-01]

c1/c2 由 `qis fetch-data` 负责；本脚本只补 c3..c{depth}。
逐条验证后才全量拉取，失败的静默跳过（不是所有品种都有那么深的曲线）。

为什么需要：两点价差（c1/c2）估 carry 的信噪比极差——本池实测
corr(信噪比, IC) = −0.53，噪声大的品种 carry 信号系统性做反。
N 点回归斜率的方差约为两点差分的 1/N（实测 4 点降噪 3.1×、6 点 6.1×）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qis.data.curve import curve_ric
from qis.data.lseg import get_source
from qis.data.store import DataStore
from qis.data.universe import Universe


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=4, help="拉到第几个月（含）")
    ap.add_argument("--start", default="2000-01-01")
    ap.add_argument("--universe", default=None)
    ap.add_argument("--report", default=str(ROOT / "data" / "curve_depth.json"))
    args = ap.parse_args()

    u = Universe.from_yaml(args.universe)
    store = DataStore()
    src = get_source()

    futs = [i for i in u.instruments if i.get("carry_leg")]
    targets: list[tuple[str, str]] = []
    for i in futs:
        for n in range(3, args.depth + 1):
            r = curve_ric(i["ric"], n)
            if r:
                targets.append((i["name"], r))

    print(f"{len(futs)} 个期货品种，需补 {len(targets)} 条 c3~c{args.depth} 序列", flush=True)
    depth: dict[str, int] = {i["name"]: 2 for i in futs}
    ok = bad = 0
    for k, (name, ric) in enumerate(targets, 1):
        try:
            df = store.update(ric, start=args.start, source=src)
            if len(df):
                n = int(ric.rsplit("c", 1)[1])
                depth[name] = max(depth[name], n)
                ok += 1
                print(f"  [{k}/{len(targets)}] {ric:<12} rows={len(df):>6} "
                      f"last={df.index[-1].date()}", flush=True)
            else:
                bad += 1
                print(f"  [{k}/{len(targets)}] {ric:<12} 空", flush=True)
        except Exception as e:
            bad += 1
            print(f"  [{k}/{len(targets)}] {ric:<12} 失败: {str(e)[:60]}", flush=True)
        time.sleep(0.05)

    Path(args.report).write_text(
        json.dumps(depth, indent=1, ensure_ascii=False), encoding="utf-8")
    from collections import Counter
    print(f"\n完成：成功 {ok}，跳过 {bad}", flush=True)
    print("曲线深度分布:", dict(sorted(Counter(depth.values()).items())), flush=True)
    print(f"深度报告: {args.report}", flush=True)


if __name__ == "__main__":
    main()
