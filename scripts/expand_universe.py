# -*- coding: utf-8 -*-
"""
标的池扩展：候选 RIC → LSEG 验证 → 全量拉取 → 重写 config/universe.yaml。

用法：uv run python scripts/expand_universe.py
候选 RIC 中可能有无效项，验证（拉 5 条日线）失败的自动剔除；
期货类标的自动配 c2 远月腿（验证同样失败则去腿保留主合约）。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qis.data.lseg import get_source
from qis.data.store import DataStore

# (name, front RIC, asset_class) —— 期货类自动补 c2 腿
CANDIDATES: list[tuple[str, str, str]] = [
    # ---- 股指期货 ----
    ("SP500", "ESc1", "equity_index"), ("NASDAQ", "NQc1", "equity_index"),
    ("RUSSELL", "RTYc1", "equity_index"), ("DOW", "YMc1", "equity_index"),
    ("ESTX50", "STXEc1", "equity_index"), ("DAX", "FDXc1", "equity_index"),
    ("CAC40", "FCEc1", "equity_index"), ("FTSE100", "FFIc1", "equity_index"),
    ("FTSEMIB", "FTMIBc1", "equity_index"), ("IBEX35", "IBEc1", "equity_index"),
    ("SMI", "FSMIc1", "equity_index"), ("AEX", "FTIc1", "equity_index"),
    ("OMX30", "OMXc1", "equity_index"), ("NIKKEI", "NIYc1", "equity_index"),
    ("TOPIX", "TOPXc1", "equity_index"), ("HSI", "HSIc1", "equity_index"),
    ("HSCEI", "HCEIc1", "equity_index"), ("KOSPI", "KSc1", "equity_index"),
    ("ASX200", "YAPc1", "equity_index"), ("MSCIEM", "MXEFc1", "equity_index"),
    ("VIX", "VXc1", "equity_index"), ("TSX60", "SXFc1", "equity_index"),
    ("BOVESPA", "INDc1", "equity_index"),
    # ---- 债券期货 ----
    ("UST2", "TUc1", "bond"), ("UST5", "FVc1", "bond"),
    ("UST10", "TYc1", "bond"), ("UST30", "USc1", "bond"),
    ("ULTRA10", "UXc1", "bond"), ("ULTRAB", "UBc1", "bond"),
    ("SCHATZ", "FGBSc1", "bond"), ("BOBL", "FGBMc1", "bond"),
    ("BUND", "FGBLc1", "bond"), ("BUXL", "FGBXc1", "bond"),
    ("BTP", "FBTPc1", "bond"), ("OAT", "FOATc1", "bond"),
    ("GILT", "FLGc1", "bond"), ("JGB10", "JGBc1", "bond"),
    ("CAN10", "CGBc1", "bond"), ("AUS10", "YTCc1", "bond"),
    # ---- 外汇现货 ----
    ("EURUSD", "EUR=", "fx"), ("USDJPY", "JPY=", "fx"),
    ("GBPUSD", "GBP=", "fx"), ("AUDUSD", "AUD=", "fx"),
    ("NZDUSD", "NZD=", "fx"), ("USDCAD", "CAD=", "fx"),
    ("USDCHF", "CHF=", "fx"), ("USDSEK", "SEK=", "fx"),
    ("USDNOK", "NOK=", "fx"), ("USDDKK", "DKK=", "fx"),
    ("USDCNH", "CNH=", "fx"), ("USDMXN", "MXN=", "fx"),
    ("USDBRL", "BRL=", "fx"), ("USDZAR", "ZAR=", "fx"),
    ("USDTRY", "TRY=", "fx"), ("USDKRW", "KRW=", "fx"),
    ("USDSGD", "SGD=", "fx"), ("USDTWD", "TWD=", "fx"),
    ("USDINR", "INR=", "fx"), ("USDIDR", "IDR=", "fx"),
    ("USDMYR", "MYR=", "fx"), ("USDPHP", "PHP=", "fx"),
    ("USDTHB", "THB=", "fx"), ("USDHUF", "HUF=", "fx"),
    ("USDPLN", "PLN=", "fx"), ("USDCZK", "CZK=", "fx"),
    ("USDILS", "ILS=", "fx"), ("USDCLP", "CLP=", "fx"),
    ("USDCOP", "COP=", "fx"), ("USDRUB", "RUB=", "fx"),
    # ---- 能源 ----
    ("WTI", "CLc1", "energy"), ("BRENT", "LCOc1", "energy"),
    ("NATGAS", "NGc1", "energy"), ("HEATOIL", "HOc1", "energy"),
    ("RBOB", "RBc1", "energy"), ("GASOIL", "LGOc1", "energy"),
    ("TTF", "TRNLTTFMc1", "energy"), ("EUA", "CFIEUc1", "energy"),
    ("COAL", "ATWMc1", "energy"),
    # ---- 金属 ----
    ("GOLD", "GCc1", "metal"), ("SILVER", "SIc1", "metal"),
    ("COPPER", "HGc1", "metal"), ("PLATINUM", "PLc1", "metal"),
    ("PALLADIUM", "PAc1", "metal"), ("ALUMINUM", "MALc1", "metal"),
    ("ZINC", "MZNc1", "metal"), ("NICKEL", "MNIc1", "metal"),
    ("IRONORE", "SZZFc1", "metal"),
    # ---- 农产品 ----
    ("CORN", "Cc1", "ags"), ("WHEAT", "Wc1", "ags"),
    ("SOYBEAN", "Sc1", "ags"), ("SOYMEAL", "SMc1", "ags"),
    ("SOYOIL", "BOc1", "ags"), ("KCWHEAT", "KWEc1", "ags"),
    ("SUGAR", "SBc1", "ags"), ("COFFEE", "KCc1", "ags"),
    ("COCOA", "CCc1", "ags"), ("COTTON", "CTc1", "ags"),
    ("LIVECATTLE", "LEc1", "ags"), ("LEANHOGS", "HEc1", "ags"),
    ("FEEDERCATTLE", "GFc1", "ags"), ("ROUGHRICE", "ZRc1", "ags"),
    ("OATS", "Oc1", "ags"), ("CANOLA", "RSc1", "ags"),
    ("OJ", "OJc1", "ags"), ("LUMBER", "LBc1", "ags"),
    ("PALMOIL", "FCPOc1", "ags"),
    # ---- 利率期货 ----
    ("SOFR3M", "SRAc1", "rates"), ("EURIBOR", "FEIc1", "rates"),
    ("SONIA3M", "SO3c1", "rates"),
    # ---- 加密 ----
    ("BTC", "BTCc1", "crypto"), ("ETH", "ETHc1", "crypto"),
]

FUTURES_CLASSES = {"equity_index", "bond", "energy", "metal", "ags", "rates", "crypto"}


def main() -> None:
    src = get_source()
    store = DataStore()

    # 1) 展开全部 RIC（主 + 腿）并逐条验证
    ric_meta: dict[str, tuple[str, str, bool]] = {}  # ric -> (name, class, is_leg)
    for name, ric, ac in CANDIDATES:
        ric_meta[ric] = (name, ac, False)
        if ac in FUTURES_CLASSES and ric.endswith("c1"):
            leg = ric[:-2] + "c2"
            ric_meta[leg] = (name, ac, True)

    ok: dict[str, int] = {}
    bad: dict[str, str] = {}
    rics = list(ric_meta)
    print(f"候选 {len(CANDIDATES)} 个标的 / {len(rics)} 条 RIC，开始验证…", flush=True)
    for i, ric in enumerate(rics, 1):
        try:
            df = src.history(ric, count=5)
            if len(df):
                ok[ric] = len(df)
            else:
                bad[ric] = "empty"
        except Exception as e:
            bad[ric] = str(e)[:80]
        if i % 20 == 0:
            print(f"  验证进度 {i}/{len(rics)}  ok={len(ok)} bad={len(bad)}", flush=True)
        time.sleep(0.05)
    print(f"验证完成：ok={len(ok)} bad={len(bad)}", flush=True)

    # 2) 幸存者全量拉取
    survivors: list[dict] = []
    for name, ric, ac in CANDIDATES:
        if ric not in ok:
            continue
        inst: dict = {"name": name, "ric": ric, "asset_class": ac}
        leg = ric[:-2] + "c2" if (ac in FUTURES_CLASSES and ric.endswith("c1")) else None
        if leg and leg in ok:
            inst["carry_leg"] = leg
        survivors.append(inst)

    print(f"幸存标的 {len(survivors)} 个，开始全量拉取…", flush=True)
    all_rics = sorted({i["ric"] for i in survivors} | {i.get("carry_leg") for i in survivors if i.get("carry_leg")})
    for i, ric in enumerate(all_rics, 1):
        try:
            df = store.update(ric, start="2000-01-01", source=src)
            print(f"  [{i}/{len(all_rics)}] {ric:<14} rows={len(df)}", flush=True)
        except Exception as e:
            print(f"  [{i}/{len(all_rics)}] {ric:<14} FAILED: {str(e)[:60]}", flush=True)

    # 3) 重写 universe.yaml + 验证报告
    order = ["equity_index", "bond", "fx", "energy", "metal", "ags", "rates", "crypto"]
    survivors.sort(key=lambda x: (order.index(x["asset_class"]), x["name"]))
    out = ROOT / "config" / "universe.yaml"
    header = (
        "# QIS 标的池（由 scripts/expand_universe.py 生成，RIC 均经 LSEG 验证）\n"
        "# ric: LSEG RIC（连续合约 c1/c2）；asset_class: 对应 settings.yaml 的 cost_bps；\n"
        "# carry_leg: 可选，carry 策略远月腿 + 换月调整\n"
    )
    with open(out, "w", encoding="utf-8") as f:
        f.write(header + "instruments:\n")
        for inst in survivors:
            line = f"  - {{ name: {inst['name']}, ric: {inst['ric']}, asset_class: {inst['asset_class']}"
            if inst.get("carry_leg"):
                line += f", carry_leg: {inst['carry_leg']}"
            f.write(line + " }\n")

    report = {"ok": ok, "bad": bad, "n_survivors": len(survivors)}
    (ROOT / "data" / "universe_validation.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"universe.yaml 已重写（{len(survivors)} 个标的）；"
          f"失败 {len(bad)} 条见 data/universe_validation.json", flush=True)


if __name__ == "__main__":
    main()
