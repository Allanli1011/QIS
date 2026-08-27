# -*- coding: utf-8 -*-
"""标的池：从 config/universe.yaml 加载 RIC 及其资产类别分组。"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

_DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "config" / "universe.yaml"


class Universe:
    """标的池。

    yaml 结构：
        instruments:
          - name: SP500
            ric: ESc1
            asset_class: equity_index
            carry_leg: ESc2        # 可选：carry 策略的远月腿
    """

    def __init__(self, instruments: list[dict]):
        self.instruments = instruments
        for inst in self.instruments:
            inst.setdefault("name", inst["ric"])
            inst.setdefault("asset_class", "default")

    @classmethod
    def from_yaml(cls, path: Optional[str | Path] = None) -> "Universe":
        p = Path(path) if path else _DEFAULT_CONFIG
        with open(p, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cls(cfg.get("instruments", []))

    # ---- 查询 ----
    def names(self) -> list[str]:
        return [i["name"] for i in self.instruments]

    def rics(self) -> list[str]:
        return [i["ric"] for i in self.instruments]

    def all_rics(self) -> list[str]:
        """主腿 + carry 腿的全部 RIC（去重，保持顺序）。"""
        seen: dict[str, None] = {}
        for i in self.instruments:
            seen.setdefault(i["ric"])
            if i.get("carry_leg"):
                seen.setdefault(i["carry_leg"])
        return list(seen)

    def asset_classes(self) -> dict[str, str]:
        """name -> asset_class"""
        return {i["name"]: i["asset_class"] for i in self.instruments}

    def ric_to_name(self) -> dict[str, str]:
        return {i["ric"]: i["name"] for i in self.instruments}

    def carry_legs(self) -> dict[str, str]:
        """name -> carry 腿 RIC（仅配置了 carry_leg 的标的）。"""
        return {i["name"]: i["carry_leg"] for i in self.instruments if i.get("carry_leg")}

    def by_class(self, asset_class: str) -> list[dict]:
        return [i for i in self.instruments if i["asset_class"] == asset_class]

    def __len__(self) -> int:
        return len(self.instruments)
