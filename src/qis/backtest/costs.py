# -*- coding: utf-8 -*-
"""交易成本：按资产类别配置单边成本率（bps）。"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

import yaml

_DEFAULT_SETTINGS = Path(__file__).resolve().parents[3] / "config" / "settings.yaml"


def load_settings(path: Optional[str | Path] = None) -> dict:
    p = Path(path) if path else _DEFAULT_SETTINGS
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def cost_bps_by_name(
    asset_classes: Mapping[str, str],
    cost_bps: Mapping[str, float],
) -> dict[str, float]:
    """
    instrument name → 单边成本（bps）。
    按资产类别查表，类别未配置时用 "default"。
    """
    default = float(cost_bps.get("default", 0.0))
    return {
        name: float(cost_bps.get(ac, default))
        for name, ac in asset_classes.items()
    }
