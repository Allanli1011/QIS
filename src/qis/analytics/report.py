# -*- coding: utf-8 -*-
"""
tearsheet：净值 / 回撤 / 滚动 Sharpe / 月度收益热力图。

使用非交互后端，输出 PNG。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qis.analytics.metrics import drawdown, sharpe, summary


def tearsheet(
    returns: pd.Series,
    turnover: Optional[pd.Series] = None,
    title: str = "QIS Strategy",
    ann_factor: int = 252,
    out_path: Optional[str | Path] = None,
) -> plt.Figure:
    r = returns.fillna(0.0)
    equity = (1.0 + r).cumprod()
    stats = summary(r, turnover, ann_factor)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"{title}   |   Sharpe {stats['sharpe']:.2f}   "
        f"AnnRet {stats['ann_return']:.1%}   AnnVol {stats['ann_vol']:.1%}   "
        f"MaxDD {stats['max_drawdown']:.1%}   Calmar {stats['calmar']:.2f}",
        fontsize=12,
    )

    # 1) 净值（对数轴）
    ax = axes[0, 0]
    equity.plot(ax=ax, logy=True, lw=1.2, title="Equity Curve (log)")
    ax.grid(alpha=0.3)

    # 2) 回撤
    ax = axes[0, 1]
    dd = drawdown(equity)
    ax.fill_between(dd.index, dd, 0, alpha=0.6)
    ax.set_title("Drawdown")
    ax.grid(alpha=0.3)

    # 3) 滚动 1 年 Sharpe
    ax = axes[1, 0]
    roll = r.rolling(ann_factor)
    roll_sharpe = roll.mean() / roll.std() * np.sqrt(ann_factor)
    roll_sharpe.plot(ax=ax, lw=1.0, title=f"Rolling {ann_factor}d Sharpe")
    ax.axhline(0, color="k", lw=0.8)
    ax.grid(alpha=0.3)

    # 4) 月度收益热力图
    ax = axes[1, 1]
    monthly = (1.0 + r).resample("ME").prod() - 1.0
    if len(monthly) >= 2:
        tab = pd.DataFrame({
            "y": monthly.index.year, "m": monthly.index.month, "r": monthly.values,
        }).pivot(index="y", columns="m", values="r")
        vals = tab.to_numpy(dtype=float, na_value=np.nan)
        lim = np.nanmax(np.abs(vals)) if np.isfinite(vals).any() else 1.0
        im = ax.imshow(vals, cmap="RdYlGn", aspect="auto", vmin=-lim, vmax=lim)
        ax.set_xticks(range(tab.shape[1]), tab.columns)
        ax.set_yticks(range(tab.shape[0]), tab.index)
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                v = vals[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.03)
    ax.set_title("Monthly Returns")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120)
    return fig
