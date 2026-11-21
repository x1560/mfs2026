# -*- coding: utf-8 -*-
"""RQ4 图（第三版）：1:99 邻居池上的质量掩蔽，0–80%，Hit@5。

数据来自 `_diag/_mask_pool_<case>_<lvl>.log`（M-DCG）与
`_diag/_maskpool_lc_<case>_<lvl>.log` / `_maskpool_base_movies_*.log`（LinCombo）。
"""
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
LEVELS = [0, 20, 40, 60, 80]
CASES = [("Books", "books"), ("Movies", "movies"), ("S2018", "s2018")]
ROW = re.compile(r'^\s{2,}([a-z_]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$')


def grab(path, metric):
    if not os.path.exists(path):
        return None
    for line in open(path, encoding='utf-8', errors='replace'):
        m = ROW.match(line)
        if m and m.group(1) == metric:
            return float(m.group(3))       # Hit@5
    return None


def series(case, pattern, metric):
    ys = []
    for lvl in LEVELS:
        p = os.path.join(HERE, pattern % (case, lvl))
        ys.append(grab(p, metric))
    return ys


def main():
    out = os.path.join(HERE, "..", "..", "..", "pdf", "pdf", "rq4_masking_row.png")
    fig, axes = plt.subplots(1, 3, figsize=(3.5, 1.30), sharey=False)
    F = {"title": 6.5, "tick": 5.0, "xlabel": 6.0, "legend": 5.0}
    for ax, (name, case) in zip(axes, CASES):
        md = series(case, '_mask_pool_%s_%d.log', 'mdcg')
        lc = series(case, '_maskpool_lc_%s_%d.log', 'lincombo')
        if any(v is None for v in lc):
            # Movies 的线性组合在 base 那批日志里
            lc = series(case, '_maskpool_base_%s_%d.log', 'lincombo')
        xs = [l for l, v in zip(LEVELS, md) if v is not None]
        md = [v for v in md if v is not None]
        lc = [v for v in lc if v is not None]
        ax.plot(xs[:len(lc)], lc, color="tab:orange", lw=1.0, marker="o", ms=2.0,
                zorder=3, label="LinCombo")
        ax.plot(xs, md, color="tab:blue", lw=1.3, marker="o", ms=2.4,
                zorder=4, label="M-DCG")
        ax.set_title(name, fontsize=F["title"])
        ax.set_xticks([0, 20, 40, 60, 80])
        ax.tick_params(labelsize=F["tick"], pad=1.5)
        ax.grid(alpha=0.25, lw=0.6)
        lo, hi = ax.get_ylim()
        pad = (hi - lo) * 0.10
        ax.set_ylim(lo - pad, hi + pad)
    axes[0].set_ylabel("Hit@5", fontsize=F["xlabel"], labelpad=2)
    axes[1].set_xlabel("masked quality ratio (%)", fontsize=F["xlabel"], labelpad=1)
    handles = [Line2D([], [], color="tab:blue", lw=1.3, marker="o", ms=2.4),
               Line2D([], [], color="tab:orange", lw=1.0, marker="o", ms=2.0)]
    fig.legend(handles, ["M-DCG", "LinCombo"], fontsize=F["legend"],
               loc="lower center", bbox_to_anchor=(0.5, 0.012), ncol=2,
               frameon=False, handlelength=1.2, borderpad=0.0,
               labelspacing=0.2, columnspacing=1.4)
    fig.subplots_adjust(left=0.115, right=0.995, top=0.88, bottom=0.30, wspace=0.34)
    fig.savefig(out, dpi=300)
    print("saved", out)


if __name__ == "__main__":
    main()
