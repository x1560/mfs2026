# -*- coding: utf-8 -*-
"""RQ4 掩蔽曲线图（第二版）。

和第一版的区别：
1. 所有线都用**圆点**，只靠颜色区分（不再用方块/虚线区分 LinCombo）；
2. **每个数据集用自己的纵轴范围**（sharey=False），否则 S2018 的 0.86 会把另外两个压扁。

x = 质量因子被遮蔽的比例 0..100%，y = within-user Hit@2。
"""
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
PREFIX = sys.argv[2] if len(sys.argv) > 2 else "_rq4paper_"
CASES = [
    ("Books", PREFIX + "books.log"),
    ("Movies", PREFIX + "movies.log"),
    ("S2018", PREFIX + "s2018.log"),
]

RE_MAIN = re.compile(r"^\s+within-user: Hit@2=([\d.]+)")
RE_SUB = re.compile(r"^\s+\[(\w+)\s*\] within-user: Hit@2=([\d.]+)")


def parse(path):
    blocks, cur = [], None
    for line in open(path, "r", encoding="utf-8", errors="replace"):
        m = RE_MAIN.match(line)
        if m:
            if cur is not None:
                blocks.append(cur)
            cur = {"M-DCG": float(m.group(1))}
            continue
        m = RE_SUB.match(line)
        if m and cur is not None:
            cur[m.group(1)] = float(m.group(2))
    if cur is not None:
        blocks.append(cur)
    return blocks


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        HERE, "..", "..", "pdf", "rq4_masking_row.png")
    fig, axes = plt.subplots(1, 3, figsize=(3.5, 1.32), sharey=False)
    F = {"title": 6.5, "tick": 5.0, "xlabel": 6.0, "legend": 5.0}
    factors = ["rel", "dif", "nov", "unpop"]
    for ax, (name, fn) in zip(axes, CASES):
        p = os.path.join(HERE, fn)
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            ax.set_title(name + " (missing)", fontsize=F["title"])
            continue
        bl = parse(p)
        n = len(bl)
        if n == 0:
            ax.set_title(name + " (empty)", fontsize=F["title"])
            continue
        xs = [100.0 * i / (n - 1) for i in range(n)] if n > 1 else [0.0]
        for fac in factors:
            ax.plot(xs, [b.get(fac, float("nan")) for b in bl],
                    color="0.72", lw=0.6, marker="o", ms=1.6, zorder=1)
        if "lincombo" in bl[0]:
            ax.plot(xs, [b["lincombo"] for b in bl], color="tab:orange", lw=1.0,
                    marker="o", ms=2.0, zorder=3, label="LinCombo")
        ax.plot(xs, [b["M-DCG"] for b in bl], color="tab:blue", lw=1.3,
                marker="o", ms=2.4, zorder=4, label="M-DCG")
        ax.set_title(name, fontsize=F["title"])
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.tick_params(labelsize=F["tick"], pad=1.5)
        ax.grid(alpha=0.25, lw=0.6)
        # 每个面板按自己的数据自动定范围，上下各留 8% 余量
        lo, hi = ax.get_ylim()
        pad = (hi - lo) * 0.08
        ax.set_ylim(lo - pad, hi + pad)
    axes[0].set_ylabel("within-user Hit@2", fontsize=F["xlabel"], labelpad=2)
    axes[1].set_xlabel("masked fraction (%)", fontsize=F["xlabel"], labelpad=1)
    handles = [Line2D([], [], color="tab:blue", lw=1.3, marker="o", ms=2.4),
               Line2D([], [], color="tab:orange", lw=1.0, marker="o", ms=2.0),
               Line2D([], [], color="0.72", lw=0.6, marker="o", ms=1.6)]
    fig.legend(handles, ["M-DCG", "LinCombo", "single factor"],
               fontsize=F["legend"], loc="lower center",
               bbox_to_anchor=(0.5, 0.012), ncol=3, frameon=False,
               handlelength=1.2, borderpad=0.0, labelspacing=0.2,
               columnspacing=1.4)
    fig.subplots_adjust(left=0.115, right=0.995, top=0.88, bottom=0.30,
                        wspace=0.34)
    fig.savefig(out, dpi=300)
    print("saved", out)


if __name__ == "__main__":
    main()
