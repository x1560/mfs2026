# -*- coding: utf-8 -*-
"""RQ4 掩蔽曲线图: 读 _rq4paper_*.log, 输出三面板折线图。

x = 质量因子被遮蔽的比例 0..100%
y = within-user Hit@2
线: M-DCG(粗), LinCombo(虚线), 四个单因子(细灰)
"""
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
# 日志前缀: 第 2 个命令行参数, 默认论文口径(不切时间窗)
_PREFIX = sys.argv[2] if len(sys.argv) > 2 else "_rq4paper_"
CASES = [
    ("Books", _PREFIX + "books.log"),
    ("Movies", _PREFIX + "movies.log"),
    ("S2018", _PREFIX + "s2018.log"),
]

RE_MAIN = re.compile(r"^\s+within-user: Hit@2=([\d.]+)")
RE_SUB = re.compile(r"^\s+\[(\w+)\s*\] within-user: Hit@2=([\d.]+)")


def parse(path):
    blocks = []
    cur = None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
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
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "..",
                                                             "pdf",
                                                             "rq4_masking.png")
    _lay = os.environ.get("RQ4_FIG_LAYOUT", "wide")
    _stack = (_lay == "column")
    if _lay == "row":
        fig, axes = plt.subplots(1, 3, figsize=(3.5, 1.42), sharey=True)
        _F = {"title": 6.5, "tick": 5.0, "xlabel": 6.0, "legend": 5.0}
    elif _stack:
        fig, axes = plt.subplots(3, 1, figsize=(3.45, 5.4), sharex=True, sharey=True)
        _F = {"title": 9, "tick": 7.5, "xlabel": 8, "legend": 7}
    else:
        fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.15), sharey=True)
        _F = {"title": 9, "tick": 7.5, "xlabel": 8, "legend": 7}
    factors = ["rel", "dif", "nov", "unpop"]
    for ax, (name, fn) in zip(axes, CASES):
        p = os.path.join(HERE, fn)
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            ax.set_title(name + " (missing)")
            continue
        bl = parse(p)
        n = len(bl)
        if n == 0:
            ax.set_title(name + " (empty)")
            continue
        xs = [100.0 * i / (n - 1) for i in range(n)] if n > 1 else [0.0]
        for fac in factors:
            ys = [b.get(fac, float("nan")) for b in bl]
            ax.plot(xs, ys, color="0.72", lw=0.9, zorder=1)
        if "lincombo" in bl[0]:
            ax.plot(xs, [b["lincombo"] for b in bl], color="tab:orange", lw=1.6,
                    ls="--", marker="s", ms=3, zorder=3, label="LinCombo")
        ax.plot(xs, [b["M-DCG"] for b in bl], color="tab:blue", lw=2.0,
                marker="o", ms=3.4, zorder=4, label="M-DCG")
        ax.set_title(name, fontsize=_F["title"])
        if _lay == "row":
            ax.set_xlabel("")
        elif (not _stack) or name == CASES[-1][0]:
            ax.set_xlabel("masked quality factor (%)", fontsize=_F["xlabel"])
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.tick_params(labelsize=_F["tick"])
        ax.grid(alpha=0.25, lw=0.6)
    axes[0].set_ylabel("within-user Hit@2", fontsize=_F["xlabel"])
    if _lay == "row":
        axes[1].set_xlabel("masked fraction (%)", fontsize=_F["xlabel"])
    # 图例: 单因子用一条灰线代表
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color="tab:blue", lw=2.0, marker="o", ms=3.4),
               Line2D([], [], color="tab:orange", lw=1.6, ls="--", marker="s", ms=3),
               Line2D([], [], color="0.72", lw=0.9)]
    axes[-1].legend(handles, ["M-DCG", "LinCombo", "single factor"],
                    fontsize=_F["legend"], loc="best", frameon=False,
                    handlelength=1.2, borderpad=0.2, labelspacing=0.2)
    fig.tight_layout(pad=0.4)
    fig.savefig(out, dpi=300)
    print("saved", out)
    # 附带输出 markdown 明细, 方便填总表
    md = [os.path.join(HERE, "_rq4_summary.md")]
    with open(md[0], "w", encoding="utf-8") as fh:
        for name, fn in CASES:
            p = os.path.join(HERE, fn)
            if not os.path.exists(p) or os.path.getsize(p) == 0:
                fh.write("\n### %s\n(缺日志)\n" % name)
                continue
            bl = parse(p)
            n = len(bl)
            fh.write("\n### %s（%d 档）\n\n" % (name, n))
            fh.write("| Mask | M-DCG | LinCombo | rel | dif | nov | unpop | qua | ndcg_rat |\n")
            fh.write("|---|---|---|---|---|---|---|---|---|\n")
            for i, b in enumerate(bl):
                lvl = int(round(100.0 * i / (n - 1))) if n > 1 else 0
                fh.write("| %d%% | %.4f | %s | %s | %s | %s | %s | %s | %s |\n" % (
                    lvl, b.get("M-DCG", float("nan")),
                    *[("%.4f" % b[k]) if k in b else "—"
                      for k in ("lincombo", "rel", "dif", "nov", "unpop", "qua", "ndcg_rat")]))
    print("wrote", md[0])
    for name, fn in CASES:
        p = os.path.join(HERE, fn)
        if os.path.exists(p) and os.path.getsize(p) > 0:
            bl = parse(p)
            print(name, "levels=", len(bl), "M-DCG0=", bl[0].get("M-DCG"),
                  "LC0=", bl[0].get("lincombo"))


if __name__ == "__main__":
    main()
