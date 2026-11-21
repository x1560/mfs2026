# -*- coding: utf-8 -*-
"""RQ4 图（第四版）：1:99 池上的质量掩蔽 + 其余基线（水平线）。

- M-DCG、LinCombo：随掩蔽变化的曲线；
- ILD / MMR / Surprise / nDCG / α-nDCG / αβ-nDCG / Random：不读质量项，
  在每个档位都一样，画成水平线（值取表 5 的 0% 档）。
"""
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
LEVELS = [0, 20, 40, 60, 80]
CASES = [("Books", "books"), ("Movies", "movies"), ("S2018", "s2018")]
DRAW_BASE = False
FLAT = [("ild", "ILD"), ("mmr", "MMR"), ("fn_surprise", "Surprise"),
        ("ndcg_rating", "nDCG"), ("alpha_ndcg", r"$\alpha$-nDCG"),
        ("abeta_ndcg", r"$\alpha\beta$-nDCG"), ("random", "Random")]
ROW = re.compile(r'^\s{2,}([a-z_]+)((?:\s+[\d.]+)+)\s*$')


def hit5(path, metric):
    if not os.path.exists(path):
        return None
    for line in open(path, encoding='utf-8', errors='replace'):
        m = ROW.match(line)
        if m and m.group(1) == metric:
            nums = [float(x) for x in m.group(2).split()]
            # 命中率表(4 个数): Hit@5,Hit@10,位次,AUC；掩蔽日志(5 个数): Hit@2,Hit@5,Hit@10,位次,AUC
            return nums[0] if len(nums) == 4 else (nums[1] if len(nums) == 5 else None)
    return None


def series(case, pattern, metric):
    return [hit5(os.path.join(HERE, pattern % (case, lvl)), metric) for lvl in LEVELS]


def main():
    out = os.path.join(HERE, "..", "..", "..", "pdf", "pdf", "rq4_masking_row.png")
    fig, axes = plt.subplots(1, 3, figsize=(3.5, 1.32), sharey=False)
    F = {"title": 6.5, "tick": 5.0, "xlabel": 6.0, "legend": 5.0}
    for ax, (name, case) in zip(axes, CASES):
        md = series(case, '_mask_pool_%s_%d.log', 'mdcg')
        lc = series(case, '_maskpool_lc_%s_%d.log', 'lincombo')
        if any(v is None for v in lc):
            lc = series(case, '_maskpool_base_%s_%d.log', 'lincombo')
        # 基线的水平线（取值来自表 5 的 0% 跑批）
        base = {}
        for key, _lab in FLAT:
            v = hit5(os.path.join(HERE, '_pw_full_%s.log' % case), key)
            if v is not None:
                base[key] = v
        if DRAW_BASE and base:
            key, v = max(base.items(), key=lambda kv: kv[1])
            lab = {"ild": "ILD", "mmr": "MMR", "fn_surprise": "Surprise",
                   "ndcg_rating": "nDCG", "alpha_ndcg": "a-nDCG",
                   "abeta_ndcg": "ab-nDCG", "random": "Random"}.get(key, key)
            ax.axhline(v, color="0.45", lw=0.7, ls=(0, (3, 2)), zorder=1)
        xs = [l for l, v in zip(LEVELS, md) if v is not None]
        ax.plot(xs[:len([v for v in lc if v is not None])],
                [v for v in lc if v is not None],
                color="tab:orange", lw=1.0, marker="o", ms=2.0, zorder=3)
        ax.plot(xs, [v for v in md if v is not None],
                color="tab:blue", lw=1.3, marker="o", ms=2.4, zorder=4)
        ax.set_title(name, fontsize=F["title"])
        ax.set_xticks([0, 20, 40, 60, 80])
        ax.tick_params(labelsize=F["tick"], pad=1.5)
        ax.grid(alpha=0.2, lw=0.5)
        # 纵轴要同时容纳曲线和那条基线水平线，否则基线会被裁掉
        vals = [v for v in md if v is not None] + [v for v in lc if v is not None]
        if DRAW_BASE and base:
            vals.append(v)
        lo, hi = min(vals), max(vals)
        pad = (hi - lo) * 0.12
        ax.set_ylim(lo - pad, hi + pad)
        if DRAW_BASE and base:
            ax.text(3, v + (hi - lo) * 0.03, lab, fontsize=4.2, color="0.35",
                    va="bottom", ha="left")
    axes[0].set_ylabel("Hit@5", fontsize=F["xlabel"], labelpad=2)
    axes[1].set_xlabel("masked quality ratio (%)", fontsize=F["xlabel"], labelpad=1)
    handles = [Line2D([], [], color="tab:blue", lw=1.3, marker="o", ms=2.4),
               Line2D([], [], color="tab:orange", lw=1.0, marker="o", ms=2.0)]
    labels = ["M-DCG", "LinCombo"]
    if DRAW_BASE:
        handles.append(Line2D([], [], color="0.45", lw=0.7, ls=(0, (3, 2))))
        labels.append("strongest other baseline")
    fig.legend(handles, labels,
               fontsize=F["legend"], loc="lower center",
               bbox_to_anchor=(0.5, 0.012), ncol=len(labels), frameon=False,
               handlelength=1.4, borderpad=0.0, labelspacing=0.2,
               columnspacing=1.2)
    fig.subplots_adjust(left=0.115, right=0.995, top=0.88, bottom=0.30, wspace=0.34)
    fig.savefig(out, dpi=300)
    print("saved", out)


if __name__ == "__main__":
    main()
