# -*- coding: utf-8 -*-
"""用 KMeans 聚类结果造【多标签】label bag，供 Fu & Niu (TKDD 2024) 的共现惊喜用。

背景
    Fu & Niu 的 Eq 1 是"**同一物品内两个 label** 的 PMI"，所以每个物品必须 >=2 个 label。
    我们的物品没有真实类别元数据（Books/Movies 的 SerenLens CSV 只有 6 列），
    但本来就有 KMeans 伪类别（cate_*.pkl）。硬分配只有 1 个 label -> 退化。
    这里取每个物品在嵌入空间里**最近的 top-k 个聚类中心**当它的 label bag。

    [WARN] 这套 label 是**合成**的（聚类共现，不是原文的"社会集体知识的主题共现"），
      论文里必须注明。

产物
    clustlab_{tag}.pkl   {item: [c3, c7, c9]}   与 cate_*.pkl 的命名一致（c0..c9）

用法
    python _diag/build_clustlab.py Books
    python _diag/build_clustlab.py Movies
    python _diag/build_clustlab.py S2018
    python _diag/build_clustlab.py Books --topk 4
"""
import os
import sys
import pickle
import argparse

import numpy as np

ROOT = os.environ.get('MDCG_ROOT') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

TAGS = {'Books': 'Seren_Books', 'Movies': 'AMZ_Movies', 'S2018': 'SEREN_S2018'}
FACTORS = {'Books': 'factors_Seren_Books.pkl',
           'Movies': 'factors_AMZ_Movies.pkl',
           'S2018': 'factors_SEREN_S2018.pkl'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data', choices=list(TAGS))
    ap.add_argument('--topk', type=int, default=3, help='每个物品取最近 k 个聚类中心当 label（默认 3）')
    ap.add_argument('--n_cate', type=int, default=10, help='与建库时一致（默认 10）')
    a = ap.parse_args()

    tag = TAGS[a.data]
    emb_path = 'emb_%s.pkl' % tag
    cate_path = 'cate_%s.pkl' % tag
    print('>>> %s : %s + %s' % (a.data, emb_path, cate_path))

    emb = pickle.load(open(emb_path, 'rb'))
    cate = pickle.load(open(cate_path, 'rb'))
    items = sorted(emb.keys())
    X = np.vstack([np.asarray(emb[i], dtype=float) for i in items])
    print('    物品 %d, 嵌入维度 %d, 现有类别数 %d' % (len(items), X.shape[1], len(set(cate.values()))))

    # ---- 复现建库时的 KMeans（build_s2018_data.py / build_books_amz.py 用的参数）----
    from sklearn.cluster import KMeans
    n_cate = int(min(a.n_cate, max(2, len(items) // 10), len(items)))
    km = KMeans(n_clusters=n_cate, random_state=42, n_init=10).fit(X)
    lab = km.labels_
    pred = {items[i]: 'c%d' % lab[i] for i in range(len(items))}
    same = sum(1 for it in items if pred[it] == cate.get(it))
    agree = 100.0 * same / max(1, len(items))
    print('    KMeans(n_clusters=%d, random_state=42, n_init=10) 与现有 cate 一致率: %.2f%%' % (n_cate, agree))
    if agree < 99.9:
        print('    [WARN] 不一致 —— 说明建库时参数不同，下面用【现有 cate_*.pkl 的标签名】对齐，')
        print('      但 top-k 的编号会按新拟合的中心顺序，可能对不上，先停下人工确认。')
    else:
        print('    [OK] 完全复现，top-k 可以放心用')

    # ---- 取 top-k 最近的聚类中心 ----
    d = km.transform(X)                       # (n_items, n_cate) 到各中心的距离
    order = np.argsort(d, axis=1)[:, :a.topk]
    topk = {}
    for i, it in enumerate(items):
        topk[it] = ['c%d' % j for j in order[i]]
    # 按距离升序，最近的在最前
    out = 'clustlab_%s.pkl' % tag
    with open(out, 'wb') as f:
        pickle.dump(topk, f, protocol=4)
    print('    -> %s (%.1f MB)' % (out, os.path.getsize(out) / 1e6))

    # ---- 统计 ----
    from collections import Counter
    cnt = Counter(len(v) for v in topk.values())
    print('    label 数分布:', dict(sorted(cnt.items())))
    print('    样例:')
    for it in items[:5]:
        dd = sorted(zip(d[items.index(it)], ['c%d' % j for j in range(n_cate)]))[:a.topk]
        print('      %-12s %s   (距离 %s)' % (it, topk[it], ['%.3f' % x[0] for x in dd]))

    # ---- 覆盖率 vs 因子表 ----
    fp = FACTORS[a.data]
    if os.path.exists(fp):
        import pandas as pd
        fdf = pd.read_pickle(fp)
        its = fdf['item'].astype(str).str.strip()
        uq = set(its)
        hit = sum(1 for x in uq if x in topk)
        multi = sum(1 for x in uq if len(topk.get(x, [])) >= 2)
        pos = set(its[fdf['label'].values == 1])
        pm = sum(1 for x in pos if len(topk.get(x, [])) >= 2)
        print()
        print('    覆盖率 vs %s: 物品 %d | 有标签 %d (%.1f%%) | **>=2 个 label %d (%.1f%%)**'
              % (fp, len(uq), hit, 100.0 * hit / max(1, len(uq)), multi, 100.0 * multi / max(1, len(uq))))
        print('    正样本 %d | **>=2 个 label %d (%.1f%%)**' % (len(pos), pm, 100.0 * pm / max(1, len(pos))))


if __name__ == '__main__':
    main()
