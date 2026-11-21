# -*- coding: utf-8 -*-
"""诊断：RQ1 现在的池子为什么只评到 60 人（Movies）/ 722 人（Books）。

复刻 eval.py 的池构造门槛：
  候选 = 目标用户的 top-50 惊喜画像邻居在**测试窗**里的已标注物品，排除目标用户自己交互过的；
  门槛 = 正例 >= min_pos(=1)、负例 >= pool_k-1(=99)。
问题在于 99 这条线是否才是真正的卡点。

输出：每个用户可得的正/负例数分布，以及把负例门槛降到 50/20 时能过线的用户数。
"""
import os
import pickle

import numpy as np
import pandas as pd
from scipy.stats import rankdata

CODE = os.path.dirname(os.path.abspath(__file__))
ARCH = os.path.join(CODE, '..', '..', 'code_archive', '_rq4_tmp')
CASES = {'Movies': ('movies', 'factors_AMZ_Movies.pkl'),
         'Books': ('books', 'factors_Seren_Books.pkl')}
TOPK, TR = 50, 0.2


def qn(v):
    r = rankdata(v, method='average')
    return (r - 1.0) / max(len(v) - 1.0, 1.0)


for name, (cache, fp) in CASES.items():
    df = pickle.load(open(os.path.join(ARCH, '_dfcache_%s.pkl' % cache), 'rb'))['df']
    f = pd.read_pickle(os.path.join(CODE, fp))
    lab = pd.to_numeric(df['label'], errors='coerce')
    m = lab.isin([0, 1]).values
    lt = dict(zip(zip(df['user_id'].values[m], df['item_id'].values[m]),
                  df['timestamp'].values[m].astype('int64')))
    ts = np.array([lt.get((u, i), 0) for u, i in zip(f['uid'], f['item'])], dtype='int64')
    b = np.sort(ts)[int(len(ts) * (1.0 - TR))]
    tr, te = ts <= b, ts > b

    f['nov_q'] = qn(f['novelty'].values)
    f['unpop_q'] = qn(f['unpopularity'].values)
    f['rare'] = 0.5 * (f['nov_q'] + f['unpop_q'])
    pos_tr = f[tr & (f['label'] == 1)]
    prof = pos_tr.groupby('uid')[['rel', 'dif', 'rare', 'qua']].mean()
    uids = list(prof.index)
    arr = prof.to_numpy()
    arr /= (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9)
    sim = arr @ arr.T
    np.fill_diagonal(sim, -1)
    top = np.argsort(-sim, axis=1)[:, :TOPK]
    nb = {uids[i]: [uids[j] for j in top[i]] for i in range(len(uids))}

    test_rows = f[te]
    by_uid = {}
    for r in test_rows.itertuples(index=False):
        by_uid.setdefault(r.uid, []).append((r.item, int(r.label)))
    seen = {}   # 用户交互过的物品（用因子表的行近似：因子表只有标注行，但已覆盖大部分交互）
    for r in f.itertuples(index=False):
        seen.setdefault(r.uid, set()).add(r.item)

    npos, nneg = [], []
    for u, lst in by_uid.items():
        us = seen.get(u, set())
        p = n = 0
        for v in nb.get(u, []):
            for it, L in by_uid.get(v, []):
                if it in us:
                    continue
                if L == 1:
                    p += 1
                else:
                    n += 1
        if p == 0:
            continue
        npos.append(p)
        nneg.append(n)
    npos, nneg = np.array(npos), np.array(nneg)
    print('\n== %s  有邻居正例的用户 %d' % (name, len(npos)))
    if len(npos) == 0:
        continue
    print('   邻居正例数: 中位 %.0f  p90 %.0f  max %d' % (
        np.median(npos), np.percentile(npos, 90), npos.max()))
    print('   邻居负例数: p10 %.0f  中位 %.0f  p90 %.0f' % (
        np.percentile(nneg, 10), np.median(nneg), np.percentile(nneg, 90)))
    for req in (99, 50, 29, 19):
        print('   负例门槛 >=%3d：能过线的用户 %4d (%.0f%%)' % (
            req, int((nneg >= req).sum()), 100.0 * (nneg >= req).mean()))
