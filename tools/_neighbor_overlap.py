# -*- coding: utf-8 -*-
"""两种“相似用户”的重合度：惊喜画像相似（标签驱动）vs 共评相似（行为驱动）。

惊喜画像 = 该用户正样本的 (rel, dif, rare, qua) 均值，余弦取 top-50；
rare = 0.5*(nov_q + unpop_q)，nov_q/unpop_q 用全表分位归一（与 eval.py 同做法）。
共评相似 = user_sim_*.pkl 里的 top-50。

输出：每个数据集 top-50 邻居的 Jaccard，均值/中位，以及“至少重合 50% 的用户”占比。
"""
import os
import pickle

import numpy as np
import pandas as pd
from scipy.stats import rankdata

CODE = os.path.dirname(os.path.abspath(__file__))
CASES = {
    'Movies': ('factors_AMZ_Movies.pkl', 'user_sim_AMZ_Movies.pkl'),
    'Books': ('factors_Seren_Books.pkl', 'user_sim_Seren_Books.pkl'),
}
TOPK = 50


def qn(v):
    r = rankdata(v, method='average')
    return (r - 1.0) / max(len(v) - 1.0, 1.0)


for name, (fpath, spath) in CASES.items():
    f = pd.read_pickle(os.path.join(CODE, fpath))
    if not os.path.exists(os.path.join(CODE, spath)):
        print(name, 'no user_sim'); continue
    us = pickle.load(open(os.path.join(CODE, spath), 'rb'))
    f['nov_q'] = qn(f['novelty'].values)
    f['unpop_q'] = qn(f['unpopularity'].values)
    f['rare'] = 0.5 * (f['nov_q'] + f['unpop_q'])
    pos = f[f['label'] == 1]
    prof = pos.groupby('uid')[['rel', 'dif', 'rare', 'qua']].mean()
    uids = list(prof.index)
    arr = prof.to_numpy()
    arr = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9)
    sim = arr @ arr.T
    np.fill_diagonal(sim, -1)
    top_idx = np.argsort(-sim, axis=1)[:, :TOPK]
    sur = {uids[i]: set(uids[j] for j in top_idx[i]) for i in range(len(uids))}

    jac = []
    for u in uids:
        a = sur.get(u, set())
        b = set(v for v, _ in us.get(u, [])[:TOPK])
        if not a or not b:
            continue
        jac.append(len(a & b) / len(a | b))
    jac = np.array(jac)
    print('== %s  有画像用户 %d  | 可比 %d' % (name, len(uids), len(jac)))
    print('   Jaccard(top50): mean %.3f  median %.3f  p90 %.3f' % (
        jac.mean(), np.median(jac), np.percentile(jac, 90)))
    print('   至少重合一半(>=0.5)的用户: %.1f%%   几乎不重合(<0.1): %.1f%%' % (
        100.0 * (jac >= 0.5).mean(), 100.0 * (jac < 0.1).mean()))
