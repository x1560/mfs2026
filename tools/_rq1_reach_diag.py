# -*- coding: utf-8 -*-
"""为什么 Movies 只有 60 人能测：那 124 个测试窗正例，落在谁手上，别人够得着吗？

三个数：
  1. 持有测试窗正例的用户，有多少个有"惊喜画像"（= 训练窗里有正例）——没有画像就不在邻居图里，
     他的正例谁也借不到；
  2. 这些持有者作为别人 top-50 邻居的入度（被多少人当作邻居）；
  3. 如果邻居图改成对称（互相都算邻居）或用共评相似补边，能测的用户能从 91 涨到多少。
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
    prof = f[tr & (f['label'] == 1)].groupby('uid')[['rel', 'dif', 'rare', 'qua']].mean()
    uids = list(prof.index)
    arr = prof.to_numpy()
    arr /= (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9)
    sim = arr @ arr.T
    np.fill_diagonal(sim, -1)
    top = np.argsort(-sim, axis=1)[:, :TOPK]
    nb = {uids[i]: set(uids[j] for j in top[i]) for i in range(len(uids))}

    holders = set(f[te & (f['label'] == 1)]['uid'].unique())
    test_users = set(f[te]['uid'].unique())
    prof_holders = holders & set(uids)
    # 入度：有多少人把某个 holder 当邻居（只统计测试用户）
    indeg = {h: 0 for h in holders}
    for u in test_users:
        for v in nb.get(u, ()):
            if v in indeg:
                indeg[v] += 1
    reach_holders = [h for h in holders if indeg[h] > 0]
    users_with_reach = sum(1 for u in test_users if nb.get(u, set()) & holders)
    users_with_reach_prof = sum(1 for u in test_users if nb.get(u, set()) & prof_holders)
    # 对称化后
    nb_sym = {u: set(nb.get(u, set())) for u in test_users}
    for u in list(nb_sym):
        for v in list(nb_sym[u]):
            nb_sym.setdefault(v, set()).add(u)
    users_sym = sum(1 for u in test_users if nb_sym.get(u, set()) & holders)

    print('\n== %s' % name)
    print('   测试窗用户 %d | 持有测试窗正例 %d 人（%d 条）' % (
        len(test_users), len(holders), int(f[te & (f['label'] == 1)].shape[0])))
    print('   持有者里**有惊喜画像**的: %d 人 (%.0f%%)  <- 没画像的人不在邻居图里，正例借不出去' % (
        len(prof_holders), 100.0 * len(prof_holders) / max(len(holders), 1)))
    print('   持有者的入度: 为 0 的 %d 人，中位 %d' % (
        sum(1 for h in holders if indeg[h] == 0), int(np.median(list(indeg.values())))))
    print('   能借到正例的测试用户: 原邻居图 %d | 仅算有画像的持有者 %d | 邻居图对称化后 %d' % (
        users_with_reach, users_with_reach_prof, users_sym))
