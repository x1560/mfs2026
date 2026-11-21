# -*- coding: utf-8 -*-
"""RQ1 池构造可行性：每个用户「自己」能提供多少测试窗负样本。

D1 甲案要求 Books/Movies 池缩到 50（1 正 + 49 负）且负例全来自用户自己。
清单里用「人均标注 113/121」估的，但那是**全量标注**；RQ1 的池只取**测试窗**，
8:2 切分下每个人能用的行数大约只有全量的 20%。先把这个数算出来，再决定池多大。

输出：每种切分下，有正样本的用户里，自有负样本数的分位数，以及能填满
      池 20 / 30 / 50 的用户占比。
"""
import os
import pickle

import numpy as np
import pandas as pd

CODE = os.path.dirname(os.path.abspath(__file__))
ARCH = os.path.join(CODE, '..', '..', 'code_archive')
TMP = os.path.join(ARCH, '_rq4_tmp')

CASES = {
    'Movies': dict(cache='movies', factors=os.path.join(CODE, 'factors_AMZ_Movies.pkl')),
    'Books': dict(cache='books', factors=os.path.join(CODE, 'factors_Seren_Books.pkl')),
    'S2018': dict(cache='s2018', factors=os.path.join(CODE, 'factors_SEREN_S2018.pkl'),
                  pairs=os.path.join(CODE, 'S2018_survey_pairs.csv')),
}


def nid(x):
    return str(x).strip().lower()


def load(case):
    cfg = CASES[case]
    df = pickle.load(open(os.path.join(TMP, '_dfcache_%s.pkl' % cfg['cache']), 'rb'))['df']
    f = pd.read_pickle(cfg['factors'])
    f['uid'] = f['uid'].map(nid)
    f['item'] = f['item'].map(nid)
    if cfg.get('pairs'):
        bp = pd.read_csv(cfg['pairs'], dtype=str)
        bs = set(zip(bp.iloc[:, 0].astype(str).str.strip().str.lower(),
                     bp.iloc[:, 1].astype(str).str.strip().str.lower()))
        keep = np.array([(u, i) in bs for u, i in zip(f['uid'], f['item'])])
        f = f[keep].reset_index(drop=True)
    return df, f


def run(case, test_ratio):
    df, f = load(case)
    lab = pd.to_numeric(df['label'], errors='coerce')
    m = lab.isin([0, 1]).values
    lt = dict(zip(zip(df['user_id'].values[m], df['item_id'].values[m]),
                  df['timestamp'].values[m].astype('int64')))
    ts = np.array([lt.get((u, i), 0) for u, i in zip(f['uid'], f['item'])], dtype='int64')
    b = np.sort(ts)[int(len(ts) * (1.0 - test_ratio))]
    te = ts > b
    lab_arr = f['label'].values.astype(int)
    cnt = pd.DataFrame({'uid': f['uid'].values[te], 'lab': lab_arr[te]})
    g = cnt.groupby('uid')['lab'].agg(['sum', 'count'])
    g['neg'] = g['count'] - g['sum']
    withpos = g[g['sum'] >= 1]
    neg = withpos['neg'].values
    print('\n== %s  test_ratio=%.1f  分割点=%d' % (case, test_ratio, b))
    print('   全量标注行/用户: p50=%.0f  mean=%.1f (总用户 %d)'
          % (g['count'].median(), g['count'].mean(), len(g)))
    print('   测试窗有正样本的用户: %d / %d' % (len(withpos), len(g)))
    if len(neg) == 0:
        return
    q = np.percentile(neg, [10, 25, 50, 75, 90])
    print('   这些用户的自有负样本数: p10=%.0f p25=%.0f p50=%.0f p75=%.0f p90=%.0f  max=%d'
          % (q[0], q[1], q[2], q[3], q[4], neg.max()))
    for k in (9, 19, 29, 49):
        n = int((neg >= k).sum())
        print('      能填满 1+%2d（自有负样本 >= %2d）: %4d 人 (%.1f%%)'
              % (k, k, n, 100.0 * n / len(withpos)))


for case in ('Movies', 'Books', 'S2018'):
    for tr in (0.2, 0.5):
        run(case, tr)
