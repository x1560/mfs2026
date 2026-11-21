# -*- coding: utf-8 -*-
"""算 rel / dif 的 ICC(1)，以及 E+D 处理（按用户负样本中心平移 + 全局百分位）之后的 ICC。

用途：论文 4.2.3 的 ICC 表目前只有 relevance，补 difference 的两行。
定义：单因素随机效应、不平衡样本的 ICC(1)
    ICC = (MSB - MSW) / (MSB + (n0 - 1) * MSW)
    n0  = (N - sum(n_i^2)/N) / (k - 1)
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

CODE = os.path.dirname(os.path.abspath(__file__))
DATASETS = [
    ("Books", "factors_Seren_Books.pkl"),
    ("Movies", "factors_AMZ_Movies.pkl"),
]


def icc1(df, col, uid_col='uid'):
    g = df.groupby(uid_col)[col]
    n_i = g.size().values.astype(float)
    N = n_i.sum()
    k = len(n_i)
    if k < 2 or N - k < 1:
        return float('nan')
    means = g.mean().values
    grand = df[col].mean()
    ssb = float(np.sum(n_i * (means - grand) ** 2))
    ssw = float(np.sum(g.apply(lambda s: ((s - s.mean()) ** 2).sum()).values))
    msb = ssb / (k - 1)
    msw = ssw / (N - k)
    n0 = (N - (np.sum(n_i ** 2) / N)) / (k - 1)
    return (msb - msw) / (msb + (n0 - 1) * msw)


def ed_transform(df, col, uid_col='uid', label_col='label'):
    """E+D：按用户负样本中心平移，再做全局百分位归一化。"""
    neg = df[df[label_col] == 0].groupby(uid_col)[col].median()
    centered = df[col] - df[uid_col].map(neg).fillna(df[col].mean())
    pct = (rankdata(centered.values) - 1.0) / max(len(centered) - 1, 1)
    return pct


def main():
    rows = []
    for name, fn in DATASETS:
        p = os.path.join(CODE, fn)
        if not os.path.exists(p):
            print("missing", fn)
            continue
        d = pd.read_pickle(p)[['uid', 'rel', 'dif', 'label']].dropna()
        out = {"dataset": name, "n": len(d), "users": d['uid'].nunique()}
        for col in ('rel', 'dif'):
            out[col] = icc1(d, col)
            d[col + '_ed'] = ed_transform(d, col)
            out[col + '_ed'] = icc1(d, col + '_ed')
        rows.append(out)
        print("%-7s n=%6d users=%4d | rel ICC %.3f (E+D %.3f) | dif ICC %.3f (E+D %.3f)"
              % (name, out['n'], out['users'], out['rel'], out['rel_ed'],
                 out['dif'], out['dif_ed']))
    pd.DataFrame(rows).to_csv(os.path.join(CODE, '_diag', '_icc_rel_dif.csv'),
                              index=False, encoding='utf-8-sig')
    print("written _diag/_icc_rel_dif.csv")


if __name__ == '__main__':
    main()
