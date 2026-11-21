# -*- coding: utf-8 -*-
"""按 M_DCG/code/analysis/02_user_profiles_icc.py:42-49 的公式，算 rel 与 dif 的 ICC。

    between = Σ n_u (mean_u - grand)^2 / (k-1)
    within  = mean of per-user variance (ddof=1)
    ICC     = between / (between + within)

三种状态：原始（raw）、E（按用户因果平移）、E+D（再全局分位归一化）。
"""
import os
import sys

import numpy as np
import pandas as pd

CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE)
os.chdir(CODE)
sys.argv = ['x']

import eval as E  # noqa: E402
import metrics as M  # noqa: E402

DATASETS = [
    ("Books", "factors_Seren_Books.pkl", "SEREN_Books", "./data/SerenLens/SerenLens_Books.csv"),
    ("Movies", "factors_AMZ_Movies.pkl", "SEREN_Movies", "./data/SerenLens/SerenLens_Movies.csv"),
    ("S2018", "factors_SEREN_S2018.pkl", "SEREN_S2018", "SerenLens_S2018.csv"),
]
MIN_C = 5


def icc(vals, groups):
    g = pd.DataFrame({'v': np.asarray(vals, float), 'u': np.asarray(groups)})
    m = g.groupby('u')['v'].agg(['mean', 'count'])
    gm = g['v'].mean()
    between = float((m['count'] * (m['mean'] - gm) ** 2).sum() / max(1, len(m) - 1))
    within = float(g.groupby('u')['v'].var(ddof=1).fillna(0).mean())
    return between / (between + within) if (between + within) > 0 else 0.0


def icc1(vals, groups):
    g = pd.DataFrame({'v': np.asarray(vals, float), 'u': np.asarray(groups)})
    gg = g.groupby('u')['v']
    n_i = gg.size().values.astype(float); N = n_i.sum(); k = len(n_i)
    means = gg.mean().values; grand = g['v'].mean()
    ssb = float(np.sum(n_i * (means - grand) ** 2))
    ssw = float(np.sum(gg.apply(lambda s: ((s - s.mean()) ** 2).sum()).values))
    msb = ssb / (k - 1); msw = ssw / (N - k)
    n0 = (N - (np.sum(n_i ** 2) / N)) / (k - 1)
    return (msb - msw) / (msb + (n0 - 1) * msw)


def run(name, fname, tag, csv):
    fdf = pd.read_pickle(fname)
    user_seq, labels, label_time, ir = E.load_serenlens(csv)
    df = E._attach_ts(fdf, label_time).sort_values('ts').reset_index(drop=True)
    ts = df['ts'].values.astype(np.int64)
    uid = df['uid'].values
    lab = df['label'].values.astype(int)
    rel = df['rel'].values.astype(float)
    dif = df['dif'].values.astype(float)
    cr, cd, cn = E._causal_running_center(uid, ts, lab, rel, dif, MIN_C)
    one = np.zeros(len(ts), dtype=np.int64)
    gr, gd, _ = E._causal_running_center(one, ts, lab, rel, dif, MIN_C)
    need = cn < MIN_C
    rel_e = np.where(need, rel - gr, rel - cr)
    dif_e = np.where(need, dif - gd, dif - cd)
    rel_t = M.quantile_normalize(rel_e)
    dif_t = M.quantile_normalize(dif_e)
    row = dict(dataset=name, n=len(df), users=len(set(uid)),
               rel_raw=icc(rel, uid), rel_E=icc(rel_e, uid), rel_ED=icc(rel_t, uid),
               dif_raw=icc(dif, uid), dif_E=icc(dif_e, uid), dif_ED=icc(dif_t, uid),
               rel_raw_i1=icc1(rel, uid), rel_E_i1=icc1(rel_e, uid),
               dif_raw_i1=icc1(dif, uid), dif_E_i1=icc1(dif_e, uid),
               fallback_pct=100.0 * need.mean())
    print("%-7s n=%6d users=%4d | rel raw %.3f  E %.3f  E+D %.3f | dif raw %.3f  E %.3f  E+D %.3f"
          % (name, row['n'], row['users'], row['rel_raw'], row['rel_E'], row['rel_ED'],
             row['dif_raw'], row['dif_E'], row['dif_ED']), flush=True)
    return row


def main():
    rows = []
    for name, fname, tag, csv in DATASETS:
        if not os.path.exists(fname):
            print("missing", fname)
            continue
        rows.append(run(name, fname, tag, csv))
    out = os.path.join(CODE, '_diag', '_icc_rel_dif_v2.csv')
    pd.DataFrame(rows).to_csv(out, index=False, encoding='utf-8-sig')
    print("written", out)


if __name__ == '__main__':
    main()
