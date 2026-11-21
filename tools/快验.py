# -*- coding: utf-8 -*-
"""快速验证：一次加载，多个打分变体，直接出 AUC / Hit@k。

用法:
    python _diag/快验.py movies

做的事
-----
1. 走 eval.py 的 _prepare_temporal（8:2 切分、训练窗拟合参数），结果缓存到
   _diag/_ctx_<case>.pkl，第二次跑直接读缓存（约 1 秒）。
2. 在测试窗上对若干个打分变体算 AUC（全部测试行）和 Hit@5/10（1 正 + 99 负池）。
3. 打印一张表。

加新变体只要在 VARIANTS 里加一行。
"""
import os
import sys
import pickle
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = r"C:\Users\Administrator\Desktop\study\论文复现\code"
sys.path.insert(0, ROOT)
import eval as E
import metrics as M

CASES = {
    'movies': ('AMZ', 'Movies', ROOT + r"\data\SerenLens\SerenLens_Movies.csv",
               ROOT + r"\factors_AMZ_Movies.pkl", ROOT + r"\emb_AMZ_Movies.pkl", None),
    'books': ('Seren', 'Books', ROOT + r"\data\SerenLens\SerenLens_Books.csv",
              ROOT + r"\factors_Seren_Books.pkl", ROOT + r"\emb_Seren_Books.pkl", None),
    's2018': ('SEREN', 'S2018', ROOT + r"\SerenLens_S2018.csv",
              ROOT + r"\factors_SEREN_S2018.pkl", ROOT + r"\emb_SEREN_S2018.pkl",
              ROOT + r"\S2018_survey_pairs.csv"),
}
POOL_K = 100


def load_ctx(case, rebuild=False):
    """带缓存地拿到 ctx（含 E+D 特征、训练窗参数）。"""
    cp = os.path.join(ROOT, '_diag', '_ctx_%s.pkl' % case)
    if os.path.exists(cp) and not rebuild:
        with open(cp, 'rb') as f:
            return pickle.load(f)
    model, data, sl, fac, embp, pairs = CASES[case]
    seq, _, label_time, _ = E.load_serenlens(sl)
    emb = {str(k).strip().lower(): np.asarray(v, float)
           for k, v in pickle.load(open(embp, 'rb')).items()}
    fdf = pd.read_pickle(fac)
    if pairs:
        p = pd.read_csv(pairs, dtype=str)
        keep = set(zip(p.iloc[:, 0].str.lower(), p.iloc[:, 1].str.lower()))
        fdf = fdf[[(str(u).lower(), str(i).lower()) in keep
                   for u, i in zip(fdf['uid'], fdf['item'])]].reset_index(drop=True)
    fdf['uid'] = fdf['uid'].astype(str).str.strip().str.lower()
    fdf['item'] = fdf['item'].astype(str).str.strip().str.lower()
    lt = {(str(u).strip().lower(), str(i).strip().lower()): t for (u, i), t in label_time.items()}
    seq = {str(u).strip().lower(): [(str(i).strip().lower(), int(t), float(r)) for i, t, r in v]
           for u, v in seq.items()}
    ctx = E._prepare_temporal(fdf, seq, lt, E.SimEngine(emb), True, 5, 0.2, 1, None, False,
                              'surprise_pool', 'median', 5, tag='(快验)', topk=50,
                              min_valid_users=1, min_neg_test=1,
                              quality_index=None, qua_mode='pit')
    with open(cp, 'wb') as f:
        pickle.dump(ctx, f)
    return ctx


# ----------------------------------------------------------------------------
# 打分变体：每个函数返回一个分数数组（越大越靠前）
# ----------------------------------------------------------------------------
def variants(ctx):
    td = ctx['test_df']
    rel = td['rel'].values.astype(float)
    dif = td['dif'].values.astype(float)
    nov = td['nov_q'].values.astype(float) if 'nov_q' in td.columns else td['novelty'].values.astype(float)
    unp = td['unpop_q'].values.astype(float) if 'unpop_q' in td.columns else td['unpopularity'].values.astype(float)
    qua = td['qua'].values.astype(float)
    PER, PACC = ctx['per_user_pp_np'], ctx['scarcity_win']
    PG, NG = ctx['pp_global'], ctx['np_global']
    uid = td['uid'].values

    def base(u_t=None, i=None):
        u = u_t if u_t is not None else uid[i]
        pp, nn = PER.get(u, (PG, NG))
        return M.f_sp_score(rel[i], dif[i], nov[i], unp[i], qua[i], pp, nn,
                            use_copula=True, user_scarcity_param=PACC.get(u))

    def arousal_only(u_t=None, i=None):
        pp, nn = PER.get(u_t if u_t is not None else uid[i], (PG, NG))
        return M.arousal_score(rel[i], dif[i], pp, nn, use_copula=True)

    def add_gate(u_t=None, i=None):
        """质量门控改加法：arousal * (1 + qn + qu) 而不是 1 + qua*(qn+qu)"""
        u = u_t if u_t is not None else uid[i]
        pp, nn = PER.get(u, (PG, NG))
        a = M.arousal_score(rel[i], dif[i], pp, nn, use_copula=True)
        qn, qu = M.accommodation_components(nov[i], unp[i], PACC.get(u))
        return a * (1.0 + qn + qu)

    # 用【训练窗】拟合两条线性融合，再在测试窗上比
    from sklearn.linear_model import LogisticRegression
    tr = ctx['train_df']
    y = tr['label'].values.astype(int)

    def _fit(cols):
        X = np.column_stack([c(tr) for c in cols])
        m = ~(np.isnan(X).any(axis=1))
        return LogisticRegression(max_iter=2000, class_weight='balanced').fit(
            X[m], y[m]).coef_[0]

    f3 = [lambda d: -d['nov_q'].values, lambda d: d['qua'].values,
          lambda d: d['rel'].values]
    f5 = f3 + [lambda d: d['dif'].values, lambda d: d['unpop_q'].values]
    w3, w5 = _fit(f3), _fit(f5)
    _V3 = np.column_stack([-nov, qua, rel])
    _V5 = np.column_stack([-nov, qua, rel, dif, unp])

    def _kernel_variant(mode):
        def f(u_t=None, i=None):
            u = u_t if u_t is not None else uid[i]
            pp, nn = PER.get(u, (PG, NG))
            a = M.arousal_score(rel[i], dif[i], pp, nn, use_copula=True)
            p = PACC.get(u) or {}
            lam = float(p.get('lambda_n', 0.5))
            ap = float(p.get('alpha_n_plus', 1.0)) or 1.0
            am = float(p.get('alpha_n_minus', 1.0)) or 1.0
            nv = nov[i]
            if mode == 'onesided':
                qn = 1.0 if nv <= lam else 1.0 / (1.0 + ((nv - lam) / ap) ** 2)
            else:                       # peak0
                qn = 1.0 / (1.0 + (nv / ap) ** 2)
            qu = M.accommodation_components(nov[i], unp[i], p)[1]
            return a * (1.0 + qua[i] * (qn + qu))
        return f

    def _gate_only(u_t=None, i=None):
        u = u_t if u_t is not None else uid[i]
        qn, qu = M.accommodation_components(nov[i], unp[i], PACC.get(u))
        return 1.0 + qua[i] * (qn + qu)

    def _additive(u_t=None, i=None):
        u = u_t if u_t is not None else uid[i]
        pp, nn = PER.get(u, (PG, NG))
        a = M.arousal_score(rel[i], dif[i], pp, nn, use_copula=True)
        qn, qu = M.accommodation_components(nov[i], unp[i], PACC.get(u))
        return a + (1.0 + qua[i] * (qn + qu))

    def _qn_pure(u_t=None, i=None):
        """只返回单边核本身，不乘 qua、不加 q_U、不乘 arousal"""
        u = u_t if u_t is not None else uid[i]
        p_ = PACC.get(u) or {}
        lam = float(p_.get('lambda_n', 0.5))
        ap = float(p_.get('alpha_n_plus', 1.0)) or 1.0
        nv = nov[i]
        return 1.0 if nv <= lam else 1.0 / (1.0 + ((nv - lam) / ap) ** 2)

    out = {
        'M-DCG(现值)': base,
        'q_N单边(纯)': _qn_pure,
        '仅门控(去arousal)': _gate_only,
        '加法(arousal+门控)': _additive,
        'q_N单边': _kernel_variant('onesided'),
        'q_N峰在0': _kernel_variant('peak0'),
        '线性(-nov,qua,rel)': lambda u_t, i: float(_V3[i] @ w3),
        '线性(五因子)': lambda u_t, i: float(_V5[i] @ w5),
        'arousal only': arousal_only,
        '质量门控改加法': add_gate,
        '-rel 单因子': lambda u_t, i: -rel[i],
        '-dif 单因子': lambda u_t, i: -dif[i],
        '-nov 单因子': lambda u_t, i: -nov[i],
        'qua 单因子': lambda u_t, i: qua[i],
    }
    return td, rel, dif, nov, unp, qua, out


def evaluate(ctx, fn):
    """全部测试行算 AUC；1 正 + 99 负池算 Hit@5/10。"""
    td = ctx['test_df']
    lab = td['label'].values.astype(int)
    uid = td['uid'].values
    s = np.array([fn(uid[i], i) for i in range(len(td))], float)
    ok = np.isfinite(s) & np.isfinite(td['rel'].values) & np.isfinite(td['dif'].values)
    # 逐用户 AUC 再平均（与流水线的 OrderAUC 同口径）；池化 AUC 会被用户间尺度差洗掉
    _per = defaultdict(list)
    for _j in np.where(ok)[0]:
        _per[uid[_j]].append(_j)
    _aucs = []
    for _u, _idx in _per.items():
        _y = lab[_idx]
        if 0 < _y.sum() < len(_y):
            _aucs.append(roc_auc_score(_y, s[_idx]))
    auc = float(np.mean(_aucs)) if _aucs else float('nan')

    rows_of = defaultdict(list)
    for j, u in enumerate(uid):
        rows_of[u].append(j)
    neg_all = np.array([j for j in range(len(td)) if lab[j] == 0 and ok[j]])
    rng = np.random.RandomState(7)
    h5, h10 = [], []
    for u, rows in rows_of.items():
        pos = [j for j in rows if lab[j] == 1 and ok[j]]
        if not pos:
            continue
        cand = [j for j in neg_all if j not in set(rows)]
        if len(cand) < POOL_K - 1:
            continue
        pool = pos[:1] + list(rng.choice(np.array(cand), POOL_K - 1, replace=False))
        order = np.argsort(-(s[pool] + 1e-9 * rng.rand(len(pool))), kind='mergesort')
        y = lab[np.array(pool)[order]]
        h5.append(1.0 if y[:5].sum() > 0 else 0.0)
        h10.append(1.0 if y[:10].sum() > 0 else 0.0)
    return auc, float(np.mean(h5)) if h5 else float('nan'), \
        float(np.mean(h10)) if h10 else float('nan'), len(h5)


def evaluate_pool(ctx, fn):
    """在【相似用户池】上算 AUC 与 Hit@k（与 exp_hit_rate 同口径：1 正 + 99 负）。"""
    td = ctx['test_df']
    uid = td['uid'].values
    SIM = ctx['surprise_sim_win']
    uis = {}
    items_by_uid = defaultdict(list)
    for j in range(len(td)):
        u = td['uid'].values[j]
        items_by_uid[u].append((td['item'].values[j], td['rel'].values[j],
                                td['dif'].values[j], td['nov_q'].values[j],
                                td['unpop_q'].values[j], td['qua'].values[j],
                                int(td['label'].values[j]), j))
    for u, rows in items_by_uid.items():
        uis[u] = set(r[0] for r in rows)

    s = np.array([fn(uid[i], i) for i in range(len(td))], float)
    lab = td['label'].values.astype(int)
    rng = np.random.RandomState(7)
    aucs, h5, h10 = [], [], []
    for u in sorted(set(td['uid'])):
        pos_pool, neg_pool = E.build_sim_user_pools(
            u, SIM, items_by_uid, uis, POOL_K, pool_dedup='none', pos_ratio_thr=0.5,
            min_pos=1, min_neg=POOL_K - 1)
        if not pos_pool or len(neg_pool) < POOL_K - 1:
            continue
        pool = [pos_pool[i] for i in rng.choice(len(pos_pool), 1, replace=False)] + \
               [neg_pool[i] for i in rng.choice(len(neg_pool), POOL_K - 1, replace=False)]
        idx = [x[7] for x in pool]
        y = np.array([x[6] for x in pool])
        sc = np.array([fn(u, j) for j in idx], float)
        ok = np.isfinite(sc)
        if ok.sum() < 2 or 0 == y[ok].sum():
            continue
        aucs.append(roc_auc_score(y[ok], sc[ok]))
        order = np.argsort(-(sc + 1e-9 * rng.rand(len(sc))), kind='mergesort')
        yy = y[order]
        h5.append(1.0 if yy[:5].sum() > 0 else 0.0)
        h10.append(1.0 if yy[:10].sum() > 0 else 0.0)
    return (float(np.mean(aucs)) if aucs else float('nan'),
            float(np.mean(h5)) if h5 else float('nan'),
            float(np.mean(h10)) if h10 else float('nan'), len(aucs))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('case', choices=sorted(CASES))
    ap.add_argument('--rebuild', action='store_true', help='忽略缓存，重新建 ctx')
    a = ap.parse_args()
    ctx = load_ctx(a.case, a.rebuild)
    td, rel, dif, nov, unp, qua, vs = variants(ctx)
    print('%s: 测试窗 %d 行 | 正样本 %d | 用户 %d'
          % (a.case, len(td), int(td['label'].sum()), td['uid'].nunique()))
    print('%-16s %8s %8s %8s %8s' % ('变体', 'AUC', 'Hit@5', 'Hit@10', '用户数'))
    print('--- 全测试窗（逐用户 AUC）---')
    for name, fn in vs.items():
        auc, h5, h10, n = evaluate(ctx, fn)
        print('%-16s %8.4f %8.4f %8.4f %8d' % (name, auc, h5, h10, n))
    print('--- 相似用户池（1 正 + 99 负，池内 AUC）---')
    for name, fn in vs.items():
        auc, h5, h10, n = evaluate_pool(ctx, fn)
        print('%-16s %8.4f %8.4f %8.4f %8d' % (name, auc, h5, h10, n))
    print('随机基准: AUC=0.5000  Hit@5=0.0500  Hit@10=0.1000')
