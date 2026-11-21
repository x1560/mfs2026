# -*- coding: utf-8 -*-
"""阈值式 div 到底有没有生效: 列表内 m 与 m_ref 的比较"""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, 'C:/Users/Administrator/Desktop/study/论文复现/code')
os.chdir('C:/Users/Administrator/Desktop/study/论文复现/code')
import numpy as np
from collections import defaultdict
import eval as E, metrics as M

K = 10; SEED = 42

def run(DS, seren, fp, ef):
    user_seq, labels, label_time, item_rating = E.load_serenlens(seren)
    fdf = E.pd.read_pickle(fp)
    sim = E.SimEngine(E.load_embeddings(ef)).sim
    fdf['nov_q'] = M.quantile_normalize(fdf['novelty'].values)
    fdf['unpop_q'] = M.quantile_normalize(fdf['unpopularity'].values)
    fdf['rare'] = 0.5 * (fdf['nov_q'] + fdf['unpop_q'])
    pos_df = fdf[fdf['label'] == 1]
    pu, pr = [], []
    for uid, g in pos_df.groupby('uid'):
        pu.append(uid); pr.append([g[c].mean() for c in E.PROFILE_COLS])
    pr = np.array(pr); pr = pr / (np.linalg.norm(pr, axis=1, keepdims=True) + 1e-9)
    sm = pr @ pr.T; ss = {}
    for i, uid in enumerate(pu):
        row = sm[i].copy(); row[i] = -1.0
        top = np.argpartition(row, -E.PROFILE_TOPK)[-E.PROFILE_TOPK:]
        top = top[np.argsort(row[top])[::-1]]
        ss[uid] = [(pu[j], float(row[j])) for j in top if row[j] >= 0]
    # m_ref: 用户惊喜物品之间的平均相似度
    mref = {}
    for uid, its in pos_df.groupby('uid')['item']:
        its = [x for x in its if x in E.load_embeddings(ef)]
        if len(its) >= 2:
            A = np.vstack([np.asarray(E.load_embeddings(ef)[x], dtype=float) for x in its])
            A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
            S = A @ A.T
            mref[uid] = float(S[np.triu_indices_from(S, 1)].mean())
    g_ref = float(np.mean(list(mref.values()))) if mref else 0.4
    ibu = defaultdict(list)
    for _, r in fdf.iterrows():
        ibu[r['uid']].append((r['item'], r['rel'], r['dif'], r['nov_q'], r['unpop_q'],
                              r['qua'], r['label'], r['uid']))
    ui = {u: set(it for it, ts, rr in seq) for u, seq in user_seq.items()}
    valid = [u for u in ss if u in ibu and u in ui]
    rng = np.random.RandomState(SEED); lists = []
    for _ in range(200):
        t = rng.choice(valid)
        pools = E.build_sim_user_pools(t, ss, ibu, ui, K, pool_dedup='none', pos_ratio_thr=0.5)
        if pools is None or pools[0] is None:
            continue
        pp, npp = pools
        lists.append((t, [pp[i] for i in rng.choice(len(pp), K, replace=False)],
                      [npp[i] for i in rng.choice(len(npp), K, replace=False)]))
    print('=== {} | m_ref: 有自身参照 {} 用户, 均值 {:.4f}, 回退 0.40 ==='.format(
        DS, len(mref), g_ref))
    print('  扰动比  位置占比(div<1)   平均div(阈值)  平均div(旧)   平均m')
    for lvl in [0.0, 0.5, 1.0]:
        nr = int(round(lvl * K)); ms, dv_t, dv_l = [], [], []
        for t, bp, bn in lists:
            its = [x[0] for x in (bp[:K - nr] + bn[:nr])[:K]]
            r = mref.get(t, 0.4)
            for p in range(1, len(its)):
                s = [sim(its[p], its[j]) for j in range(p)]
                m = float(np.mean(s)); ms.append(m)
                r_ = min(max(r, 0.0), 0.999)
                dv_t.append(float(np.clip(1.0 - max(0.0, (m - r_) / (1 - r_)), 0, 1)))
                dv_l.append(M.div_discount.__wrapped__(s) if hasattr(M.div_discount, '__wrapped__')
                            else float(np.clip(1.0 - abs(m), 0, 1)))
        ms = np.array(ms)
        print('  {:4.0f}%      {:6.1%}          {:.4f}        {:.4f}      {:.4f}'.format(
            lvl * 100, float(np.mean(ms > 0.4)), float(np.mean(dv_t)),
            float(np.mean(dv_l)), float(ms.mean())))

run('Movies', './data/SerenLens/SerenLens_Movies.csv', 'factors_SEREN_Movies.pkl', 'emb_SEREN_Movies.pkl')
run('S2018', 'SerenLens_S2018.csv', 'factors_SEREN_S2018.pkl', 'emb_SEREN_S2018.pkl')
