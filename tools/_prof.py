# -*- coding: utf-8 -*-
"""画像相似的用户, 他们的惊喜物品在物品空间里像不像?"""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, 'C:/Users/Administrator/Desktop/study/论文复现/code')
os.chdir('C:/Users/Administrator/Desktop/study/论文复现/code')
import numpy as np
from collections import defaultdict
import eval as E, metrics as M

def run(DS, fp, ef):
    fdf = E.pd.read_pickle(fp)
    emb = E.load_embeddings(ef)
    V = {}
    for it, v in emb.items():
        v = np.asarray(v, dtype=float); V[it] = v / (np.linalg.norm(v) + 1e-12)
    fdf['nov_q'] = M.quantile_normalize(fdf['novelty'].values)
    fdf['unpop_q'] = M.quantile_normalize(fdf['unpopularity'].values)
    fdf['rare'] = 0.5 * (fdf['nov_q'] + fdf['unpop_q'])
    pos = defaultdict(list)
    for _, r in fdf[fdf['label'] == 1].iterrows():
        if r['item'] in V: pos[r['uid']].append(r['item'])
    pu, pr = [], []
    for uid, g in fdf[fdf['label'] == 1].groupby('uid'):
        pu.append(uid); pr.append([g[c].mean() for c in E.PROFILE_COLS])
    pr = np.array(pr); pr = pr / (np.linalg.norm(pr, axis=1, keepdims=True) + 1e-9)
    S = pr @ pr.T
    def pair_mean(A, B):
        if A is B:
            v = [float(V[a] @ V[b]) for i, a in enumerate(A) for b in A[i + 1:]]
        else:
            v = [float(V[a] @ V[b]) for a in A for b in B]
        return float(np.mean(v)) if v else float('nan')
    within = [pair_mean(v, v) for v in pos.values() if len(v) >= 2]
    print('=== {} ==='.format(DS))
    print('  同一用户内部的惊喜互相似度: {:.4f} ({} 用户)'.format(
        float(np.nanmean(within)), len(within)))
    iu = np.triu_indices(len(pu), 1)
    pcos = S[iu]
    print('  画像 cos 分布: min={:.4f} p25={:.4f} 中位={:.4f} max={:.4f}'.format(
        pcos.min(), np.quantile(pcos, .25), np.median(pcos), pcos.max()))
    rs = np.random.RandomState(0)
    rnd = []
    for _ in range(400):
        a, b = rs.randint(len(pu)), rs.randint(len(pu))
        if a == b: continue
        v = pair_mean(pos.get(pu[a], []), pos.get(pu[b], []))
        if not np.isnan(v): rnd.append(v)
    print('  随机用户对(对照): 跨用户惊喜相似度 = {:.4f} ({} 对)'.format(
        float(np.nanmean(rnd)), len(rnd)))
    print('  按画像相似度排名分档, 与目标用户惊喜物品的跨用户相似度:')
    print('    邻居排名     画像cos      跨用户惊喜相似度')
    for lo, hi in [(1, 2), (2, 6), (6, 11), (11, 21), (21, 51)]:
        sims, cos_ = [], []
        for i, uid in enumerate(pu):
            if len(pos.get(uid, [])) < 1:
                continue
            row = S[i].copy(); row[i] = -1.0
            order = np.argsort(row)[::-1][lo - 1:hi - 1]
            for j in order:
                if row[j] <= 0: continue
                v = pair_mean(pos[uid], pos.get(pu[j], []))
                if not np.isnan(v):
                    sims.append(v); cos_.append(float(row[j]))
        if sims:
            print('    {:>3d}-{:<3d}   {:.4f}      {:.4f}  ({} 对)'.format(
                lo, hi - 1, float(np.mean(cos_)), float(np.mean(sims)), len(sims)))
run('Movies', 'factors_SEREN_Movies.pkl', 'emb_SEREN_Movies.pkl')
run('S2018', 'factors_SEREN_S2018.pkl', 'emb_SEREN_S2018.pkl')
