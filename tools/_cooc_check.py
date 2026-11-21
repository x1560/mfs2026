# -*- coding: utf-8 -*-
"""Amazon 共现空间 vs 现嵌入空间: div 的输入到底长什么样"""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, 'C:/Users/Administrator/Desktop/study/论文复现/code')
os.chdir('C:/Users/Administrator/Desktop/study/论文复现/code')
import numpy as np
from collections import defaultdict
import eval as E, metrics as M, amazon_sim

K = 10; SEED = 42
user_seq, labels, label_time, item_rating = E.load_serenlens('./data/SerenLens/SerenLens_Movies.csv')
fdf = E.pd.read_pickle('factors_SEREN_Movies.pkl')
fdf['nov_q'] = M.quantile_normalize(fdf['novelty'].values)
fdf['unpop_q'] = M.quantile_normalize(fdf['unpopularity'].values)
fdf['rare'] = 0.5 * (fdf['nov_q'] + fdf['unpop_q'])
emb = E.SimEngine(E.load_embeddings('emb_SEREN_Movies.pkl'))
import sys as _s
MU = int(_s.argv[1]) if len(_s.argv)>1 else 1
cooc = amazon_sim.build('Movies', min_user_items=MU)
print('  覆盖率: 因子表物品 {} 个, 在 Amazon 相似度里 {} 个'.format(
    fdf['item'].nunique(), sum(1 for it in fdf['item'].unique() if it in cooc.index)))
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
ibu = defaultdict(list)
for _, r in fdf.iterrows():
    ibu[r['uid']].append((r['item'], r['rel'], r['dif'], r['nov_q'], r['unpop_q'],
                          r['qua'], r['label'], r['uid']))
ui = {u: set(it for it, ts, rr in seq) for u, seq in user_seq.items()}
def stats(simf, tag):
    # 用户内: 该用户惊喜物品之间
    wu = []
    for uid, its in pos_df.groupby('uid')['item']:
        its = [x for x in its]
        if len(its) >= 2:
            v = [simf(a, b) for i, a in enumerate(its) for b in its[i + 1:]]
            if v: wu.append(float(np.mean(v)))
    rng = np.random.RandomState(SEED)
    valid = [u for u in ss if u in ibu and u in ui]
    wp, wn, cr, lm = [], [], [], []
    for _ in range(300):
        u = valid[rng.randint(len(valid))]
        pools = E.build_sim_user_pools(u, ss, ibu, ui, K, pool_dedup='none', pos_ratio_thr=0.5)
        if pools is None or pools[0] is None: continue
        pp, nn = pools
        pi = [x[0] for x in pp[:2]]; ni = [x[0] for x in nn[:8]]
        allit = pi + ni
        wp += [simf(a, b) for i, a in enumerate(pi) for b in pi[i + 1:]]
        wn += [simf(a, b) for i, a in enumerate(ni) for b in ni[i + 1:]]
        cr += [simf(a, b) for a in pi for b in ni]
        for p in range(1, len(allit)):
            lm.append(float(np.mean([simf(allit[p], allit[j]) for j in range(p)])))
    print('  {:14s} 用户内惊喜={:.4f} | 池内 pos-pos={:.4f} neg-neg={:.4f} pos-neg={:.4f} | 列表内 m={:.4f}'.format(
        tag, float(np.mean(wu)) if wu else float('nan'), float(np.mean(wp)), float(np.mean(wn)),
        float(np.mean(cr)), float(np.mean(lm))))
print()
print('=== Movies: div 的相似度在两个空间里 ===')
stats(emb.sim, '现嵌入')
stats(cooc.sim, 'Amazon cooc u>={}'.format(MU))
