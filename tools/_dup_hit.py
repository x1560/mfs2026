# -*- coding: utf-8 -*-
"""冗余池对命中率的影响: div 能不能救回来"""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, 'C:/Users/Administrator/Desktop/study/论文复现/code')
os.chdir('C:/Users/Administrator/Desktop/study/论文复现/code')
import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score
import eval as E, metrics as M
src = open(os.path.join(_HERE, '_f5.py'), encoding='utf-8').read().split('\ndef run(')[0]
exec(src)
USE_COPULA = True
C = build('Movies')
DUP = 3                      # 把"最吸引人的负样本"复制成 DUP 份(占槽位)
def fsp_of(x, target):
    pp, np_i = C['per_user_arousal'].get(target, (C['pos_params'], C['neg_params']))
    p = C['sp'].get(target)
    return M.f_sp_score(x[1], x[2], x[3], x[4], x[5], pp, np_i, USE_COPULA, p)
def greedy(pool, target, div_on):
    pp, np_i = C['per_user_arousal'].get(target, (C['pos_params'], C['neg_params']))
    trust = C['per_user_trust'].get(target, (0.48, 0.5))
    order = [0]
    rem = list(range(1, len(pool)))
    while rem:
        best, bg = None, -1e18
        for i in rem:
            trial = order + [i]
            mlst = [(pool[j][0], pool[j][1], pool[j][2], pool[j][3], pool[j][4],
                     pool[j][5], target) for j in trial]
            g = M.mdcg_score(mlst, pp, np_i,
                             sim_func=(C['sim_engine'].sim if div_on else None),
                             use_copula=USE_COPULA, k=len(trial),
                             user_scarcity_params=C['sp'], trust_params=trust)
            if g > bg: bg, best = g, i
        order.append(best); rem.remove(best)
    return order
rng = np.random.RandomState(42)
valid = [u for u in C['surprise_sim'] if u in C['items_by_uid'] and u in C['user_items_set']]
res = {d: {'h2': [], 'h4': [], 'auc': []} for d in [True, False]}
n = 0
for _ in range(500):
    u = valid[rng.randint(len(valid))]
    p = E.build_sim_user_pools(u, C['surprise_sim'], C['items_by_uid'], C['user_items_set'],
                               10, pool_dedup='none', pos_ratio_thr=0.5)
    if p is None or p[0] is None: continue
    P, N = list(p[0][:2]), list(p[1][:8])
    if len(P) < 2 or len(N) < 6: continue
    # 复制"最吸引人的负样本" DUP 份, 挤掉其它负样本
    fs = [fsp_of(x, u) for x in N]
    hot = N[int(np.argmax(fs))]
    pool = P + [hot] * DUP + N[:max(0, 8 - DUP)]
    pos_items = {x[0] for x in P}
    for div_on in [True, False]:
        order = greedy(pool, u, div_on)
        ranked = [pool[i][0] for i in order]
        labels = [pool[i][6] for i in order]
        res[div_on]['h2'].append(1.0 if any(it in pos_items for it in ranked[:2]) else 0.0)
        res[div_on]['h4'].append(1.0 if any(it in pos_items for it in ranked[:4]) else 0.0)
        if len(set(labels)) > 1:
            res[div_on]['auc'].append(roc_auc_score(labels, np.arange(len(labels))[::-1].astype(float)))
    n += 1
print('  冗余池 ({} 个池子, 把最吸引人的负样本复制 {} 份)'.format(n, DUP))
print('  {:10s} {:>8s} {:>8s} {:>10s}'.format('div', 'Hit@2', 'Hit@4', 'OrderAUC'))
for div_on in [True, False]:
    r = res[div_on]
    print('  {:10s} {:>8.4f} {:>8.4f} {:>10.4f}'.format(
        '开' if div_on else '关', np.mean(r['h2']), np.mean(r['h4']),
        float(np.nanmean(r['auc']))))
