# -*- coding: utf-8 -*-
"""验证增量贪心与 _order_gain 逐位等价"""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, 'C:/Users/Administrator/Desktop/study/论文复现/code')
os.chdir('C:/Users/Administrator/Desktop/study/论文复现/code')
import numpy as np
from collections import defaultdict
import eval as E, metrics as M
src = open(os.path.join(_HERE, '_f5.py'), encoding='utf-8').read().split('\ndef run(')[0]
exec(src)
C = build('Movies')
item_cate, item_pop = E.load_item_meta('cate_SEREN_Movies.pkl', 'pop_SEREN_Movies.pkl')
max_pop = float(max(item_pop.values())) if item_pop else 1.0
ibu = defaultdict(list)
for _, r in C['fdf'].iterrows():
    ibu[r['uid']].append((r['item'], r['rel'], r['dif'], r['nov_q'], r['unpop_q'],
                          r['qua'], r['label'], r['uid']))
ui = C['user_items_set']
valid = [u for u in C['surprise_sim'] if u in ibu and u in ui]
rng = np.random.RandomState(7)
def make_rctx(uid, pool):
    pp, np_i = C['per_user_arousal'].get(uid, (C['pos_params'], C['neg_params']))
    return dict(pool=pool, uid=uid, pos_items=set(), pp=pp, np=np_i,
                p_acc=C['sp'].get(uid), trust=(0.48, 0.5, 0.0, 'constant'),
                sf=C['sim_engine'].sim, use_copula=True,
                item_cate=item_cate or {}, item_rating_g={},
                hist_cate=[], hist_rating=[],
                item_pop=item_pop or {}, max_pop=max_pop)
n_cmp = 0; bad = 0
for _ in range(12):
    u = valid[rng.randint(len(valid))]
    p = E.build_sim_user_pools(u, C['surprise_sim'], ibu, ui, 10, pool_dedup='none', pos_ratio_thr=0.5)
    if p is None or p[0] is None: continue
    pool = list(p[0][:2]) + list(p[1][:8])
    rctx = make_rctx(u, pool)
    fsp = E._precompute_fsp(rctx)
    start = int(rng.randint(len(pool)))
    for m in ['mdcg', 'epc', 'ild', 'ndcg_seren']:
        if not E._fast_capable(m, rctx):
            print('  {} 不可增量, 跳过'.format(m)); continue
        o_old = E._greedy_order(m, rctx, start)
        o_new = E._greedy_order_fast(m, rctx, start, fsp)
        n_cmp += 1
        if o_old != o_new:
            bad += 1
            print('  ✗ {} 不一致: old={} new={}'.format(m, o_old, o_new))
print('对比 {} 组, 不一致 {} 组'.format(n_cmp, bad))
print('结论:', 'PASS' if bad == 0 else 'FAIL')
