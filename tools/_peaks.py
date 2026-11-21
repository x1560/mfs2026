# -*- coding: utf-8 -*-
"""单峰/双峰用户比例 (历史阶段 vs 惊喜池个性化后)"""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, 'C:/Users/Administrator/Desktop/study/论文复现/code')
os.chdir('C:/Users/Administrator/Desktop/study/论文复现/code')
import numpy as np
from collections import defaultdict, Counter
import eval as E, metrics as M
src = open(os.path.join(_HERE, '_f5.py'), encoding='utf-8').read().split('\ndef run(')[0]
exec(src)
for DS, seren, fp, cf, pf in [('Movies', './data/SerenLens/SerenLens_Movies.csv',
                               'factors_SEREN_Movies.pkl', 'cate_SEREN_Movies.pkl', 'pop_SEREN_Movies.pkl')]:
    user_seq, labels, label_time, item_rating = E.load_serenlens(seren)
    fdf = E.pd.read_pickle(fp)
    item_cate, item_pop = E.load_item_meta(cf, pf)
    max_pop = float(max(item_pop.values())) if item_pop else 1.0
    all_items, item_ts_first, ts_min, ts_max = set(item_pop), {}, float('inf'), 0
    for seq in user_seq.values():
        for it, ts, r in seq:
            if it not in item_ts_first or ts < item_ts_first[it]:
                item_ts_first[it] = ts
            ts_min, ts_max = min(ts_min, ts), max(ts_max, ts)
    a2 = sorted(set(item_pop) | set(item_ts_first))
    rn = np.array([(item_ts_first.get(i, ts_min) - ts_min) / (ts_max - ts_min + 1e-9) for i in a2])
    ru = np.array([1. - float(item_pop.get(i, 0)) / (max_pop + 1e-9) for i in a2])
    nm = {i: float(v) for i, v in zip(a2, E.quantile_transform_from_reference(rn, fdf['novelty'].values))}
    um = {i: float(v) for i, v in zip(a2, E.quantile_transform_from_reference(ru, fdf['unpopularity'].values))}
    rm = {i: .5 * (nm[i] + um[i]) for i in a2}
    sp_h = M.build_user_scarcity_params(user_seq, rm, nm, um, min_history=5)
    print('=== {} | 历史阶段 {} 用户 ==='.format(DS, len(sp_h)))
    for dim in ['n', 'u']:
        c = Counter(v['k_{}'.format(dim)] for v in sp_h.values())
        tot = sum(c.values())
        print('  {}: 单峰 {} ({:.1%}) | 双峰 {} ({:.1%})'.format(
            dim.upper(), c.get(1, 0), c.get(1, 0) / tot, c.get(2, 0), c.get(2, 0) / tot))
    # 个性化
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
        t = np.argpartition(row, -E.PROFILE_TOPK)[-E.PROFILE_TOPK:]
        t = t[np.argsort(row[t])[::-1]]
        ss[uid] = [(pu[j], float(row[j])) for j in t if row[j] >= 0]
    posnu = defaultdict(list)
    for _, rr in pos_df.iterrows():
        posnu[rr['uid']].append((float(rr['nov_q']), float(rr['unpop_q'])))
    sp_p = M.personalize_scarcity_centers(sp_h, ss, posnu, center='median', min_pool=5)
    print('  个性化后 ({} 用户):'.format(len(sp_p)))
    for dim in ['n', 'u']:
        c = Counter(v['k_{}'.format(dim)] for v in sp_p.values())
        tot = sum(c.values())
        print('  {}: 单峰 {} ({:.1%}) | 双峰 {} ({:.1%})'.format(
            dim.upper(), c.get(1, 0), c.get(1, 0) / tot, c.get(2, 0), c.get(2, 0) / tot))
