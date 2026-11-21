# -*- coding: utf-8 -*-
"""论文 §4 数据分析: 单因子判别力 AUC + 接受带峰数统计 (三个数据集)"""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, 'C:/Users/Administrator/Desktop/study/论文复现/code')
os.chdir('C:/Users/Administrator/Desktop/study/论文复现/code')
import numpy as np
from collections import Counter, defaultdict
import eval as E, metrics as M
from sklearn.metrics import roc_auc_score

DS_SET = [('Books', './data/SerenLens/SerenLens_Books.csv', 'factors_SEREN_Books.pkl',
           'emb_SEREN_Books.pkl', 'cate_SEREN_Books.pkl', 'pop_SEREN_Books.pkl'),
          ('Movies', './data/SerenLens/SerenLens_Movies.csv', 'factors_SEREN_Movies.pkl',
           'emb_SEREN_Movies.pkl', 'cate_SEREN_Movies.pkl', 'pop_SEREN_Movies.pkl'),
          ('S2018', 'SerenLens_S2018.csv', 'factors_SEREN_S2018.pkl',
           'emb_SEREN_S2018.pkl', 'cate_SEREN_S2018.pkl', 'pop_SEREN_S2018.pkl')]

for DS, seren, fp, ef, cf, pf in DS_SET:
    user_seq, labels, label_time, item_rating = E.load_serenlens(seren)
    fdf = E.pd.read_pickle(fp)
    item_cate, item_pop = E.load_item_meta(cf, pf)
    max_pop = float(max(item_pop.values())) if item_pop else 1.0
    fdf['nov_q'] = M.quantile_normalize(fdf['novelty'].values)
    fdf['unpop_q'] = M.quantile_normalize(fdf['unpopularity'].values)
    fdf['rare'] = 0.5 * (fdf['nov_q'] + fdf['unpop_q'])
    item_ts_first, ts_min, ts_max = {}, float('inf'), 0
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
    # 峰数统计(历史阶段)
    pk = {d: Counter(v['k_{}'.format(d)] for v in sp_h.values()) for d in ['n', 'u']}
    y = fdf['label'].values
    pp = M.fit_class_params(fdf['rel'][y == 1].values, fdf['dif'][y == 1].values, True)
    nn = M.fit_class_params(fdf['rel'][y == 0].values, fdf['dif'][y == 0].values, True)
    ar, qnn, quu, fs = [], [], [], []
    for _, r in fdf.iterrows():
        p = sp_h.get(r['uid'])
        qn_, qu_ = M.accommodation_components(r['nov_q'], r['unpop_q'], p)
        a = M.arousal_score(r['rel'], r['dif'], pp, nn, True)
        qnn.append(qn_); quu.append(qu_); ar.append(a)
        fs.append(float(np.clip((1 + r['qua'] * qn_ + r['qua'] * qu_) * a, 0, 10)))
    qnn, quu, ar, fs = map(np.array, (qnn, quu, ar, fs))
    print('=== {} (n={}, pos={}) ==='.format(DS, len(y), int(y.sum())))
    out = []
    for nm2, v in [('rel', fdf['rel'].values), ('dif', fdf['dif'].values),
                   ('novelty', fdf['nov_q'].values), ('unpopularity', fdf['unpop_q'].values),
                   ('quality', fdf['qua'].values), ('arousal', ar),
                   ('q_N', qnn), ('q_U', quu), ('f_sp', fs)]:
        out.append('{}={:.3f}'.format(nm2, roc_auc_score(y, v)))
    print('  AUC: ' + ' | '.join(out))
    tot = len(sp_h)
    print('  接受带峰数(历史阶段): N 单峰 {} ({:.1%}) 双峰 {} ({:.1%}) | U 单峰 {} ({:.1%}) 双峰 {} ({:.1%})'.format(
        pk['n'].get(1, 0), pk['n'].get(1, 0)/tot, pk['n'].get(2, 0), pk['n'].get(2, 0)/tot,
        pk['u'].get(1, 0), pk['u'].get(1, 0)/tot, pk['u'].get(2, 0), pk['u'].get(2, 0)/tot))
    print()
