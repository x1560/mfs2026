# -*- coding: utf-8 -*-
"""用 Amazon 的全部评分重建 SerenLens 数据集的因子表。
产物:emb_{tag}.pkl  cate_{tag}.pkl  pop_{tag}.pkl  factors_{tag}.pkl
用法:python build_amz_factors.py --data Books  --model AMZ
    python build_amz_factors.py --data Movies --model AMZ
"""
import argparse
import pickle
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from eval import SimEngine, build_factor_table, load_serenlens

AMZ_ORDER = {
    'Books':  ['user_id', 'item_id', 'rating', 'timestamp'],
    'Movies': ['item_id', 'user_id', 'rating', 'timestamp'],
}
SERENLENS = {
    'Books':  './data/SerenLens/SerenLens_Books.csv',
    'Movies': './data/SerenLens/SerenLens_Movies.csv',
}
AMAZON = {
    'Books':  './data/amazon/ratings_Books.csv',
    'Movies': './data/amazon/Movies_and_TV.csv',
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='Movies', choices=['Books', 'Movies'])
    ap.add_argument('--model', default='AMZ')
    ap.add_argument('--serenlens', default=None)
    ap.add_argument('--amazon', default=None)
    ap.add_argument('--emb', default=None, help='默认 amzemb_{data}_u5_i5.pkl')
    ap.add_argument('--min_history', type=int, default=2, help='计算 rel/dif 所需的最少历史交互数。必须与出厂 build_all_data.py 的默认值一致, 否则因子表行数/正样本数不可比')
    ap.add_argument('--n_cate', type=int, default=10)
    ap.add_argument('--chunksize', type=int, default=2000000)
    return ap.parse_args()


def main():
    a = parse_args()
    tag = '{}_{}'.format(a.model, a.data)
    seren_path = a.serenlens or SERENLENS[a.data]
    amz_path = a.amazon or AMAZON[a.data]
    emb_path = a.emb or 'amzemb_{}_u5_i5.pkl'.format(a.data)

    print('>>> 标注文件 {}'.format(seren_path))
    sl_seq, labels, label_time, item_rating = load_serenlens(seren_path)
    labels = {u: dict(v) for u, v in labels.items()}
    print('    {} 条 | {} 用户 | {} 物品 | 正 {}'.format(
        len(label_time), len(labels),
        len(set(it for u in labels for it in labels[u])),
        sum(1 for u in labels for it in labels[u] if labels[u][it] == 1)))
    users = set(labels.keys())
    print('>>> Amazon data: {}'.format(amz_path))
    head = pd.read_csv(amz_path, nrows=2)
    cols = list(head.columns)[:4]
    names = AMZ_ORDER[a.data]
    parts, scanned = [], 0
    for ch in pd.read_csv(amz_path, usecols=cols, chunksize=a.chunksize, dtype=str):
        scanned += len(ch)
        ch.columns = names
        ch['user_id'] = ch['user_id'].str.strip().str.lower()
        ch['item_id'] = ch['item_id'].str.strip().str.lower()
        ch = ch[ch['user_id'].isin(users)]
        if len(ch):
            parts.append(ch)
        print('    已扫 {:>10,} 行 | user in labeled data: {:>9,}'.format(
            scanned, sum(len(p) for p in parts)), flush=True)
    amz = pd.concat(parts, ignore_index=True)
    amz['timestamp'] = pd.to_numeric(amz['timestamp'], errors='coerce')
    amz['rating'] = pd.to_numeric(amz['rating'], errors='coerce')
    amz = amz.dropna(subset=['timestamp', 'rating'])
    amz['timestamp'] = amz['timestamp'].astype(np.int64)
    print('    命中 {} 条 | {} 用户 | {} 物品'.format(
        len(amz), amz['user_id'].nunique(), amz['item_id'].nunique()))

    # ---- 历史：SerenLens 的历史 ∪ Amazon 里新增的 (item, ts) ----
    user_map = defaultdict(dict)          # uid -> {(item, ts): rating}
    for u, seq in sl_seq.items():
        if u not in users:
            continue
        for it, ts, r in seq:
            user_map[u][(it, ts)] = r
    added = 0
    for uid, iid, ts, rt in zip(amz['user_id'].values, amz['item_id'].values,
                                amz['timestamp'].values, amz['rating'].values):
        d = user_map.get(uid)
        if d is None:
            continue
        k = (iid, int(ts))
        if k not in d:
            d[k] = float(rt)
            added += 1
    user_seq = {u: sorted([(it, ts, r) for (it, ts), r in d.items()], key=lambda x: x[1])
                for u, d in user_map.items()}
    print('    SerenLens {} 条 + Amazon 新增 {} 条 -> {} 用户'.format(
        sum(len(s) for s in sl_seq.values()), added, len(user_seq)))
    sizes = np.array([len(v) for v in user_seq.values()])
    print('    历史/用户: 中位 {} 均值 {:.1f} 最大 {}'.format(
        int(np.median(sizes)), float(sizes.mean()), int(sizes.max())))

    print('    标注: {} 条 | {} 用户'.format(len(label_time), len(labels)))

    # ---- 流行度 / 首次交互时间----
    pop = defaultdict(int)
    for _u, _seq in user_seq.items():
        _seen = set()
        for it, ts, _r in _seq:
            if it not in _seen:
                _seen.add(it)
                pop[it] += 1
    pop = dict(pop)
    item_ts_first, ts_max, ts_min = {}, 0, float('inf')
    for seq in user_seq.values():
        for it, ts, _r in seq:
            if it not in item_ts_first or ts < item_ts_first[it]:
                item_ts_first[it] = ts
            ts_max = max(ts_max, ts)
            ts_min = min(ts_min, ts)
    print('    流行度: {} 物品 | 时间窗 [{}, {}]'.format(len(pop), ts_min, ts_max))

    emb = pickle.load(open(emb_path, 'rb'))
    print('    嵌入 {}: {} 物品'.format(emb_path, len(emb)))
    items = sorted(emb.keys())
    X = np.vstack([np.asarray(emb[i], dtype=float) for i in items])
    n_cate = int(min(a.n_cate, max(2, len(items) // 10)))
    cl = KMeans(n_clusters=n_cate, random_state=42, n_init=10).fit_predict(X)
    cate = {items[i]: 'c{}'.format(cl[i]) for i in range(len(items))}
    for name, obj in [('emb_{}.pkl'.format(tag), emb),
                      ('cate_{}.pkl'.format(tag), cate),
                      ('pop_{}.pkl'.format(tag), pop)]:
        with open(name, 'wb') as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        print('    -> {}'.format(name))

    # ---- 因子表 ----
    t0 = time.time()
    sim_engine = SimEngine(emb)
    fdf = build_factor_table(user_seq, labels, label_time, item_rating, sim_engine,
                             pop, max(pop.values()),
                             item_ts_first=item_ts_first, ts_global_max=ts_max,
                             ts_global_min=ts_min, min_history=a.min_history,
                             quality_index=None)
    fdf.to_pickle('factors_{}.pkl'.format(tag))
    print('    factors_{}.pkl: {} 条 (pos={}, neg={}) | {:.1f}s'.format(
        tag, len(fdf), int((fdf['label'] == 1).sum()),
        int((fdf['label'] == 0).sum()), time.time() - t0))
    for c in ['rel', 'dif', 'novelty', 'unpopularity', 'qua']:
        print('      {:12s} mean={:.4f} p05={:.4f} p50={:.4f} p95={:.4f}'.format(
            c, fdf[c].mean(), fdf[c].quantile(.05), fdf[c].median(), fdf[c].quantile(.95)))


if __name__ == '__main__':
    main()
