# -*- coding: utf-8 -*-
"""基准: build_unlabeled_factors.py 的逐行循环到底多慢(只跑前 N 行, 外推)。

这是"重建时序嵌入"里唯一没有实测记录的一步。
"""
import os
import sys
import time
import pickle

import numpy as np
import pandas as pd

os.chdir(r'C:\Users\Administrator\Desktop\study\论文复现\code')
sys.path.insert(0, r'C:\Users\Administrator\Desktop\study\论文复现\code')

from build_s2018_data import to_serenlens_tables   # noqa: E402
from eval import SimEngine, compute_temporal_factors  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300

t0 = time.time()
full = pd.read_csv('SerenLens_S2018.csv')
full['user_id'] = full['user_id'].astype(str).str.strip()
full['item_id'] = full['item_id'].astype(str).str.strip()
user_seq, labels, label_time, item_rating = to_serenlens_tables(full)
emb = pickle.load(open('emb_SEREN_S2018.pkl', 'rb'))
pop = pickle.load(open('pop_SEREN_S2018.pkl', 'rb'))
sim_engine = SimEngine(emb)
item_ts_first, ts_max, ts_min = {}, 0, float('inf')
for seq in user_seq.values():
    for it, ts, _r in seq:
        if it not in item_ts_first or ts < item_ts_first[it]:
            item_ts_first[it] = ts
        ts_max = max(ts_max, ts)
        ts_min = min(ts_min, ts)
max_pop = max(pop.values())
unl = full[full['label'] == -1][['user_id', 'item_id', 'timestamp']].drop_duplicates(['user_id', 'item_id'])
print('setup %.1fs | 待算未标注行 %d' % (time.time() - t0, len(unl)))

t1 = time.time()
n = 0
for u, it, ts in unl.head(N).itertuples(index=False):
    seq = user_seq.get(u)
    if seq is None:
        continue
    compute_temporal_factors(it, int(ts), seq, sim_engine, pop, max_pop,
                             item_ts_first, ts_max, ts_min, item_rating, 2, None)
    n += 1
dt = time.time() - t1
print('跑 %d 行用 %.2fs -> %.2f ms/行' % (n, dt, 1000.0 * dt / max(n, 1)))
print('外推全部 %d 行: %.1f 分钟' % (len(unl), dt / max(n, 1) * len(unl) / 60.0))
