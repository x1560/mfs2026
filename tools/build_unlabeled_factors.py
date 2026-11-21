# -*- coding: utf-8 -*-
"""为 S2018 的「未标注行」(label=-1) 单独计算物品级因子。

用途: 零样本候选池 —— 候选来自 {标注负样本 ∪ 未标注物品},
未标注物品没有惊喜标签, 但它们的 rel/dif/novelty/unpopularity/qua
都可以从可观测数据算出, 因此可以零样本打分。

输出: factors_{model}_{data}_unlabeled.pkl  (列与 factors_*.pkl 一致)
不修改任何现有文件。
"""
import os
import sys
import pickle
import time

import numpy as np
import pandas as pd

# 路径可移植: 优先环境变量 $MDCG_ROOT, 否则用本文件上一级目录(= code 根)。
# 原来写死 'C:/Users/Administrator/Desktop/study/论文复现/code', 在 Linux 服务器上会直接崩。
_ROOT = os.environ.get('MDCG_ROOT') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)

from build_s2018_data import to_serenlens_tables          # noqa: E402
from eval import SimEngine, compute_temporal_factors        # noqa: E402
from quality import QualityIndex                            # noqa: E402

MODEL = 'SEREN'
DATA = 'S2018'
TAG = '{}_{}'.format(MODEL, DATA)
CSV = 'SerenLens_{}.csv'.format(DATA)
TRAINING = './data/serendipity-sac2018/training.csv'
MIN_HISTORY = 2

print('>>> 读取 {} ...'.format(CSV))
full = pd.read_csv(CSV)
# 与 build_s2018_data.py 一致: id 必须转字符串, 否则查不到嵌入/流行度/评分表
full['user_id'] = full['user_id'].astype(str).str.strip()
full['item_id'] = full['item_id'].astype(str).str.strip()
print('    总行 {} | label 分布: {}'.format(
    len(full), dict(full['label'].value_counts().sort_index())))

user_seq, labels, label_time, item_rating = to_serenlens_tables(full)
# to_serenlens_tables 只收录标注行, 这里换成全量评分表, 否则 qua 的 PIT 查不到更早评分
# qua 由下面 _pit_rating 单独计算(严格时序), 这里只留一个空表避免函数内回退
item_rating = {}

# ---- PIT 评分表: 每个物品在 t 之前(严格更早)的平均评分, 跨用户, 无未来信息 ----
_tr = pd.read_csv(TRAINING, usecols=['movieId', 'rating', 'timestamp'])
_tr['movieId'] = _tr['movieId'].astype(str).str.strip()
_tr = _tr.sort_values(['movieId', 'timestamp'])
_pit_ts, _pit_cum = {}, {}
for _it, _g in _tr.groupby('movieId', sort=False):
    _a = _g['timestamp'].to_numpy()
    _r = _g['rating'].to_numpy(np.float64)
    _pit_ts[_it] = _a
    _pit_cum[_it] = np.concatenate([[0.0], np.cumsum(_r)])
print('    PIT 评分表: {} 物品'.format(len(_pit_ts)))


def _pit_qua(item, ts):
    """严格早于 ts 的该物品平均评分 -> qua; 无更早评分返回 0.5 (论文口径)。"""
    a = _pit_ts.get(item)
    if a is None:
        return None
    n = int(np.searchsorted(a, ts, side='left'))
    if n == 0:
        return None
    return float(_pit_cum[item][n] / n)
print('    user_seq: {} 用户 | 标注项 {}'.format(len(user_seq), len(labels)))

emb = pickle.load(open('emb_{}.pkl'.format(TAG), 'rb'))
pop = pickle.load(open('pop_{}.pkl'.format(TAG), 'rb'))
sim_engine = SimEngine(emb)
print('    嵌入 {} 物品 | 流行度 {} 物品'.format(len(emb), len(pop)))

item_ts_first, ts_max, ts_min = {}, 0, float('inf')
for seq in user_seq.values():
    for it, ts, _r in seq:
        if it not in item_ts_first or ts < item_ts_first[it]:
            item_ts_first[it] = ts
        ts_max = max(ts_max, ts)
        ts_min = min(ts_min, ts)

# S2018 无 Amazon 质量源, qua 用 PIT 评分回退(与标注行同源), 故不需要 QualityIndex
qidx = None
max_pop = max(pop.values())

unl = full[full['label'] == -1][['user_id', 'item_id', 'timestamp']]
unl = unl.drop_duplicates(['user_id', 'item_id'])
print('>>> 待计算未标注行: {} 条'.format(len(unl)))

rows = []
t0 = time.time()
n_none = 0
for i, (u, it, ts) in enumerate(unl.itertuples(index=False)):
    seq = user_seq.get(u)
    if seq is None:
        n_none += 1
        continue
    f = compute_temporal_factors(
        it, int(ts), seq, sim_engine, pop, max_pop,
        item_ts_first, ts_max, ts_min, item_rating, MIN_HISTORY, qidx)
    if f is None:
        n_none += 1
        continue
    rel, dif, nov, unpop, qua = f
    _rt = _pit_qua(it, int(ts))
    qua = 0.5 if _rt is None else max(0.0, min(1.0, (_rt - 1.0) / 4.0))
    rows.append({'uid': u, 'item': it, 'label': -1,
                 'rel': float(rel), 'dif': float(dif), 'novelty': float(nov),
                 'unpopularity': float(unpop), 'qua': float(qua)})
    if i and i % 25000 == 0:
        print('    {}/{}  {:.0f}s'.format(i, len(unl), time.time() - t0))

df = pd.DataFrame(rows)
out = 'factors_{}_unlabeled.pkl'.format(TAG)
df.to_pickle(out)
print('>>> {}: {} 行 (跳过 {} 条, 历史不足) | {:.1f}s'.format(
    out, len(df), n_none, time.time() - t0))
for c in ['rel', 'dif', 'novelty', 'unpopularity', 'qua']:
    print('      {}: 均值 {:.4f} | p05 {:.4f} | p50 {:.4f} | p95 {:.4f}'.format(
        c, df[c].mean(), df[c].quantile(.05), df[c].median(), df[c].quantile(.95)))
