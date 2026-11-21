# -*- coding: utf-8 -*-
"""量化: S2018 的"全量嵌入" vs "只用到分割点的嵌入" 差多少。

用来回答"原本用全量嵌入, 问题大不大":
  1. 同一物品的两个向量余弦相似度 (越接近 1, 泄漏的实际影响越小)
  2. 物品-物品相似度的**排序**相关性 (排序变了才是真影响)
  3. 因子 rel 在两种嵌入下的变化, 以及正/负样本的可分性变化

不写任何产物文件。
"""
import os
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import svds

os.chdir(r'C:\Users\Administrator\Desktop\study\论文复现\code')
BOUND = 1489905036000
DIM = 64

print('>>> 读 training.csv ...')
tr = pd.read_csv('./data/serendipity-sac2018/training.csv', low_memory=False,
                 dtype={'userId': np.int32, 'movieId': np.int32,
                        'rating': np.float32, 'timestamp': np.int64})
tr = tr.rename(columns={'movieId': 'item_id', 'userId': 'user_id'})
tr['item_id'] = tr['item_id'].astype(str).str.strip()
tr['user_id'] = tr['user_id'].astype(str).str.strip()
print('    全量 %d 行 | 分割点前 %d 行 (%.2f%%)'
      % (len(tr), int((tr['timestamp'] <= BOUND).sum()),
         100.0 * (tr['timestamp'] <= BOUND).mean()))


def build_emb(frame):
    mat = frame.groupby(['user_id', 'item_id'], as_index=False).agg({'rating': 'mean'})
    users = sorted(mat['user_id'].unique())
    items = sorted(mat['item_id'].unique())
    uidx = {u: i for i, u in enumerate(users)}
    iidx = {it: i for i, it in enumerate(items)}
    ur = mat['user_id'].map(uidx).values
    ic = mat['item_id'].map(iidx).values
    M = csr_matrix((np.ones(len(mat)), (ur, ic)), shape=(len(users), len(items)))
    M.data = np.ones_like(M.data)
    uf = np.asarray(M.sum(axis=1)).flatten()
    Mw = M.T.tocsr().astype(float).dot(diags(np.log(len(items) / (uf + 1))))
    dim = int(min(DIM, min(Mw.shape) - 1))
    U, S, _ = svds(Mw, k=dim)
    o = np.argsort(S)[::-1]
    A = U[:, o] * np.sqrt(S[o])
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    return {items[i]: A[i] for i in range(len(items))}


print('>>> 建"分割点前"嵌入 ...')
emb_t = build_emb(tr[tr['timestamp'] <= BOUND])
print('    %d 物品' % len(emb_t))
emb_f = pickle.load(open('emb_SEREN_S2018.pkl', 'rb'))
print('    现用(全量) %d 物品' % len(emb_f))

shared = sorted(set(emb_f) & set(emb_t))
print()
print('共享物品 %d (占全量 %.2f%%, 占窗口版 %.2f%%)'
      % (len(shared), 100.0 * len(shared) / len(emb_f), 100.0 * len(shared) / len(emb_t)))
print('只存在于全量版(即"分割点后才首次出现")的物品: %d' % (len(emb_f) - len(shared)))

A = np.vstack([np.asarray(emb_f[i], dtype=float) for i in shared])
B = np.vstack([np.asarray(emb_t[i], dtype=float) for i in shared])
cos = np.sum(A * B, axis=1)
print()
print('同一物品两版向量的余弦: mean=%.4f  p10=%.4f  p50=%.4f  p90=%.4f'
      % (cos.mean(), np.quantile(cos, .1), np.median(cos), np.quantile(cos, .9)))

rs = np.random.RandomState(0)
idx = rs.choice(len(shared), min(1500, len(shared)), replace=False)
fa = (A[idx] @ A[idx].T)[np.triu_indices(len(idx), 1)]
fb = (B[idx] @ B[idx].T)[np.triu_indices(len(idx), 1)]
from scipy.stats import spearmanr
rho = spearmanr(fa, fb).correlation
print('物品对相似度的 Spearman rho (全量 vs 窗口): %.4f' % rho)
print('  绝对差: mean=%.4f p90=%.4f max=%.4f'
      % (np.abs(fa - fb).mean(), np.quantile(np.abs(fa - fb), .9), np.abs(fa - fb).max()))

# ---- rel 变化: 抽样标注行 ----
print()
print('>>> 抽样标注行比较 rel ...')
full = pd.read_csv('SerenLens_S2018.csv')
full['user_id'] = full['user_id'].astype(str).str.strip()
full['item_id'] = full['item_id'].astype(str).str.strip()
seq_by_u = {}
for u, g in full.sort_values('timestamp').groupby('user_id'):
    seq_by_u[u] = list(zip(g['item_id'].values, g['timestamp'].values))
rows = full[full['label'].isin([0, 1])]
rows = rows.sample(n=min(4000, len(rows)), random_state=0)


def rel_of(item, ts, seq, emb):
    prev = [h for (h, t) in seq if t < ts and h != item][-2000:]
    v = emb.get(item)
    if v is None or not prev:
        return None
    vs = [emb.get(h) for h in prev]
    vals = [float(np.dot(v, x)) for x in vs if x is not None]
    if not vals:
        return None
    vals.sort(reverse=True)
    k = min(max(3, len(vals) // 10), 10)
    return float(np.mean(vals[:k]))


rf, rt, lab = [], [], []
for u, it, ts, y in zip(rows['user_id'], rows['item_id'], rows['timestamp'], rows['label']):
    a = rel_of(it, ts, seq_by_u.get(u, []), emb_f)
    b = rel_of(it, ts, seq_by_u.get(u, []), emb_t)
    if a is None or b is None:
        continue
    rf.append(a); rt.append(b); lab.append(y)
rf, rt, lab = np.array(rf), np.array(rt), np.array(lab)
print('    可比行 %d (pos %d / neg %d)' % (len(lab), int((lab == 1).sum()), int((lab == 0).sum())))
print('    rel 全量版: pos %.4f neg %.4f | 差 %+0.4f'
      % (rf[lab == 1].mean(), rf[lab == 0].mean(), rf[lab == 1].mean() - rf[lab == 0].mean()))
print('    rel 窗口版: pos %.4f neg %.4f | 差 %+0.4f'
      % (rt[lab == 1].mean(), rt[lab == 0].mean(), rt[lab == 1].mean() - rt[lab == 0].mean()))
print('    同一行 rel 的变化: mean|Δ|=%.4f p90|Δ|=%.4f'
      % (np.abs(rf - rt).mean(), np.quantile(np.abs(rf - rt), .9)))
try:
    from sklearn.metrics import roc_auc_score
    print('    rel 单因子 AUC: 全量 %.4f | 窗口 %.4f'
          % (roc_auc_score(lab, rf), roc_auc_score(lab, rt)))
except Exception as e:
    print('    AUC 跳过:', e)
