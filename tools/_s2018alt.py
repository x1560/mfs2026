# -*- coding: utf-8 -*-
"""S2018 若不用低分负样本(改用调查里"没被标记"的条目), 相似度结构会翻转吗?"""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, 'C:/Users/Administrator/Desktop/study/论文复现/code')
os.chdir('C:/Users/Administrator/Desktop/study/论文复现/code')
import numpy as np
from collections import defaultdict
import eval as E, metrics as M

LAB6 = ["s_ser_find", "s_ser_imp", "s_ser_rec", "m_ser_find", "m_ser_imp", "m_ser_rec"]
ans = E.pd.read_csv('data/serendipity-sac2018/answers.csv', low_memory=False)
ans['uid'] = ans['userId'].astype(str); ans['item'] = ans['movieId'].astype(str)
def tb(s):
    return s.astype(str).str.strip().str.lower().isin(['true', '1', 'yes', 'y'])
ans['label'] = E.pd.DataFrame({c: tb(ans[c]) for c in LAB6}).any(axis=1).astype(int)
print('  answers: {} 条 / {} 用户 / 正 {} / 负 {}'.format(
    len(ans), ans['uid'].nunique(), int((ans['label'] == 1).sum()), int((ans['label'] == 0).sum())))
emb = E.load_embeddings('emb_SEREN_S2018.pkl')
V = {}
for it, v in emb.items():
    v = np.asarray(v, dtype=float); V[it] = v / (np.linalg.norm(v) + 1e-12)
print('  嵌入物品数 {} (答案文件里能对上的: {}/{} 行)'.format(
    len(V), int(ans['item'].isin(V).sum()), len(ans)))
def pairmean(items):
    if len(items) < 2: return None
    A = np.vstack([V[i] for i in items]); S = A @ A.T
    return float(S[np.triu_indices_from(S, 1)].mean())
pos, neg = defaultdict(list), defaultdict(list)
for _, r in ans.iterrows():
    if r['item'] in V:
        (pos if r['label'] == 1 else neg)[r['uid']].append(r['item'])
pp = [pairmean(v) for v in pos.values() if pairmean(v) is not None]
nn = [pairmean(v) for v in neg.values() if pairmean(v) is not None]
# 同一批用户内比较(避免用户群不同造成偏差)
both = [u for u in pos if u in neg and pairmean(pos[u]) is not None and pairmean(neg[u]) is not None]
print()
print('  同用户内部:')
print('    正样本×正样本  {:.4f}  ({} 用户)'.format(float(np.mean(pp)), len(pp)))
print('    负样本×负样本  {:.4f}  ({} 用户)'.format(float(np.mean(nn)), len(nn)))
if both:
    a = [pairmean(pos[u]) for u in both]; b = [pairmean(neg[u]) for u in both]
    print('    同一批 {} 个用户内配对比较: 正 {:.4f} vs 负 {:.4f}  (Δ={:+.4f})'.format(
        len(both), float(np.mean(a)), float(np.mean(b)), float(np.mean(a) - np.mean(b))))
# 对照: 当前口径(低分负样本)
fdf = E.pd.read_pickle('factors_SEREN_S2018.pkl')
pos2, neg2 = defaultdict(list), defaultdict(list)
for _, r in fdf.iterrows():
    if r['item'] in V:
        (pos2 if r['label'] == 1 else neg2)[r['uid']].append(r['item'])
p2 = [pairmean(v) for v in pos2.values() if pairmean(v) is not None]
n2 = [pairmean(v) for v in neg2.values() if pairmean(v) is not None]
print()
print('  对照 —— 当前口径(负样本=低分物品):')
print('    正×正 {:.4f} ({} 用户) | 负×负 {:.4f} ({} 用户)'.format(
    float(np.mean(p2)), len(p2), float(np.mean(n2)), len(n2)))
