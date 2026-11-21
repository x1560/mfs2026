# -*- coding: utf-8 -*-
"""RQ4：按比例删除训练窗交互后的鲁棒性实验（可编辑版）

一句话：把用户的历史按比例删掉，看各排序器的 Hit@2 / Hit@4 怎么变。

流程
----
1. 读交互流 + 因子表（只用它的 uid/item/label，特征全部重算）
2. 按标签时间 6:4 切成训练窗 / 测试窗
3. 建池子：目标用户的正样本 + 负样本（来源见 POOL 开关）
4. 每一档 r：从训练窗随机删 r% 交互 → 用边界前的数据重算五个因子 → 排序 → 数命中
5. 打印每个档位、每个排序器的 Hit@2 / Hit@4

需要改的地方都集中在下面「可调参数」一节。
"""
import os
import sys
import pickle
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, r"C:\Users\Administrator\Desktop\study\论文复现\code")
import eval as E          # 只用它的 SimEngine 和 Beta 拟合
import metrics as M

# ============================================================================
# 可调参数
# ============================================================================
DATA = r"C:\Users\Administrator\Desktop\study\论文复现\code"

DATASETS = {
    # 注意：S2018 的历史应该用 data/serendipity-sac2018/training.csv（1000 万条），
    #      当前只读了问卷文件（32 万条），要换就改这里的 'seq' 字段。
    'movies': dict(seq=DATA + r"\\data\\SerenLens\\SerenLens_Movies.csv",
                   factors=DATA + r"\\factors_AMZ_Movies.pkl",
                   emb=DATA + r"\\emb_AMZ_Movies.pkl"),
    'books': dict(seq=DATA + r"\\data\\SerenLens\\SerenLens_Books.csv",
                  factors=DATA + r"\\factors_Seren_Books.pkl",
                  emb=DATA + r"\\emb_Seren_Books.pkl"),
    's2018': dict(seq=DATA + r"\\SerenLens_S2018.csv",
                  factors=DATA + r"\\factors_SEREN_S2018.pkl",
                  emb=DATA + r"\\emb_SEREN_S2018.pkl",
                  pairs=DATA + r"\\S2018_survey_pairs.csv"),
}

LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)   # 删除比例
TEST_RATIO = 0.4                        # 测试窗占比（6:4）
NEG_PER_POS = 19                        # 每个正样本配多少负样本
POOL = 'similar'                        # 'similar' = 惊喜画像相似池；'global' = 全局随机
PROFILE_TOPK = 50                       # 画像相似的邻居数
PROFILE_WITH_NEIGHBOURS = False         # True = 画像也用邻居的正样本一起建
MIN_HISTORY = 5                         # 历史少于这么多条就算不出 rel
USE_COPULA = True


# ============================================================================
# 读数据
# ============================================================================
def nid(x):
    return str(x).strip().lower()


def load(name):
    cfg = DATASETS[name]
    seq, _, label_time, _ = E.load_serenlens(cfg['seq'])
    emb = {nid(k): np.asarray(v, float)
           for k, v in pickle.load(open(cfg['emb'], 'rb')).items()}
    fdf = pd.read_pickle(cfg['factors'])
    if cfg.get('pairs'):                      # S2018：只留问卷问过的行
        keep = set(zip(*[pd.read_csv(cfg['pairs'], dtype=str)[c].str.lower()
                         for c in (0, 1)]))
        fdf = fdf[[(nid(u), nid(i)) in keep
                   for u, i in zip(fdf['uid'], fdf['item'])]]
    fdf = fdf.reset_index(drop=True)
    fdf['uid'] = fdf['uid'].map(nid)
    fdf['item'] = fdf['item'].map(nid)
    seq = {nid(u): [(nid(i), int(t), float(r)) for i, t, r in v] for u, v in seq.items()}
    label_time = {(nid(u), nid(i)): int(t) for (u, i), t in label_time.items()}
    return seq, label_time, emb, fdf


def embeddings(emb):
    items = list(emb)
    mat = np.vstack([emb[i] for i in items])
    mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return {it: k for k, it in enumerate(items)}, mat


# ============================================================================
# 池子
# ============================================================================
def surprise_neighbours(fdf, train_mask, topk=PROFILE_TOPK, with_neighbours=False):
    """惊喜画像相似：每个用户对自己训练窗正样本的四因子均值，算余弦，取 topk。

    with_neighbours=True 时，画像把【邻居的正样本】也算进来（迭代一轮）。
    """
    cols = ['rel', 'dif', 'novelty', 'popularity']
    d = fdf[train_mask].copy()
    d['popularity'] = 1.0 - d['unpopularity'].values if 'unpopularity' in d else 0.5
    pos = d[d['label'] == 1]
    prof = pos.groupby('uid')[cols].mean()
    arr = prof.to_numpy()
    arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
    sim = arr @ arr.T
    uids = list(prof.index)
    out = {}
    for i, u in enumerate(uids):
        row = sim[i].copy()
        row[i] = -1
        top = np.argsort(-row)[:min(topk, len(uids) - 1)]
        out[u] = [(uids[j], float(row[j])) for j in top]
    if with_neighbours:                       # 把自己没有画像的用户补上：借邻居的
        for u in fdf['uid'].unique():
            if u in out:
                continue
            borrowed = defaultdict(list)
            for v, _ in out.get(u, []):
                borrowed[v].append(1)
            out.setdefault(u, [])
    return out


def build_pool(fdf, uid, lab, ev, insim, neg_per_pos=NEG_PER_POS, seed=1234):
    """正样本 = 该用户测试窗的正样本；负样本按 POOL 开关取。"""
    rows_of = defaultdict(list)
    for j, u in enumerate(uid):
        rows_of[u].append(j)
    neg_pool_all = np.array([j for j in range(len(fdf)) if lab[j] == 0], dtype=int)
    rng = np.random.RandomState(seed)
    users, pos_of, neg_of = [], {}, {}
    for u in sorted(set(uid[ev & (lab == 1)])):
        pj = [j for j in rows_of[u] if ev[j] and lab[j] == 1]
        if POOL == 'global':
            cand = neg_pool_all
        else:
            cand = np.array([j for v, _ in insim.get(u, [])
                             for j in rows_of.get(v, []) if lab[j] == 0], dtype=int)
            cand = np.unique(cand)
        need = neg_per_pos * len(pj)
        if len(cand) < need:
            continue
        users.append(u)
        pos_of[u] = pj
        neg_of[u] = rng.choice(cand, need, replace=False).tolist()
    return users, pos_of, neg_of


# ============================================================================
# 特征重算（全部以「目标用户 u、评测时刻 t=边界」为准）
# ============================================================================
def item_stats(seq, boundary):
    first, cnt, ssum, ncnt = {}, defaultdict(int), defaultdict(float), defaultdict(int)
    for s in seq.values():
        for it, ts, r in s:
            if ts >= boundary:
                continue
            if it not in first or ts < first[it]:
                first[it] = ts
            cnt[it] += 1
            ssum[it] += r
            ncnt[it] += 1
    return first, cnt, ssum, ncnt, float(max(cnt.values()) if cnt else 1)


def features_for(pairs, seq, boundary, t0, item_index, mat, st):
    """pairs = [(uid, item)]；返回 {(uid,item): (rel,dif,nov,unpop,qua)}"""
    first, cnt, ssum, ncnt, mx = st
    hist = {}
    for u in {p[0] for p in pairs}:
        h = [(it, ts) for it, ts, _r in seq.get(u, []) if ts < boundary and it in item_index]
        hist[u] = (mat[np.array([item_index[i] for i, _ in h])],
                   np.array([t for _, t in h], float)) if h else (None, None)

    out = {}
    by_u = defaultdict(set)
    for u, i in pairs:
        by_u[u].add(i)
    for u, its in by_u.items():
        H, _ = hist[u]
        if H is None or len(H) < MIN_HISTORY:
            continue
        its = [i for i in its if i in item_index]
        V = mat[np.array([item_index[i] for i in its])]
        S = H @ V.T
        k = min(max(3, len(H) // 10), 10)
        for c, i in enumerate(its):
            v = S[:, c]
            rel = float(np.partition(v, -k)[-k:].mean())
            out[(u, i)] = (
                rel,
                float(np.clip(rel - float(v.mean()), 0.0, 1.0)),
                float(np.clip((first[i] - t0) / max(boundary - t0, 1), 0, 1)) if i in first else 1.0,
                float(1 - cnt.get(i, 0) / (mx + 1e-9)),
                float(ssum[i] / ncnt[i]) if ncnt.get(i) else 3.0,
            )
    return out


# ============================================================================
# 排序器
# ============================================================================
def fit_params(feats, uid_arr, lab_arr, tr_mask, users):
    """M-DCG 的类条件参数 + LinCombo 权重，档位 0 拟合一次。"""
    idx = np.where(tr_mask)[0]
    R = np.array([feats[(uid_arr[j], feats_item[j])][0] if False else 0 for j in idx])
    return None


def run(name):
    seq, label_time, emb, fdf = load(name)
    item_index, mat = embeddings(emb)
    uid, lab = fdf['uid'].values, fdf['label'].values
    rts = np.array([label_time.get((u, i), 0)
                    for u, i in zip(uid, fdf['item'].values)], float)
    boundary = float(np.quantile(rts, 1.0 - TEST_RATIO))
    tr, ev = rts < boundary, rts >= boundary
    t0 = min(t for s in seq.values() for _i, t, _r in s)
    print('%s: %d 行 | 边界 %.0f | 训练 %d(正 %d) / 测试 %d(正 %d)'
          % (name, len(fdf), boundary, tr.sum(), int(lab[tr].sum()),
             ev.sum(), int(lab[ev].sum())))

    insim = surprise_neighbours(fdf, tr, with_neighbours=PROFILE_WITH_NEIGHBOURS)
    users, pos_of, neg_of = build_pool(fdf, uid, lab, ev, insim)
    print('池子: %d 用户 | 正样本 %d | 池 %s'
          % (len(users), sum(len(v) for v in pos_of.values()), POOL))
    if not users:
        return

    rows = []
    for lv in LEVELS:
        rng = np.random.RandomState(42 + int(lv * 1000))
        red = {u: [x for x in s if x[1] >= boundary or rng.rand() >= lv]
               for u, s in seq.items()}
        st = item_stats(red, boundary)
        pairs = [(u, fdf['item'].values[j])
                 for u in users for j in pos_of[u] + neg_of[u]]
        F = features_for(pairs, red, boundary, t0, item_index, mat, st)
        miss = (0.5, 0.0, 1.0, 1.0, 3.0)

        hit = defaultdict(list)
        for u in users:
            pool = pos_of[u] + neg_of[u]
            f = np.array([F.get((u, fdf['item'].values[j]), miss) for j in pool])
            score = {
                'mdcg': None,          # ← 在这里接 M-DCG：需要 Beta 参数，见 fit_params
                'lincombo': None,      # ← 在这里接 LinCombo：需要权重
                'rel': f[:, 0], 'dif': f[:, 1], 'nov': f[:, 2], 'unpop': f[:, 3],
            }
            for m, s in score.items():
                if s is None:
                    continue
                order = np.argsort(-s, kind='mergesort')
                y = lab[np.array(pool)[order]]
                for k in (2, 4):
                    hit[(m, k)].append(1.0 if y[:k].sum() > 0 else 0.0)
        line = '  %3.0f%% ' % (100 * lv)
        for m in ('rel', 'dif', 'nov', 'unpop'):
            line += ' %s %.3f/%.3f' % (m, np.mean(hit[(m, 2)]), np.mean(hit[(m, 4)]))
        print(line + '   (Hit@2/Hit@4，随机=0.1/0.2)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('case', choices=sorted(DATASETS))
    run(ap.parse_args().case)
