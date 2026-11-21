# -*- coding: utf-8 -*-
"""RQ4：按比例删除训练窗交互，看各排序器的 Hit@2 / Hit@4 怎么变。

流程
----
1. 读交互流 + 因子表（只用它的 uid/item/label，所有特征重算）
2. 按标签时间 6:4 切训练窗 / 测试窗
3. 建池子：目标用户测试窗正样本 + 负样本（来源见 POOL）
4. 每档 r：从训练窗随机删 r% 交互 → 以「目标用户 u、时刻=边界」重算五因子 → 排序 → 数命中
5. 打印每档每个排序器的 Hit@2 / Hit@4

参数、池子来源、排序器都集中在文件上半部分，改那里就行。
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
import eval as E
import metrics as M

# ============================================================================
# 可调参数
# ============================================================================
DATA = r"C:\Users\Administrator\Desktop\study\论文复现\code"
DATASETS = {
    # 注意：S2018 的历史本应用 data/serendipity-sac2018/training.csv（1000 万条），
    #      现在读的是问卷文件（32 万条）。换的话改 seq= 即可（列名要自己映射）。
    'movies': dict(seq=DATA + "\\data\\SerenLens\\SerenLens_Movies.csv",
                   factors=DATA + "\\factors_AMZ_Movies.pkl",
                   emb=DATA + "\\emb_AMZ_Movies.pkl"),
    'books': dict(seq=DATA + "\\data\\SerenLens\\SerenLens_Books.csv",
                  factors=DATA + "\\factors_Seren_Books.pkl",
                  emb=DATA + "\\emb_Seren_Books.pkl"),
    's2018': dict(seq=DATA + "\\SerenLens_S2018.csv",
                  factors=DATA + "\\factors_SEREN_S2018.pkl",
                  emb=DATA + "\\emb_SEREN_S2018.pkl",
                  pairs=DATA + "\\S2018_survey_pairs.csv"),
}
LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)    # 删除比例
TEST_RATIO = 0.4                         # 测试窗占比（6:4）
NEG_PER_POS = 19                         # 每个正样本配几个负样本
POOL = 'global'                          # 'global' 全局随机负样本 / 'similar' 惊喜画像相似池
#  注意：'similar' 现在只覆盖 9 个用户（画像只对有训练窗正样本的人生效），要扩得自己改 build_pool
MIN_HISTORY = 5                          # 历史少于这么多条算不出 rel
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
    if cfg.get('pairs'):                       # S2018：只留问卷问过的行
        p = pd.read_csv(cfg['pairs'], dtype=str)
        keep = set(zip(p.iloc[:, 0].str.lower(), p.iloc[:, 1].str.lower()))
        fdf = fdf[[(nid(u), nid(i)) in keep
                   for u, i in zip(fdf['uid'], fdf['item'])]].reset_index(drop=True)
    fdf['uid'] = fdf['uid'].map(nid)
    fdf['item'] = fdf['item'].map(nid)
    seq = {nid(u): [(nid(i), int(t), float(r)) for i, t, r in v] for u, v in seq.items()}
    label_time = {(nid(u), nid(i)): int(t) for (u, i), t in label_time.items()}
    return seq, label_time, emb, fdf


# ============================================================================
# 池子
# ============================================================================
def build_pool(fdf, uid, lab, ev, insim):
    """正样本 = 该用户测试窗的正样本；负样本按 POOL 取。只依赖标签 → 各档恒定。"""
    rows_of = defaultdict(list)
    for j, u in enumerate(uid):
        rows_of[u].append(j)
    all_neg = np.array([j for j in range(len(fdf)) if lab[j] == 0], dtype=int)
    rng = np.random.RandomState(1234)
    users, pos_of, neg_of = [], {}, {}
    for u in sorted(set(uid[ev & (lab == 1)])):
        pj = [j for j in rows_of[u] if ev[j] and lab[j] == 1]
        if POOL == 'global':
            cand = all_neg
        else:
            cand = np.unique([j for v, _ in insim.get(u, [])
                              for j in rows_of.get(v, []) if lab[j] == 0])
        need = NEG_PER_POS * len(pj)
        if len(cand) < need:
            continue
        users.append(u)
        pos_of[u] = pj
        neg_of[u] = rng.choice(cand, need, replace=False).tolist()
    return users, pos_of, neg_of


# ============================================================================
# 特征：以「目标用户 u、时刻 = 边界」为准
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


def features(pairs, seq, boundary, t0, item_index, mat, st):
    """pairs=[(uid,item)] → {(uid,item): (rel,dif,nov,unpop,qua)}"""
    first, cnt, ssum, ncnt, mx = st
    out = {}
    by_u = defaultdict(set)
    for u, i in pairs:
        by_u[u].add(i)
    for u, its in by_u.items():
        h = [(i, t) for i, t, _r in seq.get(u, []) if t < boundary and i in item_index]
        if len(h) < MIN_HISTORY:
            continue
        H = mat[np.array([item_index[i] for i, _ in h])]
        its = [i for i in its if i in item_index]
        S = H @ mat[np.array([item_index[i] for i in its])].T
        k = min(max(3, len(h) // 10), 10)
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
# 主流程
# ============================================================================
def run(name):
    seq, label_time, emb, fdf = load(name)
    items = list(emb)
    mat = np.vstack([emb[i] for i in items])
    mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    item_index = {it: k for k, it in enumerate(items)}

    uid = fdf['uid'].values
    lab = fdf['label'].values
    rts = np.array([label_time.get((u, i), 0)
                    for u, i in zip(uid, fdf['item'].values)], float)
    boundary = float(np.quantile(rts, 1.0 - TEST_RATIO))
    tr, ev = rts < boundary, rts >= boundary
    t0 = min(t for s in seq.values() for _i, t, _r in s)
    print('%s: %d 行 | 边界 %.0f | 训练 %d(正 %d) / 测试 %d(正 %d)'
          % (name, len(fdf), boundary, tr.sum(), int(lab[tr].sum()),
             ev.sum(), int(lab[ev].sum())))

    # 稀缺参数与画像相似图由 eval 现成函数给（含两步构造，自己写容易反向）
    ctx = E._prepare_temporal(fdf, seq, label_time, E.SimEngine(emb), USE_COPULA,
                              MIN_HISTORY, TEST_RATIO, 1, None, False,
                              'surprise_pool', 'median', 5, tag='(rq4)',
                              topk=50, min_valid_users=1, min_neg_test=1,
                              quality_index=None, qua_mode='pit')
    PACC = ctx['scarcity_win']
    SIM = ctx['surprise_sim_win']

    users, pos_of, neg_of = build_pool(fdf, uid, lab, ev, SIM)
    print('池子: %d 用户 | 正样本 %d | 来源 %s'
          % (len(users), sum(len(v) for v in pos_of.values()), POOL))
    if not users:
        return

    PAR = None
    for lv in LEVELS:
        rng = np.random.RandomState(42 + int(lv * 1000))
        red = {u: [x for x in s if x[1] >= boundary or rng.rand() >= lv]
               for u, s in seq.items()}
        st = item_stats(red, boundary)
        pairs = [(u, fdf['item'].values[j])
                 for u in users for j in pos_of[u] + neg_of[u]]

        if PAR is None:      # 档位 0 拟合一次，之后冻结
            tr_pairs = [(uid[j], fdf['item'].values[j]) for j in np.where(tr)[0]]
            Ftr = features(tr_pairs, red, boundary, t0, item_index, mat, st)
            tr_idx = np.where(tr)[0]
            fr = pd.DataFrame({
                'uid': uid[tr_idx],
                'rel': [Ftr.get(k, (0.5,))[0] for k in tr_pairs],
                'dif': [Ftr.get(k, (0.5, 0.0))[1] for k in tr_pairs],
                'nov': [Ftr.get(k, (0.5, 0.0, 1.0))[2] for k in tr_pairs],
                'unpop': [Ftr.get(k, (0.5, 0.0, 1.0, 1.0))[3] for k in tr_pairs],
                'qua': [Ftr.get(k, (0.5, 0.0, 1.0, 1.0, 3.0))[4] for k in tr_pairs],
                'label': lab[tr_idx]})
            pos, neg = fr[fr.label == 1], fr[fr.label == 0]
            PG = M.fit_class_params(pos['rel'].values, pos['dif'].values, USE_COPULA)
            NG = M.fit_class_params(neg['rel'].values, neg['dif'].values, USE_COPULA)
            LCW = LogisticRegression(max_iter=2000, class_weight='balanced').fit(
                fr[['rel', 'dif', 'nov', 'unpop', 'qua']].values, fr['label'].values).coef_[0]
            PAR = (PG, NG, LCW)
            print('  拟合: Beta惊喜=%s 非惊喜=%s'
                  % (np.round(np.asarray(PG, float), 2), np.round(np.asarray(NG, float), 2)))
        PG, NG, LCW = PAR

        F = features(pairs, red, boundary, t0, item_index, mat, st)
        miss = (0.5, 0.0, 1.0, 1.0, 3.0)
        hit = defaultdict(list)
        for u in users:
            pool = pos_of[u] + neg_of[u]
            f = np.array([F.get((u, fdf['item'].values[j]), miss) for j in pool])
            score = {
                'mdcg': [M.f_sp_score(f[c, 0], f[c, 1], f[c, 2], f[c, 3], f[c, 4],
                                      PG, NG, use_copula=USE_COPULA,
                                      user_scarcity_param=PACC.get(u))
                         for c in range(len(pool))],
                'lincombo': f @ LCW[:5],
                'rel': f[:, 0], 'dif': f[:, 1], 'nov': f[:, 2], 'unpop': f[:, 3],
            }
            for m, s in score.items():
                _j = np.random.RandomState(7 + int(lv * 100)).rand(len(pool))
                order = np.argsort(-(np.asarray(s, float) + 1e-9 * _j), kind='mergesort')
                y = lab[np.array(pool)[order]]
                for k in (2, 4):
                    hit[(m, k)].append(1.0 if y[:k].sum() > 0 else 0.0)
        line = '  %3.0f%%' % (100 * lv)
        for m in ('mdcg', 'lincombo', 'rel', 'dif', 'nov', 'unpop'):
            line += '  %-8s %.3f/%.3f' % (m, np.mean(hit[(m, 2)]), np.mean(hit[(m, 4)]))
        print(line)
    print('  每格 = Hit@2/Hit@4；随机基准 = 0.100/0.200')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('case', choices=sorted(DATASETS))
    run(ap.parse_args().case)
