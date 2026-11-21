# -*- coding: utf-8 -*-
"""RQ4 重做（v4）：删训练窗交互 + 惊喜画像相似池 + 按目标用户重算全部因子。

口径（用户 2026-10-18 定）
------------------------
1. 评测时刻 t_u = 训练窗末尾（边界），只用边界之前的数据算特征，无未来信息。
2. 池子 = 惊喜画像相似池：正样本 = 该用户测试窗的正样本；
   负样本 = 他 top-50 相似用户（ctx['surprise_sim_win']）的测试窗 label==0 行，19x 抽样；
   池子在档位 0 建一次后冻结，五档同一批候选。
3. **五个因子全部按 (目标用户 u, 物品 i, t_u) 重算** —— 池里每个物品对每个用户
   都用目标用户自己的历史，不存在"别人的特征"。
4. 参数（M-DCG 的 Beta/copula、per-user 唤醒层、接受峰、LinCombo 权重）在档位 0
   拟合一次后冻结；每个档位按 r 从训练窗随机删交互，重算特征。
5. 严格 1:19：负样本不足的用户不进池，人数五档恒定。
"""
import os
import sys
import time
import pickle
import argparse
from collections import defaultdict
from bisect import bisect_left

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = r"C:\Users\Administrator\Desktop\study\论文复现\code"
sys.path.insert(0, ROOT)
import eval as E          # noqa: E402
import metrics as M       # noqa: E402

D = ROOT
CASES = {
    'movies': dict(serenlens=os.path.join(D, 'data/SerenLens/SerenLens_Movies.csv'),
                   factors=os.path.join(D, 'factors_AMZ_Movies.pkl'),
                   emb=os.path.join(D, 'emb_AMZ_Movies.pkl')),
    'books': dict(serenlens=os.path.join(D, 'data/SerenLens/SerenLens_Books.csv'),
                  factors=os.path.join(D, 'factors_Seren_Books.pkl'),
                  emb=os.path.join(D, 'emb_Seren_Books.pkl')),
    's2018': dict(serenlens=os.path.join(D, 'SerenLens_S2018.csv'),
                  factors=os.path.join(D, 'factors_SEREN_S2018.pkl'),
                  emb=os.path.join(D, 'emb_SEREN_S2018.pkl'),
                  drop_nonsurvey=True,
                  pairs=os.path.join(D, 'S2018_survey_pairs.csv')),
}
LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)
MIN_HISTORY, MIN_CENTER = 5, 5


def log(*a):
    print(*a, flush=True)


def nid(s):
    return str(s).strip().lower()


def load_case(name):
    cfg = CASES[name]
    seq, labels, label_time, _r = E.load_serenlens(cfg['serenlens'])
    emb = {nid(k): np.asarray(v, float) for k, v in pickle.load(open(cfg['emb'], 'rb')).items()}
    fdf = pd.read_pickle(cfg['factors'])
    if cfg.get('drop_nonsurvey'):
        bp = pd.read_csv(cfg['pairs'], dtype=str)
        keep = set(zip(bp.iloc[:, 0].astype(str).str.lower(),
                       bp.iloc[:, 1].astype(str).str.lower()))
        m = np.array([(str(u).lower(), str(i).lower()) in keep
                      for u, i in zip(fdf['uid'], fdf['item'])])
        fdf = fdf[m].reset_index(drop=True)
    fdf['uid'] = fdf['uid'].map(nid)
    fdf['item'] = fdf['item'].map(nid)
    seq = {nid(u): [(nid(i), int(t), float(r)) for i, t, r in v] for u, v in seq.items()}
    label_time = {(nid(u), nid(i)): int(t) for (u, i), t in label_time.items()}
    return cfg, seq, label_time, emb, fdf


def reduce_seq(seq, boundary, frac, rng):
    if frac <= 0:
        return seq
    out = {}
    for u, s in seq.items():
        out[u] = [x for x in s if x[1] >= boundary or rng.rand() >= frac]
    return out


def item_stats(seq, boundary, t0):
    """边界前的物品级统计: 首次出现、交互数、评分和/个数。"""
    first, cnt, rsum, rcnt = {}, defaultdict(int), defaultdict(float), defaultdict(int)
    for s in seq.values():
        for it, ts, r in s:
            if ts >= boundary:
                continue
            if it not in first or ts < first[it]:
                first[it] = ts
            cnt[it] += 1
            rsum[it] += r
            rcnt[it] += 1
    mx = max(cnt.values()) if cnt else 1
    return first, cnt, rsum, rcnt, float(mx)


def user_hist(seq, boundary, item_index, mat):
    """每个用户在边界前的历史: (嵌入矩阵, 时间数组)。"""
    out = {}
    for u, s in seq.items():
        hs = [(it, ts) for it, ts, _r in s if ts < boundary and it in item_index]
        if not hs:
            out[u] = (None, None)
            continue
        idx = np.array([item_index[it] for it, _ts in hs])
        out[u] = (mat[idx], np.array([ts for _it, ts in hs], float))
    return out


def factor_for_pairs(users_items, hist, item_index, mat, boundary, t0, first, cnt, rsum, rcnt, mx):
    """对 (u, i) 配对算五个因子。返回 dict: (u,i) -> (rel,dif,nov,unpop,qua)"""
    out = {}
    by_u = defaultdict(list)
    for u, i in users_items:
        by_u[u].append(i)
    for u, items in by_u.items():
        H, Hts = hist.get(u, (None, None))
        if H is None:
            continue
        seen = set()
        uniq = [i for i in items if not (i in seen or seen.add(i))]
        ok = [i for i in uniq if i in item_index]
        if not ok:
            continue
        V = mat[np.array([item_index[i] for i in ok])]
        S = H @ V.T
        n_h = len(Hts)
        kk = min(max(3, n_h // 10), 10)
        for c, i in enumerate(ok):
            v = S[:, c]
            rel = float(np.partition(v, -kk)[-kk:].mean())
            dif = float(np.clip(rel - float(v.mean()), 0.0, 1.0))
            f = first.get(i)
            nov = float(np.clip((f - t0) / max(boundary - t0, 1.0), 0.0, 1.0)) if f else 1.0
            unpop = float(1.0 - cnt.get(i, 0) / (mx + 1e-9))
            qua = float(rsum[i] / rcnt[i]) if rcnt.get(i, 0) else 3.0
            out[(u, i)] = (rel, dif, nov, unpop, qua)
    return out


def run(name, levels=LEVELS, seed=42, test_ratio=0.4, use_copula=True, neg_per_pos=19):
    t_start = time.time()
    cfg, seq, label_time, emb, fdf = load_case(name)
    items = list(emb)
    mat = np.vstack([emb[i] for i in items])
    mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
    item_index = {it: k for k, it in enumerate(items)}
    t0 = min(ts for s in seq.values() for _i, ts, _r in s)

    rts = np.array([label_time.get((u, i), 0) for u, i in zip(fdf['uid'], fdf['item'])], float)
    b = float(np.quantile(rts, 1.0 - test_ratio))
    tr, ev = rts < b, rts >= b
    uid = fdf['uid'].values
    lab = fdf['label'].values
    log('>>> %s: seq=%d 用户 | 因子表=%d 行 | 边界 %.0f | 训练 %d(正 %d) / 测试 %d(正 %d)'
        % (name, len(seq), len(fdf), b, int(tr.sum()), int((lab[tr] == 1).sum()),
           int(ev.sum()), int((lab[ev] == 1).sum())))

    sim_engine = E.SimEngine(emb)
    ctx = E._prepare_temporal(
        fdf, seq, label_time, sim_engine, use_copula, MIN_HISTORY, test_ratio,
        1, None, False, 'surprise_pool', 'median', 5,
        tag='(hist-robust)', topk=50, min_valid_users=1, min_neg_test=1,
        quality_index=None, qua_mode='pit')
    if ctx is None:
        log('!! _prepare_temporal 返回 None'); return []
    SIM = ctx['surprise_sim_win']

    # ---- 相似池（只看标签，档位无关，因此天然五档固定）----
    rows_by_uid = defaultdict(list)
    for j, u in enumerate(uid):
        rows_by_uid[u].append(j)
    rng = np.random.RandomState(1234)
    users, pos_by_u, neg_by_u = [], {}, {}
    for u in sorted(set(uid[ev & (lab == 1)])):
        pj = [j for j in rows_by_uid[u] if ev[j] and lab[j] == 1]
        cand = []
        for v, _s in SIM.get(u, []):
            # 邻居的负样本不限窗口：相似池本来就不要求目标用户自己标注得够
            cand += [j for j in rows_by_uid.get(v, []) if lab[j] == 0]
        cand = np.array(sorted(set(cand)), dtype=int)
        need = neg_per_pos * len(pj)
        if len(cand) < need:
            continue
        users.append(u)
        pos_by_u[u] = pj
        neg_by_u[u] = rng.choice(cand, need, replace=False).tolist()
    log('    [相似池] 用户 %d | 正样本 %d | 每人负样本 19x'
        % (len(users), sum(len(v) for v in pos_by_u.values())))
    if not users:
        return []

    out, PAR = [], None
    for lv in levels:
        t1 = time.time()
        r = np.random.RandomState(seed + int(round(lv * 1000)))
        red = reduce_seq(seq, b, lv, r)
        first, cnt, rsum, rcnt, mx = item_stats(red, b, t0)
        hist = user_hist(red, b, item_index, mat)

        # 每个用户自己池子的 (u, item) 配对
        pairs = []
        for u in users:
            for j in pos_by_u[u] + neg_by_u[u]:
                pairs.append((u, fdf['item'].values[j]))
        t2 = time.time()
        F = factor_for_pairs(pairs, hist, item_index, mat, b, t0, first, cnt, rsum, rcnt, mx)
        log('    特征重算: %d 配对, 覆盖 %d, %.0fs' % (len(pairs), len(F), time.time() - t2))
        miss = (0.5, 0.0, 1.0, 1.0, 3.0)

        def feat(u, j):
            return F.get((u, fdf['item'].values[j]), miss)

        if lv == levels[0]:
            # 参数在档位 0 用训练行拟合一次（训练行仍走全表口径，供 Beta/唤醒层）
            rel_tr = np.array([feat(uid[j], j)[0] for j in np.where(tr)[0]])
            dif_tr = np.array([feat(uid[j], j)[1] for j in np.where(tr)[0]])
            y_tr = lab[tr]
            PG = M.fit_class_params(rel_tr[y_tr == 1], dif_tr[y_tr == 1], use_copula)
            NG = M.fit_class_params(rel_tr[y_tr == 0], dif_tr[y_tr == 0], use_copula)
            PACC = ctx['scarcity_win']
            _fr = pd.DataFrame({'uid': uid[tr], 'item': fdf['item'].values[tr],
                                'rel': rel_tr, 'dif': dif_tr, 'label': y_tr})
            PER, _n1, _n2 = E._build_per_user_wakeup(
                _fr, set(users), SIM, use_copula, PG, NG)
            log('    per-user 唤醒: 有画像 %d / 邻居够 %d' % (_n1, _n2))
            X = np.column_stack([rel_tr, dif_tr,
                                 np.array([feat(uid[j], j)[2] for j in np.where(tr)[0]]),
                                 np.array([feat(uid[j], j)[3] for j in np.where(tr)[0]]),
                                 np.array([feat(uid[j], j)[4] for j in np.where(tr)[0]])])
            LCW = LogisticRegression(max_iter=2000, class_weight='balanced').fit(X, y_tr).coef_[0]
            log('    [定参] Beta 惊喜=%s 非惊喜=%s | LinCombo w=%s'
                % (np.round(np.asarray(PG, float), 3), np.round(np.asarray(NG, float), 3),
                   np.round(LCW, 3)))
            PAR = (PG, NG, PER, PACC, LCW)
        PG, NG, PER, PACC, LCW = PAR

        METRICS = ('mdcg', 'lincombo', 'rel', 'dif', 'nov', 'unpop')
        hit = {m: {2: [], 4: []} for m in METRICS}
        for u in users:
            pool = pos_by_u[u] + neg_by_u[u]
            npos = len(pos_by_u[u])
            fj = [feat(u, j) for j in pool]
            rel = np.array([x[0] for x in fj])
            dif = np.array([x[1] for x in fj])
            nov = np.array([x[2] for x in fj])
            unp = np.array([x[3] for x in fj])
            qa = np.array([x[4] for x in fj])
            sc = {
                'mdcg': np.array([M.f_sp_score(rel[c], dif[c], nov[c], unp[c], qa[c],
                                               PG, NG, use_copula=use_copula,
                                               user_scarcity_param=PACC.get(u))
                                  for c in range(len(pool))]),
                'lincombo': LCW[0] * rel + LCW[1] * dif + LCW[2] * nov + LCW[3] * unp + LCW[4] * qa,
                'rel': rel, 'dif': dif, 'nov': nov, 'unpop': unp,
            }
            jit = np.random.RandomState(777 + int(round(lv * 100))).rand(len(pool))
            for m in METRICS:
                order = np.argsort(-(sc[m] + 1e-9 * jit), kind='mergesort')
                ranked = [lab[pool[i]] for i in order]
                for k in (2, 4):
                    got = int(sum(ranked[:k]))
                    hit[m][k].append(1.0 if got > 0 else 0.0)
        res = {m: (float(np.mean(hit[m][2])), float(np.mean(hit[m][4]))) for m in METRICS}
        out.append((lv, len(users), res))
        for m in METRICS:
            log('  [%3.0f%%] %-8s Hit@2=%.4f | Hit@4=%.4f' % (100 * lv, m, res[m][0], res[m][1]))
        log('          %d 用户 | %.0fs' % (len(users), time.time() - t1))

    log('\n=== %s 汇总（相似池；五因子按目标用户重算；随机 Hit@2=0.1）===' % name)
    log('   r%%    用户  ' + '  '.join('%-11s' % m for m in METRICS))
    for lv, nn, res in out:
        log('  %3.0f  %5d  ' % (100 * lv, nn)
            + '  '.join('%.3f/%.3f' % (res[m][0], res[m][1]) for m in METRICS))
    log('  （每格 = Hit@2 / Hit@4）')
    log('  总用时 %.0fs' % (time.time() - t_start))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('case', choices=sorted(CASES))
    ap.add_argument('--levels', default=None)
    ap.add_argument('--neg_per_pos', type=int, default=19)
    a = ap.parse_args()
    lv = tuple(float(x) for x in a.levels.split(',')) if a.levels else LEVELS
    _lg = os.environ.get('MDCG_RUN_LOG')
    if _lg:
        _f = open(_lg, 'a', encoding='utf-8', buffering=1)
        sys.stdout = _f
        sys.stderr = _f
        print('\n===== %s =====' % a.case, flush=True)
    run(a.case, levels=lv, neg_per_pos=a.neg_per_pos)
