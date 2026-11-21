# -*- coding: utf-8 -*-
"""RQ4 重做：删除训练窗交互（历史缩水）后的鲁棒性。只读，不改任何 pkl。

三条硬规则（用户 2026-10-18 定）
--------------------------------
1. **清 0 到底**：r = 0/25/50/75/100%，100% 就是训练窗交互全删，不留底。
2. **参数不按档重拟合**：一律用标准训练窗拟合出的那套 ——
   ctx['pp_global'] / ctx['np_global']（启动时日志打印的"惊喜类/非惊喜类"五元组）、
   ctx['per_user_pp_np']、ctx['scarcity_win']。五档之间只有特征在变。
3. **池子固定**：在档位 0 时就定死 —— 测试窗里有正样本的用户、这些正样本、
   以及按 1:19 全局随机抽到的负样本。之后五档共用同一批候选，用户数不变。
   某档里某行特征算不出来（历史被删空）→ 按"无信息"补成训练窗均值，不踢出池子。

评测集只取测试窗（8:2 → 6:4），训练行永不参与评测。
"""
import os
import sys
import time
import pickle
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = r"C:\Users\Administrator\Desktop\study\论文复现\code"
sys.path.insert(0, ROOT)
import eval as E          # noqa: E402
import metrics as M       # noqa: E402

D = ROOT
CASES = {
    'movies': dict(model='AMZ', data='Movies',
                   serenlens=os.path.join(D, 'data/SerenLens/SerenLens_Movies.csv'),
                   factors=os.path.join(D, 'factors_AMZ_Movies.pkl'),
                   emb=os.path.join(D, 'emb_AMZ_Movies.pkl')),
    'books': dict(model='Seren', data='Books',
                  serenlens=os.path.join(D, 'data/SerenLens/SerenLens_Books.csv'),
                  factors=os.path.join(D, 'factors_Seren_Books.pkl'),
                  emb=os.path.join(D, 'emb_Seren_Books.pkl')),
    's2018': dict(model='SEREN', data='S2018',
                  serenlens=os.path.join(D, 'SerenLens_S2018.csv'),
                  factors=os.path.join(D, 'factors_SEREN_S2018.pkl'),
                  emb=os.path.join(D, 'emb_SEREN_S2018.pkl'),
                  drop_nonsurvey=True, pairs=os.path.join(D, 'S2018_survey_pairs.csv')),
}
LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)
MIN_HISTORY, MIN_CENTER = 5, 5


def log(*a):
    print(*a, flush=True)


def nid(s):
    return str(s).strip().lower()


def load_case(name):
    cfg = CASES[name]
    seq, labels, label_time, rating = E.load_serenlens(cfg['serenlens'])
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


def vec_index(emb):
    items = list(emb)
    mat = np.vstack([emb[i] for i in items])
    mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
    return {it: k for k, it in enumerate(items)}, mat


def reduce_seq(seq, boundary, frac, rng):
    """按比例删训练窗(ts<=boundary)交互；frac=1.0 即清 0，不留底。"""
    if frac <= 0:
        return seq
    out = {}
    for u, s in seq.items():
        out[u] = [x for x in s if x[1] > boundary or rng.rand() >= frac]
    return out


def raw_factors(seq, rowdf, label_time, item_index, mat, t0):
    n = len(rowdf)
    rel = np.full(n, np.nan)
    dif = np.full(n, np.nan)
    first_ts, ts_by_item = {}, defaultdict(list)
    for s in seq.values():
        for it, ts, _r in s:
            if it not in first_ts or ts < first_ts[it]:
                first_ts[it] = ts
            ts_by_item[it].append(ts)
    for it in ts_by_item:
        ts_by_item[it].sort()

    uids = rowdf['uid'].values
    items = rowdf['item'].values
    rts = np.array([label_time.get((u, i), 0) for u, i in zip(uids, items)], float)
    by_user = defaultdict(list)
    for j, u in enumerate(uids):
        by_user[u].append(j)

    for u, rows in by_user.items():
        s = seq.get(u, [])
        hist = [(it, ts) for it, ts, _r in s if it in item_index]
        if not hist:
            continue
        hts = np.array([ts for _it, ts in hist], float)
        H = mat[np.array([item_index[it] for it, _ts in hist])]
        rows = [j for j in rows if items[j] in item_index and rts[j] > 0]
        if not rows:
            continue
        T = mat[np.array([item_index[items[j]] for j in rows])]
        S = H @ T.T
        for c, j in enumerate(rows):
            m = hts < rts[j]
            k_ = int(m.sum())
            if k_ < MIN_HISTORY:
                continue
            v = S[m, c]
            kk = min(max(3, k_ // 10), 10)
            top = np.partition(v, -kk)[-kk:]
            rel[j] = float(top.mean())
            dif[j] = float(np.clip(rel[j] - float(v.mean()), 0.0, 1.0))

    nov = np.full(n, np.nan)
    for j in range(n):
        f = first_ts.get(items[j])
        if f is not None and rts[j] > t0:
            nov[j] = float(np.clip((f - t0) / max(rts[j] - t0, 1.0), 0.0, 1.0))
    cnt = np.zeros(n)
    for j in range(n):
        arr = ts_by_item.get(items[j])
        if arr:
            cnt[j] = float(np.searchsorted(arr, rts[j], side='left'))
    mx = float(cnt.max()) if n and cnt.max() > 0 else 1.0
    unp = np.clip(1.0 - cnt / (mx + 1e-9), 0.0, 1.0)
    return rel, dif, nov, unp


def ed(rel, dif, nov, unp, rowdf, label_time, tr):
    n = len(rel)
    uids = rowdf['uid'].values
    items = rowdf['item'].values
    lab = rowdf['label'].values
    rts = np.array([label_time.get((u, i), 0) for u, i in zip(uids, items)], float)
    order = np.argsort(rts, kind='mergesort')

    def global_upto(v):
        g = np.full(n, np.nan)
        ss, cc = 0.0, 0.0
        for j in order:
            g[j] = ss / cc if cc > 0 else np.nan
            if lab[j] == 0 and np.isfinite(v[j]):
                ss += v[j]
                cc += 1
        return g

    def center(v):
        g = global_upto(v)
        out = np.full(n, np.nan)
        fb = float(np.nanmean(v[tr])) if np.isfinite(v[tr]).any() else 0.5
        by_user = defaultdict(list)
        for j, u in enumerate(uids):
            by_user[u].append(j)
        for u, rows in by_user.items():
            rows.sort(key=lambda j: rts[j])
            ss, cc = 0.0, 0.0
            for j in rows:
                if np.isfinite(v[j]):
                    c = (ss / cc) if cc >= MIN_CENTER else g[j]
                    if not np.isfinite(c):
                        c = fb
                    out[j] = v[j] - c
                if lab[j] == 0 and np.isfinite(v[j]):
                    ss += v[j]
                    cc += 1
        return out

    def qmap(v):
        trv = v[tr][np.isfinite(v[tr])]
        if len(trv) < 10:
            return np.full(n, np.nan)
        ref = E._fit_quantile_ref(trv)
        out = np.full(n, np.nan)
        ok = np.isfinite(v)
        if ok.any():
            out[ok] = E._quantile_transform(v[ok], ref)
        return out

    return qmap(center(rel)), qmap(center(dif)), qmap(nov), qmap(unp)


def run(name, levels=LEVELS, seed=42, test_ratio=0.4, use_copula=True, neg_per_pos=19):
    t_start = time.time()
    cfg, seq, label_time, emb, fdf = load_case(name)
    item_index, mat = vec_index(emb)
    t0 = min(ts for s in seq.values() for _i, ts, _r in s)
    rts = np.array([label_time.get((u, i), 0) for u, i in zip(fdf['uid'], fdf['item'])], float)
    b = float(np.quantile(rts, 1.0 - test_ratio))
    tr = rts < b
    ev = ~tr
    label = fdf['label'].values
    uid = fdf['uid'].values
    qua = fdf['qua'].values.astype(float)
    log('>>> %s: seq=%d 用户, 因子表=%d 行 | 边界 %.0f | 训练 %d(正 %d) / 测试 %d(正 %d)'
        % (name, len(seq), len(fdf), b, int(tr.sum()), int((label[tr] == 1).sum()),
           int(ev.sum()), int((label[ev] == 1).sum())))

    sim_engine = E.SimEngine(emb)
    ctx = E._prepare_temporal(
        fdf, seq, label_time, sim_engine, use_copula, MIN_HISTORY, test_ratio,
        1, None, False, 'surprise_pool', 'median', 5,
        tag='(hist-robust)', topk=50, min_valid_users=1, min_neg_test=1,
        quality_index=None, qua_mode='pit')
    if ctx is None:
        log('!! _prepare_temporal 返回 None'); return []
    PG, NG = ctx['pp_global'], ctx['np_global']
    PER = ctx['per_user_pp_np']
    PACC = ctx['scarcity_win']
    log('    固定参数：全局 Beta 惊喜类=%s | 非惊喜类=%s'
        % (np.round(np.asarray(PG, float), 3), np.round(np.asarray(NG, float), 3)))
    log('    per-user 覆盖 %d 用户 | 稀缺参数 %d 用户' % (len(PER), len(PACC)))

    # qua 必须用【因果口径】：截至该行时刻的物品平均分（不含本人评分）。
    # 因子表里的 qua 就是该用户自己的评分（与本数据集标签相关系数 1.0），直接用等于泄漏标签。
    _qm = {}
    for _df in (ctx['train_df'], ctx['test_df']):
        _qm.update(zip(zip(_df['uid'].values, _df['item'].values),
                       _df['qua'].values.astype(float)))
    _miss = np.nanmean(list(_qm.values())) if _qm else 0.5
    qua = np.array([_qm.get((u, i), _miss) for u, i in zip(uid, fdf['item'].values)], float)
    log('    qua 改用因果口径(物品均分): 覆盖 %d/%d 行, 本身评分口径已弃用'
        % (len(_qm), len(fdf)))

    out = []
    LCW = None           # LinCombo 权重（每档重拟合）
    pool = None          # (users, pos_idx_by_user, neg_idx_by_user) 在档位 0 定死
    for lv in levels:
        t1 = time.time()
        rng = np.random.RandomState(seed + int(round(lv * 1000)))
        red = reduce_seq(seq, b, lv, rng)
        n_kept = sum(len(v) for v in red.values())
        rel, dif, nov, unp = raw_factors(red, fdf, label_time, item_index, mat, t0)
        rq, dq, nq, uq = ed(rel, dif, nov, unp, fdf, label_time, tr)

        # 无信息 → 补训练窗均值（E+D 空间里 ≈0.5），绝不因 NaN 踢行
        def fill(v):
            m = float(np.nanmean(v[tr])) if np.isfinite(v[tr]).any() else 0.5
            out_ = v.copy()
            out_[~np.isfinite(out_)] = m
            return out_, m
        rq, m_rel = fill(rq)
        dq, m_dif = fill(dq)
        nq, m_nov = fill(nq)
        uq, m_unp = fill(uq)
        # 基线（单因子 / 线性组合）一律用【原始因子定义】，不享受 E+D
        rr, _mrr = fill(rel)
        rd, _mrd = fill(dif)
        rn, _mrn = fill(nov)
        ru, _mru = fill(unp)

        # 档位 0：用本脚本的特征拟合一次参数，之后五档共用（不重拟合）
        if lv == levels[0]:
            _fr = pd.DataFrame({'uid': uid[tr], 'item': fdf['item'].values[tr],
                                'rel': rq[tr], 'dif': dq[tr], 'label': label[tr]})
            _p = _fr[_fr['label'] == 1]
            _n = _fr[_fr['label'] == 0]
            PG = M.fit_class_params(_p['rel'].values, _p['dif'].values, use_copula)
            NG = M.fit_class_params(_n['rel'].values, _n['dif'].values, use_copula)
            PER, _np_, _ne = E._build_per_user_wakeup(
                _fr, set(uid[ev]), ctx['surprise_sim_win'], use_copula, PG, NG)
            log('    [定参] 全局 Beta 惊喜类=%s 非惊喜类=%s | per-user %d'
                % (np.round(np.asarray(PG, float), 3), np.round(np.asarray(NG, float), 3),
                   len(PER)))

        # LinCombo：每档在【剩余训练集】上重拟合一个【全局】线性模型（不分用户）。
        # 100% 档训练行特征全被填补成常数，逻辑回归拟合不出东西 —— 沿用上一档权重并标注。
        _X = np.column_stack([rr[tr], rd[tr], rn[tr], ru[tr], qua[tr]])
        _y = label[tr]
        _ok = np.isfinite(_X).all(axis=1) & ((_y == 0) | (_y == 1))
        _deg = (not _ok.any()) or any(len(np.unique(_X[_ok, c])) < 2
                                      for c in range(_X.shape[1]))
        if _deg:
            if LCW is None:
                LCW = np.zeros(_X.shape[1])
            log('    [LinCombo] 训练集退化(%d 行可用)，沿用上一档权重' % int(_ok.sum()))
        else:
            LCW = LogisticRegression(max_iter=2000, class_weight='balanced').fit(
                _X[_ok], _y[_ok]).coef_[0]
            log('    [LinCombo] 用 %d 行剩余训练集重拟合: w=%s'
                % (int(_ok.sum()), np.round(LCW, 3)))

        # 档位 0 定池子
        if lv == levels[0]:
            # 池子只看标签：测试窗里有正样本的用户全部进池，
            # 负样本也从测试窗 label==0 的行里抽 —— 特征是否可算不影响成员。
            valid = np.where(ev)[0]
            pos_by_u = defaultdict(list)
            for j in valid:
                if label[j] == 1:
                    pos_by_u[uid[j]].append(j)
            neg_pool = np.array([j for j in valid if label[j] == 0], dtype=int)
            _rng = np.random.RandomState(1234)
            users, negs = [], {}
            for u in sorted(pos_by_u):
                need = neg_per_pos * len(pos_by_u[u])
                if len(neg_pool) < need:
                    continue
                users.append(u)
                negs[u] = _rng.choice(neg_pool, need, replace=False)
            pool = (users, pos_by_u, negs)
            log('    [定池] 用户 %d | 正样本 %d | 每用户负样本 19x | 全局负样本池 %d'
                % (len(users), sum(len(pos_by_u[u]) for u in users), len(neg_pool)))

        users, pos_by_u, negs = pool
        METRICS = ('mdcg', 'lincombo', 'rel', 'dif', 'nov', 'unpop')
        hit = {m: {2: [], 4: []} for m in METRICS}
        lift = {m: {2: [], 4: []} for m in METRICS}
        _rng_tie = np.random.RandomState(777 + int(round(lv * 100)))
        for u in users:
            pj = pos_by_u[u] + [int(x) for x in negs[u]]
            npos = len(pos_by_u[u])
            pp_u, np_u = PER.get(u, (PG, NG))
            sc = {
                'mdcg': np.array([M.f_sp_score(rq[j], dq[j], nq[j], uq[j], qua[j],
                                               pp_u, np_u, use_copula=use_copula,
                                               user_scarcity_param=PACC.get(u))
                                  for j in pj]),
                'rel': rr[pj],
                'dif': rd[pj],
                'nov': rn[pj],
                'unpop': ru[pj],
            }
            sc['lincombo'] = (LCW[0] * rr[pj] + LCW[1] * rd[pj] + LCW[2] * rn[pj]
                              + LCW[3] * ru[pj] + LCW[4] * qua[pj])
            jit = _rng_tie.rand(len(pj))
            for m in METRICS:
                order = np.argsort(-(sc[m] + 1e-9 * jit), kind='mergesort')
                ranked = [pj[i] for i in order]
                for k in (2, 4):
                    got = sum(1 for j in ranked[:k] if label[j] == 1)
                    hit[m][k].append(1.0 if got > 0 else 0.0)
                    lift[m][k].append(got / (k * npos / len(pj)))
        res = {m: (float(np.mean(hit[m][2])), float(np.mean(lift[m][2])),
                   float(np.mean(hit[m][4])), float(np.mean(lift[m][4])))
               for m in METRICS}
        out.append((lv, n_kept, len(users), res))
        for m in METRICS:
            h2, l2, h4, l4 = res[m]
            log('  [%3.0f%%] %-8s Hit@2=%.4f lift=%.2f | Hit@4=%.4f lift=%.2f'
                % (100 * lv, m, h2, l2, h4, l4))
        log('          保留交互 %d | 用户 %d | %.0fs' % (n_kept, len(users), time.time() - t1))

    log('\n=== %s 汇总（删训练窗交互；池子与参数固定）===' % name)
    log('   r%%   保留交互   用户   ' + '   '.join('%-9s' % m for m in METRICS))
    for lv, nk, nn, res in out:
        log('  %3.0f  %8d  %5d   ' % (100 * lv, nk, nn)
            + '   '.join('%.3f/%.2f' % (res[m][0], res[m][1]) for m in METRICS))
    log('  （每格 = Hit@2 / lift@2）')
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
