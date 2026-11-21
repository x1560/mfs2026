# -*- coding: utf-8 -*-
"""αβ 的 γ 项: 池内物品有多少落在目标用户历史没覆盖的类别(γ=0)?"""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, 'C:/Users/Administrator/Desktop/study/论文复现/code')
os.chdir('C:/Users/Administrator/Desktop/study/论文复现/code')
import numpy as np
from collections import defaultdict
import eval as E, metrics as M

K = 10; SEED = 42; TEST_RATIO = 0.2
user_seq, labels, label_time, item_rating = E.load_serenlens('./data/SerenLens/SerenLens_Movies.csv')
fdf = E.pd.read_pickle('factors_SEREN_Movies.pkl')
item_cate, item_pop = E.load_item_meta('cate_SEREN_Movies.pkl', 'pop_SEREN_Movies.pkl')
fdf['nov_q'] = M.quantile_normalize(fdf['novelty'].values)
fdf['unpop_q'] = M.quantile_normalize(fdf['unpopularity'].values)
fdf['rare'] = 0.5 * (fdf['nov_q'] + fdf['unpop_q'])
df = fdf.copy(); df['ts'] = [label_time.get((u, i)) for u, i in zip(df['uid'], df['item'])]
df = df.dropna(subset=['ts']).sort_values('ts').reset_index(drop=True)
all_ts = df['ts'].values
boundary = all_ts[int(len(all_ts) * (1.0 - TEST_RATIO))]
train = df[df['ts'] <= boundary]
valid = sorted(set(train['uid']))
# 惊喜画像(用训练窗正样本)
pd_ = train[train['label'] == 1]
pu, pr = [], []
for uid, g in pd_.groupby('uid'):
    pu.append(uid); pr.append([g[c].mean() for c in E.PROFILE_COLS])
pr = np.array(pr); pr = pr / (np.linalg.norm(pr, axis=1, keepdims=True) + 1e-9)
sm = pr @ pr.T; ss = {}
for i, uid in enumerate(pu):
    row = sm[i].copy(); row[i] = -1.0
    top = np.argpartition(row, -E.PROFILE_TOPK)[-E.PROFILE_TOPK:]
    top = top[np.argsort(row[top])[::-1]]
    ss[uid] = [(pu[j], float(row[j])) for j in top if row[j] >= 0]
ibu = defaultdict(list)
for _, r in train.iterrows():
    ibu[r['uid']].append((r['item'], r['rel'], r['dif'], r['nov_q'], r['unpop_q'],
                          r['qua'], r['label'], r['uid']))
ui = {u: set(it for it, ts, rr in seq) for u, seq in user_seq.items()}
# 用户历史(训练窗, 排除池内物品) -> γ
hist = defaultdict(list)
for u, seq in user_seq.items():
    for it, ts, r in seq:
        if ts <= boundary and item_cate.get(it) is not None:
            hist[u].append((it, float(r)))
rng = np.random.RandomState(SEED)
n_pool = 0; z_pos = z_neg = n_pos = n_neg = 0; users = 0
for _ in range(400):
    u = valid[rng.randint(len(valid))]
    if u not in ss or u not in ibu or u not in ui:
        continue
    pools = E.build_sim_user_pools(u, ss, ibu, ui, K, pool_dedup='none', pos_ratio_thr=0.5)
    if pools is None or pools[0] is None:
        continue
    ppos, nneg = pools
    npos_k = max(1, int(round(K * 0.2))); pos_pool = ppos[:npos_k]
    pool = pos_pool + nneg[:max(0, K - len(pos_pool))]
    pset = set(x[0] for x in pool)
    hc = [item_cate[it] for it, r in hist[u] if it not in pset and item_cate.get(it) is not None]
    hr = [r for it, r in hist[u] if it not in pset and item_cate.get(it) is not None]
    users += 1
    for x in pool:
        c = item_cate.get(x[0])
        if c is None: continue
        g = M.compute_gamma(hc, hr, c, 5.0)
        n_pool += 1
        if x[6] == 1:
            n_pos += 1; z_pos += (g <= 0)
        else:
            n_neg += 1; z_neg += (g <= 0)
print('=== Movies 池内物品的 γ 覆盖 ({} 用户 / {} 物品) ==='.format(users, n_pool))
print('  正样本: n={}  其中 γ=0 占 {:.1%}'.format(n_pos, z_pos / max(n_pos, 1)))
print('  负样本: n={}  其中 γ=0 占 {:.1%}'.format(n_neg, z_neg / max(n_neg, 1)))
print('  (γ=0 意味着 αβ 的 gain 恒为 0, 该物品只能排到末尾)')

# ---- 追加: 纯按 qua 排序 vs αβ 的剖面 ----
import math
hits2 = hits4 = 0; aucs = []; ranks = []; nb = 0
for _ in range(400):
    u = valid[rng.randint(len(valid))]
    if u not in ss or u not in ibu or u not in ui:
        continue
    pools = E.build_sim_user_pools(u, ss, ibu, ui, K, pool_dedup='none', pos_ratio_thr=0.5)
    if pools is None or pools[0] is None:
        continue
    ppos, nneg = pools
    npos_k = max(1, int(round(K * 0.2))); pos_pool = ppos[:npos_k]
    pool = pos_pool + nneg[:max(0, K - len(pos_pool))]
    pos_items = set(x[0] for x in pos_pool)
    order = sorted(range(len(pool)), key=lambda i: -pool[i][5])   # 按 qua 降序
    ranked = [pool[i][0] for i in order]
    labels = [pool[i][6] for i in order]
    nb += 1
    hits2 += 1.0 if any(it in pos_items for it in ranked[:2]) else 0.0
    hits4 += 1.0 if any(it in pos_items for it in ranked[:4]) else 0.0
    try:
        from sklearn.metrics import roc_auc_score
        if len(set(labels)) > 1:
            aucs.append(roc_auc_score(labels, np.arange(len(labels))[::-1].astype(float)))
        pr = [i for i, y in enumerate(labels) if y == 1]
        if pr: ranks.append(float(np.mean(pr)))
    except Exception:
        pass
print()
print('=== 纯按 qua 排序 ({} 池) ==='.format(nb))
print('  Hit@2={:.4f}  Hit@4={:.4f}  OrderAUC={:.4f}  正样本位次={:.4f}'.format(
    hits2 / nb, hits4 / nb, float(np.mean(aucs)), float(np.mean(ranks))))

# ---- 追加2: γ 与 p_a 各自的判别力 ----
from sklearn.metrics import roc_auc_score
GY, PA, YY = [], [], []
for _ in range(400):
    u = valid[rng.randint(len(valid))]
    if u not in ss or u not in ibu or u not in ui:
        continue
    pools = E.build_sim_user_pools(u, ss, ibu, ui, K, pool_dedup='none', pos_ratio_thr=0.5)
    if pools is None or pools[0] is None:
        continue
    ppos, nneg = pools
    npos_k = max(1, int(round(K * 0.2))); pos_pool = ppos[:npos_k]
    pool = pos_pool + nneg[:max(0, K - len(pos_pool))]
    pset = set(x[0] for x in pool)
    hc = [item_cate[it] for it, r in hist[u] if it not in pset and item_cate.get(it) is not None]
    hr = [r for it, r in hist[u] if it not in pset and item_cate.get(it) is not None]
    for x in pool:
        c = item_cate.get(x[0])
        if c is None: continue
        GY.append(M.compute_gamma(hc, hr, c, 5.0))
        PA.append(M.beta_func(x[5], 5.0, 0.5))
        YY.append(x[6])
print()
print('=== 池内物品 (n={}, pos={}) ==='.format(len(YY), int(np.sum(YY))))
print('  γ(类别偏好) 对池标签 AUC = {:.4f}'.format(roc_auc_score(YY, GY)))
print('  p_a(物品评分) 对池标签 AUC = {:.4f}'.format(roc_auc_score(YY, PA)))
print('  γ × p_a     对池标签 AUC = {:.4f}'.format(roc_auc_score(YY, np.array(GY)*np.array(PA))))
print('  γ 均值: 正 {:.4f} / 负 {:.4f}'.format(
    float(np.mean(np.array(GY)[np.array(YY)==1])), float(np.mean(np.array(GY)[np.array(YY)==0]))))

# ---- 追加3: 用真实 M.abeta_ndcg + 同样的贪心, 在我的池子上重放 ----
from sklearn.metrics import roc_auc_score
def greedy_abeta(pool, irg, hc, hr):
    order = [0]
    rem = list(range(1, len(pool)))
    while rem:
        best, bg = None, -1e18
        for i in rem:
            trial = order + [i]
            its = [pool[j][0] for j in trial]
            g = M.abeta_ndcg(its, item_cate, irg, hc, hr, k=len(trial))
            if g > bg: bg, best = g, i
        order.append(best); rem.remove(best)
    return order
H2 = H4 = nb2 = 0; A = []; R = []
for _ in range(400):
    u = valid[rng.randint(len(valid))]
    if u not in ss or u not in ibu or u not in ui:
        continue
    pools = E.build_sim_user_pools(u, ss, ibu, ui, K, pool_dedup='none', pos_ratio_thr=0.5)
    if pools is None or pools[0] is None:
        continue
    ppos, nneg = pools
    npos_k = max(1, int(round(K * 0.2))); pos_pool = ppos[:npos_k]
    pool = pos_pool + nneg[:max(0, K - len(pos_pool))]
    pos_items = set(x[0] for x in pos_pool)
    pset = set(x[0] for x in pool)
    hc = [item_cate[it] for it, r in hist[u] if it not in pset and item_cate.get(it) is not None]
    hr = [r for it, r in hist[u] if it not in pset and item_cate.get(it) is not None]
    irg = {x[0]: x[5] for x in pool}          # safe 口径 = 池内 qua
    order = greedy_abeta(pool, irg, hc, hr)
    ranked = [pool[i][0] for i in order]; labels = [pool[i][6] for i in order]
    nb2 += 1
    H2 += 1.0 if any(it in pos_items for it in ranked[:2]) else 0.0
    H4 += 1.0 if any(it in pos_items for it in ranked[:4]) else 0.0
    if len(set(labels)) > 1:
        A.append(roc_auc_score(labels, np.arange(len(labels))[::-1].astype(float)))
    pr = [i for i, y in enumerate(labels) if y == 1]
    if pr: R.append(float(np.mean(pr)))
print()
print('=== 真实 abeta_ndcg 贪心 ({} 池) ==='.format(nb2))
print('  Hit@2={:.4f}  Hit@4={:.4f}  OrderAUC={:.4f}  正样本位次={:.4f}'.format(
    H2 / nb2, H4 / nb2, float(np.mean(A)), float(np.mean(R))))

# ---- 追加4: 增益是否全并列 / γ 是否恒为 0 ----
allzero = 0; tied = 0; nb3 = 0; same_order = 0; gspread = []
for _ in range(400):
    u = valid[rng.randint(len(valid))]
    if u not in ss or u not in ibu or u not in ui:
        continue
    pools = E.build_sim_user_pools(u, ss, ibu, ui, K, pool_dedup='none', pos_ratio_thr=0.5)
    if pools is None or pools[0] is None:
        continue
    ppos, nneg = pools
    npos_k = max(1, int(round(K * 0.2))); pos_pool = ppos[:npos_k]
    pool = pos_pool + nneg[:max(0, K - len(pos_pool))]
    pset = set(x[0] for x in pool)
    hc = [item_cate[it] for it, r in hist[u] if it not in pset and item_cate.get(it) is not None]
    hr = [r for it, r in hist[u] if it not in pset and item_cate.get(it) is not None]
    irg = {x[0]: x[5] for x in pool}
    gs = []
    for x in pool:
        c = item_cate.get(x[0])
        gam = M.compute_gamma(hc, hr, c, 5.0) if c is not None else 0.0
        pa = M.beta_func(irg[x[0]], 5.0, 0.5) if x[0] in irg else 0.005
        gs.append(pa * gam)
    nb3 += 1
    if max(gs) == 0: allzero += 1
    if len(set(np.round(gs, 12))) == 1: tied += 1
    gspread.append(max(gs) - min(gs))
    order = greedy_abeta(pool, irg, hc, hr)
    if order == list(range(len(pool))): same_order += 1
print()
print('=== 增益结构 ({} 池) ==='.format(nb3))
print('  全部增益=0 的池: {:.1%}'.format(allzero / nb3))
print('  增益完全并列的池: {:.1%}'.format(tied / nb3))
print('  贪心结果 == 池子原顺序 的池: {:.1%}'.format(same_order / nb3))
print('  (池子原顺序 = 正样本在前!) 增益极差中位数={:.2e}'.format(float(np.median(gspread))))

# ---- 追加5: 池内 AUC (之前算的是全局, 会被用户间量纲差抵消) ----
from sklearn.metrics import roc_auc_score as _auc
def wp_auc(vals, labs):
    if len(set(labs)) < 2: return None
    return _auc(labs, vals)
AG, AP, AQ, AB, AGP = [], [], [], [], []
for _ in range(400):
    u = valid[rng.randint(len(valid))]
    if u not in ss or u not in ibu or u not in ui:
        continue
    pools = E.build_sim_user_pools(u, ss, ibu, ui, K, pool_dedup='none', pos_ratio_thr=0.5)
    if pools is None or pools[0] is None:
        continue
    ppos, nneg = pools
    npos_k = max(1, int(round(K * 0.2))); pos_pool = ppos[:npos_k]
    pool = pos_pool + nneg[:max(0, K - len(pos_pool))]
    pset = set(x[0] for x in pool)
    hc = [item_cate[it] for it, r in hist[u] if it not in pset and item_cate.get(it) is not None]
    hr = [r for it, r in hist[u] if it not in pset and item_cate.get(it) is not None]
    g, p, q, labs = [], [], [], []
    for x in pool:
        c = item_cate.get(x[0])
        gam = M.compute_gamma(hc, hr, c, 5.0) if c is not None else 0.0
        pa = M.beta_func(x[5], 5.0, 0.5)
        g.append(gam); p.append(pa); q.append(x[5]); labs.append(x[6])
    a1 = wp_auc(g, labs); a2 = wp_auc(p, labs); a3 = wp_auc(q, labs)
    a4 = wp_auc(np.array(g) * np.array(p), labs)
    if a1: AG.append(a1)
    if a2: AP.append(a2)
    if a3: AQ.append(a3)
    if a4: AB.append(a4)
print()
print('=== 池内 AUC (每池内正负比较, 均值 ± SE) ===')
for nm, a in [('γ 类别偏好', AG), ('p_a 物品评分', AP), ('qua(时间安全)', AQ), ('γ × p_a', AB)]:
    a = np.array(a)
    print('  {:16s} {:.4f} ± {:.4f}  (n={} 池)'.format(nm, a.mean(), a.std(ddof=1)/np.sqrt(len(a)), len(a)))

# ---- 追加6: 公平重放 (随机起始位, 与 _greedy_order 一致) ----
H2r = H4r = nb4 = 0; Ar = []; Rr = []; ratios_s1 = []
for _ in range(400):
    u = valid[rng.randint(len(valid))]
    if u not in ss or u not in ibu or u not in ui:
        continue
    pools = E.build_sim_user_pools(u, ss, ibu, ui, K, pool_dedup='none', pos_ratio_thr=0.5)
    if pools is None or pools[0] is None:
        continue
    ppos, nneg = pools
    npos_k = max(1, int(round(K * 0.2))); pos_pool = ppos[:npos_k]
    pool = pos_pool + nneg[:max(0, K - len(pos_pool))]
    pos_items = set(x[0] for x in pos_pool)
    pset = set(x[0] for x in pool)
    hc = [item_cate[it] for it, r in hist[u] if it not in pset and item_cate.get(it) is not None]
    hr = [r for it, r in hist[u] if it not in pset and item_cate.get(it) is not None]
    irg = {x[0]: x[5] for x in pool}
    start = int(rng.randint(len(pool)))
    order = [start]; rem = [i for i in range(len(pool)) if i != start]
    step = 0
    while rem:
        best, bg = None, -1e18; cand = []
        for i in rem:
            its = [pool[j][0] for j in (order + [i])]
            g = M.abeta_ndcg(its, item_cate, irg, hc, hr, k=len(order) + 1)
            cand.append(g)
            if g > bg: bg, best = g, i
        if step == 0:
            ratios_s1.append(float(np.max(cand) - np.min(cand)))
        order.append(best); rem.remove(best); step += 1
    ranked = [pool[i][0] for i in order]; labels = [pool[i][6] for i in order]
    nb4 += 1
    H2r += 1.0 if any(it in pos_items for it in ranked[:2]) else 0.0
    H4r += 1.0 if any(it in pos_items for it in ranked[:4]) else 0.0
    if len(set(labels)) > 1:
        Ar.append(roc_auc_score(labels, np.arange(len(labels))[::-1].astype(float)))
    pr = [i for i, y in enumerate(labels) if y == 1]
    if pr: Rr.append(float(np.mean(pr)))
print()
print('=== αβ 贪心, 随机起始位 ({} 池) ==='.format(nb4))
print('  Hit@2={:.4f}  Hit@4={:.4f}  OrderAUC={:.4f}  正样本位次={:.4f}'.format(
    H2r / nb4, H4r / nb4, float(np.mean(Ar)), float(np.mean(Rr))))
print('  第一步各候选 dcg/idcg 的极差(中位数): {:.3e}'.format(float(np.median(ratios_s1))))

# ---- 追加7: 把贪心目标换成"原始 gain", 其余完全不变 ----
H2g = H4g = nb5 = 0; Ag = []; Rg = []
for _ in range(400):
    u = valid[rng.randint(len(valid))]
    if u not in ss or u not in ibu or u not in ui:
        continue
    pools = E.build_sim_user_pools(u, ss, ibu, ui, K, pool_dedup='none', pos_ratio_thr=0.5)
    if pools is None or pools[0] is None:
        continue
    ppos, nneg = pools
    npos_k = max(1, int(round(K * 0.2))); pos_pool = ppos[:npos_k]
    pool = pos_pool + nneg[:max(0, K - len(pos_pool))]
    pos_items = set(x[0] for x in pos_pool)
    pset = set(x[0] for x in pool)
    hc = [item_cate[it] for it, r in hist[u] if it not in pset and item_cate.get(it) is not None]
    hr = [r for it, r in hist[u] if it not in pset and item_cate.get(it) is not None]
    gam_map = {}
    for x in pool:
        c = item_cate.get(x[0])
        gam_map[x[0]] = M.compute_gamma(hc, hr, c, 5.0) if c is not None else 0.0
    start = int(rng.randint(len(pool)))
    order = [start]; rem = [i for i in range(len(pool)) if i != start]
    while rem:
        best, bg = None, -1e18
        for i in rem:
            x = pool[i]
            pa = M.beta_func(x[5], 5.0, 0.5)
            g = pa * gam_map[x[0]]          # 原始增益 (不含冗余惩罚)
            if g > bg: bg, best = g, i
        order.append(best); rem.remove(best)
    ranked = [pool[i][0] for i in order]; labels = [pool[i][6] for i in order]
    nb5 += 1
    H2g += 1.0 if any(it in pos_items for it in ranked[:2]) else 0.0
    H4g += 1.0 if any(it in pos_items for it in ranked[:4]) else 0.0
    if len(set(labels)) > 1:
        Ag.append(roc_auc_score(labels, np.arange(len(labels))[::-1].astype(float)))
    pr = [i for i, y in enumerate(labels) if y == 1]
    if pr: Rg.append(float(np.mean(pr)))
print()
print('=== 同池同起始位, 但贪心目标=原始 gain ({} 池) ==='.format(nb5))
print('  Hit@2={:.4f}  Hit@4={:.4f}  OrderAUC={:.4f}  正样本位次={:.4f}'.format(
    H2g / nb5, H4g / nb5, float(np.mean(Ag)), float(np.mean(Rg))))

# ---- 追加8: 池子先打乱后再排 (用户的方案) ----
def run_abeta(shuffle_pool, use_ratio, seed):
    r2 = r4 = nb = 0; A = []; R = []
    rr = np.random.RandomState(seed)
    for _ in range(400):
        u = valid[rr.randint(len(valid))]
        if u not in ss or u not in ibu or u not in ui:
            continue
        pools = E.build_sim_user_pools(u, ss, ibu, ui, K, pool_dedup='none', pos_ratio_thr=0.5)
        if pools is None or pools[0] is None:
            continue
        ppos, nneg = pools
        npos_k = max(1, int(round(K * 0.2))); pos_pool = ppos[:npos_k]
        pool = pos_pool + nneg[:max(0, K - len(pos_pool))]
        pos_items = set(x[0] for x in pos_pool)
        if shuffle_pool:
            perm = rr.permutation(len(pool))
            pool = [pool[i] for i in perm]
        pset = set(x[0] for x in pool)
        hc = [item_cate[it] for it, r in hist[u] if it not in pset and item_cate.get(it) is not None]
        hr = [r for it, r in hist[u] if it not in pset and item_cate.get(it) is not None]
        irg = {x[0]: x[5] for x in pool}
        gam_map = {}
        for x in pool:
            c = item_cate.get(x[0])
            gam_map[x[0]] = M.compute_gamma(hc, hr, c, 5.0) if c is not None else 0.0
        start = int(rr.randint(len(pool)))
        order = [start]; rem = [i for i in range(len(pool)) if i != start]
        while rem:
            best, bg = None, -1e18
            for i in rem:
                if use_ratio:
                    its = [pool[j][0] for j in (order + [i])]
                    g = M.abeta_ndcg(its, item_cate, irg, hc, hr, k=len(order) + 1)
                else:
                    x = pool[i]
                    g = M.beta_func(x[5], 5.0, 0.5) * gam_map[x[0]]
                if g > bg: bg, best = g, i
            order.append(best); rem.remove(best)
        ranked = [pool[i][0] for i in order]; labels = [pool[i][6] for i in order]
        nb += 1
        r2 += 1.0 if any(it in pos_items for it in ranked[:2]) else 0.0
        r4 += 1.0 if any(it in pos_items for it in ranked[:4]) else 0.0
        if len(set(labels)) > 1:
            A.append(roc_auc_score(labels, np.arange(len(labels))[::-1].astype(float)))
        pr = [i for i, y in enumerate(labels) if y == 1]
        if pr: R.append(float(np.mean(pr)))
    return nb, r2 / nb, r4 / nb, float(np.mean(A)), float(np.mean(R))
print()
for tag, sh, ur in [('原顺序 + dcg/idcg (现状)', False, True),
                    ('打乱池子 + dcg/idcg', True, True),
                    ('打乱池子 + 原始gain', True, False)]:
    n, h2, h4, a, r = run_abeta(sh, ur, 7)
    print('  {:24s} n={}  Hit@2={:.4f} Hit@4={:.4f} OrderAUC={:.4f} 位次={:.4f}'.format(
        tag, n, h2, h4, a, r))
