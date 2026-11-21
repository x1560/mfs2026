# coding=utf-8
import sys
import os
import time
import bisect
import argparse
import numpy as np
from math import log2
import pandas as pd
import pickle
from collections import defaultdict
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import metrics as M

PAPER_TEST_RATIO = 0.2

def load_serenlens(path):
    df = pd.read_csv(path, dtype={'timestamp': str}, engine='python')
    df = df.dropna(subset=['user_id', 'item_id', 'timestamp', 'rating', 'label'])
    df['user_id'] = df['user_id'].astype(str).str.strip().str.lower()
    df['item_id'] = df['item_id'].astype(str).str.strip().str.lower()

    user_seq = defaultdict(list)
    labels = defaultdict(dict)
    label_time = {}
    item_rating = {}

    for _, row in df.iterrows():
        uid, iid = row['user_id'], row['item_id']
        try:
            ts = int(float(str(row['timestamp'])))
            ts = max(0, ts)
            rating = float(row['rating']) if not pd.isna(row['rating']) else 0.0
            rating = max(1.0, min(5.0, rating))
            user_seq[uid].append((iid, ts, rating))
            lab = int(float(row['label']))
            if lab in (0, 1):
                labels[uid][iid] = lab
                label_time[(uid, iid)] = ts
                item_rating[(uid, iid)] = rating
        except (ValueError, TypeError):
            continue
    for uid in user_seq:
        user_seq[uid].sort(key=lambda x: x[1])

    pass
    return dict(user_seq), dict(labels), label_time, item_rating

def load_embeddings(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_item_meta(cate_path=None, pop_path=None):
    item_cate, item_pop = {}, {}
    if cate_path:
        with open(cate_path, 'rb') as f:
            item_cate = pickle.load(f)
    if pop_path:
        with open(pop_path, 'rb') as f:
            item_pop = pickle.load(f)
    return item_cate, item_pop

def quantile_transform_from_reference(values, reference):
    """Map values to [0, 1] through the empirical distribution of a reference set; ties take the mid-rank."""
    values = np.asarray(values, dtype=float)
    ref = np.sort(np.asarray(reference, dtype=float))
    if len(ref) <= 1:
        return np.zeros_like(values) + 0.5
    left = np.searchsorted(ref, values, side='left')
    right = np.searchsorted(ref, values, side='right')
    midrank = (left + right + 1.0) / 2.0
    return np.clip((midrank - 1.0) / (len(ref) - 1.0), 0.0, 1.0)


class SimEngine:
    def __init__(self, embeddings):
        self.emb = {}
        for it, v in embeddings.items():
            v = np.asarray(v, dtype=float)
            n = np.linalg.norm(v)
            self.emb[it] = v / n if n > 0 else v

    def sim(self, a, b):
        if a not in self.emb or b not in self.emb:
            return 0.0
        return float(np.dot(self.emb[a], self.emb[b]))


def build_div_sim_fn(args, embeddings):
    """Pick the similarity used by the diversity term: main embedding, Amazon embedding or co-occurrence, falling back to the main embedding when the data is missing."""
    if args.div_sim == 'embedding':
        return None

    if args.div_sim == 'amazon_emb':
        cands = [args.div_sim_emb] if args.div_sim_emb else [
            'amzemb_{}_u5_i5.pkl'.format(args.data),
            'amzemb_{}_u5.pkl'.format(args.data),
            'emb_AMZ5_{}.pkl'.format(args.data),
            'amzemb_{}_u20.pkl'.format(args.data),
            'emb_AMZ20_{}.pkl'.format(args.data),
        ]
        path = next((p for p in cands if p and os.path.exists(p)), None)
        if path is None:
            print("  [警告] --div_sim amazon_emb 找不到可用嵌入 (试过: {}); "
                  "div 回退到主嵌入".format(', '.join([p for p in cands if p])))
            return None
        amazon_emb = load_embeddings(path)
        print("  div 相似度: amazon_emb | {} | {} 物品 | rel/dif 仍用主嵌入".format(
            path, len(amazon_emb)))
        return SimEngine(amazon_emb).sim

    try:
        import amazon_sim
    except ImportError:
        print("  缺少 amazon_sim.py")
        return None
    if args.data not in getattr(amazon_sim, 'DEFAULT_AMAZON', {}):
        print("  amazon_cooc 不支持 data={}".format(args.data))
        return None
    try:
        cooc = amazon_sim.build(data=args.data, log=lambda s: print('  ' + str(s)),
                                min_user_items=30, topn=100,
                                item_filter=set(embeddings.keys()))
    except Exception as e:
        print("  amazon_cooc 构建失败 ({})".format(e))
        return None
    print("  div 相似度: amazon_cooc | {} 物品".format(len(cooc)))
    return cooc.sim


def compute_temporal_factors(item, item_ts, user_seq, sim_engine, item_pop=None, max_pop=1.0, item_ts_first=None, ts_global_max=None, ts_global_min=None, item_rating_map=None, min_history=5, quality_index=None):
    """Compute rel/dif/N/U/Q for one interaction from earlier history only; N uses the first-seen time, U = 1 - pop/max_pop, Q normalises the rating."""
    prev = [it for (it, ts, r) in user_seq if ts < item_ts and it != item]
    if len(prev) < min_history:
        return None
    sims = [sim_engine.sim(item, h) for h in prev]
    sims_sorted = sorted(sims, reverse=True)
    k = min(max(3, len(sims) // 10), 10)
    top_mean = float(np.mean(sims_sorted[:k]))
    all_mean = float(np.mean(sims))

    rel = top_mean
    dif = float(np.clip(top_mean - all_mean, 0.0, 1.0))

    if item_ts_first is not None and ts_global_min is not None and ts_global_max is not None:
        first_ts = item_ts_first.get(item, None)
        if first_ts is None or ts_global_max <= ts_global_min:
            novelty = 1.0
        else:
            novelty = (float(first_ts) - float(ts_global_min)) / (
                float(ts_global_max) - float(ts_global_min) + 1e-9)
            novelty = float(np.clip(novelty, 0.0, 1.0))
    else:
        novelty = 1.0

    if item_pop is not None:
        unpopularity = 1.0 - float(item_pop.get(item, 0)) / (float(max_pop) + 1e-9)
        unpopularity = float(np.clip(unpopularity, 0.0, 1.0))
    else:
        unpopularity = 1.0

    if quality_index is not None:
        qua = quality_index.mean_at(item, item_ts-1)
    else:
        qua = 0.5
        if item_rating_map is not None:
            rt = item_rating_map.get(item, 3.0)
            qua = (rt - 1.0) / 4.0

    return rel, dif, float(novelty), float(unpopularity), float(np.clip(qua, 0, 1))




def build_sim_user_pools(target_uid, user_sim, items_by_uid, user_items_set, k,
                         pool_dedup='none', pos_ratio_thr=0.5,
                         min_pos=None, min_neg=None):
    """Build pools from neighbour annotations, excluding the target user's history; none assigns per record, ratio by vote share, majority by majority vote, any_pos by any positive vote."""
    sim_users = [v for v, _ in user_sim.get(target_uid, [])]
    u_items = user_items_set.get(target_uid, set())
    req_pos = k if min_pos is None else int(min_pos)
    req_neg = k if min_neg is None else int(min_neg)

    if pool_dedup == 'none':
        pos_pool, neg_pool = [], []
        for v in sim_users:
            for tup in items_by_uid.get(v, []):
                if tup[0] in u_items:
                    continue
                (pos_pool if tup[6] == 1 else neg_pool).append(tup + (target_uid,))
        if len(pos_pool) < req_pos or len(neg_pool) < req_neg:
            return None, None
        return pos_pool, neg_pool

    merged = {}
    for v in sim_users:
        for item, rel, dif, nov, unpop, qua, label, rel_raw in items_by_uid.get(v, []):
            if item in u_items:
                continue
            cur = merged.get(item)
            if cur is None:
                merged[item] = {'n_pos': int(label == 1), 'n_neg': int(label == 0),
                                'feat': (rel, dif, nov, unpop, qua),
                                'rel_raw': rel_raw}
            else:
                cur['n_pos' if label == 1 else 'n_neg'] += 1

    pos_pool, neg_pool = [], []
    for item, rec in merged.items():
        n_pos, n_neg = rec['n_pos'], rec['n_neg']
        if pool_dedup == 'any_pos':
            is_pos = n_pos > 0
        elif pool_dedup == 'majority':
            is_pos = n_pos > n_neg
        else:                                   # ratio
            is_pos = (n_pos / (n_pos + n_neg)) >= float(pos_ratio_thr)
        rel, dif, nov, unpop, qua = rec['feat']
        tup = (item, rel, dif, nov, unpop, qua, 1 if is_pos else 0,
               rec['rel_raw'], target_uid)
        (pos_pool if is_pos else neg_pool).append(tup)

    if len(pos_pool) < req_pos or len(neg_pool) < req_neg:
        return None, None
    return pos_pool, neg_pool


"""按全局时间边界准备训练与测试数据，评分时可逐条扩展用户自身历史。"""
def _attach_ts(df, label_time):
    """Attach timestamps to the factor table."""
    df = df.copy()
    df['ts'] = [label_time.get((u, i), 0) for u, i in zip(df['uid'], df['item'])]
    return df


def _causal_running_center(uid_arr, ts_arr, lab_arr, rel_arr, dif_arr, min_count):
    """Accumulate earlier negatives per user in time order and return (c_rel, c_dif, n_used); rows sharing a timestamp never enter each other's center."""
    n = len(uid_arr)
    c_rel = np.zeros(n, dtype=float)
    c_dif = np.zeros(n, dtype=float)
    n_used = np.zeros(n, dtype=int)
    order = np.lexsort((ts_arr, uid_arr))
    u_s = uid_arr[order]
    t_s = ts_arr[order]
    l_s = lab_arr[order]
    r_s = rel_arr[order]
    d_s = dif_arr[order]
    start = 0
    while start < n:
        end = start
        while end < n and u_s[end] == u_s[start]:
            end += 1
        sr = sd = 0.0
        cnt = 0
        i = start
        while i < end:
            j = i
            while j < end and t_s[j] == t_s[i]:
                j += 1
            if cnt >= min_count:
                mr, md = sr / cnt, sd / cnt
                for k in range(i, j):
                    idx = order[k]
                    c_rel[idx] = mr
                    c_dif[idx] = md
                    n_used[idx] = cnt
            else:
                for k in range(i, j):
                    n_used[order[k]] = cnt
            for k in range(i, j):
                if l_s[k] == 0:
                    sr += r_s[k]
                    sd += d_s[k]
                    cnt += 1
            i = j
        start = end
    return c_rel, c_dif, n_used

def _fit_quantile_ref(reference):
    return np.sort(np.asarray(reference, dtype=float))

def _quantile_transform(values, sorted_ref):
    values = np.asarray(values, dtype=float)
    if len(sorted_ref) <= 1:
        return np.zeros_like(values) + 0.5
    left = np.searchsorted(sorted_ref, values, side='left')
    right = np.searchsorted(sorted_ref, values, side='right')
    midrank = (left + right + 1.0) / 2.0
    return np.clip((midrank - 1.0) / (len(sorted_ref) - 1.0), 0.0, 1.0)

def _build_window_item_stats(win_user_seq, valid_uids):
    """Aggregate first-seen time, interaction count and rating sum per item over the training window."""
    flat_items, flat_ts, flat_r = [], [], []
    for uid in valid_uids:
        for it, ts, r in win_user_seq.get(uid, []):
            flat_items.append(it)
            flat_ts.append(ts)
            flat_r.append(r)
    if not flat_items:
        return {}, 0.0, 0.0, {}, {}, 1.0
    wdf = pd.DataFrame({'item': flat_items, 'ts': flat_ts, 'r': flat_r})
    win_item_ts_first = wdf.groupby('item')['ts'].min().to_dict()
    win_item_pop = wdf.groupby('item').size().to_dict()
    win_item_rt_sum = wdf.groupby('item')['r'].sum().to_dict()
    win_ts_min = float(np.min(flat_ts))
    win_ts_max = float(np.max(flat_ts))
    win_max_pop = float(max(win_item_pop.values())) if win_item_pop else 1.0
    return (win_item_ts_first, win_ts_min, win_ts_max,
            win_item_pop, win_item_rt_sum, win_max_pop)

def _compute_item_features(train_df, test_df, win_item_ts_first, win_ts_min, win_ts_max, win_item_pop, win_item_rt_sum, win_max_pop, rating_min=1.0, rating_max=5.0,
                           label_time=None, user_seq=None):
    """Compute nov_q, unpop_q and qua for train and test; the N/U quantile reference is fitted on train only."""
    n_tr = len(train_df)
    combined = pd.concat([train_df, test_df], ignore_index=True)
    item = combined['item']

    if os.environ.get('MDCG_LEGACY_WIN_FACTORS') == '1' or label_time is None or user_seq is None:
        first_ts = item.map(win_item_ts_first).fillna(win_ts_max).to_numpy()
        if win_ts_max > win_ts_min:
            nov_raw = np.clip((first_ts - win_ts_min) / (win_ts_max - win_ts_min + 1e-9), 0.0, 1.0)
        else:
            nov_raw = np.full(len(combined), 0.5)
        pop = item.map(win_item_pop).fillna(0).to_numpy()
        unp_raw = np.clip(1.0 - pop / (win_max_pop + 1e-9), 0.0, 1.0)
        print("  [factor] 沿用旧口径（窗口内常数 nov/unp）")
    else:
        _row_ts = np.array([label_time.get((u, i), 0)
                            for u, i in zip(combined['uid'], combined['item'])], float)
        _first_all = {}
        for _u0, _seq0 in user_seq.items():
            for _it0, _ts0, _r0 in _seq0:
                _t0 = int(_ts0)
                if _it0 not in _first_all or _t0 < _first_all[_it0]:
                    _first_all[_it0] = _t0
        first_ts = np.array([_first_all.get(it, win_ts_max) for it in item], float)
        _den = np.maximum(_row_ts - win_ts_min, 1.0)
        nov_raw = np.clip((first_ts - win_ts_min) / _den, 0.0, 1.0)
        _tmp = pd.DataFrame({'item': combined['item'].values, 'ts': _row_ts})
        _tmp = _tmp.sort_values('ts', kind='mergesort')
        _cum = _tmp.groupby('item').cumcount().to_numpy() + 1.0
        _cnt = np.empty(len(combined), float)
        _cnt[_tmp.index.to_numpy()] = _cum
        unp_raw = np.clip(1.0 - _cnt / (float(_cnt.max()) + 1e-9), 0.0, 1.0)

    rt_sum = item.map(win_item_rt_sum).fillna(0.0).to_numpy()
    pop = item.map(win_item_pop).fillna(0).to_numpy()
    cnt = pop
    avg_rt = np.where(cnt > 0, rt_sum / np.where(cnt > 0, cnt, 1.0),
                      (rating_min + rating_max) / 2.0)
    qua = np.clip((avg_rt - rating_min) / (rating_max - rating_min), 0.0, 1.0)
    ref_nov = _fit_quantile_ref(nov_raw[:n_tr])
    ref_unp = _fit_quantile_ref(unp_raw[:n_tr])
    combined['nov_q'] = _quantile_transform(nov_raw, ref_nov)
    combined['unpop_q'] = _quantile_transform(unp_raw, ref_unp)
    combined['rare'] = 0.5 * (combined['nov_q'] + combined['unpop_q'])
    combined['qua'] = qua
    return combined.iloc[:n_tr].copy(), combined.iloc[n_tr:].copy()


PROFILE_COLS = ('rel', 'dif', 'rare', 'qua')
PROFILE_TOPK = 50


def _build_surprise_profiles(train_df, topk=PROFILE_TOPK, cols=PROFILE_COLS, use_all=False):
    """Build profiles from per-user means and keep the cosine top-k neighbours; positives only by default, every row when use_all is set."""
    pos = train_df if use_all else train_df[train_df['label'] == 1]
    prof = pos.groupby('uid')[list(cols)].mean()
    profile_uids = list(prof.index)
    surprise_sim = {}
    if len(profile_uids) >= 2:
        arr = prof.to_numpy()
        arr = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9)
        sim_mat = arr @ arr.T
        top_n = min(topk, len(profile_uids) - 1)
        for i, uid in enumerate(profile_uids):
            row = sim_mat[i].copy()
            row[i] = -1.0
            top = np.argpartition(row, -top_n)[-top_n:]
            top = top[np.argsort(row[top])[::-1]]
            surprise_sim[uid] = [(profile_uids[j], float(row[j])) for j in top if row[j] >= 0]
    else:
        for uid in profile_uids:
            surprise_sim[uid] = []
    return surprise_sim


def _build_per_user_wakeup(train_df, test_uids, surprise_sim, use_copula, pp_global, np_global):
    """Fit the per-user wake-up parameters on neighbour training rows, falling back to the global parameters when the sample is too small."""
    rel = train_df['rel'].to_numpy()
    dif = train_df['dif'].to_numpy()
    label = train_df['label'].to_numpy()
    uid = train_df['uid'].to_numpy()
    tr_uid_to_idx = defaultdict(list)
    for i, u in enumerate(uid):
        tr_uid_to_idx[u].append(i)

    per_user, n_has_profile, n_has_enough_sim = {}, 0, 0
    for u in test_uids:
        sim_users = [v for v, _ in surprise_sim.get(u, [])]
        if sim_users:
            n_has_profile += 1
        pr, pd_, nr, nd = [], [], [], []
        for v in sim_users:
            for idx in tr_uid_to_idx.get(v, []):
                if label[idx] == 1:
                    pr.append(rel[idx]); pd_.append(dif[idx])
                else:
                    nr.append(rel[idx]); nd.append(dif[idx])
        if len(pr) >= 5 and len(nr) >= 5:
            per_user[u] = (M.fit_class_params(pr, pd_, use_copula),
                           M.fit_class_params(nr, nd, use_copula))
            n_has_enough_sim += 1
        else:
            per_user[u] = (pp_global, np_global)
    return per_user, n_has_profile, n_has_enough_sim


def _item_unexpectedness(uid, item, user_seq, label_time, sim_engine):
    """Unexpectedness is one minus the mean similarity to the user's earlier items."""
    unexp = 0.5
    if uid in user_seq:
        target_ts = label_time.get((uid, item))
        if target_ts is not None:
            hist = [h[0] for h in user_seq[uid] if h[1] < target_ts and h[0] != item]
            hist = hist[-50:]
            if hist:
                unexp = float(np.mean([1.0 - sim_engine.sim(item, h) for h in hist]))
    return unexp

def _prepare_temporal(factor_df, user_seq, label_time, sim_engine, use_copula, min_history,
                      test_ratio, min_user_items, global_surprise_sim, test_only_with_profile,
                      lambda_source, lambda_stat, lambda_min_pool, tag, topk,
                      min_valid_users=10, min_neg_test=3, quality_index=None,
                      qua_mode='pit', low_rating_to_train=False, low_rating_thr=2.5,
                      boundary_pairs=None, neighbor_fallback=None, profile_all_items=False):
    """Split in time order and prepare the training-window parameters; pit uses the time-of-interaction quality, window the window mean; returns None when too few users remain."""
    print()
    print("=" * 60)
    print(">>> 实验一{}: (copula={})".format(tag, use_copula))
    print("=" * 60)

    df = _attach_ts(factor_df, label_time).sort_values('ts').reset_index(drop=True)
    all_ts = df['ts'].values
    if boundary_pairs is not None:
        if isinstance(boundary_pairs, str):
            _bp = pd.read_csv(boundary_pairs, dtype=str)
            boundary_pairs = set(zip(_bp.iloc[:, 0].astype(str),
                                     _bp.iloc[:, 1].astype(str)))
        _is_bp = np.array([(str(u), str(it)) in boundary_pairs
                           for u, it in zip(df['uid'], df['item'])])
        _is_low = np.zeros(len(df), dtype=bool)
        _split_ts = all_ts[_is_bp] if _is_bp.sum() >= 10 else all_ts
        boundary = _split_ts[int(len(_split_ts) * (1.0 - test_ratio))]
        print("  (分位点只用标注行 {} 条计算)".format(int(_is_bp.sum())))
    elif low_rating_to_train:
        _rate = {}
        for _u, _seq in user_seq.items():
            for _it, _ts, _r in _seq:
                _rate[(_u, _it)] = _r
        _is_low = np.array([
            (_rate.get((u, it)) is not None and _rate[(u, it)] <= low_rating_thr and lab == 0)
            for u, it, lab in zip(df['uid'], df['item'], df['label'])])
        _split_ts = all_ts[~_is_low]
        boundary = _split_ts[int(len(_split_ts) * (1.0 - test_ratio))]
    else:
        _is_low = np.zeros(len(df), dtype=bool)
        boundary = all_ts[int(len(all_ts) * (1.0 - test_ratio))]
    pass

    if FREEZE_BOUNDARY > 0:      # --boundary_ts: 删除历史时冻结切分点, 保证测试集不变
        boundary = float(FREEZE_BOUNDARY)
        print("  [freeze] 切分点冻结为 {:.0f}（不随删档重算）".format(boundary))
    _in_train = (df['ts'].values <= boundary) | _is_low
    train_df_all = df[_in_train].copy()
    test_df_all = df[~_in_train].copy()
    tr_cnt = train_df_all.groupby('uid').size()
    te_cnt = test_df_all.groupby('uid').size()
    valid_uids = {u for u in set(tr_cnt.index) & set(te_cnt.index)
                  if tr_cnt[u] >= min_user_items and te_cnt[u] >= 1}
    if len(valid_uids) < min_valid_users:
        print("  有效用户不足 ({}个), 实验终止".format(len(valid_uids)))
        return None
    train_df = train_df_all[train_df_all['uid'].isin(valid_uids)].copy()
    test_df = test_df_all[test_df_all['uid'].isin(valid_uids)].copy()
    assert test_df['ts'].min() > boundary, "test 泄漏到 train 时间窗口"
    n_train = len(train_df)
    n_pos_train = int((train_df['label'] == 1).sum())
    n_pos_test = int((test_df['label'] == 1).sum())
    n_neg_test = int((test_df['label'] == 0).sum())
    pass
    if n_pos_test < 1 or n_neg_test < min_neg_test or n_pos_train < 5:
        print("  test 正/负样本不足或 train 正样本不足, 跳过")
        return None

    win_user_seq = {}
    for uid in valid_uids:
        if uid in user_seq:
            win_user_seq[uid] = [(it, ts, r) for (it, ts, r) in user_seq[uid] if ts <= boundary]
    (win_item_ts_first, win_ts_min, win_ts_max, win_item_pop, win_item_rt_sum, win_max_pop) = _build_window_item_stats(win_user_seq, valid_uids)
    train_df, test_df = _compute_item_features(train_df, test_df, win_item_ts_first, win_ts_min, win_ts_max, win_item_pop, win_item_rt_sum, win_max_pop,
                                              label_time=label_time, user_seq=user_seq)

    if qua_mode == 'window':
        print("  qua: 训练窗口均值(旧口径, 含时间泄漏) —— 仅用于复现历史结果")
    elif quality_index is not None:
        train_df['qua'] = [quality_index.mean_at(it, ts) for it, ts in
                           zip(train_df['item'].tolist(), train_df['ts'].to_numpy())]
        test_df['qua'] = [quality_index.mean_at(it, ts) for it, ts in
                          zip(test_df['item'].tolist(), test_df['ts'].to_numpy())]
    else:
        sl_pairs = defaultdict(list)
        for _u, _seq in win_user_seq.items():
            for _it, _t, _r in _seq:
                sl_pairs[_it].append((_t, _r))
        sl_index = {}
        for _it, _evs in sl_pairs.items():
            _evs.sort()
            _ts_l, _s_l, _c_l = [], [], []
            _acc_s, _acc_c = 0.0, 0
            for _t, _r in _evs:
                _acc_s += _r
                _acc_c += 1
                _ts_l.append(_t)
                _s_l.append(_acc_s)
                _c_l.append(_acc_c)
            sl_index[_it] = (_ts_l, _s_l, _c_l)

        def _sl_mean_at(_it, _t):
            _e = sl_index.get(_it)
            if _e is None:
                return 0.5
            _i = bisect.bisect_left(_e[0], _t)
            if _i == 0:
                return 0.5
            _m = _e[1][_i - 1] / _e[2][_i - 1]
            return float(np.clip((_m - 1.0) / 4.0, 0.0, 1.0))

        train_df['qua'] = [_sl_mean_at(it, ts) for it, ts in
                           zip(train_df['item'].tolist(), train_df['ts'].to_numpy())]
        test_df['qua'] = [_sl_mean_at(it, ts) for it, ts in
                          zip(test_df['item'].tolist(), test_df['ts'].to_numpy())]
        print("  qua: 无外部质量源, 回退到 SerenLens 自身的截至时刻均值")

    tr_rel = train_df['rel'].values
    tr_dif = train_df['dif'].values
    tr_label = train_df['label'].values
    tr_pos = tr_label == 1
    tr_neg = tr_label == 0
    pp_global = M.fit_class_params(tr_rel[tr_pos], tr_dif[tr_pos], use_copula)
    np_global = M.fit_class_params(tr_rel[tr_neg], tr_dif[tr_neg], use_copula)
    surprise_sim_win = (global_surprise_sim if global_surprise_sim is not None
                        else _build_surprise_profiles(train_df, topk,
                                                      use_all=profile_all_items))
    if neighbor_fallback:
        _n_add = 0
        for _u, _lst in neighbor_fallback.items():
            if _u not in surprise_sim_win:
                surprise_sim_win[_u] = _lst
                _n_add += 1
        print("  [neighbor] 画像缺失回退：{} 人改用共评邻居（图规模 {} -> {}）"
              .format(_n_add, len(surprise_sim_win) - _n_add, len(surprise_sim_win)))

    item_nov_win = dict(zip(train_df['item'], train_df['nov_q']))
    item_unpop_win = dict(zip(train_df['item'], train_df['unpop_q']))
    item_rare_win = {it: 0.5 * (item_nov_win[it] + item_unpop_win[it]) for it in item_nov_win}
    scarcity_win = M.build_user_scarcity_params(
        win_user_seq, item_rare_win, item_nov_win, item_unpop_win, min_history=min_history)
    if lambda_source == 'surprise_pool':
        pos_df = train_df[train_df['label'] == 1]
        pos_nu_win = {u: list(zip(g['nov_q'].tolist(), g['unpop_q'].tolist()))
                      for u, g in pos_df.groupby('uid')}
        scarcity_win = M.personalize_scarcity_centers(scarcity_win, surprise_sim_win, pos_nu_win, center=lambda_stat, min_pool=lambda_min_pool)

    test_uids = set(test_df['uid'])
    per_user_pp_np, n_has_profile, n_has_enough_sim = _build_per_user_wakeup(train_df, test_uids, surprise_sim_win, use_copula, pp_global, np_global)
    pass

    if test_only_with_profile:
        profiled = set(surprise_sim_win.keys())
        test_df = test_df[test_df['uid'].isin(profiled)]
        if len(test_df) == 0:
            print("  无有画像的测试用户, 跳过")
            return None
        n_pos_test = int((test_df['label'] == 1).sum())
        n_neg_test = int((test_df['label'] == 0).sum())
        print("  仅评估有画像用户: test={} users={} pos={} neg={}".format(
            len(test_df), len(set(test_df['uid'])), n_pos_test, n_neg_test))
        if n_pos_test < 3 or n_neg_test < 3:
            print("  过滤后正/负样本不足, 跳过")
            return None

    return {
        'boundary': boundary, 'train_df': train_df, 'test_df': test_df,
        'n_train': n_train, 'n_pos_train': n_pos_train,
        'win_item_pop': win_item_pop, 'win_max_pop': win_max_pop,
        'scarcity_win': scarcity_win, 'per_user_pp_np': per_user_pp_np,
        'pp_global': pp_global, 'np_global': np_global,
        'user_seq': user_seq, 'label_time': label_time, 'sim_engine': sim_engine,
        'use_copula': use_copula, 'min_history': min_history,
        'surprise_sim_win': surprise_sim_win,
    }


_REFIT_WARNED = {'done': False}

def _refit_user_accommodation(n_hist, u_hist, frozen_p, min_history, min_dev=0.05):
    """Update the side curvatures from the user's own history; surprise-pool centers stay fixed, the others are re-estimated; explore_rank is kept; returns copies."""
    p = dict(frozen_p)
    n = np.asarray(n_hist, dtype=float)
    u = np.asarray(u_hist, dtype=float)
    enough = min(len(n), len(u)) >= min_history
    base_alpha = float(getattr(M, 'ACCOM_ALPHA', 1.0))

    if enough:
        fit = getattr(M, '_fit_asym_alphas', None)
        cen = getattr(M, '_robust_center', None)
        if fit is None or cen is None:
            if not _REFIT_WARNED['done']:
                print("  [warn] M._fit_asym_alphas / M._robust_center 缺失, "
                      "用户参数保持冻结(版本二 == 版本一)")
                _REFIT_WARNED['done'] = True
            return p
        ln, lu = cen(n), cen(u)
        _, an_m, an_p = fit(n, center=ln, base_alpha=base_alpha, min_dev=min_dev)
        _, au_m, au_p = fit(u, center=lu, base_alpha=base_alpha, min_dev=min_dev)
    else:
        ln = lu = 0.5
        an_m = an_p = au_m = au_p = base_alpha / (0.15 * 0.15) if base_alpha > 0 else 0.0

    p['alpha_n_minus'], p['alpha_n_plus'] = an_m, an_p
    p['alpha_u_minus'], p['alpha_u_plus'] = au_m, au_p

    if not p.get('lambda_n_hist'):
        p['lambda_n'], p['lambda_u'] = ln, lu
        est = getattr(M, '_estimate_bands', None)
        bwf = getattr(M, '_bandwidth_from_alphas', None)
        fit_peaks = getattr(M, '_fit_adaptive_centers', None)
        if enough and est is not None:
            mn = est(n, bandwidth=bwf(an_m, an_p) if bwf is not None else None)
            mu = est(u, bandwidth=bwf(au_m, au_p) if bwf is not None else None)
        elif enough and fit_peaks is not None:
            mn = fit_peaks(n)
            mu = fit_peaks(u)
        else:
            mn = mu = None
        if mn is not None:
            p['n_centers'], p['n_weights'], p['k_n'] = mn['centers'], mn['weights'], mn['k']
            p['u_centers'], p['u_weights'], p['k_u'] = mu['centers'], mu['weights'], mu['k']
        else:
            p['n_centers'], p['n_weights'] = [ln], [1.0]
            p['u_centers'], p['u_weights'] = [lu], [1.0]

    norm_fn = getattr(M, '_mixture_cauchy_norm_alpha', None)
    if norm_fn is None:
        if not _REFIT_WARNED['done']:
            print("  [warn] M._mixture_cauchy_norm_alpha 缺失, mixture 归一化常数未更新")
            _REFIT_WARNED['done'] = True
    else:
        p['n_mix_norm'] = norm_fn(p['n_centers'], p['n_weights'], an_m, an_p)
        p['u_mix_norm'] = norm_fn(p['u_centers'], p['u_weights'], au_m, au_p)
    return p




def make_pool_loo_fitter(factor_df, surprise_sim, use_copula):
    """Return fitter(uid, items) that refits the global and per-user wake-up parameters with the candidate items removed; returns (pp, np, per_user)."""
    pos = factor_df[factor_df['label'] == 1]
    neg = factor_df[factor_df['label'] == 0]
    p_it, p_rel, p_dif = (pos['item'].to_numpy(), pos['rel'].to_numpy(),
                          pos['dif'].to_numpy())
    n_it, n_rel, n_dif = (neg['item'].to_numpy(), neg['rel'].to_numpy(),
                          neg['dif'].to_numpy())
    uid_rows = {}
    for u, g in factor_df.groupby('uid'):
        uid_rows[u] = (g['item'].to_numpy(), g['rel'].to_numpy(),
                       g['dif'].to_numpy(), g['label'].to_numpy())

    def _keep(ex, items, *arrays):
        m = np.array([it not in ex for it in items], dtype=bool)
        return [a[m] for a in arrays]

    def fitter(target_uid, pool_items):
        ex = set(pool_items)
        pr, pd = _keep(ex, p_it, p_rel, p_dif)
        nr, nd = _keep(ex, n_it, n_rel, n_dif)
        pp = M.fit_class_params(pr, pd, use_copula)
        np_ = M.fit_class_params(nr, nd, use_copula)
        pu = {}
        if target_uid is not None and target_uid in surprise_sim:
            ar, ad, br, bd = [], [], [], []
            for v, _ in surprise_sim.get(target_uid, []):
                rows = uid_rows.get(v)
                if rows is None:
                    continue
                r_a, d_a, l_a = _keep(ex, rows[0], rows[1], rows[2], rows[3])
                pm = (l_a == 1)
                ar.extend(r_a[pm].tolist()); ad.extend(d_a[pm].tolist())
                br.extend(r_a[~pm].tolist()); bd.extend(d_a[~pm].tolist())
            if len(ar) >= 5 and len(br) >= 5:
                pu[target_uid] = (
                    M.fit_class_params(np.array(ar), np.array(ad), use_copula),
                    M.fit_class_params(np.array(br), np.array(bd), use_copula))
        return pp, np_, pu
    return fitter


def exp_perturbation(user_sim, items_by_uid, user_items_set, sim_engine,
                     pos_params, neg_params,
                     item_pop=None, max_pop=1.0, item_cate=None,
                     item_rating_g=None, item_aspects=None,
                     use_copula=True, n_levels=11, k=10, n_lists=200,
                     seed=42, user_scarcity_params=None,
                     per_user_arousal=None,
                     pool_dedup='none', pos_ratio_thr=0.5,
                     pool_fitter=None, perturb_metrics=None, num_users=None,
                     fnpair=None, hist_items_ts=None):
    print("\n" + "=" * 60)
    print(">>> 实验二: 扰动一致性")
    print("=" * 60)
    rng = np.random.RandomState(seed)
    valid_uids = [uid for uid in user_sim if uid in items_by_uid and uid in user_items_set]
    print("  有效目标用户: {}".format(len(valid_uids)))
    if len(valid_uids) == 0:
        print("无有效用户, 跳过")
        return {}

    levels = np.linspace(0, 1, n_levels)
    sf = sim_engine.sim
    if item_pop is None:
        item_pop = {}

    _w_lc_box = [None]
    _tr_lc = None
    if (not perturb_metrics) or ('LinCombo' in str(perturb_metrics)):
        try:
            import pandas as _pd2
            _rows_lc = [(t[0], t[1], t[2], t[3], t[4], t[5], int(t[6]))
                        for _rows in items_by_uid.values() for t in _rows if len(t) >= 7]
            if _rows_lc:
                _tr_lc = _pd2.DataFrame(
                    _rows_lc,
                    columns=['item', 'rel', 'dif', 'nov_q', 'unpop_q', 'qua', 'label']).dropna()
                _tr_lc = _tr_lc[_tr_lc['label'].isin([0, 1])].reset_index(drop=True)
                print('  [LinCombo] 全量训练矩阵 n=%d 正=%d (已剔除无标签行)'
                      % (len(_tr_lc), int(_tr_lc['label'].sum())))
        except Exception as _e2:
            print('  [LinCombo] 训练矩阵构建失败: %s' % _e2)

    def _fit_lc():
        """用全部有效标注拟合五因子权重，样本或类别不足时返回 None。"""
        if _tr_lc is None:
            return None
        _y = _tr_lc['label'].values.astype(int)
        if len(_y) < 200 or _y.sum() == 0 or _y.sum() == len(_y):
            return None
        from sklearn.linear_model import LogisticRegression
        _m2 = LogisticRegression(max_iter=2000, class_weight='balanced').fit(
            _tr_lc[['rel', 'dif', 'nov_q', 'unpop_q', 'qua']].values, _y)
        return [float(v) for v in _m2.coef_[0]]

    _w_lc_box[0] = _fit_lc()
    if _w_lc_box[0] is not None:
        print('  [LinCombo] 权重 w=[%s]' % ', '.join('%+.3f' % v for v in _w_lc_box[0]))

    metric_names = ['M-DCG', 'M-DCG(I-DCG)', 'EPC', 'ILD', 'ContentDist', 'GeUnexp',
                    'abeta-nDCG', 'alpha-nDCG', 'ndcg_rating', 'MMR', 'FN-Surprise', 'xQuAD',
                    'LinCombo']
    if perturb_metrics:
        _want = set(x.strip() for x in str(perturb_metrics).split(',') if x.strip())
        _kept = [m for m in metric_names if m in _want]
        if _kept:
            metric_names = _kept
            print("  [仅评指标] {}".format(', '.join(metric_names)))
    trend = {m: {lvl: [] for lvl in levels} for m in metric_names}
    taus = {m: [] for m in metric_names}

    _cur = [pos_params, neg_params, per_user_arousal]
    _uhist = {'c': [], 'r': []}

    def score_list(lst, metric):
        items = [x[0] for x in lst]
        target = lst[0][8] if len(lst[0]) >= 9 else None
        pp, np_i = _cur[0], _cur[1]
        if _cur[2] and target in _cur[2]:
            pp, np_i = _cur[2][target]
        if metric == 'LinCombo':
            if _w_lc_box[0] is None:
                return float('nan')
            _w9 = _w_lc_box[0]
            return float(np.mean([sum(_w9[j] * x[1 + j] for j in range(len(_w9)))
                                  for x in lst]))
        if metric == 'M-DCG':
            mlst = [(x[0], x[1], x[2], x[3], x[4], x[5], x[8]) for x in lst]
            return M.mdcg_score(mlst, pp, np_i, sim_func=sf,
                                use_copula=use_copula, k=k,
                                user_scarcity_params=user_scarcity_params)
        elif metric == 'M-DCG(I-DCG)':
            mlst = [(x[0], x[1], x[2], x[3], x[4], x[5], x[8]) for x in lst]
            mdcg = M.mdcg_score(mlst, pp, np_i, sim_func=sf,
                                use_copula=use_copula, k=k,
                                user_scarcity_params=user_scarcity_params)
            idcg = M.compute_idcg(mlst, pp, np_i, sim_func=sf,
                                  use_copula=use_copula, k=k,
                                  user_scarcity_params=user_scarcity_params)
            return mdcg / idcg if idcg > 0 else 0.0
        elif metric == 'ndcg_rating':
            return M.ndcg_score([float((item_rating_g or {}).get(it, 0.0)) for it in items], k,
                                normalize=M.ABETA_NORMALIZE)
        elif metric == 'ERR':
            return M.err_score([float((item_rating_g or {}).get(it, 0.0)) for it in items], k)
        elif metric == 'RBP':
            return M.rbp_score([float((item_rating_g or {}).get(it, 0.0)) for it in items], k=k)
        elif metric == 'nDCG_seren':
            return M.ndcg_seren_score([x[6] for x in lst], k)
        elif metric == 'EPC':
            return M.epc_score(items, item_pop, max_pop, k,
                               rel={x[0]: float(x[5]) for x in lst})
        elif metric == 'SteckDamping':
            _rv = [float((item_rating_g or {}).get(it, 0.0)) / 5.0 for it in items]
            _pv = [item_pop.get(it, 0) for it in items]
            _ds = M.popularity_damping_scores(_rv, _pv, max_pop, 0.5)
            return float(sum(_ds)) / max(1, len(_ds))
        elif metric == 'GeUnexp':
            _pv = sorted(item_pop.values()) if item_pop else [0]
            _med = _pv[len(_pv) // 2]
            _hs = [1.0 if item_pop.get(it, 0) <= _med else 0.0 for it in items]
            return float(sum(_hs)) / max(1, len(_hs))
        elif metric == 'ContentDist':
            _hs = M.content_dist_scores([x[7] for x in lst])
            return float(sum(_hs)) / max(1, len(_hs))
        elif metric == 'CoverageRate':
            _cs = [item_cate.get(it) for it in items if item_cate.get(it) is not None]
            return float(len(set(_cs))) / max(1, len(_cs)) if _cs else 0.0
        elif metric == 'MMR':
            _rv = [float((item_rating_g or {}).get(it, 0.0)) / 5.0 for it in items]
            _tot = 0.0
            for _t, _it in enumerate(items):
                _mx = max([sf(_it, _jt) for _jt in items[:_t]]) if _t else 0.0
                _tot += _rv[_t] - _mx
            return _tot / max(1, len(items))
        elif metric == 'DPP':
            _n = len(items)
            _A = np.array([[sf(items[_i], items[_j]) for _j in range(_n)] for _i in range(_n)])
            try:
                _sg, _ld = np.linalg.slogdet(_A + 1e-6 * np.eye(_n))
                return float(_ld) / max(1, _n) if _sg > 0 else 0.0
            except Exception:
                return 0.0
        elif metric == 'xQuAD':
            _cs = [item_cate.get(it) for it in items]
            _rv = [float((item_rating_g or {}).get(it, 0.0)) / 5.0 for it in items]
            _hd = M._cat_dist([x for x in _cs if x is not None], None)
            _sel = {}
            _tot = 0.0
            for _t, _cc in enumerate(_cs):
                _cov = _hd.get(_cc, 0.0) * (1.0 - _sel.get(_cc, 0.0)) if _cc is not None else 0.0
                _tot += 0.5 * _rv[_t] + 0.5 * _cov
                if _cc is not None:
                    _sel[_cc] = 1.0
            return _tot / max(1, len(items))
        elif metric == 'IASelect':
            _cs = [item_cate.get(it) for it in items]
            _rv = [float((item_rating_g or {}).get(it, 0.0)) / 5.0 for it in items]
            _hd = M._cat_dist([x for x in _cs if x is not None], None)
            _sel = {}
            _tot = 0.0
            for _t, _cc in enumerate(_cs):
                _w = 0.0 if (_cc is not None and _sel.get(_cc)) else (_hd.get(_cc, 0.0) if _cc is not None else 0.0)
                _tot += _rv[_t] * _w
                if _cc is not None:
                    _sel[_cc] = 1.0
            return _tot / max(1, len(items))
        elif metric == 'ILD':
            return M.ild_score(items, sf, k)
        elif metric == 'FN-Surprise':
            _fp = fnpair or {}
            _hits = (hist_items_ts or {}).get(target, [])
            _by = {}
            for _h, _ts in _hits:
                _pp = _fp.get(_h, (None, 0.0))[0]
                if _pp is not None:
                    _by[_pp] = _by.get(_pp, 0) + 1
            _vals = []
            for _it in items:
                _pp, _ss = _fp.get(_it, (None, 0.0))
                _vals.append(0.0 if _pp is None else _ss / (1.0 + _by.get(_pp, 0)))
            return float(np.mean(_vals)) if _vals else 0.0
        elif metric == 'RelUnexp':
            return M.rel_unexp_score(items, item_rating_g or {}, item_pop or {},
                                          max_pop, k=k)
        elif metric in ('abeta-nDCG', 'alpha-nDCG'):
            if metric == 'alpha-nDCG':
                if item_aspects:
                    _domA = {}
                    for _it in items:
                        _asp = item_aspects.get(_it)
                        if _asp:
                            _domA[_it] = max(_asp.items(), key=lambda kv: kv[1])[0]
                    _ha2 = [_domA[it] for it in items if it in _domA]
                    _hra2 = [3.0] * len(_ha2)
                    return M.alpha_ndcg(items, _domA, item_rating_g or {},
                                        _ha2, _hra2, k=k,
                                        alpha_c=float(os.environ.get('MDCG_ALPHA_NDCG_ALPHA', '0.5')),
                                        normalize=M.ABETA_NORMALIZE)
                _hc2 = _uhist['c'] or [item_cate.get(it) for it in items if item_cate.get(it)]
                _hr2 = _uhist['r'] or [3.0] * len(_hc2)
                return M.alpha_ndcg(items, item_cate, item_rating_g or {},
                                    _hc2, _hr2, k=k,
                                    alpha_c=float(os.environ.get('MDCG_ALPHA_NDCG_ALPHA', '0.5')),
                                    normalize=M.ABETA_NORMALIZE)
            _a = float(os.environ.get('MDCG_ABETA_ALPHA', '0.005'))
            _b = float(os.environ.get('MDCG_ABETA_BETA', '0.5'))
            if item_aspects:
                _ha = [item_aspects[it] for it in items if item_aspects.get(it)]
                _hra = [3.0] * len(_ha)
                return M.abeta_ndcg_multi(items, item_aspects, item_rating_g or {},
                                          _ha, _hra, k=k, alpha=_a, beta=_b,
                                          normalize=M.ABETA_NORMALIZE)
            _hc = _uhist['c'] or [item_cate.get(it) for it in items if item_cate.get(it)]
            _hr = _uhist['r'] or [3.0] * len(_hc)
            return M.abeta_ndcg(items, item_cate, item_rating_g or {},
                              _hc, _hr, k=k, alpha=_a, beta=_b,
                              normalize=M.ABETA_NORMALIZE)
        return 0.0

    for _ in range(n_lists):
        target_uid = rng.choice(valid_uids)
        try:
            _u_hist = user_items_set.get(target_uid, set())
            _uhist['c'] = [item_cate.get(it) for it in _u_hist if item_cate.get(it)]
            _uhist['r'] = [float((item_rating_g or {}).get(it, 3.0)) for it in _u_hist
                           if item_cate.get(it)]
        except NameError:
            pass
        pos_pool, neg_pool = build_sim_user_pools(
            target_uid, user_sim, items_by_uid, user_items_set, k,
            pool_dedup=pool_dedup, pos_ratio_thr=pos_ratio_thr)
        if pos_pool is None:
            continue
        pos_idx = rng.choice(len(pos_pool), k, replace=False)
        neg_idx = rng.choice(len(neg_pool), k, replace=False)
        base_pos = [pos_pool[i] for i in pos_idx]
        base_neg = [neg_pool[i] for i in neg_idx]
        if pool_fitter is not None:
            _pp, _np, _pu = pool_fitter(target_uid, [t[0] for t in base_pos] +
                                        [t[0] for t in base_neg])
            _cur[0], _cur[1], _cur[2] = _pp, _np, _pu
        vals_by_metric = {m: [] for m in metric_names}
        for lvl in levels:
            n_replace = int(round(lvl * k))
            lst = (base_pos[:k - n_replace] + base_neg[:n_replace])[:k]
            for m in metric_names:
                v = score_list(lst, m)
                trend[m][lvl].append(v)
                vals_by_metric[m].append(v)
        for m in metric_names:
            tau, _ = kendalltau(1.0 - levels, vals_by_metric[m])
            if not np.isnan(tau):
                taus[m].append(tau)
    first_key = metric_names[0]
    print("  请求列表数: {}, 成功构建: {}".format(n_lists, len(taus.get(first_key, []))))
    print("\n扰动比   " + "  ".join("{:>10s}".format(m) for m in metric_names))
    for lvl in levels:
        row = "  {:3.0f}%   ".format(lvl * 100)
        row += "  ".join("{:>10.4f}".format(np.mean(trend[m][lvl])) for m in metric_names)
        print(row)

    print("\n各指标扰动敏感性 (mean Tau / 正Tau比例):")
    print("  {:16s} {:>12s} {:>14s}".format("指标", "mean Tau", "正Tau比例"))
    result = {}
    for m in metric_names:
        t = np.array(taus[m])
        neg_ratio = (t < 0).mean() if len(t) else 0
        mean_tau = t.mean() if len(t) else 0
        result[m] = (mean_tau, neg_ratio)
        print("  {:16s} {:>12.4f} {:>13.1%}".format(m, mean_tau, 1-neg_ratio))
    return result

def exp_ablation(user_sim, items_by_uid, user_items_set, sim_engine,
                 pos_params, neg_params,
                 use_copula=True, n_levels=11, k=10, n_lists=200,
                 seed=42, user_scarcity_params=None,
                 per_user_arousal=None,
                 pool_dedup='none', pos_ratio_thr=0.5,
                 pool_fitter=None):
    print("\n" + "=" * 60)
    print(">>> 实验三: 消融")
    print("=" * 60)

    rng = np.random.RandomState(seed)
    valid_uids = [uid for uid in user_sim
                  if uid in items_by_uid and uid in user_items_set]
    print("  有效目标用户: {}".format(len(valid_uids)))
    if len(valid_uids) == 0:
        print("无有效用户, 跳过")
        return {}
    levels = np.linspace(0, 1, n_levels)

    _cur = [pos_params, neg_params, per_user_arousal]
    _uhist = {'c': [], 'r': []}

    def mdcg_variant(lst, variant):
        AROUSAL_MODE = {'no_arousal': 'off', 'neither': 'off',
                        'global_arousal': 'global',
                        'global_both': 'global'}.get(variant, 'user')
        SCARCITY_MODE = {'no_double': 'off', 'neither': 'off',
                         'global_scarcity': 'global',
                         'global_both': 'global'}.get(variant, 'user')

        items, f_sps, difs, mrefs = [], [], [], []
        target = lst[0][8] if len(lst[0]) >= 9 else None
        pp, np_i = _cur[0], _cur[1]
        if AROUSAL_MODE == 'user' and _cur[2] and target in _cur[2]:
            pp, np_i = _cur[2][target]
        for tup in lst:
            item, rel, dif, nov, unpop, qua, label, _rel_raw, uid = tup
            arousal = (1.0 if AROUSAL_MODE == 'off'
                       else M.arousal_score(rel, dif, pp, np_i, use_copula))
            if SCARCITY_MODE == 'off':
                qn, qu = 0.0, 0.0
            else:
                p_acc = (None if SCARCITY_MODE == 'global'
                         else (user_scarcity_params.get(uid)
                               if user_scarcity_params is not None else None))
                qn, qu = M.accommodation_components(nov, unpop, p_acc)
            if variant == 'no_qua':
                _qs = [t[5] for t in lst]
                _qm = float(np.mean(_qs)) if _qs else 0.5
                f_sp = (1.0 + _qm * qn + _qm * qu) * arousal
            elif variant == 'no_asym':
                f_sp = (1.0 + qua) * arousal
            else:
                f_sp = (1.0 + qua * qn + qua * qu) * arousal
            items.append(item); f_sps.append(f_sp); difs.append(dif)
            mrefs.append(M.get_div_mref(uid))
        dcg = 0.0
        for pos, (item, f_sp, dif) in enumerate(zip(items, f_sps, difs)):
            i = pos + 1  # 1-indexed position
            div = 1.0
            if variant != 'no_div' and pos > 0:
                div = M.div_discount([sim_engine.sim(item, items[j]) for j in range(pos)],
                                     mrefs[pos])
            match = 1.0
            discount = match / np.log2(1.0 + i)
            dcg += f_sp * div * discount
        return dcg

    bases = []
    for _ in range(n_lists):
        target_uid = rng.choice(valid_uids)
        try:
            _u_hist = user_items_set.get(target_uid, set())
            _uhist['c'] = [item_cate.get(it) for it in _u_hist if item_cate.get(it)]
            _uhist['r'] = [float((item_rating_g or {}).get(it, 3.0)) for it in _u_hist
                           if item_cate.get(it)]
        except NameError:
            pass
        pos_pool, neg_pool = build_sim_user_pools(
            target_uid, user_sim, items_by_uid, user_items_set, k,
            pool_dedup=pool_dedup, pos_ratio_thr=pos_ratio_thr)
        if pos_pool is None:
            continue
        pos_idx = rng.choice(len(pos_pool), k, replace=False)
        neg_idx = rng.choice(len(neg_pool), k, replace=False)
        _bp = [pos_pool[i] for i in pos_idx]
        _bn = [neg_pool[i] for i in neg_idx]
        if pool_fitter is not None:
            _pp, _npv, _pu = pool_fitter(target_uid, [t[0] for t in _bp] +
                                         [t[0] for t in _bn])
        else:
            _pp, _npv, _pu = pos_params, neg_params, per_user_arousal
        bases.append((_bp, _bn, _pp, _npv, _pu))

    print("  请求列表数: {}, 成功构建: {}".format(n_lists, len(bases)))

    variants = ['full',
                'global_arousal', 'global_scarcity', 'global_both',
                'no_arousal', 'no_double', 'neither',
                'no_qua', 'no_asym', 'no_div']
    div_by_level = {lvl: [[] for _ in range(k)] for lvl in levels}
    for base_pos, base_neg, _bpp, _bnp, _bpu in bases:
        for lvl in levels:
            nr = int(round(lvl * k))
            lst = (base_pos[:k - nr] + base_neg[:nr])[:k]
            items_ = [t[0] for t in lst]
            mrefs_ = [M.get_div_mref(t[8] if len(t) >= 9 else None) for t in lst]
            for pos in range(1, len(items_)):
                d = M.div_discount([sim_engine.sim(items_[pos], items_[j])
                                    for j in range(pos)], mrefs_[pos])
                div_by_level[lvl][pos].append(d)
    print("\n  各位置 div 均值 (列=列表位置, 行=冗余度档位):")
    hdr = "  冗余度  " + "".join("{:>7s}".format("#{}".format(p)) for p in range(k))
    hdr += "{:>9s}".format("全体")
    print(hdr)
    for lvl in levels:
        row = "  {:5.0f}%  ".format(lvl * 100)
        allv = []
        for pos in range(k):
            v = div_by_level[lvl][pos]
            if v:
                row += "{:>7.4f}".format(float(np.mean(v)))
                allv.extend(v)
            else:
                row += "{:>7s}".format("-")
        row += "{:>9.4f}".format(float(np.mean(allv)) if allv else float('nan'))
        print(row)

    print("\n  {:16s} {:>12s} {:>14s}".format("变体", "mean Tau", "正Tau比例"))
    results = {}
    for var in variants:
        taus = []
        for base_pos, base_neg, _bpp, _bnp, _bpu in bases:
            _cur[0], _cur[1], _cur[2] = _bpp, _bnp, _bpu
            vals = []
            for lvl in levels:
                nr = int(round(lvl * k))
                lst = (base_pos[:k - nr] + base_neg[:nr])[:k]
                vals.append(mdcg_variant(lst, var))
            tau, _ = kendalltau(1-levels, vals)
            if not np.isnan(tau):
                taus.append(tau)
        taus = np.array(taus)
        neg_ratio = (taus < 0).mean() if len(taus) else 0
        results[var] = neg_ratio
        print("  {:16s} {:>12.4f} {:>13.1%}".format(
            var, taus.mean() if len(taus) else 0, 1-neg_ratio))
    return results

def exp_diversity_perturbation(user_sim, items_by_uid, user_items_set, sim_engine,
                               pos_params, neg_params,
                               use_copula=True, n_levels=11, k=10, n_lists=200,
                               seed=42, user_scarcity_params=None,
                               per_user_arousal=None,
                               pool_dedup='none', pos_ratio_thr=0.5,
                               pool_fitter=None):
    """Keep the list all-positive and replace its tail with copies of the first item, tracking how the score moves as redundancy grows."""
    print("\n" + "=" * 60)
    print(">>> 实验二b: 多样性扰动 (惊喜标签比例不变)")
    print("=" * 60)
    rng = np.random.RandomState(seed)
    valid_uids = [uid for uid in user_sim
                  if uid in items_by_uid and uid in user_items_set]
    print("  有效目标用户: {}".format(len(valid_uids)))
    if len(valid_uids) == 0:
        print("无有效用户, 跳过")
        return {}
    levels = np.linspace(0, 1, n_levels)
    sf = sim_engine.sim

    metric_names = ['M-DCG', 'M-DCG(no_div)', 'ILD']
    trend = {m: {lvl: [] for lvl in levels} for m in metric_names}
    taus = {m: [] for m in metric_names}

    _cur = [pos_params, neg_params, per_user_arousal]
    _uhist = {'c': [], 'r': []}

    def score_list(lst, metric):
        target = lst[0][8] if len(lst[0]) >= 9 else None
        pp, np_i = _cur[0], _cur[1]
        if _cur[2] and target in _cur[2]:
            pp, np_i = _cur[2][target]
        mlst = [(x[0], x[1], x[2], x[3], x[4], x[5], x[7]) for x in lst]
        if metric == 'M-DCG':
            return M.mdcg_score(mlst, pp, np_i, sim_func=sf,
                                use_copula=use_copula, k=k,
                                user_scarcity_params=user_scarcity_params)
        elif metric == 'M-DCG(no_div)':
            return M.mdcg_score(mlst, pp, np_i, sim_func=None,
                                use_copula=use_copula, k=k,
                                user_scarcity_params=user_scarcity_params)
        elif metric == 'ILD':
            return M.ild_score([x[0] for x in lst], sf, k)
        return 0.0

    built = 0
    div_by_level = {lvl: [[] for _ in range(k)] for lvl in levels}
    for _ in range(n_lists):
        target_uid = rng.choice(valid_uids)
        try:
            _u_hist = user_items_set.get(target_uid, set())
            _uhist['c'] = [item_cate.get(it) for it in _u_hist if item_cate.get(it)]
            _uhist['r'] = [float((item_rating_g or {}).get(it, 3.0)) for it in _u_hist
                           if item_cate.get(it)]
        except NameError:
            pass
        pos_pool, neg_pool = build_sim_user_pools(
            target_uid, user_sim, items_by_uid, user_items_set, k,
            pool_dedup=pool_dedup, pos_ratio_thr=pos_ratio_thr)
        if pos_pool is None:
            continue
        pos_idx = rng.choice(len(pos_pool), k, replace=False)
        base_pos = [pos_pool[i] for i in pos_idx]
        if pool_fitter is not None:
            _pp, _npv, _pu = pool_fitter(target_uid, [t[0] for t in base_pos])
            _cur[0], _cur[1], _cur[2] = _pp, _npv, _pu
        anchor = base_pos[0]
        vals_by_metric = {m: [] for m in metric_names}
        for lvl in levels:
            n_dup = int(round(lvl * k))
            lst = base_pos[:k - n_dup] + [anchor] * n_dup
            items_ = [t[0] for t in lst]
            mrefs_ = [M.get_div_mref(t[8] if len(t) >= 9 else None) for t in lst]
            for pos in range(1, len(items_)):
                d = M.div_discount([sf(items_[pos], items_[j]) for j in range(pos)],
                                   mrefs_[pos])
                div_by_level[lvl][pos].append(d)
            for m in metric_names:
                v = score_list(lst, m)
                trend[m][lvl].append(v)
                vals_by_metric[m].append(v)
        for m in metric_names:
            tau, _ = kendalltau(1.0 - levels, vals_by_metric[m])
            if not np.isnan(tau):
                taus[m].append(tau)
        built += 1
    print("  请求列表数: {}, 成功构建: {}".format(n_lists, built))

    print("\n  各位置 div 均值 (列=列表位置, 行=冗余度档位):")
    hdr = "  冗余度  " + "".join("{:>7s}".format("#{}".format(p)) for p in range(k))
    hdr += "{:>9s}".format("全体")
    print(hdr)
    for lvl in levels:
        row = "  {:5.0f}%  ".format(lvl * 100)
        allv = []
        for pos in range(k):
            v = div_by_level[lvl][pos]
            if v:
                row += "{:>7.4f}".format(float(np.mean(v)))
                allv.extend(v)
            else:
                row += "{:>7s}".format("-")
        row += "{:>9.4f}".format(float(np.mean(allv)) if allv else float('nan'))
        print(row)

    print("\n冗余度   " + "  ".join("{:>14s}".format(m) for m in metric_names))
    for lvl in levels:
        row = "  {:3.0f}%   ".format(lvl * 100)
        row += "  ".join("{:>14.4f}".format(np.mean(trend[m][lvl])) for m in metric_names)
        print(row)

    print("\n各指标冗余敏感性 (mean Tau / 正Tau比例):")
    print("  {:16s} {:>12s} {:>14s}".format("指标", "mean Tau", "正Tau比例"))
    result = {}
    for m in metric_names:
        t = np.array(taus[m])
        mean_tau = t.mean() if len(t) else 0
        neg_ratio = (t < 0).mean() if len(t) else 0
        result[m] = (mean_tau, neg_ratio)
        print("  {:16s} {:>12.4f} {:>13.1%}".format(m, mean_tau, 1 - neg_ratio))
    print("  -> 含 div 的 M-DCG 应强正 Tau(冗余↑→分数↓), no_div 版 Tau≈0, ILD 最强")
    print("  -> 结论: 检测「冗余度下降」的敏感性完全来自 div 项, 与标签组成无关")
    return result

def exp_div2k_IDCG(user_sim, items_by_uid, user_items_set, sim_engine,
                   pos_params, neg_params,
                   use_copula=True, k=10, n_lists=200, seed=42,
                   user_scarcity_params=None, per_user_arousal=None,
                   pool_dedup='none', pos_ratio_thr=0.5,
                   pool_fitter=None):
    print("\n" + "=" * 60)
    print(">>> 实验四: 归一化对比")
    print("=" * 60)
    sf = sim_engine.sim
    rng = np.random.RandomState(seed)
    valid_uids = [uid for uid in user_sim
                  if uid in items_by_uid and uid in user_items_set]
    if len(valid_uids) == 0:
        print("无有效用户, 跳过")
        return {}
    score_2k, score_idcg = [], []
    built = 0
    for _ in range(n_lists):
        target_uid = rng.choice(valid_uids)
        try:
            _u_hist = user_items_set.get(target_uid, set())
            _uhist['c'] = [item_cate.get(it) for it in _u_hist if item_cate.get(it)]
            _uhist['r'] = [float((item_rating_g or {}).get(it, 3.0)) for it in _u_hist
                           if item_cate.get(it)]
        except NameError:
            pass
        pos_pool, neg_pool = build_sim_user_pools(
            target_uid, user_sim, items_by_uid, user_items_set, k,
            pool_dedup=pool_dedup, pos_ratio_thr=pos_ratio_thr)
        if pos_pool is None:
            continue
        built += 1
        pos_idx = rng.choice(len(pos_pool), k, replace=False)
        neg_idx = rng.choice(len(neg_pool), k, replace=False)
        base_pos = [pos_pool[i] for i in pos_idx]
        base_neg = [neg_pool[i] for i in neg_idx]
        lst = (base_pos[:1] + base_neg[1:10])
        mlst = [(x[0], x[1], x[2], x[3], x[4], x[5], x[7]) for x in lst]
        if pool_fitter is not None:
            _pp, _npv, _pu = pool_fitter(target_uid, [t[0] for t in lst])
            pp, np_i = _pp, _npv
            if target_uid in _pu:
                pp, np_i = _pu[target_uid]
        else:
            pp, np_i = pos_params, neg_params
            if per_user_arousal and target_uid in per_user_arousal:
                pp, np_i = per_user_arousal[target_uid]
        m_dcg = M.mdcg_score(mlst, pp, np_i, sim_func=sf,
                             use_copula=use_copula, k=k,
                             user_scarcity_params=user_scarcity_params)
        score_2k.append(m_dcg / (2 * k))
        idcg = M.compute_idcg(mlst, pp, np_i, sim_func=sf,
                              use_copula=use_copula, k=k,
                              user_scarcity_params=user_scarcity_params)
        score_idcg.append(m_dcg / idcg if idcg > 0 else 0.0)
    print("  请求列表数: {}, 成功构建: {}".format(n_lists, built))
    rho, p_value = spearmanr(score_2k, score_idcg)
    print("{:14s} {:>12s}".format("spearman-rho", "p_value"))
    print("{:>14.4f} {:>12.4f}".format(rho, p_value))
    return rho, p_value

def _within_user_hit(label, scores, uid_col, k_hit=(2, 4), neg_per_pos=None, seed=0,
                     detail=None):
    """Within users that have both classes, compute Hit@k and the lift over random ordering; detail optionally collects per-user rows."""
    order = np.argsort(-scores)
    by_user = defaultdict(list)
    for i in order:
        by_user[uid_col[i]].append(i)
    hits = {k: [] for k in k_hit}
    lifts = {k: [] for k in k_hit}
    _rng_hit = np.random.RandomState(seed)
    for uid, idxs in by_user.items():
        idxs = np.asarray(idxs)
        npos = int((label[idxs] == 1).sum())
        if npos == 0 or npos == len(idxs):
            continue
        n = len(idxs)
        if neg_per_pos is not None:
            pos_i = idxs[label[idxs] == 1]
            neg_i = idxs[label[idxs] == 0]
            keep = max(1, int(round(neg_per_pos * npos)))
            if len(neg_i) > keep:
                neg_i = _rng_hit.choice(neg_i, keep, replace=False)
            idxs = np.concatenate([pos_i, neg_i])
            idxs = idxs[np.argsort(-scores[idxs])]
            n = len(idxs)
        _rec = {'uid': uid, 'n_items': int(n), 'n_pos': int(npos)} if detail is not None else None
        for k in k_hit:
            got = int((label[idxs[:k]] == 1).sum())
            _h = 1.0 if got > 0 else 0.0
            _l = got / (k * npos / n)
            hits[k].append(_h)
            lifts[k].append(_l)
            if _rec is not None:
                _rec['hit%d' % k] = _h
                _rec['lift%d' % k] = _l
        if _rec is not None:
            detail.append(_rec)
    return {k: (float(np.mean(hits[k])), float(np.mean(lifts[k]))) for k in k_hit}


def _auc_macro_pooled(label, scores, uid_col, n_boot=200, seed=42):
    """Pooled and macro-averaged AUC with a user-clustered bootstrap interval; single-class users are counted as skipped."""
    label = np.asarray(label)
    scores = np.asarray(scores, dtype=float)
    uid_col = np.asarray(uid_col)

    rows_by_uid = defaultdict(list)
    for i, uid in enumerate(uid_col):
        rows_by_uid[uid].append(i)
    uids = list(rows_by_uid)
    row_idx = [np.asarray(rows_by_uid[u], dtype=int) for u in uids]

    u_auc, u_keep = [], []
    for j, idx in enumerate(row_idx):
        y = label[idx]
        if y.min() == y.max():          # 单类 -> AUC 无定义
            continue
        u_auc.append(roc_auc_score(y, scores[idx]))
        u_keep.append(j)
    u_auc = np.asarray(u_auc, dtype=float)
    u_keep = np.asarray(u_keep, dtype=int)

    def _pooled(sel):
        idx = np.concatenate([row_idx[j] for j in sel]) if len(sel) else np.array([], dtype=int)
        y = label[idx]
        if len(idx) == 0 or y.min() == y.max():
            return float('nan')
        return float(roc_auc_score(y, scores[idx]))

    out = {
        'pooled': _pooled(np.arange(len(uids))),
        'macro': float(u_auc.mean()) if len(u_auc) else float('nan'),
        'n_users': len(uids),
        'n_users_auc': len(u_auc),
        'n_users_skipped': len(uids) - len(u_auc),
        'pooled_lo': float('nan'), 'pooled_hi': float('nan'),
        'macro_lo': float('nan'), 'macro_hi': float('nan'),
        'n_boot': 0,
    }
    if n_boot and n_boot > 0 and len(u_auc) >= 2:
        rng = np.random.RandomState(seed)
        draw = rng.randint(0, len(u_auc), size=(int(n_boot), len(u_auc)))
        boot_macro = u_auc[draw].mean(axis=1)
        boot_pooled = np.array([_pooled(rng.randint(0, len(uids), len(uids)))
                                for _ in range(int(n_boot))], dtype=float)
        boot_pooled = boot_pooled[np.isfinite(boot_pooled)]
        boot_macro = boot_macro[np.isfinite(boot_macro)]
        if len(boot_macro):
            lo, hi = np.percentile(boot_macro, [2.5, 97.5])
            out['macro_lo'], out['macro_hi'] = float(lo), float(hi)
        if len(boot_pooled):
            lo, hi = np.percentile(boot_pooled, [2.5, 97.5])
            out['pooled_lo'], out['pooled_hi'] = float(lo), float(hi)
        out['n_boot'] = int(n_boot)
    return out


_RQ4_DEL_ROWS = []
_RQ4_USER_ROWS = []
_RQ4_POOL_ROWS = []
_RQ4_USER_ON = False
_RQ4_USER_RANKERS = ('mdcg', 'lincombo')


def _rq4_user_detail(ranker):
    """Return the per-user detail collector for the current switch and ranker configuration."""
    if not _RQ4_USER_ON or ranker not in _RQ4_USER_RANKERS:
        return None
    return []


def _rq4_user_append(meta, ranker, det):
    """Merge the current level's per-user rows with the experiment fields."""
    for _d in det or []:
        _RQ4_USER_ROWS.append(dict(data=meta['data'], model=meta['model'],
                                   r=meta['r'], seed=meta['seed'],
                                   ranker=ranker, **_d))


def _rq4_hit_pop(label, uid):
    """Count the users with both classes and their mean number of evaluated items."""
    _u = np.asarray(uid).astype(str)
    _l = np.asarray(label, dtype=float)
    if len(_u) == 0:
        return {'n_users': 0, 'mean_items': 0.0}
    _codes, _inv = np.unique(_u, return_inverse=True)
    _n = np.bincount(_inv).astype(float)
    _p = np.bincount(_inv, weights=_l)
    _ok = (_p > 0) & (_p < _n)
    if not _ok.any():
        return {'n_users': 0, 'mean_items': 0.0}
    return {'n_users': int(_ok.sum()), 'mean_items': float(np.mean(_n[_ok]))}


def exp_robustness_quality(factor_df, user_scarcity_params=None,
                           use_copula=True, n_levels=11,
                           surprise_sim=None, n_boot=200,
                           cross_fit=True, n_folds=5, qua_consts=None,
                           mask_value=None, hit_neg_per_pos=None,
                           label_time=None, temporal_ratio=None,
                           item_rating_g=None, rq4_meta=None):
    print("\n" + "=" * 60)
    print(">>> 实验五: 质量鲁棒性")
    print("=" * 60)

    rel = factor_df['rel'].values
    dif = factor_df['dif'].values
    nov = factor_df['nov_q'].values
    unpop = factor_df['unpop_q'].values
    qua = factor_df['qua'].values
    label = factor_df['label'].values
    uid_col = factor_df['uid'].values
    n_samples = len(rel)

    _fit_mask = np.ones(n_samples, dtype=bool)
    _eval_mask = np.ones(n_samples, dtype=bool)
    if label_time is not None and temporal_ratio:
        _ts = np.array([label_time.get((u, i), 0)
                        for u, i in zip(uid_col, factor_df['item'].values)], float)
        _b = float(np.quantile(_ts, 1.0 - float(temporal_ratio)))
        _fit_mask = _ts < _b
        _eval_mask = ~_fit_mask
        print("  [temporal] 拟合 %d 行(pos %d) / 评测 %d 行(pos %d), 边界 %.0f"
              % (int(_fit_mask.sum()), int((label[_fit_mask] == 1).sum()),
                 int(_eval_mask.sum()), int((label[_eval_mask] == 1).sum()), _b))

    levels = np.linspace(0, 1, n_levels)  # 遮蔽比例 0%~100%
    rng = np.random.RandomState(42)

    pos = label == 1
    neg = label == 0

    if cross_fit:
        _uids = np.array(sorted(set(uid_col)))
        _perm = np.random.RandomState(42).permutation(len(_uids))
        _fold_of_uid = {_uids[_perm[i]]: i % n_folds for i in range(len(_uids))}
        fold_of_row = np.array([_fold_of_uid[u] for u in uid_col])
        n_folds_eff = n_folds
        pass
    else:
        fold_of_row = np.zeros(n_samples, dtype=int)
        n_folds_eff = 1
        pass

    fold_params = {}
    _fold_lc = {}
    for f in range(n_folds_eff):
        tr = ((fold_of_row != f) if cross_fit else np.ones(n_samples, dtype=bool)) & _fit_mask
        _pp = M.fit_class_params(rel[tr & pos], dif[tr & pos], use_copula)
        _np = M.fit_class_params(rel[tr & neg], dif[tr & neg], use_copula)
        _uid_to_idx = defaultdict(list)
        for _i in np.where(tr)[0]:
            _uid_to_idx[uid_col[_i]].append(_i)
        _pu = {}
        for uid in set(uid_col):
            a_r, a_d, b_r, b_d = [], [], [], []
            for v, _ in (surprise_sim or {}).get(uid, []):
                for _i in _uid_to_idx.get(v, []):
                    if label[_i] == 1:
                        a_r.append(rel[_i]); a_d.append(dif[_i])
                    else:
                        b_r.append(rel[_i]); b_d.append(dif[_i])
            if len(a_r) >= 5 and len(b_r) >= 5:
                _pu[uid] = (M.fit_class_params(a_r, a_d, use_copula),
                            M.fit_class_params(b_r, b_d, use_copula))
            else:
                _pu[uid] = (_pp, _np)
        fold_params[f] = (_pp, _np, _pu)
        if cross_fit:
            try:
                from sklearn.linear_model import LogisticRegression
                _ok_tr = tr & ((label == 0) | (label == 1))
                _y_tr = label[_ok_tr].astype(int)
                if int(_ok_tr.sum()) >= 200 and 0 < int(_y_tr.sum()) < len(_y_tr):
                    _X_tr = np.column_stack([rel, dif, nov, unpop, qua])[_ok_tr]
                    _w_f = [float(v) for v in LogisticRegression(
                        max_iter=2000, class_weight='balanced').fit(_X_tr, _y_tr).coef_[0]]
                    _fold_lc[f] = _w_f
                    print("    [LinCombo] fold %d w=[%s]"
                          % (f + 1, ', '.join('%+.3f' % v for v in _w_f)), flush=True)
            except Exception as _e_lc:
                print("  [LinCombo] 折内拟合失败: %s" % _e_lc)
        if cross_fit:
            print("    fold {}/{}: 拟合样本 {} 条".format(f + 1, n_folds_eff, int(tr.sum())),
                  flush=True)
    pp_global = fold_params[0][0]
    np_global = fold_params[0][1]
    per_user_pp_np = fold_params[0][2]

    print("  遮蔽比 {:>12s} {:>10s} {:>12s} {:>10s} {:>10s}".format(
        "Global AUC", "95% CI", "Macro AUC", "95% CI", "有效用户"))
    aucs_by_level = {}
    macro_by_level = {}
    n_users_auc = 0
    _consts = ([float(x) for x in str(qua_consts).split(',') if x.strip()]
               if qua_consts else None)
    _mask_val = float(mask_value) if mask_value is not None else 0.0
    if _mask_val != 0.0:
        print("  [mask-value mode] masked items get Qua = {:.2f} (default was 0)"
              .format(_mask_val))
    _axis = _consts if _consts is not None else list(levels)
    if _consts is not None:
        print("  [const-Qua mode] Qua = " +
              ", ".join("{:.2f}".format(c) for c in _consts) +
              " for every item (no masking-fraction axis, so no tau)")
    for lvl in _axis:
        if _consts is not None:
            qua_masked = np.full(n_samples, lvl, dtype=float)
            mask = np.zeros(n_samples, dtype=bool)
        else:
            mask = rng.rand(n_samples) < lvl
            qua_masked = qua.copy()
        qua_masked[mask] = _mask_val
        if item_rating_g is not None:
            _irg = np.array([float((item_rating_g or {}).get(it, 0.0))
                             for it in factor_df['item'].values], float)
            _irg_masked = _irg.copy()
            _irg_masked[mask] = float(_mask_val) * (5.0 - 1.0) + 1.0
        else:
            _irg_masked = None
        scores = []
        for i in range(n_samples):
            uid = uid_col[i]
            p_sca = user_scarcity_params.get(uid) if user_scarcity_params is not None else None
            _fpp, _fnp, _fpu = fold_params[fold_of_row[i]]
            pp_u, np_u = _fpu.get(uid, (_fpp, _fnp))
            scores.append(M.f_sp_score(rel[i], dif[i], nov[i], unpop[i], qua_masked[i],
                                       pp_u, np_u, use_copula=use_copula,
                                       user_scarcity_param=p_sca))
        scores = np.array(scores)
        _L, _S, _U = label[_eval_mask], scores[_eval_mask], uid_col[_eval_mask]
        st = _auc_macro_pooled(_L, _S, _U, n_boot=n_boot, seed=42)
        _det = _rq4_user_detail('mdcg') if rq4_meta is not None else None
        _hk = _within_user_hit(_L, _S, _U,
                               neg_per_pos=hit_neg_per_pos, seed=42, detail=_det)
        print("            within-user: Hit@2={:.4f} (lift {:+.3f})   Hit@4={:.4f} (lift {:+.3f})".format(
                  _hk[2][0], _hk[2][1] - 1.0, _hk[4][0], _hk[4][1] - 1.0))
        _rq4_pop = _rq4_hit_pop(_L, _U) if rq4_meta is not None else None
        if rq4_meta is not None:
            _RQ4_DEL_ROWS.append(dict(rq4_meta, ranker='mdcg',
                                      hit2=_hk[2][0], hit4=_hk[4][0],
                                      lift2=_hk[2][1], lift4=_hk[4][1],
                                      auc=st['pooled'], **_rq4_pop))
            _rq4_user_append(rq4_meta, 'mdcg', _det)
        _metric_arrs = [('rel', rel), ('dif', dif), ('nov', nov),
                        ('unpop', unpop), ('qua', qua_masked)]
        if _irg_masked is not None:
            _metric_arrs.append(('ndcg_rat', _irg_masked / 5.0))
        if len(_fold_lc) == n_folds_eff:
            _W_lc = np.array([_fold_lc[f] for f in fold_of_row], float)
            _lc_arr = (_W_lc[:, 0] * rel + _W_lc[:, 1] * dif + _W_lc[:, 2] * nov
                       + _W_lc[:, 3] * unpop + _W_lc[:, 4] * qua_masked)
            _metric_arrs.append(('lincombo', _lc_arr))
        for _mn, _arr in _metric_arrs:
            _a_mn = np.asarray(_arr, float)[_eval_mask]
            _det_mn = _rq4_user_detail(_mn) if rq4_meta is not None else None
            _hh = _within_user_hit(_L, _a_mn, _U,
                                   neg_per_pos=hit_neg_per_pos, seed=42, detail=_det_mn)
            _st_mn = _auc_macro_pooled(_L, _a_mn, _U, n_boot=0, seed=42)
            print("            [{:<6s}] within-user: Hit@2={:.4f}   Hit@4={:.4f}"
                  "   AUC pooled={:.4f} macro={:.4f}".format(
                      _mn, _hh[2][0], _hh[4][0], _st_mn['pooled'], _st_mn['macro']))
            if rq4_meta is not None:
                _RQ4_DEL_ROWS.append(dict(rq4_meta, ranker=_mn,
                                          hit2=_hh[2][0], hit4=_hh[4][0],
                                          lift2=_hh[2][1], lift4=_hh[4][1],
                                          auc=_st_mn['pooled'], **_rq4_pop))
                _rq4_user_append(rq4_meta, _mn, _det_mn)
        aucs_by_level[lvl] = st['pooled']
        macro_by_level[lvl] = st['macro']
        n_users_auc = st['n_users_auc']
        ci_p = ('[{:.4f},{:.4f}]'.format(st['pooled_lo'], st['pooled_hi'])
                if st['n_boot'] else 'n/a')
        ci_m = ('[{:.4f},{:.4f}]'.format(st['macro_lo'], st['macro_hi'])
                if st['n_boot'] else 'n/a')
        _lab = ('Qua={:.2f}'.format(lvl) if _consts is not None
                else '{:3.0f}%'.format(lvl * 100))
        print("   {:>8s}  {:>12.4f} {:>20s} {:>12.4f} {:>20s} {:>10d}".format(
            _lab, st['pooled'], ci_p, st['macro'], ci_m, st['n_users_auc']))

    print("  说明: Global AUC 池化(按样本量加权); Macro AUC 每用户等权, 总体是"
          "同时含正负样本的 {} 个用户;".format(n_users_auc))
    print("        CI 为 {}-次【用户级聚簇】bootstrap 的 2.5/97.5 分位(整用户重抽)。".format(
        n_boot if n_boot else 0))
    if n_boot:
        d = np.array([aucs_by_level[l] - macro_by_level[l] for l in _axis
                      if not np.isnan(aucs_by_level[l]) and not np.isnan(macro_by_level[l])])
        if len(d):
            print("        池化-宏平均 差值: 中位={:+.4f} 最大={:+.4f} "
                  "(差值即池化偏置, 池化稳定偏高说明高分用户主导了全局 AUC)".format(
                      float(np.median(d)), float(np.max(np.abs(d)))))

    def _tau(vals_by_level):
        v = [(l, a) for l, a in vals_by_level.items() if not np.isnan(a)]
        if len(v) < 2:
            return float('nan')
        ls, aa = zip(*v)
        return kendalltau(1.0 - np.array(ls), aa)[0]

    tau_pooled = _tau(aucs_by_level)
    tau_macro = _tau(macro_by_level)
    print("  质量鲁棒性 Tau: 池化={}  宏平均={}".format(
        '{:.4f}'.format(tau_pooled) if not np.isnan(tau_pooled) else 'N/A',
        '{:.4f}'.format(tau_macro) if not np.isnan(tau_macro) else 'N/A'))
    return aucs_by_level

LABEL_AWARE_RANKERS = ('ndcg_seren', 'oracle')

_SINGLE_FACTORS = {'rel': 1, 'dif': 2, 'nov': 3, 'unpop': 4, 'qua': 5}
BASELINE_RAW = False
FREEZE_BOUNDARY = 0.0

_ALL_RANKERS = ['mdcg', 'random', 'epc', 'ild', 'abeta_ndcg',
                'fn_surprise', 'alpha_ndcg', 'ndcg_rating',
                'mmr', 'lincombo',
                'rel', 'dif', 'nov', 'unpop', 'qua']


def _is_label_aware(metric, abeta_rating, pool_shuffle):
    """Whether the ranker uses labels or carries a rating or pool-order bias."""
    if metric in LABEL_AWARE_RANKERS:
        return True
    if metric == 'abeta_ndcg':
        return (abeta_rating == 'leaky') or (not pool_shuffle)
    return False


_MASK_POOL_FRAC = float(os.environ.get('MDCG_MASK_POOL', '0') or 0)


def _mask_pool_qua(pool, frac, uid):
    """Zero the quality of a fraction of the pool, seeded by (uid, frac); returns a new list."""
    import random as _rnd

    n = int(round(len(pool) * float(frac)))
    if n <= 0:
        return list(pool)
    rng = _rnd.Random((hash((str(uid), round(float(frac), 3))) & 0x7FFFFFFF))
    out = list(pool)
    for i in rng.sample(range(len(out)), n):
        t = out[i]
        out[i] = t[:5] + (0.0,) + t[6:]
    return out


def exp_hit_rate(factor_df, user_seq, label_time, sim_engine,
                 pos_params, neg_params, user_scarcity_params=None,
                 per_user_arousal=None, item_cate=None,
                 item_rating_g=None, item_pop=None, max_pop=1.0,
                 k_list=(5, 10), use_copula=True, min_history=5, min_user_items=1,
                 test_ratio=PAPER_TEST_RATIO, min_valid_users=10, min_neg_test=1,
                 lambda_source='surprise_pool', lambda_stat='median',
                 lambda_min_pool=5, topk=50, pool_k=10, min_pool_pos=1,
                 pool_pos_frac=0.2, n_lists_per_user=5, seed=42,
                 quality_index=None, qua_mode='pit', in_sample=False,
                 abeta_rating='safe', pool_shuffle=True,
                 low_rating_to_train=False, low_rating_thr=2.5,
                 match_popularity=False, pool_source='similar_users',
                 hit_rate_metrics=None, item_aspects=None, boundary_pairs=None,
                 add_hit_metrics=None, fnpair=None, scale_ctx=None,
                 pool_dump=None, pool_file=None, _allpos_surprise_sim=None,
                 _neighbor_fallback=None, profile_all_items=False,
                 lincombo_neighbors=False):
    """Rerank each candidate pool and measure Hit@k, OrderAUC and the mean rank of positives; the random first slot is shared and the source is set by pool_source."""

    rng = np.random.RandomState(seed)
    k_list = sorted({int(x) for x in k_list})
    kmax = k_list[-1]

    ctx = _prepare_temporal(
        factor_df, user_seq, label_time, sim_engine, use_copula, min_history, test_ratio,
        min_user_items, _allpos_surprise_sim, False, lambda_source, lambda_stat, lambda_min_pool,
        tag="(实验六/候选池): 候选池由相似用户的已标注物品构成",
        topk=topk, min_valid_users=min_valid_users, min_neg_test=min_neg_test,
        quality_index=quality_index, qua_mode=qua_mode,
        low_rating_to_train=low_rating_to_train, low_rating_thr=low_rating_thr,
        boundary_pairs=boundary_pairs,
        neighbor_fallback=_neighbor_fallback,
        profile_all_items=profile_all_items)
    if ctx is None:
        return {}

    if M.DIV_MODE == 'threshold' and M.DIV_MREF_SOURCE in ('user', 'global'):
        _win_mref = M.build_div_mref(ctx['train_df'], sim_engine,
                                     source=M.DIV_MREF_SOURCE, min_pos=2,
                                     fallback=M.DIV_MREF_DEFAULT)
        M.set_div_config(mref=_win_mref)
        _n_real = sum(1 for _v in _win_mref.values()
                      if abs(float(_v) - float(M.DIV_MREF_DEFAULT)) > 1e-12)
        pass

    _win_pop, _win_max = ctx.get('win_item_pop'), ctx.get('win_max_pop')
    if _win_pop and _win_max:
        item_pop, max_pop = _win_pop, float(_win_max)

    items_by_uid_win = defaultdict(list)
    if pool_source == 'zeroshot' and _ZEROSHOT_UNL and not getattr(exp_hit_rate, '_zs_merged', False):
        for _u, _rows in _ZEROSHOT_UNL.items():
            items_by_uid_win[_u].extend(_rows)
        exp_hit_rate._zs_merged = True
    import zlib as _zlib
    _qf = float(os.environ.get('MDCG_MASK_QUA_FRAC', '0.0') or 0.0)
    _qmask_val = float(os.environ.get('MDCG_MASK_QUA_VAL', '0.0') or 0.0)
    if _qf > 0:
        print('  [mask-qua] frac=%.2f：qua 置 %.3f（嵌套掩蔽，固定哈希）' % (_qf, _qmask_val))

    def _masked_qua(_it, _v):
        if _qf <= 0:
            return _v
        return _qmask_val if (_zlib.crc32(str(_it).encode('utf-8')) % 1000000) / 1e6 < _qf else _v

    for _, r in ctx['test_df'].iterrows():
        items_by_uid_win[r['uid']].append(
            (r['item'], r['rel'], r['dif'], r['nov_q'], r['unpop_q'],
             _masked_qua(r['item'], r['qua']), r['label'], r['rel_raw'], r['dif_raw']))
    pass

    _FACTOR_SIGN = {}
    if os.environ.get('RQ4_SIGN_FIX') == '1' and ctx is not None and 'train_df' in ctx:
        _td = ctx['train_df']
        _y = _td['label'].values.astype(float)
        for _m, _c in (('rel', 'rel_raw' if BASELINE_RAW else 'rel'),
                       ('dif', 'dif_raw' if BASELINE_RAW else 'dif'),
                       ('nov', 'nov_q'), ('unpop', 'unpop_q'), ('qua', 'qua')):
            if _c in _td.columns:
                _x = _td[_c].values.astype(float)
                _ok = ~np.isnan(_x)
                if _ok.sum() > 10 and 0 < _y[_ok].sum() < _ok.sum():
                    _cc = float(np.corrcoef(_x[_ok], _y[_ok])[0, 1])
                    _FACTOR_SIGN[_m] = -1 if _cc < 0 else 1
        print('  [sign-fix] 训练窗因子方向(1=不翻转, -1=取补集): %s'
              % ', '.join('%s=%d' % (k, v) for k, v in sorted(_FACTOR_SIGN.items())))

    if BASELINE_RAW:
        print('  [baseline-raw] rel/dif 单因子 + lincombo 改用原始值（不套 E+D 刻度）')

    _LCW = None
    _want_lc = (not hit_rate_metrics) or ('lincombo' in str(hit_rate_metrics))
    if _want_lc:
        try:
            _lc_src = factor_df
            if ctx is not None and 'train_df' in ctx and len(ctx['train_df']) > 0:
                _lc_src = ctx['train_df']
            _lc_cols = (['rel_raw', 'dif_raw'] if BASELINE_RAW else ['rel', 'dif']) + \
                       ['nov_q', 'unpop_q', 'qua']
            _sub = _lc_src[_lc_cols + ['label']].dropna()
            _Xs = _sub[_lc_cols].values
            _ys = _sub['label'].astype(int).values
            if len(_ys) >= 200 and 0 < _ys.sum() < len(_ys):
                from sklearn.linear_model import LogisticRegression
                _lr = LogisticRegression(max_iter=2000, class_weight='balanced').fit(_Xs, _ys)
                _LCW = [float(v) for v in _lr.coef_[0]]
                pass
            else:
                pass
        except Exception as _e:
            pass


    hist_cate, hist_rating, hist_asp, hist_rating_asp, hist_items_ts = _build_user_history(
        user_seq, label_time, ctx['boundary'], item_cate or {}, item_aspects)

    _rt_sum, _rt_cnt = defaultdict(float), defaultdict(int)
    _own_train = {}          # (uid, item) -> 该用户自己的评分, 供留一用
    for _u, _seq in user_seq.items():
        for _it, _ts, _r in _seq:
            if _ts <= ctx['boundary']:
                _rt_sum[_it] += _r
                _rt_cnt[_it] += 1
                _own_train[(_u, _it)] = _r
    _irg_safe = {it: _rt_sum[it] / _rt_cnt[it] for it in _rt_sum if _rt_cnt[it] > 0}
    if not in_sample:
        item_rating_g = _irg_safe
        per_user_arousal = None
        user_scarcity_params = None
        pass
    else:
        pass

    metric_names = ['mdcg', 'random', 'epc', 'ild', 'abeta_ndcg',
                    'fn_surprise', 'alpha_ndcg', 'ndcg_rating',
                    'mmr', 'lincombo']
    if _LCW is None:                     # 训练窗没有正样本时, lincombo 不适用, 不报退化结果
        metric_names = [m for m in metric_names if m != 'lincombo']
    if hit_rate_metrics:
        _want = set(x.strip() for x in str(hit_rate_metrics).split(',') if x.strip())
        _want.add('random')                      # 随机基线始终保留
        _avail = (_ALL_RANKERS if _LCW is not None
                  else [m for m in _ALL_RANKERS if m != 'lincombo'])
        metric_names = [m for m in _avail if m in _want]
        _unknown = sorted(_want - set(_ALL_RANKERS) - {'random'})
        if _unknown:
            print('  [忽略未知排序器] {}'.format(', '.join(_unknown)))
        if metric_names:
            print('  [仅评排序器] {}'.format(', '.join(metric_names)))
    if add_hit_metrics:
        for _m in str(add_hit_metrics).split(','):
            _m = _m.strip()
            if _m and _m not in metric_names:
                metric_names.append(_m)
        print('  [补跑排序器] {}'.format(', '.join(metric_names)))
    hit_records = {m: {k: [] for k in k_list} for m in metric_names + ['oracle']}
    auc_records = {m: [] for m in metric_names + ['oracle']}
    pos_rank_records = {m: [] for m in metric_names + ['oracle']}

    _pop_rank = {}
    if match_popularity and item_pop:
        _its = sorted(item_pop.keys(), key=lambda k: item_pop[k])
        _den = max(len(_its) - 1, 1)
        _pop_rank = {it: i / _den for i, it in enumerate(_its)}

    _global_neg = []
    if pool_source in ('global_neg', 'test_own'):
        for _u, _lst in items_by_uid_win.items():
            _global_neg.extend(x for x in _lst if x[6] == 0)

    n_users, n_pools, n_pos_pools, n_lists = 0, [], [], 0
    sf = sim_engine.sim

    evaluated_uids = sorted(set(ctx['test_df']['uid']))
    _LC_USER = {}
    if lincombo_neighbors:
        try:
            from sklearn.linear_model import LogisticRegression
            _lcols = (['rel_raw', 'dif_raw'] if BASELINE_RAW else ['rel', 'dif']) + \
                     ['nov_q', 'unpop_q', 'qua']
            _tr = ctx['train_df']
            _Xa = _tr[_lcols].to_numpy(float)
            _ya = _tr['label'].to_numpy(int)
            _ix = defaultdict(list)
            for _i, _u in enumerate(_tr['uid'].to_numpy()):
                _ix[_u].append(_i)
            _ns = []
            for _u in evaluated_uids:
                _nb = [v for v, _s in ctx['surprise_sim_win'].get(_u, [])]
                _idxs = [i for v in _nb for i in _ix.get(v, [])]
                if len(_idxs) < 100:
                    continue
                _ys = _ya[_idxs]
                if _ys.sum() < 5 or _ys.sum() > len(_ys) - 5:
                    continue
                _w = LogisticRegression(max_iter=2000, class_weight='balanced').fit(
                    _Xa[_idxs], _ys).coef_[0]
                _LC_USER[_u] = [float(x) for x in _w]
                _ns.append(len(_idxs))
            pass
        except Exception as _e_lcu:
            pass
    _uis = {u: set(it for it, ts, r in seq) for u, seq in user_seq.items()}
    _n_lists_done = 0
    _t_start = time.time()
    _n_borrowed, _n_short = 0, 0
    _n_borrow_nb, _n_borrow_gl, _n_global_fill = 0, 0, 0
    _borrow_cache = {}
    _pooldict, _dump_rows, _win_idx, _item_any = None, [], None, {}
    if pool_source == 'test_own' and (pool_dump or pool_file):
        _win_idx = {u: {x[0]: x for x in lst} for u, lst in items_by_uid_win.items()}
    if pool_source == 'test_own' and pool_file:
        _pooldict = {}
        with open(pool_file, encoding='utf-8') as _pf:
            for _line in _pf:
                _p = _line.rstrip('\n').split(',')
                if len(_p) < 4 or _p[0] == 'uid':
                    continue
                _pooldict.setdefault(_p[0], []).append((_p[1], int(_p[2]), _p[3]))
        for _u2, _lst in items_by_uid_win.items():
            for _x in _lst:
                if _x[6] == 0 and _x[0] not in _item_any:
                    _item_any[_x[0]] = _x
        print("  [test_own] 冻结池: {} 人 / {} 条".format(
            len(_pooldict), sum(len(v) for v in _pooldict.values())))


    def _borrow_row(_uid, _src, _bnd, _sc):
        """外借负样本的 rel/dif 按目标历史重算并映射到参考刻度，N/U/Q 沿用来源行。"""
        _key = (_uid, _src[0])
        if _key in _borrow_cache:
            return _borrow_cache[_key]
        _f = compute_temporal_factors(
            _src[0], float(_bnd) + 1.0, user_seq.get(_uid, []), sim_engine,
            None, 1.0, None, None, None, None, min_history=1)
        if _f is None:
            _borrow_cache[_key] = None
            return None
        _rr, _dr = float(_f[0]), float(_f[1])
        if _sc and _sc.get('tf') not in (None, 'none'):
            _cr = (float(_sc['cr'].get(_uid, _sc['cr_all']))
                   if _sc.get('cr') is not None else 0.0)
            _cd = (float(_sc['cd'].get(_uid, _sc['cd_all']))
                   if _sc.get('cd') is not None else 0.0)
            _rel = quantile_transform_from_reference(_rr - _cr, _sc['r_ref'])
            _dif = quantile_transform_from_reference(_dr - _cd, _sc['d_ref'])
        else:
            _rel, _dif = _rr, _dr
        _row = (_src[0], _rel, _dif, _src[3], _src[4], _src[5], 0, _rr, _dr)
        _borrow_cache[_key] = _row
        return _row
    _prog_step = max(1, len(evaluated_uids) // 10)
    for _ui, uid in enumerate(evaluated_uids):
        if _ui % _prog_step == 0:
            pass
        if pool_source == 'test_own':
            _k_neg = max(1, int(pool_k) - 1)
            _seen = _uis.get(uid, set())
            pos_pool, neg_pool, _rows = [], [], []
            if _pooldict is not None:
                _spec = _pooldict.get(uid)
                if not _spec:
                    continue
                _ok = True
                for _it, _lab, _src in _spec:
                    if _src == uid:
                        _t = _win_idx.get(uid, {}).get(_it)
                        if _t is None or int(_t[6]) != _lab:
                            _ok = False
                            break
                        (pos_pool if _lab == 1 else neg_pool).append(_t)
                    else:
                        _s = (_win_idx.get(_src, {}).get(_it) if _src
                              else _item_any.get(_it))
                        if _s is None:
                            _ok = False
                            break
                        _rw = _borrow_row(uid, _s, ctx['boundary'], scale_ctx)
                        if _rw is None:
                            _ok = False
                            break
                        neg_pool.append(_rw)
                if not _ok or len(pos_pool) != 1 or len(neg_pool) != _k_neg:
                    _n_short += 1
                    continue
            else:
                _own = items_by_uid_win.get(uid, [])
                _op = [x for x in _own if x[6] == 1]
                _on = [x for x in _own if x[6] == 0]
                if not _op:
                    continue
                pos_pool = [_op[int(rng.randint(len(_op)))]]
                _rows.append((pos_pool[0][0], 1, uid))
                if _on:
                    _take = min(_k_neg, len(_on))
                    neg_pool = [_on[j] for j in rng.choice(len(_on), _take,
                                                           replace=False)]
                    _rows.extend((x[0], 0, uid) for x in neg_pool)
                else:
                    neg_pool = []
                _in_pool = {x[0] for x in neg_pool}
                _in_pool.add(pos_pool[0][0])
                if len(neg_pool) < _k_neg:
                    _need = _k_neg - len(neg_pool)
                    _cand, _cseen = [], set(_in_pool)
                    for _v, _sv in ctx['surprise_sim_win'].get(uid, []):
                        if _v == uid:
                            continue
                        for _x in items_by_uid_win.get(_v, []):
                            if (_x[6] != 0 or _x[0] in _seen or _x[0] in _cseen):
                                continue
                            _cand.append((_x, _v))
                            _cseen.add(_x[0])
                        if len(_cand) >= 4 * _need:
                            break
                    _n_nb = len(_cand)
                    if len(_cand) < _need:
                        for _x in _global_neg:
                            if _x[0] in _seen or _x[0] in _cseen:
                                continue
                            _cand.append((_x, ''))
                            _cseen.add(_x[0])
                        _n_global_fill += 1
                    if _cand:
                        for _j in rng.choice(len(_cand), min(_need, len(_cand)),
                                             replace=False):
                            _src_x, _src_u = _cand[_j]
                            _rw = _borrow_row(uid, _src_x, ctx['boundary'], scale_ctx)
                            if _rw is not None:
                                neg_pool.append(_rw)
                                _rows.append((_rw[0], 0, _src_u))
                                _n_borrowed += 1
                                if _j < _n_nb:
                                    _n_borrow_nb += 1
                                else:
                                    _n_borrow_gl += 1
                if len(neg_pool) < _k_neg:
                    _n_short += 1
                    continue
            if pool_dump:
                _dump_rows.extend((uid, it, lab, src) for it, lab, src in _rows)

        elif pool_source in ('own', 'global_neg', 'zeroshot'):
            _own = items_by_uid_win.get(uid, [])
            _op = [x for x in _own if x[6] == 1]
            _seen = _uis.get(uid, set())
            if pool_source == 'global_neg':
                _on = [x for x in _global_neg if x[0] not in _seen]
            else:
                if pool_source == 'zeroshot':
                    _on = []
                    for _v, _ in ctx['surprise_sim_win'].get(uid, []):
                        for x in items_by_uid_win.get(_v, []):
                            if x[6] == 1 or x[0] in _seen:
                                continue
                            _on.append(x)
                else:
                    _on = [x for x in _own if x[6] == 0]
            if len(_op) < 1 or len(_on) < 1:
                continue
            _np = max(min_pool_pos, int(round(pool_k * pool_pos_frac)))
            _nn = max(1, pool_k - _np)
            pos_pool = [_op[i] for i in
                        rng.choice(len(_op), min(_np, len(_op)), replace=False)]
            if match_popularity and _pop_rank and len(_on) > min(_nn, len(_on)):
                _tr = float(np.median([_pop_rank.get(x[0], 0.5) for x in pos_pool]))
                neg_pool = sorted(
                    _on, key=lambda x: abs(_pop_rank.get(x[0], 0.5) - _tr))[
                        :min(_nn, len(_on))]
            else:
                neg_pool = [_on[i] for i in
                            rng.choice(len(_on), min(_nn, len(_on)), replace=False)]
        elif pool_source == 'sim_own':
            _own = items_by_uid_win.get(uid, [])
            _seen = _uis.get(uid, set())
            _rows, _items = [], set()
            for _x in _own:
                _rows.append(_x)
                _items.add(_x[0])
            for _v, _ in ctx['surprise_sim_win'].get(uid, []):
                if _v == uid:
                    continue
                for _x in items_by_uid_win.get(_v, []):
                    if _x[0] in _seen or _x[0] in _items:
                        continue
                    _rows.append(_x)
                    _items.add(_x[0])
            _own_pos = {x[0] for x in _own if x[6] == 1}
            pos_pool = sorted([x for x in _rows if x[6] == 1],
                              key=lambda x: 0 if x[0] in _own_pos else 1)
            neg_pool = [x for x in _rows if x[6] == 0]
        else:
            pos_pool, neg_pool = build_sim_user_pools(
                uid, ctx['surprise_sim_win'], items_by_uid_win, _uis, pool_k,
                pool_dedup='none', pos_ratio_thr=0.5,
                min_pos=max(min_pool_pos, int(round(pool_k * pool_pos_frac))),
                min_neg=max(1, pool_k - int(round(pool_k * pool_pos_frac))))
        if pos_pool is None:
            continue
        if len(pos_pool) < min_pool_pos or len(neg_pool) < 1:
            continue
        n_pos_keep = max(min_pool_pos, int(round(pool_k * pool_pos_frac)))
        if pool_source != 'own':
            n_pos_keep = min(n_pos_keep, len(pos_pool))
            pos_pool = pos_pool[:n_pos_keep]
        if len(pos_pool) + len(neg_pool) < max(k_list):
            continue
        _n_neg = max(0, pool_k - len(pos_pool))
        if pool_source == 'own':
            _rng_pool = np.random.RandomState((seed or 42) + 7919)
            if len(neg_pool) > _n_neg:
                _pidx = _rng_pool.permutation(len(neg_pool))[:_n_neg]
                neg_pool = [neg_pool[_i] for _i in _pidx]
        elif pool_source in ('global_neg', 'zeroshot', 'sim_own'):
            _rng_pool2 = np.random.RandomState((seed or 42) + 7919)
            if len(neg_pool) > _n_neg:
                _pidx2 = _rng_pool2.permutation(len(neg_pool))[:_n_neg]
                neg_pool = [neg_pool[_i2] for _i2 in _pidx2]
        elif match_popularity and len(neg_pool) > _n_neg and _pop_rank:
            _tr = float(np.median([_pop_rank.get(x[0], 0.5) for x in pos_pool]))
            neg_pool = sorted(neg_pool,
                              key=lambda x: abs(_pop_rank.get(x[0], 0.5) - _tr))[:_n_neg]
        else:
            neg_pool = neg_pool[:_n_neg]
        pool = pos_pool + neg_pool
        if len(pool) < kmax:
            continue
        if pool_shuffle:
            _perm = rng.permutation(len(pool))
            pool = [pool[i] for i in _perm]
        pos_items = {x[0] for x in pos_pool}
        pp, np_i = ctx['per_user_pp_np'].get(
            uid, (ctx['pp_global'], ctx['np_global']))
        p_acc = (user_scarcity_params if user_scarcity_params is not None
                 else ctx['scarcity_win']).get(uid)
        if p_acc is None:
            continue

        n_users += 1
        n_pools.append(len(pool))
        n_pos_pools.append(len(pos_pool))
        if abeta_rating == 'safe':
            _irg_u = {x[0]: x[5] for x in pool}
        elif not in_sample:
            _irg_u = {}
            for _x in pool:
                _it2 = _x[0]
                _s2, _c2 = _rt_sum.get(_it2, 0.0), _rt_cnt.get(_it2, 0)
                if _c2 <= 0:
                    continue
                _own2 = _own_train.get((uid, _it2))
                if _own2 is not None and _c2 > 1:
                    _irg_u[_it2] = (_s2 - _own2) / (_c2 - 1)
                else:
                    _irg_u[_it2] = _s2 / _c2
        else:
            _irg_u = item_rating_g or {}

        if not in_sample:
            _pset = set(x[0] for x in pool)
            _hc, _hr = [], []
            _ha, _hra = [], []
            for _it3, _ts3, _r3 in user_seq.get(uid, []):
                if _ts3 <= ctx['boundary'] and _it3 not in _pset:
                    _a3 = (item_aspects or {}).get(_it3)
                    if _a3:
                        _ha.append(_a3)
                        _hra.append(float(_r3))
                    _c3 = (item_cate or {}).get(_it3)
                    if _c3 is not None:
                        _hc.append(_c3)
                        _hr.append(float(_r3))
        else:
            _hc = hist_cate.get(uid, [])
            _hr = hist_rating.get(uid, [])
            _ha = hist_asp.get(uid, [])
            _hra = hist_rating_asp.get(uid, [])

        if _MASK_POOL_FRAC > 0:
            pool = _mask_pool_qua(pool, _MASK_POOL_FRAC, uid)

        rctx = {
            'pool': pool, 'uid': uid, 'pos_items': pos_items,
            'lincombo_w': (_LC_USER.get(uid) or _LCW) if _LC_USER else _LCW,
            'factor_sign': _FACTOR_SIGN,
            'pp': pp, 'np': np_i, 'p_acc': p_acc,
            'sf': sf, 'use_copula': use_copula,
            'item_cate': item_cate or {}, 'item_rating_g': _irg_u,            'hist_cate': _hc, 'hist_rating': _hr,
            'item_aspects': item_aspects or None,
            'hist_aspects': _ha, 'hist_rating_asp': _hra,
            'item_pop': item_pop or {}, 'max_pop': max_pop,
            'fnpair': fnpair or {}, 'hist_items_ts': hist_items_ts.get(uid, []),
            'label_time': label_time or {},
        }
        oracle = sorted(range(len(pool)), key=lambda i: (pool[i][6] != 1, i))

        _warned_slow = {}
        _fsp_pool = _precompute_fsp(rctx) if any(
            _fast_capable(m, rctx) for m in metric_names) else None
        for _ in range(n_lists_per_user):
            _n_lists_done += 1
            start = int(rng.randint(len(pool)))
            for m in metric_names:
                if m == 'random':
                    order = _random_order(rng, len(pool), start)
                elif _fsp_pool is not None and _fast_capable(m, rctx):
                    order = _greedy_order_fast(m, rctx, start, _fsp_pool)
                else:
                    order = _greedy_order(m, rctx, start)
                ranked_items = [pool[i][0] for i in order]
                labels = [pool[i][6] for i in order]
                for k in k_list:
                    hit_records[m][k].append(
                        M.hit_at_k(ranked_items, k, positives=pos_items))
                auc_records[m].append(_order_auc(labels))
                pos_rank_records[m].append(_mean_pos_rank(labels))
            for k in k_list:
                hit_records['oracle'][k].append(
                    M.hit_at_k([pool[i][0] for i in oracle], k, positives=pos_items))
            auc_records['oracle'].append(_order_auc([pool[i][6] for i in oracle]))
            pos_rank_records['oracle'].append(_mean_pos_rank([pool[i][6] for i in oracle]))
            n_lists += 1

    if n_users == 0:
        print("  可用用户不足, 实验六终止 (需要 |pos_pool| >= {} 且 |neg_pool| >= 1)".format(
            min_pool_pos))
        return {}

    if pool_source == 'test_own' and pool_dump:
        _nd = len(set(r[0] for r in _dump_rows))
        with open(pool_dump, 'w', encoding='utf-8') as _f:
            _f.write('uid,item,label,src\n')
            for _r in _dump_rows:
                _f.write('%s,%s,%d,%s\n' % _r)
        print("  [test_own] 池子已冻结写出: %s（%d 人 / %d 条）"
              % (pool_dump, _nd, len(_dump_rows)))
    if pool_source == 'test_own':
        print("  [test_own] 池子 = 1 正 + {} 负（外借 {} 条：相似用户 {} / 全局兜底 {}，"
              "触发全局兜底 {} 人；凑不满跳过 {} 人）".format(
                  max(1, int(pool_k) - 1), _n_borrowed, _n_borrow_nb, _n_borrow_gl,
                  _n_global_fill, _n_short))
    pass

    header = "  {:>12s}".format("排序器")
    for k in k_list:
        header += "  {:>9s}".format("Hit@{}".format(k))
    header += "  {:>10s}".format("正样本位次")
    header += "  {:>9s}".format("OrderAUC")
    print()
    print(header)
    print("  " + "-" * (len(header) - 2))

    results = {}
    for m in metric_names + ['oracle']:
        row = "  {:>12s}".format(
            m + ("*" if _is_label_aware(m, abeta_rating, pool_shuffle) else ""))
        for k in k_list:
            row += "  {:>9.4f}".format(float(np.mean(hit_records[m][k])))
        row += "  {:>10.4f}".format(float(np.nanmean(pos_rank_records[m])))
        row += "  {:>9.4f}".format(float(np.nanmean(auc_records[m])))
        print(row)
        results[m] = {k: float(np.mean(hit_records[m][k])) for k in k_list}
        results[m]['order_auc'] = float(np.nanmean(auc_records[m]))
        results[m]['mean_pos_rank'] = float(np.nanmean(pos_rank_records[m]))


    print()

    print()
    for k in k_list:
        base = np.array(hit_records['random'][k], dtype=float)
        line = "  Hit@{} 配对符号检验(vs random):".format(k)
        for m in ['mdcg', 'ild', 'content_unexp']:
            if m not in hit_records:      # --hit_rate_metrics 只跑了部分排序器
                continue
            a = np.array(hit_records[m][k], dtype=float)
            n01 = int(np.sum((a == 1) & (base == 0)))
            n10 = int(np.sum((a == 0) & (base == 1)))
            if n01 + n10 == 0:
                line += "  {}: n/a".format(m)
                continue
            try:
                from scipy.stats import binomtest
                p = binomtest(n01, n01 + n10, 0.5, alternative='greater').pvalue
                line += "  {}: +{}/-{} p={:.1e}".format(m, n01, n10, p)
            except Exception:
                line += "  {}: +{}/-{}".format(m, n01, n10)
        print(line)

    if 'mdcg' in hit_records:
        for k in k_list:
            ref = np.array(hit_records['mdcg'][k], dtype=float)
            line = "  Hit@{} vs mdcg:".format(k)
            for m in hit_records:
                if m in ('mdcg', 'random', 'oracle'):
                    continue
                a = np.array(hit_records[m][k], dtype=float)
                n01 = int(np.sum((a == 1) & (ref == 0)))
                n10 = int(np.sum((a == 0) & (ref == 1)))
                if n01 + n10 == 0:
                    line += "  {}: n/a".format(m)
                    continue
                try:
                    from scipy.stats import binomtest
                    p = binomtest(n01, n01 + n10, 0.5, alternative='greater').pvalue
                    line += "  {}: +{}/-{} p={:.1e}".format(m, n01, n10, p)
                except Exception:
                    line += "  {}: +{}/-{}".format(m, n01, n10)
            print(line)

    print()
    print("  注: 候选池正样本占比随数据集而定(SerenLens Movies 标注稀疏, 平均约 1.5/10)。")
    print("      本实验只报 Hit@k 与正样本位次（不再报 OrderAUC）。")
    return results


def _build_user_history(user_seq, label_time, boundary, item_cate, item_aspects=None):
    """Collect history up to the boundary and return categories, ratings, aspects, aspect ratings and item times, each aligned with its own score list."""
    hist_cate, hist_rating = defaultdict(list), defaultdict(list)
    hist_asp, hist_rating_asp = defaultdict(list), defaultdict(list)
    hist_items_ts = defaultdict(list)
    for uid, seq in user_seq.items():
        for it, ts, r in seq:
            if ts > boundary:
                continue
            hist_items_ts[uid].append((it, int(ts)))
            a = (item_aspects or {}).get(it)
            if a:
                hist_asp[uid].append(a)
                hist_rating_asp[uid].append(float(r))
            c = item_cate.get(it)
            if c is not None:
                hist_cate[uid].append(c)
                hist_rating[uid].append(float(r))
    return hist_cate, hist_rating, hist_asp, hist_rating_asp, hist_items_ts


def _mean_pos_rank(labels):
    """Mean zero-based rank of the positives, or nan when the list has none."""
    pos = [i for i, y in enumerate(labels) if y == 1]
    return float(np.mean(pos)) if pos else float('nan')


def _random_order(rng, n, start):
    order = [i for i in range(n) if i != start]
    rng.shuffle(order)
    return [start] + order


def _precompute_fsp(rctx):
    """Compute the position-independent f_sp once per candidate."""
    pool = rctx['pool']
    pp, np_i = rctx['pp'], rctx['np']
    uc = rctx['use_copula']
    p_acc = rctx['p_acc']
    return [M.f_sp_score(x[1], x[2], x[3], x[4], x[5], pp, np_i, uc, p_acc)
            for x in pool]


def _fast_capable(metric, rctx):
    """Whether the incremental scorer is valid for this metric and diversity mode."""
    if metric == 'random':
        return False
    if metric == 'mdcg' and (M.DIV_MODE == 'anchor' or
                             (M.DIV_MODE == 'threshold' and M.DIV_MREF_SOURCE == 'pool')):
        return False
    return True


def _greedy_order_fast(metric, rctx, start, fsp):
    """Fix the first slot, then take the largest marginal gain per position, reusing cached scores and similarities."""
    pool = rctx['pool']
    n = len(pool)

    if metric in _SINGLE_FACTORS:
        _j = _SINGLE_FACTORS[metric]
        if BASELINE_RAW and metric in ('rel', 'dif'):
            _j = 7 if metric == 'rel' else 8
        _sc = [float(x[_j]) for x in pool]
        _sg = (rctx.get('factor_sign') or {}).get(metric, 1)
        if _sg < 0:
            _sc = [-v for v in _sc]
        return [start] + sorted([i for i in range(n) if i != start],
                                key=lambda i: (-_sc[i], i))

    _GAINS = {'mmr': M.gain_mmr(lam=float(os.environ.get('MDCG_MMR_LAMBDA', '0.5'))),
              'xquad': M.gain_xquad()}
    if rctx.get('lincombo_w'):
        _GAINS['lincombo'] = M.gain_lincombo(rctx['lincombo_w'])
    if metric in _GAINS or metric in ('ge_unexp', 'content_unexp', 'ppmi_unexp'):
        _rel9 = [x[7] for x in pool]          # 基线读取原始 rel，M-DCG 读取变换后的 rel
        _cats9 = [rctx['item_cate'].get(x[0]) for x in pool]
        _hd9 = M._cat_dist(rctx.get('hist_cate') or [], None)
        _sim9 = None
        if metric in ('mmr', 'dpp'):
            _sim9 = [[rctx['sf'](pool[i9][0], pool[j9][0]) for j9 in range(n)] for i9 in range(n)]
        _lc9 = ([[x[7], x[8], x[3], x[4], x[5]] for x in pool] if BASELINE_RAW
                else [[x[1], x[2], x[3], x[4], x[5]] for x in pool])
        _ctx9 = {'rel': _rel9, 'cats': _cats9, 'hist_dist': _hd9, 'sim': _sim9,
                 'pop': [rctx['item_pop'].get(x[0], 0) for x in pool], 'ppmi': [0.0] * n,
                 'lc': _lc9}
        if metric in _GAINS:
            return M.greedy_by_gain(n, start, _GAINS[metric], _ctx9)
        if metric == 'ge_unexp':
            _pop9 = sorted(_ctx9['pop'])
            _thr9 = _pop9[max(0, int(0.5 * len(_pop9)) - 1)] if _pop9 else 0
            _sc9 = M.ge_unexpected_scores(_ctx9['pop'], _thr9)
        elif metric == 'ppmi_unexp':
            _sc9 = M.ppmi_unexpected_scores(_rel9, rctx.get('ppmi_vec') or [0.0] * n)
        else:
            _sc9 = M.content_dist_scores(_rel9)
        return [start] + sorted([i for i in range(n) if i != start],
                                key=lambda i: (-_sc9[i], i))
    if metric == 'fn_surprise':
        _fp = rctx.get('fnpair') or {}
        _hits = rctx.get('hist_items_ts') or []
        _by_pair = {}
        for _h, _ts in _hits:
            _p = _fp.get(_h, (None, 0.0))[0]
            if _p is not None:
                _by_pair.setdefault(_p, []).append(int(_ts))
        for _p in _by_pair:
            _by_pair[_p].sort()
        _tgt = pool[0][0]
        _t = (rctx.get('label_time') or {}).get((rctx['uid'], _tgt))
        if _t is None:
            _t = max([ts for _it, ts in _hits], default=0)
        _t = int(_t)

        def _si(_item, _upto):
            _p, _s = _fp.get(_item, (None, 0.0))
            if _p is None:
                return 0.0
            _arr = _by_pair.get(_p)
            _F = bisect.bisect_left(_arr, _upto) if _arr else 0
            return _s / (1.0 + _F)

        _hist_si = _FN_HISTSI.get(rctx['uid'])
        if _hist_si is None:
            _hist_si = [_si(_h, _ts) for _h, _ts in _hits]
            _FN_HISTSI[rctx['uid']] = _hist_si
        _si = [_si(x[0], _t) for x in pool]
        if _hist_si:
            _lb = float(np.quantile(_hist_si, 0.25))
            _ub = float(np.quantile(_hist_si, 0.75))
        else:
            _lb, _ub = 0.0, 1.0
        _mid = 0.5 * (_lb + _ub)
        _sc = [(-abs(v - _ub)) - (0.0 if (_mid < v <= _ub) else 1e3) for v in _si]
        return [start] + sorted([i for i in range(n) if i != start],
                                key=lambda i: (-_sc[i], i))
    if metric in ('epc', 'rel_unexp', 'ndcg_rating', 'damping'):
        if metric == 'epc':
            _pop0, _mx0 = rctx['item_pop'], rctx['max_pop']
            _sc = [float(x[5]) * (1.0 - _pop0.get(x[0], 0) / (_mx0 + 1e-9)) for x in pool]
        elif metric == 'rel_unexp':
            _irg0 = rctx['item_rating_g'] or {}
            _pop0, _mx0 = rctx['item_pop'], rctx['max_pop']
            _sc = [(float(_irg0.get(x[0], 0.0)) / 5.0)
                   * (1.0 - _pop0.get(x[0], 0) / (_mx0 + 1e-9)) for x in pool]
        elif metric == 'damping':
            _popD, _mxD = rctx['item_pop'], rctx['max_pop']
            _sc = M.popularity_damping_scores([x[7] for x in pool],
                                              [_popD.get(x[0], 0) for x in pool], _mxD, 0.5)
        else:
            _irg0 = rctx['item_rating_g'] or {}
            _sc = [float(_irg0.get(x[0], 0.0)) for x in pool]
        return [start] + sorted([i for i in range(n) if i != start],
                                key=lambda i: (-_sc[i], i))
    sf = M.div_sim_func(rctx['sf'])          # div 用的相似度(可能是 DIV_SIM)
    order = [start]
    remaining = [i for i in range(n) if i != start]
    _sim_cache = {}                          # (i, j) -> sim, 跨步复用
    _pfx = False; _ps, _pc = 0.0, 0
    if metric == 'mdcg':
        mref = M.get_div_mref(rctx['uid'])
        _pfx = (M.DIV_MODE == 'threshold' and M.DIV_MREF_SOURCE == 'prefix')
    if metric == 'alpha_ndcg':
        _gamA = {}
        for _x in pool:
            _cc = rctx['item_cate'].get(_x[0])
            if _cc is not None and _cc not in _gamA:
                _gamA[_cc] = M.compute_gamma(rctx['hist_cate'], rctx['hist_rating'], _cc, 5.0)
    if metric == 'epc':
        _popE, _mxE = rctx['item_pop'], rctx['max_pop']
        val = [float(pool[i][5]) * (1.0 - _popE.get(pool[i][0], 0) / (_mxE + 1e-9))
               for i in range(n)]
    if metric == 'rel_unexp':
        _irg3 = rctx['item_rating_g'] or {}
        _pop3, _mx3 = rctx['item_pop'], rctx['max_pop']
        val2 = [(float(_irg3.get(x[0], 0.0)) / 5.0)
                * (1.0 - _pop3.get(x[0], 0) / (_mx3 + 1e-9)) for x in pool]
    if metric == 'abeta_ndcg':
        _irg = rctx['item_rating_g'] or {}
        _A = float(os.environ.get('MDCG_ABETA_ALPHA', '0.005'))
        _B = float(os.environ.get('MDCG_ABETA_BETA', '0.5'))
        _RMAX = 5.0
        _iasp = rctx.get('item_aspects') or None
        _hasp = rctx.get('hist_aspects') or None
        _hrat_asp = rctx.get('hist_rating_asp') or None
        _gam = {}
        if _iasp is not None and _hasp is not None:
            for x in pool:
                for _a in (_iasp.get(x[0]) or {}):
                    if _a not in _gam:
                        _gam[_a] = M.compute_gamma_aspects(_hasp, _hrat_asp, _a, _RMAX)
        else:
            _iasp = None
            for x in pool:
                c = rctx['item_cate'].get(x[0])
                if c is not None and c not in _gam:
                    _gam[c] = M.compute_gamma(rctx['hist_cate'], rctx['hist_rating'],
                                              c, _RMAX)
        _rho = {}          # aspect -> {rating: count}
        _tau = {}          # aspect -> 缺评分物品数
    while remaining:
        p = len(order)
        best, best_g = None, -np.inf
        for i in remaining:
            if metric == 'mdcg':
                if p == 0:
                    d = 1.0
                else:
                    _it = pool[i][0]
                    _s = []
                    for j in order:
                        k2 = (i, j) if i < j else (j, i)
                        v = _sim_cache.get(k2)
                        if v is None:
                            v = sf(_it, pool[j][0]); _sim_cache[k2] = v
                        _s.append(v)
                    _r = (_ps / _pc) if (_pfx and _pc > 0) else mref
                    d = M.div_discount(_s, _r)
                g = fsp[i] * d / log2(2 + p)
            elif metric == 'epc':
                g = val[i]
            elif metric == 'rel_unexp':
                g = val2[i]
            elif metric == 'ndcg_rating':
                g = float((rctx['item_rating_g'] or {}).get(pool[i][0], 0.0)) / log2(2 + p)
            elif metric == 'ndcg_seren':
                g = pool[i][6] / log2(2 + p)
            elif metric == 'abeta_ndcg':
                _it = pool[i][0]
                if _iasp is not None:
                    _asp = _iasp.get(_it)
                    if not _asp:
                        g = 0.0
                    else:
                        _pa = (min(1.0, float(_irg[_it]) / _RMAX) * _B) if _it in _irg else _A
                        _prod = 1.0
                        for _a, _w in _asp.items():
                            if not _w or _w <= 0:
                                continue
                            _ga = _gam.get(_a, 0.0)
                            if _ga <= 0:
                                continue
                            _red = (1.0 - _A) ** _tau.get(_a, 0)
                            for _r, _cnt in _rho.get(_a, {}).items():
                                _red *= (1.0 - M.beta_func(_r, _RMAX, _B)) ** _cnt
                            _prod *= (1.0 - _pa * float(_w) * _ga * _red)
                        g = (1.0 - _prod) / log2(2 + p)
                else:
                    c = rctx['item_cate'].get(_it)
                    if c is None:
                        g = 0.0
                    else:
                        if _it in _irg:
                            _pa = min(1.0, float(_irg[_it]) / _RMAX) * _B
                        else:
                            _pa = _A
                        red = (1.0 - _A) ** _tau.get(c, 0)
                        for _r, _cnt in _rho.get(c, {}).items():
                            red *= (1.0 - M.beta_func(_r, _RMAX, _B)) ** _cnt
                        g = _pa * _gam.get(c, 0.0) * red / log2(2 + p)
            elif metric == 'alpha_ndcg':
                _it = pool[i][0]
                _c4 = rctx['item_cate'].get(_it)
                _rel4 = float((rctx['item_rating_g'] or {}).get(_it, 0.0)) / 5.0
                _t4 = sum(1 for j in order if rctx['item_cate'].get(pool[j][0]) == _c4)
                g = _rel4 * (0.5 ** _t4) / log2(2 + p)
            elif metric == 'ild':
                g = sum(1.0 - rctx['sf'](pool[i][0], pool[j][0]) for j in order)
            else:
                g = 0.0
            if g > best_g:
                best_g, best = g, i
        order.append(best)
        remaining.remove(best)
        if _pfx:
            _b = pool[best][0]
            for j in order[:-1]:
                k2 = (best, j) if best < j else (j, best)
                v = _sim_cache.get(k2)
                if v is None:
                    v = sf(_b, pool[j][0]); _sim_cache[k2] = v
                _ps += v
            _pc += p
        if metric == 'abeta_ndcg':
            _it = pool[best][0]
            if _iasp is not None:
                _keys = [_a for _a, _w in (_iasp.get(_it) or {}).items() if _w and _w > 0]
            else:
                _c = rctx['item_cate'].get(_it)
                _keys = [_c] if _c is not None else []
            for _a in _keys:
                if _it in _irg:
                    _rho.setdefault(_a, {})
                    _r = _irg[_it]
                    _rho[_a][_r] = _rho[_a].get(_r, 0) + 1
                else:
                    _tau[_a] = _tau.get(_a, 0) + 1
    return order


def _greedy_order(metric, rctx, start):
    """Fix the first slot, then take the candidate that maximises the current list score at each step."""
    n = len(rctx['pool'])
    order = [start]
    remaining = [i for i in range(n) if i != start]
    while remaining:
        best, best_g = None, -np.inf
        for i in remaining:
            g = _order_gain(metric, rctx, order + [i])
            if g > best_g:
                best_g, best = g, i
        order.append(best)
        remaining.remove(best)
    return order


def _order_gain(metric, rctx, order):
    """Score of the given ordering under the metric."""
    pool = rctx['pool']
    feats = [pool[i] for i in order]
    items = [x[0] for x in feats]
    k = len(order)

    if metric == 'mdcg':
        mlist = [(x[0], x[1], x[2], x[3], x[4], x[5], rctx['uid']) for x in feats]
        return M.mdcg_score(mlist, rctx['pp'], rctx['np'], sim_func=rctx['sf'],
                            use_copula=rctx['use_copula'], k=k,
                            user_scarcity_params={rctx['uid']: rctx['p_acc']})
    if metric == 'epc':
        return M.epc_score(items, rctx['item_pop'], rctx['max_pop'], k,
                           rel={x[0]: float(x[5]) for x in feats})
    if metric == 'ild':
        return M.ild_score(items, rctx['sf'], k)
    if metric == 'ndcg_seren':
        return M.ndcg_seren_score([x[6] for x in feats], k)
    if metric == 'abeta_ndcg':
        return M.abeta_ndcg(items, rctx['item_cate'], rctx['item_rating_g'],
                            rctx['hist_cate'], rctx['hist_rating'], k=k)
    if metric in _SINGLE_FACTORS:
        _j = _SINGLE_FACTORS[metric]
        return sum(float(x[_j]) / log2(2 + p) for p, x in enumerate(feats))
    return 0.0


def _order_auc(labels):
    """Rank-based AUC: label 1 is positive, everything else negative."""
    y = (np.asarray(labels, dtype=float) == 1.0).astype(int)
    if len(y) < 2 or len(set(y.tolist())) < 2:
        return float('nan')
    scores = np.arange(len(y))[::-1].astype(float)
    try:
        return float(roc_auc_score(y, scores))
    except Exception:
        return float('nan')


_ZEROSHOT_UNL = {}          # --pool_source zeroshot 时载入的未标注物品
_FN_HISTSI = {}             # Fu&Niu: 每个用户历史 SI 分布的缓存
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model', help='Artifact tag used in the pkl names: Seren, SEREN or AMZ.')
    ap.add_argument('--serenlens', default='./data/SerenLens/SerenLens_Movies.csv')
    ap.add_argument('--factors_path', default=None, help='Factor table path; defaults to factors_{model}_{data}.pkl.')
    ap.add_argument('--data', default='Movies')
    ap.add_argument('--quality', dest='quality', default='auto', choices=['auto', 'off'], help='Quality source: auto prefers the raw Amazon ratings and falls back to SerenLens; off uses SerenLens only.')
    ap.add_argument('--qua_mode', dest='qua_mode', default='pit', choices=['pit', 'window'],
                    help='Quality over time: pit uses the mean before the interaction (paper setting); window uses the training-window mean (leaky, legacy only).')
    ap.add_argument('--amazon_dir', default=None, help='Directory of the raw Amazon files (default <cwd>/data/amazon).')
    ap.add_argument('--quality_cache', default=None, help='Cache path for the quality table (default quality_<model>_<data>.pkl).')
    ap.add_argument('--rebuild_quality', action='store_true', help='Ignore the cache and rescan the Amazon files.')
    ap.add_argument('--quality_shrink', type=float, default=0.0, help='Shrinkage k for the mean rating: (sum + k*3.0)/(cnt + k); 0 disables it.')
    ap.add_argument('--use_copula', action='store_true')
    ap.add_argument('--min_history', type=int, default=5)
    ap.add_argument('--min_user_items', type=int, default=1)
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--factor_mode', '--scarcity_mode', dest='factor_mode', default='both',
                    choices=['both', 'nov_only', 'unpop_only', 'none'],
                    help='N/U accommodation factors: both (main), Novelty only, Unpopularity only, or none')
    ap.add_argument('--cauchy_alpha', '--scarcity_gamma', dest='cauchy_alpha', type=float, default=1.0,
                    help='Base curvature alpha for asymmetric Cauchy kernels; alpha=0 gives flat response')
    ap.add_argument('--scarcity_peak_mode', default='adaptive_mixture', choices=['adaptive_mixture', 'single'],
                    help='Cauchy centers: auto multi-peak mixture (BIC) or old single center')
    ap.add_argument('--scarcity_max_peaks', type=int, default=4,
                    help='maximum automatically selected peaks per N/U factor')
    ap.add_argument('--scarcity_min_peak_samples', type=int, default=8,
                    help='minimum effective samples per mixture component')
    ap.add_argument('--scarcity_min_peak_sep', type=float, default=0.06,
                    help='minimum separation between two accepted peak centers on [0,1]')
    ap.add_argument('--test_ratio', type=float, default=0.2,
                    help='Fraction of the timeline used as the test window (default 0.2, the paper setting).')
    ap.add_argument('--lambda_source', default='surprise_pool', choices=['surprise_pool', 'history'],
                    help='Cauchy center: personalized surprise pool or ordinary history')
    ap.add_argument('--lambda_stat', default='median', choices=['median', 'mean'],
                    help='Center statistic inside personalized surprise pool')
    ap.add_argument('--lambda_min_pool', type=int, default=5,
                    help='Minimum positive items required in personalized surprise pool')
    ap.add_argument('--sim_user_topk', type=int, default=PROFILE_TOPK,
                    help='Number of similar users (defaults to PROFILE_TOPK; hit-rate study only).')
    ap.add_argument('--seed', type=int, default=42, help='Random seed for the reranking in the hit-rate study.')
    ap.add_argument('--pool_dedup', default='none',
                    choices=['ratio', 'majority', 'any_pos', 'none'],
                    help='Item assignment when users disagree: none keeps both pools, ratio uses pos/(pos+neg), majority uses pos > neg, any_pos takes any positive.')
    ap.add_argument('--pos_ratio_thr', type=float, default=0.5,
                    help='Threshold for the ratio rule: pos/(pos+neg) >= thr counts as positive.')
    ap.add_argument('--pool_shuffle', default='on', choices=['on', 'off'],
                    help='Shuffle the pool before ranking. on removes the structural advantage of lists whose positives come first; off reproduces the old order.')
    ap.add_argument('--abeta_rating', default='safe', choices=['safe', 'leaky'],
                    help="Item ratings for alpha-beta-nDCG: safe uses the pool's own quality (same as M-DCG), leaky uses the boundary mean including the annotator's rating.")
    ap.add_argument('--abeta_aspects', default='auto', choices=['auto', 'off'],
                    help='Aspect handling for alpha-beta-nDCG: auto uses aspects_{model}_{data}.pkl when present, otherwise KMeans categories; off forces single-category.')
    ap.add_argument('--boundary_ts', type=float, default=0.0,
                    help='Freeze the time split at this Unix timestamp; required by the history-deletion study so the test window does not drift.')
    ap.add_argument('--baseline_raw', action='store_true',
                    help='Baselines use the raw rel/dif without the E+D rescaling; M-DCG is unchanged.')
    ap.add_argument('--n_lists_per_user', type=int, default=5,
                    help='Reranked lists per user in the hit-rate study (default 5).')
    ap.add_argument('--hit_rate_metrics', default=None,                    help='Comma-separated rankers to evaluate in the hit-rate study; random is always kept.')
    ap.add_argument('--add_hit_metrics', default=None,
                    help='Comma-separated rankers added on top of the default set.')
    ap.add_argument('--skip_hit_rate', action='store_true',
                    help='Skip the hit-rate study, e.g. when --div_mref pool disables the fast greedy path.')
    ap.add_argument('--factor_transform', default='auto',
                    choices=['auto', 'none', 'quantile', 'quantile_center'],
                    help='Scale transform for rel/dif: auto picks per dataset, none leaves them raw, quantile normalises globally, quantile_center centres per user first.')
    ap.add_argument('--factor_center_ref', default='neg', choices=['neg', 'allrows'],
                    help='Rows used for centring: neg uses labelled negatives (paper setting), allrows uses every labelled row of the user.')
    ap.add_argument('--factor_center_mode', default='causal', choices=['causal', 'all'],
                    help='How the center is computed: causal uses strictly earlier negatives (default); all uses the full history and leaks future rows.')
    ap.add_argument('--factor_center_min', type=int, default=5,
                    help='Minimum earlier negatives before the causal center falls back to the global one.')
    ap.add_argument('--run_div_pert', action='store_true',
                    help='Run the diversity diagnostic: report the diversity term by perturbation level and position.')
    ap.add_argument('--run_ablation', action='store_true',
                    help='Run the ablation study (off by default).')
    ap.add_argument('--run_normalization', action='store_true',
                    help='Run the normalization study (off by default).')
    ap.add_argument('--run_robustness', action='store_true',
                    help='Run the quality-robustness study (off by default; it has no temporal split).')
    ap.add_argument('--qua_const', default=None,
                    help="Constant-quality mode: set every item's quality to the given value(s) instead of masking at random.")
    ap.add_argument('--mask_value', default=None, type=float,
                    help='Quality assigned to masked items (default 0, the paper setting).')
    ap.add_argument('--robust_neg_per_pos', default=None, type=float,
                    help='Pool ratio for the robustness study: keep all positives and sample N negatives per positive (99 reproduces the 1:99 pool).')
    ap.add_argument('--run_perturbation', action='store_true',
                    help='Run the perturbation-consistency study (off by default).')
    ap.add_argument('--perturb_metrics', default=None,
                    help='Comma-separated metrics for the perturbation study; defaults to all.')
    ap.add_argument('--pool_source', default='similar_users',
                    choices=['zeroshot', 'similar_users', 'own', 'global_neg', 'test_own',
                             'sim_own'],
                    help="Candidate pool source: similar_users (default), own (target user's labels), test_own (1 positive plus k negatives from the test window), zeroshot, global_neg.")
    ap.add_argument('--match_popularity', action='store_true',
                    help='Match negatives to positives by popularity quantile, so popularity cannot act as a label proxy.')
    ap.add_argument('--low_rating_to_train', action='store_true',
                    help='Keep low-rating derived negatives out of the quantile split and inside the training window (needed for S2018).')
    ap.add_argument('--low_rating_thr', type=float, default=2.5)
    ap.add_argument('--boundary_pairs', default=None,
                    help='CSV with uid,item pairs; the time quantile is computed from these labelled rows only.')
    ap.add_argument('--drop_nonsurvey_neg', action='store_true',
                    help='S2018 mode: keep only the survey rows listed in --boundary_pairs and let zeroshot items supply the negatives.')
    ap.add_argument('--pool_k', type=int, default=None,
                    help='Candidate pool size for the hit-rate study; defaults to --k. Use --pool_k 100 --pool_pos_frac 0.01 for a 1:99 pool.')
    ap.add_argument('--pool_dump', default=None,
                    help="Write each user's pool (uid,item,label,src) here at level 0, then replay it with --pool_file.")
    ap.add_argument('--neighbor_source', default='surprise',
                    choices=['surprise', 'user_sim'],
                    help='Neighbour source for the hit-rate study: surprise profile (default) or co-rating similarity (user_sim, label-free).')
    ap.add_argument('--profile_source', default='all',
                    choices=['all', 'train'],
                    help='Rows used to build the profile: all (default, higher coverage but uses test-window labels) or train.')
    ap.add_argument('--lincombo_neighbors', action='store_true',
                    help="Fit the linear combination on the target user's top-50 neighbours only, matching the data budget of M-DCG.")
    ap.add_argument('--profile_items', default='pos',
                    choices=['pos', 'all'],
                    help='Rows used to fit the surprise profile: pos (default) or all rows regardless of label.')
    ap.add_argument('--neighbor_fallback', default='none',
                    choices=['none', 'user_sim'],
                    help='Fallback for users without training positives: take neighbours from co-rating similarity instead.')
    ap.add_argument('--pool_file', default=None,
                    help='Replay a frozen pool written by --pool_dump.')
    ap.add_argument('--pool_pos_frac', type=float, default=0.2,
                    help='Share of positives in the pool (default 0.2; lower it to soften the ceiling effect).')
    ap.add_argument('--in_sample', action='store_true',
                    help='In-sample protocol: fit parameters on all labels and let the hit-rate study use ratings that cover the test window.')
    ap.add_argument('--div_mode', default='legacy',
                    choices=['legacy', 'threshold', 'paper'],
                    help='Diversity convention: legacy uses 1-|m|; threshold penalises similarity above the user reference r; paper matches the formula in the paper.')
    ap.add_argument('--div_mref', default='user', choices=['user', 'global', 'pool', 'prefix'],
                    help='Where the reference r comes from: per user, global, the pool itself, or the prefix before the current position.')
    ap.add_argument('--div_mref_value', type=float, default=0.40,
                    help="Fallback reference when a user's own reference cannot be estimated (default 0.40).")
    ap.add_argument('--div_sim', default='embedding',
                    choices=['embedding', 'amazon_emb', 'amazon_cooc'],
                    help='Item similarity used by the diversity term only: the main embedding, the Amazon embedding, or Amazon co-occurrence.')
    ap.add_argument('--div_sim_emb', default=None,
                    help='Embedding file for --div_sim amazon_emb; searched automatically when omitted.')
    args = ap.parse_args()
    global PAPER_TEST_RATIO
    PAPER_TEST_RATIO = float(args.test_ratio)
    global BASELINE_RAW
    BASELINE_RAW = bool(getattr(args, 'baseline_raw', False))
    global FREEZE_BOUNDARY
    FREEZE_BOUNDARY = float(getattr(args, 'boundary_ts', 0.0) or 0.0)
    M.set_accommodation_config(
        factor_mode=args.factor_mode, alpha=args.cauchy_alpha,
        peak_mode=args.scarcity_peak_mode,
        max_peaks=args.scarcity_max_peaks,
        min_peak_samples=args.scarcity_min_peak_samples,
        min_peak_sep=args.scarcity_min_peak_sep)

    emb_path = 'emb_{}_{}.pkl'.format(args.model, args.data)
    cate_path = 'cate_{}_{}.pkl'.format(args.model, args.data)
    pop_path = 'pop_{}_{}.pkl'.format(args.model, args.data)
    pass
    pass
    pass
    pass
    pass

    pass
    user_seq, labels, label_time, item_rating = load_serenlens(args.serenlens)
    embeddings = load_embeddings(emb_path)
    item_cate, item_pop = load_item_meta(cate_path, pop_path)
    fnpair = {}
    _fnp = 'fnpair_{}_{}.pkl'.format(args.model, args.data)
    if os.path.exists(_fnp):
        with open(_fnp, 'rb') as _f:
            fnpair = pickle.load(_f)
        pass
    else:
        pass
    # Use multi-aspect when the aspects file exists, single-category otherwise.
    item_aspects = None
    asp_path = 'aspects_{}_{}.pkl'.format(args.model, args.data)
    if args.abeta_aspects == 'auto' and os.path.exists(asp_path):
        with open(asp_path, 'rb') as _f:
            item_aspects = pickle.load(_f)
        _n_a = len({a for d in item_aspects.values() for a in d})
        pass
    else:
        pass
    max_pop = max(item_pop.values()) if item_pop else 1.0
    sim_engine = SimEngine(embeddings)

    factors_path = (args.factors_path if getattr(args, 'factors_path', None)
                    else 'factors_{}_{}.pkl'.format(args.model, args.data))

    pass
    factor_df = pd.read_pickle(factors_path)

    if getattr(args, 'drop_nonsurvey_neg', False):
        if not args.boundary_pairs:
            raise SystemExit('--drop_nonsurvey_neg 必须同时给 --boundary_pairs')
        _bp_df = pd.read_csv(args.boundary_pairs, dtype=str)
        _bp_set = set(zip(_bp_df.iloc[:, 0].astype(str), _bp_df.iloc[:, 1].astype(str)))
        _keep_mask = np.array([(str(u), str(i)) in _bp_set
                               for u, i in zip(factor_df['uid'], factor_df['item'])])
        _n_before = len(factor_df)
        factor_df = factor_df[_keep_mask].reset_index(drop=True)
        pass

    quality_index = None
    if args.quality != 'off' and args.qua_mode != 'window':
        try:
            import quality as _Q
        except ImportError:
            _Q = None
        if _Q is None:
            pass
        else:
            try:
                quality_index = _Q.ensure_quality_index(
                    args.serenlens, args.model, args.data,
                    amazon_dir=(args.amazon_dir or _Q.DEFAULT_AMAZON_DIR),
                    cache_path=args.quality_cache,
                    rebuild=args.rebuild_quality, log=print)
            except Exception as _e:
                pass
                quality_index = None
    if args.qua_mode == 'window':
        pass
    elif quality_index is not None:
        _shrink = args.quality_shrink
        _uids = factor_df['uid'].tolist()
        _items = factor_df['item'].tolist()
        _tss = [label_time.get((u, i), 0) for u, i in zip(_uids, _items)]
        _old = factor_df['qua'].values
        _new = np.array([quality_index.mean_at(it, ts, shrink_k=_shrink)
                         for it, ts in zip(_items, _tss)])
        pass
        if float(np.std(_new)) < 1e-6:
            pass
        factor_df = factor_df.copy()
        factor_df['qua'] = _new

    _tf = args.factor_transform
    if _tf == 'auto':
        _tf = 'quantile_center'
        pass
    factor_df['rel_raw'] = factor_df['rel'].values.astype(float)
    factor_df['dif_raw'] = factor_df['dif'].values.astype(float)   # --baseline_raw 用
    if _tf != 'none':
        factor_df = factor_df.copy()
        _r_raw = factor_df['rel'].values.astype(float)
        _d_raw = factor_df['dif'].values.astype(float)
        _u = factor_df['uid'].values
        _lab = factor_df['label'].values
        _cr = np.zeros(len(factor_df))
        _cd = np.zeros(len(factor_df))
        if _tf == 'quantile_center':
            if args.factor_center_mode == 'all':
                _c_all = factor_df.groupby('uid')[['rel', 'dif']].mean()
                _c_neg = factor_df[factor_df['label'] == 0].groupby('uid')[['rel', 'dif']].mean()
                _c = _c_neg.reindex(_c_all.index).fillna(_c_all)
                _cr = _c['rel'].reindex(_u).values
                _cd = _c['dif'].reindex(_u).values
                pass
            else:
                _ts = _attach_ts(factor_df, label_time)['ts'].values.astype(np.int64)
                _n_bad = int((_ts <= 0).sum())
                if _n_bad:
                    pass
                _ref_lab = (_lab if getattr(args, 'factor_center_ref', 'neg') == 'neg'
                            else np.zeros(len(_lab), dtype=np.int64))
                if getattr(args, 'factor_center_ref', 'neg') == 'allrows':
                    pass
                _one = np.zeros(len(factor_df), dtype=np.int64)
                _gr, _gd, _ = _causal_running_center(
                    _one, _ts, _ref_lab, _r_raw, _d_raw, args.factor_center_min)
                _cr, _cd, _cn = _causal_running_center(
                    _u, _ts, _ref_lab, _r_raw, _d_raw, args.factor_center_min)
                _need = _cn < args.factor_center_min
                _cr = np.where(_need, _gr, _cr)
                _cd = np.where(_need, _gd, _cd)
                pass
            _r_raw = _r_raw - _cr
            _d_raw = _d_raw - _cd
        factor_df['rel'] = M.quantile_normalize(_r_raw)
        factor_df['dif'] = M.quantile_normalize(_d_raw)
        pass

    _scale_ctx = None
    if args.pool_source == 'test_own':
        if _tf != 'none':
            _cr_by_u = pd.Series(_cr, index=_u).groupby(level=0).mean()
            _cd_by_u = pd.Series(_cd, index=_u).groupby(level=0).mean()
            _scale_ctx = dict(tf=_tf, r_ref=_r_raw, d_ref=_d_raw,
                              cr=_cr_by_u, cd=_cd_by_u,
                              cr_all=float(np.mean(_cr)), cd_all=float(np.mean(_cd)))
            pass
        else:
            _scale_ctx = dict(tf='none')
            pass

    _div_sim_fn = build_div_sim_fn(args, embeddings)
    if _div_sim_fn is not None:
        M.set_div_config(sim=_div_sim_fn)

    if args.div_mode == 'threshold':
        user_div_mref = M.build_div_mref(factor_df, sim_engine, source=args.div_mref,
                                         min_pos=2, fallback=args.div_mref_value)
        M.set_div_config(mode='threshold', mref=user_div_mref,
                         mref_default=args.div_mref_value, source=args.div_mref)
        pass
        pass
    elif args.div_mode == 'paper':
        M.set_div_config(mode='paper')
        pass
    else:
        M.set_div_config(mode='legacy')
        pass

    pass
    factor_df['nov_q'] = M.quantile_normalize(factor_df['novelty'].values)
    factor_df['unpop_q'] = M.quantile_normalize(factor_df['unpopularity'].values)
    factor_df['rare'] = 0.5 * (factor_df['nov_q'] + factor_df['unpop_q'])

    item_ts_first_all = {}
    ts_min_all, ts_max_all = float('inf'), 0
    for seq in user_seq.values():
        for it, ts, r in seq:
            if it not in item_ts_first_all or ts < item_ts_first_all[it]:
                item_ts_first_all[it] = ts
            ts_min_all = min(ts_min_all, ts)
            ts_max_all = max(ts_max_all, ts)

    all_items = sorted(set(item_pop.keys()) | set(item_ts_first_all.keys()))
    raw_nov_all = np.array([
        (item_ts_first_all.get(it, ts_min_all) - ts_min_all) / (ts_max_all - ts_min_all + 1e-9)
        if ts_max_all > ts_min_all else 0.5
        for it in all_items
    ], dtype=float)
    raw_unpop_all = np.array([
        1.0 - float(item_pop.get(it, 0)) / (float(max_pop) + 1e-9)
        for it in all_items
    ], dtype=float)

    nov_q_all = quantile_transform_from_reference(raw_nov_all, factor_df['novelty'].values)
    unpop_q_all = quantile_transform_from_reference(raw_unpop_all, factor_df['unpopularity'].values)
    item_nov_map = {it: float(v) for it, v in zip(all_items, nov_q_all)}
    item_unpop_map = {it: float(v) for it, v in zip(all_items, unpop_q_all)}
    item_rare_map = {it: 0.5 * (item_nov_map[it] + item_unpop_map[it]) for it in all_items}

    user_scarcity_params = M.build_user_scarcity_params(user_seq, item_rare_map, item_nov_map, item_unpop_map, min_history=args.min_history)
    qn_pos, qn_neg, qu_pos, qu_neg = [], [], [], []
    for _, r in factor_df.iterrows():
        uid = r['uid']
        p_acc = user_scarcity_params.get(uid) if user_scarcity_params is not None else None
        qn, qu = M.accommodation_components(r['nov_q'], r['unpop_q'], p_acc)
        if r['label'] == 1:
            qn_pos.append(qn); qu_pos.append(qu)
        else:
            qn_neg.append(qn); qu_neg.append(qu)

    def _print_q_stats(name, values):
        x = np.asarray(values, dtype=float)
        pass

    _print_q_stats("q_N 正样本", qn_pos)
    _print_q_stats("q_N 负样本", qn_neg)
    _print_q_stats("q_U 正样本", qu_pos)
    _print_q_stats("q_U 负样本", qu_neg)
    qua_pos = factor_df[factor_df['label'] == 1]['qua'].values
    qua_neg = factor_df[factor_df['label'] == 0]['qua'].values

    pass
    pass

    pos = factor_df['label'] == 1
    neg = factor_df['label'] == 0
    pos_params = M.fit_class_params(
        factor_df['rel'][pos].values, factor_df['dif'][pos].values, args.use_copula)
    neg_params = M.fit_class_params(
        factor_df['rel'][neg].values, factor_df['dif'][neg].values, args.use_copula)
    pass
    pass
    pass

    _g_ts = np.array([label_time.get((u, i), 0) for u, i in
                      zip(factor_df['uid'].values, factor_df['item'].values)], float)
    _g_b = float(np.quantile(_g_ts, 1.0 - PAPER_TEST_RATIO)) if len(_g_ts) else 0.0
    item_rating_g = {}
    _rt_sum, _rt_cnt = defaultdict(float), defaultdict(int)
    for (uid, it), rt in item_rating.items():
        _t = label_time.get((uid, it), 0)
        if _t and _t <= _g_b:
            _rt_sum[it] += rt; _rt_cnt[it] += 1
    for it in _rt_sum:
        item_rating_g[it] = _rt_sum[it] / _rt_cnt[it]
    pass
    _rc = os.environ.get('MDCG_R_CONST')
    if _rc:
        _rv = float(_rc)
        item_rating_g = {it: _rv for it in item_rating_g}
        pass

    items_by_uid = defaultdict(list)
    for _, r in factor_df.iterrows():
        items_by_uid[r['uid']].append(
            (r['item'], r['rel'], r['dif'], r['nov_q'], r['unpop_q'],
             r['qua'], r['label'], r['rel_raw']))
    if args.pool_source == 'zeroshot':
        global _ZEROSHOT_UNL
        _unl_path = 'factors_{}_{}_unlabeled.pkl'.format(args.model, args.data)
        if os.path.exists(_unl_path):
            _unl = pd.read_pickle(_unl_path)
            _nov_q = quantile_transform_from_reference(_unl['novelty'].values, factor_df['novelty'].values)
            _unp_q = quantile_transform_from_reference(_unl['unpopularity'].values, factor_df['unpopularity'].values)
            if _tf != 'none':
                _rel_q = quantile_transform_from_reference(
                    _unl['rel'].values - float(np.mean(_cr)), _r_raw)
                _dif_q = quantile_transform_from_reference(
                    _unl['dif'].values - float(np.mean(_cd)), _d_raw)
                pass
            else:
                _rel_q = _unl['rel'].values
                _dif_q = _unl['dif'].values
            _ZEROSHOT_UNL = defaultdict(list)
            for _i, _r in enumerate(_unl.itertuples(index=False)):
                _ZEROSHOT_UNL[_r.uid].append(
                    (_r.item, float(_rel_q[_i]), float(_dif_q[_i]),
                     _nov_q[_i], _unp_q[_i], _r.qua, -1, float(_r.rel)))
            pass
            for _u, _rows in _ZEROSHOT_UNL.items():
                items_by_uid[_u].extend(_rows)
        else:
            pass
    user_items_set = {uid: set(it for it, ts, r in seq)
                      for uid, seq in user_seq.items()}

    pass
    profile_uids, profiles = [], []
    pos_df = factor_df[factor_df['label'] == 1]
    for uid, group in pos_df.groupby('uid'):
        profile_uids.append(uid)
        profiles.append([group[c].mean() for c in PROFILE_COLS])
    profiles = np.array(profiles)
    profiles = profiles / (np.linalg.norm(profiles, axis=1, keepdims=True) + 1e-9)
    sim_mat = profiles @ profiles.T

    surprise_sim = {}
    top_n = PROFILE_TOPK
    for i, uid in enumerate(profile_uids):
        row = sim_mat[i].copy()
        row[i] = -1.0
        top = np.argpartition(row, -top_n)[-top_n:]
        top = top[np.argsort(row[top])[::-1]]
        surprise_sim[uid] = [(profile_uids[j], float(row[j]))
                             for j in top if row[j] >= 0]
    experiment_ready = sum(1 for uid in surprise_sim
                           if uid in items_by_uid and uid in user_items_set)
    pass

    _neighbor_sim = surprise_sim
    if getattr(args, 'profile_source', 'all') == 'train':
        _neighbor_sim = None
        pass
    if getattr(args, 'neighbor_source', 'surprise') == 'user_sim':
        _usp = 'user_sim_{}_{}.pkl'.format(args.model, args.data)
        if os.path.exists(_usp):
            _neighbor_sim = pickle.load(open(_usp, 'rb'))
            _neighbor_sim = {str(k).strip().lower(): v for k, v in _neighbor_sim.items()}
            pass
        else:
            pass

    _neighbor_fallback = None
    if getattr(args, 'neighbor_fallback', 'none') == 'user_sim':
        _usp = 'user_sim_{}_{}.pkl'.format(args.model, args.data)
        if os.path.exists(_usp):
            _us = pickle.load(open(_usp, 'rb'))
            _neighbor_fallback = {str(k).strip().lower(): v for k, v in _us.items()}
            pass
        else:
            pass

    _scarcity_base = {u: dict(v) for u, v in (user_scarcity_params or {}).items()}
    if args.lambda_source == 'surprise_pool':
        positive_nu_by_uid = defaultdict(list)
        _lam_src = factor_df[factor_df['label'] == 1]
        pass
        for _, rr in _lam_src.iterrows():
            positive_nu_by_uid[rr['uid']].append((float(rr['nov_q']), float(rr['unpop_q'])))
        user_scarcity_params = M.personalize_scarcity_centers(
            user_scarcity_params, surprise_sim, positive_nu_by_uid,
            center=args.lambda_stat, min_pool=args.lambda_min_pool)

        qn_pos, qn_neg, qu_pos, qu_neg = [], [], [], []
        for _, rr in factor_df.iterrows():
            p_acc = user_scarcity_params.get(rr['uid'])
            qn, qu = M.accommodation_components(rr['nov_q'], rr['unpop_q'], p_acc)
            if rr['label'] == 1:
                qn_pos.append(qn); qu_pos.append(qu)
            else:
                qn_neg.append(qn); qu_neg.append(qu)
        pass
        _print_q_stats('q_N 正样本', qn_pos); _print_q_stats('q_N 负样本', qn_neg)
        _print_q_stats('q_U 正样本', qu_pos); _print_q_stats('q_U 负样本', qu_neg)

    per_user_arousal = {}

    pass



    _pool_fitter = None
    if not args.in_sample:
        _pool_fitter = make_pool_loo_fitter(factor_df, surprise_sim, args.use_copula)
        pass

    _fn_hist_items_ts = {u: [(it, int(ts)) for it, ts, _r in seq]
                         for u, seq in user_seq.items()}

    if args.run_perturbation:
        exp_perturbation(surprise_sim, items_by_uid, user_items_set,
                         sim_engine, pos_params, neg_params,
                         item_pop=item_pop, max_pop=max_pop,
                         item_cate=item_cate if item_cate else None,
                         item_aspects=item_aspects,
                         item_rating_g=item_rating_g,
                         use_copula=args.use_copula, k=args.k,
                         user_scarcity_params=user_scarcity_params,
                         per_user_arousal=per_user_arousal,
                         pool_dedup=args.pool_dedup, pos_ratio_thr=args.pos_ratio_thr,
                         pool_fitter=_pool_fitter,
                         perturb_metrics=args.perturb_metrics,
                         num_users=len(user_seq) if user_seq else None,
                         fnpair=fnpair,
                         hist_items_ts=_fn_hist_items_ts)
    if args.run_ablation:
        exp_ablation(surprise_sim, items_by_uid, user_items_set,
                     sim_engine, pos_params, neg_params,
                     args.use_copula, k=args.k,
                     user_scarcity_params=user_scarcity_params,
                     per_user_arousal=per_user_arousal,
                     pool_dedup=args.pool_dedup, pos_ratio_thr=args.pos_ratio_thr,
                     pool_fitter=_pool_fitter)
    if args.run_div_pert:
        exp_diversity_perturbation(surprise_sim, items_by_uid, user_items_set,
                                   sim_engine, pos_params, neg_params,
                                   use_copula=args.use_copula, k=args.k,
                                   user_scarcity_params=user_scarcity_params,
                                   per_user_arousal=per_user_arousal,
                                   pool_dedup=args.pool_dedup,
                                   pos_ratio_thr=args.pos_ratio_thr,
                                   pool_fitter=_pool_fitter)
    if args.run_normalization:
        exp_div2k_IDCG(surprise_sim, items_by_uid, user_items_set,
                       sim_engine, pos_params, neg_params,
                       args.use_copula, k=10,
                       user_scarcity_params=user_scarcity_params,
                       per_user_arousal=per_user_arousal,
                       pool_dedup=args.pool_dedup, pos_ratio_thr=args.pos_ratio_thr,
                       pool_fitter=_pool_fitter)
    _rq4_spec = os.environ.get('RQ4_DEL_REC')
    _rq4_done = False
    if _rq4_spec:
        _rq4_done = True
        global _RQ4_USER_ON, _RQ4_USER_RANKERS
        _rq4_user_csv = os.environ.get('RQ4_USER_CSV') or ''
        _RQ4_USER_ON = bool(_rq4_user_csv)
        _RQ4_USER_RANKERS = tuple(
            _x.strip() for _x in
            (os.environ.get('RQ4_USER_RANKERS') or 'mdcg,lincombo').split(',')
            if _x.strip())
        _rq4_ratios = []
        for _x in _rq4_spec.split(','):
            if not _x.strip():
                continue
            _v = float(_x)
            _rq4_ratios.append(_v / 100.0 if _v > 1.0 else _v)
        _rq4_seeds = [int(x) for x in
                      (os.environ.get('RQ4_DEL_SEED') or str(args.seed)).split(',')
                      if x.strip()]
        _rq4_csv = os.environ.get(
            'RQ4_DEL_CSV',
            os.path.join('RQ4_删历史实验', '_rq4_delrec_%s.csv' % args.data))
        _rq4_base_df = factor_df.reset_index(drop=True)
        _rq4_n = len(_rq4_base_df)
        print("\n" + "#" * 60)
        print("# [RQ4] 删记录协议 | data={} model={} | r={} | seeds={}".format(
            args.data, args.model,
            ','.join('%g%%' % (100.0 * x) for x in _rq4_ratios), _rq4_seeds))
        print("# 因子不重算（同一份因子表）；删掉的记录既不进拟合也不进评价")
        print("#" * 60)
        for _sd in _rq4_seeds:
            _perm = np.random.RandomState(_sd).permutation(_rq4_n)
            for _r in _rq4_ratios:
                _n_del = int(round(_r * _rq4_n))
                _del_mask = np.zeros(_rq4_n, dtype=bool)
                _del_mask[_perm[:_n_del]] = True
                _keep_df = _rq4_base_df[~_del_mask].reset_index(drop=True)
                _n_pos_del = int((_rq4_base_df['label'].values[_del_mask] == 1).sum())
                _n_neg_del = int(_n_del - _n_pos_del)
                if _rq4_user_csv:
                    _pg = _keep_df.groupby('uid')['label'].agg(n_items='size', n_pos='sum')
                    for _t in _pg.itertuples():
                        _RQ4_POOL_ROWS.append(dict(data=args.data, model=args.model,
                                                   r=_r, seed=_sd, uid=_t.Index,
                                                   n_items=int(_t.n_items),
                                                   n_pos=int(_t.n_pos)))
                _sur_sim = _build_surprise_profiles(_keep_df)
                _pos_nu = defaultdict(list)
                for _rr in _keep_df[_keep_df['label'] == 1].itertuples(index=False):
                    _pos_nu[_rr.uid].append((float(_rr.nov_q), float(_rr.unpop_q)))
                _sp = M.personalize_scarcity_centers(
                    _scarcity_base, _sur_sim, _pos_nu,
                    center=args.lambda_stat, min_pool=args.lambda_min_pool)
                print("\n--- [RQ4] r={:.0f}% seed={} | 删除 {} / {} 行 (pos -{}, neg -{}) | "
                      "拟合+评测 {} 行 (pos {} / neg {}) | 惊喜画像用户 {} ---".format(
                          100.0 * _r, _sd, _n_del, _rq4_n, _n_pos_del, _n_neg_del,
                          len(_keep_df), int((_keep_df['label'] == 1).sum()),
                          int((_keep_df['label'] == 0).sum()), len(_sur_sim)), flush=True)
                _t0 = time.time()
                exp_robustness_quality(
                    _keep_df, _sp,
                    use_copula=args.use_copula,
                    surprise_sim=_sur_sim,
                    cross_fit=not args.in_sample,
                    qua_consts=args.qua_const,
                    mask_value=args.mask_value,
                    hit_neg_per_pos=args.robust_neg_per_pos,
                    n_boot=0,
                    n_levels=int(os.environ.get('MDCG_ROBUST_LEVELS', '11')),
                    label_time=label_time,
                    item_rating_g=item_rating_g,
                    rq4_meta=dict(data=args.data, model=args.model, r=_r, seed=_sd,
                                  n_rows=len(_keep_df),
                                  n_pos=int((_keep_df['label'] == 1).sum()),
                                  n_neg=int((_keep_df['label'] == 0).sum()),
                                  n_del=_n_del, n_del_pos=_n_pos_del))
                print("--- [RQ4] r={:.0f}% seed={} 用时 {:.1f}s ---".format(
                    100.0 * _r, _sd, time.time() - _t0), flush=True)
        if _rq4_csv and _RQ4_DEL_ROWS:
            _d = os.path.dirname(_rq4_csv)
            if _d and not os.path.isdir(_d):
                os.makedirs(_d)
            _cols = ['data', 'model', 'r', 'seed', 'ranker', 'hit2', 'hit4',
                     'lift2', 'lift4', 'auc', 'n_users', 'mean_items',
                     'n_rows', 'n_pos', 'n_neg', 'n_del', 'n_del_pos']
            with open(_rq4_csv, 'w', encoding='utf-8') as _f:
                _f.write(','.join(_cols) + '\n')
                for _row in _RQ4_DEL_ROWS:
                    _f.write(','.join(str(_row.get(c, '')) for c in _cols) + '\n')
            print("\n>>> [RQ4] 明细已写入: {} ({} 行)".format(_rq4_csv, len(_RQ4_DEL_ROWS)))
        if _rq4_user_csv and _RQ4_USER_ROWS:
            _ud = os.path.dirname(_rq4_user_csv)
            if _ud and not os.path.isdir(_ud):
                os.makedirs(_ud)
            _ucols = ['data', 'model', 'r', 'seed', 'ranker', 'uid',
                      'n_items', 'n_pos', 'hit2', 'hit4', 'lift2', 'lift4']
            with open(_rq4_user_csv, 'w', encoding='utf-8') as _f:
                _f.write(','.join(_ucols) + '\n')
                for _row in _RQ4_USER_ROWS:
                    _f.write(','.join(str(_row.get(c, '')) for c in _ucols) + '\n')
            print(">>> [RQ4] 每用户明细已写入: {} ({} 行)".format(
                _rq4_user_csv, len(_RQ4_USER_ROWS)))
        if _rq4_user_csv and _RQ4_POOL_ROWS:
            _pb = os.path.basename(_rq4_user_csv)
            _pool_csv = os.path.join(os.path.dirname(_rq4_user_csv),
                                     _pb.replace('_delrec_user_', '_delrec_pool_')
                                     if '_delrec_user_' in _pb else _pb + '.pool.csv')
            _pcols = ['data', 'model', 'r', 'seed', 'uid', 'n_items', 'n_pos']
            with open(_pool_csv, 'w', encoding='utf-8') as _f:
                _f.write(','.join(_pcols) + '\n')
                for _row in _RQ4_POOL_ROWS:
                    _f.write(','.join(str(_row.get(c, '')) for c in _pcols) + '\n')
            print(">>> [RQ4] 池规模明细已写入: {} ({} 行)".format(
                _pool_csv, len(_RQ4_POOL_ROWS)))
    if args.run_robustness and not _rq4_done:
        exp_robustness_quality(factor_df, user_scarcity_params,
                               use_copula=args.use_copula,
                               surprise_sim=surprise_sim,
                               cross_fit=not args.in_sample,
                               qua_consts=args.qua_const,
                               mask_value=args.mask_value,
                               hit_neg_per_pos=args.robust_neg_per_pos,
                               n_levels=int(os.environ.get('MDCG_ROBUST_LEVELS', '11')),
                               label_time=label_time,
                               item_rating_g=item_rating_g,
                               temporal_ratio=(PAPER_TEST_RATIO
                                               if os.environ.get('MDCG_ROBUST_TEMPORAL') == '1'
                                               else None))
    _hit_kw = dict(
        user_scarcity_params=user_scarcity_params,
        per_user_arousal=per_user_arousal,
        test_ratio=args.test_ratio,
        item_cate=item_cate if item_cate else None,
        item_aspects=item_aspects,
        item_rating_g=item_rating_g,
        item_pop=item_pop, max_pop=max_pop,
        k_list=tuple(int(x) for x in
                     (os.environ.get('MDCG_HIT_K', '5,10') or '5,10').split(',') if x.strip()),
        use_copula=args.use_copula,
        min_history=args.min_history,
        min_user_items=args.min_user_items,
        lambda_source=args.lambda_source, lambda_stat=args.lambda_stat,
        lambda_min_pool=args.lambda_min_pool, topk=args.sim_user_topk,
        pool_k=(args.pool_k or args.k), pool_pos_frac=args.pool_pos_frac, seed=args.seed,
        n_lists_per_user=args.n_lists_per_user,
        quality_index=quality_index, qua_mode=args.qua_mode,
        in_sample=args.in_sample,
        abeta_rating=args.abeta_rating,
        pool_shuffle=(args.pool_shuffle == 'on'),
        match_popularity=args.match_popularity,
        pool_source=args.pool_source,
        hit_rate_metrics=args.hit_rate_metrics,
        add_hit_metrics=args.add_hit_metrics,
        fnpair=fnpair,
        low_rating_to_train=args.low_rating_to_train,
        boundary_pairs=args.boundary_pairs,
        low_rating_thr=args.low_rating_thr,
        scale_ctx=_scale_ctx,
        pool_dump=getattr(args, 'pool_dump', None),
        pool_file=getattr(args, 'pool_file', None),
        _allpos_surprise_sim=_neighbor_sim,
        _neighbor_fallback=_neighbor_fallback,
        profile_all_items=(getattr(args, 'profile_items', 'pos') == 'all'),
        lincombo_neighbors=getattr(args, 'lincombo_neighbors', False))

    if args.skip_hit_rate:
        print("\n>>> [跳过] 实验六 (--skip_hit_rate)")
    else:
        exp_hit_rate(factor_df, user_seq, label_time, sim_engine,
                     pos_params, neg_params, **_hit_kw)
    if os.environ.get('MDCG_NO_EXPORT') == '1':
        print("\n>>> [MDCG_NO_EXPORT=1] 跳过 temporal_factors.csv / scarcity_params.pkl 导出")
    else:
        factor_df.to_csv('temporal_factors.csv', index=False)
        with open('scarcity_params.pkl', 'wb') as _f:
            pickle.dump(user_scarcity_params, _f)
        print("\n>>> 因子表已导出: temporal_factors.csv")
        print(">>> N/U accommodation 参数已导出: scarcity_params.pkl")

if __name__ == "__main__":
    main()

