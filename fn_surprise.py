# -*- coding: utf-8 -*-
"""Fu & Niu (ACM TKDD 18(1), 2024) 
    Eq 1  用 **PMI** 衡量"两个 label 一起出现有多不可能" —— pairwise surprise
              s(e_i,e_j) = -log[ p(e_i,e_j) / (p(e_i)·p(e_j)) ]
          PMI 越低（越不可能共现）-> s 越高 -> 越惊喜。
    Eq 2  每个物品取 s 最高的那个 label 对 (e1,e2) 作为它的特征对。
    Eq 3  个性化因子 P^t_{u,i}：F^t_{u,i} = 用户 u 在时刻 t **之前**访问过的、
          与物品 i 共享那个 label 对的物品数。F 越大 -> 已经见过这类组合 -> 越不惊喜。
    Eq 4  SI^t_{u,i} = 客观惊喜 × P^t_{u,i}   （个性化惊喜 = 刺激强度）

"""
import os
import math
import pickle
from collections import defaultdict
from itertools import combinations

ROOT = os.environ.get('MDCG_ROOT') or os.path.dirname(os.path.abspath(__file__))

TAG = {'Books': 'Seren_Books', 'Movies': 'AMZ_Movies', 'S2018': 'SEREN_S2018'}


def load_labels(data):
    """{item: [label, ...]}"""
    p = os.path.join(ROOT, 'clustlab_%s.pkl' % TAG[data])
    with open(p, 'rb') as f:
        return pickle.load(f)


# --------------------------------------------------------------------- Eq 1
def build_pair_surprise(labels_by_item):
    """Eq 1: 对共现的 label 对算 PMI，取负号，越大越惊喜。

    返回 (pair_s, label_p)：pair_s[(a,b)] = s，label_p[a] = p(a)
    """
    n_item = 0
    p_pair = defaultdict(int)
    p_single = defaultdict(int)
    for labels in labels_by_item.values():
        L = sorted(set(labels))
        if len(L) < 2:
            p_single[L[0]] += 1 if L else 0
            n_item += 1
            continue
        for a in L:
            p_single[a] += 1
        for a, b in combinations(L, 2):
            p_pair[(a, b)] += 1
        n_item += 1
    if n_item == 0:
        return {}, {}
    p_single = {k: v / n_item for k, v in p_single.items()}
    pair_s = {}
    for (a, b), c in p_pair.items():
        p_ab = c / n_item
        denom = p_single.get(a, 0.0) * p_single.get(b, 0.0)
        if denom <= 0:
            continue
        pair_s[(a, b)] = -math.log(p_ab / denom)      # = -PMI
    return pair_s, p_single


# --------------------------------------------------------------------- Eq 2
def item_pairs(labels_by_item, pair_s):
    """Eq 2: 每个物品取 s 最高的 label 对
    返回 {item: (best_pair, s)}
    """
    out = {}
    for it, labels in labels_by_item.items():
        L = sorted(set(labels))
        if len(L) < 2:
            out[it] = (None, 0.0)
            continue
        best, bs = None, -1e18
        for a, b in combinations(L, 2):
            s = pair_s.get((a, b))
            if s is None:
                continue
            if s > bs:
                bs, best = s, (a, b)
        out[it] = (best, bs if best is not None else 0.0)
    return out


# ------------------------------------------------------------------- Eq 3/4
def personalization(F, form='inv'):
    """Eq 3 的折减因子
    F = F^t_{u,i} = 用户在 t 之前访问过的、与 i 共享那个 label 对的物品数。
    F 越大，越熟悉，折减越狠。
    """
    if form == 'inv':
        return 1.0 / (1.0 + F)
    if form == 'exp':
        return math.exp(-F)
    if form == 'none':
        return 1.0
    raise ValueError('未知的 Eq 3 形式: %s' % form)


def si(objective_s, F, form='inv'):
    """Eq 4: SI = 客观惊喜 × 个性化因子。"""
    return objective_s * personalization(F, form)



def dump_pairs(data):
    """把 Eq 2 的结果存成 pkl，供 eval.py 的排序器直接加载。

    产物: fnpair_{tag}.pkl = {item: (pair, s_obj)}   pair 为 None 表示只有 <2 个 label
    """
    labels = load_labels(data)
    pair_s, _ = build_pair_surprise(labels)
    ip = item_pairs(labels, pair_s)
    out = os.path.join(ROOT, 'fnpair_%s.pkl' % TAG[data])
    with open(out, 'wb') as f:
        pickle.dump(ip, f, protocol=4)
    n_pair = sum(1 for _, (p, _s) in ip.items() if p is not None)
    print('%s -> %s | 物品 %d | 有 label 对 %d (%.1f%%)'
          % (data, out, len(ip), n_pair, 100.0 * n_pair / max(1, len(ip))))
    return ip


if __name__ == '__main__':
    import sys
    args = sys.argv[1:]
    if args and args[0] == '--dump':
        for d in (args[1:] or ['Books', 'Movies', 'S2018']):
            dump_pairs(d)
