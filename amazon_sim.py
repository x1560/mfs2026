# coding=utf-8
"""Amazon 共现余弦相似度
用途: 给 div（列表级多样性折扣）提供**源自 Amazon 全量评分**的物品间相似度，
定义:
    X = 物品 x 用户 的二值矩阵（Amazon 原始评分，谁评过谁）
    sim(i, j) = cos(x_i, x_j) = |U_i ∩ U_j| / sqrt(|U_i| * |U_j|)
    amzcooc_{data}.pkl —— 归一化后的稀疏相似度矩阵 + 索引。
"""
import os
import pickle
import numpy as np

DEFAULT_AMAZON = {
    # Amazon 2018: item, user, rating, timestamp 
    'Movies': ('data/amazon/Movies_and_TV.csv', 0, 1),
    # Amazon 2014: user, item, rating, timestamp 
    'Books': ('data/amazon/ratings_Books.csv', 1, 0),
}


class AmazonCoocSim(object):
    """基于 Amazon 共现的物品相似度。"""

    def __init__(self, items, mat):
        self.items = items
        self.index = {it: i for i, it in enumerate(items)}
        self.mat = mat
        self.emb = self.index

    def sim(self, a, b):
        ia = self.index.get(a)
        ib = self.index.get(b)
        if ia is None or ib is None:
            return 0.0
        return float(self.mat[ia, ib])

    def __len__(self):
        return len(self.items)


def build(data='Movies', amazon_path=None, cache_path=None, log=print,
          min_user_items=1, item_filter=None, topn=None):
    default_path, item_col, user_col = DEFAULT_AMAZON[data]
    path = amazon_path or default_path
    cache = cache_path or 'amzcooc_{}{}.npz'.format(
        data, '' if min_user_items <= 1 else '_u{}'.format(min_user_items))
    if not cache.endswith('.npz'):
        cache += '.npz'
    if os.path.exists(cache):
        log('  amazon cooc: hit cache: {}'.format(cache))
        z = np.load(cache, allow_pickle=True)
        from scipy.sparse import csr_matrix
        mat = csr_matrix((z['data'], z['indices'], z['indptr']), shape=tuple(z['shape']))
        return AmazonCoocSim(list(z['items']), mat)

    log('  amazon cooc: scanning {} ...'.format(path))
    items, users = {}, {}
    uc, ic = [], []                              # user idx / item idx
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        f.readline()
        for line in f:
            p = line.split(',')
            if len(p) < 4:
                continue
            it = p[item_col].strip().lower()
            u = p[user_col].strip().lower()
            if it not in items:
                items[it] = len(items)
            if u not in users:
                users[u] = len(users)
            uc.append(users[u]); ic.append(items[it])
    log('    {} 条评分 | {} 用户 | {} 物品'.format(len(uc), len(users), len(items)))
    if min_user_items > 1:
        from collections import Counter
        cu = Counter(uc)
        keep = np.array([cu[u] >= min_user_items for u in uc])
        uc = list(np.array(uc)[keep]); ic = list(np.array(ic)[keep])
        _u2i = {}; _i2 = {}
        for a, b in zip(uc, ic):
            _u2i.setdefault(a, len(_u2i)); _i2.setdefault(b, None)
        uc = [_u2i[a] for a in uc]
        _inv = {v: k for k, v in users.items()}
        log('    u>={}: 保留 {} 条 / {} 用户'.format(min_user_items, len(uc), len(_u2i)))
        users = {_inv[k]: v for k, v in _u2i.items()}

    if item_filter is not None:
        _inv_all = {v: k for k, v in items.items()}
        _imap, _ic, _uc = {}, [], []
        for a, b in zip(uc, ic):
            if _inv_all[b] not in item_filter:
                continue
            if b not in _imap:
                _imap[b] = len(_imap)
            _ic.append(_imap[b]); _uc.append(a)
        uc, ic = _uc, _ic
        items = {_inv_all[b]: k for b, k in _imap.items()}
        log('    item_filter: 保留 {} 物品 / {} 条'.format(len(items), len(ic)))

    from scipy.sparse import csr_matrix
    X = csr_matrix((np.ones(len(uc), dtype=np.float32), (ic, uc)),
                   shape=(len(items), len(users)))
    X.data = np.ones_like(X.data) 
    norm = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).flatten())
    norm[norm == 0] = 1.0
    Xn = X.multiply(1.0 / norm[:, None]).tocsr()
    S = (Xn @ Xn.T).tocsr()                      # 物品 x 物品 稀疏余弦
    S.setdiag(0.0)
    S.eliminate_zeros()
    if topn:                                   # 每物品只保留 top-N 邻居
        S = _topn_per_row(S, topn)
    inv = {v: k for k, v in items.items()}
    item_list = [inv[i] for i in range(len(inv))]
    log('    相似度矩阵: nnz={} ({:.4f}% 稠密)'.format(
        S.nnz, 100.0 * S.nnz / max(S.shape[0] ** 2, 1)))
    S = S.astype(np.float32)
    np.savez_compressed(cache, items=np.array(item_list, dtype=object),
                        data=S.data, indices=S.indices, indptr=S.indptr,
                        shape=np.array(S.shape))
    log('    -> {}  ({} 物品, nnz={})'.format(cache, S.shape[0], S.nnz))
    return AmazonCoocSim(item_list, S)


def _topn_per_row(S, n):
    from scipy.sparse import csr_matrix as _csr
    indptr, indices, data = [0], [], []
    for r in range(S.shape[0]):
        s, e = S.indptr[r], S.indptr[r + 1]
        if e > s:
            row = S.data[s:e]; col = S.indices[s:e]
            if len(row) > n:
                sel = np.argpartition(row, -n)[-n:]
                row, col = row[sel], col[sel]
            indices.extend(col.tolist()); data.extend(row.tolist())
        indptr.append(len(indices))
    return _csr((np.array(data, dtype=np.float32), np.array(indices), np.array(indptr)),
                shape=S.shape)


def build_emb(data='Movies', min_user_items=20, dim=64, item_filter=None,
              cache_path=None, log=print, min_item_items=None):
    """用 Amazon 评分训练物品 SVD 嵌入。给 div 当相似度用（--div_sim amazon_emb）。
    """
    if data not in DEFAULT_AMAZON:
        raise ValueError('未知数据集: {}'.format(data))
    default_path, item_col, user_col = DEFAULT_AMAZON[data]
    _core = not (min_item_items is None or min_item_items <= 1)
    cache = cache_path or 'amzemb_{}_u{}{}.pkl'.format(
        data, min_user_items, '_i{}'.format(min_item_items) if _core else '')
    if os.path.exists(cache):
        log('  amazon emb: hit cache: {}'.format(cache))
        with open(cache, 'rb') as f:
            return pickle.load(f)
    log('  amazon emb: scanning {} ...'.format(default_path))
    items, users = {}, {}
    uc, ic = [], []
    with open(default_path, 'r', encoding='utf-8', errors='ignore') as f:
        f.readline()
        for line in f:
            p = line.split(',')
            if len(p) < 4:
                continue
            it = p[item_col].strip().lower()
            u = p[user_col].strip().lower()
            if it not in items:
                items[it] = len(items)
            if u not in users:
                users[u] = len(users)
            uc.append(users[u]); ic.append(items[it])
    from collections import Counter
    uc = np.asarray(uc); ic = np.asarray(ic)
    if not _core:
        cu = Counter(uc)
        keep = np.array([cu[u] >= min_user_items for u in uc])
        uc, ic = uc[keep], ic[keep]
    else:
        for _ in range(50):
            cu = Counter(uc); ci = Counter(ic)
            keep = np.array([cu[a] >= min_user_items and ci[b] >= min_item_items
                             for a, b in zip(uc, ic)])
            uc, ic = uc[keep], ic[keep]
            if len(keep) == 0 or keep.all():
                break
    u2 = {}; uc2 = np.empty(len(uc), dtype=np.int64)
    for i, a in enumerate(uc):
        if a not in u2:
            u2[a] = len(u2)
        uc2[i] = u2[a]
    if _core:
        _i2 = {}; _ic2 = np.empty(len(ic), dtype=np.int64)
        for i, b in enumerate(ic):
            if b not in _i2:
                _i2[b] = len(_i2)
            _ic2[i] = _i2[b]
        _inv_i = {v: k for k, v in items.items()}
        items = {_inv_i[b]: k for b, k in _i2.items()}
        ic = _ic2

    if item_filter is not None:
        _inv_all = {v: k for k, v in items.items()}
        _imap, _ic, _uc = {}, [], []
        for a, b in zip(uc2, ic):
            if _inv_all[b] not in item_filter:
                continue
            if b not in _imap:
                _imap[b] = len(_imap)
            _ic.append(_imap[b]); _uc.append(int(a))
        uc2 = np.asarray(_uc, dtype=np.int64); ic = _ic
        items = {_inv_all[b]: k for b, k in _imap.items()}
    log('    u>={}: {} 条 / {} 用户 / {} 物品'.format(
        min_user_items, len(uc2), len(u2), len(items)))
    from scipy.sparse import csr_matrix, diags
    from scipy.sparse.linalg import svds
    R = csr_matrix((np.ones(len(uc2), dtype=np.float32), (uc2, ic)),
                   shape=(len(u2), len(items)))
    R.data = np.ones_like(R.data)
    uf = np.asarray(R.sum(axis=1)).flatten()
    Mx = R.T.tocsr().astype(float).dot(diags(np.log(len(items) / (uf + 1))))
    U, S, _ = svds(Mx, k=min(dim, min(Mx.shape) - 1))
    o = np.argsort(S)[::-1]
    A = U[:, o] * np.sqrt(S[o])
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    inv = {v: k for k, v in items.items()}
    emb = {inv[i]: A[i] for i in range(len(inv))}
    with open(cache, 'wb') as f:
        pickle.dump(emb, f, protocol=pickle.HIGHEST_PROTOCOL)
    log('    -> {} ({} 物品)'.format(cache, len(emb)))
    return emb