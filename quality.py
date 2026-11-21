# coding=utf-8
"""
qua: 物品平均得分。

qua = (avg - 1) / 4 归一到 [0, 1]。

本模块负责把平均分按【时间】整理好, 使 eval.py 在任何时刻 t 取到的都是「截至 t 为止」的均值。
------------------
"""
import os
import csv
import time
import pickle
import bisect

import numpy as np

# 各数据集的 Amazon 源文件与列序。
AMAZON_SPECS = {
    'Movies': {
        'filename': 'Movies_and_TV.csv',
        'item_col': 0,      # Amazon 2018: item, user, rating, timestamp
        'rating_col': 2,
        'ts_col': 3,
    },
    'Books': {
        'filename': 'ratings_Books.csv',
        'item_col': 1,      # Amazon 2014: user, item, rating, timestamp
        'rating_col': 2,
        'ts_col': 3,
    },
    'S2018': None,
}

DEFAULT_AMAZON_DIR = os.path.join('data', 'amazon')

#   v1: {'_schema': 1, 'table': {item: (ts_list, cum_sum, cum_cnt)}}
_SCHEMA = 1


def default_cache_path(model, data):
    return 'quality_{}_{}.pkl'.format(model, data)


def read_serenlens_items(serenlens_path):
    """读取 SerenLens 的全部 distinct 物品 id, 小写后返回。
    """
    _raise_csv_field_limit()
    items = set()
    with open(serenlens_path, 'r', newline='', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) >= 2:
                it = row[1].strip().lower()
                if it:
                    items.add(it)
    return items


def _raise_csv_field_limit():
    try:
        csv.field_size_limit(2 ** 31 - 1)
    except OverflowError:
        csv.field_size_limit(2 ** 30)


def scan_amazon_ratings(amazon_path, wanted_items,
                        item_col=0, rating_col=2, ts_col=3,
                        log=None, log_every=2000000):
    """
    扫描 Amazon 评论, 收集被查询物品的 (时间戳 -> 评分和/条数)。
    返回 {item: {ts: [评分和, 条数]}}。
    """
    _raise_csv_field_limit()
    acc = {}
    n_rows = 0
    n_hit = 0
    t0 = time.time()
    with open(amazon_path, 'r', newline='', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        for row in reader:
            n_rows += 1
            if log and log_every and n_rows % log_every == 0:
                log("    ...已扫描 {} 行, 命中 {} ({:.0f}s)".format(
                    n_rows, n_hit, time.time() - t0))
            if len(row) <= ts_col:
                continue
            item = row[item_col].strip().lower()
            if item not in wanted_items:
                continue
            try:
                rt = float(row[rating_col])
                ts = int(float(row[ts_col]))
            except (ValueError, TypeError):
                continue
            per_item = acc.get(item)
            if per_item is None:
                per_item = {}
                acc[item] = per_item
            bucket = per_item.get(ts)
            if bucket is None:
                per_item[ts] = [rt, 1]
            else:
                bucket[0] += rt
                bucket[1] += 1
            n_hit += 1
    if log:
        log("    扫描完成: {} 行, 命中 {} 条, 覆盖 {} 个物品 ({:.0f}s)".format(
            n_rows, n_hit, len(acc), time.time() - t0))
        if n_rows >= log_every and len(acc) * 2 < len(wanted_items):
            log("    覆盖率低 ({}/{} = {:.1%})".format(
                    len(acc), len(wanted_items),
                    float(len(acc)) / max(1, len(wanted_items))))
    return acc


def _collapse(per_item):
    """{ts: [和, 条数]} -> (时间戳升序, 累计评分和, 累计条数)。
    """
    ts_list = sorted(per_item)
    sums = []
    cnts = []
    s = 0.0
    c = 0
    for ts in ts_list:
        v = per_item[ts]
        s += v[0]
        c += v[1]
        sums.append(s)
        cnts.append(c)
    return ts_list, sums, cnts


def collapse_quality_table(acc):
    """{item: (ts_list, cum_sum, cum_cnt)}。"""
    return {item: _collapse(per_item) for item, per_item in acc.items()}


def _normalize_table(table):
    if not table:
        return table
    for v in table.values():
        if isinstance(v, tuple) and len(v) == 3:
            return table
        break
    return {item: _collapse(per_item) for item, per_item in table.items()}


class QualityIndex(object):
    """
    按时间点查询物品的历史平均评分。stats_at(item, t) 只统计时间戳 < t 的评分。
    """

    def __init__(self, table):
        self._t = _normalize_table(table)

    def __len__(self):
        return len(self._t)

    def __contains__(self, item):
        return item in self._t

    def stats_at(self, item, ts):
        """(累计评分和, 累计条数), 统计时间戳 < ts 的评分。
        未知物品或 t 早于/等于该物品首条评分 -> (0.0, 0)。
        """
        e = self._t.get(item)
        if e is None:
            return 0.0, 0
        ts_list, sums, cnts = e
        i = bisect.bisect_left(ts_list, ts)
        if i == 0:
            return 0.0, 0
        return sums[i - 1], cnts[i - 1]

    def mean_at(self, item, ts, rating_min=1.0, rating_max=5.0,
                shrink_k=0.0, prior=None):
        """
        截至 ts 的平均评分, 归一化到 [0,1]。
        无评分时返回0.5; 
        shrink_k > 0 时用于缓解冷门物品的估计噪声: (sum + k*prior) / (cnt + k)。
        """
        s, c = self.stats_at(item, ts)
        mid = (rating_min + rating_max) / 2.0
        if c <= 0:
            m = mid if prior is None else prior
        else:
            m = s / float(c)
            if shrink_k > 0:
                p = mid if prior is None else prior
                m = (s + shrink_k * p) / (c + shrink_k)
        return float(np.clip((m - rating_min) / (rating_max - rating_min), 0.0, 1.0))

    def batch_stats_at(self, items, ts):
        """向量化批量查询: 返回 (sum 数组, cnt 数组)。"""
        n = len(items)
        sums = np.zeros(n, dtype=float)
        cnts = np.zeros(n, dtype=np.int64)
        for k, it in enumerate(items):
            s, c = self.stats_at(it, ts)
            if c:
                sums[k] = s
                cnts[k] = c
        return sums, cnts


def save_quality_table(path, table, log=None):
    collapsed = _normalize_table(table)
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump({'_schema': _SCHEMA, 'table': collapsed}, f, protocol=4)
    os.replace(tmp, path)
    if log:
        size_mb = os.path.getsize(path) / 1048576.0
        log("    质量表已缓存: {} ({:.1f} MB, {} 物品)".format(path, size_mb, len(collapsed)))


def load_quality_table(path, log=None):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and '_schema' in obj and 'table' in obj:
        table = obj['table']
    else:
        table = obj
    table = _normalize_table(table)
    if log:
        log("    质量表已载入: {} ({} 物品)".format(path, len(table)))
    return table


def build_quality_table(serenlens_path, amazon_dir=DEFAULT_AMAZON_DIR,
                        data='Movies', log=print):
    """
    返回 {item: {ts: [和, 条数]}}。
    """
    spec = AMAZON_SPECS.get(data)
    if spec is None:
        if log:
            log("  quality: 数据集 {} 无 Amazon 对应源, qua 回退到 SerenLens 自身评分".format(data))
        return None
    amazon_path = os.path.join(amazon_dir, spec['filename'])
    if not os.path.exists(amazon_path):
        if log:
            log("  quality: 未找到 {}, qua 回退到 SerenLens 自身评分".format(amazon_path))
        return None

    wanted = read_serenlens_items(serenlens_path)
    if log:
        log("  quality: SerenLens 物品 {} 个, 开始扫描 {}".format(
            len(wanted), spec['filename']))
    return scan_amazon_ratings(
        amazon_path, wanted,
        item_col=spec['item_col'], rating_col=spec['rating_col'], ts_col=spec['ts_col'],
        log=log)


def ensure_quality_index(serenlens_path, model, data,
                         amazon_dir=DEFAULT_AMAZON_DIR,
                         cache_path=None, rebuild=False, log=print):
    if cache_path is None:
        cache_path = default_cache_path(model, data)
    if os.path.exists(cache_path) and not rebuild:
        return QualityIndex(load_quality_table(cache_path, log=log))
    acc = build_quality_table(serenlens_path, amazon_dir=amazon_dir, data=data, log=log)
    if acc is None:
        return None
    if acc:
        save_quality_table(cache_path, acc, log=log)
    return QualityIndex(collapse_quality_table(acc))
