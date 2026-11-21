# -*- coding: utf-8 -*-
"""从 McAuley Amazon metadata 里抽出【物品 -> 类别列表】，并量覆盖率。

用途: Fu & Niu (TKDD 2024) 的 surprise 是"同一物品内两个 label 的 PMI"
      (Eq 1-2)，需要每个物品有一个 label bag。Amazon metadata 的
      `categories` 字段正好给这个（形如 [["Sports & Outdoors","Dance"]]）。

输入（自行下载，版本必须与我们的评分文件对应）:
    Books  (2014 版, 22,507,155 条评分)  meta_Books.json.gz
        https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Books.json.gz
    Movies (2018 版, 8,765,568 条评分)   meta_Movies_and_TV.json.gz
        https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_Movies_and_TV.json.gz

输出:
    amzcat_{data}.pkl     {asin: [leaf 类别, ...]}   几百 KB ~ 几 MB
    （大文件只在这一次用到，之后可以删）

用法:
    python _diag/build_amazon_cate.py Books  ./data/amazon/meta_Books.json.gz
    python _diag/build_amazon_cate.py Movies ./data/amazon/meta_Movies_and_TV.json.gz
"""
import os
import sys
import gzip
import json
import pickle
from collections import Counter

ROOT = os.environ.get('MDCG_ROOT') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

# 我们各数据集的 tag（model_data），以及真正参与评测的物品从哪里取
TAGS = {
    'Books':  ('Seren_Books',  'factors_Seren_Books.pkl'),
    'Movies': ('AMZ_Movies',   'factors_AMZ_Movies.pkl'),
    'S2018':  ('SEREN_S2018',  'factors_SEREN_S2018.pkl'),
}


def parse_line(line):
    """2014 版是 loose json（要 eval），2018 版是严格 json。两种都吃。"""
    s = line.strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        try:
            return eval(s)
        except Exception:
            return None


def labels_of(obj, mode='all'):
    """categories 是若干条类别路径。取该物品的 label 集合（去重）。

    mode='all'  → 每条路径的**所有层级**都算 label
                  ["Books","Literature & Fiction","Poetry"] -> {books, literature & fiction, poetry}
                  （2014 版很多书的 categories 只有 [["Books"]]，只取叶子就只剩 1 个 label，
                    所以默认取全部层级，保证 label bag 够大）
    mode='leaf' → 只取每条的叶子（更接近"主题"，但覆盖率低）
    """
    out, seen = [], set()
    for path in (obj.get('categories') or []):
        if not isinstance(path, (list, tuple)) or not path:
            continue
        els = [str(e) for e in path] if mode == 'all' else [str(path[-1])]
        for e in els:
            e = e.strip().lower()
            if e and e not in seen:
                seen.add(e)
                out.append(e)
    return out


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    data, gz_path = sys.argv[1], sys.argv[2]
    if data not in TAGS:
        raise SystemExit('未知数据集: %s（可选 %s）' % (data, '/'.join(TAGS)))
    tag, factors_file = TAGS[data]

    print('>>> 扫描 %s ...' % gz_path)
    if not os.path.exists(gz_path):
        raise SystemExit('找不到文件: %s' % gz_path)
    print('    大小 %.2f GB' % (os.path.getsize(gz_path) / 1e9))

    cat_map = {}
    leaf_map = {}
    n_lines = n_no_cat = n_bad = 0
    per_item = Counter()
    with gzip.open(gz_path, 'rt', encoding='utf-8', errors='replace') as f:
        for line in f:
            n_lines += 1
            obj = parse_line(line)
            if obj is None:
                n_bad += 1
                continue
            asin = str(obj.get('asin', '')).strip().lower()
            if not asin:
                n_bad += 1
                continue
            cates = labels_of(obj, 'all')
            if not cates:
                n_no_cat += 1
                continue
            cat_map[asin] = cates
            leaf_map[asin] = labels_of(obj, 'leaf')
            per_item[len(cates)] += 1
            if n_lines % 500000 == 0:
                print('    已扫 %d 行 | 有类别 %d 个物品' % (n_lines, len(cat_map)))

    print('    扫完: %d 行 | 解析失败 %d | 无类别 %d | **有类别物品 %d**'
          % (n_lines, n_bad, n_no_cat, len(cat_map)))
    # 两种口径的 label 数分布
    for name, m in (('全层级(all)', cat_map), ('只取叶子(leaf)', leaf_map)):
        c = Counter(len(v) for v in m.values())
        multi = sum(1 for v in m.values() if len(v) >= 2)
        print('    [%s] label 数分布 top: %s | >=2 个 label 的物品 %.1f%%'
              % (name, dict(sorted(c.items())[:6]), 100.0 * multi / max(1, len(m))))

    out = 'amzcat_%s.pkl' % data
    with open(out, 'wb') as f:
        pickle.dump(cat_map, f, protocol=4)
    print('    -> %s (%.1f MB)' % (out, os.path.getsize(out) / 1e6))

    # ---- 覆盖率：真正参与评测的物品有多少能查到类别 ----
    print()
    print('>>> 覆盖率 vs %s' % factors_file)
    if not os.path.exists(factors_file):
        print('    找不到 %s，跳过覆盖率' % factors_file)
        return
    import pandas as pd
    fdf = pd.read_pickle(factors_file)
    items = fdf['item'].astype(str).str.strip().str.lower()
    uniq = set(items)
    hit = sum(1 for it in uniq if it in cat_map)
    print('    因子表物品 %d 个 | 查到类别 %d 个 (%.1f%%)' % (len(uniq), hit, 100.0 * hit / max(1, len(uniq))))
    pos = set(items[fdf['label'].values == 1]) if 'label' in fdf.columns else set()
    if pos:
        hp = sum(1 for it in pos if it in cat_map)
        print('    其中正样本 %d 个 | 查到类别 %d 个 (%.1f%%)' % (len(pos), hp, 100.0 * hp / max(1, len(pos))))
    # 每个物品的 label 数分布（Fu & Niu 的 Eq 1 需要 >= 2）
    cnt = Counter(len(cat_map.get(it, [])) for it in uniq)
    print('    label 数分布（物品数）: %s' % dict(sorted(cnt.items())[:8]))
    multi = sum(1 for it in uniq if len(cat_map.get(it, [])) >= 2)
    print('    **>=2 个 label 的物品**: %d / %d (%.1f%%)  <- Eq 1 需要这个'
          % (multi, len(uniq), 100.0 * multi / max(1, len(uniq))))
    top = Counter(c for it in uniq for c in cat_map.get(it, []))
    print('    最常见类别 top10: %s' % top.most_common(10))


if __name__ == '__main__':
    main()
