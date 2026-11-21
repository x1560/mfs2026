# coding=utf-8
"""
  1. 输出文件名改成{kind}_{model}_{data}（emb_Seren_S2018.pkl 等）；
  2. 输出 factors_Seren_S2018.pkl
  3. qua 用 training.csv 的「截至该条交互时刻」物品均分
  4. 低评分负样本默认关闭，用 --low_rating_neg 打开

输入 (data/serendipity-sac2018/)：
    answers.csv        调查答案 + 惊喜标注（2150 条 / 481 用户 / 1678 物品）
    training.csv       历史评分（9,997,850 条 / 104,661 用户 / 49,151 物品）
输出（当前目录）：
    SerenLens_S2018.csv          eval.py 的 --serenlens 输入
    emb_Seren_S2018.pkl          物品嵌入（rating 矩阵 SVD）
    pop_Seren_S2018.pkl          物品流行度
    cate_Seren_S2018.pkl         物品类别（嵌入 KMeans 类别，供 αβ-nDCG 用）
    factors_Seren_S2018.pkl      时序因子表（rel/dif/novelty/unpopularity/qua）

用法：
    python build_s2018_data.py
    python build_s2018_data.py --label_mode all8
    python build_s2018_data.py --low_rating_neg          # 打开低评分负样本
    python build_s2018_data.py --history_scope all       # CSV 含全部 1000 万条（加载很慢）
"""
import argparse
import os
import pickle
import time
from bisect import bisect_right
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import svds
from sklearn.cluster import KMeans

from eval import SimEngine, build_factor_table

LABEL_COLS_PAPER6 = ["s_ser_find", "s_ser_imp", "s_ser_rec",
                     "m_ser_find", "m_ser_imp", "m_ser_rec"]
# 八列全取并集
LABEL_COLS_ALL8 = ["s_ser_rel", "s_ser_find", "s_ser_imp", "s_ser_rec",
                   "m_ser_rel", "m_ser_find", "m_ser_imp", "m_ser_rec"]

# （Li Kang et al., RecSys'25, https://github.com/Leah-HKBU/SerenEva）的
# Process_Data/data_process.ipynb：
#     df['seren'] = df[['s4', 's5', 's6']].mean(axis=1).round()
LABEL_COLS_UNEXP3 = ["s4", "s5", "s6"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="./data/serendipity-sac2018")
    p.add_argument("--model", default="SEREN")
    p.add_argument("--data", default="S2018")
    p.add_argument("--emb_mode", choices=["binary_idf", "rating"], default="binary_idf")
    p.add_argument("--dim", type=int, default=64, help="SVD 维度，与论文 d=64 对齐")
    p.add_argument("--n_cate", type=int, default=10, help="KMeans 伪类别数")
    p.add_argument("--label_mode", choices=["unexp3", "paper6", "all8", "single"], default="unexp3",
                   help="unexp3=三个意外性陈述(s4,s5,s6)求平均后 round, >3 判惊喜；"
                        "paper6=原论文六种定义取并集；all8=八列并集；single=单列")
    p.add_argument("--single_label", default="s_ser_rel", help="be used when label_mode=single")
    p.add_argument("--unexp_cols", default="s4,s5,s6", help="be used when label_mode=unexp3")
    p.add_argument("--unexp_thr", type=float, default=3.0, help="the threshold of label_mode=unexp3")
    p.add_argument("--unexp_no_round", action="store_true")
    p.add_argument("--csv_out", default=None,
                   help="the output name --serenlens CSV。default: SerenLens_{data}.csv")
    p.add_argument("--low_rating_neg", action="store_true", help="把 training 中 rating<=--low_rating_thr 的交互当负样")
    p.add_argument("--low_rating_thr", type=float, default=2.5)
    p.add_argument("--history_scope", choices=["surveyed", "all"], default="surveyed",
                   help="CSV 纳入哪些用户的历史：surveyed=仅被调查的 481 用户；all=全部 10M 行")
    p.add_argument("--min_history", type=int, default=2, help="the min_history which are used to build factor table")
    return p.parse_args()


def to_bool(s):
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])


def build_labels(answers, mode, single_label, unexp_cols="s4,s5,s6",
                 unexp_thr=3.0, unexp_round=True):
    if mode == "unexp3":
        cols = [c.strip() for c in str(unexp_cols).split(",") if c.strip()]
        missing = [c for c in cols if c not in answers.columns]
        if missing:
            raise ValueError("answers.csv need cols: {}".format(missing))
        score = answers[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        key = score.round() if unexp_round else score
        lab = (key > unexp_thr).fillna(False).astype(int)
        print("    mode: unexp3 | col: {} | {} > {} | pos {} / {} | NA {}".format(
            ", ".join(cols), "round" if unexp_round else "original mean",
            unexp_thr, int(lab.sum()), len(lab), int(score.isna().sum())))
        return lab
    if mode == "paper6":
        cols = LABEL_COLS_PAPER6
    elif mode == "all8":
        cols = LABEL_COLS_ALL8
    else:
        cols = [single_label]
    missing = [c for c in cols if c not in answers.columns]
    if missing:
        raise ValueError("answers.csv need cols: {}".format(missing))
    lab = pd.DataFrame({c: to_bool(answers[c]) for c in cols}).any(axis=1).astype(int)
    print("    mode {} | col: {}".format(mode, ", ".join(cols)))
    return lab


class QualityIndex(object):
    """物品的「截至t」平均分，用 training.csv 构建。
    物品没有更早评分时返回 0.5。
    """
    def __init__(self, training):
        t0 = time.time()
        d = training[["item_id", "timestamp", "rating"]].sort_values(["item_id", "timestamp"])
        d = d.reset_index(drop=True)
        d["cum"] = d.groupby("item_id")["rating"].cumsum()
        d["cnt"] = d.groupby("item_id").cumcount() + 1
        self.ts, self.cum, self.cnt = {}, {}, {}
        for item, g in d.groupby("item_id", sort=False):
            self.ts[item] = g["timestamp"].to_numpy()
            self.cum[item] = g["cum"].to_numpy()
            self.cnt[item] = g["cnt"].to_numpy()
        self.fallback = 0.5
        print("    {} items, time used {:.1f}s".format(len(self.ts), time.time() - t0))

    def mean_at(self, item, ts, shrink_k=0.0):
        t = self.ts.get(item)
        if t is None:
            return self.fallback
        i = bisect_right(t, ts)
        if i == 0:
            return self.fallback
        s, c = self.cum[item][i - 1], self.cnt[item][i - 1]
        if shrink_k > 0:
            s, c = s + shrink_k * 3.0, c + shrink_k
        return float(np.clip((s / c - 1.0) / 4.0, 0.0, 1.0))


def to_serenlens_tables(df):
    user_seq, labels, label_time, item_rating = defaultdict(list), defaultdict(dict), {}, {}
    for r in df.itertuples(index=False):
        uid, iid = r.user_id, r.item_id
        ts = max(0, int(r.timestamp))
        rating = float(min(5.0, max(1.0, float(r.rating))))
        user_seq[uid].append((iid, ts, rating))
        lab = int(r.label)
        if lab in (0, 1):
            labels[uid][iid] = lab
            label_time[(uid, iid)] = ts
            item_rating[(uid, iid)] = rating
    for uid in user_seq:
        user_seq[uid].sort(key=lambda x: x[1])
    return dict(user_seq), dict(labels), label_time, item_rating


def main():
    args = parse_args()
    tag = "{}_{}".format(args.model, args.data)
    answers_path = os.path.join(args.data_dir, "answers.csv")
    training_path = os.path.join(args.data_dir, "training.csv")

    print("loading Serendipity2018 ...")
    answers = pd.read_csv(answers_path, low_memory=False)
    training = pd.read_csv(training_path, low_memory=False,
                           dtype={"userId": np.int32, "movieId": np.int32,
                                  "rating": np.float32, "timestamp": np.int64})
    answers = answers.rename(columns={"movieId": "item_id", "userId": "user_id"})
    training = training.rename(columns={"movieId": "item_id", "userId": "user_id"})
    for df in (answers, training):
        df["user_id"] = df["user_id"].astype(str).str.strip()
        df["item_id"] = df["item_id"].astype(str).str.strip()
    answers["timestamp"] = pd.to_numeric(answers["timestamp"], errors="coerce").astype(np.int64)
    answers["rating"] = pd.to_numeric(answers["rating"], errors="coerce").astype(float)
    answers = answers.dropna(subset=["user_id", "item_id", "timestamp", "rating"])
    answers["label"] = build_labels(answers, args.label_mode, args.single_label,
                                    args.unexp_cols, args.unexp_thr, not args.unexp_no_round)

    surveyed = set(answers["user_id"].unique())
    print("    answers: {} 条 | {} 用户 | {} 物品 | 正 {} 负 {}".format(
        len(answers), answers["user_id"].nunique(), answers["item_id"].nunique(),
        int((answers["label"] == 1).sum()), int((answers["label"] == 0).sum())))
    print("    training: {} 条 | {} 用户 | {} 物品".format(
        len(training), training["user_id"].nunique(), training["item_id"].nunique()))

    hist = training[["user_id", "item_id", "timestamp", "rating"]].copy()
    if args.history_scope == "surveyed":
        hist = hist[hist["user_id"].isin(surveyed)].copy()
        print("    被调查用户的全部历史 {} 条".format(len(hist)))
    else:
        print("    历史范围: {} 条".format(len(hist)))
    hist["label"] = -1
    ans_out = answers[["user_id", "item_id", "timestamp", "rating", "label"]]
    ans_out = ans_out.sort_values("timestamp").drop_duplicates(["user_id", "item_id"], keep="last")

    labeled = ans_out
    if args.low_rating_neg:
        low = hist[hist["rating"] <= args.low_rating_thr].copy()
        low["label"] = 0
        low = low[["user_id", "item_id", "timestamp", "rating", "label"]]
        print("    低评分负样本: {} 条 (rating<={})".format(len(low), args.low_rating_thr))
        labeled = pd.concat([ans_out, low], ignore_index=True)

    full = pd.concat([labeled, hist], ignore_index=True)
    full = full.sort_values("label", ascending=False).drop_duplicates(["user_id", "item_id"], keep="first")
    full = full.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    csv_path = args.csv_out or "SerenLens_{}.csv".format(args.data)
    full.to_csv(csv_path, index=False)
    print("    {}: {} 条（标注 {} | 正 {} | 负 {} | 历史 {}）".format(
        csv_path, len(full), int((full["label"] >= 0).sum()),
        int((full["label"] == 1).sum()), int((full["label"] == 0).sum()), int((full["label"] == -1).sum())))

    print("\n item's embedding (mode={}, dim={}) ...".format(args.emb_mode, args.dim))
    t0 = time.time()
    mat = training.groupby(["user_id", "item_id"], as_index=False).agg({"rating": "mean"})
    users = sorted(mat["user_id"].unique())
    items = sorted(mat["item_id"].unique())
    uidx = {u: i for i, u in enumerate(users)}
    iidx = {it: i for i, it in enumerate(items)}
    ur = mat["user_id"].map(uidx).values
    ic = mat["item_id"].map(iidx).values
    if args.emb_mode == "binary_idf":
        M = csr_matrix((np.ones(len(mat)), (ur, ic)), shape=(len(users), len(items)))
        M.data = np.ones_like(M.data)
        user_freq = np.asarray(M.sum(axis=1)).flatten()
        idf = np.log(len(items) / (user_freq + 1))
        M = M.T.tocsr().astype(float).dot(diags(idf))      # (items x users), 加权
    else:
        M = csr_matrix((mat["rating"].astype(float).values, (ur, ic)),
                       shape=(len(users), len(items))).T.tocsr().astype(float)
    dim = int(min(args.dim, min(M.shape) - 1))
    item_vectors, singular_values, _ = svds(M, k=dim)
    order = np.argsort(singular_values)[::-1]
    item_emb = item_vectors[:, order] * np.sqrt(singular_values[order])
    item_emb = item_emb / (np.linalg.norm(item_emb, axis=1, keepdims=True) + 1e-9)
    emb = {items[i]: item_emb[i] for i in range(len(items))}
    with open("emb_{}.pkl".format(tag), "wb") as f:
        pickle.dump(emb, f, protocol=pickle.HIGHEST_PROTOCOL)
    miss = set(ans_out["item_id"]) - set(emb)
    print("    emb_{}.pkl: {} 物品, dim={}, {:.1f}s".format(tag, len(emb), dim, time.time() - t0))
    if miss:
        n_rows = int(ans_out["item_id"].isin(miss).sum())
        print("{} 个标注物品不在 training 中（涉及 {} 条标注, {:.2f}%），rel/dif 退化为 0".format(len(miss), n_rows, 100.0 * n_rows / len(ans_out)))
    print("\n popularity ...")
    pop = training.groupby("item_id").size().astype(int).to_dict()
    pop = {str(k): int(v) for k, v in pop.items()}
    with open("pop_{}.pkl".format(tag), "wb") as f:
        pickle.dump(pop, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("    pop_{}.pkl: {} 物品".format(tag, len(pop)))

    print("\n>>> KMeans ...")
    n_cate = int(min(args.n_cate, max(2, len(items) // 10), len(items)))
    km = KMeans(n_clusters=n_cate, random_state=42, n_init=10)
    cl = km.fit_predict(item_emb)
    cate = {items[i]: "c{}".format(cl[i]) for i in range(len(items))}
    with open("cate_{}.pkl".format(tag), "wb") as f:
        pickle.dump(cate, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("    cate_{}.pkl: {} 物品, {} 类".format(tag, len(cate), n_cate))

    print("\n>>> [4/4] temporal factor table ...")
    t0 = time.time()
    user_seq, labels, label_time, item_rating = to_serenlens_tables(full)
    print("    user_seq: {} users | labeled users {}".format(len(user_seq), len(labels)))
    sim_engine = SimEngine(emb)
    item_ts_first, ts_max, ts_min = {}, 0, float("inf")
    for seq in user_seq.values():
        for it, ts, _r in seq:
            if it not in item_ts_first or ts < item_ts_first[it]:
                item_ts_first[it] = ts
            ts_max = max(ts_max, ts)
            ts_min = min(ts_min, ts)
    qidx = QualityIndex(training)
    factor_df = build_factor_table(
        user_seq, labels, label_time, item_rating, sim_engine, pop, max(pop.values()),
        item_ts_first=item_ts_first, ts_global_max=ts_max, ts_global_min=ts_min,
        min_history=args.min_history, quality_index=qidx)
    factor_df.to_pickle("factors_{}.pkl".format(tag))
    print("    factors_{}.pkl: {} rows (pos={}, neg={}) | qua mean {:.3f} | {:.1f}s".format(
        tag, len(factor_df), int((factor_df["label"] == 1).sum()),
        int((factor_df["label"] == 0).sum()), float(factor_df["qua"].mean()), time.time() - t0))
    for c in ["rel", "dif", "novelty", "unpopularity", "qua"]:
        print("      {}: mean {:.4f} | p05 {:.4f} | p50 {:.4f} | p95 {:.4f}".format(
            c, factor_df[c].mean(), factor_df[c].quantile(.05),
            factor_df[c].median(), factor_df[c].quantile(.95)))

    print("Next step: python eval.py --serenlens SerenLens_{}.csv --data {} --model {} --use_copula "
          "--trust_shape ushape --trust_alpha 8".format(args.data, args.data, args.model))


if __name__ == "__main__":
    main()
