# coding=utf-8
"""
从 SerenLens 一次生成评估需要的全部数据:
  emb_SEREN.pkl      : 共现SVD嵌入
  pop_SEREN.pkl      : system-level 物品交互次数
  cate_SEREN.pkl     : 类别(用聚类生成类别)
  user_sim_SEREN.pkl : 与该用户相似的前k个用户
  factors_SEREN.pkl  : 每个物品-用户交互对中，该物品的五个因子值
用法: python build_all_data.py [--dim 64] [--n_cate 50]
"""
import argparse
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import svds
from sklearn.cluster import KMeans
from eval import load_serenlens, SimEngine, build_factor_table
 
ap = argparse.ArgumentParser()
ap.add_argument('--serenlens', default='./data/SerenLens/SerenLens_Movies.csv')
ap.add_argument('--data', default='Movies', help="Movies || Books || S2018")
ap.add_argument('--model', default='Seren')
ap.add_argument('--dim', type=int, default=64)
ap.add_argument('--n_cate', type=int, default=10, help='聚类')
# 注意: 这个 --min_history 只决定「能不能算出 rel/dif」(每个交互需要多少条更早的历史),
# 与 eval.py 的 --min_history(拟合 N/U accommodation 参数所需的最少历史)是两个概念。
# 放宽到 2 可回收大量早期交互 —— 那批交互的正样本密度是全局的数倍。
ap.add_argument('--min_history', type=int, default=2, help='计算 rel/dif 所需的最少历史交互数')
ap.add_argument('--sim_top_n', type=int, default=50, help="用于设置user_sim_SEREN.pkl中相似用户的数量")
# qua 的质量数据源: 默认读 Amazon 原始评论(与 SerenLens 同源, 覆盖率 ~98.6%,
# 样本量约为 SerenLens 自身的 90 倍), 按「截至每条交互时刻」的平均分计算。
ap.add_argument('--quality', dest='quality', default='auto', choices=['auto', 'off'],
                help='qua 数据源: auto=优先 Amazon 原始评论(缺失则回退 SerenLens), '
                     'off=不用 Amazon, 只用 SerenLens 自身评分')
ap.add_argument('--amazon_dir', default=None, help='Amazon 原始文件目录 (默认 data/amazon)')
ap.add_argument('--quality_cache', default=None, help='质量表缓存路径')
ap.add_argument('--rebuild_quality', action='store_true', help='忽略缓存重新扫描 Amazon')
ap.add_argument('--quality_shrink', type=float, default=0.0,
                help='质量均值的收缩强度 k: (sum + k*3.0)/(cnt + k)。0=不收缩(默认)')
args = ap.parse_args()

emb_path = 'emb_{}_{}.pkl'.format(args.model, args.data)
pop_path = 'pop_{}_{}.pkl'.format(args.model, args.data)
cate_path = 'cate_{}_{}.pkl'.format(args.model, args.data)
user_sim_path = 'user_sim_{}_{}.pkl'.format(args.model, args.data)
factors_path = 'factors_{}_{}.pkl'.format(args.model, args.data)

print(">>> loading...")
df = pd.read_csv(args.serenlens)
df['user_id'] = df['user_id'].astype(str).str.strip().str.lower()
df['item_id'] = df['item_id'].astype(str).str.strip().str.lower()
users = sorted(df['user_id'].unique())
items = sorted(df['item_id'].unique())
uidx = {u: i for i, u in enumerate(users)}
iidx = {it: i for i, it in enumerate(items)}
print("    {} users, {} items, {} interactions".format(len(users), len(items), len(df)))
 
# ========== 1. 共现SVD嵌入 ==========
print("\n>>> [1/5] 计算共现嵌入...")
rows = df['user_id'].map(uidx).values
cols = df['item_id'].map(iidx).values
UI = csr_matrix((np.ones(len(df)), (rows, cols)), shape=(len(users), len(items)))#u-i矩阵
UI.data = np.ones_like(UI.data)
IU = UI.T.tocsr().astype(float)
user_freq = np.asarray(UI.sum(axis=1)).flatten()
idf = np.log(len(items) / (user_freq + 1))
IU_w = IU.dot(diags(idf))
dim = min(args.dim, min(IU.shape) - 1)
U, S, Vt = svds(IU_w, k=dim)
user_emb = Vt.T * np.sqrt(S)
item_emb = U * np.sqrt(S)
norms = np.linalg.norm(item_emb, axis=1, keepdims=True)
item_emb = item_emb / (norms + 1e-9)
emb_dict = {items[i]: item_emb[i] for i in range(len(items))}
with open(emb_path, 'wb') as f:
    pickle.dump(emb_dict, f)
print("    emb_SEREN_Movies.pkl: {} 物品嵌入 (dim={})".format(len(emb_dict), dim))
 
# ========== 2. 流行度 ==========
print("\n>>> [2/5] 计算流行度...")
pop_dict = df.groupby('item_id').size().to_dict()
with open(pop_path, 'wb') as f:
    pickle.dump(pop_dict, f)
print("    pop_SEREN_Movies.pkl: {} 物品流行度".format(len(pop_dict)))
 
# ========== 3. 对嵌入聚类 ==========
print("\n>>> [3/5] KMeans聚类嵌入...")
n_cate = min(args.n_cate, len(items) // 10)
km = KMeans(n_clusters=n_cate, random_state=42, n_init=10)
labels = km.fit_predict(item_emb)
cate_dict = {items[i]: 'c{}'.format(labels[i]) for i in range(len(items))}
with open(cate_path, 'wb') as f:
    pickle.dump(cate_dict, f)
print("    cate_SEREN_Movies.pkl: {} 物品, {} 个聚类类别".format(len(cate_dict), n_cate))


# ========== 4.用户嵌入余弦相似度 ==========
print("\n>>> [4/5] 计算用户相似度...")
norms = np.linalg.norm(user_emb, axis=1, keepdims=True)
user_emb = user_emb / (norms + 1e-9)    
sim_mat = (user_emb @ user_emb.T)
top_n = args.sim_top_n
user_sim = {}
for idx, uid in enumerate(users):
    row_sim = sim_mat[idx].copy()
    row_sim[idx] = -1.0
    #前n个索引
    top_indices = np.argpartition(row_sim, -top_n)[-top_n:] 
    top_indices = top_indices[np.argsort(row_sim[top_indices])[::-1]]
    user_sim[uid] = [(users[i], float(row_sim[i])) for i in top_indices if row_sim[i] >= 0]

with open(user_sim_path, 'wb') as f:
    pickle.dump(dict(user_sim), f)

print("    user_sim_SEREN_Movies.pkl: {} 用户, 前{} 个相似用户".format(len(users), args.sim_top_n))

# ========== 5. 构建时序因子 (rel/dif/novelty/unpopularity/qua) ==========
print("\n>>> [5/5] 构建时序因子表...")

# qua 的时序质量索引: 截至每条交互时刻的物品平均分(不含未来信息)。
# 数据源优先 Amazon 原始评论(与 SerenLens 同源, 覆盖率 ~98.6%), 缺失时回退 SerenLens。
quality_index = None
if args.quality != 'off':
    try:
        import quality as _Q
        _amazon_dir = args.amazon_dir or _Q.DEFAULT_AMAZON_DIR
        quality_index = _Q.ensure_quality_index(
            args.serenlens, args.model, args.data, amazon_dir=_amazon_dir,
            cache_path=args.quality_cache, rebuild=args.rebuild_quality, log=print)
        if quality_index is None:
            print("    qua 将回退到 SerenLens 自身的截至时刻均值")
    except Exception as _e:
        print("    quality 初始化失败 ({}), qua 回退到 SerenLens 自身评分".format(_e))
        quality_index = None

user_seq, labels, label_time, item_rating = load_serenlens(args.serenlens)
sim_engine = SimEngine(emb_dict) 
max_pop = max(pop_dict.values()) if pop_dict else 1.0

item_ts_first, ts_global_max, ts_global_min = {}, 0, float('inf')
for uid, seq in user_seq.items():
    for it, ts, r in seq:
        if it not in item_ts_first or ts < item_ts_first[it]:
            item_ts_first[it] = ts
        ts_global_max = max(ts_global_max, ts)
        ts_global_min = min(ts_global_min, ts)

factor_df = build_factor_table(
    user_seq, labels, label_time, item_rating,
    sim_engine, pop_dict, max_pop,
    item_ts_first=item_ts_first, ts_global_max=ts_global_max,
    ts_global_min=ts_global_min, min_history=args.min_history,
    quality_index=quality_index)

factor_df.to_pickle(factors_path)
print("    factors_SEREN_Movies.pkl: {} 条因子 (pos={}, neg={}) | min_history={}".format(
    len(factor_df), int((factor_df['label'] == 1).sum()),
    int((factor_df['label'] == 0).sum()), args.min_history))

