#!/usr/bin/env bash
#   bash build_all.sh
#   PY=/path/to/python bash build_all.sh
set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-python}"
SL_BOOKS="${SL_BOOKS:-./data/SerenLens/SerenLens_Books.csv}"
SL_MOVIES="${SL_MOVIES:-./data/SerenLens/SerenLens_Movies.csv}"
mkdir -p _diag

# 1) Amazon 物品嵌入
"$PY" -c "import amazon_sim; amazon_sim.build_emb('Books',  min_user_items=5, min_item_items=5)" \
    2>&1 | tee _diag/build_amzemb_books.log
"$PY" -c "import amazon_sim; amazon_sim.build_emb('Movies', min_user_items=5, min_item_items=5)" \
    2>&1 | tee _diag/build_amzemb_movies.log

# 2) Books / Movies 因子表
"$PY" build_amz_factors.py --data Books --model Seren --emb amzemb_Books_u5_i5.pkl \
      --serenlens "$SL_BOOKS"  2>&1 | tee _diag/build_books.log
"$PY" build_amz_factors.py --data Movies --model AMZ  --emb amzemb_Movies_u5_i5.pkl \
      --serenlens "$SL_MOVIES" 2>&1 | tee _diag/build_movies.log

# 3) S2018：CSV + 嵌入 + 因子表
"$PY" build_s2018_data.py --model SEREN --data S2018 \
      --low_rating_neg --low_rating_thr 2.5 2>&1 | tee _diag/build_s2018.log

# # 4) S2018 零样本因子表
# MDCG_ROOT="$PWD" "$PY" _diag/build_unlabeled_factors.py 2>&1 | tee _diag/build_unlabeled.log

echo "next: bash run_all.sh"
