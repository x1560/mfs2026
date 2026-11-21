#!/usr/bin/env bash
#   bash run_all.sh
#   CASES="books" bash run_all.sh
#   PY=/path/to/python bash run_all.sh
set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-python}"
CASES="${CASES:-books movies s2018_low s2018_nolow}"
SL_BOOKS="${SL_BOOKS:-./data/SerenLens/SerenLens_Books.csv}"
SL_MOVIES="${SL_MOVIES:-./data/SerenLens/SerenLens_Movies.csv}"
SL_S2018="${SL_S2018:-SerenLens_S2018.csv}"

COMMON=(--factor_transform quantile_center --factor_center_mode causal
        --use_copula 
        --div_mode threshold --div_mref user --div_mref_value 0.40
        --pool_k 100 --pool_pos_frac 0.01)
EXP=(--run_perturbation --run_div_pert --run_ablation
     --run_normalization --run_robustness --skip_hit_rate)

mkdir -p _diag

case_args() {
  case "$1" in
    books)       echo "Seren --data Books  --serenlens $SL_BOOKS" ;;
    movies)      echo "AMZ   --data Movies --serenlens $SL_MOVIES" ;;
    s2018_low)   echo "SEREN --data S2018 --serenlens $SL_S2018 --abeta_aspects off
                        --boundary_pairs S2018_survey_pairs.csv --pool_source zeroshot" ;;
    s2018_nolow) echo "SEREN --data S2018 --serenlens $SL_S2018 --abeta_aspects off
                        --boundary_pairs S2018_survey_pairs.csv --pool_source zeroshot --drop_nonsurvey_neg" ;;
    *) echo "未知 case: $1" >&2; exit 1 ;;
  esac
}

for c in $CASES; do
  H="$(case_args "$c")"
  echo "== $c  hit =="
  "$PY" eval.py $H "${COMMON[@]}"        2>&1 | tee "_diag/_full_${c}_hit.log"
  echo "== $c  exp =="
  "$PY" eval.py $H "${COMMON[@]}" "${EXP[@]}" 2>&1 | tee "_diag/_full_${c}_exp.log"
done

echo "log： _diag/_full_<case>_{hit,exp}.log"
