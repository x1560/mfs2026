# M-DCG

Code for the paper *A Multi-Factor-Aware Serendipity Evaluation Framework for Top-K Recommendation*.

## Environment

Python 3.8 or newer.

```bash
pip install -r requirements.txt
```

Tested with Python 3.8.20, pandas 2.0.3, numpy 1.24.3, scipy 1.10.1, scikit-learn 1.3.2.

## Data

Put the files in the repository root:

```
data/SerenLens/SerenLens_Books.csv      # SerenLens Books
data/SerenLens/SerenLens_Movies.csv     # SerenLens Movies
SerenLens_S2018.csv                     # SerenLens S2018 (or set SL_S2018)
S2018_survey_pairs.csv                  # S2018 survey pairs (uid,item), used by --boundary_pairs
```

and the precomputed item artifacts next to the scripts:

```
emb_{tag}_{data}.pkl  cate_{tag}_{data}.pkl  pop_{tag}_{data}.pkl
factors_{tag}_{data}.pkl  quality_{tag}_{data}.pkl
fnpair_{tag}_{data}.pkl  clustlab_{tag}_{data}.pkl  user_sim_{tag}_{data}.pkl
```

`{data}` is `Books`, `Movies` or `S2018`, `{tag}` is `Seren`, `AMZ` or `SEREN` (see below).
Download them from the release assets, or rebuild them with `bash build_all.sh`
(needs the raw Amazon ratings under `data/amazon/`).

The tag in the file names is what `--model` selects:

| dataset | `--model` |
|---------|-----------|
| Books   | `Seren`   |
| Movies  | `AMZ`     |
| S2018   | `SEREN`   |

## Run

Every setting (books, movies, s2018_low, s2018_nolow), all six experiments each:

```bash
bash run_all.sh
```

One setting only:

```bash
CASES="movies" bash run_all.sh
```

`CASES` accepts any combination, e.g. `CASES="books movies"`.

Each setting is executed as two passes: the hit-rate study first, then the perturbation,
diversity, ablation, normalization and robustness studies:

```bash
python eval.py AMZ --data Movies --serenlens ./data/SerenLens/SerenLens_Movies.csv \
  --factor_transform quantile_center --factor_center_mode causal --use_copula \
  --div_mode threshold --div_mref user --div_mref_value 0.40 \
  --pool_k 100 --pool_pos_frac 0.01

python eval.py AMZ --data Movies --serenlens ./data/SerenLens/SerenLens_Movies.csv \
  --factor_transform quantile_center --factor_center_mode causal --use_copula \
  --div_mode threshold --div_mref user --div_mref_value 0.40 \
  --pool_k 100 --pool_pos_frac 0.01 \
  --run_perturbation --run_div_pert --run_ablation --run_normalization --run_robustness \
  --skip_hit_rate
```

## Output

Logs go to `_diag/_full_<case>_hit.log` and `_diag/_full_<case>_exp.log`.
A run also writes `temporal_factors.csv` and `scarcity_params.pkl` to the working directory.

`python eval.py --help` lists every option.
