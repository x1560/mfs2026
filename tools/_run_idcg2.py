# -*- coding: utf-8 -*-
"""DCG 类基线：IDCG 归一化 on/off 对照（按 2026-10-09 的时效口径重跑）。

为什么重跑
    老的 `13_三数据集IDCG对照.py` 只覆盖 books/s2018 且用的是旧口径（Books 用 SEREN、
    S2018 没有 --pool_source zeroshot）；movies 的数来自另一份日志。
    而且 α-nDCG 去掉了 γ 乘子 -> 它的 on/off 两侧都变了，旧值不可用。

口径（与论文现值、30_全量跑.py 的 CASES 一致）
    books / movies / s2018_low（带低分）

只跑这几个指标（nDCG 家族 + 不变的对照）
    M-DCG(I-DCG)  alpha-nDCG  abeta-nDCG  ndcg_rating

日志: _diag/_idcg2_<case>_{on,off}.log
用法: python _diag/_run_idcg2.py [case ...]
"""
import os
import sys
import time
import runpy
import traceback

os.chdir(os.environ.get('MDCG_ROOT') or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())
import metrics as M

COMMON = ['--factor_transform', 'quantile_center',
          '--factor_center_mode', 'causal',
          '--use_copula', '--trust_shape', 'off',
          '--div_mode', 'threshold', '--div_mref', 'user', '--div_mref_value', '0.40',
          '--pool_k', '100', '--pool_pos_frac', '0.01',
          '--run_perturbation', '--skip_hit_rate']

HEADS = {
    'books': ['SEREN', '--data', 'Books',
              '--serenlens', './data/SerenLens/SerenLens_Books.csv'],
    'movies': ['AMZ', '--data', 'Movies',
               '--serenlens', './data/SerenLens/SerenLens_Movies.csv'],
    's2018_low': ['SEREN', '--data', 'S2018', '--serenlens', 'SerenLens_S2018.csv',
                  '--abeta_aspects', 'off',
                  '--boundary_pairs', 'S2018_survey_pairs.csv',
                  '--pool_source', 'zeroshot'],
}

WANT = 'M-DCG(I-DCG),alpha-nDCG,abeta-nDCG,ndcg_rating'


def run(case, normalize):
    tag = '%s_%s' % (case, 'on' if normalize else 'off')
    log = os.path.join('_diag', '_idcg2_%s.log' % tag)
    f = open(log, 'w', encoding='utf-8', buffering=1)
    old = sys.stdout, sys.stderr
    sys.stdout = f
    sys.stderr = f
    M.ABETA_NORMALIZE = normalize
    sys.argv = (['eval.py'] + HEADS[case] + COMMON
                + ['--perturb_metrics', WANT])
    t0 = time.time()
    try:
        runpy.run_path('eval.py', run_name='__main__')
        f.write('\n>>> RUN OK %.1fs (ABETA_NORMALIZE=%s)\n' % (time.time() - t0, normalize))
    except SystemExit as e:
        f.write('\n>>> SystemExit %s\n' % e)
    except Exception:
        traceback.print_exc(file=f)
    finally:
        sys.stdout, sys.stderr = old
        f.close()
    print('DONE %-16s %.1fs  log=%s' % (tag, time.time() - t0, log), flush=True)


if __name__ == '__main__':
    cases = sys.argv[1:] or ['books', 'movies', 's2018_low']
    for c in cases:
        if c not in HEADS:
            raise SystemExit('未知 case: %s（可选 %s）' % (c, '/'.join(HEADS)))
    for c in cases:
        for nz in (True, False):
            run(c, nz)
