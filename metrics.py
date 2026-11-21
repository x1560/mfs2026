# coding=utf-8
"""
惊喜推荐评估指标统一模块 (serendipity_metrics.py)
 
包含:
  M-DCG       : 本文指标 (时序因子 + 判别唤醒层 g1/g0 + 非对称包容 + 列表级框架)
  αβ-nDCG     : Parapar & Radlinski RecSys2021 (类别多样性+准确性)
  nDCG        : 标准 nDCG
  EPC         : Vargas & Castells 2011 (Expected Popularity Complement, 新颖度)
  ILD         : Intra-List Diversity (列表内多样性)
  NDCG_seren  : 惊喜版 nDCG (惊喜标注0/1代入折损)
  Hit@k       : 推荐命中率 (前 k 条是否命中至少一条惊喜正样本)
 
  - 因子 rel/dif/novelty/unpopularity/qua 均在 [0,1]
  - 物品级因子由compute_temporal_factors 提供
"""
import numpy as np
import itertools
from math import log2
 
from scipy.stats import beta as _beta, norm as _norm
 
 
def _fit_beta_ab(x):
    x = np.clip(np.asarray(x, dtype=float), 1e-3, 1 - 1e-3)
    if len(x) < 5:
        return 2.0, 2.0
    try:
        a, b, _, _ = _beta.fit(x, floc=0, fscale=1)
        return float(np.clip(a, 0.1, 50)), float(np.clip(b, 0.1, 50))
    except Exception:
        return 2.0, 2.0
 
 
def _fit_rho(rel, dif, a_r, b_r, a_d, b_d):
    """拟合 Gaussian Copula 的 rho """
    rel = np.clip(np.asarray(rel, dtype=float), 1e-3, 1 - 1e-3)
    dif = np.clip(np.asarray(dif, dtype=float), 1e-3, 1 - 1e-3)
    if len(rel) < 3:
        return 0.0
    u_r = _beta.cdf(rel, a_r, b_r)
    u_d = _beta.cdf(dif, a_d, b_d)
    z_r = _norm.ppf(np.clip(u_r, 1e-4, 1 - 1e-4))
    z_d = _norm.ppf(np.clip(u_d, 1e-4, 1 - 1e-4))
    rho = np.corrcoef(z_r, z_d)[0, 1]
    return float(np.clip(rho, -0.95, 0.95)) if not np.isnan(rho) else 0.0
 
 
def fit_class_params(rel, dif, use_copula=True):
    """拟合惊喜或非惊喜的 Beta(-Copula) 参数。"""
    a_r, b_r = _fit_beta_ab(rel)
    a_d, b_d = _fit_beta_ab(dif)
    rho = _fit_rho(rel, dif, a_r, b_r, a_d, b_d) if use_copula else 0.0
    return (a_r, b_r, a_d, b_d, rho)
 
 
def _class_density(rel, dif, params, use_copula=True):
    """g_k = Beta(rel)·Beta(dif)·Copula。"""
    a_r, b_r, a_d, b_d, rho = params
    rel = np.clip(np.asarray(rel, dtype=float), 1e-3, 1 - 1e-3)
    dif = np.clip(np.asarray(dif, dtype=float), 1e-3, 1 - 1e-3)
    dens = _beta.pdf(rel, a_r, b_r) * _beta.pdf(dif, a_d, b_d)
    if use_copula:
        u_r = _beta.cdf(rel, a_r, b_r)
        u_d = _beta.cdf(dif, a_d, b_d)
        z_r = _norm.ppf(np.clip(u_r, 1e-4, 1 - 1e-4))
        z_d = _norm.ppf(np.clip(u_d, 1e-4, 1 - 1e-4))
        rho2 = rho ** 2
        cop = (1.0 / np.sqrt(1 - rho2 + 1e-9)) * np.exp(
            -(rho2 * (z_r ** 2 + z_d ** 2) - 2 * rho * z_r * z_d) / (2 * (1 - rho2) + 1e-9))
        dens = dens * cop
    return dens
 
 
def arousal_score(rel, dif, pos_params, neg_params, use_copula=True):
    """g1/(g1+g0) ∈ [0,1]。"""
    g1 = _class_density(rel, dif, pos_params, use_copula)
    g0 = _class_density(rel, dif, neg_params, use_copula)
    return g1 / (g1 + g0 + 1e-12)
 
 
from scipy.stats import rankdata as _rankdata


def quantile_normalize(values):
    """Rank-based quantile normalization to [0,1]."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n <= 1:
        return np.zeros_like(arr) + 0.5
    ranks = _rankdata(arr, method='average')
    return (ranks - 1.0) / (n - 1.0)


ACCOM_FACTOR_MODE = 'both'
ACCOM_ALPHA = 1.0

ACCOM_BAND_MODE = 'kde'
ACCOM_MAX_BANDS = 2
ACCOM_BAND_GRID = 256
ACCOM_BAND_MIN_SAMPLES = 10

ACCOM_PEAK_MODE = 'adaptive_mixture'
ACCOM_MAX_PEAKS = 3
ACCOM_MIN_PEAK_SAMPLES = 8
ACCOM_MIN_PEAK_SEP = 0.06

SCARCITY_MODE = ACCOM_FACTOR_MODE
SCARCITY_GAMMA = ACCOM_ALPHA
SCARCITY_PEAK_MODE = ACCOM_PEAK_MODE
SCARCITY_MAX_PEAKS = ACCOM_MAX_PEAKS
SCARCITY_MIN_PEAK_SAMPLES = ACCOM_MIN_PEAK_SAMPLES
SCARCITY_MIN_PEAK_SEP = ACCOM_MIN_PEAK_SEP


ABETA_NORMALIZE = __import__('os').environ.get('MDCG_ABETA_NORM', '1') != '0'   # IDCG 归一化开关(MDCG_ABETA_NORM=0 关闭)


def set_accommodation_config(factor_mode='both', alpha=1.0,
                             peak_mode='adaptive_mixture', max_peaks=3,
                             min_peak_samples=8, min_peak_sep=0.06):
    """配置 Novelty / Unpopularity 的独立非对称多峰 Cauchy 响应。

    factor_mode 只用于兼容单因子/消融调用：
      both       : 同时保留 q_N 与 q_U
      nov_only   : 只使用 q_N
      unpop_only : 只使用 q_U
      none       : 不使用 N/U accommodation

      alpha^- = alpha / d_-^2,  alpha^+ = alpha / d_+^2
    其中 d_- / d_+ 是目标用户历史在中心左右两侧的稳健偏离尺度。
    最终 Cauchy 公式中没有 tau 参数。
    """
    global ACCOM_FACTOR_MODE, ACCOM_ALPHA
    global ACCOM_BAND_MODE
    global ACCOM_PEAK_MODE, ACCOM_MAX_PEAKS
    global ACCOM_MIN_PEAK_SAMPLES, ACCOM_MIN_PEAK_SEP
    global SCARCITY_MODE, SCARCITY_GAMMA
    global SCARCITY_PEAK_MODE, SCARCITY_MAX_PEAKS
    global SCARCITY_MIN_PEAK_SAMPLES, SCARCITY_MIN_PEAK_SEP

    valid = {'both', 'nov_only', 'unpop_only', 'none'}
    if factor_mode not in valid:
        raise ValueError('未知 factor_mode: {}'.format(factor_mode))
    if float(alpha) < 0:
        raise ValueError('alpha must >= 0')
    if peak_mode not in ('single', 'adaptive_mixture'):
        raise ValueError('peak_mode should be single or adaptive_mixture')
    if int(max_peaks) < 1:
        raise ValueError('max_peaks must >= 1')
    if int(min_peak_samples) < 2:
        raise ValueError('min_peak_samples must >= 2')
    if float(min_peak_sep) < 0:
        raise ValueError('min_peak_sep must >= 0')

    ACCOM_FACTOR_MODE = factor_mode
    ACCOM_ALPHA = float(alpha)

    if peak_mode == 'single':
        ACCOM_BAND_MODE = 'off'
    ACCOM_PEAK_MODE = peak_mode
    ACCOM_MAX_PEAKS = int(max_peaks)
    ACCOM_MIN_PEAK_SAMPLES = int(min_peak_samples)
    ACCOM_MIN_PEAK_SEP = float(min_peak_sep)

    SCARCITY_MODE = ACCOM_FACTOR_MODE
    SCARCITY_GAMMA = ACCOM_ALPHA
    SCARCITY_PEAK_MODE = ACCOM_PEAK_MODE
    SCARCITY_MAX_PEAKS = ACCOM_MAX_PEAKS
    SCARCITY_MIN_PEAK_SAMPLES = ACCOM_MIN_PEAK_SAMPLES
    SCARCITY_MIN_PEAK_SEP = ACCOM_MIN_PEAK_SEP




def _robust_center(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return 0.5
    return float(np.median(vals))


def _robust_deviation(vals, center, min_dev=0.05):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return 0.15
    dev = np.abs(vals - float(center))
    nz = dev[dev > 1e-12]
    d = float(np.median(nz)) if len(nz) >= 2 else float(np.std(vals))
    if not np.isfinite(d) or d <= 1e-12:
        d = 0.15
    return float(np.clip(max(d, min_dev), min_dev, 1.0))


def _fit_asym_alphas(vals, center=None, base_alpha=None, min_dev=0.05,
                      min_side_samples=2):
    """由目标用户历史估计左右曲率 alpha^- / alpha^+。
    若一侧样本不足，则使用整体稳健偏离作为回退。
    """
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if center is None:
        center = _robust_center(vals)
    center = float(center)
    base_alpha = ACCOM_ALPHA if base_alpha is None else max(float(base_alpha), 0.0)

    overall = _robust_deviation(vals, center, min_dev=min_dev)
    left = center - vals[vals < center - 1e-12]
    right = vals[vals > center + 1e-12] - center

    if len(left) >= int(min_side_samples):
        d_minus = float(np.median(left))
    else:
        d_minus = overall
    if len(right) >= int(min_side_samples):
        d_plus = float(np.median(right))
    else:
        d_plus = overall

    d_minus = float(np.clip(max(d_minus, min_dev), min_dev, 1.0))
    d_plus = float(np.clip(max(d_plus, min_dev), min_dev, 1.0))

    alpha_minus = base_alpha / (d_minus * d_minus) if base_alpha > 0 else 0.0
    alpha_plus = base_alpha / (d_plus * d_plus) if base_alpha > 0 else 0.0
    return center, float(alpha_minus), float(alpha_plus)


def _cauchy_alpha_asym(x, lam, alpha_minus, alpha_plus):
    """左右非对称 Cauchy：1 / [1 + alpha_side (x-lambda)^2]."""
    x = float(x)
    lam = float(lam)
    alpha_side = float(alpha_minus) if x < lam else float(alpha_plus)
    alpha_side = max(alpha_side, 0.0)
    d = x - lam
    return float(1.0 / (1.0 + alpha_side * d * d))


def _fit_adaptive_centers(vals, max_peaks=None, min_peak_samples=None,
                          min_peak_sep=None, random_state=42):
    """用 GaussianMixture + BIC 自动选择峰数、中心和权重。"""
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    vals = np.clip(vals, 0.0, 1.0)

    max_peaks = ACCOM_MAX_PEAKS if max_peaks is None else int(max_peaks)
    min_peak_samples = (ACCOM_MIN_PEAK_SAMPLES if min_peak_samples is None
                        else int(min_peak_samples))
    min_peak_sep = ACCOM_MIN_PEAK_SEP if min_peak_sep is None else float(min_peak_sep)

    if len(vals) == 0:
        return {'k': 1, 'centers': [0.5], 'weights': [1.0], 'bic': float('nan')}
    if len(vals) < max(5, 2 * min_peak_samples) or len(np.unique(vals)) < 3:
        return {'k': 1, 'centers': [float(np.median(vals))],
                'weights': [1.0], 'bic': float('nan')}

    try:
        from sklearn.mixture import GaussianMixture
    except Exception:
        return {'k': 1, 'centers': [float(np.median(vals))],
                'weights': [1.0], 'bic': float('nan')}

    X = vals.reshape(-1, 1)
    k_cap = min(max_peaks, len(np.unique(vals)), len(vals) // min_peak_samples)
    k_cap = max(1, int(k_cap))
    best = None
    for k in range(1, k_cap + 1):
        try:
            gm = GaussianMixture(n_components=k, covariance_type='full',
                                 reg_covar=1e-4, n_init=5,
                                 random_state=random_state)
            gm.fit(X)
            means = gm.means_.ravel()
            weights = gm.weights_.ravel()
            if np.min(weights * len(vals)) < min_peak_samples:
                continue
            order = np.argsort(means)
            means = means[order]
            weights = weights[order]
            if len(means) > 1 and np.min(np.diff(means)) < min_peak_sep:
                continue
            bic = float(gm.bic(X))
            cand = (bic, means, weights)
            if best is None or bic < best[0]:
                best = cand
        except Exception:
            continue

    if best is None:
        return {'k': 1, 'centers': [float(np.median(vals))],
                'weights': [1.0], 'bic': float('nan')}

    bic, means, weights = best
    weights = weights / (weights.sum() + 1e-12)
    return {'k': int(len(means)),
            'centers': [float(x) for x in means],
            'weights': [float(x) for x in weights],
            'bic': float(bic)}


def _bandwidth_from_alphas(a_minus, a_plus, default=0.15):
    """由计算曲率反解接受半宽, 取几何平均作为 KDE 带宽。
    """
    base = float(ACCOM_ALPHA) if float(ACCOM_ALPHA) > 0 else 1.0
    hs = []
    for a in (a_minus, a_plus):
        try:
            a = float(a)
        except (TypeError, ValueError):
            a = 0.0
        hs.append(np.sqrt(base / a) if np.isfinite(a) and a > 0 else float(default))
    return float(np.sqrt(hs[0] * hs[1]))


def _kde_bands(vals, bandwidth=None, max_bands=None, grid_size=None):
    """高斯核密度估计 -> 密度局部极大 -> 接受带中心。
    返回 {'k','centers','weights'}。
    """
    max_bands = ACCOM_MAX_BANDS if max_bands is None else int(max_bands)
    grid_size = ACCOM_BAND_GRID if grid_size is None else int(grid_size)

    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    vals = np.clip(vals, 0.0, 1.0)
    if len(vals) == 0:
        return {'k': 1, 'centers': [0.5], 'weights': [1.0]}
    if len(vals) < ACCOM_BAND_MIN_SAMPLES or len(np.unique(vals)) < 3:
        return {'k': 1, 'centers': [float(np.median(vals))], 'weights': [1.0]}

    h = 0.15 if bandwidth is None else float(bandwidth)
    if not np.isfinite(h) or h <= 0:
        h = 0.15
    lo = max(0.0, float(np.min(vals)) - h)
    hi = min(1.0, float(np.max(vals)) + h)
    if hi - lo < 1e-9:
        return {'k': 1, 'centers': [float(np.median(vals))], 'weights': [1.0]}

    grid = np.linspace(lo, hi, grid_size)
    z = (grid[:, None] - vals[None, :]) / h
    dens = np.exp(-0.5 * z * z).sum(axis=1)

    idx = np.where((dens[1:-1] > dens[:-2]) & (dens[1:-1] > dens[2:]))[0] + 1
    if len(idx) == 0:
        centers = [float(grid[int(np.argmax(dens))])]
    else:
        keep = idx[np.argsort(dens[idx])[::-1][:max_bands]]
        centers = [float(c) for c in np.sort(grid[keep])]
    k = len(centers)
    return {'k': k, 'centers': centers, 'weights': [1.0 / k] * k}


def _estimate_bands(vals, bandwidth=None, mode=None, **kw):
    mode = ACCOM_BAND_MODE if mode is None else mode
    if mode == 'adaptive_mixture':
        return _fit_adaptive_centers(vals, **kw)
    return _kde_bands(vals, bandwidth=bandwidth)


def _mixture_cauchy_norm_alpha(centers, weights, alpha_minus, alpha_plus):
    return 1.0


def _cauchy_multipeak_alpha_asym(x, centers, alpha_minus, alpha_plus):
    """
    q(x) = max_k 1 / [1 + alpha_side * (x - lambda_k)^2]
    """
    centers = np.asarray(centers, dtype=float)
    centers = centers[np.isfinite(centers)]
    if len(centers) == 0:
        centers = np.asarray([0.5], dtype=float)

    responses = [
        _cauchy_alpha_asym(x, c, alpha_minus, alpha_plus)
        for c in centers
    ]
    return float(np.clip(max(responses), 0.0, 1.0))




def build_user_accommodation_params(user_seq, item_rare_map, item_nov_map=None,
                                    item_unpop_map=None, min_history=5,
                                    factor_mode=None, min_dev=0.05,
                                    scarcity_mode=None, scarcity_weight=None,
                                    min_tau=None):
    """
    从目标用户历史构建 N/U 的左右曲率参数
    核心参数：
      lambda_n, alpha_n_minus, alpha_n_plus
      lambda_u, alpha_u_minus, alpha_u_plus
    多峰中心/权重可先由历史估计，若后续使用 surprise pool，则仅重估
    K / centers / weights；左右 alpha 始终来自 target user 自身历史。
    """
    if min_tau is not None:
        min_dev = float(min_tau)
    item_nov_map = item_nov_map or {}
    item_unpop_map = item_unpop_map or {}
    item_rare_map = item_rare_map or {}

    histories, explore = {}, {}
    for uid, seq in user_seq.items():
        n = np.array([item_nov_map.get(it, 0.5) for it, ts, r in seq], dtype=float)
        u = np.array([item_unpop_map.get(it, 0.5) for it, ts, r in seq], dtype=float)
        s = np.array([item_rare_map.get(it, 0.5) for it, ts, r in seq], dtype=float)
        n, u, s = n[np.isfinite(n)], u[np.isfinite(u)], s[np.isfinite(s)]
        histories[uid] = (n, u, s)
        explore[uid] = float(np.mean(s)) if len(s) >= min_history else 0.5

    uids = list(explore.keys())
    vals = np.array([explore[u] for u in uids], dtype=float)
    if len(vals) <= 1:
        ranks = np.zeros_like(vals) + 0.5
    else:
        ranks = (_rankdata(vals, method='average') - 1.0) / (len(vals) - 1.0)
        ranks = np.clip(ranks, 0.0, 1.0)
    rank_map = {u: float(r) for u, r in zip(uids, ranks)}

    out = {}
    default_dev = 0.15
    default_alpha = ACCOM_ALPHA / (default_dev * default_dev) if ACCOM_ALPHA > 0 else 0.0
    for uid in uids:
        n, u, s = histories[uid]
        if min(len(n), len(u)) < min_history:
            ln = lu = 0.5
            an_m = an_p = au_m = au_p = default_alpha
        else:
            ln = _robust_center(n)
            lu = _robust_center(u)
            _, an_m, an_p = _fit_asym_alphas(
                n, center=ln, base_alpha=ACCOM_ALPHA, min_dev=min_dev)
            _, au_m, au_p = _fit_asym_alphas(
                u, center=lu, base_alpha=ACCOM_ALPHA, min_dev=min_dev)

        p = {
            'lambda_n': ln,
            'alpha_n_minus': an_m, 'alpha_n_plus': an_p,
            'lambda_u': lu,
            'alpha_u_minus': au_m, 'alpha_u_plus': au_p,
            'explore_rank': rank_map.get(uid, 0.5)
        }

        use_bands = (ACCOM_BAND_MODE != 'off'
                     and min(len(n), len(u)) >= min_history)
        if use_bands:
            mn = _estimate_bands(n, bandwidth=_bandwidth_from_alphas(an_m, an_p))
            mu = _estimate_bands(u, bandwidth=_bandwidth_from_alphas(au_m, au_p))
            p['n_centers'] = mn['centers']; p['n_weights'] = mn['weights']; p['k_n'] = mn['k']
            p['u_centers'] = mu['centers']; p['u_weights'] = mu['weights']; p['k_u'] = mu['k']
        else:
            p['n_centers'] = [ln]; p['n_weights'] = [1.0]; p['k_n'] = 1
            p['u_centers'] = [lu]; p['u_weights'] = [1.0]; p['k_u'] = 1

        p['n_mix_norm'] = _mixture_cauchy_norm_alpha(
            p['n_centers'], p['n_weights'], an_m, an_p)
        p['u_mix_norm'] = _mixture_cauchy_norm_alpha(
            p['u_centers'], p['u_weights'], au_m, au_p)
        out[uid] = p

    if out:
        def ms(key):
            a = np.array([v[key] for v in out.values()], dtype=float)
            return float(a.mean()), float(a.std())
        ln, lns = ms('lambda_n')
        anm, anms = ms('alpha_n_minus'); anp, anps = ms('alpha_n_plus')
        lu, lus = ms('lambda_u')
        aum, aums = ms('alpha_u_minus'); aup, aups = ms('alpha_u_plus')
        rr, rrs = ms('explore_rank')
        pass
        if ACCOM_BAND_MODE != 'off':
            kn = np.array([v.get('k_n', 1) for v in out.values()], dtype=float)
            ku = np.array([v.get('k_u', 1) for v in out.values()], dtype=float)
            if ACCOM_BAND_MODE == 'adaptive_mixture':
                kdesc = 'Kmax={} minComp={} minSep={:.3f}'.format(
                    ACCOM_MAX_PEAKS, ACCOM_MIN_PEAK_SAMPLES, ACCOM_MIN_PEAK_SEP)
            else:
                kdesc = 'Kmax={} (KDE, 带宽'.format(ACCOM_MAX_BANDS)
            pass
    return out


build_user_scarcity_params = build_user_accommodation_params
build_user_cauchy_params = build_user_accommodation_params


def personalize_accommodation_centers(base_params, surprise_sim, positive_nu_by_uid,
                                       center='median', min_pool=5,
                                       adaptive_peaks=None, max_peaks=None,
                                       min_peak_samples=None, min_peak_sep=None):
    """用个性化惊喜池只重估 N/U 的带数、带中心和权重。
    max_peaks / min_peak_samples / min_peak_sep 仅对旧口径
    (ACCOM_BAND_MODE == 'adaptive_mixture') 生效, 新口径下被忽略。
    """
    stat = np.median if center == 'median' else np.mean
    use_bands = (ACCOM_BAND_MODE != 'off'
                 if adaptive_peaks is None else bool(adaptive_peaks))

    out = {uid: dict(v) for uid, v in (base_params or {}).items()}
    used = 0
    sizes = []
    k_ns, k_us = [], []

    for uid, p in out.items():
        ns, us = [], []
        for v, sim in (surprise_sim or {}).get(uid, []):
            for n, u in positive_nu_by_uid.get(v, []):
                if np.isfinite(n) and np.isfinite(u):
                    ns.append(float(n)); us.append(float(u))

        p['surprise_pool_n'] = len(ns)
        if len(ns) < min_pool or len(us) < min_pool:
            continue

        p['lambda_n_hist'] = p.get('lambda_n', 0.5)
        p['lambda_u_hist'] = p.get('lambda_u', 0.5)

        if use_bands:
            mn = _estimate_bands(ns,
                                 bandwidth=_bandwidth_from_alphas(
                                     p.get('alpha_n_minus'), p.get('alpha_n_plus')),
                                 max_peaks=max_peaks,
                                 min_peak_samples=min_peak_samples,
                                 min_peak_sep=min_peak_sep)
            mu = _estimate_bands(us,
                                 bandwidth=_bandwidth_from_alphas(
                                     p.get('alpha_u_minus'), p.get('alpha_u_plus')),
                                 max_peaks=max_peaks,
                                 min_peak_samples=min_peak_samples,
                                 min_peak_sep=min_peak_sep)
            p['n_centers'] = mn['centers']; p['n_weights'] = mn['weights']; p['k_n'] = mn['k']
            p['u_centers'] = mu['centers']; p['u_weights'] = mu['weights']; p['k_u'] = mu['k']
            p['lambda_n'] = float(np.sum(np.asarray(mn['centers']) * np.asarray(mn['weights'])))
            p['lambda_u'] = float(np.sum(np.asarray(mu['centers']) * np.asarray(mu['weights'])))
            k_ns.append(mn['k']); k_us.append(mu['k'])
        else:
            p['lambda_n'] = float(stat(ns))
            p['lambda_u'] = float(stat(us))
            p['n_centers'] = [p['lambda_n']]; p['n_weights'] = [1.0]; p['k_n'] = 1
            p['u_centers'] = [p['lambda_u']]; p['u_weights'] = [1.0]; p['k_u'] = 1
            k_ns.append(1); k_us.append(1)

        p['n_mix_norm'] = _mixture_cauchy_norm_alpha(
            p['n_centers'], p['n_weights'],
            p.get('alpha_n_minus', 1.0), p.get('alpha_n_plus', 1.0))
        p['u_mix_norm'] = _mixture_cauchy_norm_alpha(
            p['u_centers'], p['u_weights'],
            p.get('alpha_u_minus', 1.0), p.get('alpha_u_plus', 1.0))
        used += 1
        sizes.append(len(ns))

    if out:
        ln = np.array([v.get('lambda_n', .5) for v in out.values()], dtype=float)
        lu = np.array([v.get('lambda_u', .5) for v in out.values()], dtype=float)
        pass
        if use_bands and k_ns:
            kn = np.asarray(k_ns, dtype=float); ku = np.asarray(k_us, dtype=float)
            if ACCOM_BAND_MODE == 'adaptive_mixture':
                kdesc = 'Kmax={} minComp={} minSep={:.3f}'.format(
                    ACCOM_MAX_PEAKS if max_peaks is None else int(max_peaks),
                    ACCOM_MIN_PEAK_SAMPLES if min_peak_samples is None else int(min_peak_samples),
                    ACCOM_MIN_PEAK_SEP if min_peak_sep is None else float(min_peak_sep))
            else:
                kdesc = 'Kmax={} (KDE, 带宽)'.format(ACCOM_MAX_BANDS)
            pass
    return out


personalize_scarcity_centers = personalize_accommodation_centers


def accommodation_components(nov, unpop, user_param=None):
    """返回 (q_N, q_U)。"""
    p = user_param or {}
    n_centers = p.get('n_centers', [p.get('lambda_n', 0.5)])
    n_weights = p.get('n_weights', [1.0])
    u_centers = p.get('u_centers', [p.get('lambda_u', 0.5)])
    u_weights = p.get('u_weights', [1.0])

    default_alpha = ACCOM_ALPHA / (0.15 * 0.15) if ACCOM_ALPHA > 0 else 0.0
    an_m = p.get('alpha_n_minus', default_alpha)
    an_p = p.get('alpha_n_plus', default_alpha)
    au_m = p.get('alpha_u_minus', default_alpha)
    au_p = p.get('alpha_u_plus', default_alpha)

    qn = _cauchy_multipeak_alpha_asym(
        nov, n_centers, an_m, an_p)
    qu = _cauchy_multipeak_alpha_asym(
        unpop, u_centers, au_m, au_p)
    return qn, qu








def f_sp_score(rel, dif, nov, unpop, qua, pos_params, neg_params,
               use_copula=True, user_scarcity_param=None, scarcity_mode=None):
    """Item-level score with independent N/U accommodation responses.
        f_sp = [1 + Qua*q_N + Qua*q_U] * f_user
    """
    arousal = arousal_score(rel, dif, pos_params, neg_params, use_copula)
    qn, qu = accommodation_components(nov, unpop, user_scarcity_param)

    legacy = {'separate_avg': 'both', 'separate_product': 'both',
              'scalar_avg': 'both', None: ACCOM_FACTOR_MODE,
              'both': 'both', 'nov_only': 'nov_only',
              'unpop_only': 'unpop_only', 'none': 'none'}
    mode = legacy.get(scarcity_mode, scarcity_mode)
    qua = float(qua)
    if mode == 'nov_only':
        gain = 1.0 + qua * qn
    elif mode == 'unpop_only':
        gain = 1.0 + qua * qu
    elif mode == 'none':
        gain = 1.0 + qua
    else:
        gain = 1.0 + qua * qn + qua * qu
    return float(np.clip(gain * arousal, 0, 10))


DIV_MODE = 'legacy'
DIV_MREF_DEFAULT = 0.40      # 无该用户参照时的回退值
DIV_MREF = {}                # uid -> 该用户惊喜物品之间的平均两两相似度
DIV_MREF_SOURCE = 'user'     # 'user' | 'global' | 'pool'
DIV_SIM = None
DIV_CAT_BETA = 0.5


def set_div_config(mode=None, mref=None, mref_default=None, source=None, sim=None,
                   cat_beta=None):
    """设置 div 口径。
    mode   : 'legacy' | 'threshold'
    mref   : {uid: float} 或 None
    source : 'user' | 'global' | 'pool'
    sim    : callable(a, b)；None = 用 sim_func
    """
    global DIV_MODE, DIV_MREF, DIV_MREF_DEFAULT, DIV_MREF_SOURCE, DIV_SIM, DIV_CAT_BETA
    if mode is not None:
        if mode not in ('legacy', 'threshold', 'anchor', 'cat', 'paper'):
            raise ValueError(
                "DIV_MODE 必须是 legacy/threshold/anchor/cat/paper, 收到 {}".format(mode))
        DIV_MODE = mode
    if mref is not None:
        DIV_MREF = dict(mref)
    if mref_default is not None:
        DIV_MREF_DEFAULT = float(mref_default)
    if source is not None:
        if source not in ('user', 'global', 'pool', 'prefix'):
            raise ValueError("DIV_MREF_SOURCE 必须是 user/global/pool/prefix, 收到 {}".format(source))
        DIV_MREF_SOURCE = source
    if sim is not None:
        DIV_SIM = sim
    if cat_beta is not None:
        DIV_CAT_BETA = float(cat_beta)


def anchor_div(list_items, list_fsps, sim_fn, c_ref):
    """锚点一致性折扣（--div_mode anchor）: anchor = 列表中 f_sp 最高的物品。

    与 anchor 的相似度 s:
        s >= c_ref  -> div = 1   (属于 anchor 的口味簇, 不罚)  ← "和它很像就不罚"
        s <  c_ref  -> div = s / c_ref 线性下降                ← "和它很不像就罚"
        s -> 0      -> div -> 0
    """
    n = len(list_items)
    out = [1.0] * n
    if n <= 1 or sim_fn is None:
        return out
    a = int(np.argmax(list_fsps))
    _others = [float(sim_fn(list_items[p], list_items[a]))
               for p in range(n) if p != a]
    if _others:
        _m = float(np.mean(_others))
        if _m > 1e-9:
            c_ref = _m
    if c_ref is None or c_ref <= 0:
        return out
    for p in range(n):
        if p == a:
            continue
        s = float(sim_fn(list_items[p], list_items[a]))
        out[p] = float(np.clip(s / float(c_ref), 0.0, 1.0))
    return out


def div_sim_func(fallback):
    """div 实际使用的相似度函数."""
    return DIV_SIM if DIV_SIM is not None else fallback


def get_div_mref(uid):
    """取某用户的口味参照相似度."""
    if DIV_MODE != 'threshold':
        return None
    return DIV_MREF.get(uid, DIV_MREF_DEFAULT)


def build_div_mref(factor_df, sim_engine, source='user', min_pos=2, fallback=None):
    """从标注正样本估计每个用户的「口味参照相似度」m_ref。

    source='user'  : 该用户惊讶物品之间的平均两两相似度 (>= min_pos 个才够算)
    source='global': 所有用户统一用同一个值 (上面那个量的全用户均值)
    返回 {uid: float}。
    """
    fb = DIV_MREF_DEFAULT if fallback is None else float(fallback)
    out, stats = {}, []
    pos_df = factor_df[factor_df['label'] == 1]
    for uid, g in pos_df.groupby('uid'):
        its = [it for it in g['item'] if it in sim_engine.emb]
        if len(its) < min_pos:
            continue
        s = [sim_engine.sim(a, b)
             for i, a in enumerate(its) for b in its[i + 1:]]
        if s:
            out[uid] = float(np.mean(s))
            stats.append(out[uid])
    ds_mean = float(np.mean(stats)) if stats else fb
    fill = ds_mean if source == 'global' else fb
    if source == 'global':
        out = {}
    for uid in factor_df['uid'].unique():
        out.setdefault(uid, fill)
    pass
    return out


def div_discount(sims, m_ref=None):
    """列表级多样性折扣，值域 [0, 1]。
      legacy    : 两侧都做折扣, 保证 d ∈ [0, 1]
                    mean_sim >= 0 : d = 1 - m
                    mean_sim <  0 : d = 1 + m
      threshold : 以该用户的口味参照 m_ref 为零点, 只有超出参照的部分才算冗余
                    d = 1 - clip((m - m_ref) / (1 - m_ref), 0, 1)
                  m <= m_ref -> d = 1 ; m -> 1 -> d -> 0
    """
    if not sims:
        return 1.0
    if DIV_MODE == 'paper':
        return float(np.mean([1.0 - (s if s > 0.0 else 0.0) for s in sims]))
    m = float(np.mean(sims))
    if DIV_MODE == 'threshold':
        r = DIV_MREF_DEFAULT if m_ref is None else float(m_ref)
        r = float(np.clip(r, 0.0, 0.999))
        d = 1.0 - max(0.0, (m - r) / (1.0 - r))
        return float(np.clip(d, 0.0, 1.0))
    d = 1.0 - m if m >= 0.0 else 1.0 + m
    return float(np.clip(d, 0.0, 1.0))


def trust_threshold(E_s, theta, i, k, shape='increasing', shape_p=None):
    """位置 i(1-indexed) 上的信任容忍阈值 E_i。

    阈值越高, 高 dif 物品越不容易被压制(match 越大)。形状决定"哪个位置的
    门槛更低", 即哪个位置更容易接纳高 dif(更意外)的物品。
      increasing : E_i 随位置递增 —— 前严后松。
      decreasing : E_i 随位置递减 —— 前松后严。
      ushape     : 两端高、中间低 —— 中段最容易接纳意外物品。
                   shape_p = (lo, span, power, center)
                   E = lo + span * (|i - center| / center_scale) ** power
                   center: 最松位置的绝对索引(1-indexed), 默认 5.0
                   center_scale: 归一化尺度, 默认取 center(使 i=1 时
                                 |i-center|/center_scale = 1)
                   默认 (0.30, 0.55, 1.5, 5.0)
      constant   : 所有位置同一阈值 E_s。
    """
    if shape == 'increasing':
        return E_s + (1.0 - E_s) * (1.0 - 1.0 / (1.0 + theta * (i - 1)))
    if shape == 'decreasing':
        t = (i - 1) / float(max(k - 1, 1))
        return (1.0 - t) * 0.85 + t * 0.30
    if shape == 'ushape':
        if shape_p is None:
            lo, span, power, center = 0.30, 0.55, 1.5, 5.0
        elif len(shape_p) >= 4:
            lo, span, power, center = shape_p[:4]
        else:
            lo, span, power = shape_p[:3]
            center = 5.0
        scale = max(float(center), 1e-9)
        return min(1.0, lo + span * (abs(i - center) / scale) ** power)
    if shape == 'constant':
        return E_s
    raise ValueError('未知trust shape: {}'.format(shape))


def _unpack_trust(trust_params, k):
    """
    trust_params -> (E_s, theta, alpha, shape, shape_p)。
    None                                 -> 默认
    (E_s, theta)                         -> 旧写法
    (E_s, theta, alpha)                  -> 加力度
    (E_s, theta, alpha, shape[, shape_p])-> 加形状
    """
    if trust_params is None:
        return 0.48, 0.5, 1.0, 'increasing', None
    tp = tuple(trust_params)
    E_s = float(tp[0]) if len(tp) > 0 else 0.48
    th = float(tp[1]) if len(tp) > 1 else 0.5
    alpha = float(tp[2]) if len(tp) > 2 else 1.0
    shape = tp[3] if len(tp) > 3 else 'increasing'
    shape_p = tp[4] if len(tp) > 4 else None
    return E_s, th, alpha, shape, shape_p


def mdcg_score(ranked_factors, pos_params, neg_params, sim_func=None,
               use_copula=True, k=10, user_scarcity_params=None,
               trust_params=None, scarcity_mode=None, item_cates=None):
    """列表级 M-DCG@k。

    trust_params 形如 (E_s, theta[, alpha[, shape[, shape_p]]]):
      E_s, theta  : 容忍阈值曲线的参数(仅 increasing 形状用)
      trust_alpha : 超量惩罚的力度系数, 默认 1.0。
                    match = max(0, 1 - trust_alpha * max(0, dif - E_i))
      shape       : 阈值曲线形状, 见 trust_threshold(); 默认 'increasing'
      shape_p     : 形状参数, 仅 ushape 用
    """
    E_s, th, t_alpha, t_shape, t_shape_p = _unpack_trust(trust_params, k)
    ranked = ranked_factors[:k]
    f_sps, items, difs, mrefs = [], [], [], []
    for tup in ranked:
        item_id, rel, dif, nov, unpop, qua = tup[:6]
        uid = tup[6] if len(tup) >= 7 else None
        p = user_scarcity_params.get(uid) if user_scarcity_params is not None and uid is not None else None
        f_sp = f_sp_score(rel, dif, nov, unpop, qua, pos_params, neg_params,
                          use_copula, p, scarcity_mode)
        f_sps.append(f_sp); items.append(item_id); difs.append(dif)
        mrefs.append(get_div_mref(uid))

    _dsim = div_sim_func(sim_func)
    if DIV_MODE == 'threshold' and DIV_MREF_SOURCE == 'pool' and _dsim is not None \
            and len(items) > 1:
        _pv = [_dsim(items[a], items[b])
               for a in range(len(items)) for b in range(a + 1, len(items))]
        if _pv:
            mrefs = [float(np.mean(_pv))] * len(mrefs)
    elif DIV_MODE == 'threshold' and DIV_MREF_SOURCE == 'prefix' and _dsim is not None \
            and len(items) > 1:
        # 前缀不足 2 个物品时无参照, 回退该用户自身的参照点。
        _s, _c, _pref = 0.0, 0, []
        for a in range(len(items)):
            _pref.append(mrefs[a] if _c == 0 else _s / _c)
            for b in range(a):
                _s += _dsim(items[a], items[b]); _c += 1
        mrefs = _pref

    _cat_divs = None
    if DIV_MODE == 'cat' and item_cates is not None \
            and sim_func is not None and len(items) > 1:
        _cat_divs = [1.0] * len(items)
        for _p in range(1, len(items)):
            _c = item_cates.get(items[_p])
            if _c is None:
                continue
            _ns = sum(1 for _j in range(_p) if item_cates.get(items[_j]) == _c)
            _cat_divs[_p] = float((1.0 - DIV_CAT_BETA) ** _ns)
    _anchor_divs = None
    if DIV_MODE == 'anchor' and sim_func is not None and len(items) > 1:
        _a = int(np.argmax(f_sps))
        _anchor_divs = anchor_div(items, f_sps, _dsim, mrefs[_a] or DIV_MREF_DEFAULT)
    dcg = 0.0
    for pos, (item_id, f_sp) in enumerate(zip(items, f_sps)):
        if _cat_divs is not None:
            div_disc = _cat_divs[pos]
        elif _anchor_divs is not None:
            div_disc = _anchor_divs[pos]
        elif sim_func is not None and pos > 0:
            div_disc = div_discount([_dsim(item_id, items[j]) for j in range(pos)],
                                    mrefs[pos])
        else:
            div_disc = 1.0
        E = trust_threshold(E_s, th, pos + 1, k, t_shape, t_shape_p)
        match = max(0.0, 1.0 - t_alpha * max(0.0, difs[pos] - E))
        dcg += f_sp * div_disc * match / log2(2 + pos)
    return dcg


def compute_idcg(ranked_factors, pos_params, neg_params, sim_func=None,
                 use_copula=True, k=10, user_scarcity_params=None,
                 trust_params=None, scarcity_mode=None, item_cates=None):
    """对给定 top-k 候选集合用 DP 求 M-DCG 的理想排列。"""
    E_s, th, t_alpha, t_shape, t_shape_p = _unpack_trust(trust_params, k)
    ranked = ranked_factors[:k]
    f_sps, items, difs, mrefs = [], [], [], []
    for tup in ranked:
        item_id, rel, dif, nov, unpop, qua = tup[:6]
        uid = tup[6] if len(tup) >= 7 else None
        p = user_scarcity_params.get(uid) if user_scarcity_params is not None and uid is not None else None
        f_sp = f_sp_score(rel, dif, nov, unpop, qua, pos_params, neg_params,
                          use_copula, p, scarcity_mode)
        f_sps.append(f_sp); items.append(item_id); difs.append(dif)
        mrefs.append(get_div_mref(uid))          # div 参照点 (legacy 下为 None)

    _dsim = div_sim_func(sim_func)
    if DIV_MODE == 'threshold' and DIV_MREF_SOURCE == 'pool' and _dsim is not None \
            and len(items) > 1:
        _pv = [_dsim(items[a], items[b])
               for a in range(len(items)) for b in range(a + 1, len(items))]
        if _pv:
            mrefs = [float(np.mean(_pv))] * len(mrefs)

    _mref_pref = None
    if DIV_MODE == 'threshold' and DIV_MREF_SOURCE == 'prefix' and _dsim is not None \
            and len(items) > 1:
        _n = len(items)
        _mref_pref = [None] * (1 << _n)
        for _m in range(1 << _n):
            _idx = [_j for _j in range(_n) if _m & (1 << _j)]
            if len(_idx) < 2:
                continue
            _vv = [_dsim(items[_idx[_x]], items[_idx[_y]])
                   for _x in range(len(_idx)) for _y in range(_x + 1, len(_idx))]
            if _vv:
                _mref_pref[_m] = float(np.mean(_vv))

    _cat_divs = None
    if DIV_MODE == 'cat' and item_cates is not None and sim_func is not None \
            and len(items) > 1:
        _cat_divs = [1.0] * len(items)
        for _p in range(1, len(items)):
            _c = item_cates.get(items[_p])
            if _c is None:
                continue
            _ns = sum(1 for _j in range(_p) if item_cates.get(items[_j]) == _c)
            _cat_divs[_p] = float((1.0 - DIV_CAT_BETA) ** _ns)
    _anchor_divs = None
    if DIV_MODE == 'anchor' and sim_func is not None and len(items) > 1:
        _a = int(np.argmax(f_sps))
        _anchor_divs = anchor_div(items, f_sps, _dsim, mrefs[_a] or DIV_MREF_DEFAULT)
    n = len(ranked)
    dp = [-1.0] * (1 << n)
    dp[0] = 0.0
    for mask in range(1 << n):
        if dp[mask] < 0:
            continue
        ppos = bin(mask).count('1')
        if ppos >= n:
            continue
        E = trust_threshold(E_s, th, ppos + 1, k, t_shape, t_shape_p)
        disc = 1.0 / np.log2(2 + ppos)
        for i in range(n):
            if mask & (1 << i):
                continue
            if _cat_divs is not None:
                div = _cat_divs[i]
            elif _anchor_divs is not None:
                div = _anchor_divs[i]
            elif sim_func is not None and mask != 0:
                sims = [_dsim(items[i], items[j]) for j in range(n) if mask & (1 << j)]
                _rf = mrefs[i]
                if _mref_pref is not None and _mref_pref[mask] is not None:
                    _rf = _mref_pref[mask]
                div = div_discount(sims, _rf)
            else:
                div = 1.0
            match = max(0.0, 1.0 - t_alpha * max(0.0, difs[i] - E))
            gain = f_sps[i] * div * match * disc
            nm = mask | (1 << i)
            dp[nm] = max(dp[nm], dp[mask] + gain)
    return dp[(1 << n) - 1]

def spade_frontier(pop, sim):
    """SPADE 的用户专属帕累托前沿。
    p 在前沿上 ⟺ **不存在** k 使 pop(k) > pop(p) **且** sim(k) > sim(p)
    按 pop 降序分组, 组内比较用的是"严格更大 pop"的那批点的最大 sim。
    """
    n = len(pop)
    on = [False] * n
    order = sorted(range(n), key=lambda i: (-pop[i], -sim[i], i))
    best_strict = -1.0
    a = 0
    while a < n:
        b = a
        while b < n and pop[order[b]] == pop[order[a]]:
            b += 1
        grp = order[a:b]
        for i in grp:
            on[i] = sim[i] >= best_strict
        for i in grp:
            if sim[i] > best_strict:
                best_strict = sim[i]
        a = b
    return on


def _spade_minmax(v):
    lo, hi = min(v), max(v)
    if hi - lo < 1e-12:
        return [0.0] * len(v)
    return [(x - lo) / (hi - lo) for x in v]


def spade_distances(pop, sim, on=None, scale_both=True):
    """
    每个物品到前沿的**最小欧氏距离**。
    距离越大 = 越"跳出"热门×像历史的边界 = SPADE 认为越有惊喜潜力。
    """
    if on is None:
        on = spade_frontier(pop, sim)
    if not scale_both:
        p, s = list(pop), list(sim)
    else:
        p = _spade_minmax(pop)
        s = _spade_minmax(sim)
    p = _spade_minmax(pop)
    s = _spade_minmax(sim)
    fp = [(p[i], s[i]) for i in range(len(pop)) if on[i]]
    if not fp:
        return [0.0] * len(pop)
    out = []
    for i in range(len(pop)):
        out.append(min(((p[i] - x) ** 2 + (s[i] - y) ** 2) ** 0.5 for (x, y) in fp))
    return out



def exact_idcg(k, gain_of_prefix):
    """
    子集 DP: 在**固定的 k 个物品**上求使 DCG 最大的排列, 即精确 IDCG。
    gain_of_prefix(mask, i) : 把物品 i 追加到"已选集合 mask"之后时的未折扣增益
    返回 (最优 DCG, 最优排列[0-based 序号列表])。
    """
    full = 1 << k
    neg = -1.0
    dp = [neg] * full
    par = [-1] * full
    dp[0] = 0.0
    for mask in range(full):
        if dp[mask] < 0.0:
            continue
        p = bin(mask).count('1')
        if p == k:
            continue
        disc = log2(2 + p)
        for i in range(k):
            if (mask >> i) & 1:
                continue
            v = dp[mask] + gain_of_prefix(mask, i) / disc
            nm = mask | (1 << i)
            if v > dp[nm]:
                dp[nm] = v
                par[nm] = i
    order = []
    mask = full - 1
    while mask:
        i = par[mask]
        order.append(i)
        mask ^= (1 << i)
    order.reverse()
    return dp[full - 1], order

def ndcg_score(ranked_rels, k=10, normalize=True):
    """nDCG@k。ranked_rels"""
    ranked = ranked_rels[:k]
    dcg = sum(r / log2(2 + pos) for pos, r in enumerate(ranked))
    if not normalize:
        return dcg
    ideal = sorted(ranked_rels, reverse=True)[:k]
    idcg = sum(r / log2(2 + pos) for pos, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0
 
def ndcg_seren_score(ranked_seren_labels, k=10):
    """惊喜版 NDCG@k, 对齐 Fu et al. SIGIR'23 论文 Eq.(16)。

        NDCG_seren@k = Σ_{i=1..k} serendipity_score(0/1) / log2(i+1)

    ranked_seren_labels: 按推荐顺序的惊喜标注(1=惊喜, 0=非惊喜)。
    """
    ranked = list(ranked_seren_labels)[:k]
    return float(sum(float(r) / log2(i + 2) for i, r in enumerate(ranked)))
 
def epc_score(ranked_items, item_popularity, max_pop, k=10, num_users=None, rel=None):
    """EPC (Expected Popularity Complement, Vargas & Castells, RecSys 2011)。
        EPC = C · Σ_k disc(k) · p(rel|i_k,u) · (1 − p(seen|i_k))
            C    = 1/Σ disc
            disc(k)     = 1/log2(k+2)
            p(seen|i)   = pop(i)/|U|     ← |U| 是**用户数**
            p(rel|i,u)  = **ground-truth 二元相关性**
    参数:
        max_pop     : 仅有 num_users 缺失时用作 |U| 的兜底
        num_users   : |U|。
        rel         : {item: p(rel|i,u)} 的 ground-truth 相关性。不给则整项置 1,
    """
    ranked = list(ranked_items)[:k]
    _U = float(num_users) if num_users else float(max_pop)
    num, den = 0.0, 0.0
    for pos, item in enumerate(ranked):
        disc = 1.0 / log2(2 + pos)
        p_seen = float(item_popularity.get(item, 0)) / (_U + 1e-9)
        p_rel = 1.0 if rel is None else float(rel.get(item, 0.0))
        num += disc * p_rel * (1.0 - p_seen)
        den += disc
    return num / den if den > 0 else 0.0
 
def ild_score(ranked_items, sim_func, k=10):
    """ILD@k = 列表内物品两两距离(1-sim)的平均。
    sim_func(a,b)： 相似度。"""
    items = ranked_items[:k]
    if len(items) < 2:
        return 0.0
    dists = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            dists.append(1.0 - sim_func(items[i], items[j]))
    return float(np.mean(dists)) if dists else 0.0
 
def beta_func(r, r_max, beta, r_min=1.0):
    """Eq.6: β(u,r) = (r/r_max) * β. 评分置信度。"""
    return (float(r) / float(r_max)) * beta
 
def compute_gamma(user_hist_cates, user_hist_ratings, cate, r_max):
    """Eq.9: γ^u_φ = aspect φ 对用户 u 的整体兴趣。 = 用户历史中属于该类别的物品评分和 / 用户历史所有评分和。
    用户历史: user_hist_cates[i]=类别, user_hist_ratings[i]=评分。"""
    num = sum(r for c, r in zip(user_hist_cates, user_hist_ratings) if c == cate)
    den = sum(user_hist_ratings)
    if den <= 0:
        return 0.0
    return num / den

def hit_at_k(ranked_items, k, positives):
    """
    ranked_items : 已按推荐顺序排好的物品列表
    positives    : 该次评估认定的正样本集合 (惊喜标注为1的物品)
    """
    ranked = list(ranked_items)[:k]
    if not ranked:
        return 0.0
    pos = set(positives)
    return 1.0 if any(it in pos for it in ranked) else 0.0


def hit_random_baseline(k, n_candidates, n_positive):
    """
    Hit@k 的随机排序期望：1 - C(n_cand-n_pos, k) / C(n_cand, k)。用作 Reference
    """
    n = int(n_candidates)
    npos = int(n_positive)
    k = int(k)
    if n <= 0 or k <= 0 or npos <= 0 or k > n:
        return 0.0
    if npos >= n:
        return 1.0
    # P(全不命中) = C(n-npos, k) / C(n, k);用对数连乘避免大组合数溢出
    nneg = n - npos
    if k > nneg:
        return 1.0
    log_p = 0.0
    for i in range(k):
        log_p += np.log(nneg - i) - np.log(n - i)
    return float(np.clip(1.0 - np.exp(log_p), 0.0, 1.0))


def abeta_ndcg(ranked_items, item_cate, item_rating, user_hist_cates,
               user_hist_ratings, k=10, alpha=0.005, beta=0.5, r_max=5.0,
               normalize=True):
    """计算一个推荐列表的 αβ-nDCG@k。
    参数:
      ranked_items: 推荐列表(物品id, 已按推荐顺序排)
      item_cate: {item_id: cate_id} 物品类别映射
      item_rating: {item_id: rating} 用户对物品的评分
      user_hist_cates, user_hist_ratings: 用户历史的类别和评分
      k: cutoff
      alpha, beta: 论文参数
      r_max: 最大评分
    返回: αβ-nDCG@k 值
    """
    ranked = ranked_items[:k]
    all_cates = set(item_cate.get(it) for it in ranked if item_cate.get(it) is not None)
    gamma = {c: compute_gamma(user_hist_cates, user_hist_ratings, c, r_max) for c in all_cates}
 
    def gain_at(ranked_list, pos):
        """计算 ranked_list 中位置 pos(0-indexed) 物品的 gain G。
        需要 pos 之前的物品来算冗余惩罚(τ, ρ)。"""
        item = ranked_list[pos]
        cate = item_cate.get(item)
        if cate is None:
            return 0.0
        if item in item_rating:
            p_a_given_ui = beta_func(item_rating[item], r_max, beta)
        else:
            p_a_given_ui = alpha
        g = gamma.get(cate, 0.0)
        tau = 0
        rho = {}  # rating -> count
        for j in range(pos):
            pj = ranked_list[j]
            if item_cate.get(pj) == cate:
                if pj in item_rating:
                    r = item_rating[pj]
                    rho[r] = rho.get(r, 0) + 1
                else:
                    tau += 1
        redundancy = (1 - alpha) ** tau
        for r, cnt in rho.items():
            redundancy *= (1 - beta_func(r, r_max, beta)) ** cnt
        contrib = p_a_given_ui * g * redundancy
        return contrib
 
    dcg = 0.0
    for pos in range(len(ranked)):
        g = gain_at(ranked, pos)
        dcg += g / log2(2 + pos)
 
    _k = len(ranked)
    _items = list(ranked)

    def _gain_of_prefix(mask, i):
        lst = [_items[j] for j in range(_k) if (mask >> j) & 1] + [_items[i]]
        return gain_at(lst, len(lst) - 1)

    if _k <= 12:
        idcg, ideal_order = exact_idcg(_k, _gain_of_prefix)
    else:
        remaining = list(_items)
        ideal_order = []
        while remaining:
            best, best_g = None, -1
            for it in remaining:
                g = gain_at(ideal_order + [it], len(ideal_order))
                if g > best_g:
                    best_g, best = g, it
            ideal_order.append(best)
            remaining.remove(best)
        idcg = 0.0
        for pos in range(len(ideal_order)):
            idcg += gain_at(ideal_order, pos) / log2(2 + pos)

    if idcg <= 0:
        return 0.0
    return (dcg / idcg) if normalize else dcg


def compute_gamma_aspects(hist_aspects, hist_ratings, aspect, r_max=5.0):
    """
    Eq.9 的多 aspect 版: γ_u(a) = Σ_h r_h·w(a|h) / Σ_h r_h。
    """
    num = 0.0
    for asp, r in zip(hist_aspects, hist_ratings):
        if not asp:
            continue
        w = asp.get(aspect)
        if w:
            num += float(r) * float(w)
    den = float(sum(hist_ratings))
    if den <= 0:
        return 0.0
    return num / den


def rel_unexp_score(ranked_items, item_rating, item_pop, max_pop, k=10,
                    r_max=5.0, item_novelty=None, use_arousal=False):
    """相关性 x 意外性基线: serendipity = relevance x unexpectedness。
    - relevance      : 物品评分 r / r_max
    - unexpectedness : 默认用流行度补 1 - pop/max_pop;
                       use_arousal=True 时改用 Wundt 倒 U 曲线 A(n) = 4n(1-n) over 新奇度 n
    返回未归一化的 DCG 累加值。
    """
    s = 0.0
    for j, it in enumerate(list(ranked_items)[:k]):
        rel = float(item_rating.get(it, 0.0)) / float(r_max)
        if use_arousal and item_novelty is not None:
            n = float(item_novelty.get(it, 0.0))
            unexp = 4.0 * n * (1.0 - n)
        else:
            unexp = 1.0 - float(item_pop.get(it, 0)) / (float(max_pop) + 1e-9)
        s += rel * unexp / log2(2 + j)
    return s

def alpha_ndcg(ranked_items, item_cate, item_rating, user_hist_cates,
               user_hist_ratings, k=10, alpha_c=0.5, r_max=5.0,
               normalize=True):
    """
    Clarke et al. 2008 的 alpha-nDCG
    """
    ranked = list(ranked_items)[:k]

    def gain_at(lst, pos):
        item = lst[pos]
        cate = item_cate.get(item)
        if cate is None:
            return 0.0
        rel = float(item_rating.get(item, 0.0)) / float(r_max)
        t = 0
        for j in range(pos):
            if item_cate.get(lst[j]) == cate:
                t += 1
        return rel * ((1.0 - alpha_c) ** t)

    dcg = 0.0
    for pos in range(len(ranked)):
        dcg += gain_at(ranked, pos) / log2(2 + pos)

    _k = len(ranked)
    _items = list(ranked)

    def _gain_of_prefix(mask, i):
        lst = [_items[j] for j in range(_k) if (mask >> j) & 1] + [_items[i]]
        return gain_at(lst, len(lst) - 1)

    if _k <= 12:
        idcg, ideal_order = exact_idcg(_k, _gain_of_prefix)
    else:
        remaining = list(_items)
        ideal_order = []
        while remaining:
            best, best_g = None, -1
            for it in remaining:
                g = gain_at(ideal_order + [it], len(ideal_order))
                if g > best_g:
                    best_g, best = g, it
            ideal_order.append(best)
            remaining.remove(best)
        idcg = 0.0
        for pos in range(len(ideal_order)):
            idcg += gain_at(ideal_order, pos) / log2(2 + pos)

    if idcg <= 0:
        return 0.0
    return (dcg / idcg) if normalize else dcg

def abeta_ndcg_multi(ranked_items, item_aspects, item_rating, user_hist_aspects,
               user_hist_ratings, k=10, alpha=0.005, beta=0.5, r_max=5.0,
               normalize=True):
    """αβ-nDCG 的多 aspect 版
      P(a|u,i) = β(u,r_ui)·w(a|i)        (缺评分时 = α·w(a|i))
      red_a(i) = (1-α)^{τ_a}·Π_r (1-β(u,r))^{ρ_{a,r}}
                 τ_a / ρ_{a,r} 只统计**该物品之前**、且带 aspect a 的物品
    """
    ranked = list(ranked_items)[:k]

    def gain_at(seq, pos):
        item = seq[pos]
        asp = item_aspects.get(item)
        if not asp:
            return 0.0
        if item in item_rating:
            p_a = beta_func(item_rating[item], r_max, beta)
        else:
            p_a = alpha
        prod = 1.0
        for a, w in asp.items():
            if not w or w <= 0:
                continue
            g_a = compute_gamma_aspects(user_hist_aspects, user_hist_ratings, a, r_max)
            if g_a <= 0:
                continue
            tau, rho = 0, {}
            for j in range(pos):
                aspj = item_aspects.get(seq[j])
                if not aspj or not aspj.get(a):
                    continue
                pj = seq[j]
                if pj in item_rating:
                    r = item_rating[pj]
                    rho[r] = rho.get(r, 0) + 1
                else:
                    tau += 1
            red = (1.0 - alpha) ** tau
            for r, cnt in rho.items():
                red *= (1.0 - beta_func(r, r_max, beta)) ** cnt
            prod *= (1.0 - p_a * float(w) * g_a * red)
        return 1.0 - prod

    dcg = 0.0
    for pos in range(len(ranked)):
        dcg += gain_at(ranked, pos) / log2(2 + pos)

    remaining = list(ranked)
    ideal = []
    while remaining:
        best, best_g = None, -1.0
        for it in remaining:
            g = gain_at(ideal + [it], len(ideal))
            if g > best_g:
                best_g, best = g, it
        ideal.append(best)
        remaining.remove(best)
    idcg = 0.0
    for pos in range(len(ideal)):
        idcg += gain_at(ideal, pos) / log2(2 + pos)
    if idcg <= 0:
        return 0.0
    return (dcg / idcg) if normalize else dcg

if __name__ == "__main__":
    np.random.seed(0)
    pos_rel = np.random.beta(3, 2, 100); pos_dif = np.random.beta(2, 3, 100)
    neg_rel = np.random.beta(2, 3, 100); neg_dif = np.random.beta(3, 2, 100)
    pp = fit_class_params(pos_rel, pos_dif, use_copula=False)
    np_ = fit_class_params(neg_rel, neg_dif, use_copula=False)
    pass
    pass
    s_sur = arousal_score(0.8, 0.2, pp, np_, use_copula=False)
    s_norm = arousal_score(0.2, 0.8, pp, np_, use_copula=False)
    pass
    pass
    pass

    ranked = [(1, 0.7, 0.3, 0.5, 0.6, 0.8),
              (2, 0.3, 0.7, 0.4, 0.5, 0.6),
              (3, 0.6, 0.4, 0.6, 0.4, 0.7)]
    m = mdcg_score(ranked, pp, np_, use_copula=False, k=3)
    pass

    pass
    pass
    pass


def ppmi_positive(n_ij, n_i, n_j, n_users, alpha=1.0):
    """SPADE 用的平滑正 PMI（原文 §3.1, α=1）：PPMI_α(i,j) = max{0, log( (n_ij + α) / ( (1/n)·n_i·n_j + α ) ) }

    n_ij = 同时交互过 i 与 j 的用户数; n_i / n_j = 各自用户数; n = 总用户数。
    """
    den = (float(n_i) * float(n_j)) / max(1.0, float(n_users))
    v = log2((float(n_ij) + alpha) / (den + alpha)) * 0.6931471805599453  # log -> ln
    return v if v > 0.0 else 0.0


def avg_recommendation_popularity(items, item_pop, max_pop, k=10):
    """ARP: 推荐列表的平均流行度（按 max_pop 归一到 [0,1]）。越低说明越不依赖热门物品。"""
    lst = list(items)[:k]
    if not lst:
        return 0.0
    mp = float(max_pop) + 1e-9
    return float(sum(float(item_pop.get(x, 0)) for x in lst) / (len(lst) * mp))


def mean_self_information(items, item_pop, total, k=10):
    """Novelty: -log2(p(i))，p(i)=pop_i/总交互数。越大越"新"。"""
    lst = list(items)[:k]
    if not lst or float(total) <= 0:
        return 0.0
    s = 0.0
    for x in lst:
        p = max(1e-12, float(item_pop.get(x, 0)) / float(total))
        s += -log2(p)
    return s / len(lst)


def calibration(items, item_cate, hist_cates, k=10):
    """
    Steck 2018 的 calibration: 1 - 0.5*sum_c |p(c|推荐) - p(c|历史)|，取值 [0,1]。
    衡量"推荐列表的类目分布与用户历史类目分布有多一致"，1 = 完全一致。
    """
    lst = [x for x in list(items)[:k] if item_cate.get(x) is not None]
    hc = [x for x in hist_cates if x is not None]
    if not lst or not hc:
        return 0.0
    rc, hcc = {}, {}
    for x in lst:
        c = item_cate.get(x)
        rc[c] = rc.get(c, 0) + 1
    for c in hc:
        hcc[c] = hcc.get(c, 0) + 1
    d = 0.0
    for c in set(rc) | set(hcc):
        d += abs(rc.get(c, 0) / float(len(lst)) - hcc.get(c, 0) / float(len(hc)))
    return max(0.0, 1.0 - 0.5 * d)


def exposure_stats(items, item_pop=None):
    """目录覆盖率(相对 item_pop 的目录) / 曝光 Gini(越低越分散) / 长尾曝光占比。
    长尾定义: 流行度不超过(item_pop 的)中位数的物品。
    """
    if not items:
        return 0.0, 0.0, 0.0
    cnt = {}
    for x in items:
        cnt[x] = cnt.get(x, 0) + 1
    n_items = len(item_pop) if item_pop else len(cnt)
    cov = float(len(cnt)) / float(n_items) if n_items else 0.0
    v = sorted(cnt.values())
    n = len(v)
    tot = float(sum(v))
    if n == 0 or tot <= 0:
        gini = 0.0
    else:
        acc = 0.0
        for i, x in enumerate(v, 1):
            acc += (2 * i - n - 1) * x
        gini = acc / (n * tot)
    tail = 0.0
    if item_pop and tot > 0:
        pops = sorted(item_pop.values())
        med = pops[len(pops) // 2] if pops else 0
        tail = sum(c for x, c in cnt.items() if float(item_pop.get(x, 0)) <= med) / tot
    return cov, gini, tail


def greedy_by_gain(n, start, gain, ctx):
    """通用增量贪心，同分取序号小者。返回 0-based 序号列表。"""
    order = [start]
    rem = [i for i in range(n) if i != start]
    while rem:
        best, bg = None, -1e18
        for i in rem:
            g = gain(i, order, ctx)
            if g > bg:
                bg, best = g, i
        order.append(best)
        rem.remove(best)
    return order


def _cat_dist(cats, hist_cats):
    """类目分布（列表 -> {cat: prob}），用于 xQuAD / IA-Select / PM-2 / Calibration。"""
    d = {}
    for c in cats:
        if c is not None:
            d[c] = d.get(c, 0.0) + 1.0
    s = sum(d.values())
    if s <= 0:
        return {}
    for k in d:
        d[k] /= s
    return d


def gain_mmr(lam=0.5):
    """MMR（Carbonell & Goldstein, SIGIR 1998）: lam*rel - (1-lam)*max_{j in sel} sim(i,j)。"""
    def g(i, order, ctx):
        r = float(ctx['rel'][i])
        if not order or ctx.get('sim') is None:
            return r
        m = max(float(ctx['sim'][i][j]) for j in order)
        return lam * r - (1.0 - lam) * m
    return g


def gain_lincombo(w):
    """最优线性组合基线: 五因子的线性加权 w·x。贪心退化成"按分数排序"。
    """
    def g(i, order, ctx):
        x = ctx['lc'][i]
        return float(sum(w[k] * x[k] for k in range(len(w))))
    return g


def gain_xquad(lam=0.5):
    """xQuAD（Santos, Macdonald & Ounis, SIGIR 2010）: (1-lam)*rel + lam*sum_a p(a|hist)*[1-p(a|sel)]*w(a|i)。"""
    def g(i, order, ctx):
        hd = ctx['hist_dist']
        sel = _cat_dist([ctx['cats'][j] for j in order], None) if order else {}
        c = ctx['cats'][i]
        rel = float(ctx['rel'][i])
        cover = 0.0
        if c is not None:
            cover = hd.get(c, 0.0) * (1.0 - sel.get(c, 0.0))
        return (1.0 - lam) * rel + lam * cover
    return g














def ge_unexpected_scores(pop, pop_thr):
    """Ge et al. (RecSys 2010) 的 primitive 意外性: 不在流行度基线 top 段： 1, 否则 0。"""
    return [1.0 if float(p) <= pop_thr else 0.0 for p in pop]


def content_dist_scores(rel):
    """内容距离意外性（Shani & Gunawardana, Recommender Systems Handbook）: unexp = 1 - 与历史相似度。"""
    return [1.0 - float(r) for r in rel]


def ppmi_unexpected_scores(rel, ppmi):
    """共现意外性（Kaminskas & Bridge）: 与历史共现越低越意外; 乘相关性。"""
    return [float(rel[i]) * (1.0 / (1.0 + max(0.0, float(ppmi[i])))) for i in range(len(rel))]




def err_score(rels, k=10, r_max=5.0, use_grade=True):
    """ERR = sum_t (1/t) * R_t * prod_{j<t} (1 - R_j)。

    R=(2^r - 1)/2^r_max
    """
    surv = 1.0
    tot = 0.0
    for t, r in enumerate(list(rels)[:k]):
        if use_grade:
            R = (2.0 ** float(r) - 1.0) / (2.0 ** float(r_max))
        else:
            R = float(r) / float(r_max)
        R = min(max(R, 0.0), 1.0)
        tot += surv * R / (t + 1.0)
        surv *= (1.0 - R)
    return tot


def rbp_score(rels, p=0.8, k=10, r_max=5.0):
    """RBP = (1 - p) * sum_t p^(t-1) * (r_t / r_max)。p 默认 0.8。"""
    tot = 0.0
    w = 1.0 - p
    for r in list(rels)[:k]:
        tot += w * (float(r) / float(r_max))
        w *= p
    return tot




def popularity_damping_scores(rel, pop, max_pop, alpha=0.5, eps=1e-6):
    """Steck (RecSys 2011) 的流行度阻尼: rel * (pop/max_pop + eps)^(-alpha)。alpha=0 退化为纯相关性。"""
    mp = float(max_pop) + 1e-9
    out = []
    for i in range(len(rel)):
        p = float(pop[i]) / mp
        out.append(float(rel[i]) * ((p + eps) ** (-float(alpha))))
    return out


