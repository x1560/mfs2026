# -*- coding: utf-8 -*-
"""4.2 数据分析（上半）: 判别力与分布 —— 回答两个问题
  (1) 单因子 / 最优线性组合 是否都无法分开惊喜与非惊喜?
      额外给出"联合结构"的增益: 2D 直方图密度比 (无模型假设) 与灵活模型的 AUC,
      用来量化"边缘分布重合但联合结构不同"。
  (2) 为何用 Beta 分布: 逐因子逐类拟合 Beta vs 高斯 vs 截断高斯,
      报 KS、AIC 与"高斯落在 [0,1] 之外的概率质量"。

口径与 eval.py 完全一致: 先做因果 E(逐用户居中) + D(全局百分位) 再统计。
输出: _diag/_42_auc_ovl.txt
"""
import io
import os
import sys

import numpy as np
import pandas as pd
from scipy import optimize, stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, 'C:/Users/Administrator/Desktop/study/论文复现/code')
os.chdir('C:/Users/Administrator/Desktop/study/论文复现/code')
sys.argv = ['x']

import eval as E      # noqa: E402
import metrics as M   # noqa: E402

OUT = io.open('_diag/_42_auc_ovl.txt', 'w', encoding='utf-8', buffering=1)


def P(*a):
    s = ' '.join(str(x) for x in a)
    OUT.write(s + '\n')
    print(s.encode('ascii', 'replace').decode('ascii'), flush=True)


DS = [('Books', 'SEREN_Books', './data/SerenLens/SerenLens_Books.csv'),
      ('Movies', 'SEREN_Movies', './data/SerenLens/SerenLens_Movies.csv'),
      ('S2018', 'SEREN_S2018', 'SerenLens_S2018.csv')]
FAC = ['rel', 'dif', 'novelty', 'unpopularity', 'qua']   # 原始因子值(未经 D+E / 百分位)


def load(tag, csv):
    fdf = pd.read_pickle('factors_%s.pkl' % tag)
    if tag == 'SEREN_S2018':
        # [nolow] 与 --drop_nonsurvey_neg 同口径: 只留问卷标注行
        _bp = pd.read_csv('S2018_survey_pairs.csv', dtype=str)
        _s = set(zip(_bp.iloc[:, 0].astype(str), _bp.iloc[:, 1].astype(str)))
        _keep = [ (str(u), str(i)) in _s for u, i in zip(fdf['uid'], fdf['item']) ]
        fdf = fdf[_keep].reset_index(drop=True)
        print('  [nolow] 因子表 -> %d 行 (pos=%d neg=%d)' % (len(fdf), int((fdf['label']==1).sum()), int((fdf['label']==0).sum())))
    user_seq, labels, label_time, ir = E.load_serenlens(csv)
    df = E._attach_ts(fdf, label_time).sort_values('ts').reset_index(drop=True)
    ts = df['ts'].values.astype(np.int64)
    uid = df['uid'].values
    lab = df['label'].values.astype(int)
    rel, dif = df['rel'].values.astype(float), df['dif'].values.astype(float)
    cr, cd, cn = E._causal_running_center(uid, ts, lab, rel, dif, 5)
    one = np.zeros(len(ts), dtype=np.int64)
    gr, gd, _ = E._causal_running_center(one, ts, lab, rel, dif, 5)
    need = cn < 5
    rel = np.where(need, rel - gr, rel - cr)
    dif = np.where(need, dif - gd, dif - cd)
    df['rel_t'] = M.quantile_normalize(rel)
    df['dif_t'] = M.quantile_normalize(dif)
    # eval.py:2843-2844 对 nov/unpop 也做分位归一化, 分析必须在同一空间
    df['nov_q'] = M.quantile_normalize(df['novelty'].values)
    df['unpop_q'] = M.quantile_normalize(df['unpopularity'].values)
    return df, lab


def ovl(x1, x2, bins=60):
    """两类分布的重叠系数(OVL): 直方图密度逐格取小后求和。"""
    lo = min(x1.min(), x2.min())
    hi = max(x1.max(), x2.max())
    if hi <= lo:
        return 1.0
    h1, e = np.histogram(x1, bins=bins, range=(lo, hi), density=True)
    h2, _ = np.histogram(x2, bins=bins, range=(lo, hi), density=True)
    return float(np.minimum(h1, h2).sum() * (e[1] - e[0]))


def auc_dir(y, x):
    a = roc_auc_score(y, x)
    return (a, '+') if a >= 0.5 else (1.0 - a, '-')


def crossfit_auc(X, y, kind='logit', seed=0):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        if kind == 'logit':
            sc = StandardScaler().fit(X[tr])
            m = LogisticRegression(max_iter=2000, class_weight='balanced')
            m.fit(sc.transform(X[tr]), y[tr])
            oof[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
        else:
            m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                               random_state=seed)
            m.fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, oof), oof


def crossfit_auc_2d(x, z, y, nb=20, seed=0):
    """(x,z) 上的 2D 直方图密度比, 交叉拟合 —— 无模型假设的"联合结构"上限。"""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    ex = np.linspace(np.percentile(x, 0.5), np.percentile(x, 99.5), nb + 1)
    ez = np.linspace(np.percentile(z, 0.5), np.percentile(z, 99.5), nb + 1)
    oof = np.zeros(len(y))
    for tr, te in skf.split(x, y):
        Hp, _, _ = np.histogram2d(x[tr][y[tr] == 1], z[tr][y[tr] == 1], bins=[ex, ez])
        Hn, _, _ = np.histogram2d(x[tr][y[tr] == 0], z[tr][y[tr] == 0], bins=[ex, ez])
        Hp = (Hp + 0.5) / (Hp.sum() + 0.5 * Hp.size)
        Hn = (Hn + 0.5) / (Hn.sum() + 0.5 * Hn.size)
        ix = np.clip(np.digitize(x[te], ex) - 1, 0, nb - 1)
        iz = np.clip(np.digitize(z[te], ez) - 1, 0, nb - 1)
        oof[te] = np.log(Hp[ix, iz] / Hn[ix, iz])
    return roc_auc_score(y, oof), oof


def beta_fit(x):
    a, b, _, _ = stats.beta.fit(x, floc=0, fscale=1)
    return float(a), float(b)


def tg_cdf(x, mu, sg):
    """截断高斯(限在[0,1])的 CDF。自己算, 不用 scipy.truncnorm.cdf
    —— 后者在边界极端时会递归爆栈(RecursionError)。"""
    lo = float(stats.norm.cdf((0.0 - mu) / sg))
    hi = float(stats.norm.cdf((1.0 - mu) / sg))
    den = max(hi - lo, 1e-12)
    return (stats.norm.cdf((np.asarray(x, dtype=float) - mu) / sg) - lo) / den


def trunclnorm_fit(x):
    """截断高斯的 MLE (参数带界, 避免退化解)。返回 (mu, sigma, loglik)。"""
    def nll(p):
        mu = float(np.clip(p[0], -1.0, 2.0))
        sg = float(np.exp(np.clip(p[1], np.log(0.01), np.log(2.0))))
        lo = float(stats.norm.cdf((0.0 - mu) / sg))
        hi = float(stats.norm.cdf((1.0 - mu) / sg))
        ll = (stats.norm.logpdf((x - mu) / sg) - np.log(sg)
              - np.log(max(hi - lo, 1e-12)))
        return -float(np.sum(ll))
    best, bv = None, np.inf
    for mu0 in (0.3, 0.8):
        for s0 in (-2.5, -1.0):
            r = optimize.minimize(nll, [mu0, s0], method='Nelder-Mead',
                                  options={'maxiter': 150, 'xatol': 1e-4, 'fatol': 1e-3})
            if r.fun < bv:
                bv, best = r.fun, r.x
    mu = float(np.clip(best[0], -1.0, 2.0))
    sg = float(np.exp(np.clip(best[1], np.log(0.01), np.log(2.0))))
    return mu, sg, float(-bv)


def fit_stats(v, rng, cap=20000):
    """对一个因子的一个类别: Beta / 高斯 / 截断高斯 (都是 2 参数) 的 KS 与 AIC。

    为速度考虑, 在固定随机子样本(<=2e4)上拟合与评估 —— 三个分布用**同一份**样本,
    所以 AIC 之间可比。截断高斯的 MLE 若用全量 25 万点会慢到不可接受。
    """
    v = v[np.isfinite(v)]
    if len(v) > cap:
        v = rng.choice(v, cap, replace=False)
    edge = float(((v <= 0.0) | (v >= 1.0)).mean())
    vc = np.clip(v, 1e-6, 1 - 1e-6)          # Beta 的 MLE 要求严格落在 (0,1)
    a, b = beta_fit(vc)
    ks_b = stats.kstest(vc, 'beta', args=(a, b)).statistic
    aic_b = -2 * float(np.sum(stats.beta.logpdf(vc, a, b))) + 4
    mu, sg = float(v.mean()), float(v.std())
    ks_g = stats.kstest(v, 'norm', args=(mu, sg)).statistic
    aic_g = -2 * float(np.sum(stats.norm.logpdf(v, mu, sg))) + 4
    mass = float(stats.norm.cdf(0, mu, sg) + 1 - stats.norm.cdf(1, mu, sg))
    mu2, sg2, ll_t = trunclnorm_fit(vc)
    try:
        ks_t = stats.kstest(vc, lambda q: tg_cdf(q, mu2, sg2)).statistic
    except Exception:
        ks_t = float('nan')
    aic_t = -2 * ll_t + 4
    if a < 1 and b < 1:
        shape = 'U形'
    elif abs(a - 1) < 0.08 and abs(b - 1) < 0.08:
        shape = '~均匀'
    elif a > 1 and b > 1:
        shape = '单峰'
    else:
        shape = '单边'
    return dict(a=a, b=b, ks_b=ks_b, aic_b=aic_b, mu=mu, sg=sg, ks_g=ks_g,
                aic_g=aic_g, mass=mass, mu2=mu2, sg2=sg2, ks_t=ks_t,
                aic_t=aic_t, edge=edge, shape=shape)


def main():
    for name, tag, csv in DS:
        df, y = load(tag, csv)
        if tag == 'SEREN_S2018':
            # 4.2 的 S2018 只用 survey 答案行(2150): 正 = round(mean(s4,s5,s6))>3, 负 = survey 自己的非惊喜
            _a = pd.read_csv('data/serendipity-sac2018/answers.csv')
            _a['uid'] = _a['userId'].astype(str); _a['item'] = _a['movieId'].astype(str)
            _a['y2'] = (_a[['s4', 's5', 's6']].mean(axis=1).round() > 3).astype(int)
            _df = df.copy(); _df['uid'] = _df['uid'].astype(str); _df['item'] = _df['item'].astype(str)
            _m = _df.merge(_a[['uid', 'item', 'y2']], on=['uid', 'item'], how='inner')
            df = _m.drop(columns=['y2']).reset_index(drop=True)
            y = _m['y2'].values
            P('  [survey-only] S2018 只保留 survey 答案行: %d (pos %d / neg %d)' % (len(y), int(y.sum()), int((1 - y).sum())))
        rng = np.random.RandomState(0)
        n, npos = len(df), int(y.sum())
        P('=' * 78)
        P('### %s | %d 行 | 正样本 %d (%.2f%%)' % (name, n, npos, 100.0 * npos / n))

        # ---------- S1 单因子判别力 ----------
        P('\n[S1] 单因子判别力 (AUC 已取>0.5, 方向列给出符号; OVL 越低越可分)')
        P('  %-9s | %-22s | %-22s' % ('factor', 'raw: AUC(dir) / OVL', 'E+D 后: AUC(dir) / OVL'))
        for f in FAC:
            if f == 'rel':
                xr, xt = df['rel'].values, df['rel_t'].values
            elif f == 'dif':
                xr, xt = df['dif'].values, df['dif_t'].values
            else:
                xr = xt = df[f].values.astype(float)
            a1, d1 = auc_dir(y, xr)
            o1 = ovl(xr[y == 1], xr[y == 0])
            a2, d2 = auc_dir(y, xt)
            o2 = ovl(xt[y == 1], xt[y == 0])
            P('  %-9s | %.4f (%s) / %.3f     | %.4f (%s) / %.3f' % (f, a1, d1, o1, a2, d2, o2))

        # ---------- S2 组合模型 ----------
        X = np.column_stack([df['rel'].values, df['dif'].values,
                             df['novelty'].values, df['unpopularity'].values,
                             df['qua'].values])
        a_lin, s_lin = crossfit_auc(X, y, 'logit')
        o_lin = ovl(s_lin[y == 1], s_lin[y == 0])
        a_hgb, _ = crossfit_auc(X, y, 'hgb')
        a_2d, _ = crossfit_auc_2d(df['rel'].values, df['dif'].values, y)
        P('\n[S2] 组合模型 (5 折交叉拟合, out-of-fold)')
        P('  最优线性 (logistic, 5 因子) : AUC %.4f | OVL %.3f' % (a_lin, o_lin))
        P('  仅 (rel,dif) 的 2D 联合结构  : AUC %.4f   <- 无模型假设' % a_2d)
        P('  灵活模型 (梯度提升, 5 因子)  : AUC %.4f' % a_hgb)
        P('  -> 线性 vs 联合 的增益        : %+.4f' % (a_2d - a_lin))
        P('  -> 线性 vs 灵活 的增益        : %+.4f' % (a_hgb - a_lin))

        # ---------- S3 联合结构 (copula 参数) ----------
        P('\n[S3] (rel, dif) 两类各自的联合结构 (Beta-Copula 拟合, raw 因子)')
        pr = df['rel'].values      # raw 因子(4.2 统一空间)
        pd_ = df['dif'].values
        for lab_, nm in [(1, 'Ser=1'), (0, 'Ser=0')]:
            try:
                pp = M.fit_class_params(pr[y == lab_], pd_[y == lab_], use_copula=True)
                P('  %s : rel Beta(%.2f, %.2f) | dif Beta(%.2f, %.2f) | copula rho=%+.3f'
                  % (nm, pp[0], pp[1], pp[2], pp[3], pp[4]))
            except Exception as e:
                P('  %s : 拟合失败 %s' % (nm, e))
        try:
            r1 = M.fit_class_params(pr[y == 1], pd_[y == 1], use_copula=True)[4]
            r0 = M.fit_class_params(pr[y == 0], pd_[y == 0], use_copula=True)[4]
            P('  -> 两类 rho 之差 = %+.3f (联合结构不同才有意义)' % (r1 - r0))
        except Exception:
            pass
        P('  样本 Kendall tau(rel,dif): Ser=1 %.4f | Ser=0 %.4f'
          % (stats.kendalltau(pr[y == 1], pd_[y == 1])[0],
             stats.kendalltau(pr[y == 0], pd_[y == 0])[0]))

        # ---------- S4 Beta vs Gaussian ----------
        P('\n[S4] 分布拟合（**模型输入空间**：rel/dif 已 E+D，nov/unpop 已分位归一化）')
        P('  同 2 参数对比: Beta(a,b) vs 高斯 vs **截断高斯**(真值限在[0,1], 公平对手)')
        P('  %-9s %-6s | %-14s %-5s %-8s | %-13s %-7s | %s'
          % ('factor', 'class', 'Beta(a,b)', '形状', 'KS', '截断高斯(mu,sd)', 'KS',
             'AIC: Beta-截断(负=截断更好)'))
        for f in FAC:
            col = {'rel': 'rel_t', 'dif': 'dif_t'}.get(f, f)
            x = df[col].values.astype(float)
            for lab_, nm in [(1, 'Ser=1'), (0, 'Ser=0')]:
                v = x[y == lab_]
                if len(v) < 50:
                    continue
                s = fit_stats(v, rng)
                P('  %-9s %-6s | B(%.2f,%.2f) %-5s %7.4f | %.2f/%.2f %8.4f | %+.0f'
                  % (f, nm, s['a'], s['b'], s['shape'], s['ks_b'],
                     s['mu2'], s['sg2'], s['ks_t'], s['aic_b'] - s['aic_t']))
        P('\n[S4b] 原始空间 vs 变换后空间 (AIC 差 = Beta - 截断高斯; 负数 = 截断高斯更好)')
        P('  %-9s %-6s %14s %14s' % ('factor', 'class', 'raw', 'E+D/归一化后'))
        for f in FAC:
            col_t = {'rel': 'rel_t', 'dif': 'dif_t'}.get(f, f)
            for lab_, nm in [(1, 'Ser=1'), (0, 'Ser=0')]:
                v_r = df[f].values.astype(float)[y == lab_]
                v_t = df[col_t].values.astype(float)[y == lab_]
                if len(v_r) < 50:
                    continue
                sr, stt = fit_stats(v_r, rng), fit_stats(v_t, rng)
                P('  %-9s %-6s %+14.0f %+14.0f   (Beta 形状 raw=%s -> 变换后=%s)'
                  % (f, nm, sr['aic_b'] - sr['aic_t'], stt['aic_b'] - stt['aic_t'],
                     sr['shape'], stt['shape']))
        P('')
    OUT.close()
    print('-> _diag/_42_auc_ovl.txt')


if __name__ == '__main__':
    main()


