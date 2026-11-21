# -*- coding: utf-8 -*-
"""用户个体层面的【非对称包容曲线】图 + 非对称度分布图。

模型（metrics.py）:
    单峰: q(x) = 1 / [1 + alpha_side (x - lambda)^2],  alpha_side = alpha_minus (x<lambda) / alpha_plus (x>=lambda)
    多峰: q(x) = max_k 1 / [1 + alpha_side (x - lambda_k)^2]      (_cauchy_multipeak_alpha_asym, max-envelope)

数据: scarcity_params.pkl = {uid: {lambda_n, alpha_n_minus, alpha_n_plus, n_centers, k_n, ...}}
输出: M_DCG/pdf/asym_tolerance.png          极端样例（默认）
      M_DCG/pdf/asym_tolerance_moderate.png 中等非对称样例
      M_DCG/pdf/asym_ratio_dist.png         非对称度 log10(a-/a+) 的分布

用法: python _diag/plot_asym_tolerance.py [both|extreme|moderate]
"""
import os
import sys
import pickle

import numpy as np

# matplotlib 默认把字体缓存写到 ~/.matplotlib（工作区外，沙箱会拒绝）-> 指到工作区内
ROOT = os.environ.get('MDCG_ROOT') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('MPLCONFIGDIR', os.path.join(ROOT, '_diag', '_mplcache'))
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.chdir(ROOT)
PDFDIR = os.path.join(ROOT, '..', 'pdf')
X = np.linspace(0.0, 1.0, 501)


def curve(x, centers, am, ap):
    """复刻 _cauchy_multipeak_alpha_asym（max-envelope，左右 α 不同）。"""
    centers = np.asarray(centers, dtype=float)
    resp = []
    for c in centers:
        a = np.where(x < c, float(am), float(ap))
        resp.append(1.0 / (1.0 + a * (x - c) ** 2))
    return np.clip(np.max(np.vstack(resp), axis=0), 0.0, 1.0)


def log_ratio(p, fac='n'):
    am, ap = p['alpha_%s_minus' % fac], p['alpha_%s_plus' % fac]
    return float(np.log10(max(am, 1e-9) / max(ap, 1e-9)))


# 按论文那段描述挑好的三类典型（条件见下方注释；uid 固定以保证可复现）
#   exploratory : alpha=(7.8, 4.3)    两侧都温和，半宽 0.36/0.48
#   conservative: alpha=(266.1, 60.2) 两侧都窄，半宽 0.06/0.13
#   two-peak    : alpha=(5.1, 35.1)   峰距 0.46，两个可接受区域
PICKS = {'exploratory': 'a16jcog68dgpty',
         'conservative': 'a3q7zl6tprx2y',
         'two-peak': 'a2v3cqlct9ot28'}


def pick_users(P, mode):
    """优先用上面锁定的 uid；缺失时按同样的条件回退搜索。"""
    def falls(kind, p):
        am, ap = p['alpha_n_minus'], p['alpha_n_plus']
        mx, mn = max(am, ap), min(am, ap)
        kn = int(p.get('k_n', 1))
        if kind == 'exploratory':
            return kn == 1 and mx <= 20
        if kind == 'conservative':
            return kn == 1 and mn >= 60 and mx <= 300
        c = np.sort(np.asarray(p['n_centers'], dtype=float))
        return kn >= 2 and mn <= 25 and mx <= 100 and float(c[-1] - c[0]) >= 0.4

    out = []
    for kind in ('exploratory', 'conservative', 'two-peak'):
        uid = PICKS.get(kind)
        if uid is None or uid not in P or not falls(kind, P[uid]):
            cand = [u for u, p in P.items() if falls(kind, p)]
            uid = cand[0] if cand else list(P.keys())[0]
        out.append((kind, uid))
    return out


def draw_tolerance(P, mode, out):
    users = pick_users(P, mode)
    rng = np.random.RandomState(0)
    bg = list(rng.choice(list(P.keys()), size=min(250, len(P)), replace=False))

    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.6), sharex=True, sharey=True)
    for col, (kind, uid) in enumerate(users):
        p = P[uid]
        for row, fac in enumerate(('n', 'u')):
            ax = axes[row][col]
            for b in bg:
                pb = P[b]
                ax.plot(X, curve(X, pb['%s_centers' % fac],
                                 pb['alpha_%s_minus' % fac], pb['alpha_%s_plus' % fac]),
                        color='0.85', lw=0.5, zorder=1)
            cs = p['%s_centers' % fac]
            am, ap = p['alpha_%s_minus' % fac], p['alpha_%s_plus' % fac]
            ax.plot(X, curve(X, cs, am, ap), color='#c0392b', lw=2.0, zorder=3)
            for c in cs:
                ax.axvline(c, color='#2471a3', ls='--', lw=1.0, zorder=2)
            ax.set_ylim(-0.03, 1.06)
            ax.set_xlim(0, 1)
            ax.grid(alpha=0.25, lw=0.4)
            lab = ('novelty' if fac == 'n' else 'unpopularity')
            # 下排不重复用户标签：'two-peak' 说的是该用户的 novelty 核，其 unpopularity 可能仍是单峰
            ttl = ('%s: %s   $k=%d$' % (lab, kind, len(cs))) if fac == 'n' \
                else ('%s   $k=%d$' % (lab, len(cs)))
            # 面板标题已移到论文 caption，这里不再画
            # 半宽 = q 掉到 0.5 处的距离 = 1/sqrt(alpha)，比 alpha 直观
            hw_m, hw_p = 1.0 / np.sqrt(max(am, 1e-9)), 1.0 / np.sqrt(max(ap, 1e-9))
            ax.text(0.03, 0.05,
                    r'$\alpha_-=%.1f,\ \alpha_+=%.1f$' % (am, ap) + '\n'
                    + r'half-width $%.2f\,/\,%.2f$' % (hw_m, hw_p),
                    transform=ax.transAxes, fontsize=7.0, va='bottom',
                    bbox=dict(fc='white', ec='0.8', lw=0.4, pad=1.6))
            if row == 1:
                ax.set_xlabel('factor value', fontsize=9)
            if col == 0:
                ax.set_ylabel('tolerance  $q(x)$', fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 1])
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print('saved:', os.path.abspath(out))
    for kind, uid in users:
        p = P[uid]
        print('  %-6s uid=%s  k_n=%d k_u=%d | a_n=(%.0f,%.0f) a_u=(%.0f,%.0f) | n_centers=%s'
              % (kind, uid, p['k_n'], p['k_u'], p['alpha_n_minus'], p['alpha_n_plus'],
                 p['alpha_u_minus'], p['alpha_u_plus'],
                 [round(float(c), 3) for c in p['n_centers']]))


def draw_dist(P, out):
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    for ax, fac, name in ((axes[0], 'n', 'novelty'), (axes[1], 'u', 'unpopularity')):
        r = np.array([log_ratio(p, fac) for p in P.values()])
        r = r[np.isfinite(r)]
        bins = np.linspace(-2.2, 2.2, 45)
        ax.hist(r, bins=bins, color='#5d8aa8', edgecolor='white', lw=0.4)
        ax.axvline(0, color='0.3', lw=1.0)
        for s in (-0.5, 0.5):
            ax.axvline(s, color='#c0392b', ls='--', lw=1.0)
        ax.set_title('%s   (left-steep %d | symmetric %d | right-steep %d)'
                     % (name, (r > 0.5).sum(), (np.abs(r) <= 0.5).sum(), (r < -0.5).sum()),
                     fontsize=9)
        ax.set_xlabel(r'$\log_{10}(\alpha_-/\alpha_+)$', fontsize=9)
        ax.set_ylabel('users', fontsize=9)
        ax.grid(alpha=0.25, lw=0.4)
    fig.suptitle('Asymmetry of the per-user tolerance kernel '
                 '(dashed: a factor of three in curvature)', fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print('saved:', os.path.abspath(out))


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'both'
    P = pickle.load(open('scarcity_params.pkl', 'rb'))
    if mode in ('both', 'extreme'):
        draw_tolerance(P, 'extreme', os.path.join(PDFDIR, 'asym_tolerance.png'))
    if mode in ('both', 'moderate'):
        draw_tolerance(P, 'moderate', os.path.join(PDFDIR, 'asym_tolerance_moderate.png'))
    if mode == 'both':
        draw_dist(P, os.path.join(PDFDIR, 'asym_ratio_dist.png'))


if __name__ == '__main__':
    main()
