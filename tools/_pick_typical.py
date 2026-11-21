# -*- coding: utf-8 -*-
"""为三类典型找合格用户（收紧了条件）。"""
import pickle
import numpy as np

P = pickle.load(open('scarcity_params.pkl', 'rb'))


def sep_of(p):
    c = np.sort(np.asarray(p['n_centers'], float))
    return float(c[-1] - c[0]) if len(c) > 1 else 0.0


one = [(u, p) for u, p in P.items() if int(p.get('k_n', 1)) == 1]
two = [(u, p) for u, p in P.items() if int(p.get('k_n', 1)) >= 2]

print('== conservative 候选: 两侧都窄 (min_a>=60 且 max_a<=300)，左右差别最大 ==')
c = [(u, p) for u, p in one
     if min(p['alpha_n_minus'], p['alpha_n_plus']) >= 60
     and max(p['alpha_n_minus'], p['alpha_n_plus']) <= 300]
for u, p in sorted(c, key=lambda t: -np.log10(max(t[1]['alpha_n_minus'], t[1]['alpha_n_plus'])
                                              / min(t[1]['alpha_n_minus'], t[1]['alpha_n_plus'])))[:6]:
    print('  %-16s am=%6.1f ap=%6.1f lam=%.3f' % (u, p['alpha_n_minus'], p['alpha_n_plus'], p['n_centers'][0]))
print('  合格人数: %d' % len(c))
print()

print('== two-peak 候选: min_a<=25 且 max_a<=100（两侧都不尖），峰距最大 ==')
c2 = [(u, p) for u, p in two
      if min(p['alpha_n_minus'], p['alpha_n_plus']) <= 25
      and max(p['alpha_n_minus'], p['alpha_n_plus']) <= 100]
for u, p in sorted(c2, key=lambda t: -sep_of(t[1]))[:8]:
    print('  %-16s am=%6.1f ap=%6.1f 峰距=%.3f centers=%s'
          % (u, p['alpha_n_minus'], p['alpha_n_plus'], sep_of(p),
             [round(float(x), 3) for x in p['n_centers']]))
print('  合格人数: %d' % len(c2))
print()

print('== 放宽到 max_a<=200 ==')
c3 = [(u, p) for u, p in two
      if min(p['alpha_n_minus'], p['alpha_n_plus']) <= 25
      and max(p['alpha_n_minus'], p['alpha_n_plus']) <= 200]
for u, p in sorted(c3, key=lambda t: -sep_of(t[1]))[:8]:
    print('  %-16s am=%6.1f ap=%6.1f 峰距=%.3f centers=%s'
          % (u, p['alpha_n_minus'], p['alpha_n_plus'], sep_of(p),
             [round(float(x), 3) for x in p['n_centers']]))
print('  合格人数: %d' % len(c3))
