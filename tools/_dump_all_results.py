# -*- coding: utf-8 -*-
"""四口径完整结果抽取（v3：锚点行 + 向下取；修掉"负值被正则挡掉"的 bug）。"""
import os
import re

os.chdir(r'C:\Users\Administrator\Desktop\study\论文复现\code')
D = '_diag'
CASES = [('books', 'Books'), ('movies', 'Movies'),
         ('s2018_low', 'S2018带低分'), ('s2018_nolow', 'S2018不带低分')]
MAIN = ['mdcg', 'random', 'ild', 'abeta_ndcg', 'rel_unexp', 'alpha_ndcg',
        'ndcg_rating', 'mmr', 'content_unexp']
EXTRA = ['xquad', 'ge_unexp', 'epc']
RQ2 = ['M-DCG', 'M-DCG(I-DCG)', 'ILD', 'ContentDist', 'abeta-nDCG', 'alpha-nDCG',
       'ndcg_rating', 'MMR', 'RelUnexp']
ABL = ['full', 'global_arousal', 'global_scarcity', 'global_both', 'no_arousal',
       'no_double', 'neither', 'no_qua', 'no_asym', 'no_div', 'no_trust', 'no_div_trust']


def read(p):
    fp = os.path.join(D, p)
    return open(fp, encoding='utf-8', errors='replace').read().splitlines() if os.path.exists(fp) else []


def block(lines, marker, anchor, n=26):
    """在含 marker 的行之后, 找到含 anchor 的行, 返回它及其后 n 行"""
    for i, l in enumerate(lines):
        if marker in l:
            for j in range(i, min(i + 500, len(lines))):
                if anchor in lines[j]:
                    return lines[j:j + n]
    return []


def table(lines, n=14):
    """定位命中率表头: 同一行同时含 '排序器' 和 'Hit@2'。
    注意不能只找 '排序器' —— 驱动会打印 '[补跑排序器] ...' 那行, 会撞上。"""
    for i, l in enumerate(lines):
        if '排序器' in l and 'Hit@2' in l:
            return lines[i:i + n]
    return []


def grab(body, keys):
    res = {}
    for l in body:
        s = l.strip()
        for k in keys:
            if k in res or not s.startswith(k):
                continue
            rest = s[len(k):].strip()
            if re.match(r'^-?\d', rest):          # ← 允许负号开头
                res[k] = rest.split()
    return res


hit, extra, rq2, abl, div, rob, nrm = {}, {}, {}, {}, {}, {}, {}
for tag, _ in CASES:
    h = read('_full_%s_hit.log' % tag)
    hit[tag] = grab(table(h), MAIN)
    extra[tag] = grab(table(read('_full_%s_extra_hit.log' % tag)), EXTRA)
    x = read('_full_%s_exp.log' % tag)
    rq2[tag] = grab(block(x, '实验二', '正Tau比例'), RQ2)
    abl[tag] = grab(block(x, '实验三', '正Tau比例'), ABL)
    div[tag] = grab(block(x, '多样性扰动', '正Tau比例'), ['M-DCG', 'M-DCG(no_div)', 'ILD'])
    rb = [l for l in block(x, '质量鲁棒性', 'Tau', 60) if 'Tau' in l]
    rob[tag] = rb[-1].split('Tau')[-1].lstrip('：: ').strip() if rb else ''
    nm = [l for l in block(x, '归一化', 'spearman', 8) if re.match(r'^\s*-?\d', l)]
    nrm[tag] = ' / '.join(nm[0].split()[:2]) if nm else ''


def md(title, keys, data):
    L = ['### ' + title, '', '| 变体 | ' + ' | '.join(c[1] for c in CASES) + ' |',
         '|' + '---|' * (len(CASES) + 1)]
    for k in keys:
        L.append('| %s | %s |' % (k, ' | '.join(
            ' / '.join(data[t].get(k, [])) or '—' for t, _ in CASES)))
    L.append('')
    return L


out = ['# 新指标口径 · 四口径完整结果（2026-10-09）', '',
       '来源 `_diag/_full_<case>_{hit,exp}.log`；三行补跑 `_diag/_full_<case>_extra_hit.log`。',
       'M-DCG 代码未改动，四口径逐位复现。', '', '---', '']
out += md('RQ1 命中率：Hit@2 / Hit@4 / OrderAUC / 正样本位次（主表 9 个）', MAIN, hit)
out += md('RQ1 命中率：三行补跑（已移出主表）', EXTRA, extra)
out += md('RQ2 扰动一致性：mean Kendall τ / 正方向比例', RQ2, rq2)
out += md('实验三 消融：mean Kendall τ / 正方向比例', ABL, abl)
out += md('实验二b div 扰动：mean Kendall τ / 正方向比例', ['M-DCG', 'M-DCG(no_div)', 'ILD'], div)
out += ['### 其余', '', '| 板块 | ' + ' | '.join(c[1] for c in CASES) + ' |',
        '|' + '---|' * (len(CASES) + 1),
        '| 实验四 归一化 spearman（rho / p）| ' + ' | '.join(nrm[t] or '—' for t, _ in CASES) + ' |',
        '| 实验五 质量鲁棒性 Tau | ' + ' | '.join(rob[t] or '—' for t, _ in CASES) + ' |', '']

txt = '\n'.join(out)
open('_diag/_results_full_dump.md', 'w', encoding='utf-8').write(txt)
print(txt)
