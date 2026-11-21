# -*- coding: utf-8 -*-
"""比对两份 hit-rate 日志的排序器表，用于验证元组重构是否伤到无关排序器。

用法: python _diag\_cmp_hit_logs.py <旧日志> <新日志>

判读:
  本仓库里**会读 rel** 的排序器（口径修正后它们允许变）:
    content_unexp  1-rel
    mmr / iaselect / pm2 / calibgreedy / coveragegreedy / dpp  经 _ctx9['rel'] 取候选相关性
    damping        rel * pop^-alpha
    ppmi_unexp     经 _rel9
  其余（epc / ild / random / abeta_ndcg / alpha_ndcg / ndcg_rating /
        fu_serendipity / ge_unexp / ndcg_seren / oracle）**必须逐位一致**。
"""
import re
import sys

ROW = re.compile(r'^\s*([A-Za-z_][A-Za-z0-9_]*\*?)\s+'
                 r'(-?[\d.]+|nan)\s+(-?[\d.]+|nan)\s+(-?[\d.]+|nan)\s+(-?[\d.]+|nan)\s*$')
REL_RANKERS = {'content_unexp', 'mmr', 'xquad', 'damping', 'ppmi_unexp',
               'iaselect', 'pm2', 'calibgreedy', 'coveragegreedy', 'dpp'}


def parse(path):
    lines = None
    for enc in ('utf-8', 'gbk', 'latin-1'):
        try:
            lines = open(path, encoding=enc).read().split('\n')
            break
        except UnicodeDecodeError:
            continue
    out = {}
    for ln in lines:
        m = ROW.match(ln)
        if m:
            out[m.group(1)] = tuple(m.group(i) for i in range(2, 6))
    return out


a, b = parse(sys.argv[1]), parse(sys.argv[2])
print('旧表 %d 行: %s' % (len(a), ', '.join(sorted(a))))
print('新表 %d 行: %s' % (len(b), ', '.join(sorted(b))))
only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
if only_a:
    print('!! 新表缺行:', only_a)
if only_b:
    print('!! 新表多行:', only_b)

same, diff_ok, diff_bad = [], [], []
for k in sorted(set(a) & set(b)):
    if a[k] == b[k]:
        same.append(k)
    elif k in REL_RANKERS:
        diff_ok.append(k)
    else:
        diff_bad.append(k)

print()
print('逐位一致 (%d): %s' % (len(same), ', '.join(same)))
print()
print('读 rel、允许变 (%d):' % len(diff_ok))
for k in diff_ok:
    print('   %-16s 旧 %s   ->   新 %s' % (k, ' '.join(a[k]), ' '.join(b[k])))
print()
if diff_bad:
    print('!!! 不该变却变了 (%d) —— 元组重构伤到了无关排序器:' % len(diff_bad))
    for k in diff_bad:
        print('   %-16s 旧 %s   ->   新 %s' % (k, ' '.join(a[k]), ' '.join(b[k])))
else:
    print('OK: 不读 rel 的排序器全部逐位一致。')
