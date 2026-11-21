# -*- coding: utf-8 -*-
"""把所有 .pkl 从 pickle protocol 5 降级重写成 protocol 4。

背景: 本地是 Python 3.8, `pickle.HIGHEST_PROTOCOL` = 5; 服务器是 Python 3.6,
只能读到 protocol 4 → `ValueError: unsupported pickle protocol: 5`。

protocol 4 从 Python 3.4 起就支持, 3.8 也能读, 所以就地降级是安全的。
原文件先备份到 _bak_pkl_proto5_<时间戳>/。

注意: 这只解决 **protocol** 问题。如果服务器上的 pandas/numpy 比本地(2.0.3/1.24.3)
老很多, `factors_*.pkl` 这类 **DataFrame** 仍可能因为 pandas 内部结构变化读不了。
"""
import os
import glob
import pickle
import shutil
import time
import datetime

os.chdir(r'C:\Users\Administrator\Desktop\study\论文复现\code')
TS = datetime.datetime.now().strftime('%Y%m%d_%H%M')
BK = os.path.join('_bak_pkl_proto5_' + TS)
os.makedirs(BK, exist_ok=True)

files = sorted(glob.glob('*.pkl'))
print('找到 %d 个 pkl' % len(files))
n_ok = n_skip = n_fail = 0
for p in files:
    try:
        with open(p, 'rb') as f:
            obj = pickle.load(f)
        shutil.copy2(p, os.path.join(BK, p))
        with open(p, 'wb') as f:
            pickle.dump(obj, f, protocol=4)
        n_ok += 1
        print('  proto4  %-42s %s' % (p, type(obj).__name__))
    except Exception as e:
        n_fail += 1
        print('  FAIL    %-42s %s' % (p, e))
print()
print('完成: 降级 %d | 失败 %d | 备份 -> %s' % (n_ok, n_fail, BK))
