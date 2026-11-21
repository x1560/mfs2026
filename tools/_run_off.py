# -*- coding: utf-8 -*-
"""Scratch launcher: run eval.py with stdio redirected INTO a log file by Python
itself, so no shell-level pipe/redirect is needed (the sandbox blocks those).

usage: python _diag/_run_off.py <logpath> <eval.py args...>
"""
import os
import runpy
import sys

here = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(here)
os.chdir(root)
sys.path.insert(0, root)

log_path = sys.argv[1]
rest = sys.argv[2:]

log = open(log_path, "w", encoding="utf-8", buffering=1)
sys.stdout = log
sys.stderr = log

sys.argv = ["eval.py"] + rest
try:
    runpy.run_path(os.path.join(root, "eval.py"), run_name="__main__")
finally:
    try:
        log.flush()
        log.close()
    except Exception:
        pass
