#!/usr/bin/env python3
"""
exp23 - Per-user CatBoost 6h (n=3 anchors, 50 iters per user)
Heavy personalization: global 500it on 750k rows + per-user 50it on 3 rows -> ~6h on 50c server (85ms/user), ~1h on M1
Validation target: 1.638 on fold_03 (1k pilot), prod -> fold_end
Run: .venv/bin/python archive/exp23/src/train_6h.py  (or src/per_user_full.py with ITER=50)
"""
# This is a placeholder for the 6h run - full logic in src/per_user_full.py with ITER_PER_USER=50
# See src/per_user_full.py and reports/per_user_full.json (n=3 it20 1.624) for validation
# For n=3 it50, change ITER_PER_USER=50 and re-run; estimated wall 5.9h on server (50c) with 85ms/user
# See reports/test_n5_it25.json and per_user_dense_pilot.json for ablation of n/iter
import pathlib
print("exp23 6h per-user: see src/per_user_full.py with ITER_PER_USER=50, n=3")
print("Pilot: n=3 it50 1k RMSLE 1.63803 (global 1.69284) mean 16.12ms")
print("Full 250k est 5.9h on 50c server, 1.6h on M1")
