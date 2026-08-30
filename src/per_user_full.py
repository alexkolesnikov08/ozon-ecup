#!/usr/bin/env python3
"""
Full per-user personalization on 250k users
n=3 anchors (fold_00..02), iter=20, 141 feats (features_ext)
Logs per-user timing to ms, checkpoints every 10k
"""
import time
import json
import numpy as np
import polars as pl
from pathlib import Path
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from collections import defaultdict

SAMPLE_N = 250_000
SEED = 42
GLOBAL_ITERS = 500
ITER_PER_USER = 20
BATCH_SIZE = 5000
FOLDS_TRAIN = ["fold_00","fold_01","fold_02"]
FOLD_TEST = "fold_03"
OUT_DIR = Path("data/v2/features_ext")
REPORT_PATH = Path("reports/per_user_full.json")
CKPT_DIR = Path("reports/ckpt_per_user_full")
PRED_PATH = Path("reports/per_user_full_preds.npz")

t0 = time.perf_counter()
def log(msg):
    print(f"[{time.perf_counter()-t0:.3f}s] {msg}", flush=True)

log(f"START FULL n=3 iter={ITER_PER_USER} batch={BATCH_SIZE}")
CKPT_DIR.mkdir(parents=True, exist_ok=True)
Path("reports").mkdir(exist_ok=True)

# Load data for global train
log("Loading GLOBAL train folds 00..02 (all 250k)...")
train_dfs=[]
for fold in FOLDS_TRAIN:
    df = pl.scan_parquet(str(OUT_DIR / fold / "batch_*.parquet")).collect()
    train_dfs.append(df)
    log(f"  {fold} {df.shape}")
train_all = pl.concat(train_dfs)
log(f"train_all {train_all.shape}")

log(f"Loading TEST fold_03...")
test_all = pl.scan_parquet(str(OUT_DIR / FOLD_TEST / "batch_*.parquet")).collect()
log(f"test_all {test_all.shape}")

feature_cols = [c for c in train_all.columns if c not in ("anchor_date","user_id","target")]
log(f"Features {len(feature_cols)}")

def to_numpy(df, cols):
    X = df.select(cols).to_numpy()
    y = df["target"].to_numpy()
    y_z = np.log1p(y)
    return X, y_z

X_train_all, y_train_z_all = to_numpy(train_all, feature_cols)
X_test_all, y_test_z_all = to_numpy(test_all, feature_cols)
user_ids_test_all = test_all["user_id"].to_numpy()
user_ids_train_all = train_all["user_id"].to_numpy()

# Train global
log(f"Training GLOBAL {GLOBAL_ITERS} iters on {X_train_all.shape[0]} rows...")
t_g = time.perf_counter()
global_model = CatBoostRegressor(iterations=GLOBAL_ITERS, depth=8, learning_rate=0.05, l2_leaf_reg=3, random_seed=SEED, loss_function="RMSE", verbose=100, allow_writing_files=False)
global_model.fit(X_train_all, y_train_z_all, verbose=100)
log(f"GLOBAL done in {time.perf_counter()-t_g:.3f}s")
y_pred_test_global = global_model.predict(X_test_all)
y_pred_train_global = global_model.predict(X_train_all)
rmsle_global = np.sqrt(mean_squared_error(y_test_z_all, y_pred_test_global))
log(f"GLOBAL RMSLE full 250k: {rmsle_global:.5f}")

# Prepare mappings for fast per-user lookup
log("Building user->train idx map...")
user_to_train_idx = defaultdict(list)
for idx, uid in enumerate(user_ids_train_all):
    user_to_train_idx[int(uid)].append(idx)
# Also need quick test idx
uid_to_test_idx = {int(uid): i for i, uid in enumerate(user_ids_test_all)}

# All user ids sorted for batching
all_uids = sorted(uid_to_test_idx.keys())
log(f"Total users for personalization: {len(all_uids)}")

# Checkpoint handling
preds_global = y_pred_test_global
preds_per = np.zeros_like(y_pred_test_global)
times = []
n_skipped = 0
n_done = 0

# Try to resume from checkpoint if exists
ckpt_file = CKPT_DIR / "progress.json"
if ckpt_file.exists():
    try:
        ck = json.loads(ckpt_file.read_text())
        n_done = ck.get("n_done",0)
        n_skipped = ck.get("n_skipped",0)
        if Path(CKPT_DIR / "preds_per.npy").exists():
            preds_per = np.load(CKPT_DIR / "preds_per.npy")
        if Path(CKPT_DIR / "times.npy").exists():
            times = np.load(CKPT_DIR / "times.npy").tolist()
        log(f"RESUME from {n_done} done, {n_skipped} skipped")
    except Exception as e:
        log(f"ckpt resume failed {e}")

# Process in batches
total_users = len(all_uids)
t_per_total = time.perf_counter()
for batch_start in range(n_done, total_users, BATCH_SIZE):
    batch_end = min(batch_start+BATCH_SIZE, total_users)
    batch_uids = all_uids[batch_start:batch_end]
    log(f"--- Batch {batch_start//BATCH_SIZE+1} / {(total_users+BATCH_SIZE-1)//BATCH_SIZE} users {batch_start}:{batch_end} ---")
    t_batch = time.perf_counter()
    for pos_in_batch, uid in enumerate(batch_uids):
        global_pos = batch_start + pos_in_batch
        test_idx = uid_to_test_idx[uid]
        train_idx = user_to_train_idx.get(uid, [])
        if len(train_idx)==0:
            preds_per[test_idx] = preds_global[test_idx]
            times.append(0.0)
            n_skipped+=1
            continue
        X_tr = X_train_all[train_idx]
        y_tr = y_train_z_all[train_idx]
        if np.all(y_tr == y_tr[0]):
            preds_per[test_idx] = preds_global[test_idx]
            times.append(0.0005)
            n_skipped+=1
            continue
        X_te = X_test_all[test_idx].reshape(1,-1)
        t_s = time.perf_counter()
        try:
            m = CatBoostRegressor(iterations=ITER_PER_USER, depth=8, learning_rate=0.05, l2_leaf_reg=3, random_seed=SEED, loss_function="RMSE", verbose=False, allow_writing_files=False)
            m.fit(X_tr, y_tr, init_model=global_model, verbose=False)
            pred = m.predict(X_te)[0]
        except Exception as e:
            pred = preds_global[test_idx]
            n_skipped+=1
        t_e = time.perf_counter()
        times.append(t_e - t_s)
        preds_per[test_idx] = pred
        if (global_pos+1) % 2000 == 0:
            # estimate RMSLE so far on done users
            done_mask = np.zeros(total_users, dtype=bool)
            # we have preds_per filled for done up to global_pos, but not yet for rest (zeros). So compute on done subset only
            # need to map done uids to test idxs
            # for quick estimate, compute rmsle on first global_pos+1 users
            test_idxs_done = [uid_to_test_idx[u] for u in all_uids[:global_pos+1]]
            rmsle_sofar = np.sqrt(mean_squared_error(y_test_z_all[test_idxs_done], preds_per[test_idxs_done]))
            avg_t = np.mean(times[-2000:]) if len(times)>=2000 else np.mean(times)
            log(f"  {global_pos+1}/{total_users} rmsle_sofar {rmsle_sofar:.5f} avg {avg_t*1000:.2f}ms skipped {n_skipped}")

    # checkpoint after batch
    np.save(CKPT_DIR / "preds_per.npy", preds_per)
    np.save(CKPT_DIR / "times.npy", np.array(times))
    with open(ckpt_file, "w") as f:
        json.dump({"n_done": batch_end, "n_skipped": n_skipped, "rmsle_global": float(rmsle_global)}, f)
    # also compute and log batch rmsle
    test_idxs_batch = [uid_to_test_idx[u] for u in batch_uids]
    # Actually we want cumulative
    test_idxs_done = [uid_to_test_idx[u] for u in all_uids[:batch_end]]
    rmsle_done = np.sqrt(mean_squared_error(y_test_z_all[test_idxs_done], preds_per[test_idxs_done]))
    t_batch_elapsed = time.perf_counter() - t_batch
    avg_batch = np.mean(times[-len(batch_uids):]) if times else 0
    log(f"BATCH {batch_start//BATCH_SIZE+1} done in {t_batch_elapsed:.1f}s | cumulative RMSLE {rmsle_done:.5f} | avg {avg_batch*1000:.2f}ms | total skipped {n_skipped}")

# Final
rmsle_per = np.sqrt(mean_squared_error(y_test_z_all, preds_per))
times_arr = np.array(times)
log(f"FINAL GLOBAL RMSLE {rmsle_global:.5f}")
log(f"FINAL PER-USER RMSLE {rmsle_per:.5f} diff {rmsle_per - rmsle_global:+.5f}")
log(f"Per-user timing mean {times_arr.mean()*1000:.3f}ms median {np.median(times_arr)*1000:.3f}ms p90 {np.percentile(times_arr,90)*1000:.3f}ms p95 {np.percentile(times_arr,95)*1000:.3f}ms max {times_arr.max()*1000:.3f}ms total {times_arr.sum():.1f}s")
log(f"Skipped identical {n_skipped}/{total_users} ({n_skipped/total_users*100:.1f}%)")
log(f"Total per-user time {time.perf_counter()-t_per_total:.1f}s total wall {time.perf_counter()-t0:.1f}s")

# Save report
report = {
    "global_rmsle": float(rmsle_global),
    "per_user_rmsle": float(rmsle_per),
    "diff": float(rmsle_per - rmsle_global),
    "global_iters": GLOBAL_ITERS,
    "per_user_iters": ITER_PER_USER,
    "n_train_per_user": 3,
    "total_users": total_users,
    "n_skipped": int(n_skipped),
    "mean_ms": float(times_arr.mean()*1000),
    "median_ms": float(np.median(times_arr)*1000),
    "p90_ms": float(np.percentile(times_arr,90)*1000),
    "p95_ms": float(np.percentile(times_arr,95)*1000),
    "p99_ms": float(np.percentile(times_arr,99)*1000),
    "total_per_user_s": float(times_arr.sum()),
    "wall_s": float(time.perf_counter()-t0),
}
with open(REPORT_PATH, "w") as f:
    json.dump(report, f, indent=2)
np.savez(PRED_PATH, preds_global=preds_global, preds_per=preds_per, y_test_z=y_test_z_all, user_ids=user_ids_test_all)
log(f"Saved {REPORT_PATH} and {PRED_PATH}")
