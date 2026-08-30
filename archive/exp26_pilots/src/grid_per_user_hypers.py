#!/usr/bin/env python3
"""
Grid search per-user hyperparams on pilot 1k
Base: CatBoost 2000 it depth8 lr0.05 l2=3 141f (ext+btyd) -> per-user init_model
Sweep: n, it, depth, lr, l2
"""
import time, json, itertools, pathlib
import numpy as np
import polars as pl
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from collections import defaultdict

SEED=42
SAMPLE_N=1000  # pilot
GLOBAL_ITERS=500  # keep 500 for speed, 2000 for final validation in separate run
FOLDS_TRAIN=["fold_00","fold_01","fold_02"]
FOLD_TEST="fold_03"
FEAT_DIR=pathlib.Path("data/v2/features_ext")
BTYD_DIR=pathlib.Path("data/v2/features_bgnbd")
OUT=pathlib.Path("reports/grid_per_user_hypers.json")

# --- load 1k pilot data subset for grid ---
# Use first 1k user_ids sorted for reproducibility
t0=time.perf_counter()
def log(m): print(f"[{time.perf_counter()-t0:.1f}s] {m}", flush=True)

log("Loading folds for pilot 1k ...")
# load ext + btyd to get 141f
def load_fold(name):
    feats=pl.scan_parquet(str(FEAT_DIR/name/"batch_*.parquet")).collect()
    btyd=pl.read_parquet(BTYD_DIR/f"{name}.parquet")
    df=feats.join(btyd, on=["anchor_date","user_id"], how="left")
    # fill btyd nulls (same as train_stack_v2)
    for c in [c for c in btyd.columns if c not in ("anchor_date","user_id")]:
        if c in ("bgnbd_tx","bgnbd_en30","eb_lambda_n30","bgnbd_e_gmv30","eb_e_gmv30"):
            df=df.with_columns(pl.col(c).fill_null(0.0))
        else:
            df=df.with_columns(pl.col(c).fill_null(-1.0))
    return df

folds={f: load_fold(f) for f in FOLDS_TRAIN+[FOLD_TEST]}
# pick 1k uids from test fold
all_uids_test=folds[FOLD_TEST]["user_id"].unique().sort().to_list()[:SAMPLE_N]
# also need maybe 1k from train? Actually we use same 1k uids across folds
pilot_uids=all_uids_test
log(f"Pilot uids {len(pilot_uids)} sample {pilot_uids[:5]}")

# filter folds to pilot uids
for k in folds:
    folds[k]=folds[k].filter(pl.col("user_id").is_in(pilot_uids))
    log(f"{k} filtered {folds[k].shape}")

feature_cols=[c for c in folds["fold_00"].columns if c not in ("anchor_date","user_id","target")]
log(f"Features {len(feature_cols)}: {feature_cols[:5]} ...")

def to_np(df, cols):
    X=df.select(cols).to_numpy().astype(np.float32)
    y=np.log1p(df["target"].to_numpy().astype(float))
    u=df["user_id"].to_numpy()
    return X,y,u

# prepare train_all for global
train_all=pl.concat([folds[f] for f in FOLDS_TRAIN])
X_train_all, y_train_all, u_train_all = to_np(train_all, feature_cols)
X_test, y_test, u_test = to_np(folds[FOLD_TEST], feature_cols)
# mappings for per-user
user_to_train_idx=defaultdict(list)
for idx, uid in enumerate(u_train_all):
    user_to_train_idx[int(uid)].append(idx)
uid_to_test_idx={int(uid):i for i,uid in enumerate(u_test)}

log(f"Train {X_train_all.shape}, Test {X_test.shape}")

# train global once (500it)
log(f"Training GLOBAL {GLOBAL_ITERS} it depth8 lr0.05 l2=3 on {X_train_all.shape[0]} rows ...")
gm=CatBoostRegressor(iterations=GLOBAL_ITERS, depth=8, learning_rate=0.05, l2_leaf_reg=3, random_seed=SEED, loss_function="RMSE", verbose=False, allow_writing_files=False)
gm.fit(X_train_all, y_train_all, verbose=False)
pred_global=gm.predict(X_test)
rmsle_global=np.sqrt(mean_squared_error(y_test, pred_global))
log(f"GLOBAL RMSLE pilot 1k: {rmsle_global:.5f}")

# define grid
# Keep n=3 for depth/lr/l2 sweep first, then expand n
grid_depth=[4,6,8]
grid_lr=[0.03,0.05,0.10]
grid_l2=[1,3,9]
grid_it=[10,20,30,50]
# Also n sweep with best depth/lr/l2
grid_n=[3,5,10]  # 5 and 10 need extra anchors with 7d step? For now use 3 base n=3 only; for n=5/10 we need to build n=5 data via same 3 folds + extra? For pilot we simulate n as using only n anchors among the 3? Actually n=3 max for current 3 folds. So for n=5 we need dense anchors. For now sweep n=3 only and later dense grid.
# So main grid: depth x lr x l2 x it for n=3

configs=list(itertools.product(grid_depth, grid_lr, grid_l2, grid_it))
log(f"Grid size {len(configs)} for n=3")

results=[]
best=None
for idx,(depth,lr,l2,it) in enumerate(configs):
    t_s=time.perf_counter()
    preds=np.zeros_like(pred_global)
    skipped=0
    times=[]
    for uid in pilot_uids:
        test_idx=uid_to_test_idx[uid]
        train_idx=user_to_train_idx.get(uid, [])
        if not train_idx:
            preds[test_idx]=pred_global[test_idx]; skipped+=1; continue
        X_tr=X_train_all[train_idx]
        y_tr=y_train_all[train_idx]
        if np.all(y_tr==y_tr[0]):
            preds[test_idx]=pred_global[test_idx]; skipped+=1; continue
        X_te=X_test[test_idx].reshape(1,-1)
        ts=time.perf_counter()
        try:
            m=CatBoostRegressor(iterations=it, depth=depth, learning_rate=lr, l2_leaf_reg=l2, random_seed=SEED, loss_function="RMSE", verbose=False, allow_writing_files=False, thread_count=1)
            m.fit(X_tr, y_tr, init_model=gm, verbose=False)
            preds[test_idx]=m.predict(X_te)[0]
        except Exception as e:
            preds[test_idx]=pred_global[test_idx]; skipped+=1
        times.append(time.perf_counter()-ts)
    rmsle=np.sqrt(mean_squared_error(y_test, preds))
    mean_ms=np.mean(times)*1000 if times else 0
    diff=rmsle-rmsle_global
    row={"depth":depth,"lr":lr,"l2":l2,"it":it,"n":3,"rmsle":float(rmsle),"diff":float(diff),"mean_ms":float(mean_ms),"skipped":int(skipped)}
    results.append(row)
    tag="*" if best is None or rmsle<best else " "
    if best is None or rmsle<best:
        best=rmsle
    log(f"{tag} [{idx+1}/{len(configs)}] d{depth} lr{lr} l2{l2} it{it:2d} -> {rmsle:.5f} diff {diff:+.5f} {mean_ms:.1f}ms")

# sort
results_sorted=sorted(results, key=lambda x: x["rmsle"])
log("TOP 10")
for r in results_sorted[:10]:
    log(f"  {r['rmsle']:.5f} diff {r['diff']:+.5f} d{r['depth']} lr{r['lr']} l2{r['l2']} it{r['it']} ms{r['mean_ms']:.1f}")

# also n sweep with best hyperparams
best_cfg=results_sorted[0]
log(f"BEST cfg for n=3: {best_cfg}")
# Save
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({"global": {"rmsle":float(rmsle_global), "iters":GLOBAL_ITERS, "depth":8,"lr":0.05,"l2":3, "n":3, "features":len(feature_cols)}, "grid":results_sorted, "best":best_cfg}, indent=2))
log(f"Saved {OUT}")
