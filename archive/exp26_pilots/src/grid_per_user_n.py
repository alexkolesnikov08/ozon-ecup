#!/usr/bin/env python3
"""
Grid n=3/5/10 with best hypers depth4 lr0.1 l2=1 it30/50 on pilot 1k
Uses 141f global 500, same as previous
Also test 5k pilot for validation
"""
import time, json, pathlib
import numpy as np, polars as pl
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from collections import defaultdict

SEED=42
SAMPLE_N=1000
FOLDS_TRAIN=["fold_00","fold_01","fold_02"]
FOLD_TEST="fold_03"
FEAT_DIR=pathlib.Path("data/v2/features_ext")
BTYD_DIR=pathlib.Path("data/v2/features_bgnbd")

t0=time.perf_counter()
def log(m): print(f"[{time.perf_counter()-t0:.1f}s] {m}", flush=True)

def load_fold(name):
    feats=pl.scan_parquet(str(FEAT_DIR/name/"batch_*.parquet")).collect()
    btyd=pl.read_parquet(BTYD_DIR/f"{name}.parquet")
    df=feats.join(btyd, on=["anchor_date","user_id"], how="left")
    for c in [c for c in btyd.columns if c not in ("anchor_date","user_id")]:
        if c in ("bgnbd_tx","bgnbd_en30","eb_lambda_n30","bgnbd_e_gmv30","eb_e_gmv30"):
            df=df.with_columns(pl.col(c).fill_null(0.0))
        else:
            df=df.with_columns(pl.col(c).fill_null(-1.0))
    return df

# For n=5/10 we need dense anchors: build on fly from train.parquet
# Simpler: we emulate n=5 by taking 5 anchors: 00,01,02 plus two intermediate 7d steps between them
# For pilot we will build dense features via src/per_user_full_n5_it25.py logic but quickly for 1k users only
# Instead we can reuse earlier per_user_dense_pilot json for reference and just validate n=3 hypers here
# So this script validates best hypers depth4 lr0.1 l2=1 it50 on 5k pilot and estimates 32c wall

# Load 1k then 5k
for SAMPLE_N in [1000,5000]:
    log(f"\n=== SAMPLE_N={SAMPLE_N} ===")
    folds={f: load_fold(f) for f in FOLDS_TRAIN+[FOLD_TEST]}
    all_uids_test=folds[FOLD_TEST]["user_id"].unique().sort().to_list()[:SAMPLE_N]
    pilot_uids=all_uids_test
    for k in folds:
        folds[k]=folds[k].filter(pl.col("user_id").is_in(pilot_uids))
    feature_cols=[c for c in folds["fold_00"].columns if c not in ("anchor_date","user_id","target")]
    def to_np(df, cols):
        X=df.select(cols).to_numpy().astype(np.float32)
        y=np.log1p(df["target"].to_numpy().astype(float))
        u=df["user_id"].to_numpy()
        return X,y,u
    train_all=pl.concat([folds[f] for f in FOLDS_TRAIN])
    X_train_all, y_train_all, u_train_all = to_np(train_all, feature_cols)
    X_test, y_test, u_test = to_np(folds[FOLD_TEST], feature_cols)
    user_to_train_idx=defaultdict(list)
    for idx, uid in enumerate(u_train_all):
        user_to_train_idx[int(uid)].append(idx)
    uid_to_test_idx={int(uid):i for i,uid in enumerate(u_test)}
    log(f"Train {X_train_all.shape}, Test {X_test.shape}")
    gm=CatBoostRegressor(iterations=500, depth=8, learning_rate=0.05, l2_leaf_reg=3, random_seed=SEED, loss_function="RMSE", verbose=False, allow_writing_files=False)
    gm.fit(X_train_all, y_train_all, verbose=False)
    pred_global=gm.predict(X_test)
    rmsle_global=np.sqrt(mean_squared_error(y_test, pred_global))
    log(f"GLOBAL {rmsle_global:.5f}")

    # test few hypers
    cfgs=[(4,0.1,1,50),(4,0.1,1,30),(4,0.05,1,50),(6,0.05,3,20)]
    for depth,lr,l2,it in cfgs:
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
            except:
                preds[test_idx]=pred_global[test_idx]; skipped+=1
            times.append(time.perf_counter()-ts)
        rmsle=np.sqrt(mean_squared_error(y_test, preds))
        mean_ms=np.mean(times)*1000 if times else 0
        wall_32c_ms = mean_ms*250000/32/1000  # seconds
        log(f"  d{depth} lr{lr} l2{l2} it{it} -> {rmsle:.5f} diff {rmsle-rmsle_global:+.5f} {mean_ms:.1f}ms wall32c {wall_32c_ms/60:.1f}min")
