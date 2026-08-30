#!/usr/bin/env python3
import time, pathlib
from datetime import date, timedelta
import numpy as np, polars as pl
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error

t0=time.perf_counter()
def log(m): print(f"[{time.perf_counter()-t0:.1f}s] {m}", flush=True)

DATA=pl.read_parquet("data/train.parquet").with_columns(pl.col("event_date").cast(pl.Date), pl.col("gmv").cast(pl.Float64))
# pilot sizes
for SAMPLE_N in [1000, 10000]:
    log(f"\n=== SAMPLE {SAMPLE_N} ===")
    all_uids=DATA["user_id"].unique().sort().to_list()[:SAMPLE_N]
    ANCHORS_TRAIN=[date(2025,12,3),date(2025,12,17),date(2025,12,31)]
    ANCHOR_TEST=date(2026,1,14)
    L=30
    def build_lag(anchor, uids):
        # grid
        dates=pl.date_range(anchor-timedelta(days=L-1), anchor, "1d", eager=True)
        grid=pl.DataFrame({"user_id":uids}).join(pl.DataFrame({"event_date":dates}), how="cross")
        # gmv window
        win=DATA.filter(pl.col("event_date").is_between(anchor-timedelta(days=L-1), anchor)).filter(pl.col("user_id").is_in(uids)).select(["user_id","event_date","gmv"])
        # join
        df=grid.join(win, on=["user_id","event_date"], how="left").with_columns(pl.col("gmv").fill_null(0.0))
        # sort
        df=df.sort(["user_id","event_date"])
        # group to list
        # create lag columns via pivot: we can use group_by and then explode
        # Use agg list
        grouped=df.group_by("user_id").agg(pl.col("gmv").alias("seq"))
        # seq is list of 30 values oldest->newest, we need lag1 newest
        # convert to numpy
        # create matrix
        mat=np.vstack([np.array(s[::-1], dtype=np.float32) for s in grouped["seq"].to_list()]) # lag1 is anchor
        # log1p
        mat=np.log1p(mat)
        # target
        tgt=DATA.filter(pl.col("event_date").is_between(anchor+timedelta(days=1), anchor+timedelta(days=30))).filter(pl.col("user_id").is_in(uids)).group_by("user_id").agg(pl.col("gmv").sum().alias("target"))
        idx=pl.DataFrame({"user_id":uids})
        tgt_df=idx.join(tgt, on="user_id", how="left").with_columns(pl.col("target").fill_null(0.0)).sort("user_id")
        y=np.log1p(tgt_df["target"].to_numpy().astype(float))
        # ensure order matches grouped order (sorted user_id)
        # grouped is sorted by user_id because group_by sort? Need to sort
        grouped_sorted=grouped.sort("user_id")
        # mat already in sorted order? We used grouped order which may not be sorted, but we sorted after?
        # Let's ensure mat order matches sorted user_id
        # Rebuild mat in sorted order
        # Instead create dict
        user_to_seq={uid: seq for uid, seq in zip(grouped["user_id"].to_list(), grouped["seq"].to_list())}
        sorted_uids=sorted(uids)
        mat_sorted=np.vstack([np.log1p(np.array(user_to_seq[uid][::-1], dtype=np.float32)) for uid in sorted_uids])
        return mat_sorted, y, sorted_uids

    # Build train
    mats=[]; ys=[]
    for a in ANCHORS_TRAIN:
        m,y,_=build_lag(a, all_uids)
        mats.append(m); ys.append(y)
        log(f" anchor {a} mat {m.shape}")
    X_train=np.vstack(mats)
    y_train=np.concatenate(ys)
    log(f"X_train {X_train.shape}")
    # Test
    X_test,y_test,_=build_lag(ANCHOR_TEST, all_uids)
    log(f"X_test {X_test.shape}")
    # Train CatBoost
    log("Training CatBoost lag 500it ...")
    gm=CatBoostRegressor(iterations=500, depth=8, learning_rate=0.05, l2_leaf_reg=3, random_seed=42, loss_function="RMSE", verbose=False, allow_writing_files=False)
    gm.fit(X_train, y_train, verbose=False)
    pred=gm.predict(X_test)
    rmsle=np.sqrt(mean_squared_error(y_test, pred))
    log(f"Lag {L}d RMSLE {rmsle:.5f} (vs agg 1.671)")
    # also try with aggregates baseline on same sample for fair compare: use 90f?
    # quick compare: train aggregates 90f on same sample via per_user_n7 logic? Skip for now, just show lag
    # Try also 60d lags
    # For 10k, we will also test 30d but already
