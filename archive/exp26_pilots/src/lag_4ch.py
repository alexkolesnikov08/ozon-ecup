#!/usr/bin/env python3
import time
from datetime import date, timedelta
import numpy as np, polars as pl
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
t0=time.perf_counter()
def log(m): print(f"[{time.perf_counter()-t0:.1f}s] {m}", flush=True)
DATA=pl.read_parquet("data/train.parquet").with_columns(pl.col("event_date").cast(pl.Date), pl.col("gmv").cast(pl.Float64), pl.col("searches").cast(pl.Float64), pl.col("to_ord").cast(pl.Float64), pl.col("to_cart").cast(pl.Float64))
SAMPLE_N=10000
all_uids=DATA["user_id"].unique().sort().to_list()[:SAMPLE_N]
ANCHORS_TRAIN=[date(2025,12,3),date(2025,12,17),date(2025,12,31)]
ANCHOR_TEST=date(2026,1,14)
L=30
def build_4ch(anchor, uids):
    dates=pl.date_range(anchor-timedelta(days=L-1), anchor, "1d", eager=True)
    grid=pl.DataFrame({"user_id":uids}).join(pl.DataFrame({"event_date":dates}), how="cross")
    win=DATA.filter(pl.col("event_date").is_between(anchor-timedelta(days=L-1), anchor)).filter(pl.col("user_id").is_in(uids)).select(["user_id","event_date","gmv","searches","to_ord","to_cart"])
    df=grid.join(win, on=["user_id","event_date"], how="left").with_columns([pl.col("gmv").fill_null(0.0),pl.col("searches").fill_null(0.0),pl.col("to_ord").fill_null(0.0),pl.col("to_cart").fill_null(0.0)]).sort(["user_id","event_date"])
    # group
    grouped=df.group_by("user_id").agg([pl.col("gmv").alias("gmv_seq"),pl.col("searches").alias("s_seq"),pl.col("to_ord").alias("o_seq"),pl.col("to_cart").alias("c_seq")])
    user_to_seq={uid: (g,s,o,c) for uid,g,s,o,c in zip(grouped["user_id"].to_list(), grouped["gmv_seq"].to_list(), grouped["s_seq"].to_list(), grouped["o_seq"].to_list(), grouped["c_seq"].to_list())}
    # Actually need to handle 4 seq correctly
    sorted_uids=sorted(uids)
    mats=[]
    for uid in sorted_uids:
        rec=grouped.filter(pl.col("user_id")==uid)
        g=np.array(rec["gmv_seq"][0][::-1], dtype=np.float32)
        s=np.array(rec["s_seq"][0][::-1], dtype=np.float32)
        o=np.array(rec["o_seq"][0][::-1], dtype=np.float32)
        c=np.array(rec["c_seq"][0][::-1], dtype=np.float32)
        # log1p for g, raw for others? Use log1p for all
        mats.append(np.concatenate([np.log1p(g), np.log1p(s), np.log1p(o), np.log1p(c)]))
    mat=np.vstack(mats)
    tgt=DATA.filter(pl.col("event_date").is_between(anchor+timedelta(days=1), anchor+timedelta(days=30))).filter(pl.col("user_id").is_in(uids)).group_by("user_id").agg(pl.col("gmv").sum().alias("target"))
    idx=pl.DataFrame({"user_id":sorted_uids})
    tgt_df=idx.join(tgt, on="user_id", how="left").with_columns(pl.col("target").fill_null(0.0))
    y=np.log1p(tgt_df["target"].to_numpy().astype(float))
    return mat, y

mats=[]; ys=[]
for a in ANCHORS_TRAIN:
    m,y=build_4ch(a, all_uids)
    mats.append(m); ys.append(y)
    log(f"anchor {a} {m.shape}")
X_train=np.vstack(mats); y_train=np.concatenate(ys)
log(f"X_train {X_train.shape}")
X_test,y_test=build_4ch(ANCHOR_TEST, all_uids)
log(f"X_test {X_test.shape}")
gm=CatBoostRegressor(iterations=500, depth=8, learning_rate=0.05, l2_leaf_reg=3, random_seed=42, loss_function="RMSE", verbose=False, allow_writing_files=False)
gm.fit(X_train, y_train, verbose=False)
pred=gm.predict(X_test)
rmsle=np.sqrt(mean_squared_error(y_test, pred))
log(f"4ch 30d*4 RMSLE {rmsle:.5f}")

# Also try 60d*4 =240 feats quickly
L=60
def build_4ch_60(anchor, uids):
    dates=pl.date_range(anchor-timedelta(days=L-1), anchor, "1d", eager=True)
    grid=pl.DataFrame({"user_id":uids}).join(pl.DataFrame({"event_date":dates}), how="cross")
    win=DATA.filter(pl.col("event_date").is_between(anchor-timedelta(days=L-1), anchor)).filter(pl.col("user_id").is_in(uids)).select(["user_id","event_date","gmv","searches","to_ord","to_cart"])
    df=grid.join(win, on=["user_id","event_date"], how="left").with_columns([pl.col("gmv").fill_null(0.0),pl.col("searches").fill_null(0.0),pl.col("to_ord").fill_null(0.0),pl.col("to_cart").fill_null(0.0)]).sort(["user_id","event_date"])
    grouped=df.group_by("user_id").agg([pl.col("gmv").alias("gmv_seq"),pl.col("searches").alias("s_seq"),pl.col("to_ord").alias("o_seq"),pl.col("to_cart").alias("c_seq")])
    sorted_uids=sorted(uids)
    mats=[]
    for uid in sorted_uids:
        rec=grouped.filter(pl.col("user_id")==uid)
        g=np.array(rec["gmv_seq"][0][::-1], dtype=np.float32)
        s=np.array(rec["s_seq"][0][::-1], dtype=np.float32)
        o=np.array(rec["o_seq"][0][::-1], dtype=np.float32)
        c=np.array(rec["c_seq"][0][::-1], dtype=np.float32)
        mats.append(np.concatenate([np.log1p(g), np.log1p(s), np.log1p(o), np.log1p(c)]))
    mat=np.vstack(mats)
    tgt=DATA.filter(pl.col("event_date").is_between(anchor+timedelta(days=1), anchor+timedelta(days=30))).filter(pl.col("user_id").is_in(uids)).group_by("user_id").agg(pl.col("gmv").sum().alias("target"))
    idx=pl.DataFrame({"user_id":sorted_uids})
    tgt_df=idx.join(tgt, on="user_id", how="left").with_columns(pl.col("target").fill_null(0.0))
    y=np.log1p(tgt_df["target"].to_numpy().astype(float))
    return mat, y

mats=[]; ys=[]
for a in ANCHORS_TRAIN:
    m,y=build_4ch_60(a, all_uids)
    mats.append(m); ys.append(y)
X_train=np.vstack(mats); y_train=np.concatenate(ys)
log(f"60d X_train {X_train.shape}")
X_test,y_test=build_4ch_60(ANCHOR_TEST, all_uids)
gm.fit(X_train, y_train, verbose=False)
pred=gm.predict(X_test)
rmsle=np.sqrt(mean_squared_error(y_test, pred))
log(f"4ch 60d*4 RMSLE {rmsle:.5f}")
