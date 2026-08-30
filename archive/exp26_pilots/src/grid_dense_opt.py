#!/usr/bin/env python3
import time, pathlib, json
import numpy as np, polars as pl
from datetime import date, timedelta
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from collections import defaultdict

SEED=42
FEATURES=141

# dense anchors as in per_user_full_n5_it25.py but for 1k pilot
ANCHORS_N3=[date(2025,12,3),date(2025,12,17),date(2025,12,31)]
ANCHORS_N5=[date(2025,12,17),date(2025,12,24),date(2025,12,31),date(2026,1,7),date(2026,1,14)]
ANCHORS_N10=[date(2025,12,3)+timedelta(days=i*4) for i in range(10)] # 4d step approx

t0=time.perf_counter()
def log(m): print(f"[{time.perf_counter()-t0:.1f}s] {m}", flush=True)

# load raw data for building features
log("Loading train.parquet for dense building ...")
data=pl.read_parquet("data/train.parquet").with_columns(pl.col("event_date").cast(pl.Date), pl.col("gmv").cast(pl.Float64))
# pick 1k users
all_uids=data["user_id"].unique().sort().to_list()[:1000]
log(f"Pilot {len(all_uids)} users")

# Use existing feature builders logic: reuse build_dataset_v3? Simpler reuse per_user_full_n5_it25 helpers but copy here
VALUE_COLS=["gmv","searches","to_ord","to_cart"]
WINDOWS=[(7,"7d"),(14,"14d"),(30,"30d"),(60,"60d"),(90,"90d")]
RECENCY_NONE=999.0
HORIZON=30

def win_mask(a,d): return pl.col("event_date").is_between(a-timedelta(days=d-1), a)
def wsum(a,d,col,name): return (pl.when(pl.col("event_date").is_between(a-timedelta(days=d-1), a)).then(pl.col(col)).otherwise(0.0).sum().alias(name))
def base_exprs(a):
    exprs=[]
    for w_days,w_name in WINDOWS:
        mask=pl.col("event_date").is_between(a-timedelta(days=w_days-1), a)
        for col in VALUE_COLS:
            c=pl.col(col)
            exprs.append(pl.when(mask).then(c).otherwise(0.0).sum().alias(f"{col}_sum_{w_name}"))
            exprs.append(pl.when(mask).then(c).otherwise(None).max().alias(f"{col}_max_{w_name}"))
            exprs.append(pl.when(mask).then(c).otherwise(None).mean().alias(f"{col}_mean_{w_name}"))
    m30=pl.col("event_date").is_between(a-timedelta(days=29), a)
    last_ord=pl.col("event_date").filter(pl.col("to_ord")>0).max()
    last_srch=pl.col("event_date").filter(pl.col("searches")>0).max()
    rec_ord=pl.when(last_ord.is_null()).then(RECENCY_NONE).otherwise((a-last_ord).dt.total_days())
    rec_srch=pl.when(last_srch.is_null()).then(RECENCY_NONE).otherwise((a-last_srch).dt.total_days())
    exprs+=[
        pl.when(m30 & ((pl.col("gmv")>0)|(pl.col("searches")>0))).then(1).otherwise(0).sum().cast(pl.Float64).alias("active_days_30d"),
        rec_ord.cast(pl.Float64).alias("recency_to_ord_days"),
        rec_srch.cast(pl.Float64).alias("recency_searches_days"),
        (a-pl.col("event_date").min()).dt.total_days().cast(pl.Float64).alias("tenure_days"),
        pl.when(pl.col("to_ord")>0).then(1).otherwise(0).sum().cast(pl.Float64).alias("order_days_total"),
        pl.len().cast(pl.Float64).alias("row_days_total"),
    ]
    s90=pl.col("searches_sum_90d"); tc=pl.col("to_cart_sum_90d"); to=pl.col("to_ord_sum_90d"); g9=pl.col("gmv_sum_90d")
    conv=[(to/s90.clip(lower_bound=1)).alias("conv_to_ord_per_search_90d"),(tc/s90.clip(lower_bound=1)).alias("conv_to_cart_per_search_90d"),(to/tc.clip(lower_bound=1)).alias("conv_to_ord_per_cart_90d"),(g9/to.clip(lower_bound=1)).alias("gmv_per_order_90d")]
    return exprs, conv

def ext_agg(a):
    m14=pl.col("event_date").is_between(a-timedelta(days=13), a)
    return [wsum(a,14,"searches","x_searches_sum_14d"),wsum(a,14,"to_cart","x_cart_sum_14d"),wsum(a,14,"to_ord","x_ord_sum_14d"),wsum(a,14,"gmv","x_gmv_sum_14d"),wsum(a,30,"gmv_search","x_gmv_search_sum_30d"),wsum(a,30,"gmv_cat","x_gmv_cat_sum_30d"),wsum(a,30,"search_to_ord","x_search_to_ord_30d"),wsum(a,30,"cat_to_ord","x_cat_to_ord_30d"),wsum(a,30,"cat_to_cart","x_cat_to_cart_30d"),(pl.when(m14&(pl.col("to_cart")>0)&(pl.col("to_ord")==0)).then(1).otherwise(0).sum().cast(pl.Float64).alias("x_cart_no_ord_days_14d")),(pl.when(pl.col("event_date").is_between(a-timedelta(days=29),a)&(pl.col("searches")==0)&(pl.col("to_cart")==0)&(pl.col("to_ord")==0)&(pl.col("gmv")==0)).then(1).otherwise(0).sum().cast(pl.Float64).alias("x_visit_only_days_30d")),(pl.when(m14&(pl.col("searches")>0)).then(1).otherwise(0).sum().cast(pl.Float64).alias("x_search_days_14d"))]

def ext_post():
    def ratio(n,d,na): return pl.when(pl.col(d)>0).then(pl.col(n)/pl.col(d)).otherwise(None).alias(na)
    return [ratio("x_gmv_search_sum_30d","gmv_sum_30d","x_share_gmv_search_30d"),ratio("x_gmv_cat_sum_30d","gmv_sum_30d","x_share_gmv_cat_30d"),ratio("x_search_to_ord_30d","searches_sum_14d","x_conv_s2o_14d"),ratio("x_cat_to_ord_30d","x_cat_to_cart_30d","x_conv_c2o_30d"),ratio("x_gmv_sum_14d","gmv_sum_30d","x_gmv_share_14_of_30"),(pl.when(pl.col("x_ord_sum_14d")==0).then(1.0).otherwise(0.0).alias("x_intent_no_ord_14d")),(pl.when(pl.col("to_ord_sum_30d")>0).then(pl.col("gmv_sum_30d")/pl.col("to_ord_sum_30d")).otherwise(None).alias("x_aov_30d")),((pl.col("recency_to_ord_days").clip(upper_bound=365)/((pl.col("tenure_days")+1.0)/(pl.col("order_days_total")+1.0)).clip(lower_bound=1.0)).clip(upper_bound=60.0).alias("x_due_ratio"))]

def build_anchor(a, uids):
    hist=data.filter(pl.col("event_date")<=a).filter(pl.col("user_id").is_in(uids))
    be, ce=base_exprs(a)
    feats=hist.group_by("user_id").agg([*be, *ext_agg(a)]).with_columns(ce).with_columns(ext_post())
    idx=pl.DataFrame({"user_id":uids})
    out=idx.join(feats, on="user_id", how="left")
    t=data.filter(pl.col("event_date").is_between(a+timedelta(days=1), a+timedelta(days=HORIZON))).filter(pl.col("user_id").is_in(uids)).group_by("user_id").agg(pl.col("gmv").sum().alias("target"))
    out=out.join(t, on="user_id", how="left").with_columns(pl.col("target").fill_null(0.0))
    feat_cols=[c for c in out.columns if c not in ("user_id","target")]
    out=out.with_columns([pl.col(c).cast(pl.Float64) for c in feat_cols]).with_columns([pl.col(c).fill_null(0.0) for c in feat_cols])
    # join btyd quickly from existing files for speed? For now zero out btyd cols - but we need btyd for 141f. Load from file for this anchor.
    # Try to load btyd for closest anchor
    # Map to fold name: for n3 anchors they are folds, for dense use nearest
    return out

# For 141f we need btyd, load from data/v2/features_bgnbd
import pathlib
BTYD_DIR=pathlib.Path("data/v2/features_bgnbd")
def attach_btyd(df, anchor):
    # find closest fold
    folds_map={date(2025,12,3):"fold_00",date(2025,12,17):"fold_01",date(2025,12,31):"fold_02",date(2026,1,14):"fold_03"}
    # for dense anchors, use nearest fold's btyd params? Actually btyd depends on anchor's RFM, so nearest is approx
    # For pilot, just use fold_02 btyd for all dense anchors beyond 02? Not accurate but pilot only for timing/ranking
    # Use 02 for simplicity
    b=pl.read_parquet(BTYD_DIR/"fold_02.parquet").select(["user_id","bgnbd_p_alive","bgnbd_en30","eb_lambda_n30","bgnbd_e_gmv30","eb_e_gmv30"])
    # fill others with 0 for missing cols
    for c in ["bgnbd_tx","bgnbd_en30","eb_lambda_n30","bgnbd_e_gmv30","eb_e_gmv30"]:
        if c not in b.columns:
            pass
    df=df.join(b, on="user_id", how="left")
    # fill remaining btyd cols expected 11? But we use 141f: need all btyd cols from manifest: 11? Actually 141f includes all btyd cols (11). For pilot we approximate with 5 available.
    # Fill nulls
    for c in ["bgnbd_p_alive","bgnbd_en30","eb_lambda_n30","bgnbd_e_gmv30","eb_e_gmv30","bgnbd_tx","bgnbd_T","bgnbd_n_occasions","bgnbd_mon_freq","bgnbd_mbar","gg_e_value"]:
        if c in df.columns:
            df=df.with_columns(pl.col(c).fill_null(0.0))
    return df

def prepare_for_anchors(anchors, uids):
    parts=[]
    for a in anchors:
        part=build_anchor(a, uids)
        # approximate btyd attach using fold_02
        part=attach_btyd(part, a)
        parts.append(part)
    df=pl.concat(parts)
    return df

# Build for n=3 and n=5 for 1k
for n, anchors in [("n3",ANCHORS_N3),("n5",ANCHORS_N5)]:
    log(f"Building {n} anchors {anchors}...")
    df=prepare_for_anchors(anchors, all_uids)
    log(f"{n} df {df.shape} cols {len([c for c in df.columns if c!='user_id' and c!='target'])}")
    # quick rmsle test with best hypers
    feature_cols=[c for c in df.columns if c not in ("user_id","target")]
    # For test we need to split train vs test anchor fold_03 is separate, but our dense building already includes train anchors only, test is fold_03 built separately via same method with anchor 2026-01-14
    # Build test fold 03
    test_df=build_anchor(date(2026,1,14), all_uids)
    test_df=attach_btyd(test_df, date(2026,1,14))
    # Align columns
    common_cols=[c for c in feature_cols if c in test_df.columns]
    log(f"Common {len(common_cols)}")
    # Convert to numpy
    X_train=df.select(common_cols).to_numpy().astype(np.float32)
    y_train=np.log1p(df["target"].to_numpy().astype(float))
    X_test=test_df.select(common_cols).to_numpy().astype(np.float32)
    y_test=np.log1p(test_df["target"].to_numpy().astype(float))
    u_train=df["user_id"].to_numpy()
    u_test=test_df["user_id"].to_numpy()
    # Train global
    gm=CatBoostRegressor(iterations=500, depth=8, learning_rate=0.05, l2_leaf_reg=3, random_seed=42, loss_function="RMSE", verbose=False, allow_writing_files=False)
    gm.fit(X_train, y_train, verbose=False)
    pred_global=gm.predict(X_test)
    rmsle_global=np.sqrt(mean_squared_error(y_test, pred_global))
    log(f"{n} GLOBAL {rmsle_global:.5f}")
    # per-user with best hypers d4 lr0.1 l2=1 it50 and d4 lr0.05 l2=1 it50
    for depth,lr,l2,it in [(4,0.1,1,50),(4,0.05,1,50),(4,0.1,1,30)]:
        # map
        user_to_train_idx=defaultdict(list)
        for idx, uid in enumerate(u_train):
            user_to_train_idx[int(uid)].append(idx)
        uid_to_test_idx={int(uid):i for i,uid in enumerate(u_test)}
        preds=np.zeros_like(pred_global)
        skipped=0
        times=[]
        for uid in all_uids:
            test_idx=uid_to_test_idx[int(uid)]
            train_idx=user_to_train_idx.get(int(uid), [])
            if not train_idx:
                preds[test_idx]=pred_global[test_idx]; skipped+=1; continue
            X_tr=X_train[train_idx]
            y_tr=y_train[train_idx]
            if np.all(y_tr==y_tr[0]):
                preds[test_idx]=pred_global[test_idx]; skipped+=1; continue
            X_te=X_test[test_idx].reshape(1,-1)
            ts=time.perf_counter()
            m=CatBoostRegressor(iterations=it, depth=depth, learning_rate=lr, l2_leaf_reg=l2, random_seed=42, loss_function="RMSE", verbose=False, allow_writing_files=False, thread_count=1)
            m.fit(X_tr, y_tr, init_model=gm, verbose=False)
            preds[test_idx]=m.predict(X_te)[0]
            times.append(time.perf_counter()-ts)
        rmsle=np.sqrt(mean_squared_error(y_test, preds))
        log(f"  d{depth} lr{lr} l2{l2} it{it} -> {rmsle:.5f} diff {rmsle-rmsle_global:+.5f} {np.mean(times)*1000:.1f}ms")
