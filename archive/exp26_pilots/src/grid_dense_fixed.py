#!/usr/bin/env python3
import time, pathlib
import numpy as np, polars as pl
from datetime import date, timedelta
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from collections import defaultdict

SEED=42
t0=time.perf_counter()
def log(m): print(f"[{time.perf_counter()-t0:.1f}s] {m}", flush=True)

# anchors честные до fold_03 2026-01-14
ANCHORS_N3=[date(2025,12,3),date(2025,12,17),date(2025,12,31)]
# n5: 7д шаг, 5 якорей до 14.01: 10.12,17.12,24.12,31.12,07.01
ANCHORS_N5=[date(2025,12,10),date(2025,12,17),date(2025,12,24),date(2025,12,31),date(2026,1,7)]
# n10: 4д шаг, 10 якорей 08.12-13.01? 08.12,12.12,16.12,20.12,24.12,28.12,01.01,05.01,09.01,13.01 but end before 14.01 => 07.01 is last? Use 07.01 as last
ANCHORS_N10=[date(2025,12,10)+timedelta(days=i*4) for i in range(10)] # 10.12 ... 15.01? Actually 10.12+36d=15.01 >07.01, need cap
ANCHORS_N10=[d for d in ANCHORS_N10 if d < date(2026,1,14)]
# adjust to 10 items: 05.12,09.12,13.12,17.12,21.12,25.12,29.12,02.01,06.01,10.01? Simpler: use 4d from 05.12
ANCHORS_N10=[date(2025,12,5)+timedelta(days=i*4) for i in range(10)]
ANCHORS_N10=[d for d in ANCHORS_N10 if d < date(2026,1,14)]
log(f"N10 anchors {ANCHORS_N10}")

BTYD_DIR=pathlib.Path("data/v2/features_bgnbd")
DATA=pl.read_parquet("data/train.parquet").with_columns(pl.col("event_date").cast(pl.Date), pl.col("gmv").cast(pl.Float64))
all_uids=DATA["user_id"].unique().sort().to_list()[:1000]
log(f"Pilot 1k uids {len(all_uids)}")

# helpers from grid_dense_opt - same base/ext
VALUE_COLS=["gmv","searches","to_ord","to_cart"]
WINDOWS=[(7,"7d"),(14,"14d"),(30,"30d"),(60,"60d"),(90,"90d")]
RECENCY_NONE=999.0
HORIZON=30
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
    def wsum(a,d,col,name): return (pl.when(pl.col("event_date").is_between(a-timedelta(days=d-1), a)).then(pl.col(col)).otherwise(0.0).sum().alias(name))
    return [wsum(a,14,"searches","x_searches_sum_14d"),wsum(a,14,"to_cart","x_cart_sum_14d"),wsum(a,14,"to_ord","x_ord_sum_14d"),wsum(a,14,"gmv","x_gmv_sum_14d"),wsum(a,30,"gmv_search","x_gmv_search_sum_30d"),wsum(a,30,"gmv_cat","x_gmv_cat_sum_30d"),wsum(a,30,"search_to_ord","x_search_to_ord_30d"),wsum(a,30,"cat_to_ord","x_cat_to_ord_30d"),wsum(a,30,"cat_to_cart","x_cat_to_cart_30d"),(pl.when(m14&(pl.col("to_cart")>0)&(pl.col("to_ord")==0)).then(1).otherwise(0).sum().cast(pl.Float64).alias("x_cart_no_ord_days_14d")),(pl.when(pl.col("event_date").is_between(a-timedelta(days=29),a)&(pl.col("searches")==0)&(pl.col("to_cart")==0)&(pl.col("to_ord")==0)&(pl.col("gmv")==0)).then(1).otherwise(0).sum().cast(pl.Float64).alias("x_visit_only_days_30d")),(pl.when(m14&(pl.col("searches")>0)).then(1).otherwise(0).sum().cast(pl.Float64).alias("x_search_days_14d"))]

def ext_post():
    def ratio(n,d,na): return pl.when(pl.col(d)>0).then(pl.col(n)/pl.col(d)).otherwise(None).alias(na)
    return [ratio("x_gmv_search_sum_30d","gmv_sum_30d","x_share_gmv_search_30d"),ratio("x_gmv_cat_sum_30d","gmv_sum_30d","x_share_gmv_cat_30d"),ratio("x_search_to_ord_30d","searches_sum_14d","x_conv_s2o_14d"),ratio("x_cat_to_ord_30d","x_cat_to_cart_30d","x_conv_c2o_30d"),ratio("x_gmv_sum_14d","gmv_sum_30d","x_gmv_share_14_of_30"),(pl.when(pl.col("x_ord_sum_14d")==0).then(1.0).otherwise(0.0).alias("x_intent_no_ord_14d")),(pl.when(pl.col("to_ord_sum_30d")>0).then(pl.col("gmv_sum_30d")/pl.col("to_ord_sum_30d")).otherwise(None).alias("x_aov_30d")),((pl.col("recency_to_ord_days").clip(upper_bound=365)/((pl.col("tenure_days")+1.0)/(pl.col("order_days_total")+1.0)).clip(lower_bound=1.0)).clip(upper_bound=60.0).alias("x_due_ratio"))]

def build_anchor(a, uids):
    hist=DATA.filter(pl.col("event_date")<=a).filter(pl.col("user_id").is_in(uids))
    be, ce=base_exprs(a)
    feats=hist.group_by("user_id").agg([*be, *ext_agg(a)]).with_columns(ce).with_columns(ext_post())
    idx=pl.DataFrame({"user_id":uids})
    out=idx.join(feats, on="user_id", how="left")
    t=DATA.filter(pl.col("event_date").is_between(a+timedelta(days=1), a+timedelta(days=HORIZON))).filter(pl.col("user_id").is_in(uids)).group_by("user_id").agg(pl.col("gmv").sum().alias("target"))
    out=out.join(t, on="user_id", how="left").with_columns(pl.col("target").fill_null(0.0))
    feat_cols=[c for c in out.columns if c not in ("user_id","target")]
    out=out.with_columns([pl.col(c).cast(pl.Float64) for c in feat_cols]).with_columns([pl.col(c).fill_null(0.0) for c in feat_cols])
    return out

# also need pca/btyd: for pilot speed we skip pca/btyd and use 95 cols, relative ranking stays
for n, anchors in [("n3",ANCHORS_N3),("n5",ANCHORS_N5),("n10",ANCHORS_N10)]:
    log(f"\n=== {n} anchors {anchors} ===")
    df=pl.concat([build_anchor(a, all_uids) for a in anchors])
    log(f"train {df.shape}")
    test_df=build_anchor(date(2026,1,14), all_uids)
    log(f"test {test_df.shape}")
    feature_cols=[c for c in df.columns if c not in ("user_id","target")]
    common=[c for c in feature_cols if c in test_df.columns]
    log(f"features {len(common)}")
    X_train=df.select(common).to_numpy().astype(np.float32)
    y_train=np.log1p(df["target"].to_numpy().astype(float))
    X_test=test_df.select(common).to_numpy().astype(np.float32)
    y_test=np.log1p(test_df["target"].to_numpy().astype(float))
    u_train=df["user_id"].to_numpy()
    u_test=test_df["user_id"].to_numpy()
    # global
    gm=CatBoostRegressor(iterations=500, depth=8, learning_rate=0.05, l2_leaf_reg=3, random_seed=42, loss_function="RMSE", verbose=False, allow_writing_files=False)
    gm.fit(X_train, y_train, verbose=False)
    pred_global=gm.predict(X_test)
    rmsle_global=np.sqrt(mean_squared_error(y_test, pred_global))
    log(f"GLOBAL {rmsle_global:.5f} ({n})")
    for depth,lr,l2,it in [(4,0.1,1,50),(4,0.05,1,50),(4,0.1,1,30),(6,0.1,1,30)]:
        user_to_train_idx=defaultdict(list)
        for idx, uid in enumerate(u_train):
            user_to_train_idx[int(uid)].append(idx)
        uid_to_test_idx={int(uid):i for i,uid in enumerate(u_test)}
        preds=np.zeros_like(pred_global)
        skipped=0
        times=[]
        for uid in all_uids:
            ti=uid_to_test_idx[int(uid)]
            tr_idx=user_to_train_idx.get(int(uid), [])
            if not tr_idx:
                preds[ti]=pred_global[ti]; skipped+=1; continue
            X_tr=X_train[tr_idx]
            y_tr=y_train[tr_idx]
            if np.all(y_tr==y_tr[0]):
                preds[ti]=pred_global[ti]; skipped+=1; continue
            X_te=X_test[ti].reshape(1,-1)
            ts=time.perf_counter()
            m=CatBoostRegressor(iterations=it, depth=depth, learning_rate=lr, l2_leaf_reg=l2, random_seed=42, loss_function="RMSE", verbose=False, allow_writing_files=False, thread_count=1)
            m.fit(X_tr, y_tr, init_model=gm, verbose=False)
            preds[ti]=m.predict(X_te)[0]
            times.append(time.perf_counter()-ts)
        rmsle=np.sqrt(mean_squared_error(y_test, preds))
        log(f"  d{depth} lr{lr} l2{l2} it{it:2d} -> {rmsle:.5f} diff {rmsle-rmsle_global:+.5f} {np.mean(times)*1000:.1f}ms skipped {skipped}")

log("DONE")
