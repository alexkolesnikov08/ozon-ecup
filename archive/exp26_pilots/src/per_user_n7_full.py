#!/usr/bin/env python3
"""
Full 250k n=7 per-user -> LB submit on Desktop
Anchors 7: 26.11,03.12,10.12,17.12,24.12,31.12,07.01 (all < 14.01 fold_03)
Test anchors: fold_03 14.01 (for RMSLE), fold_end 13.02 (for submit)
Hypers: d4 lr0.05 l2=1 it30 (top 5k 1.57) / it50 variant
Parallel per-user via ProcessPool
"""
import time, pathlib, json, multiprocessing as mp
from datetime import date, timedelta
from collections import defaultdict
import numpy as np, polars as pl
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error

SEED=42
HORIZON=30
GLOBAL_ITERS=500
PER_ITERS=30
PER_DEPTH=4
PER_LR=0.05
PER_L2=1
N_JOBS=8  # M1 10c, leave 2
BATCH_CP=5000
ANCHORS_TRAIN=[date(2025,11,26),date(2025,12,3),date(2025,12,10),date(2025,12,17),date(2025,12,24),date(2025,12,31),date(2026,1,7)]
ANCHOR_TEST=date(2026,1,14)
ANCHOR_SUBMIT=date(2026,2,13)

DATA_PATH=pathlib.Path("data/train.parquet")
OUT_DESKTOP=pathlib.Path.home()/"Desktop"/"submission_n7_it30.csv"
REPORT=pathlib.Path("reports/per_user_n7_full.json")
CKPT_DIR=pathlib.Path("reports/ckpt_n7_full")

t0=time.perf_counter()
def log(m): print(f"[{time.perf_counter()-t0:.1f}s] {m}", flush=True)

VALUE_COLS=["gmv","searches","to_ord","to_cart"]
WINDOWS=[(7,"7d"),(14,"14d"),(30,"30d"),(60,"60d"),(90,"90d")]
RECENCY_NONE=999.0

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

def build_anchor(data, anchor, uids):
    hist=data.filter(pl.col("event_date")<=anchor).filter(pl.col("user_id").is_in(uids))
    be, ce=base_exprs(anchor)
    feats=hist.group_by("user_id").agg([*be, *ext_agg(anchor)]).with_columns(ce).with_columns(ext_post())
    idx=pl.DataFrame({"user_id":uids})
    out=idx.join(feats, on="user_id", how="left")
    t=data.filter(pl.col("event_date").is_between(anchor+timedelta(days=1), anchor+timedelta(days=HORIZON))).filter(pl.col("user_id").is_in(uids)).group_by("user_id").agg(pl.col("gmv").sum().alias("target"))
    out=out.join(t, on="user_id", how="left").with_columns(pl.col("target").fill_null(0.0))
    feat_cols=[c for c in out.columns if c not in ("user_id","target")]
    out=out.with_columns([pl.col(c).cast(pl.Float64) for c in feat_cols]).with_columns([pl.col(c).fill_null(0.0) for c in feat_cols])
    return out

log(f"Loading {DATA_PATH} ...")
data=pl.read_parquet(DATA_PATH).with_columns(pl.col("event_date").cast(pl.Date), pl.col("gmv").cast(pl.Float64))
all_uids=data["user_id"].unique().sort().to_list()
log(f"Users {len(all_uids)}")

# Build train 7 anchors
log(f"Building TRAIN {ANCHORS_TRAIN} ...")
CKPT_DIR.mkdir(parents=True, exist_ok=True)
train_parts=[]
for a in ANCHORS_TRAIN:
    t=time.time()
    part=build_anchor(data, a, all_uids)
    train_parts.append(part)
    log(f" anchor {a} {part.shape} {time.time()-t:.1f}s")
train_all=pl.concat(train_parts)
log(f"train_all {train_all.shape}")
feature_cols=[c for c in train_all.columns if c not in ("user_id","target")]
X_train=train_all.select(feature_cols).to_numpy().astype(np.float32)
y_train=np.log1p(train_all["target"].to_numpy().astype(float))
u_train=train_all["user_id"].to_numpy()
log(f"X_train {X_train.shape} feat {len(feature_cols)}")

# Build test and submit
log(f"Building TEST {ANCHOR_TEST} ...")
test_df=build_anchor(data, ANCHOR_TEST, all_uids)
X_test=test_df.select(feature_cols).to_numpy().astype(np.float32)
y_test=np.log1p(test_df["target"].to_numpy().astype(float))
u_test=test_df["user_id"].to_numpy()
log(f"X_test {X_test.shape}")

log(f"Building SUBMIT {ANCHOR_SUBMIT} ...")
submit_df=build_anchor(data, ANCHOR_SUBMIT, all_uids)
X_submit=submit_df.select(feature_cols).to_numpy().astype(np.float32)
u_submit=submit_df["user_id"].to_numpy()
log(f"X_submit {X_submit.shape}")

# Train global
log(f"Training GLOBAL {GLOBAL_ITERS} on {X_train.shape[0]} rows ...")
gm=CatBoostRegressor(iterations=GLOBAL_ITERS, depth=8, learning_rate=0.05, l2_leaf_reg=3, random_seed=SEED, loss_function="RMSE", verbose=50, allow_writing_files=False)
gm.fit(X_train, y_train, verbose=50)
pred_global_test=gm.predict(X_test)
rmsle_global=np.sqrt(mean_squared_error(y_test, pred_global_test))
log(f"GLOBAL RMSLE fold_03 250k: {rmsle_global:.5f}")
pred_global_submit=gm.predict(X_submit)

# Per-user parallel
log(f"Per-user n=7 it{PER_ITERS} d{PER_DEPTH} lr{PER_LR} l2{PER_L2} jobs={N_JOBS} ...")
# mappings
user_to_train_idx=defaultdict(list)
for idx, uid in enumerate(u_train):
    user_to_train_idx[int(uid)].append(idx)
uid_to_test_idx={int(uid):i for i,uid in enumerate(u_test)}
uid_to_submit_idx={int(uid):i for i,uid in enumerate(u_submit)}

# Prepare shared arrays for worker
# Use init_model via global model object pickled - catboost can handle init_model as model object if fork
# For mp we need to pass gm via global var
import pickle, tempfile, os

# Save gm to temp file for workers to load (avoid pickle issues with catboost)
tmp_gm=CKPT_DIR/"global.cbm"
gm.save_model(str(tmp_gm))
log(f"Saved global {tmp_gm}")

# Split users into chunks for Pool
all_uids_sorted=sorted(uid_to_test_idx.keys())
total=len(all_uids_sorted)
# Create list of batches
BATCH=5000
batches=[all_uids_sorted[i:i+BATCH] for i in range(0,total,BATCH)]
log(f"Total batches {len(batches)} x {BATCH}")

# Worker function
def process_batch(batch_uids):
    # load global inside worker
    from catboost import CatBoostRegressor
    import numpy as np
    gm_local=CatBoostRegressor()
    gm_local.load_model(str(tmp_gm))
    # need X_train etc as global? We will capture via closure - but mp needs pickled arrays, we use shared via global variables set in initializer
    # Instead we pass arrays via global in main and use initializer to set them
    global X_train_g, y_train_g, X_test_g, X_submit_g, user_to_train_idx_g, uid_to_test_idx_g, uid_to_submit_idx_g
    preds_test=np.zeros(len(batch_uids))
    preds_submit=np.zeros(len(batch_uids))
    times=[]
    for j, uid in enumerate(batch_uids):
        test_idx=uid_to_test_idx_g[int(uid)]
        submit_idx=uid_to_submit_idx_g[int(uid)]
        train_idx=user_to_train_idx_g.get(int(uid), [])
        if not train_idx:
            preds_test[j]= X_test_g[test_idx]  # placeholder will be replaced in main? Actually we need pred_global
            preds_submit[j]= X_submit_g[submit_idx]
            continue
        # will use pred_global arrays passed
        # This worker needs pred_global arrays
        # Let's instead handle skipping in main loop but we need X/Y
        X_tr=X_train_g[train_idx]
        y_tr=y_train_g[train_idx]
        if np.all(y_tr==y_tr[0]):
            # use global preds
            preds_test[j]=pred_global_test_g[test_idx]
            preds_submit[j]=pred_global_submit_g[submit_idx]
            times.append(0.0)
            continue
        X_te=X_test_g[test_idx].reshape(1,-1)
        X_su=X_submit_g[submit_idx].reshape(1,-1)
        ts=time.perf_counter()
        try:
            m=CatBoostRegressor(iterations=PER_ITERS, depth=PER_DEPTH, learning_rate=PER_LR, l2_leaf_reg=PER_L2, random_seed=SEED, loss_function="RMSE", verbose=False, allow_writing_files=False, thread_count=1)
            m.fit(X_tr, y_tr, init_model=gm_local, verbose=False)
            preds_test[j]=m.predict(X_te)[0]
            preds_submit[j]=m.predict(X_su)[0]
        except:
            preds_test[j]=pred_global_test_g[test_idx]
            preds_submit[j]=pred_global_submit_g[submit_idx]
        times.append(time.perf_counter()-ts)
    return preds_test, preds_submit, times, batch_uids

# Use initializer to share arrays
X_train_g=X_train
y_train_g=y_train
X_test_g=X_test
X_submit_g=X_submit
user_to_train_idx_g=user_to_train_idx
uid_to_test_idx_g=uid_to_test_idx
uid_to_submit_idx_g=uid_to_submit_idx
pred_global_test_g=pred_global_test
pred_global_submit_g=pred_global_submit

def init_worker(xt,yt,xte,xsu,ut2t,ut2s,pt,ps):
    global X_train_g, y_train_g, X_test_g, X_submit_g, user_to_train_idx_g, uid_to_test_idx_g, uid_to_submit_idx_g, pred_global_test_g, pred_global_submit_g
    X_train_g=xt; y_train_g=yt; X_test_g=xte; X_submit_g=xsu; user_to_train_idx_g=ut2t; uid_to_test_idx_g=ut2s[0]; uid_to_submit_idx_g=ut2s[1]; pred_global_test_g=pt; pred_global_submit_g=ps

# Actually our process_batch already uses globals set above, but for Pool we need to pass
# Simplify: use Pool without initializer, capture via closure with pickling - may be heavy but okay for 1.75M rows (300MB per worker copy)
# Instead run sequential batches with Pool inside main loop per batch to avoid copying huge arrays many times: we will process batches sequentially but parallel inside batch via Pool of batch size

# Approach: for each batch, run Pool on that batch's users partitioned

log("Starting per-user parallel_batches ...")
preds_test_all=np.zeros(total, dtype=np.float32)
preds_submit_all=np.zeros(total, dtype=np.float32)
# we will use pred_global as default and overwrite where per-user succeeds
# init with global
for i, uid in enumerate(all_uids_sorted):
    preds_test_all[i]=pred_global_test[uid_to_test_idx[uid]]
    preds_submit_all[i]=pred_global_submit[uid_to_submit_idx[uid]]

t_per=time.perf_counter()
n_skipped=0
all_times=[]

# For progress, iterate batches
for bi, batch_uids in enumerate(batches):
    t_batch=time.perf_counter()
    # split batch into sub-batches for Pool
    # Use Pool with N_JOBS, each worker processes one user
    # To avoid overhead of 5000 tasks, chunk into N_JOBS slices
    def worker_one(uid):
        test_idx=uid_to_test_idx[int(uid)]
        submit_idx=uid_to_submit_idx[int(uid)]
        train_idx=user_to_train_idx.get(int(uid), [])
        if not train_idx:
            return (uid, pred_global_test[test_idx], pred_global_submit[submit_idx], 0.0, True)
        X_tr=X_train[train_idx]
        y_tr=y_train[train_idx]
        if np.all(y_tr==y_tr[0]):
            return (uid, pred_global_test[test_idx], pred_global_submit[submit_idx], 0.0, True)
        X_te=X_test[test_idx].reshape(1,-1)
        X_su=X_submit[submit_idx].reshape(1,-1)
        ts=time.perf_counter()
        try:
            # each worker loads global model from file
            gm_local=CatBoostRegressor()
            gm_local.load_model(str(tmp_gm))
            m=CatBoostRegressor(iterations=PER_ITERS, depth=PER_DEPTH, learning_rate=PER_LR, l2_leaf_reg=PER_L2, random_seed=SEED, loss_function="RMSE", verbose=False, allow_writing_files=False, thread_count=1)
            m.fit(X_tr, y_tr, init_model=gm_local, verbose=False)
            pt=m.predict(X_te)[0]
            ps=m.predict(X_su)[0]
            return (uid, pt, ps, time.perf_counter()-ts, False)
        except Exception as e:
            return (uid, pred_global_test[test_idx], pred_global_submit[submit_idx], 0.0, True)

    with mp.Pool(N_JOBS) as pool:
        results=pool.map(worker_one, batch_uids)
    # collect
    for uid, pt, ps, tm, skipped in results:
        idx=all_uids_sorted.index(uid)  # slow, use dict
        # better use position
        pass

log("Oops index mapping slow, redo with position")
