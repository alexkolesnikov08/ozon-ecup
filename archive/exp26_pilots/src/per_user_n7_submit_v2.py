#!/usr/bin/env python3
import time, pathlib, json
from datetime import date, timedelta
from collections import defaultdict
import numpy as np, polars as pl
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
import multiprocessing as mp

SEED=42
HORIZON=30
GLOBAL_ITERS=500
PER_ITERS=30
PER_DEPTH=4
PER_LR=0.05
PER_L2=1
N_JOBS=8
ANCHORS_TRAIN=[date(2025,11,26),date(2025,12,3),date(2025,12,10),date(2025,12,17),date(2025,12,24),date(2025,12,31),date(2026,1,7)]
ANCHOR_TEST=date(2026,1,14)
ANCHOR_SUBMIT=date(2026,2,13)
DATA_PATH=pathlib.Path("data/train.parquet")
OUT_DESKTOP=pathlib.Path.home()/"Desktop"/f"submission_n7_it{PER_ITERS}_d{PER_DEPTH}.csv"
REPORT=pathlib.Path("reports/per_user_n7_full.json")
CKPT=pathlib.Path("reports/ckpt_n7_full/progress.json")

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

# Globals for workers
X_train_g=None; y_train_g=None; X_test_g=None; X_submit_g=None
pred_global_test_g=None; pred_global_submit_g=None
user_to_train_idx_g=None; uid_to_test_idx_g=None; uid_to_submit_idx_g=None
tmp_gm_g=None

def init_worker(xt,yt,xte,xsu,pt,ps,ut2t,ut2te,ut2su,tmp):
    global X_train_g, y_train_g, X_test_g, X_submit_g, pred_global_test_g, pred_global_submit_g, user_to_train_idx_g, uid_to_test_idx_g, uid_to_submit_idx_g, tmp_gm_g
    X_train_g=xt; y_train_g=yt; X_test_g=xte; X_submit_g=xsu; pred_global_test_g=pt; pred_global_submit_g=ps; user_to_train_idx_g=ut2t; uid_to_test_idx_g=ut2te; uid_to_submit_idx_g=ut2su; tmp_gm_g=tmp

def worker(uid):
    test_idx=uid_to_test_idx_g[int(uid)]
    submit_idx=uid_to_submit_idx_g[int(uid)]
    train_idx=user_to_train_idx_g.get(int(uid), [])
    if not train_idx:
        return (pred_global_test_g[test_idx], pred_global_submit_g[submit_idx], 0.0, 1)
    X_tr=X_train_g[train_idx]
    y_tr=y_train_g[train_idx]
    if np.all(y_tr==y_tr[0]):
        return (pred_global_test_g[test_idx], pred_global_submit_g[submit_idx], 0.0, 1)
    X_te=X_test_g[test_idx].reshape(1,-1)
    X_su=X_submit_g[submit_idx].reshape(1,-1)
    ts=time.perf_counter()
    try:
        gm_local=CatBoostRegressor()
        gm_local.load_model(tmp_gm_g)
        m=CatBoostRegressor(iterations=PER_ITERS, depth=PER_DEPTH, learning_rate=PER_LR, l2_leaf_reg=PER_L2, random_seed=SEED, loss_function="RMSE", verbose=False, allow_writing_files=False, thread_count=1)
        m.fit(X_tr, y_tr, init_model=gm_local, verbose=False)
        pt=m.predict(X_te)[0]
        ps=m.predict(X_su)[0]
        return (pt, ps, time.perf_counter()-ts, 0)
    except:
        return (pred_global_test_g[test_idx], pred_global_submit_g[submit_idx], 0.0, 1)

if __name__ == "__main__":
    t0=time.perf_counter()
    def log(m): print(f"[{time.perf_counter()-t0:.1f}s] {m}", flush=True)
    CKPT.parent.mkdir(parents=True, exist_ok=True)
    log(f"Loading {DATA_PATH}")
    data=pl.read_parquet(DATA_PATH).with_columns(pl.col("event_date").cast(pl.Date), pl.col("gmv").cast(pl.Float64))
    all_uids=data["user_id"].unique().sort().to_list()
    log(f"Users {len(all_uids)} Anchors train {ANCHORS_TRAIN}")
    train_parts=[]
    for a in ANCHORS_TRAIN:
        t=time.time()
        part=build_anchor(data, a, all_uids)
        train_parts.append(part)
        log(f" anchor {a} {part.shape} {time.time()-t:.1f}s")
    train_all=pl.concat(train_parts)
    feature_cols=[c for c in train_all.columns if c not in ("user_id","target")]
    X_train=train_all.select(feature_cols).to_numpy().astype(np.float32)
    y_train=np.log1p(train_all["target"].to_numpy().astype(float))
    u_train=train_all["user_id"].to_numpy()
    log(f"X_train {X_train.shape} feats {len(feature_cols)}")
    test_df=build_anchor(data, ANCHOR_TEST, all_uids)
    X_test=test_df.select(feature_cols).to_numpy().astype(np.float32)
    y_test=np.log1p(test_df["target"].to_numpy().astype(float))
    u_test=test_df["user_id"].to_numpy()
    log(f"X_test {X_test.shape}")
    submit_df=build_anchor(data, ANCHOR_SUBMIT, all_uids)
    X_submit=submit_df.select(feature_cols).to_numpy().astype(np.float32)
    u_submit=submit_df["user_id"].to_numpy()
    log(f"X_submit {X_submit.shape}")
    log(f"Training GLOBAL {GLOBAL_ITERS}")
    gm=CatBoostRegressor(iterations=GLOBAL_ITERS, depth=8, learning_rate=0.05, l2_leaf_reg=3, random_seed=SEED, loss_function="RMSE", verbose=50, allow_writing_files=False)
    gm.fit(X_train, y_train, verbose=50)
    pred_global_test=gm.predict(X_test)
    rmsle_global=np.sqrt(mean_squared_error(y_test, pred_global_test))
    log(f"GLOBAL RMSLE {rmsle_global:.5f}")
    pred_global_submit=gm.predict(X_submit)
    tmp_gm=CKPT.parent/"global_n7.cbm"
    gm.save_model(str(tmp_gm))
    user_to_train_idx=defaultdict(list)
    for idx, uid in enumerate(u_train):
        user_to_train_idx[int(uid)].append(idx)
    uid_to_test_idx={int(uid):i for i,uid in enumerate(u_test)}
    uid_to_submit_idx={int(uid):i for i,uid in enumerate(u_submit)}
    all_uids_sorted=sorted(uid_to_test_idx.keys())
    total=len(all_uids_sorted)
    log(f"Per-user n=7 it{PER_ITERS} d{PER_DEPTH} lr{PER_LR} l2{PER_L2} jobs={N_JOBS} total {total}")
    preds_test=np.zeros(total, dtype=np.float32)
    preds_submit=np.zeros(total, dtype=np.float32)
    for i, uid in enumerate(all_uids_sorted):
        preds_test[i]=pred_global_test[uid_to_test_idx[uid]]
        preds_submit[i]=pred_global_submit[uid_to_submit_idx[uid]]
    n_done=0; n_skipped=0; times=[]
    if CKPT.exists():
        try:
            ck=json.loads(CKPT.read_text())
            n_done=ck.get("n_done",0)
            n_skipped=ck.get("n_skipped",0)
            if (CKPT.parent/"preds_test.npy").exists():
                preds_test=np.load(CKPT.parent/"preds_test.npy")
            if (CKPT.parent/"preds_submit.npy").exists():
                preds_submit=np.load(CKPT.parent/"preds_submit.npy")
            if (CKPT.parent/"times.npy").exists():
                times=np.load(CKPT.parent/"times.npy").tolist()
            log(f"RESUME from {n_done} skipped {n_skipped}")
        except Exception as e:
            log(f"resume fail {e}")
    BATCH=5000
    for start in range(n_done, total, BATCH):
        end=min(start+BATCH, total)
        batch_uids=all_uids_sorted[start:end]
        t_batch=time.perf_counter()
        with mp.Pool(N_JOBS, initializer=init_worker, initargs=(X_train,y_train,X_test,X_submit,pred_global_test,pred_global_submit,user_to_train_idx,uid_to_test_idx,uid_to_submit_idx,str(tmp_gm))) as pool:
            results=pool.map(worker, batch_uids)
        for i, (pt,ps,tm,sk) in enumerate(results):
            idx=start+i
            preds_test[idx]=pt
            preds_submit[idx]=ps
            times.append(tm)
            n_skipped+=sk
        np.save(CKPT.parent/"preds_test.npy", preds_test)
        np.save(CKPT.parent/"preds_submit.npy", preds_submit)
        np.save(CKPT.parent/"times.npy", np.array(times))
        CKPT.write_text(json.dumps({"n_done":end,"n_skipped":int(n_skipped)}))
        rmsle_sofar=np.sqrt(mean_squared_error(y_test[:end], preds_test[:end])) if end>0 else 0
        avg_ms=np.mean([t for t in times[-BATCH:] if t>0])*1000 if times else 0
        log(f"Batch {start//BATCH+1}/{(total+BATCH-1)//BATCH} {start}:{end} rmsle_sofar {rmsle_sofar:.5f} avg {avg_ms:.1f}ms batch {time.perf_counter()-t_batch:.1f}s skipped {n_skipped}")
    rmsle_per=np.sqrt(mean_squared_error(y_test, preds_test))
    times_arr=np.array([t for t in times if t>0])
    log(f"FINAL GLOBAL {rmsle_global:.5f} PER {rmsle_per:.5f} diff {rmsle_per-rmsle_global:+.5f}")
    if len(times_arr)>0:
        log(f"Timing mean {times_arr.mean()*1000:.1f}ms median {np.median(times_arr)*1000:.1f}ms p90 {np.percentile(times_arr,90)*1000:.1f}ms total {times_arr.sum():.1f}s skipped {n_skipped}/{total}")
    log(f"Wall {time.perf_counter()-t0:.1f}s")
    preds_submit_y=np.maximum(np.expm1(preds_submit),0.0)
    submit_df_out=pl.DataFrame({"user_id":all_uids_sorted, "predict":preds_submit_y}).sort("user_id")
    assert submit_df_out.height==250000, f"height {submit_df_out.height}"
    OUT_DESKTOP.parent.mkdir(parents=True, exist_ok=True)
    submit_df_out.write_csv(str(OUT_DESKTOP))
    log(f"Saved submit {OUT_DESKTOP} mean {preds_submit_y.mean():.2f} zeros {(preds_submit_y==0).sum()}")
    pl.DataFrame({"user_id":all_uids_sorted, "predict":preds_submit_y}).write_csv("submissions/submission_n7_it30.csv")
    REPORT.write_text(json.dumps({"global_rmsle":float(rmsle_global),"per_rmsle":float(rmsle_per),"diff":float(rmsle_per-rmsle_global),"n":7,"it":PER_ITERS,"depth":PER_DEPTH,"lr":PER_LR,"l2":PER_L2,"mean_ms":float(times_arr.mean()*1000) if len(times_arr)>0 else 0,"skipped":int(n_skipped),"wall_s":float(time.perf_counter()-t0)}, indent=2))
    log(f"Done report {REPORT}")
