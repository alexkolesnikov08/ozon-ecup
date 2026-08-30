#!/usr/bin/env python3
"""Full n=5 it=25 prod: 5 anchors -> fold_end submit"""
import time, json, numpy as np, polars as pl
from pathlib import Path
from datetime import date, timedelta
from catboost import CatBoostRegressor
from collections import defaultdict

SEED=42
GLOBAL_ITERS=500
ITER_PER_USER=25
BATCH_SIZE=5000
# 5 anchors ending at fold_03 (2026-01-14) with 7d step
ANCHORS_TRAIN=[date(2025,12,17), date(2025,12,24), date(2025,12,31), date(2026,1,7), date(2026,1,14)]
ANCHOR_PRED=date(2026,2,13)
OUT_DIR=Path.home()/"Desktop"
REPORT=Path("reports/per_user_full_n5_it25.json")
CKPT_DIR=Path("reports/ckpt_n5_it25")
HORIZON=30
VALUE_COLS=["gmv","searches","to_ord","to_cart"]
WINDOWS=[(7,"7d"),(14,"14d"),(30,"30d"),(60,"60d"),(90,"90d")]
RECENCY_NONE=999.0

t0=time.perf_counter()
def log(m): print(f"[{time.perf_counter()-t0:.3f}s] {m}", flush=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)
Path("reports").mkdir(exist_ok=True)
Path("submissions").mkdir(exist_ok=True)

def win_mask(a,d): return pl.col("event_date").is_between(a-timedelta(days=d-1), a)
def wsum(a,d,col,name): return (pl.when(win_mask(a,d)).then(pl.col(col)).otherwise(0.0).sum().alias(name))
def base_exprs(a):
    exprs=[]
    for w_days,w_name in WINDOWS:
        mask=win_mask(a,w_days)
        for col in VALUE_COLS:
            c=pl.col(col)
            exprs.append(pl.when(mask).then(c).otherwise(0.0).sum().alias(f"{col}_sum_{w_name}"))
            exprs.append(pl.when(mask).then(c).otherwise(None).max().alias(f"{col}_max_{w_name}"))
            exprs.append(pl.when(mask).then(c).otherwise(None).mean().alias(f"{col}_mean_{w_name}"))
    m30=win_mask(a,30)
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
    m14=win_mask(a,14)
    return [wsum(a,14,"searches","x_searches_sum_14d"),wsum(a,14,"to_cart","x_cart_sum_14d"),wsum(a,14,"to_ord","x_ord_sum_14d"),wsum(a,14,"gmv","x_gmv_sum_14d"),wsum(a,30,"gmv_search","x_gmv_search_sum_30d"),wsum(a,30,"gmv_cat","x_gmv_cat_sum_30d"),wsum(a,30,"search_to_ord","x_search_to_ord_30d"),wsum(a,30,"cat_to_ord","x_cat_to_ord_30d"),wsum(a,30,"cat_to_cart","x_cat_to_cart_30d"),(pl.when(m14&(pl.col("to_cart")>0)&(pl.col("to_ord")==0)).then(1).otherwise(0).sum().cast(pl.Float64).alias("x_cart_no_ord_days_14d")),(pl.when(win_mask(a,30)&(pl.col("searches")==0)&(pl.col("to_cart")==0)&(pl.col("to_ord")==0)&(pl.col("gmv")==0)).then(1).otherwise(0).sum().cast(pl.Float64).alias("x_visit_only_days_30d")),(pl.when(m14&(pl.col("searches")>0)).then(1).otherwise(0).sum().cast(pl.Float64).alias("x_search_days_14d"))]
def ext_post():
    def ratio(n,d,na): return pl.when(pl.col(d)>0).then(pl.col(n)/pl.col(d)).otherwise(None).alias(na)
    return [ratio("x_gmv_search_sum_30d","gmv_sum_30d","x_share_gmv_search_30d"),ratio("x_gmv_cat_sum_30d","gmv_sum_30d","x_share_gmv_cat_30d"),ratio("x_search_to_ord_30d","searches_sum_14d","x_conv_s2o_14d"),ratio("x_cat_to_ord_30d","x_cat_to_cart_30d","x_conv_c2o_30d"),ratio("x_gmv_sum_14d","gmv_sum_30d","x_gmv_share_14_of_30"),(pl.when(pl.col("x_ord_sum_14d")==0).then(1.0).otherwise(0.0).alias("x_intent_no_ord_14d")),(pl.when(pl.col("to_ord_sum_30d")>0).then(pl.col("gmv_sum_30d")/pl.col("to_ord_sum_30d")).otherwise(None).alias("x_aov_30d")),((pl.col("recency_to_ord_days").clip(upper_bound=365)/((pl.col("tenure_days")+1.0)/(pl.col("order_days_total")+1.0)).clip(lower_bound=1.0)).clip(upper_bound=60.0).alias("x_due_ratio"))]
def build_anchor(df_all, anchor, uids):
    hist=df_all.filter(pl.col("event_date")<=anchor).filter(pl.col("user_id").is_in(uids))
    be, ce=base_exprs(anchor)
    feats=hist.group_by("user_id").agg([*be, *ext_agg(anchor)]).with_columns(ce).with_columns(ext_post())
    idx=pl.DataFrame({"user_id":uids})
    out=idx.join(feats, on="user_id", how="left")
    t=df_all.filter(pl.col("event_date").is_between(anchor+timedelta(days=1), anchor+timedelta(days=HORIZON))).filter(pl.col("user_id").is_in(uids)).group_by("user_id").agg(pl.col("gmv").sum().alias("target"))
    out=out.join(t, on="user_id", how="left").with_columns(pl.col("target").fill_null(0.0))
    feat_cols=[c for c in out.columns if c not in ("user_id","target")]
    out=out.with_columns([pl.col(c).cast(pl.Float64) for c in feat_cols]).with_columns([pl.col(c).fill_null(0.0) for c in feat_cols])
    return out

log(f"Loading train.parquet...")
data=pl.read_parquet("data/train.parquet").with_columns(pl.col("event_date").cast(pl.Date), pl.col("gmv").cast(pl.Float64))
all_uids=data["user_id"].unique().sort().to_list()
log(f"Users {len(all_uids)} Anchors train {ANCHORS_TRAIN} pred {ANCHOR_PRED}")

# Build train for 250k x5
log("Building TRAIN tables 5 anchors x 250k...")
train_parts=[]
for a in ANCHORS_TRAIN:
    t=time.time()
    part=build_anchor(data, a, all_uids)
    train_parts.append(part)
    log(f"  anchor {a} {part.shape} {time.time()-t:.1f}s")
train_all=pl.concat(train_parts)
log(f"train_all {train_all.shape}")
feature_cols=[c for c in train_all.columns if c not in ("user_id","target")]
X_train=train_all.select(feature_cols).to_numpy()
y_train_z=np.log1p(train_all["target"].to_numpy())
train_uids=train_all["user_id"].to_numpy()

# Build pred table for fold_end
log(f"Building PRED table anchor {ANCHOR_PRED}...")
pred_df=build_anchor(data, ANCHOR_PRED, all_uids)  # target will be 0 but we need features
# For pred we need features only, but build_anchor with target gives target as 0 for future (since data ends 2026-02-13, target for 02-13 is 0 as no future). That's fine, we just need features.
X_pred=pred_df.select(feature_cols).to_numpy()
pred_uids=pred_df["user_id"].to_numpy()
log(f"pred {pred_df.shape}")

# Train global
log(f"Training GLOBAL {GLOBAL_ITERS} on {X_train.shape[0]} rows...")
t_g=time.time()
gm=CatBoostRegressor(iterations=GLOBAL_ITERS, depth=8, learning_rate=0.05, l2_leaf_reg=3, random_seed=SEED, loss_function="RMSE", verbose=100, allow_writing_files=False)
gm.fit(X_train, y_train_z, verbose=100)
log(f"GLOBAL done {time.time()-t_g:.1f}s")
y_pred_global=gm.predict(X_pred)

# Per-user
user_to_idx=defaultdict(list)
for i,uid in enumerate(train_uids):
    user_to_idx[int(uid)].append(i)
uid_to_pred={int(uid):i for i,uid in enumerate(pred_uids)}
total=len(all_uids)
preds_per=np.zeros(total)
times=[]
n_skipped=0
n_done=0
ckpt=CKPT_DIR/"progress.json"
if ckpt.exists():
    try:
        ck=json.loads(ckpt.read_text())
        n_done=ck.get("n_done",0)
        n_skipped=ck.get("n_skipped",0)
        if (CKPT_DIR/"preds.npy").exists():
            preds_per=np.load(CKPT_DIR/"preds.npy")
        if (CKPT_DIR/"times.npy").exists():
            times=np.load(CKPT_DIR/"times.npy").tolist()
        log(f"RESUME {n_done}")
    except: pass

t_per=time.time()
for bs in range(n_done, total, BATCH_SIZE):
    be=min(bs+BATCH_SIZE, total)
    batch=all_uids[bs:be]
    log(f"Batch {bs//BATCH_SIZE+1}/{(total+BATCH_SIZE-1)//BATCH_SIZE} {bs}:{be}")
    t_b=time.time()
    for pos, uid in enumerate(batch):
        gp=bs+pos
        pred_idx=uid_to_pred[uid]
        idx=user_to_idx.get(uid, [])
        if not idx:
            preds_per[gp]=y_pred_global[pred_idx]; times.append(0); n_skipped+=1; continue
        X_tr=X_train[idx]; y_tr=y_train_z[idx]
        if np.all(y_tr==y_tr[0]):
            preds_per[gp]=y_pred_global[pred_idx]; times.append(0.0005); n_skipped+=1; continue
        X_te=X_pred[pred_idx].reshape(1,-1)
        t_s=time.time()
        try:
            m=CatBoostRegressor(iterations=ITER_PER_USER, depth=8, learning_rate=0.05, l2_leaf_reg=3, random_seed=SEED, loss_function="RMSE", verbose=False, allow_writing_files=False)
            m.fit(X_tr, y_tr, init_model=gm, verbose=False)
            pred=m.predict(X_te)[0]
        except:
            pred=y_pred_global[pred_idx]; n_skipped+=1
        times.append(time.time()-t_s)
        preds_per[gp]=pred
        if (gp+1)%10000==0:
            log(f"  {gp+1}/{total} avg {np.mean(times[-10000:])*1000:.1f}ms")
    np.save(CKPT_DIR/"preds.npy", preds_per)
    np.save(CKPT_DIR/"times.npy", np.array(times))
    ckpt.write_text(json.dumps({"n_done":be, "n_skipped":n_skipped}))
    log(f"Batch done {time.time()-t_b:.1f}s skipped {n_skipped}")

preds_y=np.maximum(np.expm1(preds_per),0.0)
submit=pl.DataFrame({"user_id":all_uids, "predict":preds_y}).sort("user_id")
out_path=OUT_DIR / "submission_n5_it25.csv"
submit.write_csv(str(out_path))
Path("submissions").mkdir(exist_ok=True)
submit.write_csv("submissions/submission_n5_it25.csv")
log(f"Saved {out_path} mean {preds_y.mean():.2f} zeros {(preds_y==0).sum()} wall {time.time()-t0:.1f}s")

# report
times_arr=np.array(times)
report={"n_train":5,"iter":25,"global_iters":GLOBAL_ITERS,"total":total,"n_skipped":int(n_skipped),"mean_ms":float(times_arr.mean()*1000),"median_ms":float(np.median(times_arr)*1000),"wall_s":float(time.time()-t0)}
with open(REPORT,"w") as f:
    json.dump(report,f,indent=2)
log(f"Saved {REPORT}")

# Simple plots for this run
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.figure(figsize=(10,4))
plt.hist(times_arr*1000, bins=80, color="#4C78A8", edgecolor="white")
plt.axvline(times_arr.mean()*1000, color="red", linestyle="--", label=f"mean {times_arr.mean()*1000:.1f}ms")
plt.xlabel("ms per user"); plt.ylabel("count"); plt.title("n=5 it=25 per-user time 250k")
plt.legend(); plt.tight_layout()
plt.savefig(str(OUT_DIR/"per_user_n5_it25_speed.png"), dpi=150)
plt.savefig("reports/figures/per_user_n5_it25_speed.png", dpi=150)
log("Plots done")
print("MAYAK SUBMIT DONE")
