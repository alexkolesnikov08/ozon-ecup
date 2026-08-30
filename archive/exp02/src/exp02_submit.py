"""exp02 part D: final refit on all 4 CV folds -> predict fold_end ->
YoY calibration -> submissions/submission_exp02.csv.

Consumes the "decision" block of reports/exp02_metrics.json written by
src/exp02_train.py (best feature config, n_estimators, loss, beta, s_hat).
Calibration is applied in z-space: z_cal = z_pred + beta * ln(s_hat),
then y = expm1(z_cal), clip >= 0. Also reports the honest refit-model score
on fold_03 (same protocol as exp01) for comparison with their 1.68125.
"""

import json
import time
from pathlib import Path

import numpy as np
import polars as pl
from catboost import CatBoostRegressor

SEED = 42
DROP_COLS = ["anchor_date", "user_id", "target"]
CV_FOLDS = ["fold_00", "fold_01", "fold_02", "fold_03"]
METRICS_PATH = Path("reports/exp02_metrics.json")

BASE_FEATURES = [
    c for c in pl.read_parquet("data/v2/features/fold_00/batch_0000.parquet").columns
    if c not in DROP_COLS
]

BLOCKS = {
    "ewma": ["ewma_gmv_hl7", "ewma_gmv_hl30", "ewma_to_ord_hl7", "ewma_to_ord_hl30"],
    "trend": ["trend_gmv_7v30", "trend_gmv_30v90", "slope_loggmv_60d"],
    "conv": ["conv_s2o", "conv_c2o", "conv_o2c"],
    "decomp": ["aov_30", "ord_days_30"],
    "due": ["due_ratio"],
    "shares": [
        "share_gmv_search_90", "share_gmv_cat_90",
        "share_gmv_search_trend", "share_gmv_cat_trend",
    ],
    "pop": ["pct_rank_gmv30"],
}


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    lt = np.log1p(np.clip(y_true, 0, None))
    lp = np.log1p(np.clip(y_pred, 0, None))
    return float(np.sqrt(np.mean((lt - lp) ** 2)))


def join_fold(name: str, extra_cols: list[str]) -> pl.DataFrame:
    base_dir = Path(f"data/v2/features/{name}")
    if not base_dir.exists():
        base_dir = Path(f"data/v2/features_exp02/{name}_base")
        assert base_dir.exists(), f"no base cache for {name}"
    base = pl.read_parquet(str(base_dir / "batch_*.parquet"))
    if not extra_cols:
        return add_pct_rank(base)
    extra = pl.read_parquet(f"data/v2/features_exp02/{name}/batch_*.parquet")
    j = base.join(
        extra.select(["user_id", "anchor_date", *extra_cols]),
        on="user_id", how="inner", suffix="_x2",
    )
    assert j.height == base.height == 250_000
    assert (j["anchor_date"] == j["anchor_date_x2"]).all()
    j = j.drop("anchor_date_x2")
    return add_pct_rank(j)


def add_pct_rank(df: pl.DataFrame) -> pl.DataFrame:
    n = df.height
    return df.with_columns(
        ((pl.col("gmv_sum_30d").rank(method="average") - 1) / (n - 1))
        .cast(pl.Float64)
        .alias("pct_rank_gmv30")
    )


def main() -> None:
    metrics = json.loads(METRICS_PATH.read_text())
    dec = metrics["decision"]
    cfg, n_est, loss = dec["feature_config"], dec["n_estimators"], dec["loss"]
    beta, s_hat = dec["beta"], dec["s_hat"]
    blocks = dec["blocks"]
    extra_cols = [f for b in blocks for f in BLOCKS[b]]
    feat_cols = BASE_FEATURES + extra_cols
    print(f"decision: config={cfg} ({len(feat_cols)} feats), n_estimators={n_est}, "
          f"loss={loss}, beta={beta}, s_hat={s_hat}", flush=True)

    print("loading folds...", flush=True)
    train_df = pl.concat([join_fold(f, extra_cols) for f in CV_FOLDS], how="vertical")
    end_df = join_fold("fold_end", extra_cols)
    print(f"train: {train_df.shape}, fold_end: {end_df.shape}", flush=True)

    X_tr = train_df.select(feat_cols).to_numpy()
    y_tr = np.log1p(np.clip(train_df["target"].to_numpy(), 0, None))
    X_end = end_df.select(feat_cols).to_numpy()

    model = CatBoostRegressor(
        loss_function=loss,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3,
        n_estimators=n_est,
        thread_count=-1,
        random_seed=SEED,
        verbose=0,
        allow_writing_files=False,
    )
    t0 = time.time()
    model.fit(X_tr, y_tr)
    fit_time = time.time() - t0
    print(f"final model ({n_est} iters, {loss}) trained in {fit_time:.1f}s", flush=True)

    val_df = join_fold("fold_03", extra_cols)
    X_val = val_df.select(feat_cols).to_numpy()
    y_val_raw = np.clip(val_df["target"].to_numpy(), 0, None)
    refit_score = rmsle(y_val_raw, np.clip(np.expm1(model.predict(X_val)), 0, None))
    print(f"refit-model RMSLE on fold_03: {refit_score:.5f} (exp01: 1.68125)", flush=True)

    ln_s = float(np.log(s_hat))
    pred_raw = np.expm1(model.predict(X_end) + beta * ln_s)
    pred = np.clip(pred_raw, 0, None).astype(np.float64)
    assert np.isfinite(pred).all() and (pred >= 0).all()

    sample = pl.read_csv("sample_submit.csv")
    sub = pl.DataFrame({"user_id": end_df["user_id"], "predict": pred}).join(
        sample.select("user_id"), on="user_id", how="semi"
    )
    assert set(sub["user_id"]) == set(sample["user_id"]), "user_id set mismatch"
    sub = sub.join(sample.select("user_id").with_row_index("__ord"), on="user_id")
    sub = sub.sort("__ord").drop("__ord")
    assert sub.height == 250_000 and sub.width == 2

    Path("submissions").mkdir(exist_ok=True)
    out = Path("submissions/submission_exp02.csv")
    sub.write_csv(out)

    chk = pl.read_csv(out)
    assert chk.height == 250_000 and chk.width == 2
    assert chk.columns == ["user_id", "predict"]
    assert chk["predict"].is_finite().all() and (chk["predict"] >= 0).all()
    print(f"submission saved: {out} ({chk.height} rows, "
          f"pred mean={chk['predict'].mean():.2f}, "
          f"median={chk['predict'].median():.4f})", flush=True)

    metrics["submission"] = {
        "path": str(out),
        "config": cfg,
        "n_features": len(feat_cols),
        "n_estimators": n_est,
        "loss": loss,
        "beta": beta,
        "s_hat": s_hat,
        "fit_time_sec": round(fit_time, 1),
        "refit_rmsle_fold03": round(refit_score, 5),
        "exp01_refit_rmsle_fold03": 1.68125,
        "pred_mean": round(float(chk["predict"].mean()), 4),
        "pred_median": round(float(chk["predict"].median()), 4),
        "pred_min": round(float(chk["predict"].min()), 6),
        "pred_max": round(float(chk["predict"].max()), 2),
        "rows": chk.height,
        "columns": chk.columns,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print("metrics updated", flush=True)


if __name__ == "__main__":
    main()
