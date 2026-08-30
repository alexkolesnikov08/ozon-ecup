"""Train a hierarchical time-series model and create a competition submission.

Input is produced by :mod:`build_segmented_features` directly from the base
``data/train.parquet``.  The hierarchy is:

1. one global CatBoost model;
2. one local model for every ADI/CV^2 demand class;
3. one local model for every cluster inside a demand class;
4. a convex blend selected on an inner time fold.

The validation protocol is intentionally time ordered:

* fold_00+01 -> fold_02: choose blend weights;
* fold_00+01+02 -> fold_03: honest model comparison;
* fold_00+01+02+03 -> fold_end: refit and submission.

Run from the repository root::

    python src/train_segmented_submit.py --threads 16
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from catboost import CatBoostRegressor
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURES_DIR = Path("data/segmented_base")
SAMPLE_PATH = Path("sample_submit.csv")
SUBMISSION_PATH = Path("submissions/submission_segmented.csv")
REPORT_PATH = Path("reports/segmented_pipeline.json")
PREDICTIONS_DIR = Path("data/segmented_predictions/classical")

TRAIN_INNER = ["fold_00", "fold_01"]
EVAL_INNER = "fold_02"
TRAIN_HOLDOUT = ["fold_00", "fold_01", "fold_02"]
EVAL_HOLDOUT = "fold_03"
TRAIN_FINAL = ["fold_00", "fold_01", "fold_02", "fold_03"]
EVAL_FINAL = "fold_end"

META_COLS = {"anchor_date", "user_id", "target"}
CLASS_NAMES = {
    0: "smooth",
    1: "erratic",
    2: "intermittent",
    3: "lumpy",
}

CLUSTER_FEATURE_CANDIDATES = [
    "adi_26w",
    "cv2_demand_26w",
    "demand_weeks_26w",
    "zero_week_share_26w",
    "positive_week_gmv_mean_26w",
    "weekly_gmv_mean_26w",
    "weekly_gmv_max_26w",
    "gmv_sum_7d",
    "gmv_sum_30d",
    "gmv_sum_90d",
    "gmv_sum_180d",
    "searches_sum_30d",
    "searches_sum_90d",
    "to_ord_sum_30d",
    "to_ord_sum_90d",
    "to_cart_sum_30d",
    "recency_order_days",
    "recency_search_days",
    "tenure_days",
    "gmv_recent_share_7_30",
    "gmv_recent_share_30_90",
    *[f"week_gmv_{i:02d}" for i in range(8)],
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Train global + class + cluster models and write submission."
    )
    ap.add_argument("--features-dir", type=Path, default=FEATURES_DIR)
    ap.add_argument("--sample", type=Path, default=SAMPLE_PATH)
    ap.add_argument("--submission", type=Path, default=SUBMISSION_PATH)
    ap.add_argument("--report", type=Path, default=REPORT_PATH)
    ap.add_argument("--predictions-dir", type=Path, default=PREDICTIONS_DIR)
    ap.add_argument("--threads", type=int, default=-1)
    ap.add_argument("--clusters-per-class", type=int, default=3)
    ap.add_argument("--min-local-rows", type=int, default=2_500)
    ap.add_argument("--min-hist-rows", type=int, default=15_000)
    ap.add_argument("--min-cat-rows", type=int, default=100_000)
    ap.add_argument("--max-cluster-fit-rows", type=int, default=300_000)
    ap.add_argument("--global-iters", type=int, default=1_000)
    ap.add_argument("--local-iters", type=int, default=500)
    ap.add_argument("--hist-iters", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Fast infrastructure check: fewer boosting iterations.",
    )
    args = ap.parse_args()
    if args.clusters_per_class < 1:
        ap.error("--clusters-per-class must be positive")
    if args.min_local_rows < 100:
        ap.error("--min-local-rows must be at least 100")
    if args.quick:
        args.global_iters = min(args.global_iters, 120)
        args.local_iters = min(args.local_iters, 80)
        args.hist_iters = min(args.hist_iters, 80)
    return args


def rmsle_z(y_raw: np.ndarray, z_pred: np.ndarray) -> float:
    z_true = np.log1p(np.clip(y_raw.astype(np.float64), 0.0, None))
    z_safe = np.clip(np.asarray(z_pred, dtype=np.float64), 0.0, 20.0)
    return float(np.sqrt(np.mean((z_true - z_safe) ** 2)))


def feature_columns(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS]


def load_stage(
    features_dir: Path,
    train_folds: list[str],
    eval_fold: str,
) -> tuple[pl.DataFrame, pl.DataFrame, list[str]]:
    paths = [features_dir / f"{name}.parquet" for name in train_folds + [eval_fold]]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing feature folds. Run src/build_segmented_features.py first: "
            + ", ".join(missing)
        )
    train_parts = [pl.read_parquet(features_dir / f"{name}.parquet") for name in train_folds]
    eval_df = pl.read_parquet(features_dir / f"{eval_fold}.parquet")
    cols = feature_columns(train_parts[0])
    expected = ["anchor_date", "user_id", *cols, "target"]
    for name, df in zip(train_folds, train_parts):
        if df.columns != expected:
            raise ValueError(f"{name}: feature schema/order mismatch")
        if df["target"].null_count():
            raise ValueError(f"{name}: training target contains NULL")
    if feature_columns(eval_df) != cols:
        raise ValueError(f"{eval_fold}: feature schema/order mismatch")
    train_df = pl.concat(train_parts, how="vertical")
    return train_df, eval_df, cols


def to_matrix(df: pl.DataFrame, cols: list[str]) -> np.ndarray:
    x = df.select(cols).to_numpy().astype(np.float32, copy=False)
    return np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=0.0)


@dataclass
class DemandClusterer:
    clusters_per_class: int
    max_fit_rows: int
    min_local_rows: int
    seed: int
    cluster_feature_indices: list[int]
    models: dict[int, tuple[StandardScaler, MiniBatchKMeans]]

    @classmethod
    def fit(
        cls,
        x: np.ndarray,
        demand_class: np.ndarray,
        feature_names: list[str],
        clusters_per_class: int,
        max_fit_rows: int,
        min_local_rows: int,
        seed: int,
    ) -> "DemandClusterer":
        cluster_cols = [c for c in CLUSTER_FEATURE_CANDIDATES if c in feature_names]
        if len(cluster_cols) < 8:
            raise ValueError(
                "Too few clustering features. Was the dataset built with "
                "src/build_segmented_features.py?"
            )
        indices = [feature_names.index(c) for c in cluster_cols]
        rng = np.random.default_rng(seed)
        models: dict[int, tuple[StandardScaler, MiniBatchKMeans]] = {}

        for class_id in sorted(np.unique(demand_class).astype(int)):
            rows = np.flatnonzero(demand_class == class_id)
            possible = max(1, len(rows) // max(min_local_rows, 1))
            n_clusters = min(clusters_per_class, possible)
            if n_clusters <= 1:
                continue
            if len(rows) > max_fit_rows:
                rows = rng.choice(rows, size=max_fit_rows, replace=False)
            values = np.log1p(np.clip(x[rows][:, indices], 0.0, 1e9))
            scaler = StandardScaler().fit(values)
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                batch_size=min(8_192, max(1_024, len(rows))),
                n_init=10,
                max_iter=200,
                random_state=seed + class_id,
                reassignment_ratio=0.01,
            ).fit(scaler.transform(values))
            models[class_id] = (scaler, kmeans)
        return cls(
            clusters_per_class=clusters_per_class,
            max_fit_rows=max_fit_rows,
            min_local_rows=min_local_rows,
            seed=seed,
            cluster_feature_indices=indices,
            models=models,
        )

    def predict(self, x: np.ndarray, demand_class: np.ndarray) -> np.ndarray:
        # A stable global group id: class_id * requested_K + in-class cluster.
        result = demand_class.astype(np.int32) * self.clusters_per_class
        for class_id, (scaler, kmeans) in self.models.items():
            rows = np.flatnonzero(demand_class == class_id)
            if not len(rows):
                continue
            values = np.log1p(
                np.clip(x[rows][:, self.cluster_feature_indices], 0.0, 1e9)
            )
            result[rows] += kmeans.predict(scaler.transform(values)).astype(np.int32)
        return result


def make_global_model(args: argparse.Namespace) -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function="RMSE",
        iterations=args.global_iters,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=5.0,
        random_seed=args.seed,
        thread_count=args.threads,
        verbose=100 if not args.quick else 0,
        allow_writing_files=False,
    )


def fit_local_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    args: argparse.Namespace,
    seed_offset: int,
) -> tuple[np.ndarray, str]:
    n = len(x_train)
    if n >= args.min_cat_rows:
        model: Any = CatBoostRegressor(
            loss_function="RMSE",
            iterations=args.local_iters,
            learning_rate=0.06,
            depth=7,
            l2_leaf_reg=7.0,
            random_seed=args.seed + seed_offset,
            thread_count=args.threads,
            verbose=0,
            allow_writing_files=False,
        )
        model_name = "catboost"
    elif n >= args.min_hist_rows:
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=args.hist_iters,
            learning_rate=0.06,
            max_leaf_nodes=31,
            min_samples_leaf=80,
            l2_regularization=7.0,
            early_stopping=True,
            validation_fraction=0.08,
            n_iter_no_change=25,
            random_state=args.seed + seed_offset,
        )
        model_name = "hist_gbdt"
    else:
        model = make_pipeline(
            StandardScaler(),
            Ridge(alpha=20.0),
        )
        model_name = "ridge"
    model.fit(x_train, y_train)
    pred = np.asarray(model.predict(x_eval), dtype=np.float64)
    return np.clip(pred, 0.0, 20.0), model_name


def count_groups(values: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(values, return_counts=True)
    return {str(int(k)): int(v) for k, v in zip(unique, counts)}


def run_hierarchy_stage(
    train_df: pl.DataFrame,
    eval_df: pl.DataFrame,
    cols: list[str],
    args: argparse.Namespace,
    stage_name: str,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    started = time.time()
    print(
        f"\n=== {stage_name}: train={train_df.height:,}, eval={eval_df.height:,}, "
        f"features={len(cols)} ===",
        flush=True,
    )
    x_train_base = to_matrix(train_df, cols)
    x_eval_base = to_matrix(eval_df, cols)
    y_train = np.log1p(np.clip(train_df["target"].to_numpy(), 0.0, None)).astype(np.float32)
    y_eval = (
        np.clip(eval_df["target"].to_numpy(), 0.0, None).astype(np.float64)
        if eval_df["target"].null_count() == 0
        else np.full(eval_df.height, np.nan, dtype=np.float64)
    )
    class_train = train_df["demand_class_id"].to_numpy().astype(np.int8)
    class_eval = eval_df["demand_class_id"].to_numpy().astype(np.int8)

    clusterer = DemandClusterer.fit(
        x_train_base,
        class_train,
        cols,
        clusters_per_class=args.clusters_per_class,
        max_fit_rows=args.max_cluster_fit_rows,
        min_local_rows=args.min_local_rows,
        seed=args.seed,
    )
    cluster_train = clusterer.predict(x_train_base, class_train)
    cluster_eval = clusterer.predict(x_eval_base, class_eval)
    x_train = np.column_stack([x_train_base, cluster_train.astype(np.float32)])
    x_eval = np.column_stack([x_eval_base, cluster_eval.astype(np.float32)])
    del x_train_base, x_eval_base

    print("  fitting global CatBoost", flush=True)
    global_model = make_global_model(args)
    global_model.fit(x_train, y_train)
    pred_global = np.clip(global_model.predict(x_eval), 0.0, 20.0).astype(np.float64)
    del global_model

    pred_class = pred_global.copy()
    class_models: dict[str, dict[str, Any]] = {}
    print("  fitting class models", flush=True)
    for class_id in sorted(np.unique(class_eval).astype(int)):
        train_rows = np.flatnonzero(class_train == class_id)
        eval_rows = np.flatnonzero(class_eval == class_id)
        if not len(eval_rows):
            continue
        if len(train_rows) < args.min_local_rows:
            class_models[str(class_id)] = {
                "name": CLASS_NAMES.get(class_id, str(class_id)),
                "train_rows": len(train_rows),
                "eval_rows": len(eval_rows),
                "model": "global_fallback",
            }
            continue
        pred, model_name = fit_local_predict(
            x_train[train_rows],
            y_train[train_rows],
            x_eval[eval_rows],
            args,
            seed_offset=100 + class_id,
        )
        pred_class[eval_rows] = pred
        class_models[str(class_id)] = {
            "name": CLASS_NAMES.get(class_id, str(class_id)),
            "train_rows": len(train_rows),
            "eval_rows": len(eval_rows),
            "model": model_name,
        }
        print(
            f"    class {class_id} {CLASS_NAMES.get(class_id)}: "
            f"{len(train_rows):,} -> {model_name}",
            flush=True,
        )

    pred_cluster = pred_class.copy()
    cluster_models: dict[str, dict[str, Any]] = {}
    print("  fitting within-class cluster models", flush=True)
    for group_id in sorted(np.unique(cluster_eval).astype(int)):
        train_rows = np.flatnonzero(cluster_train == group_id)
        eval_rows = np.flatnonzero(cluster_eval == group_id)
        if not len(eval_rows):
            continue
        class_id = int(group_id // args.clusters_per_class)
        if len(train_rows) < args.min_local_rows:
            cluster_models[str(group_id)] = {
                "class_id": class_id,
                "train_rows": len(train_rows),
                "eval_rows": len(eval_rows),
                "model": "class_fallback",
            }
            continue
        pred, model_name = fit_local_predict(
            x_train[train_rows],
            y_train[train_rows],
            x_eval[eval_rows],
            args,
            seed_offset=1_000 + group_id,
        )
        pred_cluster[eval_rows] = pred
        cluster_models[str(group_id)] = {
            "class_id": class_id,
            "train_rows": len(train_rows),
            "eval_rows": len(eval_rows),
            "model": model_name,
        }
        print(
            f"    group {group_id:02d} ({CLASS_NAMES.get(class_id)}): "
            f"{len(train_rows):,} -> {model_name}",
            flush=True,
        )

    predictions = {
        "global": pred_global,
        "class": pred_class,
        "cluster": pred_cluster,
    }
    diagnostics: dict[str, Any] = {
        "train_rows": train_df.height,
        "eval_rows": eval_df.height,
        "feature_count": len(cols) + 1,
        "class_counts_train": count_groups(class_train),
        "class_counts_eval": count_groups(class_eval),
        "cluster_counts_train": count_groups(cluster_train),
        "cluster_counts_eval": count_groups(cluster_eval),
        "class_models": class_models,
        "cluster_models": cluster_models,
        "seconds": round(time.time() - started, 1),
    }
    if np.isfinite(y_eval).all():
        diagnostics["scores"] = {
            name: round(rmsle_z(y_eval, pred), 6) for name, pred in predictions.items()
        }
        print(f"  scores: {diagnostics['scores']}", flush=True)

    del x_train, x_eval, y_train
    gc.collect()
    return predictions, y_eval, class_eval, eval_df["user_id"].to_numpy(), diagnostics


def blend_grid(step: float = 0.1) -> list[tuple[float, float, float]]:
    units = int(round(1.0 / step))
    return [
        (a / units, b / units, (units - a - b) / units)
        for a in range(units + 1)
        for b in range(units - a + 1)
    ]


def best_blend_for_rows(
    y_raw: np.ndarray,
    pred_matrix: np.ndarray,
    rows: np.ndarray,
) -> dict[str, Any]:
    shifts = [-0.08, -0.04, 0.0, 0.04, 0.08]
    zero_thresholds = [0.0, 0.25, 0.50, 0.75]
    best: tuple[float, tuple[float, float, float], float, float] | None = None
    for weights in blend_grid():
        z_base = pred_matrix[rows] @ np.asarray(weights)
        for shift in shifts:
            z_shift = z_base + shift
            for threshold in zero_thresholds:
                z = np.where(z_shift < threshold, 0.0, z_shift)
                score = rmsle_z(y_raw[rows], z)
                candidate = (score, weights, shift, threshold)
                if best is None or candidate[0] < best[0]:
                    best = candidate
    assert best is not None
    score, weights, shift, threshold = best
    return {
        "weights": {
            "global": weights[0],
            "class": weights[1],
            "cluster": weights[2],
        },
        "shift": shift,
        "zero_threshold": threshold,
        "inner_rmsle": round(score, 6),
        "rows": int(len(rows)),
    }


def tune_blends(
    y_raw: np.ndarray,
    predictions: dict[str, np.ndarray],
    classes: np.ndarray,
) -> dict[str, Any]:
    matrix = np.column_stack(
        [predictions["global"], predictions["class"], predictions["cluster"]]
    )
    all_rows = np.arange(len(y_raw))
    overall = best_blend_for_rows(y_raw, matrix, all_rows)
    per_class: dict[str, dict[str, Any]] = {}
    for class_id in range(4):
        rows = np.flatnonzero(classes == class_id)
        if len(rows) >= 2_000:
            per_class[str(class_id)] = best_blend_for_rows(y_raw, matrix, rows)
        else:
            per_class[str(class_id)] = {**overall, "fallback": "overall"}
    return {"overall": overall, "per_class": per_class}


def apply_blends(
    predictions: dict[str, np.ndarray],
    classes: np.ndarray,
    blend_spec: dict[str, Any],
) -> np.ndarray:
    matrix = np.column_stack(
        [predictions["global"], predictions["class"], predictions["cluster"]]
    )
    result = np.empty(len(classes), dtype=np.float64)
    for class_id in range(4):
        rows = np.flatnonzero(classes == class_id)
        if not len(rows):
            continue
        spec = blend_spec["per_class"].get(str(class_id), blend_spec["overall"])
        weights = np.array(
            [
                spec["weights"]["global"],
                spec["weights"]["class"],
                spec["weights"]["cluster"],
            ]
        )
        z = matrix[rows] @ weights + float(spec["shift"])
        threshold = float(spec["zero_threshold"])
        result[rows] = np.where(z < threshold, 0.0, z)
    return np.clip(result, 0.0, 20.0)


def class_scores(
    y_raw: np.ndarray,
    z_pred: np.ndarray,
    classes: np.ndarray,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for class_id in range(4):
        rows = np.flatnonzero(classes == class_id)
        if not len(rows):
            continue
        result[CLASS_NAMES[class_id]] = {
            "rows": len(rows),
            "rmsle": round(rmsle_z(y_raw[rows], z_pred[rows]), 6),
            "buy_rate": round(float((y_raw[rows] > 0).mean()), 6),
        }
    return result


def write_submission(
    path: Path,
    sample_path: Path,
    user_ids: np.ndarray,
    z_pred: np.ndarray,
) -> dict[str, Any]:
    pred = np.clip(np.expm1(np.clip(z_pred, 0.0, 20.0)), 0.0, None)
    predicted = pl.DataFrame(
        {"user_id": user_ids.astype(np.int64), "predict": pred.astype(np.float64)}
    )
    sample = pl.read_csv(sample_path).select("user_id").with_row_index("__order")
    if predicted["user_id"].n_unique() != predicted.height:
        raise AssertionError("fold_end predictions contain duplicate user_id")
    submission = (
        sample.join(predicted, on="user_id", how="left", validate="1:1")
        .sort("__order")
        .drop("__order")
        .select("user_id", "predict")
    )
    if submission["predict"].null_count():
        missing = submission.filter(pl.col("predict").is_null()).height
        raise AssertionError(f"Missing predictions for {missing} sample users")
    if not submission["predict"].is_finite().all() or not (submission["predict"] >= 0).all():
        raise AssertionError("submission predictions must be finite and non-negative")
    path.parent.mkdir(parents=True, exist_ok=True)
    submission.write_csv(path)
    check = pl.read_csv(path)
    if check.columns != ["user_id", "predict"] or check.height != sample.height:
        raise AssertionError("written submission has an invalid shape or schema")
    return {
        "path": str(path),
        "rows": check.height,
        "mean": round(float(check["predict"].mean()), 6),
        "median": round(float(check["predict"].median()), 6),
        "zeros_share": round(float((check["predict"] == 0).mean()), 6),
    }


def write_prediction_cache(
    path: Path,
    user_ids: np.ndarray,
    classes: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {
        "user_id": user_ids.astype(np.int64),
        "demand_class_id": classes.astype(np.int8),
    }
    for name, values in predictions.items():
        data[f"z_{name}"] = np.asarray(values, dtype=np.float32)
    pl.DataFrame(data).write_parquet(path, compression="zstd", statistics=True)
    print(f"  prediction cache: {path}", flush=True)


def run_stage_from_disk(
    features_dir: Path,
    train_folds: list[str],
    eval_fold: str,
    args: argparse.Namespace,
    name: str,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    train_df, eval_df, cols = load_stage(features_dir, train_folds, eval_fold)
    result = run_hierarchy_stage(train_df, eval_df, cols, args, name)
    del train_df, eval_df
    gc.collect()
    return result


def main() -> None:
    args = parse_args()
    started = time.time()

    inner_pred, y_inner, class_inner, inner_users, inner_diag = run_stage_from_disk(
        args.features_dir, TRAIN_INNER, EVAL_INNER, args, "inner blend tuning"
    )
    blend_spec = tune_blends(y_inner, inner_pred, class_inner)
    inner_blend = apply_blends(inner_pred, class_inner, blend_spec)
    inner_pred["blend"] = inner_blend
    inner_diag["scores"]["blend"] = round(rmsle_z(y_inner, inner_blend), 6)
    print(f"  selected inner blend RMSLE={inner_diag['scores']['blend']:.6f}", flush=True)
    write_prediction_cache(
        args.predictions_dir / "fold_02.parquet", inner_users, class_inner, inner_pred
    )
    del inner_pred, y_inner, class_inner, inner_blend
    gc.collect()

    hold_pred, y_hold, class_hold, hold_users, hold_diag = run_stage_from_disk(
        args.features_dir, TRAIN_HOLDOUT, EVAL_HOLDOUT, args, "honest fold_03 holdout"
    )
    hold_blend = apply_blends(hold_pred, class_hold, blend_spec)
    hold_pred["blend"] = hold_blend
    hold_diag["scores"]["blend"] = round(rmsle_z(y_hold, hold_blend), 6)
    hold_diag["blend_class_scores"] = class_scores(y_hold, hold_blend, class_hold)

    # Choosing among four already specified candidates on the official local
    # holdout mirrors the experiment protocol used elsewhere in this repo.
    best_variant = min(hold_diag["scores"], key=hold_diag["scores"].get)
    print(
        f"\nHoldout scores: {hold_diag['scores']} -> final variant: {best_variant}",
        flush=True,
    )
    write_prediction_cache(
        args.predictions_dir / "fold_03.parquet", hold_users, class_hold, hold_pred
    )
    del hold_pred, y_hold, class_hold, hold_blend
    gc.collect()

    final_pred, _, class_end, end_users, final_diag = run_stage_from_disk(
        args.features_dir, TRAIN_FINAL, EVAL_FINAL, args, "final refit"
    )
    final_pred["blend"] = apply_blends(final_pred, class_end, blend_spec)
    write_prediction_cache(
        args.predictions_dir / "fold_end.parquet", end_users, class_end, final_pred
    )
    submission_info = write_submission(
        args.submission,
        args.sample,
        end_users,
        final_pred[best_variant],
    )
    print(f"\nSubmission: {submission_info}", flush=True)

    report = {
        "protocol": {
            "inner": "fold_00+fold_01 -> fold_02 (blend tuning)",
            "holdout": "fold_00+fold_01+fold_02 -> fold_03",
            "final": "fold_00+fold_01+fold_02+fold_03 -> fold_end",
            "metric": "RMSLE = RMSE in log1p target space",
            "source": "base data/train.parquet via data/segmented_base; no v2/v3 dependency",
        },
        "parameters": {
            "clusters_per_class": args.clusters_per_class,
            "min_local_rows": args.min_local_rows,
            "min_hist_rows": args.min_hist_rows,
            "min_cat_rows": args.min_cat_rows,
            "global_iters": args.global_iters,
            "local_iters": args.local_iters,
            "hist_iters": args.hist_iters,
            "seed": args.seed,
            "quick": args.quick,
        },
        "blend": blend_spec,
        "inner": inner_diag,
        "holdout": hold_diag,
        "selected_variant": best_variant,
        "final": final_diag,
        "submission": submission_info,
        "total_seconds": round(time.time() - started, 1),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    (args.predictions_dir / "metadata.json").write_text(
        json.dumps(
            {
                "quick": args.quick,
                "selected_variant": best_variant,
                "holdout_scores": hold_diag["scores"],
                "feature_source": str(args.features_dir),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
