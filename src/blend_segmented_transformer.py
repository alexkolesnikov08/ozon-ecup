"""Blend the classical segmented hierarchy with the weekly Transformer.

Weights are selected on fold_02, compared honestly on fold_03, and then applied
to fold_end.  All operations are performed in z=log1p(target) space.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


CLASSICAL_DIR = Path("data/segmented_predictions/classical")
TRANSFORMER_DIR = Path("data/segmented_predictions/transformer")
FEATURES_DIR = Path("data/segmented_base")
SAMPLE_PATH = Path("sample_submit.csv")
SUBMISSION_PATH = Path("submissions/submission_segmented_transformer.csv")
REPORT_PATH = Path("reports/segmented_transformer_blend.json")

CLASS_NAMES = {
    0: "smooth",
    1: "erratic",
    2: "intermittent",
    3: "lumpy",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Blend classical and Transformer branches.")
    ap.add_argument("--classical-dir", type=Path, default=CLASSICAL_DIR)
    ap.add_argument("--transformer-dir", type=Path, default=TRANSFORMER_DIR)
    ap.add_argument("--features-dir", type=Path, default=FEATURES_DIR)
    ap.add_argument("--sample", type=Path, default=SAMPLE_PATH)
    ap.add_argument("--submission", type=Path, default=SUBMISSION_PATH)
    ap.add_argument("--report", type=Path, default=REPORT_PATH)
    return ap.parse_args()


def rmsle_z(y_raw: np.ndarray, z_pred: np.ndarray) -> float:
    z_true = np.log1p(np.clip(y_raw.astype(np.float64), 0.0, None))
    z_safe = np.clip(np.asarray(z_pred, dtype=np.float64), 0.0, 20.0)
    return float(np.sqrt(np.mean((z_true - z_safe) ** 2)))


def read_metadata(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def load_prediction_fold(args: argparse.Namespace, fold: str) -> pl.DataFrame:
    classical_path = args.classical_dir / f"{fold}.parquet"
    transformer_path = args.transformer_dir / f"{fold}.parquet"
    feature_path = args.features_dir / f"{fold}.parquet"
    missing = [str(p) for p in [classical_path, transformer_path, feature_path] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing prediction inputs: " + ", ".join(missing))

    classical = pl.read_parquet(classical_path).select(
        "user_id", "demand_class_id", "z_blend"
    )
    transformer = pl.read_parquet(transformer_path).select("user_id", "z_transformer")
    target = pl.read_parquet(feature_path, columns=["user_id", "target"])
    if classical["user_id"].n_unique() != classical.height:
        raise ValueError(f"{classical_path}: duplicate user_id")
    if transformer["user_id"].n_unique() != transformer.height:
        raise ValueError(f"{transformer_path}: duplicate user_id")
    joined = (
        classical.join(transformer, on="user_id", how="inner", validate="1:1")
        .join(target, on="user_id", how="inner", validate="1:1")
    )
    if joined.height != classical.height or joined.height != transformer.height:
        raise ValueError(f"{fold}: classical/Transformer user sets differ")
    values = joined.select("z_blend", "z_transformer").to_numpy()
    if not np.isfinite(values).all():
        raise ValueError(f"{fold}: non-finite predictions")
    return joined


def tune_one(
    y_raw: np.ndarray,
    z_classical: np.ndarray,
    z_transformer: np.ndarray,
    rows: np.ndarray,
) -> dict[str, Any]:
    best: tuple[float, float, float, float] | None = None
    for transformer_weight in np.linspace(0.0, 1.0, 21):
        z_base = (
            (1.0 - transformer_weight) * z_classical[rows]
            + transformer_weight * z_transformer[rows]
        )
        for shift in [-0.08, -0.04, 0.0, 0.04, 0.08]:
            z_shift = z_base + shift
            for zero_threshold in [0.0, 0.25, 0.50, 0.75]:
                z = np.where(z_shift < zero_threshold, 0.0, z_shift)
                score = rmsle_z(y_raw[rows], z)
                candidate = (score, float(transformer_weight), shift, zero_threshold)
                if best is None or candidate[0] < best[0]:
                    best = candidate
    assert best is not None
    return {
        "transformer_weight": best[1],
        "classical_weight": 1.0 - best[1],
        "shift": best[2],
        "zero_threshold": best[3],
        "inner_rmsle": round(best[0], 6),
        "rows": len(rows),
    }


def tune(
    y_raw: np.ndarray,
    z_classical: np.ndarray,
    z_transformer: np.ndarray,
    classes: np.ndarray,
) -> dict[str, Any]:
    overall = tune_one(
        y_raw, z_classical, z_transformer, np.arange(len(y_raw), dtype=np.int64)
    )
    per_class: dict[str, dict[str, Any]] = {}
    for class_id in range(4):
        rows = np.flatnonzero(classes == class_id)
        if len(rows) >= 2_000:
            per_class[str(class_id)] = tune_one(
                y_raw, z_classical, z_transformer, rows
            )
        else:
            per_class[str(class_id)] = {**overall, "fallback": "overall"}
    return {"overall": overall, "per_class": per_class}


def apply(
    z_classical: np.ndarray,
    z_transformer: np.ndarray,
    classes: np.ndarray,
    spec: dict[str, Any],
) -> np.ndarray:
    result = np.empty(len(classes), dtype=np.float64)
    for class_id in range(4):
        rows = np.flatnonzero(classes == class_id)
        if not len(rows):
            continue
        params = spec["per_class"].get(str(class_id), spec["overall"])
        z = (
            params["classical_weight"] * z_classical[rows]
            + params["transformer_weight"] * z_transformer[rows]
            + params["shift"]
        )
        result[rows] = np.where(z < params["zero_threshold"], 0.0, z)
    return np.clip(result, 0.0, 20.0)


def component_arrays(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        df["z_blend"].to_numpy().astype(np.float64),
        df["z_transformer"].to_numpy().astype(np.float64),
        df["demand_class_id"].to_numpy().astype(np.int8),
    )


def score_by_class(
    y: np.ndarray, z: np.ndarray, classes: np.ndarray
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for class_id, name in CLASS_NAMES.items():
        rows = np.flatnonzero(classes == class_id)
        if len(rows):
            result[name] = {
                "rows": len(rows),
                "rmsle": round(rmsle_z(y[rows], z[rows]), 6),
            }
    return result


def write_submission(
    path: Path,
    sample_path: Path,
    users: np.ndarray,
    z_pred: np.ndarray,
) -> dict[str, Any]:
    predicted = pl.DataFrame(
        {
            "user_id": users.astype(np.int64),
            "predict": np.clip(np.expm1(np.clip(z_pred, 0.0, 20.0)), 0.0, None),
        }
    )
    sample = pl.read_csv(sample_path).select("user_id").with_row_index("__order")
    out = (
        sample.join(predicted, on="user_id", how="left", validate="1:1")
        .sort("__order")
        .drop("__order")
        .select("user_id", "predict")
    )
    if out["predict"].null_count() or not out["predict"].is_finite().all():
        raise ValueError("Invalid or missing final predictions")
    if not (out["predict"] >= 0).all() or out.height != sample.height:
        raise ValueError("Invalid submission shape or negative predictions")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(path)
    return {
        "path": str(path),
        "rows": out.height,
        "mean": round(float(out["predict"].mean()), 6),
        "median": round(float(out["predict"].median()), 6),
        "zeros_share": round(float((out["predict"] == 0).mean()), 6),
    }


def main() -> None:
    args = parse_args()
    classical_meta = read_metadata(args.classical_dir / "metadata.json")
    transformer_meta = read_metadata(args.transformer_dir / "metadata.json")
    if classical_meta.get("quick") or transformer_meta.get("quick"):
        print("WARNING: at least one prediction branch was trained with --quick.")

    inner = load_prediction_fold(args, "fold_02")
    zc_inner, zt_inner, class_inner = component_arrays(inner)
    y_inner = inner["target"].to_numpy().astype(np.float64)
    spec = tune(y_inner, zc_inner, zt_inner, class_inner)
    z_inner = apply(zc_inner, zt_inner, class_inner, spec)
    inner_scores = {
        "classical": round(rmsle_z(y_inner, zc_inner), 6),
        "transformer": round(rmsle_z(y_inner, zt_inner), 6),
        "blend": round(rmsle_z(y_inner, z_inner), 6),
    }

    holdout = load_prediction_fold(args, "fold_03")
    zc_hold, zt_hold, class_hold = component_arrays(holdout)
    y_hold = holdout["target"].to_numpy().astype(np.float64)
    z_hold = apply(zc_hold, zt_hold, class_hold, spec)
    holdout_scores = {
        "classical": round(rmsle_z(y_hold, zc_hold), 6),
        "transformer": round(rmsle_z(y_hold, zt_hold), 6),
        "blend": round(rmsle_z(y_hold, z_hold), 6),
    }
    selected = min(holdout_scores, key=holdout_scores.get)
    print(f"inner: {inner_scores}")
    print(f"holdout: {holdout_scores} -> {selected}")

    final = load_prediction_fold(args, "fold_end")
    zc_end, zt_end, class_end = component_arrays(final)
    z_end_blend = apply(zc_end, zt_end, class_end, spec)
    candidates = {
        "classical": zc_end,
        "transformer": zt_end,
        "blend": z_end_blend,
    }
    submission = write_submission(
        args.submission,
        args.sample,
        final["user_id"].to_numpy(),
        candidates[selected],
    )

    report = {
        "protocol": {
            "tuning": "fold_02",
            "honest_holdout": "fold_03",
            "final": "fold_end",
            "space": "z=log1p(target)",
        },
        "branch_metadata": {
            "classical": classical_meta,
            "transformer": transformer_meta,
        },
        "weights": spec,
        "inner_scores": inner_scores,
        "holdout_scores": holdout_scores,
        "holdout_blend_by_class": score_by_class(y_hold, z_hold, class_hold),
        "selected_variant": selected,
        "submission": submission,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Submission: {submission}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
