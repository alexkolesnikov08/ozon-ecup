"""exp02 auxiliary check: is the YoY beta grid optimum interior?

Retrains the decision model (train fold_00..02) and evaluates the calibration
RMSLE on the proxy fold over an extended beta range [0, 1.5] to document
whether the coarse-grid argmin at beta=1.0 sits on the boundary.
Read-only with respect to all caches; prints a table, writes nothing.
"""

import json
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from catboost import CatBoostRegressor

SEED = 42
DROP_COLS = ["anchor_date", "user_id", "target"]
CV_FOLDS = ["fold_00", "fold_01", "fold_02"]

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
}


def load(name: str, extra_cols: list[str]) -> pl.DataFrame:
    base_dir = Path(f"data/v2/features/{name}")
    if not base_dir.exists():
        base_dir = Path(f"data/v2/features_exp02/{name}_base")
    base = pl.read_parquet(str(base_dir / "batch_*.parquet"))
    if not extra_cols:
        return base
    extra = pl.read_parquet(f"data/v2/features_exp02/{name}/batch_*.parquet")
    j = base.join(extra.select(["user_id", *extra_cols]), on="user_id", how="inner")
    assert j.height == base.height == 250_000
    return j


def main() -> None:
    dec = json.loads(Path("reports/exp02_metrics.json").read_text())["decision"]
    extra_cols = [f for b in dec["blocks"] for f in BLOCKS[b]]
    feat_cols = BASE_FEATURES + extra_cols

    tr = pl.concat([load(f, extra_cols) for f in CV_FOLDS], how="vertical")
    px = load("fold_proxy", extra_cols)
    X_tr = tr.select(feat_cols).to_numpy()
    y_tr = np.log1p(np.clip(tr["target"].to_numpy(), 0, None))
    X_px = px.select(feat_cols).to_numpy()
    z_true = np.log1p(np.clip(px["target"].to_numpy(), 0, None))

    model = CatBoostRegressor(
        loss_function=dec["loss"], learning_rate=0.05, depth=8, l2_leaf_reg=3,
        n_estimators=dec["n_estimators"], thread_count=-1, random_seed=SEED,
        verbose=0, allow_writing_files=False,
    )
    model.fit(X_tr, y_tr)
    z_pred = model.predict(X_px)

    raw = pl.read_parquet("data/train.parquet", columns=["event_date", "gmv"])
    num = raw.filter(pl.col("event_date").is_between(
        date(2025, 2, 14), date(2025, 3, 15)))["gmv"].sum()
    den = raw.filter(pl.col("event_date").is_between(
        date(2025, 1, 15), date(2025, 2, 13)))["gmv"].sum()
    ln_s = float(np.log(num / den))

    print(f"{'beta':>6} {'proxy RMSLE':>12}")
    best = (None, float("inf"))
    for b in np.arange(0.0, 1.5001, 0.1):
        s = float(np.sqrt(np.mean((z_true - (z_pred + b * ln_s)) ** 2)))
        print(f"{b:>6.2f} {s:>12.5f}")
        if s < best[1]:
            best = (float(b), s)
    print(f"\nextended-grid argmin: beta={best[0]:.2f} RMSLE={best[1]:.5f} "
          f"(coarse-grid choice was beta={dec['beta']})")


if __name__ == "__main__":
    main()
