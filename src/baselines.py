"""Naive baselines on fold_03 (exp00)."""

import json
from pathlib import Path

import numpy as np
import polars as pl


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    lt = np.log1p(np.clip(y_true, 0, None))
    lp = np.log1p(np.clip(y_pred, 0, None))
    return float(np.sqrt(np.mean((lt - lp) ** 2)))


def main() -> None:
    f3 = pl.read_parquet("data/v2/features/fold_03/batch_*.parquet")
    y = f3["target"].to_numpy()

    preds = {
        "naive_gmv30": f3["gmv_sum_30d"].to_numpy(),
        "zeros": np.zeros_like(y),
        "median": np.full_like(y, float(np.median(y))),
    }

    metrics = {name: rmsle(y, p) for name, p in preds.items()}
    for name, m in metrics.items():
        print(f"{name:>14}: RMSLE = {m:.5f}")

    Path("reports").mkdir(exist_ok=True)
    with open("reports/exp00_baselines.json", "w") as fh:
        json.dump(metrics, fh, indent=2)


if __name__ == "__main__":
    main()
