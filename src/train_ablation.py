"""Hyperparameter / iteration ablation on exp01 base features.

Train = fold_00..02, validation = fold_03 (same protocol as exp01).
All configs use quadratic loss on z = log1p(target) (RMSLE-equivalent),
varying iterations / lr / depth / regularization / bootstrap.

Run from repo root:  .venv/bin/python src/train_ablation.py
Writes reports/train_ablation.json
"""

import json
import time
from pathlib import Path

import numpy as np
import polars as pl
from catboost import CatBoostRegressor, Pool

DROP_COLS = ["anchor_date", "user_id", "target"]
SEED = 42

CONFIGS = [
    {"name": "parity_d8_lr05_1000", "n_estimators": 1000, "learning_rate": 0.05,
     "depth": 8, "l2_leaf_reg": 3},
    {"name": "d8_lr05_2000", "n_estimators": 2000, "learning_rate": 0.05,
     "depth": 8, "l2_leaf_reg": 3},
    {"name": "d8_lr05_3000", "n_estimators": 3000, "learning_rate": 0.05,
     "depth": 8, "l2_leaf_reg": 3},
    {"name": "d8_lr05_4000_od200", "n_estimators": 4000, "learning_rate": 0.05,
     "depth": 8, "l2_leaf_reg": 3, "early_stopping_rounds": 200},
    {"name": "d8_lr03_3000", "n_estimators": 3000, "learning_rate": 0.03,
     "depth": 8, "l2_leaf_reg": 3},
    {"name": "d8_lr03_5000_od300", "n_estimators": 5000, "learning_rate": 0.03,
     "depth": 8, "l2_leaf_reg": 3, "early_stopping_rounds": 300},
    {"name": "d6_lr05_2000", "n_estimators": 2000, "learning_rate": 0.05,
     "depth": 6, "l2_leaf_reg": 3},
    {"name": "d10_lr05_2000", "n_estimators": 2000, "learning_rate": 0.05,
     "depth": 10, "l2_leaf_reg": 3},
    {"name": "d8_lr05_2000_l2reg10", "n_estimators": 2000, "learning_rate": 0.05,
     "depth": 8, "l2_leaf_reg": 10},
    {"name": "d8_lr05_2000_bern08", "n_estimators": 2000, "learning_rate": 0.05,
     "depth": 8, "l2_leaf_reg": 3, "bootstrap_type": "Bernoulli", "subsample": 0.8},
]


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    lt = np.log1p(np.clip(y_true, 0, None))
    lp = np.log1p(np.clip(y_pred, 0, None))
    return float(np.sqrt(np.mean((lt - lp) ** 2)))


def load_fold(name: str) -> pl.DataFrame:
    return pl.read_parquet(f"data/v2/features/{name}/batch_*.parquet")


def xy(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = df.drop(DROP_COLS).to_numpy()
    y = np.log1p(np.clip(df["target"].to_numpy(), 0, None))
    return X, y


def main() -> None:
    print("loading folds...", flush=True)
    train_df = pl.concat([load_fold(f"fold_{i:02d}") for i in range(3)], how="vertical")
    val_df = load_fold("fold_03")
    feature_names = [c for c in train_df.columns if c not in DROP_COLS]
    print(f"train: {train_df.shape}, val: {val_df.shape}, features: {len(feature_names)}")

    X_tr, y_tr = xy(train_df)
    X_val, _ = xy(val_df)
    y_val_raw = np.clip(val_df["target"].to_numpy(), 0, None)

    val_pool = Pool(X_val, label=np.log1p(y_val_raw), feature_names=feature_names)

    results = {}
    for cfg in CONFIGS:
        name = cfg["name"]
        es = cfg.pop("early_stopping_rounds", None)
        params = {
            "loss_function": "RMSE",
            "thread_count": -1,
            "random_seed": SEED,
            "verbose": 0,
            **{k: v for k, v in cfg.items() if k != "name"},
        }
        model = CatBoostRegressor(**params)
        t0 = time.time()
        if es is not None:
            model.fit(Pool(X_tr, label=y_tr, feature_names=feature_names),
                      eval_set=val_pool, early_stopping_rounds=es)
        else:
            model.fit(Pool(X_tr, label=y_tr, feature_names=feature_names))
        fit_time = time.time() - t0

        pred_raw = np.clip(np.expm1(model.predict(X_val)), 0, None)
        score = rmsle(y_val_raw, pred_raw)
        n_used = model.get_best_iteration() + 1 if es is not None else params["n_estimators"]
        results[name] = {
            "params": {k: v for k, v in params.items()},
            "rmsle_fold03": round(score, 5),
            "fit_time_sec": round(fit_time, 1),
            "iterations_used": int(n_used),
        }
        print(f"{name:>26}: RMSLE={score:.5f}  iters={n_used}  {fit_time:.0f}s", flush=True)
        del model

    ranked = sorted(results.items(), key=lambda kv: kv[1]["rmsle_fold03"])
    print("\n=== LEADERBOARD ===")
    for name, r in ranked:
        print(f"{r['rmsle_fold03']:.5f}  {name}")

    Path("reports").mkdir(exist_ok=True)
    with open("reports/train_ablation.json", "w") as fh:
        json.dump(
            {
                "protocol": "train fold_00..02 -> val fold_03, RMSE on log1p(target)",
                "seed": SEED,
                "results": results,
                "best": {"name": ranked[0][0], **ranked[0][1]},
            },
            fh,
            indent=2,
        )
    print("saved reports/train_ablation.json")


if __name__ == "__main__":
    main()
