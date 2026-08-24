"""CatBoost training on extended features (exp02-lite).

Same protocol as exp01/train.py: train = fold_00..02, val = fold_03,
RMSE on z = log1p(target). Run from repo root:

    .venv/bin/python src/train_e2.py                 # default config
    .venv/bin/python src/train_e2.py --iters 2000 --lr 0.05 --depth 8
Writes reports/train_e2_metrics.json (appends by config key).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
from catboost import CatBoostRegressor, Pool

DROP_COLS = ["user_id", "target", "anchor_date"]
SEED = 42
FEAT_DIR = "data/v2/features_e2"
REPORT = Path("reports/train_e2_metrics.json")


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    lt = np.log1p(np.clip(y_true, 0, None))
    lp = np.log1p(np.clip(y_pred, 0, None))
    return float(np.sqrt(np.mean((lt - lp) ** 2)))


def load_fold(name: str) -> pl.DataFrame:
    return pl.read_parquet(f"{FEAT_DIR}/{name}/batch_*.parquet")


def xy(df: pl.DataFrame):
    X = df.drop([c for c in DROP_COLS if c in df.columns]).to_numpy()
    y = np.log1p(np.clip(df["target"].to_numpy(), 0, None))
    return X, y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--l2", type=float, default=3.0)
    ap.add_argument("--save-model", default=None)
    args = ap.parse_args()

    print("loading folds...", flush=True)
    train_df = pl.concat([load_fold(f"fold_{i:02d}") for i in range(3)], how="vertical")
    val_df = load_fold("fold_03")
    feature_names = [c for c in train_df.columns if c not in DROP_COLS]
    print(f"train: {train_df.shape}, val: {val_df.shape}, features: {len(feature_names)}")

    X_tr, y_tr = xy(train_df)
    X_val, _ = xy(val_df)
    y_val_raw = np.clip(val_df["target"].to_numpy(), 0, None)

    model = CatBoostRegressor(
        loss_function="RMSE", learning_rate=args.lr, depth=args.depth,
        l2_leaf_reg=args.l2, n_estimators=args.iters,
        thread_count=-1, random_seed=SEED, verbose=0,
    )
    t0 = time.time()
    model.fit(Pool(X_tr, label=y_tr, feature_names=feature_names))
    fit_time = time.time() - t0

    pred_raw = np.clip(np.expm1(model.predict(X_val)), 0, None)
    score = rmsle(y_val_raw, pred_raw)
    print(f"config iters={args.iters} lr={args.lr} depth={args.depth} l2={args.l2}: "
          f"RMSLE={score:.5f} ({fit_time:.0f}s)")

    imp = sorted(zip(feature_names, model.get_feature_importance()),
                 key=lambda kv: -kv[1])[:20]

    rec = {
        "rmsle_fold03": round(score, 5),
        "fit_time_sec": round(fit_time, 1),
        "params": {"n_estimators": args.iters, "learning_rate": args.lr,
                   "depth": args.depth, "l2_leaf_reg": args.l2},
        "top20_importance": [(n, round(v, 2)) for n, v in imp],
    }
    data = {}
    if REPORT.exists():
        data = json.loads(REPORT.read_text())
    data[f"i{args.iters}_lr{args.lr}_d{args.depth}_l2{args.l2}"] = rec
    REPORT.write_text(json.dumps(data, indent=2))

    if args.save_model:
        model.save_model(args.save_model)
    print("saved", REPORT)


if __name__ == "__main__":
    main()
