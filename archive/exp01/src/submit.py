"""Final model on all 4 CV folds -> predict fold_end -> submission_exp01.csv."""

import json
import time
from pathlib import Path

import numpy as np
import polars as pl
from catboost import CatBoostRegressor, Pool

from train import DROP_COLS, SEED, load_fold, rmsle, xy

BEST_N_ESTIMATORS = 1000


def main() -> None:
    print("loading folds...", flush=True)
    train_df = pl.concat(
        [load_fold(f"fold_{i:02d}") for i in range(4)], how="vertical"
    )
    end_df = load_fold("fold_end")
    feature_names = [c for c in train_df.columns if c not in DROP_COLS]
    print(f"train: {train_df.shape}, fold_end: {end_df.shape}")

    X_tr, y_tr = xy(train_df)
    X_end = end_df.drop(DROP_COLS).to_numpy()

    model = CatBoostRegressor(
        loss_function="RMSE",
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3,
        n_estimators=BEST_N_ESTIMATORS,
        thread_count=-1,
        random_seed=SEED,
        verbose=0,
    )
    t0 = time.time()
    model.fit(Pool(X_tr, label=y_tr, feature_names=feature_names))
    fit_time = time.time() - t0
    print(f"final model ({BEST_N_ESTIMATORS} iters) trained in {fit_time:.1f}s")

    # honest check: same config scored on holdout fold_03 with this exact seed
    val_df = load_fold("fold_03")
    X_val, _ = xy(val_df)
    y_val_raw = np.clip(val_df["target"].to_numpy(), 0, None)
    score = rmsle(y_val_raw, np.clip(np.expm1(model.predict(X_val)), 0, None))
    print(f"refit-model RMSLE on fold_03: {score:.5f}")

    pred = np.clip(np.expm1(model.predict(X_end)), 0, None).astype(np.float64)
    assert not np.isnan(pred).any() and (pred >= 0).all()

    sample = pl.read_csv("sample_submit.csv")
    sub = (
        pl.DataFrame({"user_id": end_df["user_id"], "predict": pred})
        .join(sample.select("user_id"), on="user_id", how="semi")
    )
    assert set(sub["user_id"]) == set(sample["user_id"]), "user_id set mismatch"
    sub = sub.join(sample.select("user_id").with_row_index("__ord"), on="user_id")
    sub = sub.sort("__ord").drop("__ord")

    Path("submissions").mkdir(exist_ok=True)
    out = Path("submissions/submission_exp01.csv")
    sub.write_csv(out)

    chk = pl.read_csv(out)
    assert chk.height == 250_000 and chk.width == 2
    assert chk.columns == ["user_id", "predict"]
    assert chk["predict"].is_finite().all() and (chk["predict"] >= 0).all()
    print(f"submission saved: {out} ({chk.height} rows, "
          f"pred mean={chk['predict'].mean():.2f}, median={chk['predict'].median():.4f})")


if __name__ == "__main__":
    main()
