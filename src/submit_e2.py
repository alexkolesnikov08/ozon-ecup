"""Final exp02-lite submission: refit best config on all 4 CV folds,
predict fold_end, optionally apply level calibration from OOF study.

Run from repo root:
    .venv/bin/python src/submit_e2.py --iters 1000 --shift 0.0
Writes submissions/submission_exp02.csv (and _cal variant if shift != 0).
"""

import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl
from catboost import CatBoostRegressor, Pool

from train_e2 import DROP_COLS, FEAT_DIR, SEED, load_fold, rmsle, xy

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--l2", type=float, default=3.0)
    ap.add_argument("--shift", type=float, default=0.0,
                    help="additive z-space calibration from OOF study")
    args = ap.parse_args()

    print("loading folds...", flush=True)
    train_df = pl.concat([load_fold(f"fold_{i:02d}") for i in range(4)], how="vertical")
    end_df = load_fold("fold_end")
    feature_names = [c for c in train_df.columns if c not in DROP_COLS]
    print(f"train: {train_df.shape}, fold_end: {end_df.shape}")

    X_tr, y_tr = xy(train_df)
    model = CatBoostRegressor(
        loss_function="RMSE", learning_rate=args.lr, depth=args.depth,
        l2_leaf_reg=args.l2, n_estimators=args.iters,
        thread_count=-1, random_seed=SEED, verbose=0,
    )
    t0 = time.time()
    model.fit(Pool(X_tr, label=y_tr, feature_names=feature_names))
    print(f"final model ({args.iters} iters) trained in {time.time() - t0:.1f}s")

    # honest reference: refit model on holdout fold_03
    val_df = load_fold("fold_03")
    X_val, _ = xy(val_df)
    y_val_raw = np.clip(val_df["target"].to_numpy(), 0, None)
    score = rmsle(y_val_raw, np.clip(np.expm1(model.predict(X_val)), 0, None))
    print(f"refit-model RMSLE on fold_03: {score:.5f}")

    z_end = model.predict(end_df.drop(DROP_COLS).to_numpy())
    pred = np.clip(np.expm1(z_end), 0, None).astype(np.float64)

    sample = pl.read_csv("sample_submit.csv")

    def write_sub(name: str, values: np.ndarray) -> None:
        sub = (
            pl.DataFrame({"user_id": end_df["user_id"], "predict": values})
            .join(sample.select("user_id"), on="user_id", how="semi")
        )
        assert set(sub["user_id"]) == set(sample["user_id"]), "user_id set mismatch"
        sub = sub.join(sample.select("user_id").with_row_index("__ord"), on="user_id")
        sub = sub.sort("__ord").drop("__ord")
        Path("submissions").mkdir(exist_ok=True)
        out = Path(f"submissions/{name}")
        sub.write_csv(out)
        chk = pl.read_csv(out)
        assert chk.height == 250_000 and chk.columns == ["user_id", "predict"]
        assert chk["predict"].is_finite().all() and (chk["predict"] >= 0).all()
        print(f"saved {out} (mean={values.mean():.2f}, median={np.median(values):.4f})")

    tag = f"i{args.iters}_lr{args.lr}_d{args.depth}"
    write_sub(f"submission_exp02_{tag}.csv", pred)

    if args.shift != 0.0:
        pred_cal = np.clip(np.expm1(z_end + args.shift), 0, None).astype(np.float64)
        write_sub(f"submission_exp02_{tag}_cal.csv", pred_cal)


if __name__ == "__main__":
    main()
