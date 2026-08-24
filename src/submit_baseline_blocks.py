"""Baseline (exp01 protocol) trained on the new feature datasets -> CSVs.

Same model/config as archive/exp01 (CatBoost RMSE on log1p(target), lr=0.05,
depth=8, l2_leaf_reg=3, 1000 iters, seed=42), four feature-block variants:

    base            — windowed aggregates (exp01 parity rebuild, sanity check)
    base+ext        — + intent/EWMA/trend/frequency block      (exp02)
    base+ext+pca    — + 32 PCA panel components                 (exp07)
    base+ext+pca+btyd — + BG/NBD & Gamma-Gamma features         (exp04)

Protocol per variant: train fold_00..02 -> score fold_03; refit on all four
folds -> predict fold_end -> submissions/submission_bl_<variant>.csv
(user_id,predict, sample_submit order). Metrics -> reports/bl_blocks.json.

Run from repo root:
    .venv/bin/python src/submit_baseline_blocks.py
"""

import json
import time
from pathlib import Path

import numpy as np
import polars as pl
from catboost import CatBoostRegressor

FEAT_DIR = Path("data/v2/features_ext")
BTYD_DIR = Path("data/v2/features_bgnbd")
REPORT_PATH = Path("reports/bl_blocks.json")

META_COLS = {"anchor_date", "user_id", "target"}
SEED = 42
PARAMS = dict(
    loss_function="RMSE",
    learning_rate=0.05,
    depth=8,
    l2_leaf_reg=3,
    n_estimators=1000,
    thread_count=-1,
    random_seed=SEED,
    verbose=0,
)

BTYD_FILL_ZERO = [
    "bgnbd_tx", "bgnbd_en30", "eb_lambda_n30", "bgnbd_e_gmv30", "eb_e_gmv30",
]


def load_fold(name: str) -> pl.DataFrame:
    feats = pl.read_parquet(FEAT_DIR / name / "batch_*.parquet")
    btyd = pl.read_parquet(BTYD_DIR / f"{name}.parquet")
    df = feats.join(btyd, on=["anchor_date", "user_id"], how="left")
    return df.with_columns([
        pl.col(c).fill_null(0.0) if c in BTYD_FILL_ZERO else pl.col(c).fill_null(-1.0)
        for c in btyd.columns if c not in ("anchor_date", "user_id")
    ])


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    lt = np.log1p(np.clip(y_true, 0, None))
    lp = np.log1p(np.clip(y_pred, 0, None))
    return float(np.sqrt(np.mean((lt - lp) ** 2)))


def main() -> None:
    t0 = time.time()
    print("loading folds...", flush=True)
    train_df = pl.concat([load_fold(f"fold_{i:02d}") for i in range(4)], how="vertical")
    val_df = load_fold("fold_03")
    end_df = load_fold("fold_end")
    print(f"train4: {train_df.shape}, val: {val_df.shape}, end: {end_df.shape} "
          f"(loaded in {time.time() - t0:.0f}s)")

    all_cols = [c for c in train_df.columns if c not in META_COLS]
    base_cols = [c for c in all_cols if not c.startswith(("x_", "pca_", "bgnbd_", "eb_"))]
    ext_cols = [c for c in all_cols if c.startswith("x_")]
    pca_cols = [c for c in all_cols if c.startswith("pca_")]
    btyd_cols = [c for c in all_cols if c.startswith(("bgnbd_", "eb_"))]
    variants = {
        "base": base_cols,
        "base+ext": base_cols + ext_cols,
        "base+ext+pca": base_cols + ext_cols + pca_cols,
        "base+ext+pca+btyd": base_cols + ext_cols + pca_cols + btyd_cols,
    }
    print({k: len(v) for k, v in variants.items()})

    def mat(df: pl.DataFrame) -> np.ndarray:
        return df.select(all_cols).to_numpy().astype(np.float32)

    M_tr, M_val, M_end = mat(train_df), mat(val_df), mat(end_df)
    col_pos = {c: i for i, c in enumerate(all_cols)}
    y_tr_log = np.log1p(np.clip(train_df["target"].to_numpy(), 0, None))
    y_val_raw = np.clip(val_df["target"].to_numpy(), 0, None)

    sample = pl.read_csv("sample_submit.csv")
    order = sample.select("user_id").with_row_index("__ord")
    Path("submissions").mkdir(exist_ok=True)

    results: dict[str, dict] = {}
    for name, cols in variants.items():
        idx = [col_pos[c] for c in cols]
        model = CatBoostRegressor(**PARAMS)

        tv = time.time()
        model.fit(M_tr[:, idx], y_tr_log)
        pred_val = np.clip(np.expm1(model.predict(M_val[:, idx])), 0, None)
        score = rmsle(y_val_raw, pred_val)
        t_fit = time.time() - tv

        imp = dict(sorted(zip(cols, model.get_feature_importance()),
                          key=lambda kv: -kv[1])[:12])
        results[name] = {
            "rmsle_fold03": round(score, 5),
            "n_features": len(cols),
            "fit_time_sec": round(t_fit, 1),
            "top_importance": {k: round(float(v), 2) for k, v in imp.items()},
        }
        print(f"{name:<20}: RMSLE(fold_03)={score:.5f} ({t_fit:.0f}s)", flush=True)

        out_path = Path(f"submissions/submission_bl_{name.replace('+', '_')}.csv")
        pred_end = np.clip(np.expm1(model.predict(M_end[:, idx])), 0, None)
        assert np.isfinite(pred_end).all() and (pred_end >= 0).all()
        sub = (
            pl.DataFrame({"user_id": end_df["user_id"].cast(pl.Int64), "predict": pred_end})
            .join(order, on="user_id", how="inner")
            .sort("__ord")
            .drop("__ord")
            .select(["user_id", "predict"])
        )
        assert set(sub["user_id"]) == set(sample["user_id"]), "user_id set mismatch"
        sub.write_csv(out_path)
        chk = pl.read_csv(out_path)
        assert chk.height == 250_000 and chk.columns == ["user_id", "predict"]
        assert chk["predict"].is_finite().all() and (chk["predict"] >= 0).all()
        results[name]["submission"] = str(out_path)
        results[name]["pred_mean"] = round(float(chk["predict"].mean()), 2)
        results[name]["pred_median"] = round(float(chk["predict"].median()), 2)
        print(f"{'':<20}  saved {out_path} (mean={results[name]['pred_mean']}, "
              f"median={results[name]['pred_median']})", flush=True)

    print("\nLEADERBOARD (RMSLE fold_03; refs: naive 2.19506, exp01 1.70261):")
    for k, v in sorted(results.items(), key=lambda kv: kv[1]["rmsle_fold03"]):
        print(f"  {v['rmsle_fold03']:.5f}  {k}")

    REPORT_PATH.write_text(json.dumps({
        "protocol": {**PARAMS, "seed": SEED,
                     "train": "fold_00..02 -> val fold_03; refit fold_00..03 -> fold_end"},
        "results": results,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nmetrics -> {REPORT_PATH}")
    print(f"ALL DONE in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
