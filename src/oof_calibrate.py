"""Level calibration study (exp06): OOF z-shift x isotonic x zero-threshold.

Protocol:
- leave-one-fold-out OOF: for each CV fold k train CatBoost (parity config,
  RMSE on log1p target) on the remaining three folds, predict z on fold k;
- on pooled OOF fit: (a) scalar shift c = mean(z_true - z_pred),
  (b) isotonic regression z_pred -> z_true;
- zero-thresholding grid applied in z-space on top of each base variant;
- report RMSLE (raw space, expm1) per fold + pooled for every variant.

Run from repo root:  .venv/bin/python src/oof_calibrate.py [--features e2|base]
Writes reports/oof_calibration.json and reports/oof_preds.parquet.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
from catboost import CatBoostRegressor, Pool
from sklearn.isotonic import IsotonicRegression

DROP_COLS = ["anchor_date", "user_id", "target"]
SEED = 42
FOLDS = [f"fold_{i:02d}" for i in range(4)]
PARAMS = dict(loss_function="RMSE", learning_rate=0.05, depth=8,
              l2_leaf_reg=3, n_estimators=1000, thread_count=-1,
              random_seed=SEED, verbose=0)
TAUS = np.round(np.arange(0.0, 0.55, 0.05), 2)


def rmsle_raw(y_true: np.ndarray, z_pred: np.ndarray) -> float:
    lt = np.log1p(np.clip(y_true, 0, None))
    lp = np.clip(z_pred, 0, None)
    return float(np.sqrt(np.mean((lt - lp) ** 2)))


def load_folds(feat_dir: str):
    def one(name):
        return pl.read_parquet(f"data/v2/{feat_dir}/{name}/batch_*.parquet")
    return {f: one(f) for f in FOLDS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features_e2",
                    choices=["features_e2", "features"])
    args = ap.parse_args()
    feat_dir = args.features

    print(f"loading feature set: {feat_dir}", flush=True)
    folds = load_folds(feat_dir)
    feature_names = [c for c in folds[FOLDS[0]].columns if c not in DROP_COLS]

    # ---- pass 1: LOO OOF predictions ------------------------------------
    preds = {}
    t00 = time.time()
    for held_out in FOLDS:
        tr = pl.concat([folds[f] for f in FOLDS if f != held_out], how="vertical")
        va = folds[held_out]
        X_tr = tr.drop(DROP_COLS).to_numpy()
        y_tr = np.log1p(np.clip(tr["target"].to_numpy(), 0, None))
        X_va = va.drop(DROP_COLS).to_numpy()
        model = CatBoostRegressor(**PARAMS)
        t0 = time.time()
        model.fit(Pool(X_tr, label=y_tr, feature_names=feature_names))
        z = model.predict(X_va)
        rmsle = rmsle_raw(np.clip(va["target"].to_numpy(), 0, None), z)
        preds[held_out] = pl.DataFrame({
            "user_id": va["user_id"],
            "anchor": held_out,
            "z_pred": z,
            "y_true": np.clip(va["target"].to_numpy(), 0, None),
        })
        print(f"  OOF {held_out}: RMSLE={rmsle:.5f} ({time.time() - t0:.0f}s)", flush=True)
        del model, X_tr, y_tr, X_va
    print(f"OOF fits done in {(time.time() - t00) / 60:.1f} min")

    oof = pl.concat(preds.values(), how="vertical")
    Path("reports").mkdir(exist_ok=True)
    oof.write_parquet("reports/oof_preds.parquet")

    z_p = oof["z_pred"].to_numpy()
    y = oof["y_true"].to_numpy()
    z_t = np.log1p(y)
    c_shift = float(np.mean(z_t - z_p))

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(z_p, z_t)

    variants: dict[str, np.ndarray] = {
        "raw": z_p,
        "shift": z_p + c_shift,
        "isotonic": iso.predict(z_p),
    }
    for tau in TAUS:
        for base_name in ("raw", "shift", "isotonic"):
            zv = np.where(variants[base_name] < tau, 0.0, variants[base_name])
            variants[f"{base_name}+thr{tau:.2f}"] = zv

    rows = []
    for name, zv in variants.items():
        rec = {"variant": name}
        for f in FOLDS:
            m = (oof["anchor"] == f).to_numpy()
            rec[f] = round(rmsle_raw(y[m], zv[m]), 5)
        rec["pooled"] = round(rmsle_raw(y, zv), 5)
        rows.append(rec)

    res = pl.DataFrame(rows).sort("pooled")
    print("\n=== CALIBRATION LEADERBOARD (top 15) ===")
    with pl.Config(tbl_rows=16, tbl_cols=8, tbl_width_chars=200):
        print(res.head(15))
    print(f"\nscalar shift c = {c_shift:+.4f}")

    out = {
        "feature_set": feat_dir,
        "protocol": "leave-one-fold-out CatBoost, RMSE on log1p",
        "params": PARAMS,
        "scalar_shift_c": round(c_shift, 5),
        "leaderboard": res.to_dicts(),
    }
    with open(f"reports/oof_calibration_{feat_dir}.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"saved reports/oof_calibration_{feat_dir}.json")


if __name__ == "__main__":
    main()
