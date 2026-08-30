"""Measure classification accuracy (buyer / non-buyer) of the RMSLE-trained model.

Uses honest leave-one-fold-out OOF predictions (CatBoost, RMSE on z=log1p(target),
parity config with exp02 features). Reports per-fold and pooled:
- RMSLE (the competition metric)
- binary accuracy of "y>0" prediction at several z-thresholds
- AUC / precision / recall at best-accuracy threshold
- reference accuracies: predict-all-buy / predict-all-zero / predict-by-mean-z

Run from repo root:  .venv/bin/python src/exp_accuracy.py
Writes reports/accuracy_oof.json.
"""

import json
import time
from pathlib import Path

import numpy as np
import polars as pl
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

DROP_COLS = ["anchor_date", "user_id", "target"]
SEED = 42
FOLDS = [f"fold_{i:02d}" for i in range(4)]
FEAT_DIR = "features_exp02"
PARAMS = dict(loss_function="RMSE", learning_rate=0.05, depth=8,
              l2_leaf_reg=3, n_estimators=1000, thread_count=-1,
              random_seed=SEED, verbose=0)
THRESHOLDS = np.round(np.arange(0.0, 3.05, 0.25), 2)


def rmsle_raw(y_true: np.ndarray, z_pred: np.ndarray) -> float:
    lt = np.log1p(np.clip(y_true, 0, None))
    lp = np.clip(z_pred, 0, None)
    return float(np.sqrt(np.mean((lt - lp) ** 2)))


def load_folds():
    def one(name):
        return pl.read_parquet(f"data/v2/{FEAT_DIR}/{name}/batch_*.parquet")
    return {f: one(f) for f in FOLDS}


def main() -> None:
    folds = load_folds()
    feature_names = [c for c in folds[FOLDS[0]].columns if c not in DROP_COLS]

    preds = {}
    t00 = time.time()
    for held_out in FOLDS:
        tr = pl.concat([folds[f] for f in FOLDS if f != held_out], how="vertical")
        va = folds[held_out]
        model = CatBoostRegressor(**PARAMS)
        model.fit(
            Pool(tr.drop(DROP_COLS).to_numpy(),
                 label=np.log1p(np.clip(tr["target"].to_numpy(), 0, None)),
                 feature_names=feature_names),
            silent=True,
        )
        z = model.predict(va.drop(DROP_COLS).to_numpy())
        y = np.clip(va["target"].to_numpy(), 0, None)
        preds[held_out] = pl.DataFrame({
            "user_id": va["user_id"], "anchor": held_out,
            "z_pred": z, "y_true": y,
        })
        print(f"  OOF {held_out}: RMSLE={rmsle_raw(y, z):.5f} ({time.time() - t00:.0f}s)", flush=True)
    print(f"OOF done in {(time.time() - t00) / 60:.1f} min")

    oof = pl.concat(preds.values(), how="vertical")
    z_p = oof["z_pred"].to_numpy()
    y = oof["y_true"].to_numpy()
    yb = (y > 0).astype(int)
    p_buy = yb.mean()

    rows = []
    for name, m in ({"pooled": np.ones_like(yb, dtype=bool)} | {
        f: (oof["anchor"] == f).to_numpy() for f in FOLDS
    }).items():
        r = {"scope": name, "n": int(m.sum())}
        r["p_buy_true"] = round(float(yb[m].mean()), 5)
        r["rmsle"] = round(rmsle_raw(y[m], z_p[m]), 5)
        r["auc_buy"] = round(float(roc_auc_score(yb[m], z_p[m])), 5)
        r["acc_all_buy"] = round(float(accuracy_score(yb[m], np.ones(m.sum(), dtype=int))), 5)
        best = {}
        for t in THRESHOLDS:
            pred = (z_p[m] > t).astype(int)
            acc = accuracy_score(yb[m], pred)
            best[t] = round(float(acc), 5)
        r["acc_by_thr"] = best
        t_best = max(best, key=best.get)
        pred_best = (z_p[m] > t_best).astype(int)
        r["t_best"] = float(t_best)
        r["acc_best"] = best[t_best]
        r["precision_t_best"] = round(float(precision_score(yb[m], pred_best, zero_division=0)), 5)
        r["recall_t_best"] = round(float(recall_score(yb[m], pred_best, zero_division=0)), 5)
        r["pred_buy_share"] = round(float(pred_best.mean()), 5)
        rows.append(r)

    res = pl.DataFrame(rows)
    with pl.Config(tbl_rows=10, tbl_cols=10, tbl_width_chars=220):
        print("\n=== ACCURACY / RMSLE (OOF, honest) ===")
        print(res)

    Path("reports").mkdir(exist_ok=True)
    oof.write_parquet("reports/accuracy_oof.parquet")
    with open("reports/accuracy_oof.json", "w") as fh:
        json.dump({
            "protocol": "LOO CatBoost RMSE on z=log1p, features_exp02, parity with exp02",
            "params": PARAMS,
            "results": [dict(r) for r in res.iter_rows(named=True)],
        }, fh, indent=2)
    print("saved reports/accuracy_oof.json and reports/accuracy_oof.parquet")


if __name__ == "__main__":
    main()
