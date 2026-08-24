"""Stacked ensemble with hyperparameter search (z = log1p target space).

Protocol (honest, no leakage into reported fold_03 metric):

  Stage A — tuning: compact grids per base model on inner time split
            (train fold_00+01 -> validate fold_02), best config by RMSLE.
  Stage B — OOF: for each k in {fold_00..02} fit bases on the other two
            folds (early stopping on fold_k) -> pooled Z_oof [750k x M].
  Stage B2 — production bases: same best configs trained on all three
            folds (early stopping on fold_02, inner-only) predict
            fold_03 (Z_val) and fold_end (Z_end).
  Stage C — meta: RidgeCV / OLS over base z-predictions fitted on pooled
            OOF; intercept absorbs global level shift (exp06 built-in).
            Stack scored once on fold_03.
  Stage D — submissions: fold_end predictions for every base + stack.

Base models: LightGBM, HistGradientBoosting, CatBoost, Ridge(imputed+scaled).
Run from repo root:
    .venv/bin/python src/train_stack.py [--quick]
Artifacts:
    submissions/submission_stack_<meta>.csv, submissions/submission_bl_<model>.csv,
    reports/stack_search.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEAT_DIR = Path("data/v2/features_ext")
BTYD_DIR = Path("data/v2/features_bgnbd")
REPORT_PATH = Path("reports/stack_search.json")
SUB_DIR = Path("submissions")

SEED = 42
TRAIN_FOLDS = ["fold_00", "fold_01", "fold_02"]
INNER_VAL_FOLD = "fold_02"
META_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]

BTYD_FILL_ZERO = [
    "bgnbd_tx", "bgnbd_en30", "eb_lambda_n30", "bgnbd_e_gmv30", "eb_e_gmv30",
]

GRIDS = {
    "lgbm": [
        {"n_estimators": 1500, "learning_rate": lr, "num_leaves": nl,
         "min_child_samples": mc, "feature_fraction": ff}
        for lr in (0.05, 0.1) for nl in (63, 127)
        for mc, ff in ((50, 0.8), (200, 1.0))
    ],
    "hist": [
        {"max_iter": 800, "learning_rate": lr, "max_leaf_nodes": ml,
         "l2_regularization": l2, "min_samples_leaf": ms}
        for lr in (0.05, 0.1) for ml in (31, 63)
        for l2, ms in ((0.0, 40), (5.0, 100))
    ],
    "cat": [
        {"iterations": 1200, "learning_rate": 0.05, "depth": d, "l2_leaf_reg": l2}
        for d in (6, 8) for l2 in (3, 10)
    ],
    "ridge": [{"alpha": a} for a in (1.0, 10.0)],
}


def load_fold(name: str) -> pl.DataFrame:
    feats = pl.read_parquet(FEAT_DIR / name / "batch_*.parquet")
    btyd = pl.read_parquet(BTYD_DIR / f"{name}.parquet")
    df = feats.join(btyd, on=["anchor_date", "user_id"], how="left")
    return df.with_columns([
        pl.col(c).fill_null(0.0) if c in BTYD_FILL_ZERO else pl.col(c).fill_null(-1.0)
        for c in btyd.columns if c not in ("anchor_date", "user_id")
    ])


def rmsle_z(y_raw: np.ndarray, z_pred: np.ndarray) -> float:
    lt = np.log1p(np.clip(y_raw, 0, None))
    lp = np.clip(z_pred, None, 30.0)
    return float(np.sqrt(np.mean((lt - lp) ** 2)))


def make_model(name: str, params: dict):
    if name == "lgbm":
        p = dict(params)
        n_est = p.pop("n_estimators")
        return LGBMRegressor(**p, n_estimators=n_est, objective="l2",
                             random_state=SEED, n_jobs=-1, verbosity=-1)
    if name == "hist":
        return HistGradientBoostingRegressor(
            early_stopping=True, validation_fraction=0.08,
            n_iter_no_change=30, random_state=SEED, **params
        )
    if name == "cat":
        return CatBoostRegressor(**params, loss_function="RMSE",
                                 random_seed=SEED, verbose=0,
                                 allow_writing_files=False)
    if name == "ridge":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             RidgeCV(alphas=tuple(META_ALPHAS)))
    raise ValueError(name)


def fit_predict(name: str, params: dict, Xtr, ytr, Xs, eval_set=None):
    model = make_model(name, params)
    t0 = time.time()
    if name == "lgbm" and eval_set is not None:
        model.fit(Xtr, ytr, eval_set=[eval_set],
                  callbacks=[early_stopping(100, verbose=False),
                             log_evaluation(0)])
    elif name == "cat" and eval_set is not None:
        model.fit(Xtr, ytr, eval_set=eval_set, early_stopping_rounds=100)
    else:
        model.fit(Xtr, ytr)
    preds = [np.clip(model.predict(X), None, 30.0) for X in Xs]
    print(f"    fit {time.time() - t0:.0f}s", flush=True)
    return preds, model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny grids/iters smoke")
    args = ap.parse_args()

    t00 = time.time()
    print("loading folds...", flush=True)
    folds = {f: load_fold(f) for f in TRAIN_FOLDS + ["fold_03", "fold_end"]}
    meta_cols = {"anchor_date", "user_id", "target"}
    feats = [c for c in folds["fold_00"].columns if c not in meta_cols]
    print(f"features: {len(feats)}")

    X, Y_raw = {}, {}
    for f, df in folds.items():
        X[f] = df.select(feats).to_numpy().astype(np.float32)
        Y_raw[f] = np.clip(df["target"].to_numpy(), 0, None)
    Y = {f: np.log1p(Y_raw[f]) for f in folds}

    tune_tr_X = np.concatenate([X["fold_00"], X["fold_01"]])
    tune_tr_y = np.concatenate([Y["fold_00"], Y["fold_01"]])
    tune_val_X, tune_val_y = X[INNER_VAL_FOLD], Y[INNER_VAL_FOLD]

    grids = {k: v[:2] for k, v in GRIDS.items()} if args.quick else GRIDS

    print("\n=== STAGE A: hyperparameter search (train 00+01 -> val 02) ===")
    best_params: dict[str, dict] = {}
    for name, grid in grids.items():
        if args.quick and name == "cat":
            grid = [{**g, "iterations": 50} for g in grid]
        if args.quick and name == "lgbm":
            grid = [{**g, "n_estimators": 60} for g in grid]
        rows = []
        for i, params in enumerate(grid):
            preds, _ = fit_predict(name, params, tune_tr_X, tune_tr_y,
                                   [tune_val_X], eval_set=(tune_val_X, tune_val_y))
            score = rmsle_z(Y_raw[INNER_VAL_FOLD], preds[0])
            rows.append((score, params))
            print(f"  [{name} {i + 1}/{len(grid)}] RMSLE={score:.5f}", flush=True)
        best_params[name] = min(rows, key=lambda r: r[0])[1]
        print(f"  BEST {name}: {best_params[name]}")

    model_names = list(grids.keys())
    n_oof = sum(len(Y[f]) for f in TRAIN_FOLDS)
    Z_oof = np.full((n_oof, len(model_names)), np.nan)
    oof_y = np.concatenate([Y[f] for f in TRAIN_FOLDS])
    oof_raw = np.concatenate([Y_raw[f] for f in TRAIN_FOLDS])

    print("\n=== STAGE B: OOF matrix (leave-one-fold-out) ===")
    parts = {m: [] for m in range(len(model_names))}
    for k in TRAIN_FOLDS:
        tr_folds = [f for f in TRAIN_FOLDS if f != k]
        Xtr = np.concatenate([X[f] for f in tr_folds])
        ytr = np.concatenate([Y[f] for f in tr_folds])
        for m, name in enumerate(model_names):
            preds, _ = fit_predict(name, best_params[name], Xtr, ytr, [X[k]],
                                   eval_set=(X[k], Y[k]))
            parts[m].append(preds[0])
    for m in range(len(model_names)):
        Z_oof[:, m] = np.concatenate(parts[m])

    print("\n=== STAGE B2: production bases (all 3 train folds) ===")
    Xtr_all = np.concatenate([X[f] for f in TRAIN_FOLDS])
    ytr_all = np.concatenate([Y[f] for f in TRAIN_FOLDS])
    es_ref = (X[INNER_VAL_FOLD], Y[INNER_VAL_FOLD])
    prod_models, Z_val, Z_end = {}, [], []
    for m, name in enumerate(model_names):
        preds, model = fit_predict(name, best_params[name], Xtr_all, ytr_all,
                                   [X["fold_03"], X["fold_end"]],
                                   eval_set=es_ref)
        prod_models[name] = model
        Z_val.append(preds[0])
        Z_end.append(preds[1])
    Z_val = np.column_stack(Z_val)
    Z_end = np.column_stack(Z_end)

    print("\n=== STAGE C: meta-model on pooled OOF ===")
    metas = {
        "ridgecv": RidgeCV(alphas=META_ALPHAS),
        "ols": LinearRegression(positive=True),
    }
    stack_scores, coefs = {}, {}
    for mname, meta in metas.items():
        meta.fit(Z_oof, oof_y)
        s = rmsle_z(Y_raw["fold_03"], meta.predict(Z_val))
        stack_scores[mname] = round(s, 5)
        coefs[mname] = dict(zip(model_names,
                                np.round(meta.coef_, 4).tolist()))
        coefs[mname]["intercept"] = round(float(meta.intercept_), 4)
        print(f"  meta={mname}: RMSLE(fold_03)={s:.5f} {coefs[mname]}")

    solo = {n: round(rmsle_z(Y_raw["fold_03"], Z_val[:, i]), 5)
            for i, n in enumerate(model_names)}
    best_meta_name = min(stack_scores, key=stack_scores.get)

    print(f"\nLEADERBOARD fold_03 (refs: naive 2.19506, exp01-catboost 1.70261):")
    board = sorted({**solo, **stack_scores}.items(), key=lambda kv: kv[1])
    for n, s in board:
        tag = "STACK" if n in stack_scores else "solo "
        print(f"  {s:.5f}  [{tag}] {n}")
    print(f"\nBEST STACK: meta={best_meta_name} -> {stack_scores[best_meta_name]:.5f}")

    print("\n=== STAGE D: submissions ===")
    SUB_DIR.mkdir(exist_ok=True)
    sample = pl.read_csv("sample_submit.csv")
    order = sample.select("user_id").with_row_index("__ord")
    uids = folds["fold_end"]["user_id"].cast(pl.Int64)

    def save(pred_z: np.ndarray, fname: str) -> dict:
        pred = np.clip(np.expm1(pred_z), 0, None)
        assert np.isfinite(pred).all() and (pred >= 0).all()
        out = SUB_DIR / fname
        (pl.DataFrame({"user_id": uids, "predict": pred})
         .join(order, on="user_id", how="inner")
         .sort("__ord").drop("__ord").select(["user_id", "predict"])
         .write_csv(out))
        chk = pl.read_csv(out)
        assert chk.height == 250_000 and chk.columns == ["user_id", "predict"]
        return {"file": str(out),
                "pred_mean": round(float(chk["predict"].mean()), 2),
                "pred_median": round(float(chk["predict"].median()), 2)}

    subs = {f"stack_{best_meta_name}":
            save(metas[best_meta_name].predict(Z_end),
                 f"submission_stack_{best_meta_name}.csv")}
    for i, n in enumerate(model_names):
        subs[n] = save(Z_end[:, i], f"submission_bl_{n}.csv")
    for k, v in subs.items():
        print(f"  {v['file']} (mean={v['pred_mean']})")

    REPORT_PATH.write_text(json.dumps({
        "protocol": "A: tune on 00+01->02 | B: OOF leave-one-fold-out | "
                    "B2: production bases on 00..02, ES on 02 | "
                    "C: meta on pooled OOF | fold_03 = holdout report",
        "features": len(feats),
        "grids": grids,
        "best_params": best_params,
        "rmsle_fold03": {"stack": stack_scores, "solo": solo},
        "meta_coefs": coefs,
        "submissions": subs,
        "runtime_min": round((time.time() - t00) / 60, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nreport -> {REPORT_PATH}")
    print(f"ALL DONE in {(time.time() - t00) / 60:.1f} min")


if __name__ == "__main__":
    main()
