"""Stacking v2: expanded diverse ensemble + deeper tuning + OOF calibration.

Goal: improve on v1 stack (fold_03 1.67103, LB ~1.66) toward 1.64.

Changes vs src/train_stack.py:
- members: lgbm_l2, lgbm_q50 (median regression), hist, cat_rmse, cat_q50,
  xgb_l2 — diversity via algorithms AND losses (quantile members stabilize
  the heavy zero-mass region);
- deeper compact grids per member on inner split fold_00+01 -> fold_02;
- meta candidates compared honestly inside OOF (fit on folds 00+01 rows,
  validate on fold_02 rows): RidgeCV / OLS-positive / shallow LGBM-meta;
- post-calibration grid (global z-shift c x zero-threshold tau) tuned on the
  same inner OOF slice, applied to production; single verification on fold_03.

Run from repo root:
    .venv/bin/python src/train_stack_v2.py [--quick]
Artifacts:
    submissions/submission_stack_v2.csv, submission_stack_v2_calib.csv,
    reports/stack_search_v2.json
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
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression, RidgeCV
from xgboost import XGBRegressor

FEAT_DIR = Path("data/v2/features_ext")
BTYD_DIR = Path("data/v2/features_bgnbd")
REPORT_PATH = Path("reports/stack_search_v2.json")
SUB_DIR = Path("submissions")

SEED = 42
TRAIN_FOLDS = ["fold_00", "fold_01", "fold_02"]
INNER_VAL_FOLD = "fold_02"
MEMBERS = ["lgbm", "lgbm_q50", "hist", "cat", "cat_q50", "xgb"]

BTYD_FILL_ZERO = [
    "bgnbd_tx", "bgnbd_en30", "eb_lambda_n30", "bgnbd_e_gmv30", "eb_e_gmv30",
]

GRIDS = {
    "lgbm": [
        {"n_estimators": 2500, "learning_rate": 0.04, "num_leaves": nl,
         "min_child_samples": mc, "feature_fraction": 0.85}
        for nl in (63, 127) for mc in (100, 300)
    ],
    "hist": [
        {"max_iter": 900, "learning_rate": lr, "max_leaf_nodes": ml,
         "l2_regularization": 5.0, "min_samples_leaf": 80}
        for lr in (0.04, 0.08) for ml in (31, 63)
    ],
    "cat": [
        {"iterations": 2000, "learning_rate": 0.05, "depth": d, "l2_leaf_reg": l2}
        for d in (8, 10) for l2 in (3, 9)
    ],
    "xgb": [
        {"n_estimators": 1800, "learning_rate": 0.05, "max_depth": d,
         "min_child_weight": mcw, "colsample_bytree": 0.8, "subsample": 0.85}
        for d in (6, 10) for mcw in (30, 100)
    ],
}

META_ALPHAS = [0.01, 0.1, 1.0, 10.0]
CALIB_C = [-0.06, -0.03, -0.015, -0.0075, 0.0, 0.0075, 0.015, 0.03]
CALIB_TAU = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]


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


def make_member(name: str, params: dict):
    if name == "lgbm":
        p = dict(params)
        n_est = p.pop("n_estimators")
        return LGBMRegressor(**p, n_estimators=n_est, objective="l2",
                             random_state=SEED, n_jobs=-1, verbosity=-1)
    if name == "lgbm_q50":
        p = dict(params)
        n_est = p.pop("n_estimators")
        return LGBMRegressor(**p, n_estimators=n_est, objective="quantile",
                             alpha=0.5, random_state=SEED, n_jobs=-1,
                             verbosity=-1)
    if name == "hist":
        return HistGradientBoostingRegressor(
            early_stopping=True, validation_fraction=0.08,
            n_iter_no_change=30, random_state=SEED, **params)
    if name == "cat":
        return CatBoostRegressor(**params, loss_function="RMSE",
                                 random_seed=SEED, verbose=0,
                                 allow_writing_files=False)
    if name == "cat_q50":
        return CatBoostRegressor(**params, loss_function="Quantile:alpha=0.5",
                                 random_seed=SEED, verbose=0,
                                 allow_writing_files=False)
    if name == "xgb":
        p = dict(params)
        n_est = p.pop("n_estimators")
        return XGBRegressor(**p, n_estimators=n_est, tree_method="hist",
                            random_state=SEED, n_jobs=-1, verbosity=0,
                            early_stopping_rounds=100)
    raise ValueError(name)


def fit_predict(name: str, params: dict, Xtr, ytr, Xs, eval_set=None):
    model = make_member(name, params)
    t0 = time.time()
    if name.startswith("lgbm") and eval_set is not None:
        model.fit(Xtr, ytr, eval_set=[eval_set],
                  callbacks=[early_stopping(100, verbose=False),
                             log_evaluation(0)])
    elif name.startswith("cat") and eval_set is not None:
        model.fit(Xtr, ytr, eval_set=eval_set, early_stopping_rounds=150)
    elif name == "xgb" and eval_set is not None:
        model.fit(Xtr, ytr, eval_set=[eval_set], verbose=False)
    else:
        model.fit(Xtr, ytr)
    preds = [np.clip(model.predict(X), None, 30.0) for X in Xs]
    print(f"    fit {time.time() - t0:.0f}s", flush=True)
    return preds, model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
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
    tv_X, tv_y = X[INNER_VAL_FOLD], Y[INNER_VAL_FOLD]

    grids = {k: [{**g} for g in v[:2]] for k, v in GRIDS.items()} \
        if args.quick else GRIDS

    print("\n=== STAGE A: tuning (train 00+01 -> val 02) ===")
    best_params: dict[str, dict] = {}
    for name, grid in grids.items():
        rows = []
        for i, params in enumerate(grid):
            preds, _ = fit_predict(name, params, tune_tr_X, tune_tr_y,
                                   [tv_X], eval_set=(tv_X, tv_y))
            s = rmsle_z(Y_raw[INNER_VAL_FOLD], preds[0])
            rows.append((s, params))
            print(f"  [{name} {i + 1}/{len(grid)}] RMSLE={s:.5f}", flush=True)
        best_params[name] = min(rows, key=lambda r: r[0])[1]
        print(f"  BEST {name}: {best_params[name]}", flush=True)

    q_members = {"lgbm_q50": ("lgbm", best_params["lgbm"]),
                 "cat_q50": ("cat", best_params["cat"])}
    all_params = dict(best_params)
    all_params.update({q: base_params for q, (_, base_params) in q_members.items()})

    print("\n=== STAGE B: OOF matrix (leave-one-fold-out) ===")
    Z_oof = {m: [] for m in MEMBERS}
    oof_fold_id = []
    for k in TRAIN_FOLDS:
        tr_folds = [f for f in TRAIN_FOLDS if f != k]
        Xtr = np.concatenate([X[f] for f in tr_folds])
        ytr = np.concatenate([Y[f] for f in tr_folds])
        for m in MEMBERS:
            preds, _ = fit_predict(m, all_params[m], Xtr, ytr, [X[k]],
                                   eval_set=(X[k], Y[k]))
            Z_oof[m].append(preds[0])
        oof_fold_id += [k] * len(X[k])
    Z_oof = {m: np.concatenate(v) for m, v in Z_oof.items()}
    oof_y = np.concatenate([Y[f] for f in TRAIN_FOLDS])
    oof_raw = np.concatenate([Y_raw[f] for f in TRAIN_FOLDS])
    fold_id = np.array(oof_fold_id)

    print("\n=== STAGE B2: production bases (all 3 train folds) ===")
    Xtr_all = np.concatenate([X[f] for f in TRAIN_FOLDS])
    ytr_all = np.concatenate([Y[f] for f in TRAIN_FOLDS])
    es_ref = (X[INNER_VAL_FOLD], Y[INNER_VAL_FOLD])
    Z_val_m, Z_end_m = {}, {}
    for m in MEMBERS:
        preds, _ = fit_predict(m, all_params[m], Xtr_all, ytr_all,
                               [X["fold_03"], X["fold_end"]], eval_set=es_ref)
        Z_val_m[m], Z_end_m[m] = preds
    Z_val = np.column_stack([Z_val_m[m] for m in MEMBERS])
    Z_end = np.column_stack([Z_end_m[m] for m in MEMBERS])

    def mat_of(d: dict[str, np.ndarray]) -> np.ndarray:
        return np.column_stack([d[m] for m in MEMBERS])

    print("\n=== STAGE C: meta selection inside OOF ===")
    tr_mask = np.isin(fold_id, ["fold_00", "fold_01"])
    va_mask = fold_id == INNER_VAL_FOLD
    Zo_fit, y_fit = mat_of(Z_oof)[tr_mask], oof_y[tr_mask]
    Zo_va, yr_va = mat_of(Z_oof)[va_mask], oof_raw[va_mask]

    meta_candidates = {
        "ridgecv": RidgeCV(alphas=META_ALPHAS),
        "ols_pos": LinearRegression(positive=True),
        "lgbm_meta": LGBMRegressor(n_estimators=300, learning_rate=0.05,
                                   num_leaves=15, min_child_samples=500,
                                   random_state=SEED, verbosity=-1, n_jobs=-1),
    }
    meta_scores = {}
    for mname, meta in meta_candidates.items():
        meta.fit(Zo_fit, y_fit)
        s = rmsle_z(yr_va, meta.predict(Zo_va))
        meta_scores[mname] = round(s, 5)
        print(f"  meta={mname}: inner-OOF RMSLE={s:.5f}")
    best_meta_name = min(meta_scores, key=meta_scores.get)

    print("\n=== STAGE D: calibration grid on inner-OOF slice ===")
    meta_best = meta_candidates[best_meta_name].fit(mat_of(Z_oof), oof_y)
    z_va = meta_candidates[best_meta_name].predict(Zo_va)

    def apply_calib(z: np.ndarray, c: float, tau: float) -> np.ndarray:
        zc = z + c
        return np.where(zc < tau, 0.0, zc)

    grid_rows = []
    for c in CALIB_C:
        for tau in CALIB_TAU:
            s = rmsle_z(yr_va, apply_calib(z_va, c, tau))
            grid_rows.append((s, c, tau))
    grid_rows.sort(key=lambda r: r[0])
    (best_c, best_tau), raw_inner = grid_rows[0][1:], rmsle_z(yr_va, z_va)
    print(f"  raw inner={raw_inner:.5f}; best calib c={best_c} tau={best_tau} "
          f"-> {grid_rows[0][0]:.5f}")

    print("\n=== STAGE E: final scores & submissions ===")
    stack_val = meta_best.predict(Z_val)
    solo = {m: round(rmsle_z(Y_raw["fold_03"], Z_val_m[m]), 5) for m in MEMBERS}
    stack_raw_score = round(rmsle_z(Y_raw["fold_03"], stack_val), 5)
    calib_val = apply_calib(stack_val, best_c, best_tau)
    stack_calib_score = round(rmsle_z(Y_raw["fold_03"], calib_val), 5)

    print(f"LEADERBOARD fold_03 (v1 stack 1.67103, exp01 1.70261):")
    board = sorted({**solo,
                    "STACK_v2_" + best_meta_name: stack_raw_score,
                    "STACK_v2_calib": stack_calib_score}.items(),
                   key=lambda kv: kv[1])
    for n, s in board:
        tag = "STK " if "STACK" in n else "solo"
        print(f"  {s:.5f}  [{tag}] {n}")

    SUB_DIR.mkdir(exist_ok=True)
    sample = pl.read_csv("sample_submit.csv")
    order = sample.select("user_id").with_row_index("__ord")
    uids = folds["fold_end"]["user_id"].cast(pl.Int64)

    def save(z: np.ndarray, fname: str) -> dict:
        pred = np.clip(np.expm1(z), 0, None)
        assert np.isfinite(pred).all() and (pred >= 0).all()
        out = SUB_DIR / fname
        (pl.DataFrame({"user_id": uids, "predict": pred})
         .join(order, on="user_id", how="inner").sort("__ord").drop("__ord")
         .select(["user_id", "predict"]).write_csv(out))
        chk = pl.read_csv(out)
        assert chk.height == 250_000
        return {"file": str(out), "pred_mean": round(float(chk["predict"].mean()), 2)}

    subs = {
        "stack_v2": save(stack_val, "submission_stack_v2.csv"),
        "stack_v2_calib": save(calib_val, "submission_stack_v2_calib.csv"),
    }
    for k, v in subs.items():
        print(f"  {v['file']} mean={v['pred_mean']}")

    REPORT_PATH.write_text(json.dumps({
        "protocol": "A tune 00+01->02 | B OOF LOFO | B2 prod bases on 00..02 "
                    "| C meta chosen on inner OOF split | D calibration grid "
                    "on inner OOF | fold_03 holdout",
        "features": len(feats),
        "members": MEMBERS,
        "grids": grids,
        "best_params": all_params,
        "meta_selection_inner_oof": meta_scores,
        "calibration": {"chosen": {"c": best_c, "tau": best_tau},
                        "inner_raw": round(raw_inner, 5),
                        "inner_calib": round(grid_rows[0][0], 5)},
        "rmsle_fold03": {"solo": solo,
                         "stack_raw": stack_raw_score,
                         "stack_calib": stack_calib_score},
        "submissions": subs,
        "runtime_min": round((time.time() - t00) / 60, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nreport -> {REPORT_PATH}")
    print(f"ALL DONE in {(time.time() - t00) / 60:.1f} min")


if __name__ == "__main__":
    main()
