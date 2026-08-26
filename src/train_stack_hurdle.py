"""exp13 — Hurdle-ансамбль поверх стека v3: P(y>0) x E[z|y>0], z=log1p(y).

Тождество моментов: E[z|X] = P(y>0|X) · E[z|y>0,X] (z=0 при y=0).
  Stage 1: LGBM binary классификатор P(y>0) (без scale_pos_weight — не ломаем
           калибровку вероятностей), изотоника p̂ по OOF (выбор raw/iso на inner).
  Stage 2: текущий стек v3 (lgbm/lgbm_q50/hist/cat/cat_q50/xgb), ОБУЧЕННЫЙ только
           на покупателях (y>0), предсказывает E[z|buy] на ВСЕ строки;
           мета как в v3 (ridgecv/ols_pos/lgbm_meta, выбор на inner-OOF).
  Итог:   ẑ = p̂ · ẑ2 поэлементно на всех строках, калибровка (k,c,tau) и бленд-α
          с single-stage референсом выбираются строго на inner-slice (fold_02);
          fold_03 — чистый холдаут; expm1 один раз в конце.

Протокол как в train_stack_v3.py:
  A  тюнинг гридов: train fold_00+01 -> val fold_02
  B  OOF leave-one-fold-out по fold_00..02 (обе ступени + single-референс)
  B2 прод-бейзы на fold_00..02 -> предикты fold_03 и fold_end
  C  мета stage-2 выбирается внутри OOF (fit 00+01, val 02)
  D  изотоника p̂, калибровка k x c x tau, бленд α — всё на inner-slice
  E  скоры fold_03 + сабмиты
  F  мат-ожидание в RAW-пространстве E[y|X] = P(buy) x E[y|buy]: expm1(E[z])
     занижает из-за Йенсена; коррекции (smearing/lognorm/ratio, p-calib) по
     OOF-остаткам покупателей, вариант выбирается на inner-slice по калибровке
     суммарной выручки

Run from repo root:
    .venv/bin/python src/train_stack_hurdle.py [--quick] [--fresh]
Artifacts:
    reports/stack_search_hurdle.json,
    submissions/submission_{hurdle,hurdle_calib,hurdle_blend_calib,hurdle_ev}.csv
    reports/ckpt_stack_hurdle/state.npz  (чекпоинт OOF/прод-предиктов, авто-resume)
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
from catboost import CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping, log_evaluation
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from xgboost import XGBRegressor

V3_DIR = Path("data/v3")
REPORT_PATH = Path("reports/stack_search_hurdle.json")
CKPT_PATH = Path("reports/ckpt_stack_hurdle/state.npz")
SUB_DIR = Path("submissions")

SEED = 42
TRAIN_FOLDS = ["fold_00", "fold_01", "fold_02"]
INNER_VAL_FOLD = "fold_02"
HOLDOUT = "fold_03"
ALL_FOLDS = TRAIN_FOLDS + [HOLDOUT, "fold_end"]

MEMBERS = ["lgbm", "lgbm_q50", "hist", "cat", "cat_q50", "xgb"]

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

CLF_GRID = [
    {"n_estimators": 2500, "learning_rate": 0.04, "num_leaves": nl,
     "min_child_samples": mc, "feature_fraction": 0.85,
     "bagging_fraction": 0.9, "bagging_freq": 1}
    for nl in (63, 127) for mc in (100, 300)
]

# single-stage референс (для диагностики декорреляции и бленда):
# лучшие параметры lgbm из reports/stack_search_v3.json
SINGLE_PARAMS = {"n_estimators": 2500, "learning_rate": 0.04, "num_leaves": 63,
                 "min_child_samples": 100, "feature_fraction": 0.85}

META_ALPHAS = [0.01, 0.1, 1.0, 10.0]
CALIB_K = [0.85, 0.95, 1.0, 1.05, 1.15]
CALIB_C = [-0.12, -0.09, -0.06, -0.04, -0.02, 0.0, 0.02]
CALIB_TAU = [0.0, 0.25]
BLEND_ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]

QUICK_SCALE = {"n_estimators": 0.24, "max_iter": 0.33, "iterations": 0.3}


def scale_params(params: dict) -> dict:
    return {k: (int(v * QUICK_SCALE[k]) if k in QUICK_SCALE else v)
            for k, v in params.items()}


def load_fold(name: str) -> pl.DataFrame:
    return pl.read_parquet(V3_DIR / f"{name}.parquet")


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


def fit_member(name: str, params: dict, Xtr, ytr, Xs, eval_set=None):
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
    print(f"      fit {time.time() - t0:.0f}s", flush=True)
    return preds


def fit_clf(params: dict, Xtr, ybin_tr, Xs, eval_set=None):
    model = LGBMClassifier(**params, objective="binary",
                           random_state=SEED, n_jobs=-1, verbosity=-1)
    t0 = time.time()
    if eval_set is not None:
        model.fit(Xtr, ybin_tr, eval_set=[eval_set], eval_metric="binary_logloss",
                  callbacks=[early_stopping(100, verbose=False),
                             log_evaluation(0)])
    else:
        model.fit(Xtr, ybin_tr)
    preds = [model.predict_proba(X)[:, 1].astype(np.float64) for X in Xs]
    print(f"      clf fit {time.time() - t0:.0f}s", flush=True)
    return preds


def apply_calib(z: np.ndarray, k: float, c: float, tau: float) -> np.ndarray:
    zc = z * k + c
    return np.where(zc < tau, 0.0, zc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="урезанные гриды и итерации для смоука")
    ap.add_argument("--fresh", action="store_true", help="игнорировать чекпоинт")
    args = ap.parse_args()
    quick = args.quick

    t00 = time.time()
    print("loading v3 folds...", flush=True)
    folds = {f: load_fold(f) for f in ALL_FOLDS}
    meta_cols = {"anchor_date", "user_id", "target"}
    feats = [c for c in folds["fold_00"].columns if c not in meta_cols]
    print(f"features: {len(feats)}")

    X, Y_raw, Y, BUY = {}, {}, {}, {}
    for f, df in folds.items():
        X[f] = df.select(feats).to_numpy().astype(np.float32)
        Y_raw[f] = np.clip(df["target"].to_numpy(), 0, None)
        Y[f] = np.log1p(Y_raw[f])
        BUY[f] = Y_raw[f] > 0
        print(f"  {f}: {df.height} rows, zeros={1 - BUY[f].mean():.3f}")

    grids = GRIDS if not quick else \
        {k: [scale_params(g) for g in v[:1]] for k, v in GRIDS.items()}
    clf_grid = CLF_GRID if not quick else [scale_params(CLF_GRID[0])]
    single_params = SINGLE_PARAMS if not quick else scale_params(SINGLE_PARAMS)

    ckpt_ok = (not args.fresh) and CKPT_PATH.exists()
    state: dict = {}
    if ckpt_ok:
        print(f"resume from {CKPT_PATH}", flush=True)
        state = dict(np.load(CKPT_PATH, allow_pickle=True))

    def save_ckpt() -> None:
        CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CKPT_PATH, **state)

    def have(*keys: str) -> bool:
        return ckpt_ok and all(k in state for k in keys)

    # ---------- STAGE A: tuning ----------
    tune_tr_X = np.concatenate([X["fold_00"], X["fold_01"]])
    tune_tr_yraw = np.concatenate([Y_raw["fold_00"], Y_raw["fold_01"]])
    tune_tr_yz = np.log1p(tune_tr_yraw)
    tune_tr_Xb = tune_tr_X[tune_tr_yraw > 0]
    tune_tr_yz_b = tune_tr_yz[tune_tr_yraw > 0]
    tv_Xb = X[INNER_VAL_FOLD][BUY[INNER_VAL_FOLD]]
    tv_yzb = Y[INNER_VAL_FOLD][BUY[INNER_VAL_FOLD]]
    tv_X = X[INNER_VAL_FOLD]
    tv_ybin = (Y_raw[INNER_VAL_FOLD] > 0).astype(np.int8)

    best_clf_params, best_params = None, {}
    if have("best_clf_params", "best_params_json"):
        best_clf_params = json.loads(str(state["best_clf_params"]))
        best_params = json.loads(str(state["best_params_json"]))
        print("stage A loaded from checkpoint")
    else:
        print("\n=== STAGE A: tuning (train 00+01 -> val 02) ===")
        rows = []
        for i, params in enumerate(clf_grid):
            (pv,) = fit_clf(params, tune_tr_X,
                            (tune_tr_yraw > 0).astype(np.int8),
                            [tv_X], eval_set=(tv_X, tv_ybin))
            auc = float(roc_auc_score(tv_ybin, pv))
            rows.append((auc, params))
            print(f"  [clf {i + 1}/{len(clf_grid)}] AUC={auc:.5f}", flush=True)
        best_clf_params = max(rows, key=lambda r: r[0])[1]
        print(f"  BEST clf: {best_clf_params}", flush=True)

        for name, grid in grids.items():
            rows = []
            for i, params in enumerate(grid):
                (pv,) = fit_member(name, params, tune_tr_Xb, tune_tr_yz_b, [tv_Xb],
                                   eval_set=(tv_Xb, tv_yzb))
                s = rmsle_z(np.expm1(tv_yzb), pv)
                rows.append((s, params))
                print(f"  [{name} {i + 1}/{len(grid)}] RMSLE(buyers)={s:.5f}",
                      flush=True)
            best_params[name] = min(rows, key=lambda r: r[0])[1]
            print(f"  BEST {name}: {best_params[name]}", flush=True)
        best_params["lgbm_q50"] = best_params["lgbm"]
        best_params["cat_q50"] = best_params["cat"]
        state["best_clf_params"] = json.dumps(best_clf_params)
        state["best_params_json"] = json.dumps(best_params)
        save_ckpt()

    # ---------- STAGE B: OOF matrices ----------
    if have("P_oof", "oof_fold_id", "oof_yraw",
            *[f"Z2_oof_{m}" for m in MEMBERS], "Zs_oof"):
        P_oof = state["P_oof"]
        oof_fold_id = state["oof_fold_id"]
        oof_yraw = state["oof_yraw"]
        Z2_oof = {m: state[f"Z2_oof_{m}"] for m in MEMBERS}
        Zs_oof = state["Zs_oof"]
        oof_buy = oof_yraw > 0
        print("stage B loaded from checkpoint")
    else:
        print("\n=== STAGE B: OOF (leave-one-fold-out) ===")
        P_parts, Zs_parts, fid_parts, yraw_parts = [], [], [], []
        Z2_parts = {m: [] for m in MEMBERS}
        for k in TRAIN_FOLDS:
            tr_folds = [f for f in TRAIN_FOLDS if f != k]
            Xtr = np.concatenate([X[f] for f in tr_folds])
            yraw_tr = np.concatenate([Y_raw[f] for f in tr_folds])
            yz_tr = np.log1p(yraw_tr)
            ybin_tr = (yraw_tr > 0).astype(np.int8)
            buy_tr = yraw_tr > 0

            (pk,) = fit_clf(best_clf_params, Xtr, ybin_tr, [X[k]],
                            eval_set=(X[k], (Y_raw[k] > 0).astype(np.int8)))
            P_parts.append(pk)

            # члены стека учатся только на покупателях, предсказывают все строки фолда
            Xtr_b, yz_tr_b = Xtr[buy_tr], yz_tr[buy_tr]
            es_b = (X[k][BUY[k]], Y[k][BUY[k]])
            for m in MEMBERS:
                (zk,) = fit_member(m, best_params[m], Xtr_b, yz_tr_b, [X[k]],
                                   eval_set=es_b)
                Z2_parts[m].append(zk)

            (zs,) = fit_member("lgbm", single_params, Xtr, yz_tr, [X[k]],
                               eval_set=(X[k], Y[k]))
            Zs_parts.append(zs)

            fid_parts += [k] * len(Y_raw[k])
            yraw_parts.append(Y_raw[k])

        P_oof = np.concatenate(P_parts)
        Zs_oof = np.concatenate(Zs_parts)
        Z2_oof = {m: np.concatenate(v) for m, v in Z2_parts.items()}
        oof_fold_id = np.array(fid_parts)
        oof_yraw = np.concatenate(yraw_parts)
        oof_buy = oof_yraw > 0
        state.update({"P_oof": P_oof, "oof_fold_id": oof_fold_id,
                      "oof_yraw": oof_yraw, "Zs_oof": Zs_oof,
                      **{f"Z2_oof_{m}": Z2_oof[m] for m in MEMBERS}})
        save_ckpt()

    # ---------- STAGE C/D: meta E[z|buy], iso, k/c/tau, alpha (inner=fold_02) ----------
    print("\n=== STAGE C/D: meta + calibration on inner slice (val=fold_02) ===")
    tr_mask = np.isin(oof_fold_id, ["fold_00", "fold_01"])
    va_mask = oof_fold_id == INNER_VAL_FOLD

    Zo_mat = np.column_stack([Z2_oof[m] for m in MEMBERS])
    Zo_fit, yz_fit = Zo_mat[tr_mask & oof_buy], np.log1p(oof_yraw[tr_mask & oof_buy])
    Zo_va = Zo_mat[va_mask]
    p_va = P_oof[va_mask]

    meta_candidates = {
        "ridgecv": RidgeCV(alphas=META_ALPHAS),
        "ols_pos": LinearRegression(positive=True),
        "lgbm_meta": LGBMRegressor(n_estimators=300, learning_rate=0.05,
                                   num_leaves=15, min_child_samples=500,
                                   random_state=SEED, verbosity=-1, n_jobs=-1),
    }
    # мета выбирается по ИТОГОВОМУ hurdle-RMSLE на inner-slice (p_va * E[z|buy])
    meta_scores, meta_z2_va = {}, {}
    for mname, meta in meta_candidates.items():
        meta.fit(Zo_fit, yz_fit)
        z2v = meta.predict(Zo_va)
        meta_z2_va[mname] = z2v
        s = rmsle_z(oof_yraw[va_mask], np.clip(p_va * z2v, None, 30.0))
        meta_scores[mname] = round(s, 5)
        print(f"  meta={mname}: inner hurdle RMSLE={s:.5f}")
    best_meta_name = min(meta_scores, key=meta_scores.get)
    z2_va = meta_z2_va[best_meta_name]

    # изотоника p̂: fit строго до inner-фолда
    iso = IsotonicRegression(out_of_bounds="clip").fit(
        P_oof[tr_mask], (oof_yraw[tr_mask] > 0).astype(np.int8))
    p_iso_va = iso.predict(p_va)

    z_h_raw_va = np.clip(p_va * z2_va, None, 30.0)
    z_h_iso_va = np.clip(p_iso_va * z2_va, None, 30.0)
    s_raw = rmsle_z(oof_yraw[va_mask], z_h_raw_va)
    s_iso = rmsle_z(oof_yraw[va_mask], z_h_iso_va)
    use_iso = s_iso < s_raw
    print(f"  hurdle inner: raw p {s_raw:.5f} | isotonic p {s_iso:.5f} "
          f"-> {'iso' if use_iso else 'raw'}")
    p_src_va = p_iso_va if use_iso else p_va
    z_h_va = np.clip(p_src_va * z2_va, None, 30.0)

    grid_rows = []
    for kk in CALIB_K:
        for cc in CALIB_C:
            for tt in CALIB_TAU:
                s = rmsle_z(oof_yraw[va_mask], apply_calib(z_h_va, kk, cc, tt))
                grid_rows.append((s, kk, cc, tt))
    grid_rows.sort(key=lambda r: r[0])
    best_k, best_c, best_tau = grid_rows[0][1:]
    raw_inner = rmsle_z(oof_yraw[va_mask], z_h_va)
    print(f"  calib grid: raw inner={raw_inner:.5f}; "
          f"best k={best_k} c={best_c} tau={best_tau} -> {grid_rows[0][0]:.5f}")

    zs_va = Zs_oof[va_mask]
    corr_inner = float(np.corrcoef(z_h_va, zs_va)[0, 1])
    blend_rows = []
    for a in BLEND_ALPHAS:
        zb = a * z_h_va + (1 - a) * zs_va
        blend_rows.append((rmsle_z(oof_yraw[va_mask],
                                   apply_calib(zb, best_k, best_c, best_tau)), a))
    blend_rows.sort(key=lambda r: r[0])
    best_alpha = blend_rows[0][1]
    print(f"  corr(hurdle,single) inner={corr_inner:.4f}; "
          f"best blend alpha={best_alpha} -> {blend_rows[0][0]:.5f}")

    # ---------- STAGE B2: production bases ----------
    if have("P_val", "P_end", "Z_val_mat", "Z_end_mat", "Zs_val", "Zs_end"):
        P_val, P_end = state["P_val"], state["P_end"]
        Z_val_mat, Z_end_mat = state["Z_val_mat"], state["Z_end_mat"]
        Zs_val, Zs_end = state["Zs_val"], state["Zs_end"]
        print("stage B2 loaded from checkpoint")
    else:
        print("\n=== STAGE B2: production bases (train fold_00..02) ===")
        Xtr_all = np.concatenate([X[f] for f in TRAIN_FOLDS])
        yraw_all = np.concatenate([Y_raw[f] for f in TRAIN_FOLDS])
        yz_all = np.log1p(yraw_all)
        ybin_all = (yraw_all > 0).astype(np.int8)
        buy_all = yraw_all > 0
        es_ref = (X[INNER_VAL_FOLD], Y[INNER_VAL_FOLD])

        P_val, P_end = fit_clf(best_clf_params, Xtr_all, ybin_all,
                               [X[HOLDOUT], X["fold_end"]],
                               eval_set=(tv_X, tv_ybin))

        Xtr_b, yz_b = Xtr_all[buy_all], yz_all[buy_all]
        es_b_ref = (tv_Xb, tv_yzb)
        Z_val_m, Z_end_m = {}, {}
        for m in MEMBERS:
            pv, pe = fit_member(m, best_params[m], Xtr_b, yz_b,
                                [X[HOLDOUT], X["fold_end"]], eval_set=es_b_ref)
            Z_val_m[m], Z_end_m[m] = pv, pe
        Z_val_mat = np.column_stack([Z_val_m[m] for m in MEMBERS])
        Z_end_mat = np.column_stack([Z_end_m[m] for m in MEMBERS])

        Zs_val, Zs_end = fit_member("lgbm", single_params, Xtr_all, yz_all,
                                    [X[HOLDOUT], X["fold_end"]], eval_set=es_ref)
        state.update({"P_val": P_val, "P_end": P_end, "Z_val_mat": Z_val_mat,
                      "Z_end_mat": Z_end_mat, "Zs_val": Zs_val, "Zs_end": Zs_end})
        save_ckpt()

    # ---------- STAGE E: final combine, scores, submissions ----------
    print("\n=== STAGE E: final scores & submissions ===")
    meta_full = meta_candidates[best_meta_name].fit(
        Zo_mat[oof_buy], np.log1p(oof_yraw[oof_buy]))
    # если изотоника выбрана — рефит на всём pooled OOF (паритет с рефитом мета)
    p_use_val = iso.predict(P_val) if use_iso else P_val
    p_use_end = iso.predict(P_end) if use_iso else P_end

    z_h_val = np.clip(p_use_val * meta_full.predict(Z_val_mat), None, 30.0)
    z_h_end = np.clip(p_use_end * meta_full.predict(Z_end_mat), None, 30.0)
    z_s_val, z_s_end = Zs_val, Zs_end

    ybin_val = (Y_raw[HOLDOUT] > 0).astype(np.int8)
    auc_val = float(roc_auc_score(ybin_val, P_val))
    print(f"  clf fold_03: AUC={auc_val:.5f}; mean p={P_val.mean():.4f} "
          f"vs actual buy share={ybin_val.mean():.4f}")

    solo_hurdle = {}
    for i, m in enumerate(MEMBERS):
        zv = np.clip(p_use_val * Z_val_mat[:, i], None, 30.0)
        solo_hurdle[m] = round(rmsle_z(Y_raw[HOLDOUT], zv), 5)

    stack_raw_score = round(rmsle_z(Y_raw[HOLDOUT], z_h_val), 5)
    stack_calib_score = round(rmsle_z(Y_raw[HOLDOUT],
                                      apply_calib(z_h_val, best_k, best_c, best_tau)), 5)
    corr_val = float(np.corrcoef(z_h_val, z_s_val)[0, 1])
    single_score = round(rmsle_z(Y_raw[HOLDOUT], z_s_val), 5)
    blend_val_calib = apply_calib(best_alpha * z_h_val + (1 - best_alpha) * z_s_val,
                                  best_k, best_c, best_tau)
    blend_score = round(rmsle_z(Y_raw[HOLDOUT], blend_val_calib), 5)

    board = sorted({**solo_hurdle,
                    "HURDLE_STACK": stack_raw_score,
                    "HURDLE_CALIB": stack_calib_score,
                    f"BLEND_a{best_alpha}_calib": blend_score,
                    "SINGLE_lgbm_ref": single_score}.items(), key=lambda kv: kv[1])
    print("LEADERBOARD fold_03 (stack_v3 raw 1.67278 / calib 1.67150 | exp01 1.70261):")
    for n, s in board:
        tag = ("HRD" if "HURDLE" in n or "BLEND" in n
               else "ref" if "SINGLE" in n else "mbr")
        print(f"  {s:.5f}  [{tag}] {n}")
    print(f"  corr(hurdle, single) fold_03={corr_val:.4f}")

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
        assert chk["user_id"].equals(order["user_id"]), "user_id order mismatch"
        return {"file": str(out), "pred_mean": round(float(chk["predict"].mean()), 2),
                "zeros_share": round(float((chk["predict"] == 0).mean()), 4)}

    subs = {
        "hurdle": save(z_h_val, "submission_hurdle.csv"),
        "hurdle_calib": save(apply_calib(z_h_val, best_k, best_c, best_tau),
                             "submission_hurdle_calib.csv"),
        "hurdle_blend_calib": save(blend_val_calib,
                                   "submission_hurdle_blend_calib.csv"),
    }
    for kk, vv in subs.items():
        print(f"  {vv['file']} mean={vv['pred_mean']} zeros={vv['zeros_share']}")

    # ---------- STAGE F: expectation E[y|X] = P(buy) * E[y|buy] (raw space) ----------
    # Тождество в RAW-пространстве: E[y|X] = P(buy|X)*E[y|buy,X]. Текущий сабмит
    # expm1(p*z2) систематически занижает сумму (Йенсен): E[expm1(z)] >= expm1(E[z]).
    # Коррекции E[y|buy] строятся по OOF-остаткам покупателей inner-фолда; выбор
    # варианта — по калибровке СУММАРНОЙ выручки на inner-slice.
    print("\n=== STAGE F: expectation variants (revenue view) ===")
    buy_va = oof_buy[va_mask]
    y_va = oof_yraw[va_mask]
    e_va = (np.log1p(y_va) - z2_va)[buy_va]
    s2_ev = float(np.var(e_va, ddof=1))
    k_ratio = float(y_va[buy_va].sum() / np.expm1(z2_va[buy_va]).sum())
    p_share_fix = float(buy_va.mean() / max(p_src_va.mean(), 1e-9))
    rng_ev = np.random.default_rng(SEED)
    e_smp = rng_ev.choice(e_va, size=min(len(e_va), 2000), replace=False)

    def ev_smear(z2: np.ndarray) -> np.ndarray:
        out = np.empty_like(z2)
        for s in range(0, z2.shape[0], 20_000):
            m = z2[s:s + 20_000][:, None]
            out[s:s + 20_000] = np.mean(
                np.expm1(np.clip(m + e_smp[None, :], None, 30.0)), axis=1)
        return out

    EV_CORR = {
        "base": lambda z2: np.expm1(np.clip(z2, None, 30.0)),
        "lognorm": lambda z2: np.expm1(np.clip(z2 + s2_ev / 2.0, None, 30.0)),
        "smear": ev_smear,
        "ratio": lambda z2: k_ratio * np.expm1(np.clip(z2, None, 30.0)),
    }
    EV_NAMES = list(EV_CORR) + ["smear_pcal"]

    def ev_pred(name: str, p: np.ndarray, z2: np.ndarray) -> np.ndarray:
        corr = EV_CORR["smear" if name == "smear_pcal" else name](z2)
        pp = np.clip(p * (p_share_fix if name == "smear_pcal" else 1.0), 0.0, 1.0)
        return np.clip(pp * corr, 0.0, None)

    def ev_row(name: str, pred_raw: np.ndarray, y_raw: np.ndarray) -> dict:
        pred_raw = np.clip(pred_raw, 0.0, None)
        return {"rmsle": round(rmsle_z(y_raw, np.log1p(pred_raw)), 5),
                "rev_err_pct": round(100.0 * (pred_raw.sum() - y_raw.sum())
                                     / y_raw.sum(), 2)}

    inner_tbl = {n: ev_row(n, ev_pred(n, p_src_va, z2_va), y_va) for n in EV_NAMES}
    inner_tbl["current_hurdle_raw"] = ev_row(
        "ref", np.expm1(z_h_va), y_va)
    inner_tbl["current_hurdle_calib"] = ev_row(
        "ref", np.clip(np.expm1(apply_calib(z_h_va, best_k, best_c, best_tau)), 0, None), y_va)
    print("  inner fold_02: variant -> rmsle | rev err %")
    for n, r in sorted(inner_tbl.items(), key=lambda kv: abs(kv[1]["rev_err_pct"])):
        print(f"    {r['rmsle']:.5f} | {r['rev_err_pct']:+7.2f}%  {n}")
    ev_best = min(EV_NAMES, key=lambda n: abs(inner_tbl[n]["rev_err_pct"]))
    print(f"  chosen (min |rev err| on inner): {ev_best}")

    z2_val_f = meta_full.predict(Z_val_mat)
    z2_end_f = meta_full.predict(Z_end_mat)
    pred_val_ev = ev_pred(ev_best, p_use_val, z2_val_f)
    pred_end_ev = ev_pred(ev_best, p_use_end, z2_end_f)
    ev_f03_tbl = {n: ev_row(n, ev_pred(n, p_use_val, z2_val_f), Y_raw[HOLDOUT])
                  for n in EV_NAMES}
    ev_f03_tbl["current_hurdle_raw"] = ev_row("ref", np.expm1(z_h_val), Y_raw[HOLDOUT])
    ev_f03_tbl["current_hurdle_calib"] = ev_row(
        "ref", np.clip(np.expm1(apply_calib(z_h_val, best_k, best_c, best_tau)), 0, None),
        Y_raw[HOLDOUT])
    ev_f03_tbl["single_lgbm_ref"] = ev_row("ref", np.expm1(np.clip(z_s_val, None, 30.0)),
                                           Y_raw[HOLDOUT])
    print("  fold_03 all variants (chosen first):")
    for n, r in sorted(ev_f03_tbl.items(), key=lambda kv: abs(kv[1]["rev_err_pct"])):
        print(f"    {r['rmsle']:.5f} | {r['rev_err_pct']:+7.2f}%  {n}")

    subs["hurdle_ev"] = save(np.log1p(pred_end_ev), "submission_hurdle_ev.csv")
    print(f"  {subs['hurdle_ev']['file']} mean={subs['hurdle_ev']['pred_mean']} "
          f"zeros={subs['hurdle_ev']['zeros_share']}")

    REPORT_PATH.write_text(json.dumps({
        "protocol": "A tune 00+01->02 | B OOF LOFO (clf + members-on-buyers + "
                    "single ref) | C meta E[z|buy] chosen on inner OOF | D iso/k,c,tau"
                    "/alpha on inner slice | B2 prod bases | fold_03 holdout",
        "dataset": "data/v3 (85 features)",
        "features": len(feats),
        "stage1": {"model": "LGBMClassifier(binary)", "params": best_clf_params,
                   "isotonic": bool(use_iso),
                   "auc_fold03": round(auc_val, 5),
                   "mean_p_fold03": round(float(P_val.mean()), 4),
                   "actual_buy_share_fold03": round(float(ybin_val.mean()), 4)},
        "stage2_members": MEMBERS,
        "grids": grids,
        "best_params": best_params,
        "single_ref_params": single_params,
        "meta_selection_inner_oof": meta_scores,
        "meta_chosen": best_meta_name,
        "calibration": {"chosen": {"k": best_k, "c": best_c, "tau": best_tau},
                        "inner_hurdle_raw_p": round(s_raw, 5),
                        "inner_hurdle_isotonic_p": round(s_iso, 5),
                        "inner_before_calib": round(raw_inner, 5),
                        "inner_after_calib": round(grid_rows[0][0], 5)},
        "blend_with_single": {"alpha": best_alpha,
                              "corr_inner": round(corr_inner, 4),
                              "corr_fold03": round(corr_val, 4),
                              "rmsle_fold03": blend_score},
        "rmsle_fold03": {"solo_hurdle_members": solo_hurdle,
                         "hurdle_raw": stack_raw_score,
                         "hurdle_calib": stack_calib_score,
                         "hurdle_blend_calib": blend_score,
                         "single_lgbm_ref": single_score,
                         f"hurdle_ev_{ev_best}": ev_f03_tbl[ev_best]["rmsle"]},
        "expectation": {"resid_var_buyers_inner": round(s2_ev, 4),
                        "ratio_k_inner": round(k_ratio, 4),
                        "p_share_fix_inner": round(p_share_fix, 4),
                        "inner_table": inner_tbl,
                        "chosen_by_abs_rev_err_inner": ev_best,
                        "fold03_table": ev_f03_tbl,
                        "fold03_chosen": ev_f03_tbl[ev_best]},
        "references": {"stack_v3_raw": 1.67278, "stack_v3_calib": 1.6715,
                       "exp01_catboost": 1.70261},
        "submissions": subs,
        "runtime_min": round((time.time() - t00) / 60, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nreport -> {REPORT_PATH}")
    print(f"ALL DONE in {(time.time() - t00) / 60:.1f} min")


if __name__ == "__main__":
    main()
