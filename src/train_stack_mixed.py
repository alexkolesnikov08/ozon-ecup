"""Стек смешанных семейств на блоке base+ext+pca (БЕЗ BTYD-фичей).

Мотивация: бленд bl_base_ext_pca на LB 1.66035 — лучший; проверяем, даст ли
полноценный стек на том же блоке фичей больше, и помогает ли разнообразие
алгоритмов (не только деревья).

Члены стека (9):
  деревья/бустинг: lgbm, cat, xgb, hist           (нелинейности, взаимодействия)
  линейные:        ridge, linsvr (LinearSVR)      (стабильность, простые сигналы)
  лес:             rf (RandomForest)              (устойчивость к выбросам)
  окрестности:     knn по PCA-компонентам         (другая геометрия границ)
  гибрид:          hurdle_lin (LogisticRegression P(buy) x Ridge E[y|buy])
                   — прямой учет 46% нулей таргета

Протокол как в train_stack_v2_remote.py:
  A' OOF leave-one-fold-out по fold_00..02 -> матрица Z для мета
  B' продакшн-базы на fold_00..02 (ES fold_02) -> fold_03 + fold_end
  C' мета: RidgeCV vs OLS-positive, выбор на inner-val (OOF fold_02)
  D' калибровка c x tau на inner-val
  E' скоры fold_03 + сабмиты
Все фиты чекпоинтятся (reports/ckpt_stack_mixed/state.npz) — падение/обрыв
не страшны, перезапуск продолжает.

Запуск из корня репо:
    python src/train_stack_mixed.py
Артефакты:
    reports/stack_search_mixed.json
    submissions/submission_stack_mixed.csv, submission_stack_mixed_calib.csv
"""

import json
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import nnls
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR

FEAT_DIR = Path("data/v2/features_ext")
REPORT_PATH = Path("reports/stack_search_mixed.json")
SUB_DIR = Path("submissions")
CKPT_DIR = Path("reports/ckpt_stack_mixed")
CKPT_PATH = CKPT_DIR / "state.npz"

TRAIN_FOLDS = ["fold_00", "fold_01", "fold_02"]
META_COLS = {"anchor_date", "user_id", "target"}
KNN_REFS = 120_000
SEED = 42

CALIB_C = [-0.12, -0.09, -0.06, -0.04, -0.02, 0.0, 0.02, 0.04]
CALIB_TAU = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]


def rmsle_z(y_raw, z_pred):
    lt = np.log1p(np.clip(y_raw, 0, None))
    lp = np.clip(z_pred, None, 30.0)
    return float(np.sqrt(np.mean((lt - lp) ** 2)))


class Ckpt:
    def __init__(self, path: Path):
        self.path = path
        self.data = {}
        if path.exists():
            with np.load(path) as z:
                self.data = {k: z[k] for k in z.files}

    def has(self, key):
        return key in self.data

    def get(self, key):
        return self.data[key]

    def put(self, key, arr):
        self.data[key] = arr
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.path, **self.data)


def load_folds():
    X, Y, Yraw, users = {}, {}, {}, {}
    for f in TRAIN_FOLDS + ["fold_03", "fold_end"]:
        df = pl.read_parquet(FEAT_DIR / f / "batch_*.parquet")
        users[f] = df["user_id"]
        Yraw[f] = np.clip(df["target"].to_numpy(), 0, None) if f != "fold_end" else None
        Y[f] = np.log1p(Yraw[f]) if f != "fold_end" else None
        cols = [c for c in df.columns if c not in META_COLS
                and not c.startswith(("bgnbd_", "eb_"))]
        X[f] = df.select(cols).to_numpy().astype(np.float32)
        print(f"  {f}: {X[f].shape}", flush=True)
    pca_cols = [i for i, c in enumerate(cols) if c.startswith("pca_")]
    return X, Y, Yraw, users, pca_cols


def fit_predict(name, Xtr, ytr, sw_ignore, Xs, es):
    """Возвращает список предиктов (log-space) по массивам Xs."""
    t0 = time.time()
    if name == "lgbm":
        from lightgbm import LGBMRegressor, early_stopping, log_evaluation
        m = LGBMRegressor(n_estimators=2000, learning_rate=0.04, num_leaves=63,
                          min_child_samples=300, feature_fraction=0.85,
                          random_state=SEED, n_jobs=8, verbosity=-1)
        m.fit(Xtr, ytr, eval_set=[es],
              callbacks=[early_stopping(100, verbose=False), log_evaluation(0)])
    elif name == "cat":
        from catboost import CatBoostRegressor
        m = CatBoostRegressor(iterations=2000, learning_rate=0.05, depth=8,
                              l2_leaf_reg=3, random_seed=SEED, thread_count=8,
                              verbose=0, allow_writing_files=False)
        m.fit(Xtr, ytr, eval_set=es, early_stopping_rounds=150)
    elif name == "xgb":
        from xgboost import XGBRegressor
        m = XGBRegressor(n_estimators=1800, learning_rate=0.05, max_depth=6,
                         min_child_weight=100, colsample_bytree=0.9,
                         subsample=0.85, tree_method="hist", random_state=SEED,
                         n_jobs=8, verbosity=0)
        m.fit(Xtr, ytr, eval_set=[es], verbose=False)
    elif name == "hist":
        m = HistGradientBoostingRegressor(
            max_iter=900, learning_rate=0.04, max_leaf_nodes=63,
            l2_regularization=5.0, min_samples_leaf=80,
            early_stopping=True, validation_fraction=0.08,
            n_iter_no_change=30, random_state=SEED)
        m.fit(Xtr, ytr)
    elif name == "rf":
        m = RandomForestRegressor(
            n_estimators=200, max_depth=14, min_samples_leaf=100,
            max_features=0.5, max_samples=0.7, n_jobs=-1, random_state=SEED)
        m.fit(Xtr, ytr)
    elif name in ("ridge", "linsvr", "knn", "hurdle_lin"):
        # x_* ratio-фичи содержат NaN (нулевой знаменатель): линейные модели
        # и knn их не принимают — заполняем 0 (политика как у base-блока).
        Xtr = np.nan_to_num(Xtr, nan=0.0, posinf=0.0, neginf=0.0)
        Xs = [np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0) for x in Xs]
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr)
        Xs_s = [sc.transform(x) for x in Xs]
        if name == "ridge":
            m = Ridge(alpha=10.0)
            m.fit(Xtr_s, ytr)
            preds = [np.clip(p, None, 30.0) for p in (m.predict(x) for x in Xs_s)]
        elif name == "linsvr":
            m = LinearSVR(C=0.5, epsilon=0.1, dual=True, tol=1e-3, max_iter=20000,
                          random_state=SEED)
            m.fit(Xtr_s, ytr)
            preds = [np.clip(p, None, 30.0) for p in (m.predict(x) for x in Xs_s)]
        elif name == "knn":
            rng = np.random.default_rng(SEED)
            ref = rng.choice(len(Xtr_s), size=min(KNN_REFS, len(Xtr_s)),
                             replace=False)
            m = KNeighborsRegressor(n_neighbors=100, weights="distance",
                                    algorithm="brute", n_jobs=-1)
            m.fit(Xtr_s[ref][:, _PCA_SLICE[0]:_PCA_SLICE[1]]
                  if isinstance(_PCA_SLICE, tuple) else Xtr_s[ref], ytr[ref])
            preds = [np.clip(p, None, 30.0)
                     for p in (m.predict(x[:, _PCA_SLICE[0]:_PCA_SLICE[1]]
                                         if isinstance(_PCA_SLICE, tuple) else x)
                               for x in Xs_s)]
        else:  # hurdle_lin
            ybin = (ytr > 0).astype(np.int8)
            clf = LogisticRegression(C=1.0, max_iter=3000, random_state=SEED)
            clf.fit(Xtr_s, ybin)
            mask = ytr > 0
            reg = Ridge(alpha=10.0).fit(Xtr_s[mask], ytr[mask])
            preds = []
            for x in Xs_s:
                p = np.clip(clf.predict_proba(x)[:, 1], 0.0, 1.0)
                e_raw = np.clip(np.expm1(np.clip(reg.predict(x), None, 30.0)),
                                0, None)
                preds.append(np.clip(np.log1p(p * e_raw), None, 30.0))
        print(f"      fit {name} {time.time() - t0:.0f}s", flush=True)
        return preds
    preds = [np.clip(p, None, 30.0) for p in
             (m.predict(x) for x in Xs)]
    print(f"      fit {name} {time.time() - t0:.0f}s", flush=True)
    return preds


MEMBERS = ["lgbm", "cat", "xgb", "hist", "ridge", "linsvr", "rf", "knn",
           "hurdle_lin"]
_PCA_SLICE = None  # (start, stop) колонок pca_ для knn; заполняется в main


def main():
    global _PCA_SLICE
    ckpt = Ckpt(CKPT_PATH)
    print("loading folds (base+ext+pca, без BTYD)...", flush=True)
    X, Y, Yraw, users, pca_idx = load_folds()
    _PCA_SLICE = (min(pca_idx), max(pca_idx) + 1)

    # ---------- STAGE A': OOF matrix ----------
    print("\n=== STAGE A': OOF leave-one-fold-out ===", flush=True)
    oof_parts = {m: [] for m in MEMBERS}
    fold_id = []
    for k in TRAIN_FOLDS:
        tr = [f for f in TRAIN_FOLDS if f != k]
        Xtr = np.concatenate([X[f] for f in tr])
        ytr = np.concatenate([Y[f] for f in tr])
        for m in MEMBERS:
            key = f"oof__{m}__{k}"
            if ckpt.has(key):
                pred = ckpt.get(key)
                print(f"  {key}: checkpoint hit", flush=True)
            else:
                pred = fit_predict(m, Xtr, ytr, None, [X[k]], (X[k], Y[k]))[0]
                ckpt.put(key, pred)
            oof_parts[m].append(pred)
        fold_id += [k] * len(X[k])
        del Xtr, ytr
    Z_oof = {m: np.concatenate(v) for m, v in oof_parts.items()}
    oof_y = np.concatenate([Y[f] for f in TRAIN_FOLDS])
    oof_yraw = np.concatenate([Yraw[f] for f in TRAIN_FOLDS])
    fold_id = np.array(fold_id)

    # ---------- STAGE B': production bases ----------
    print("\n=== STAGE B': production bases (00+01+02, ES fold_02) ===",
          flush=True)
    Xtr_all = np.concatenate([X[f] for f in TRAIN_FOLDS])
    ytr_all = np.concatenate([Y[f] for f in TRAIN_FOLDS])
    es_ref = (X["fold_02"], Y["fold_02"])
    Z_val_m, Z_end_m = {}, {}
    for m in MEMBERS:
        kv, ke = f"val__{m}", f"end__{m}"
        if ckpt.has(kv) and ckpt.has(ke):
            Z_val_m[m], Z_end_m[m] = ckpt.get(kv), ckpt.get(ke)
            print(f"  {m}: checkpoint hit", flush=True)
        else:
            preds = fit_predict(m, Xtr_all, ytr_all, None,
                                [X["fold_03"], X["fold_end"]], es_ref)
            Z_val_m[m], Z_end_m[m] = preds
            ckpt.put(kv, preds[0])
            ckpt.put(ke, preds[1])
    Z_val = np.column_stack([Z_val_m[m] for m in MEMBERS])
    Z_end = np.column_stack([Z_end_m[m] for m in MEMBERS])
    del Xtr_all, ytr_all

    print("\n=== solo scores fold_03 ===")
    solo = {}
    for i, m in enumerate(MEMBERS):
        solo[m] = round(rmsle_z(Yraw["fold_03"], Z_val[:, i]), 5)
        print(f"  {m:11s} {solo[m]}")

    # ---------- STAGE C': meta ----------
    print("\n=== STAGE C': meta (RidgeCV vs OLS-pos) ===")
    tr_mask = np.isin(fold_id, ["fold_00", "fold_01"])
    va_mask = fold_id == "fold_02"
    Zf, yf = np.column_stack([Z_oof[m] for m in MEMBERS])[tr_mask], oof_y[tr_mask]
    Zv, yv_raw = np.column_stack([Z_oof[m] for m in MEMBERS])[va_mask], oof_yraw[va_mask]

    def apply_ols_pos(w, Z):
        return Z @ w[:-1] + w[-1]

    ridge_m = RidgeCV(alphas=[0.3, 1, 3, 10, 30, 100]).fit(Zf, yf)
    w_pos, _ = nnls(np.column_stack([Zf, np.ones(len(Zf))]), yf)
    cands = {
        "ridgecv": (ridge_m.predict(Zv), None),
        "ols_pos": (apply_ols_pos(w_pos, Zv), ("ols_pos", w_pos)),
    }
    meta_scores = {k: round(rmsle_z(yv_raw, zv), 5) for k, (zv, _) in cands.items()}
    best_meta = min(meta_scores, key=meta_scores.get)
    print(" ", meta_scores, "->", best_meta)
    if best_meta == "ridgecv":
        blend_val, blend_end = ridge_m.predict(Z_val), ridge_m.predict(Z_end)
    else:
        blend_val, blend_end = (apply_ols_pos(w_pos, Z_val),
                                apply_ols_pos(w_pos, Z_end))

    # ---------- STAGE D': calibration ----------
    print("\n=== STAGE D': calibration c x tau (inner val) ===")
    zv_oof = (blend_val if best_meta != "ridgecv" else blend_val)
    best = min(
        ((rmsle_z(yv_raw, np.where(zv_oof + c < tau, 0.0, zv_oof + c)), c, tau)
         for c in CALIB_C for tau in CALIB_TAU), key=lambda x: x[0])
    score_cal_inner, c_best, tau_best = best
    print(f"  c={c_best} tau={tau_best} (inner {score_cal_inner:.5f})")

    val_cal = np.where(blend_val + c_best < tau_best, 0.0, blend_val + c_best)
    end_cal = np.where(blend_end + c_best < tau_best, 0.0, blend_end + c_best)

    # ---------- STAGE E': scores & submissions ----------
    print("\n=== STAGE E': fold_03 scores ===")
    score_raw = rmsle_z(Yraw["fold_03"], blend_val)
    score_cal = rmsle_z(Yraw["fold_03"], val_cal)
    print(f"  stack_mixed raw {score_raw:.5f} | calib {score_cal:.5f}")

    s = pl.read_csv("sample_submit.csv").select("user_id").with_row_index("__ord")
    uids = users["fold_end"].cast(pl.Int64)
    for tag, pred in (("stack_mixed", blend_end), ("stack_mixed_calib", end_cal)):
        out = SUB_DIR / f"submission_{tag}.csv"
        (pl.DataFrame({"user_id": uids, "predict": np.clip(np.expm1(pred), 0, None)})
         .join(s, on="user_id", how="inner").sort("__ord")
         .drop("__ord").select(["user_id", "predict"]).write_csv(out))
        chk = pl.read_csv(out)
        print(f"  saved {out} | rows {chk.height} | mean "
              f"{chk['predict'].mean():.2f} | order ok "
              f"{chk['user_id'].to_list() == s['user_id'].to_list()}")

    REPORT_PATH.write_text(json.dumps({
        "protocol": "mixed stack base+ext+pca (no BTYD) | OOF LOFO | ES fold_02 | holdout fold_03",
        "block": "base+ext+pca",
        "members": MEMBERS, "solo_fold03": solo,
        "meta_scores_inner": meta_scores, "meta": best_meta,
        "calibration": {"c": c_best, "tau": tau_best},
        "rmsle_fold03": {"stack_raw": round(score_raw, 5),
                         "stack_calib": round(score_cal, 5)},
        "submissions": {"stack_mixed": str(SUB_DIR / "submission_stack_mixed.csv"),
                        "stack_mixed_calib": str(SUB_DIR / "submission_stack_mixed_calib.csv")},
    }, indent=1))
    print(f"\nreport -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
