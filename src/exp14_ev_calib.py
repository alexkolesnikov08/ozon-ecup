"""exp14 — EV-калибровка мат-ожидания на лучшем бейзлайне (bl_base_ext_pca).

Бейзлайн: одиночный CatBoost (exp01-конфиг, RMSE на log1p) на base+ext+pca —
локальный fold_03 1.64487, LB 1.66035 (лучший результат проекта). Строим поверх
него hurdle E[y|X] = P(buy) x E[y|buy] и калибруем мат-ожидание:

  Stage 1  CatBoostClassifier P(buy); тюнинг грида и отбор по GINI = 2*AUC - 1
           на inner-slice (train 00+01 -> val 02);
  Stage 2  CatBoost RMSE на покупателях E[z|buy] (как у бейзлайна);
  H1  пер-бинная коррекция по предсказанному уровню z2 (десили/квинтили среди
      покупателей inner), фактор shrunk к глобальному (EB, lam=1000);
  H2  пер-сегментная коррекция по сетке recency x intent (exp03-сегменты),
      shrinkage lam=500; стабильность факторов fold_02 vs pooled(00+01) как
      критерий переноса.

Протокол: A тюнинг clf 00+01->02 по Gini | B LOFO OOF по fold_00..02 | inner =
fold_02, fold_03 холдаут для валидации переноса; финальная коррекция рефитится
на pooled OOF (свежайшие данные) -> сабмит fold_end.

Run from repo root:
    .venv/bin/python src/exp14_ev_calib.py [--quick] [--fresh]
Artifacts:
    reports/exp14_ev_calib.json,
    submissions/submission_hurdle_bl_ev.csv
    reports/ckpt_exp14_gini/state.npz (чекпоинт, авто-resume)
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import roc_auc_score

FEAT_DIR = Path("data/v2/features_ext")
REPORT_PATH = Path("reports/exp14_ev_calib.json")
CKPT_PATH = Path("reports/ckpt_exp14_gini/state.npz")
SUB_DIR = Path("submissions")

SEED = 42
TRAIN_FOLDS = ["fold_00", "fold_01", "fold_02"]
INNER_VAL_FOLD = "fold_02"
HOLDOUT = "fold_03"
ALL_FOLDS = TRAIN_FOLDS + [HOLDOUT, "fold_end"]

CAT_PARAMS = dict(loss_function="RMSE", learning_rate=0.05, depth=8,
                  l2_leaf_reg=3, n_estimators=1000, thread_count=-1,
                  random_seed=SEED, verbose=0, allow_writing_files=False)

CLF_GRID = [
    {"iterations": 1200, "learning_rate": 0.05, "depth": d, "l2_leaf_reg": l2}
    for d in (6, 8) for l2 in (3, 9)
]

RECENCY_EDGES = [0, 7, 14, 30, 90]
RECENCY_LABELS = ["0-7d", "7-14d", "14-30d", "30-90d", ">90d"]
INTENT_EDGES = [0.5, 1, 4, 12]
INTENT_LABELS = ["s=0", "s=1", "s=2-4", "s=5-12", "s>=12"]

H1_BINS_GRID = [5, 10, 20]
LAM_BIN = 1000.0
LAM_SEG = 500.0

QUICK_SCALE = {"iterations": 0.25}


def scale_params(params: dict) -> dict:
    return {k: (int(v * QUICK_SCALE[k]) if k in QUICK_SCALE else v)
            for k, v in params.items()}


def gini(y_true: np.ndarray, p: np.ndarray) -> float:
    return 2.0 * float(roc_auc_score(y_true, p)) - 1.0


def load_fold(name: str) -> pl.DataFrame:
    return pl.read_parquet(FEAT_DIR / name / "batch_*.parquet")


def gini(y_true: np.ndarray, p: np.ndarray) -> float:
    return 2.0 * float(roc_auc_score(y_true, p)) - 1.0


def make_clf(params: dict) -> CatBoostClassifier:
    return CatBoostClassifier(**params, loss_function="Logloss",
                              eval_metric="AUC", thread_count=-1,
                              random_seed=SEED, verbose=0,
                              allow_writing_files=False)


def seg_labels(df: pl.DataFrame) -> np.ndarray:
    rec = df["recency_to_ord_days"].to_numpy()
    s14 = df["x_searches_sum_14d"].to_numpy()
    rec_idx = np.minimum(np.digitize(rec, RECENCY_EDGES, right=False),
                         len(RECENCY_LABELS) - 1)
    rec_lab = np.array(RECENCY_LABELS, dtype=object)[rec_idx]
    rec_lab = np.where(rec >= 999.0, "never", rec_lab)
    s_idx = np.digitize(s14, INTENT_EDGES, right=False)
    s_lab = np.array(INTENT_LABELS, dtype=object)[s_idx]
    return np.array([f"{r}|{s}" for r, s in zip(rec_lab, s_lab)], dtype=object)


def rmsle_z(y_raw: np.ndarray, z_pred: np.ndarray) -> float:
    lt = np.log1p(np.clip(y_raw, 0, None))
    lp = np.clip(z_pred, None, 30.0)
    return float(np.sqrt(np.mean((lt - lp) ** 2)))


def ev_row(pred_raw: np.ndarray, y_raw: np.ndarray) -> dict:
    pred_raw = np.clip(pred_raw, 0.0, None)
    return {"rmsle": round(rmsle_z(y_raw, np.log1p(pred_raw)), 5),
            "rev_err_pct": round(100.0 * (pred_raw.sum() - y_raw.sum())
                                 / y_raw.sum(), 2)}


def binned_ratio_fit(m: np.ndarray, y: np.ndarray, n_bins: int):
    edges = np.quantile(m, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
    idx = np.searchsorted(edges, m)
    k_glob = y.sum() / np.expm1(m).sum()
    centers, factors, counts = [], [], []
    for b in range(n_bins):
        mb = m[idx == b]
        nb = len(mb)
        if nb == 0:
            continue
        k_b = y[idx == b].sum() / np.expm1(mb).sum()
        f_b = (nb * k_b + LAM_BIN * k_glob) / (nb + LAM_BIN)
        centers.append(float(mb.mean()))
        factors.append(float(f_b))
        counts.append(nb)
    return np.array(centers), np.array(factors), np.array(counts), float(k_glob)


def make_binned_apply(centers, factors):
    def apply(m: np.ndarray) -> np.ndarray:
        return np.interp(m, centers, factors)
    return apply


def seg_factor_fit(seg: np.ndarray, m: np.ndarray, y: np.ndarray) -> dict:
    k_glob = y.sum() / np.expm1(m).sum()
    out = {"__global__": (int(len(m)), float(k_glob))}
    for s in np.unique(seg):
        mask = seg == s
        ns = int(mask.sum())
        if ns < 50:
            out[str(s)] = (ns, float(k_glob))
            continue
        k_s = y[mask].sum() / np.expm1(m[mask]).sum()
        f_s = (ns * k_s + LAM_SEG * k_glob) / (ns + LAM_SEG)
        out[str(s)] = (ns, float(f_s))
    return out


def make_seg_apply(factors: dict):
    k_glob = factors["__global__"][1]

    def apply(seg: np.ndarray, m: np.ndarray) -> np.ndarray:
        f = np.full(len(m), k_glob, dtype=np.float64)
        for s, (_, fs) in factors.items():
            if s == "__global__":
                continue
            f[seg == s] = fs
        return f
    return apply


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()
    quick = args.quick
    cat_params = scale_params(CAT_PARAMS) if quick else CAT_PARAMS
    clf_grid = [scale_params(g) for g in CLF_GRID[:1]] if quick else CLF_GRID

    t00 = time.time()
    print("loading folds...", flush=True)
    data = {}
    for f in ALL_FOLDS:
        df = load_fold(f)
        meta_cols = {"anchor_date", "user_id", "target"}
        feats = [c for c in df.columns
                 if c not in meta_cols and not c.startswith(("bgnbd_", "eb_"))]
        X = df.select(feats).to_numpy().astype(np.float32)
        yraw = np.clip(df["target"].to_numpy(), 0, None) \
            if "target" in df.columns else None
        data[f] = {"X": X, "df": df, "feats": feats, "yraw": yraw,
                   "seg": seg_labels(df)}
        ys = f", zeros={1 - (yraw > 0).mean():.3f}" if yraw is not None else ""
        print(f"  {f}: {df.height} rows, {len(feats)} feats{ys}", flush=True)
    n_feats = len(data["fold_00"]["feats"])

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

    # ---------- STAGE A: clf tuning by Gini (train 00+01 -> val 02) ----------
    if have("best_clf_params"):
        best_clf_params = json.loads(str(state["best_clf_params"]))
        clf_tune_tbl = json.loads(str(state["clf_tune_tbl"]))
        print("stage A loaded from checkpoint")
    else:
        print("\n=== STAGE A: clf tuning (Gini, train 00+01 -> val 02) ===")
        Xa = np.concatenate([data[f]["X"] for f in TRAIN_FOLDS[:2]])
        ya = np.concatenate([(data[f]["yraw"] > 0).astype(np.int8)
                             for f in TRAIN_FOLDS[:2]])
        Xv = data[INNER_VAL_FOLD]["X"]
        yv = (data[INNER_VAL_FOLD]["yraw"] > 0).astype(np.int8)
        rows = []
        for i, params in enumerate(clf_grid):
            t0 = time.time()
            model = make_clf(params)
            model.fit(Xa, ya, eval_set=(Xv, yv), early_stopping_rounds=100,
                      metric_period=100)
            g = gini(yv, model.predict_proba(Xv)[:, 1])
            rows.append((g, params))
            print(f"  [clf {i + 1}/{len(clf_grid)}] Gini={g:.5f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        best_clf_params = max(rows, key=lambda r: r[0])[1]
        clf_tune_tbl = {json.dumps(p): round(g, 5) for g, p in rows}
        print(f"  BEST clf by Gini: {best_clf_params}", flush=True)
        state["best_clf_params"] = json.dumps(best_clf_params)
        state["clf_tune_tbl"] = json.dumps(clf_tune_tbl)
        save_ckpt()

    # ---------- STAGE B: LOFO OOF ----------
    if have("P_oof", "Z2_oof", "oof_yraw", "oof_fold_id", "oof_seg"):
        P_oof, Z2_oof = state["P_oof"], state["Z2_oof"]
        oof_yraw, oof_fold_id = state["oof_yraw"], state["oof_fold_id"]
        oof_seg = state["oof_seg"]
        print("stage B loaded from checkpoint")
    else:
        print("\n=== STAGE B: LOFO OOF (cat-clf P(buy) + cat-on-buyers) ===")
        P_parts, Z_parts, fid, yraw_p, seg_p = [], [], [], [], []
        for k in TRAIN_FOLDS:
            tr = [f for f in TRAIN_FOLDS if f != k]
            Xtr = np.concatenate([data[f]["X"] for f in tr])
            yraw_tr = np.concatenate([data[f]["yraw"] for f in tr])
            buy_tr = yraw_tr > 0

            t0 = time.time()
            clf = make_clf(best_clf_params)
            clf.fit(Xtr, (yraw_tr > 0).astype(np.int8),
                    eval_set=(data[k]["X"],
                              (data[k]["yraw"] > 0).astype(np.int8)),
                    early_stopping_rounds=100, metric_period=200)
            pk = clf.predict_proba(data[k]["X"])[:, 1].astype(np.float64)
            print(f"  [{k}] cat-clf {time.time() - t0:.0f}s "
                  f"Gini={gini((data[k]['yraw'] > 0).astype(np.int8), pk):.5f}",
                  flush=True)

            t0 = time.time()
            cat = CatBoostRegressor(**cat_params)
            cat.fit(Xtr[buy_tr], np.log1p(yraw_tr[buy_tr]))
            zk = np.clip(cat.predict(data[k]["X"]), None, 30.0)
            print(f"  [{k}] cat-on-buyers {time.time() - t0:.0f}s", flush=True)

            P_parts.append(pk)
            Z_parts.append(zk)
            fid += [k] * data[k]["X"].shape[0]
            yraw_p.append(data[k]["yraw"])
            seg_p.append(data[k]["seg"])

        P_oof = np.concatenate(P_parts)
        Z2_oof = np.concatenate(Z_parts)
        oof_yraw = np.concatenate(yraw_p)
        oof_fold_id = np.array(fid)
        oof_seg = np.concatenate(seg_p)
        state.update({"P_oof": P_oof, "Z2_oof": Z2_oof, "oof_yraw": oof_yraw,
                      "oof_fold_id": oof_fold_id, "oof_seg": oof_seg})
        save_ckpt()

    # ---------- STAGE B2: prod bases (train fold_00..02) ----------
    if have("P_val", "P_end", "Z_val", "Z_end", "Seg_val", "Seg_end",
            "Zs_val", "Zs_end"):
        P_val, P_end = state["P_val"], state["P_end"]
        Z_val, Z_end = state["Z_val"], state["Z_end"]
        Seg_val, Seg_end = state["Seg_val"], state["Seg_end"]
        Zs_val, Zs_end = state["Zs_val"], state["Zs_end"]
        print("stage B2 loaded from checkpoint")
    else:
        print("\n=== STAGE B2: prod bases (train fold_00..02) ===")
        Xtr = np.concatenate([data[f]["X"] for f in TRAIN_FOLDS])
        yraw_tr = np.concatenate([data[f]["yraw"] for f in TRAIN_FOLDS])
        buy_tr = yraw_tr > 0
        es = (data[INNER_VAL_FOLD]["X"],
              (data[INNER_VAL_FOLD]["yraw"] > 0).astype(np.int8))

        t0 = time.time()
        clf = make_clf(best_clf_params)
        clf.fit(Xtr, (yraw_tr > 0).astype(np.int8), eval_set=es,
                early_stopping_rounds=100, metric_period=200)
        P_val, P_end = [clf.predict_proba(data[f]["X"])[:, 1].astype(np.float64)
                        for f in (HOLDOUT, "fold_end")]
        print(f"  cat-clf {time.time() - t0:.0f}s", flush=True)

        t0 = time.time()
        cat = CatBoostRegressor(**cat_params)
        cat.fit(Xtr[buy_tr], np.log1p(yraw_tr[buy_tr]))
        Z_val, Z_end = [np.clip(cat.predict(data[f]["X"]), None, 30.0)
                        for f in (HOLDOUT, "fold_end")]
        print(f"  cat-on-buyers {time.time() - t0:.0f}s", flush=True)

        t0 = time.time()
        cat_all = CatBoostRegressor(**cat_params)
        cat_all.fit(Xtr, np.log1p(yraw_tr))
        Zs_val, Zs_end = [np.clip(cat_all.predict(data[f]["X"]), None, 30.0)
                          for f in (HOLDOUT, "fold_end")]
        print(f"  single-cat-all-rows {time.time() - t0:.0f}s", flush=True)

        Seg_val = data[HOLDOUT]["seg"]
        Seg_end = data["fold_end"]["seg"]
        state.update({"P_val": P_val, "P_end": P_end, "Z_val": Z_val,
                      "Z_end": Z_end, "Seg_val": Seg_val, "Seg_end": Seg_end,
                      "Zs_val": Zs_val, "Zs_end": Zs_end})
        save_ckpt()

    # ---------- STAGE C: corrections on inner (fit fold_02) ----------
    print("\n=== STAGE C: H1/H2 corrections fit on inner (fold_02) ===")
    va = oof_fold_id == INNER_VAL_FOLD
    p_va, z2_va = P_oof[va], Z2_oof[va]
    y_va, seg_va = oof_yraw[va], oof_seg[va].astype(object)
    buy_va = y_va > 0

    # стабильность факторов: fold_02 vs pooled(fold_00+01)
    tr_prev = oof_fold_id != INNER_VAL_FOLD
    stab = {}
    for n_bins in H1_BINS_GRID:
        _, fac_a, _, _ = binned_ratio_fit(z2_va[buy_va], y_va[buy_va], n_bins)
        c_b, fac_b, _, _ = binned_ratio_fit(
            Z2_oof[tr_prev][oof_yraw[tr_prev] > 0],
            oof_yraw[tr_prev][oof_yraw[tr_prev] > 0], n_bins)
        if len(fac_a) == len(fac_b):
            stab[f"h1_bins={n_bins}"] = round(
                float(np.median(np.abs(fac_a / fac_b - 1))) * 100, 2)
    seg_a = seg_factor_fit(seg_va[buy_va], z2_va[buy_va], y_va[buy_va])
    prev_buy = oof_yraw[tr_prev] > 0
    seg_b = seg_factor_fit(oof_seg[tr_prev][prev_buy],
                           Z2_oof[tr_prev][prev_buy],
                           oof_yraw[tr_prev][prev_buy])
    deltas = [abs(seg_a[s][1] / seg_b[s][1] - 1)
              for s in seg_a if s in seg_b and s != "__global__"]
    stab["h2_segments"] = round(float(np.median(deltas)) * 100, 2)
    print(f"  stability (median |factor ratio - 1|, %): {stab}")
    print("  h2 per-segment factors fold_02 vs 00+01:")
    for s in sorted(seg_a):
        if s == "__global__":
            continue
        ka, kb = seg_a[s][1], seg_b.get(s, (0, float("nan")))[1]
        print(f"    {s:<16} n={seg_a[s][0]:>7} k={ka:.3f} vs {kb:.3f} "
              f"(d={abs(ka / kb - 1) * 100:.1f}%)")

    corrections: dict[str, callable] = {
        "base": lambda p, m, sg: p * np.expm1(np.clip(m, None, 30.0)),
    }
    bin_fits = {}
    for n_bins in H1_BINS_GRID:
        c, f, n, kg = binned_ratio_fit(z2_va[buy_va], y_va[buy_va], n_bins)
        bin_fits[n_bins] = (c, f, n, kg)
        corrections[f"h1_bins={n_bins}"] = (
            lambda p, m, sg, c=c, f=f: p * np.expm1(np.clip(m, None, 30.0))
            * make_binned_apply(c, f)(m))
    seg_factors = seg_factor_fit(seg_va[buy_va], z2_va[buy_va], y_va[buy_va])
    seg_apply = make_seg_apply(seg_factors)
    corrections["h2_segments"] = (
        lambda p, m, sg: p * np.expm1(np.clip(m, None, 30.0)) * seg_apply(sg, m))

    inner_tbl = {}
    for name, fn in corrections.items():
        pred = fn(p_va, z2_va, seg_va)
        inner_tbl[name] = ev_row(pred, y_va)
    print("  inner fold_02:")
    for n, r in sorted(inner_tbl.items(), key=lambda kv: abs(kv[1]["rev_err_pct"])):
        print(f"    {r['rmsle']:.5f} | {r['rev_err_pct']:+7.2f}%  {n}")
    ev_best = min(inner_tbl, key=lambda n: abs(inner_tbl[n]["rev_err_pct"]))
    print(f"  chosen by |rev err| on inner: {ev_best}")

    # ---------- STAGE D: validate transfer on fold_03 ----------
    print("\n=== STAGE D: fold_03 validation ===")
    y_val = data[HOLDOUT]["yraw"]
    f03_tbl = {}
    for name, fn in corrections.items():
        pred = fn(P_val, Z_val, Seg_val)
        f03_tbl[name] = ev_row(pred, y_val)
    f03_tbl["ref_single_bl"] = ev_row(np.expm1(np.clip(
        Zs_val, None, 30.0)), y_val)
    ybin_val = (y_val > 0).astype(np.int8)
    clf_gini_f03 = gini(ybin_val, P_val)
    print(f"  stage1 fold_03: Gini={clf_gini_f03:.5f} "
          f"(AUC={(clf_gini_f03 + 1) / 2:.5f}); mean p={P_val.mean():.4f} "
          f"vs buy share={ybin_val.mean():.4f}")
    print("  fold_03 all variants:")
    for n, r in sorted(f03_tbl.items(), key=lambda kv: abs(kv[1]["rev_err_pct"])):
        tag = " <- chosen" if n == ev_best else ""
        print(f"    {r['rmsle']:.5f} | {r['rev_err_pct']:+7.2f}%  {n}{tag}")

    # ---------- STAGE E: refit on pooled OOF -> submission ----------
    print("\n=== STAGE E: pooled refit & submission ===")
    buy_all = oof_yraw > 0

    def refit(name: str):
        if name == "base":
            return lambda p, m, sg: p * np.expm1(np.clip(m, None, 30.0))
        if name.startswith("h1_bins="):
            nb = int(name.split("=")[1])
            c, f, _, _ = binned_ratio_fit(Z2_oof[buy_all], oof_yraw[buy_all], nb)
            ba = make_binned_apply(c, f)
            return lambda p, m, sg: p * np.expm1(np.clip(m, None, 30.0)) * ba(m)
        sf = seg_factor_fit(oof_seg[buy_all], Z2_oof[buy_all], oof_yraw[buy_all])
        sa = make_seg_apply(sf)
        return lambda p, m, sg: p * np.expm1(np.clip(m, None, 30.0)) * sa(sg, m)

    final_fn = refit(ev_best)
    pred_end = final_fn(P_end, Z_end, Seg_end)
    pred_val_best = final_fn(P_val, Z_val, Seg_val)
    pooled_check = ev_row(pred_val_best, y_val)
    print(f"  pooled-refit [{ev_best}] on fold_03: rmsle={pooled_check['rmsle']:.5f} "
          f"rev={pooled_check['rev_err_pct']:+.2f}%")

    SUB_DIR.mkdir(exist_ok=True)
    sample = pl.read_csv("sample_submit.csv")
    order = sample.select("user_id").with_row_index("__ord")
    uids = data["fold_end"]["df"]["user_id"].cast(pl.Int64)
    pred_end = np.clip(pred_end, 0.0, None)
    assert np.isfinite(pred_end).all() and (pred_end >= 0).all()
    out = SUB_DIR / "submission_hurdle_bl_ev.csv"
    (pl.DataFrame({"user_id": uids, "predict": pred_end})
     .join(order, on="user_id", how="inner").sort("__ord").drop("__ord")
     .select(["user_id", "predict"]).write_csv(out))
    chk = pl.read_csv(out)
    assert chk.height == 250_000 and chk["user_id"].equals(order["user_id"])
    sub_info = {"file": str(out), "variant": ev_best,
                "pred_mean": round(float(chk["predict"].mean()), 2),
                "pred_sum_mln": round(float(chk["predict"].sum()) / 1e6, 2)}
    print(f"  saved {out} mean={sub_info['pred_mean']} "
          f"sum={sub_info['pred_sum_mln']}M")

    REPORT_PATH.write_text(json.dumps({
        "protocol": "LOFO OOF 00..02 (clf P(buy) + cat-on-buyers E[z|buy]) | "
                    "corrections fit inner=fold_02 | transfer check fold_03 | "
                    "final refit pooled OOF -> fold_end submission",
        "baseline": "single CatBoost exp01-config on base+ext+pca "
                    "(LB 1.66035, local fold_03 1.64487)",
        "stage1": {"model": "CatBoostClassifier(Logloss, eval_metric=AUC)",
                   "selection_metric": "Gini = 2*AUC - 1",
                   "params": best_clf_params,
                   "tune_table_inner": clf_tune_tbl,
                   "gini_fold03": round(clf_gini_f03, 5),
                   "mean_p_fold03": round(float(P_val.mean()), 4),
                   "buy_share_fold03": round(float(ybin_val.mean()), 4)},
        "quick": bool(quick),
        "stability_median_pct": stab,
        "h2_segment_factors_inner": {s: {"n": seg_a[s][0], "k": round(seg_a[s][1], 4)}
                                     for s in seg_a if s != "__global__"},
        "inner_table": inner_tbl,
        "chosen_by_abs_rev_err_inner": ev_best,
        "fold03_table": f03_tbl,
        "fold03_pooled_refit_chosen": pooled_check,
        "submission": sub_info,
        "runtime_min": round((time.time() - t00) / 60, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nreport -> {REPORT_PATH}")
    print(f"ALL DONE in {(time.time() - t00) / 60:.1f} min")


if __name__ == "__main__":
    main()
