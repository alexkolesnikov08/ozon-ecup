"""b01 part C: shape calibration of ready-made OOF predictions in z-space.

Works on artifacts of src/b01_train.py:
- oof_pool.parquet       honest LOFO preds (fold_00..03), 66-feat exp02 config
- pseudo_pool.parquet    pseudo-anchor 2025-02-13 preds (model on fold_00..02)
- fold_end_pred.parquet  refit-model preds for anchor 2026-02-13
- refit_fold03.parquet   refit-model preds on fold_03 (in-sample reference)

Calibrators (no retraining of boosters):
(a) isotonic regression m*(zh)->z_true, monotone, y_min=0;
(b) zeroing: zh->0 if below threshold t*, grid {0, 0.05, ..., 2.0};
orders iso->zero and zero->iso.
HONESTY: every variant/pool/t is scored CROSS-FIT — the calibrator is fitted on
the pool minus the evaluated fold (+pseudo for pool=cvp), never on the fold
being scored. Selection = argmin mean cross-fit RMSLE (ties -> simpler variant,
pool 'cv'). Winner is refitted on the FULL pool and applied to fold_end preds
-> submissions/b01_submission_calibrated.csv.

Outputs: reports/b01_metrics.json, reports/b01_figures/*.png, submission csv.
Idempotent: recomputes everything deterministically (cheap, pure post-processing).
"""

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

ART_DIR = Path("data/v2/b01_pseudo")
FIG_DIR = Path("reports/b01_figures")
METRICS_PATH = Path("reports/b01_metrics.json")
SUB_PATH = Path("submissions/b01_submission_calibrated.csv")

CV_FOLDS = ["fold_00", "fold_01", "fold_02", "fold_03"]
T_GRID = np.round(np.arange(0.0, 2.0001, 0.05), 2)

REF_LOFO = {"fold_00": 1.81545, "fold_01": 1.77420,
            "fold_02": 1.71090, "fold_03": 1.69277}
REF_MEAN = float(np.mean(list(REF_LOFO.values())))  # 1.74833
ADOPT_MEAN_TOL = 0.003          # >= 0.3% mean improvement required
ADOPT_F03_MAX = REF_LOFO["fold_03"] - 0.001

VARIANT_RANK = {"raw": 0, "iso": 1, "zero": 2, "iso_zero": 3, "zero_iso": 4}


def log1p_clip(y: np.ndarray) -> np.ndarray:
    return np.log1p(np.clip(y, 0.0, None))


def rmsle_z(z_true: np.ndarray, z_cal: np.ndarray) -> float:
    return float(np.sqrt(np.mean((z_true - z_cal) ** 2)))


def fit_iso(zh: np.ndarray, zt: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(y_min=0.0, increasing=True, out_of_bounds="clip")
    iso.fit(zh, zt)
    return iso


def assert_monotone(iso: IsotonicRegression, name: str) -> None:
    x = np.linspace(float(np.min(iso.X_thresholds_)), float(np.max(iso.X_thresholds_)), 2001)
    p = iso.predict(x)
    assert np.all(np.diff(p) >= -1e-12), f"{name}: isotonic curve not monotone"
    assert (p >= -1e-12).all(), f"{name}: isotonic curve below 0"


def apply_variant(variant: str, iso: IsotonicRegression | None,
                  zh: np.ndarray, t: float | None) -> np.ndarray:
    if variant == "raw":
        return zh.copy()
    if variant == "iso":
        return iso.predict(zh)
    if variant == "zero":
        return np.where(zh < t, 0.0, zh)
    if variant == "iso_zero":
        z1 = iso.predict(zh)
        return np.where(z1 < t, 0.0, z1)
    if variant == "zero_iso":
        zh1 = np.where(zh < t, 0.0, zh)
        return iso.predict(zh1)
    raise ValueError(variant)


def main() -> None:
    t_start = time.time()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    M: dict = {}

    # ---------- load ----------
    oof = pl.read_parquet(ART_DIR / "oof_pool.parquet")
    pseudo = pl.read_parquet(ART_DIR / "pseudo_pool.parquet")
    fend = pl.read_parquet(ART_DIR / "fold_end_pred.parquet")
    rf03 = pl.read_parquet(ART_DIR / "refit_fold03.parquet")
    train_times = json.loads((ART_DIR / "train_times.json").read_text())
    assert oof.height == 1_000_000 and pseudo.height == 250_000
    assert fend.height == rf03.height == 250_000

    fold_lbl = oof["fold"].to_numpy()
    zh_all = oof["z_pred"].to_numpy().astype(np.float64)
    zt_all = log1p_clip(oof["target"].to_numpy())

    zh_px = pseudo["z_pred"].to_numpy().astype(np.float64)
    zt_px = log1p_clip(pseudo["target"].to_numpy())
    zh_fe = fend["z_pred"].to_numpy().astype(np.float64)
    zh_r3 = rf03["z_pred"].to_numpy().astype(np.float64)
    zt_r3 = log1p_clip(rf03["target"].to_numpy())

    masks_va = {f: fold_lbl == f for f in CV_FOLDS}

    # ---------- stage 0: reproduction of exp02 LOFO ----------
    raw_scores = {f: rmsle_z(zt_all[masks_va[f]], zh_all[masks_va[f]]) for f in CV_FOLDS}
    raw_mean = float(np.mean(list(raw_scores.values())))
    repro_devs = {f: abs(raw_scores[f] - REF_LOFO[f]) for f in CV_FOLDS}
    repro_ok = all(d <= 0.002 for d in repro_devs.values())
    print(f"raw OOF reproduction: {raw_scores} mean={raw_mean:.5f} ok={repro_ok}", flush=True)

    # equivalence of z-space and original-space RMSLE (once)
    _zt = zt_all[masks_va["fold_03"]]
    _zc = np.maximum(zh_all[masks_va["fold_03"]], 0.0)
    _y_true = oof.filter(masks_va["fold_03"])["target"].to_numpy()
    r_z = rmsle_z(_zt, _zc)
    r_o = float(np.sqrt(np.mean(
        (np.log1p(np.clip(_y_true, 0, None)) - np.log1p(np.clip(np.expm1(_zc), 0, None)))**2)))
    assert abs(r_z - r_o) < 1e-9, (r_z, r_o)

    # target diagnostics
    zero_share_by_fold = {
        f: float(np.mean(oof.filter(masks_va[f])["target"].to_numpy() == 0)) for f in CV_FOLDS
    }

    # shrinkage diagnostic: OLS slope of z_true ~ zh on pooled OOF
    slope = float(np.polyfit(zh_all, zt_all, 1)[0])

    # ---------- stage 1: cross-fit evaluation ----------
    # per (pool, fold): fitted isotonic on train part only
    isos: dict[tuple[str, str], IsotonicRegression] = {}
    for pool in ("cv", "cvp"):
        for f in CV_FOLDS:
            tr = ~masks_va[f]
            if pool == "cvp":
                zh_tr = np.concatenate([zh_all[tr], zh_px])
                zt_tr = np.concatenate([zt_all[tr], zt_px])
            else:
                zh_tr, zt_tr = zh_all[tr], zt_all[tr]
            iso = fit_iso(zh_tr, zt_tr)
            assert_monotone(iso, f"{pool}/{f}")
            isos[(pool, f)] = iso

    def crossfit(variant: str, pool: str, t: float | None):
        scores, cal_parts = {}, []
        for f in CV_FOLDS:
            va = masks_va[f]
            zc = apply_variant(variant, isos[(pool, f)], zh_all[va], t)
            scores[f] = rmsle_z(zt_all[va], zc)
            cal_parts.append(zc)
        cal = np.concatenate(cal_parts)
        return scores, float(np.mean(list(scores.values()))), cal

    results: dict = {}
    cal_arrays: dict[tuple, np.ndarray] = {}

    results["raw"] = {"scores": {k: round(v, 5) for k, v in raw_scores.items()},
                      "mean": round(raw_mean, 5)}

    for pool in ("cv", "cvp"):
        sc, mn, cal = crossfit("iso", pool, None)
        results.setdefault("iso", {})[pool] = {
            "scores": {k: round(v, 5) for k, v in sc.items()}, "mean": round(mn, 5)}
        cal_arrays[("iso", pool, None)] = cal

    grid_store: dict[tuple, list] = {}
    for variant in ("zero", "iso_zero", "zero_iso"):
        for pool in ("cv", "cvp"):
            rows = []
            for t in T_GRID:
                sc, mn, cal = crossfit(variant, pool, float(t))
                rows.append({"t": round(float(t), 2),
                             "scores": {k: round(v, 5) for k, v in sc.items()},
                             "mean": round(mn, 5)})
                cal_arrays[(variant, pool, float(t))] = cal
            grid_store[(variant, pool)] = rows
            best_row = min(rows, key=lambda r: r["mean"])
            results.setdefault(variant, {})[pool] = {
                "grid": rows,
                "best_t": best_row["t"],
                "best_mean": best_row["mean"],
                "best_scores": best_row["scores"],
            }
            print(f"{variant:>8}/{pool}: best t={best_row['t']:.2f} "
                  f"mean={best_row['mean']:.5f}", flush=True)

    for pool in ("cv", "cvp"):
        print(f"iso/{pool}: mean={results['iso'][pool]['mean']:.5f}", flush=True)

    # ---------- stage 2: selection ----------
    cands = [("raw", None, None, raw_mean)]
    for vname, pools_d in results.items():
        if vname == "raw":
            continue
        for pool, d in pools_d.items():
            if vname == "iso":
                cands.append((vname, pool, None, d["mean"]))
            else:
                cands.append((vname, pool, d["best_t"], d["best_mean"]))
    cands.sort(key=lambda c: (round(c[3], 6), VARIANT_RANK[c[0]], 0 if c[1] == "cv" else 1))
    best_variant, best_pool, best_t, best_score = cands[0][0], cands[0][1], cands[0][2], cands[0][3]

    chosen_scores = results[best_variant][best_pool]["scores"] if best_t is None \
        else results[best_variant][best_pool]["best_scores"]
    chosen_f03 = chosen_scores["fold_03"]
    improvement = (REF_MEAN - best_score) / REF_MEAN
    adopt_mean = improvement >= ADOPT_MEAN_TOL
    adopt_f03 = chosen_f03 <= ADOPT_F03_MAX
    adopted = bool(adopt_mean and adopt_f03)
    print(f"\nSELECTION: {best_variant}/pool={best_pool}/t={best_t} "
          f"mean={best_score:.5f} fold_03={chosen_f03:.5f} adopted={adopted}", flush=True)

    # ---------- diagnostics: why calibration does/does not help ----------
    # (1) oracle per-fold zeroing threshold (upper bound of the zeroing headroom,
    #     NOT achievable honestly)
    oracle = {}
    for f in CV_FOLDS:
        va = masks_va[f]
        per_t = [(rmsle_z(zt_all[va], np.where(zh_all[va] < t, 0.0, zh_all[va])),
                  round(float(t), 2)) for t in T_GRID]
        s, t = min(per_t)
        oracle[f] = {"raw": round(raw_scores[f], 5), "best_oracle": round(s, 5),
                     "t_oracle": t, "gain": round(raw_scores[f] - s, 5)}

    # (2) isotonic in-sample vs cross-fit transfer gap (pooled over folds):
    #     fit on 3 folds, score on those SAME 3 folds vs honest held-out
    iso_insample_scores = []
    for f in CV_FOLDS:
        tr = ~masks_va[f]
        iso_tr = fit_iso(zh_all[tr], zt_all[tr])
        assert_monotone(iso_tr, f"diag/{f}")
        iso_insample_scores.append(rmsle_z(zt_all[tr], iso_tr.predict(zh_all[tr])))
    iso_transfer_gap = float(np.mean(iso_insample_scores)) - results["iso"]["cv"]["mean"]

    # (3) binned bias of raw preds + share near zero (deciles of zh, pooled OOF)
    qs = np.quantile(zh_all, np.linspace(0, 1, 11))
    bins = np.digitize(zh_all, qs[1:-1])
    bias_bins = []
    for b in range(10):
        m = bins == b
        bias_bins.append({
            "zh_lo": round(float(qs[b]), 3), "zh_hi": round(float(qs[b + 1]), 3),
            "n": int(m.sum()),
            "mean_zh": round(float(np.mean(zh_all[m])), 4),
            "mean_z_true": round(float(np.mean(zt_all[m])), 4),
            "bias": round(float(np.mean(zt_all[m] - zh_all[m])), 4),
            "target_zero_share": round(float(np.mean(zt_all[m] <= 1e-12)), 4),
        })

    M_diag = {
        "oracle_zeroing_per_fold": oracle,
        "oracle_note": "оракул видит сам валидируемый фолд — это ВЕРХНЯЯ граница "
                       "выгоды зануления, не достижимая честно",
        "isotonic_insample_minus_crossfit_mean": round(iso_transfer_gap, 5),
        "isotonic_insample_pooled_mean": round(float(np.mean(iso_insample_scores)), 5),
        "binned_bias_deciles": bias_bins,
    }

    # ---------- stage 3: final calibrator on FULL pool -> submission ----------
    if best_variant == "raw":
        iso_full = None
    else:
        if best_pool == "cvp":
            zh_full = np.concatenate([zh_all, zh_px])
            zt_full = np.concatenate([zt_all, zt_px])
        else:
            zh_full, zt_full = zh_all, zt_all
        iso_full = fit_iso(zh_full, zt_full)
        assert_monotone(iso_full, "final")

    zc_fe = apply_variant(best_variant, iso_full, zh_fe, best_t)
    y_fe = np.clip(np.expm1(zc_fe), 0, None).astype(np.float64)
    assert np.isfinite(y_fe).all() and (y_fe >= 0).all()

    sample = pl.read_csv("sample_submit.csv")
    sub = pl.DataFrame({"user_id": fend["user_id"], "predict": y_fe}).join(
        sample.select("user_id"), on="user_id", how="semi"
    )
    assert set(sub["user_id"]) == set(sample["user_id"]), "user_id set mismatch"
    sub = sub.join(sample.select("user_id").with_row_index("__ord"), on="user_id")
    sub = sub.sort("__ord").drop("__ord")
    assert sub.height == 250_000 and sub.width == 2
    SUB_PATH.parent.mkdir(exist_ok=True)
    sub.write_csv(SUB_PATH)

    chk = pl.read_csv(SUB_PATH)
    assert chk.height == 250_000 and chk.width == 2
    assert chk.columns == ["user_id", "predict"]
    assert chk["predict"].is_finite().all() and (chk["predict"] >= 0).all()
    assert (chk["user_id"] == sample["user_id"]).all(), "order mismatch"

    # in-sample reference: full-pool calibrator applied to refit model on fold_03
    zc_r3 = apply_variant(best_variant, iso_full, zh_r3, best_t)
    insample_raw = rmsle_z(zt_r3, zh_r3)
    insample_cal = rmsle_z(zt_r3, zc_r3)
    zero_share_fe_before = float(np.mean(zh_fe <= 1e-12))
    zero_share_fe_after = float(np.mean(y_fe <= 0))

    # ---------- figures ----------
    # fig 1: calibration curves (cross-fit pooled, chosen pipeline)
    cal_chosen = cal_arrays[(best_variant, best_pool, best_t)]
    n_bins = 40
    qs = np.quantile(zh_all, np.linspace(0, 1, n_bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    bins = np.digitize(zh_all, qs[1:-1])
    bx, by_raw, by_cal = [], [], []
    for b in range(n_bins):
        m = bins == b
        if m.sum() < 50:
            continue
        bx.append(float(np.mean(zh_all[m])))
        by_raw.append(float(np.mean(zt_all[m])))
        by_cal.append(float(np.mean(cal_chosen[m])))
    lo = min(min(bx), min(by_raw), min(by_cal))
    hi = max(max(bx), max(by_raw), max(by_cal))
    plt.figure(figsize=(7, 7))
    plt.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
    plt.plot(bx, by_raw, "o-", ms=4, color="darkorange", label="до калибровки (OOF)")
    plt.plot(bx, by_cal, "o-", ms=4, color="seagreen",
             label=f"после ({best_variant}" + (f", t*={best_t}" if best_t is not None else "") + ")")
    if iso_full is not None:
        xs = np.linspace(max(lo, iso_full.X_min_), min(hi, iso_full.X_max_), 400)
        plt.plot(xs, iso_full.predict(xs), "-", lw=1.2, color="steelblue", alpha=0.8,
                 label="изотоника m*(ẑ), полный пул")
    plt.xlabel("предикт модели ẑ = log1p(ŷ)")
    plt.ylabel("средний факт z = log1p(target) в бине")
    plt.title(f"b01: калибровочная кривая (cross-fit OOF, пул {best_pool})")
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "b01_calibration_curve.png", dpi=150)
    plt.close()

    # fig 2: RMSLE vs t*
    plt.figure(figsize=(8, 5))
    colors = {"zero": "tab:brown", "iso_zero": "seagreen", "zero_iso": "steelblue"}
    styles = {"cv": "-", "cvp": "--"}
    for variant in ("zero", "iso_zero", "zero_iso"):
        for pool in ("cv", "cvp"):
            rows = grid_store[(variant, pool)]
            ts = [r["t"] for r in rows]
            ms = [r["mean"] for r in rows]
            plt.plot(ts, ms, styles[pool], color=colors[variant], lw=1.4,
                     label=f"{variant} / {pool}")
    plt.axhline(results["iso"][best_pool]["mean"], color="gray", ls=":", lw=1,
                label=f"изотоника без зануления ({best_pool})")
    plt.axhline(raw_mean, color="black", ls=":", lw=1, label="без калибровки")
    if best_t is not None:
        plt.axvline(best_t, color="crimson", ls="--", lw=1, label=f"t*={best_t}")
    plt.xlabel("порог зануления t* (в z-пространстве)")
    plt.ylabel("средний LOFO RMSLE (cross-fit)")
    plt.title("b01: зависимость RMSLE от порога зануления")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "b01_t_grid.png", dpi=150)
    plt.close()

    # ---------- metrics ----------
    M["versions"] = {
        "polars": pl.__version__, "numpy": np.__version__,
        "sklearn": __import__("sklearn").__version__,
        "catboost": __import__("catboost").__version__,
    }
    M["config"] = {
        "features": "exp02 accepted: 56 base + conv/decomp/due/shares = 66",
        "catboost": "RMSE on z=log1p(target), lr=0.05, depth=8, l2_leaf_reg=3, seed=42, x1000",
        "t_grid": [round(float(x), 2) for x in T_GRID[[0, 1, 2, -2, -1]]],
        "t_grid_note": "{0, 0.05, ..., 2.0}, 41 points",
        "isotonic": "IsotonicRegression(increasing=True, y_min=0, out_of_bounds=clip)",
    }
    M["references"] = {"exp02_lofo": REF_LOFO, "exp02_lofo_mean": round(REF_MEAN, 5)}
    M["oof_reproduction"] = {
        "raw_scores": {k: round(v, 5) for k, v in raw_scores.items()},
        "raw_mean": round(raw_mean, 5),
        "abs_deviations": {k: round(v, 5) for k, v in repro_devs.items()},
        "within_tol_0.002": bool(repro_ok),
    }
    M["diagnostics"] = {
        "target_zero_share_by_fold": {k: round(v, 4) for k, v in zero_share_by_fold.items()},
        "ols_slope_z_true_over_z_pred_pooled": round(slope, 4),
        "pred_zero_share_foldend_before": round(zero_share_fe_before, 5),
        "pred_zero_share_submission_after": round(zero_share_fe_after, 5),
        "pseudo_raw_rmsle_m3": train_times.get("pseudo_raw_rmsle"),
    }
    M["diagnostics_headroom"] = M_diag
    M["crossfit_results"] = {
        "protocol": "calibrator fitted on pool minus evaluated fold (+pseudo for cvp); "
                    "never sees the scored fold",
        "iso": {p: {"scores": {k: round(v, 5) for k, v in d["scores"].items()},
                    "mean": round(d["mean"], 5)}
                for p, d in results["iso"].items()},
        "zero": {p: {"best_t": d["best_t"], "best_mean": d["best_mean"],
                     "best_scores": d["best_scores"]}
                 for p, d in results["zero"].items()},
        "iso_zero": {p: {"best_t": d["best_t"], "best_mean": d["best_mean"],
                         "best_scores": d["best_scores"]}
                     for p, d in results["iso_zero"].items()},
        "zero_iso": {p: {"best_t": d["best_t"], "best_mean": d["best_mean"],
                         "best_scores": d["best_scores"]}
                     for p, d in results["zero_iso"].items()},
    }
    M["grids"] = {f"{v}/{p}": grid_store[(v, p)] for v in ("zero", "iso_zero", "zero_iso")
                  for p in ("cv", "cvp")}
    M["selection"] = {
        "chosen": best_variant, "pool": best_pool, "t_star": best_t,
        "mean_crossfit_rmsle": round(best_score, 5),
        "scores_by_fold": chosen_scores,
        "top3_candidates": [
            {"variant": c[0], "pool": c[1], "t": c[2], "mean": round(c[3], 5)}
            for c in cands[:3]
        ],
        "tie_break": "min mean, ties -> simpler variant, pool cv",
    }
    M["adoption"] = {
        "threshold_mean_improvement_pct": 0.3,
        "improvement_vs_ref_mean_pct": round(improvement * 100, 4),
        "mean_criterion_met": bool(adopt_mean),
        "fold03_value": chosen_f03,
        "fold03_max_allowed": round(ADOPT_F03_MAX, 5),
        "fold03_criterion_met": bool(adopt_f03),
        "verdict": "adopted" if adopted else "rejected",
    }
    M["final_pipeline"] = {
        "steps": ["CatBoost 66ф LOFO OOF / рефит на 4 фолдах"]
                 + (["изотоника m*(ẑ), фит на полном пуле (" + best_pool + ")"]
                    if "iso" in best_variant else [])
                 + ([f"зануление z<t*={best_t}"]
                    if best_variant in ("zero", "iso_zero") and (best_t or 0) > 0 else [])
                 + ["expm1, clip>=0"],
        "order": best_variant,
        "note": ("t*=0: зануление вырождается в клип отрицательных z (доля "
                 f"{zero_share_fe_before:.4%} точек) — фактически тождественно "
                 "прогнозу без калибровки") if (best_variant == "zero" and best_t == 0.0) else None,
    }
    M["submission"] = {
        "path": str(SUB_PATH), "rows": chk.height, "columns": chk.columns,
        "pred_mean": round(float(chk["predict"].mean()), 4),
        "pred_median": round(float(chk["predict"].median()), 4),
        "pred_min": round(float(chk["predict"].min()), 6),
        "pred_max": round(float(chk["predict"].max()), 2),
        "asserts_passed": True,
    }
    M["insample_reference"] = {
        "note": "рефит-модель видела fold_03 в обучении — цифра in-sample, для сравнения",
        "raw": round(insample_raw, 5),
        "calibrated": round(insample_cal, 5),
    }
    M["timings_sec"] = {
        "train_stage": train_times.get("total_sec"),
        "fits": {k: v for k, v in train_times.items() if k.startswith("fit")},
        "calibrate_total": round(time.time() - t_start, 1),
        "features_pseudo_total_min": "<1 (см. reports/b01_features.log)",
    }
    METRICS_PATH.write_text(json.dumps(M, indent=2, ensure_ascii=False))
    print(f"CALIBRATION DONE in {(time.time() - t_start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
