"""b03 step 3: ablations (i)-(iv) on fold_03 + proxy-fold comparison + deciles.

Protocol of exp01/exp02: train fold_00+01+02, validate fold_03, CatBoost RMSE
on log1p target (variant iii: log1p(target/M_anchor)), lr=0.05 depth=8 l2=3,
seed=42, 1000 iters, expm1 once, clip >= 0.

Variants:
  i   reference 66f on OFFICIAL caches (reproduction gate vs 1.69277);
  ii  history normalised (b03 features), raw targets;
  iii history + window-normalised targets (strict causal M), denormalised
      prediction *M(anchor) - the V1' method;
  iv  reference + scalar s^beta, beta=1 (exp02 YoY calibration transplanted:
      fold_03 uses its closest causal analog s_hat_f03; proxy uses 1.16278).
Proxy diagnostics additionally include iii-assist (oracle-calendar at proxy
time, symmetric information to exp02's realised-s_hat protocol) and ii+s.

Writes reports/b03_metrics.json and figures. The "decision" block gates
src/b03_submit.py.
"""

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import platform  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from catboost import CatBoostRegressor, Pool  # noqa: E402

from b03_common import (  # noqa: E402
    CV_FOLDS, FIG_DIR, METRICS_PATH, OUT_DIR, SEED, TRAIN_FOLDS, rmsle,
)
from b03_features import FEATURES_66_ORDER  # noqa: E402

REF_FOLD03_RMSLE = 1.69277
ADOPT_THRESHOLD_F03 = 1.68938          # -0.2% vs reference
PROXY_REF_BETA0, PROXY_REF_BETA1 = 1.853, 1.835
ADOPT_THRESHOLD_PROXY = 1.825          # >=0.5% win vs beta=1

EXTRA_COLS = ["conv_s2o", "conv_c2o", "conv_o2c", "aov_30", "ord_days_30",
              "due_ratio", "share_gmv_search_90", "share_gmv_cat_90",
              "share_gmv_search_trend", "share_gmv_cat_trend"]
FEAT_NAMES = [*FEATURES_66_ORDER, "pct_rank_gmv30"]

FIG_DIR.mkdir(parents=True, exist_ok=True)


def fit_catboost(X_tr, y_tr, feat_names, n_estimators=1000,
                 X_val=None, y_val=None):
    model = CatBoostRegressor(
        loss_function="RMSE",
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3,
        n_estimators=n_estimators,
        thread_count=-1,
        random_seed=SEED,
        verbose=0,
        allow_writing_files=False,
    )
    train_pool = Pool(X_tr, label=y_tr, feature_names=feat_names)
    eval_pool = Pool(X_val, label=y_val, feature_names=feat_names) if X_val is not None else None
    t0 = time.time()
    model.fit(train_pool, eval_set=eval_pool)
    return model, time.time() - t0


def predict_raw(model, X) -> np.ndarray:
    return np.clip(np.expm1(model.predict(X)), 0, None)


def add_pct_rank(df: pl.DataFrame, col: str = "gmv_sum_30d") -> pl.DataFrame:
    n = df.height
    return df.with_columns(
        ((pl.col(col).rank(method="average") - 1) / (n - 1)).cast(pl.Float64)
        .alias("pct_rank_gmv30")
    )


def join_ref(name: str) -> pl.DataFrame:
    base_dir = Path(f"data/v2/features/{name}")
    if not base_dir.exists():
        base_dir = Path(f"data/v2/features_exp02/{name}_base")
    assert base_dir.exists(), f"no base cache for {name}"
    base = pl.read_parquet(str(base_dir / "batch_*.parquet"))
    extra = pl.read_parquet(f"data/v2/features_exp02/{name}/batch_*.parquet")
    j = base.join(
        extra.select(["user_id", "anchor_date", *EXTRA_COLS]),
        on="user_id", how="inner", suffix="_x2",
    )
    assert j.height == 250_000 and (j["anchor_date"] == j["anchor_date_x2"]).all()
    j = j.drop("anchor_date_x2")
    got = [c for c in j.columns if c not in ("anchor_date", "user_id", "target")]
    assert set(got) == set(FEATURES_66_ORDER), f"{name}: col mismatch {set(got)}"
    return add_pct_rank(j)


def join_b03(name: str) -> pl.DataFrame:
    j = pl.read_parquet(str(OUT_DIR / name / "batch_*.parquet")).select(
        ["anchor_date", "user_id", *FEATURES_66_ORDER, "target"])
    assert j.height == 250_000
    return add_pct_rank(j)


def canonical(df: pl.DataFrame) -> pl.DataFrame:
    """Sort by user_id.

    Row orders differ between the official caches and the b03 cache
    (parallel group_by order), so ANY positional mixing of frames would pair
    predictions of one user with labels of another. Sorting both cache views
    into the same canonical order removes the hazard; the cross-cache
    identity assert below proves alignment.
    """
    return df.sort("user_id")


class DS:
    def __init__(self, name, df):
        self.name = name
        self.df = df
        self.y_raw = np.clip(df["target"].to_numpy(), 0, None)

    def X(self):
        return self.df.select(FEAT_NAMES).to_numpy()


def Xtr_df(df):
    return df.select(FEAT_NAMES).to_numpy()


def Ytr_log1p(df) -> np.ndarray:
    return np.log1p(np.clip(df["target"].to_numpy(), 0, None))


def Ytr_norm(df, lm_by_fold: dict[str, float]) -> np.ndarray:
    """z = log1p(target / M(anchor)); anchor ISO-date -> log_M map."""
    y = np.clip(df["target"].to_numpy(), 0, None)
    lm = (
        df.select(
            pl.col("anchor_date").cast(pl.String)
            .replace_strict(lm_by_fold, return_dtype=pl.Float64).alias("lm")
        )["lm"].to_numpy()
    )
    assert not np.isnan(lm).any(), "unmapped anchor in lm_by_fold"
    return np.log1p(y / np.exp(lm))


def decile_table(y_true, pred_ref, pred_best, activity) -> list[dict]:
    # ordinal rank breaks the large zero-tie blocks arbitrarily but guarantees
    # exactly 10 non-empty buckets; fine for a reporting table
    rk = pl.Series(activity).rank(method="ordinal").to_numpy()
    q = np.ceil(rk / len(activity) * 10).astype(int)
    rows = []
    for d in range(1, 11):
        m = q == d
        r_ref = rmsle(y_true[m], pred_ref[m])
        r_best = rmsle(y_true[m], pred_best[m])
        rows.append({
            "decile": d, "n": int(m.sum()),
            "activity_mean_gmv90": round(float(activity[m].mean()), 2),
            "rmsle_ref": round(r_ref, 5),
            "rmsle_best": round(r_best, 5),
            "delta_best_minus_ref": round(r_best - r_ref, 5),
        })
    return rows


def main() -> None:
    global ANCHOR_KEYS
    t_start = time.time()
    summary_idx = json.loads((OUT_DIR / "index_summary.json").read_text())
    anchor_keys = [*CV_FOLDS, "fold_proxy", "fold_end"]
    ANCHORS_ISO = {f: summary_idx["M"][f]["strict"]["anchor"] for f in anchor_keys}
    M_strict = {f: summary_idx["M"][f]["strict"]["log_M"] for f in anchor_keys}
    LM_strict = {summary_idx["M"][f]["strict"]["anchor"]: M_strict[f]
                 for f in anchor_keys}  # keyed by anchor ISO date
    M_assist = {f: summary_idx["M"][f]["assist"]["log_M"]
                for f in ("fold_00", "fold_01", "fold_02", "fold_proxy")}
    LM_assist = {summary_idx["M"][f]["assist"]["anchor"]: M_assist[f] for f in M_assist}
    shat_fest = summary_idx["yoy_facts"]["s_hat_exp02_festive"]
    shat_f03 = summary_idx["yoy_facts"]["fold03_analog"]["s_hat"]

    METRICS: dict = {
        "versions": {
            "python": platform.python_version(), "polars": pl.__version__,
            "catboost": __import__("catboost").__version__, "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "protocol": {
            "train": TRAIN_FOLDS, "val": "fold_03", "n_estimators": 1000, "seed": SEED,
            "n_features": len(FEAT_NAMES),
            "references": {"exp02_66f_fold03": REF_FOLD03_RMSLE,
                           "proxy_beta0": PROXY_REF_BETA0,
                           "proxy_beta1": PROXY_REF_BETA1},
            "adoption": {"threshold_fold03": ADOPT_THRESHOLD_F03,
                         "threshold_proxy_vs_beta1": ADOPT_THRESHOLD_PROXY},
        },
        "M_used_per_fold": {
            f: {"log_M_strict": M_strict[f],
                "prior_year_cov": summary_idx["M"][f]["strict"]["prior_year_coverage_frac"],
                **({"log_M_assist": M_assist[f]} if f in M_assist else {})}
            for f in anchor_keys},
    }

    def dump() -> None:
        METRICS_PATH.write_text(json.dumps(METRICS, indent=2, ensure_ascii=False))

    print("loading folds...", flush=True)
    refs, b03s = {}, {}
    for f in [*anchor_keys]:
        refs[f] = DS(f, canonical(join_ref(f)))
        b03s[f] = DS(f, canonical(join_b03(f)))
        # alignment proof: same users, same order, identical targets
        assert refs[f].df["user_id"].to_numpy().tolist() == \
            b03s[f].df["user_id"].to_numpy().tolist(), f"{f}: user order mismatch"
        tr_ref = refs[f].df["target"].to_numpy()
        tr_b03 = b03s[f].df["target"].to_numpy()
        both_nan = np.isnan(tr_ref) & np.isnan(tr_b03)
        assert np.allclose(tr_ref, tr_b03, atol=1e-9, equal_nan=True), \
            f"{f}: target mismatch between caches"
        print(f"  {f}: ref{refs[f].df.shape} b03{b03s[f].df.shape} "
              f"target_nulls={int(np.isnan(tr_ref).sum())} "
              f"alignment=OK(user-sorted, targets identical)", flush=True)

    tr_r = pl.concat([refs[f].df for f in TRAIN_FOLDS], how="vertical")
    tr_n = pl.concat([b03s[f].df for f in TRAIN_FOLDS], how="vertical")

    results = {}

    # ---------- (i) reference reproduction ----------
    print("\n=== (i) reference 66f reproduction ===", flush=True)
    m_i, ft = fit_catboost(
        Xtr_df(tr_r), Ytr_log1p(tr_r), FEAT_NAMES,
        X_val=refs["fold_03"].X(), y_val=np.log1p(refs["fold_03"].y_raw))
    p_i = predict_raw(m_i, refs["fold_03"].X())
    s_i = rmsle(refs["fold_03"].y_raw, p_i)
    dev = abs(s_i - REF_FOLD03_RMSLE)
    print(f"(i) ref fold_03 RMSLE={s_i:.5f} ({ft:.0f}s), dev vs exp02 {dev:.5f}", flush=True)
    assert dev <= 0.002, f"reference reproduction failed: {s_i}"
    results["i_ref"] = {"rmsle_fold03": round(s_i, 5), "fit_sec": round(ft, 1),
                        "dev_vs_exp02": round(dev, 5)}
    METRICS["ablation_fold03"] = results
    dump()

    # ---------- (ii) history normalised ----------
    print("\n=== (ii) history-normalised features, raw target ===", flush=True)
    m_ii, ft = fit_catboost(
        Xtr_df(tr_n), Ytr_log1p(tr_n), FEAT_NAMES,
        X_val=b03s["fold_03"].X(), y_val=np.log1p(b03s["fold_03"].y_raw))
    p_ii = predict_raw(m_ii, b03s["fold_03"].X())
    s_ii = rmsle(b03s["fold_03"].y_raw, p_ii)
    print(f"(ii) fold_03 RMSLE={s_ii:.5f} ({ft:.0f}s)", flush=True)
    results["ii_histnorm"] = {"rmsle_fold03": round(s_ii, 5), "fit_sec": round(ft, 1)}
    dump()

    # ---------- (iii) + window-normalised target (strict causal V1') ----------
    print("\n=== (iii) history+target normalisation (V1' strict) ===", flush=True)
    lm_tr_strict = {ANCHORS_ISO[f]: M_strict[f] for f in TRAIN_FOLDS}
    m_iii, ft = fit_catboost(
        Xtr_df(tr_n), Ytr_norm(tr_n, lm_tr_strict), FEAT_NAMES,
        X_val=b03s["fold_03"].X(),
        y_val=Ytr_norm(b03s["fold_03"].df, {ANCHORS_ISO["fold_03"]: M_strict["fold_03"]}))
    zn_iii = m_iii.predict(b03s["fold_03"].X())
    p_iii = np.clip(np.expm1(zn_iii) * float(np.exp(M_strict["fold_03"])), 0, None)
    s_iii = rmsle(refs["fold_03"].y_raw, p_iii)
    print(f"(iii) fold_03 RMSLE={s_iii:.5f} ({ft:.0f}s)", flush=True)
    results["iii_full_strict"] = {
        "rmsle_fold03": round(s_iii, 5), "fit_sec": round(ft, 1),
        "log_M_fold03": M_strict["fold_03"],
        "bias_z_mean_pred_minus_actualnorm": round(
            float(np.mean(zn_iii - np.log1p(refs["fold_03"].y_raw
                                            / float(np.exp(M_strict["fold_03"]))))), 4)}

    # ---------- (iv) reference + s^beta, beta=1 ----------
    s_iv = float(np.sqrt(np.mean((
        np.log1p(refs["fold_03"].y_raw)
        - (np.log1p(p_i) + np.log(shat_f03))) ** 2)))
    print(f"(iv) ref + s^1 (analog s={shat_f03}): fold_03 RMSLE={s_iv:.5f}", flush=True)
    results["iv_ref_beta1"] = {
        "rmsle_fold03": round(s_iv, 5), "s_hat_used": shat_f03,
        "note": "closest causal analog of exp02 s_hat for the Jan15-Feb13 twin "
                "window; exp02 never calibrated fold_03"}
    METRICS["ablation_fold03"] = results
    dump()

    # ---------- proxy fold ----------
    print("\n=== proxy fold (anchor 2025-02-14, festive-window simulation) ===",
          flush=True)
    y_px = refs["fold_proxy"].y_raw
    zpx = np.log1p(predict_raw(m_i, refs["fold_proxy"].X()))
    beta_rows = []
    for b in (0.0, 0.25, 0.5, 0.75, 1.0):
        sc = float(np.sqrt(np.mean(
            (np.log1p(y_px) - (zpx + b * np.log(shat_fest))) ** 2)))
        beta_rows.append({"beta": b, "rmsle_proxy": round(sc, 5)})
        print(f"  ref + beta={b:g}: {sc:.5f}", flush=True)
    proxy = {"ref_beta_grid": beta_rows}
    proxy["replication_check"] = {
        "beta0_expected": PROXY_REF_BETA0, "beta1_expected": PROXY_REF_BETA1,
        "beta0_got": beta_rows[0]["rmsle_proxy"], "beta1_got": beta_rows[-1]["rmsle_proxy"],
        "tol": 0.02,
        "ok_within_tol": bool(abs(beta_rows[0]["rmsle_proxy"] - PROXY_REF_BETA0) < 0.02
                              and abs(beta_rows[-1]["rmsle_proxy"] - PROXY_REF_BETA1) < 0.02),
        "note": "exp02 published grid: 1.85297 -> 1.83478; small systematic shift "
                "expected from model-transfer amplification on the off-distribution "
                "proxy population (45d histories), fold_03 dev is 20x smaller"}

    p_px_ii = predict_raw(m_ii, b03s["fold_proxy"].X())
    s_px_ii = rmsle(y_px, p_px_ii)
    s_px_ii_s = float(np.sqrt(np.mean((
        np.log1p(y_px) - (np.log1p(p_px_ii) + np.log(shat_fest))) ** 2)))
    proxy["ii_histnorm"] = {"rmsle_proxy": round(s_px_ii, 5)}
    proxy["ii_plus_beta1"] = {"rmsle_proxy": round(s_px_ii_s, 5)}
    print(f"  (ii): {s_px_ii:.5f}; (ii)+s^1: {s_px_ii_s:.5f}", flush=True)

    p_px_iii = np.clip(np.expm1(m_iii.predict(b03s["fold_proxy"].X()))
                       * float(np.exp(M_strict["fold_proxy"])), 0, None)
    s_px_iii = rmsle(y_px, p_px_iii)
    proxy["iii_strict_causal"] = {
        "rmsle_proxy": round(s_px_iii, 5), "log_M_proxy_strict": M_strict["fold_proxy"],
        "note": "fully causal at proxy time (no 2024 data -> no annual term)"}
    print(f"  (iii strict): {s_px_iii:.5f}", flush=True)

    lm_tr_assist = {ANCHORS_ISO[f]: M_assist[f] for f in TRAIN_FOLDS}
    m_as, ft_as = fit_catboost(Xtr_df(tr_n), Ytr_norm(tr_n, lm_tr_assist), FEAT_NAMES)
    p_px_as = np.clip(np.expm1(m_as.predict(b03s["fold_proxy"].X()))
                      * float(np.exp(M_assist["fold_proxy"])), 0, None)
    s_px_as = rmsle(y_px, p_px_as)
    proxy["iii_assist_oracle_calendar"] = {
        "rmsle_proxy": round(s_px_as, 5), "log_M_proxy_assist": M_assist["fold_proxy"],
        "fit_sec": round(ft_as, 1),
        "note": "oracle calendar knowledge at proxy time (realised m_hat of the "
                "window itself), symmetric to exp02's realised-s_hat protocol"}
    print(f"  (iii assist): {s_px_as:.5f}", flush=True)
    METRICS["proxy"] = proxy
    dump()

    # ---------- deciles ----------
    best_name, best_score, best_kind, best_pred = min(
        [("ii_histnorm", s_ii, "ii", p_ii), ("iii_full_strict", s_iii, "iii", p_iii)],
        key=lambda kv: kv[1])
    decile_block: dict = {}
    decile_ok = True
    if best_score < s_i:
        act = refs["fold_03"].df["gmv_sum_90d"].to_numpy()
        dtab = decile_table(refs["fold_03"].y_raw, p_i, best_pred, act)
        worst = max(r["delta_best_minus_ref"] for r in dtab)
        decile_ok = worst <= 0.01
        decile_block = {
            "by": "deciles of raw gmv_sum_90d (official cache), fold_03",
            "best_variant": best_name,
            "worst_decile_delta": round(worst, 5), "no_degradation_flag": bool(decile_ok),
            "table": dtab}
        print(f"\ndeciles [{best_name}]: worst decile delta={worst:.5f} "
              f"({'OK' if decile_ok else 'DEGRADED'})", flush=True)
    else:
        decile_block = {"skipped": "no variant better than reference on fold_03"}
    METRICS["deciles_fold03"] = decile_block
    dump()

    # ---------- decision ----------
    adopted = False
    reason = ""
    px_key = "iii_assist_oracle_calendar"
    if best_score >= s_i:
        reason = ("no improvement over reference on fold_03 "
                  f"(best {best_name}={best_score:.5f} vs ref {s_i:.5f})")
    elif not decile_ok:
        reason = "decile degradation detected"
    elif best_score <= ADOPT_THRESHOLD_F03:
        adopted, reason = True, (f"fold_03 {best_score:.5f} <= {ADOPT_THRESHOLD_F03} "
                                 "(direct -0.2% threshold)")
    elif best_kind == "iii" and proxy[px_key]["rmsle_proxy"] <= ADOPT_THRESHOLD_PROXY:
        adopted = True
        reason = (f"proxy-assist {proxy[px_key]['rmsle_proxy']} <= {ADOPT_THRESHOLD_PROXY} "
                  f"(>=0.5% vs beta=1) with fold_03 non-degradation "
                  f"({best_score:.5f} <= {s_i:.5f})")
    else:
        reason = (f"thresholds not met: fold_03 {best_score:.5f} > {ADOPT_THRESHOLD_F03}"
                  + ("" if best_kind != "iii" else
                     f", proxy {proxy[px_key]['rmsle_proxy']} > {ADOPT_THRESHOLD_PROXY}"))
    METRICS["decision"] = {
        "adopted": bool(adopted), "reason": reason,
        "variant": best_kind if adopted else None,
        "variant_name": best_name,
        "best_rmsle_fold03": round(best_score, 5),
        "ref_rmsle_fold03_reproduced": round(s_i, 5),
        "calibration_for_submit": ("per-window strict M(anchor), fully causal"
                                   if best_kind == "iii" and adopted else
                                   ("none (raw predictions)" if best_kind == "ii" else None))}
    dump()
    print(f"\nDECISION: adopted={adopted}\n  {reason}", flush=True)

    # ---------- figures ----------
    names = ["(i)\nref 66f", "(ii)\nhist-norm", "(iii)\nfull V1'",
             f"(iv)\nref+s (s={shat_f03:.3f})"]
    vals = [s_i, s_ii, s_iii, s_iv]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    bars = ax.bar(names, vals, color=["gray", "tab:blue", "tab:green", "tab:red"])
    ax.axhline(REF_FOLD03_RMSLE, ls="--", color="k", lw=1, label="exp02 ref 1.69277")
    ax.axhline(ADOPT_THRESHOLD_F03, ls=":", color="seagreen", lw=1.2,
               label=f"adoption {ADOPT_THRESHOLD_F03}")
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.5f}", (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("RMSLE fold_03")
    ax.legend(fontsize=8)
    ax.set_title("b03 ablation on fold_03 (train fold_00-02)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "b03_variants_fold03.png", dpi=150)
    plt.close(fig)

    if "table" in decile_block:
        tab = decile_block["table"]
        xs = np.arange(10)
        w = 0.38
        fig, ax = plt.subplots(figsize=(10, 4.6))
        ax.bar(xs - w / 2, [r["rmsle_ref"] for r in tab], w, label="ref (i)",
               color="gray")
        ax.bar(xs + w / 2, [r["rmsle_best"] for r in tab], w,
               label=f"best ({best_name})", color="tab:green")
        ax.set_xticks(xs)
        ax.set_xticklabels(str(i + 1) for i in range(10))
        ax.set_xlabel("decile of gmv_sum_90d (1 = least active)")
        ax.set_ylabel("RMSLE fold_03")
        ax.legend()
        ax.set_title("b03: RMSLE by user-activity decile, ref vs best")
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "b03_deciles.png", dpi=150)
        plt.close(fig)

    px_names = ["ref\nbeta=0", "ref\nbeta=1 (exp02)", "(ii)\nhist-norm",
                "(ii)+s", "(iii)\nstrict", "(iii)\nassist"]
    px_vals = [beta_rows[0]["rmsle_proxy"], beta_rows[-1]["rmsle_proxy"],
               s_px_ii, s_px_ii_s, s_px_iii, s_px_as]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    bars = ax.bar(px_names, px_vals,
                  color=["gray", "darkred", "tab:blue", "tab:cyan", "tab:green", "olive"])
    ax.axhline(PROXY_REF_BETA1, ls="--", color="k", lw=1, label="exp02 beta=1: 1.835")
    ax.axhline(ADOPT_THRESHOLD_PROXY, ls=":", color="seagreen", lw=1.2,
               label=f"adoption {ADOPT_THRESHOLD_PROXY}")
    for b, v in zip(bars, px_vals):
        ax.annotate(f"{v:.4f}", (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("RMSLE proxy fold")
    ax.legend(fontsize=8)
    ax.set_title("b03 proxy fold (anchor 2025-02-14, festive-window simulation)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "b03_proxy.png", dpi=150)
    plt.close(fig)

    METRICS["runtime_train_sec"] = round(time.time() - t_start, 1)
    dump()
    print(f"\nTRAIN DONE in {(time.time() - t_start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
