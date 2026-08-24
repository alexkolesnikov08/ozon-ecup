"""b03 step 3b: diagnostics after the ablation decision (negative result).

Adds to reports/b03_metrics.json:
  - calibration_shift: RMSLE of reference fold_03 / proxy predictions as a
    function of an additive z-shift ln(s); reports s* (argmin) - the optimal
    SCALAR calibration. Explains the mechanism: proxy window genuinely runs
    ~+15% above the model's implicit expectation (festive ramp), while
    fold_03 needs no shift - hence exp02's beta helps there and nothing
    similar helps fold_03;
  - decile table for the best non-reference variant regardless of adoption.
"""

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import time  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from b03_common import FIG_DIR, METRICS_PATH, OUT_DIR, TRAIN_FOLDS  # noqa: E402
from b03_train import (  # noqa: E402
    FEAT_NAMES, Xtr_df, Ytr_log1p, canonical, decile_table, fit_catboost,
    join_b03, join_ref, predict_raw,
)


def main() -> None:
    t0 = time.time()
    METRICS = json.loads(METRICS_PATH.read_text())
    print("refitting reference model for diagnostics...", flush=True)
    tr_r = pl.concat([join_ref(f) for f in TRAIN_FOLDS], how="vertical")
    f3 = join_ref("fold_03")
    px = join_ref("fold_proxy")
    tr_r, f3, px = (canonical(d) for d in (tr_r, f3, px))

    m, _ = fit_catboost(Xtr_df(tr_r), Ytr_log1p(tr_r), FEAT_NAMES,
                        X_val=f3.select(FEAT_NAMES).to_numpy(),
                        y_val=np.log1p(np.clip(f3["target"].to_numpy(), 0, None)))
    z3 = m.predict(f3.select(FEAT_NAMES).to_numpy())
    zpx = m.predict(px.select(FEAT_NAMES).to_numpy())

    def shift_curve(z_pred, y_true):
        lt = np.log1p(np.clip(y_true, 0, None))
        grid = np.arange(-0.05, 0.301, 0.005)
        scores = [float(np.sqrt(np.mean((lt - (z_pred + sh)) ** 2))) for sh in grid]
        i = int(np.argmin(scores))
        return {"grid_s": [round(float(np.exp(g)), 4) for g in grid[::10]],
                "grid_rmsle": [round(s, 5) for s in scores[::10]],
                "s_star": round(float(np.exp(grid[i])), 4),
                "ln_shift_star": round(float(grid[i]), 4),
                "rmsle_at_star": round(scores[i], 5)}

    d3 = shift_curve(z3, np.clip(f3["target"].to_numpy(), 0, None))
    dpx = shift_curve(zpx, np.clip(px["target"].to_numpy(), 0, None))
    print(f"fold_03: s*={d3['s_star']} rmsle*={d3['rmsle_at_star']}", flush=True)
    print(f"proxy:   s*={dpx['s_star']} rmsle*={dpx['rmsle_at_star']}", flush=True)

    idx_sum = json.loads((OUT_DIR / "index_summary.json").read_text())
    shat_fest = idx_sum["yoy_facts"]["s_hat_exp02_festive"]
    METRICS["calibration_shift"] = {
        "method": ("reference model refit; RMSLE(z + ln s) over s-grid; "
                   "s* = optimal scalar calibration"),
        "fold_03": d3,
        "proxy": dpx,
        "proxy_exp02_beta1_equivalent": round(float(shat_fest), 4),
        "interpretation": ("proxy window runs ~+15-16% above the model's implicit "
                           "level (festive ramp) -> beta=1 helps there; fold_03 "
                           "window needs no shift -> neither beta nor M helps"),
    }

    # decile table for the best non-reference variant (forced, for reporting)
    ablation = METRICS.get("ablation_fold03", {})
    best_name = min(("ii_histnorm", "iii_full_strict"),
                    key=lambda k: ablation.get(k, {}).get("rmsle_fold03", 9e9))
    lm_tr_strict = {idx_sum["M"][f]["strict"]["anchor"]: idx_sum["M"][f]["strict"]["log_M"]
                    for f in TRAIN_FOLDS}
    tr_n = pl.concat([canonical(join_b03(f)) for f in TRAIN_FOLDS], how="vertical")
    f3n = canonical(join_b03("fold_03"))
    if best_name == "ii_histnorm":
        m_b, _ = fit_catboost(Xtr_df(tr_n), Ytr_log1p(tr_n), FEAT_NAMES)
        p_best = predict_raw(m_b, f3n.select(FEAT_NAMES).to_numpy())
    else:
        from b03_train import Ytr_norm
        m_b, _ = fit_catboost(Xtr_df(tr_n), Ytr_norm(tr_n, lm_tr_strict), FEAT_NAMES)
        import json as _json
        lm3 = _json.loads((OUT_DIR / "index_summary.json").read_text())[
            "M"]["fold_03"]["strict"]["log_M"]
        p_best = np.clip(np.expm1(m_b.predict(f3n.select(FEAT_NAMES).to_numpy()))
                         * float(np.exp(lm3)), 0, None)
    y_true = np.clip(f3["target"].to_numpy(), 0, None)  # identical to f3n target (sorted)
    act = f3["gmv_sum_90d"].to_numpy()
    tab = decile_table(y_true, predict_raw(m, f3.select(FEAT_NAMES).to_numpy()),
                       p_best, act)
    worst = max(r["delta_best_minus_ref"] for r in tab)
    prev = METRICS.get("deciles_fold03", {})
    METRICS["deciles_fold03"] = {
        "by": "deciles of raw gmv_sum_90d (official cache), fold_03",
        "best_variant": best_name,
        "worst_decile_delta": round(worst, 5),
        "no_degradation_flag": bool(worst <= 0.01),
        "note": "produced for reporting despite negative adoption decision",
        "table": tab}
    print(f"deciles[{best_name}]: worst delta={worst:.5f}", flush=True)

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.plot(d3["grid_s"], d3["grid_rmsle"], lw=2, label="fold_03", color="tab:blue")
    ax.plot(dpx["grid_s"], dpx["grid_rmsle"], lw=2, label="proxy", color="tab:red")
    ax.axvline(d3["s_star"], ls=":", color="tab:blue", lw=1)
    ax.axvline(dpx["s_star"], ls=":", color="tab:red", lw=1)
    ax.axvline(shat_fest, ls="--", color="k", lw=1, label=f"exp02 s_hat={shat_fest:.3f}")
    ax.set_xscale("log")
    ax.set_xlabel("scalar calibration s (prediction x s)")
    ax.set_ylabel("RMSLE")
    ax.set_title("b03: optimal scalar shift - proxy wants ~1.16, fold_03 wants ~1.0")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "b03_calibration_shift.png", dpi=150)
    plt.close(fig)

    METRICS["runtime_diag_sec"] = round(time.time() - t0, 1)
    METRICS_PATH.write_text(json.dumps(METRICS, indent=2, ensure_ascii=False))
    print("metrics updated", flush=True)


if __name__ == "__main__":
    main()
