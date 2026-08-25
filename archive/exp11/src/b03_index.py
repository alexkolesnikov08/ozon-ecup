"""b03 step 1: causal platform STL index per anchor + M_window table.

Outputs:
  data/v2/b03_plsi/index/<fold>.parquet      - index fitted on data <= anchor
  data/v2/b03_plsi/index/full_oracle.parquet - fit on ALL days (assist diag only)
  data/v2/b03_plsi/index_summary.json        - M_window(anchor), scheme facts,
                                               s_hat references, timings
  reports/b03_figures/b03_index_overview.png

Causality is demonstrated functionally: for sample folds the slice-index is
recomputed after DOUBLING all post-anchor gmv rows; exact equality asserted.
"""

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import time  # noqa: E402
from datetime import date, timedelta  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from b03_common import (  # noqa: E402
    ANCHORS, CV_FOLDS, END_ANCHOR, FIG_DIR, INDEX_DIR, METRICS_PATH,
    OUT_DIR, PROXY_ANCHOR, index_from_frame, m_window, platform_daily,
)

SAMPLE_CAUSALITY_FOLDS = ["fold_02", "fold_03", "fold_end"]


def s_hat_festive(daily: pl.DataFrame) -> float:
    num = daily.filter(
        pl.col("event_date").is_between(date(2025, 2, 14), date(2025, 3, 15)))
    den = daily.filter(
        pl.col("event_date").is_between(date(2025, 1, 15), date(2025, 2, 13)))
    return float(num["y"].sum() / den["y"].sum())


def s_hat_fold03_analog(daily: pl.DataFrame) -> dict:
    """Closest causal analog of exp02's s_hat for the fold_03 window.

    exp02: festive window / preceding 30d. For fold_03 the twin window
    (2025-01-15..02-13) has its preceding 30d partly outside the data start,
    so the baseline level is flat-rate extended from 2025-01-01..01-14.
    """
    twin = daily.filter(
        pl.col("event_date").is_between(date(2025, 1, 15), date(2025, 2, 13)))
    partial = daily.filter(
        pl.col("event_date").is_between(date(2025, 1, 1), date(2025, 1, 14)))
    base30 = float(partial["y"].sum()) / 14 * 30
    return {
        "definition": ("GMV(2025-01-15..02-13) / [GMV(2025-01-01..01-14)/14*30]; "
                       "preceding-30d baseline flat-rate extended (no Dec-2024 data)"),
        "s_hat": round(float(twin["y"].sum() / base30), 6),
    }


def s_hat_decomposition(idx_full: pl.DataFrame) -> dict:
    """Decompose exp02's s_hat into transferable vs non-transferable parts:
    s_hat = trend_ratio * mhat_ratio; the STL trend carries the annual shape
    (New-Year dip + recovery), m_hat the calendar deviations around it."""
    fest = idx_full.filter(
        pl.col("event_date").is_between(date(2025, 2, 14), date(2025, 3, 15)))
    prev = idx_full.filter(
        pl.col("event_date").is_between(date(2025, 1, 15), date(2025, 2, 13)))
    s_raw = float(fest["y"].sum() / prev["y"].sum())
    t_ratio = float(fest["trend"].mean() / prev["trend"].mean())
    m_ratio = float(fest["m_hat"].mean() / prev["m_hat"].mean())
    return {
        "s_hat": round(s_raw, 6),
        "trend_ratio_component": round(t_ratio, 4),
        "mhat_ratio_component": round(m_ratio, 4),
        "product_check": round(t_ratio * m_ratio, 6),
        "note": ("~all of the +16.3% 'festive uplift' is the smooth STL trend "
                 "slope between mid-Jan and mid-Feb 2025 (New-Year V-recovery), "
                 "not a calendar deviation of the festive days themselves"),
    }


def main() -> None:
    t0 = time.time()
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log = []

    def p(msg: str) -> None:
        line = f"[{time.time() - t0:7.1f}s] {msg}"
        print(line, flush=True)
        log.append(line)

    daily = platform_daily()
    start, end = daily["event_date"].min(), daily["event_date"].max()
    p(f"platform series: {daily.height} days, {start}..{end}, "
      f"total gmv={daily['y'].sum():.3e}")

    # ---- YoY facts ----
    shat_fest = s_hat_festive(daily)
    yoy = (
        daily.filter(pl.col("event_date").is_between(date(2026, 1, 15), date(2026, 2, 13)))["y"].sum()
        / daily.filter(pl.col("event_date").is_between(date(2025, 1, 15), date(2025, 2, 13)))["y"].sum()
    )
    shat_f03 = s_hat_fold03_analog(daily)
    p(f"s_hat exp02 festive = {shat_fest:.6f}; YoY jan15-feb13 = {yoy:.4f}; "
      f"fold_03 analog = {shat_f03['s_hat']}")

    # ---- full-sample oracle index (diagnostics/assist only) ----
    idx_full = index_from_frame(daily)
    idx_full.write_parquet(INDEX_DIR / "full_oracle.parquet")
    p(f"full oracle index built ({idx_full.height} days)")

    # ---- causal per-anchor indices ----
    summary: dict = {
        "scheme": {
            "series": "daily platform sum of gmv over all users",
            "stl": f"statsmodels STL period={7} robust=True on raw daily sums",
            "index": "m_hat_t = y_t / trend_t (ratio-to-smooth-trend; captures "
                     "DoW + holiday dips/spikes + short noise around smooth trend)",
            "causality": "per-anchor refit on the slice event_date <= anchor only",
            "M_definition": "M(a)=mean_{k=1..30} G(a,k)*base(a+k); G=clip(T_a/T_{a-W},0.85,1.5)**(k/W), "
                            "W=91d; base=prior-year same-date m_hat with dow correction "
                            "wp(dow d)/wp(dow d-365), fallback trailing-70d weekly profile",
            "modes": {"strict": "fully causal (used for train/val/submit)",
                      "assist": "base(d):=realised m_hat_d, oracle diagnostic on proxy "
                                "(symmetric to exp02's realised s_hat protocol)"},
        },
        "yoy_facts": {
            "s_hat_exp02_festive": round(shat_fest, 6),
            "yoy_jan15_feb13": round(float(yoy), 4),
            "fold03_analog": shat_f03,
        },
        "folds": {},
        "M": {},
        "timings_sec": {},
    }
    summary["yoy_facts"]["s_hat_decomposition"] = s_hat_decomposition(idx_full)
    p(f"s_hat decomposition: trend_ratio={summary['yoy_facts']['s_hat_decomposition']['trend_ratio_component']}, "
      f"mhat_ratio={summary['yoy_facts']['s_hat_decomposition']['mhat_ratio_component']}")

    for fold, anchor in ANCHORS.items():
        tf = time.time()
        sl = daily.filter(pl.col("event_date") <= anchor)
        idx = index_from_frame(sl)
        idx.write_parquet(INDEX_DIR / f"{fold}.parquet")
        mw_strict = m_window(anchor, idx, mode="strict")
        summary["folds"][fold] = {
            "anchor": anchor.isoformat(),
            "slice_days": idx.height,
            "slice_start": str(idx["event_date"].min()),
            "m_hat_mean": round(float(idx["m_hat"].mean()), 5),
            "m_hat_min": round(float(idx["m_hat"].min()), 5),
            "m_hat_max": round(float(idx["m_hat"].max()), 5),
        }
        summary["M"][fold] = {"strict": mw_strict}
        if fold == "fold_proxy":
            summary["M"][fold]["assist"] = m_window(
                anchor, idx, mode="assist", realized_idx=idx_full)
        summary["timings_sec"][f"index_{fold}"] = round(time.time() - tf, 3)
        p(f"{fold}: anchor {anchor}, slice {idx.height}d, M_strict={mw_strict['M']} "
          f"(growth={mw_strict['growth_annual']}, method={mw_strict['growth_method']}, "
          f"cov={mw_strict['prior_year_coverage_frac']})")

    # assist-consistent targets for training folds (oracle diagnostic only)
    for fold in ("fold_00", "fold_01", "fold_02"):
        a = ANCHORS[fold]
        summary["M"][fold]["assist"] = m_window(
            a, pl.read_parquet(INDEX_DIR / f"{fold}.parquet"),
            mode="assist", realized_idx=idx_full)
        p(f"{fold}: M_assist={summary['M'][fold]['assist']['M']} (oracle diagnostic)")

    # ---- functional causality check ----
    rng = np.random.default_rng(42)
    checks = {}
    for fold in SAMPLE_CAUSALITY_FOLDS:
        anchor = ANCHORS[fold]
        pert = daily.with_columns(
            pl.when(pl.col("event_date") > anchor)
            .then(pl.col("y") * 2.0)
            .otherwise(pl.col("y"))
            .alias("y")
        )
        sl = pert.filter(pl.col("event_date") <= anchor)
        idx_pert = index_from_frame(sl)
        idx_orig = pl.read_parquet(INDEX_DIR / f"{fold}.parquet")
        assert idx_pert.height == idx_orig.height
        assert idx_pert["event_date"].equals(idx_orig["event_date"])
        max_abs = float((idx_pert["m_hat"] - idx_orig["m_hat"]).abs().max())
        assert max_abs == 0.0, f"CAUSALITY VIOLATION in {fold}: max|dm|={max_abs}"
        checks[fold] = {"max_abs_diff_post_anchor_doubling": max_abs,
                        "n_slice_days": idx_orig.height}
        p(f"causality check {fold}: post-anchor x2 perturbation -> identical "
          f"slice index (max|diff|={max_abs})")
    summary["causality_check"] = {
        "method": "STL refit on slice<=anchor after doubling all post-anchor gmv; "
                  "exact equality of m_hat required",
        **checks,
    }

    # ---- figure ----
    dl = idx_full["event_date"].to_list()
    y_full = idx_full["y"].to_numpy()
    tr_full = idx_full["trend"].to_numpy()
    mh_full = idx_full["m_hat"].to_numpy()
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                             gridspec_kw={"height_ratios": [1.4, 1]})
    ax = axes[0]
    ax.plot(dl, y_full, lw=0.7, color="steelblue", label="daily platform gmv")
    ax.plot(dl, tr_full, lw=1.6, color="firebrick", label="STL trend")
    for hdate, lbl in [(date(2025, 2, 23), "23 Feb"), (date(2025, 3, 8), "8 Mar"),
                       (date(2025, 12, 31), "31 Dec"), (date(2026, 1, 1), "1 Jan"),
                       (date(2026, 1, 7), "7 Jan"), (date(2026, 2, 23), "23 Feb"),
                       (date(2026, 3, 8), "8 Mar")]:
        if start <= hdate <= end:
            ax.axvline(hdate, color="gray", ls=":", lw=0.8)
            ax.annotate(lbl, (hdate, ax.get_ylim()[1]), rotation=90,
                        va="top", ha="right", fontsize=7, color="dimgray")
    for fold in CV_FOLDS + ["fold_proxy", "fold_end"]:
        a = ANCHORS[fold]
        ax.axvline(a, color="seagreen", lw=0.8, alpha=0.6)
    ax.set_title("b03: platform daily gmv, STL trend, anchors (green) and holidays (dotted)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    ax = axes[1]
    ax.plot(dl, mh_full, lw=0.7, color="darkorange", label="m_hat (full fit, oracle view)")
    for fold, colr in zip(CV_FOLDS + ["fold_proxy"],
                          ["tab:blue", "tab:cyan", "tab:green", "tab:red", "tab:purple"]):
        sl = pl.read_parquet(INDEX_DIR / f"{fold}.parquet")
        ax.plot(sl["event_date"].to_list(), sl["m_hat"].to_numpy(),
                lw=0.8, color=colr, alpha=0.75,
                label=f"causal slice {fold} (fit<={ANCHORS[fold]})")
    ax.axhline(1.0, color="k", lw=0.6)
    ax.set_ylabel("m_hat = y/trend")
    ax.set_title("b03: multiplicative index - causal slice fits vs full fit")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "b03_index_overview.png", dpi=150)
    plt.close(fig)
    p("figure saved: b03_index_overview.png")

    summary["timings_sec"]["total"] = round(time.time() - t0, 1)
    (OUT_DIR / "index_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    p(f"DONE in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
