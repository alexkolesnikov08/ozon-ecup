"""Hurdle stage-0 priors (exp03): empirical P(buy) / E[z|buy] lookup tables.

For every CV anchor builds a segment grid over order-recency x search intent
and saves per-segment target statistics in z = log1p space:

    reports/exp03_hurdle_priors.json
    {fold: {overall: {...}, segments: {"rec=..|intent=..": {...}}}}

Segment keys: recency_to_ord_days bucket x searches-14d bucket
("never" = no orders in history).

Usage by the training agent: hurdle baseline pred_z = p_buy * mean_z_buyers
per segment; also calibration reference for learned P(buy|X) * E[z|buy,X].
Run from repo root:
    .venv/bin/python src/exp03_hurdle_priors.py
"""

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

FEAT_DIR = Path("data/v2/features_ext")
OUT_PATH = Path("reports/exp03_hurdle_priors.json")

CV_ANCHORS = {
    "fold_00": date(2025, 12, 3),
    "fold_01": date(2025, 12, 17),
    "fold_02": date(2025, 12, 31),
    "fold_03": date(2026, 1, 14),
}

RECENCY_EDGES = [0, 7, 14, 30, 90]
RECENCY_LABELS = ["0-7d", "7-14d", "14-30d", "30-90d", ">90d"]
INTENT_EDGES = [0.5, 1, 4, 12]
INTENT_LABELS = ["s=0", "s=1", "s=2-4", "s=5-12", "s>=12"]


def main() -> None:
    report: dict[str, dict] = {}

    for fold_name, anchor in CV_ANCHORS.items():
        feats = pl.read_parquet(FEAT_DIR / fold_name / "batch_*.parquet")
        y = feats["target"].to_numpy()
        z = np.log1p(y)
        buyers = y > 0

        rec = feats["recency_to_ord_days"].to_numpy()
        rec_idx = np.minimum(np.digitize(rec, RECENCY_EDGES, right=False), len(RECENCY_LABELS) - 1)
        rec_lab = np.array(RECENCY_LABELS, dtype=object)[rec_idx]
        never = ~np.isfinite(rec) | (rec >= 10 ** 8 - 1)
        rec_lab = np.where(never, "never", rec_lab)

        s14 = feats["x_searches_sum_14d"].to_numpy()
        s_idx = np.digitize(s14, INTENT_EDGES, right=False)
        s_lab = np.array(INTENT_LABELS, dtype=object)[s_idx]

        segments: dict[str, dict] = {}
        for rl in RECENCY_LABELS + ["never"]:
            for sl in INTENT_LABELS:
                m = (rec_lab == rl) & (s_lab == sl)
                n = int(m.sum())
                if n == 0:
                    continue
                mb = m & buyers
                segments[f"rec={rl}|{sl}"] = {
                    "n": n,
                    "p_buy": round(float(buyers[m].mean()), 4),
                    "mean_y": round(float(y[m].mean()), 2),
                    "mean_z_buyers": round(float(z[mb].mean()), 4) if mb.any() else None,
                    "mean_z_all": round(float(z[m].mean()), 4),
                }

        report[fold_name] = {
            "anchor": str(anchor),
            "target_window": f"[{anchor + timedelta(days=1)}, {anchor + timedelta(days=30)}]",
            "overall": {
                "n": int(len(y)),
                "p_buy": round(float(buyers.mean()), 4),
                "mean_z_buyers": round(float(z[buyers].mean()), 4),
                "mean_z_all": round(float(z.mean()), 4),
            },
            "segments": segments,
        }
        print(f"{fold_name}: {len(segments)} segments, overall p_buy="
              f"{report[fold_name]['overall']['p_buy']}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"\nsaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
