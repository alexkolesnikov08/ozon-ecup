"""Level-drift calibration indices (exp06).

Computes platform-level multipliers for post-hoc calibration of predictions
in z space:  z' = z + c + beta * log(s_index)

Outputs reports/exp06_calibration.json with:

- s_seasonal      — twin-window index: GMV [14.02–15.03.25] vs prior month
                    [15.01–13.02.25] (holiday-season uplift of the target window)
- yoy_cohort_user — same-cohort per-user YoY growth ([15.01–13.02] 2026/2025,
                    identical users) — structural platform drift
- local_drift     — per-active-user GMV ratio of the last two aligned 30d
                    windows in data: [14.01–12.02.26] vs [15.12–13.01.26]
                    (short-term momentum right before the submit anchor)
- recommended     — assembled c and beta suggestions to grid-search

Run from repo root:
    .venv/bin/python src/exp06_calibration_indices.py
"""

import json
from datetime import date
from pathlib import Path

import polars as pl

DATA_PATH = Path("data/train.parquet")
OUT_PATH = Path("reports/exp06_calibration.json")


def win_gmv(daily: pl.DataFrame, d0: date, d1: date, col: str = "gmv") -> float:
    return float(
        daily.filter(pl.col("event_date").is_between(d0, d1))[col].sum()
    )


def main() -> None:
    daily = (
        pl.read_parquet(DATA_PATH)
        .group_by("event_date")
        .agg(pl.col("gmv").sum().alias("gmv"), pl.col("user_id").len().alias("rows"))
        .sort("event_date")
    )

    s_seasonal = win_gmv(daily, date(2025, 2, 14), date(2025, 3, 15)) / \
        win_gmv(daily, date(2025, 1, 15), date(2025, 2, 13))

    raw = pl.read_parquet(DATA_PATH)
    w25 = raw.filter(pl.col("event_date").is_between(date(2025, 1, 15), date(2025, 2, 13))) \
        .group_by("user_id").agg(pl.col("gmv").sum().alias("g25"))
    w26 = raw.filter(pl.col("event_date").is_between(date(2026, 1, 15), date(2026, 2, 13))) \
        .group_by("user_id").agg(pl.col("gmv").sum().alias("g26"))
    coh = w25.join(w26, on="user_id", how="inner")
    yoy_cohort_user = float(coh["g26"].mean() / coh["g25"].mean())

    recent = daily.with_columns(
        (pl.col("gmv") / pl.col("rows")).alias("gmv_per_row_day")
    )
    m_pre_submit = float(
        recent.filter(pl.col("event_date").is_between(date(2026, 1, 14), date(2026, 2, 13)))
        ["gmv_per_row_day"].mean()
    )
    m_before = float(
        recent.filter(pl.col("event_date").is_between(date(2025, 12, 16), date(2026, 1, 14)))
        ["gmv_per_row_day"].mean()
    )
    local_drift = m_pre_submit / m_before

    out = {
        "s_seasonal_twin_window": round(s_seasonal, 4),
        "yoy_cohort_user": round(yoy_cohort_user, 4),
        "local_drift_last_30d_vs_prev_30d": round(local_drift, 4),
        "windows": {
            "seasonal_num": "[2025-02-14 .. 2025-03-15]",
            "seasonal_den": "[2025-01-15 .. 2025-02-13]",
            "yoy_num": "[2026-01-15 .. 2026-02-13] mean gmv/user",
            "yoy_den": "[2025-01-15 .. 2025-02-13] mean gmv/user, same cohort",
            "drift_num": "[2026-01-14 .. 2026-02-13] gmv/row/day",
            "drift_den": "[2025-12-16 .. 2026-01-14] gmv/row/day",
        },
        "recommended": {
            "formula_z_space": "z' = z + c + beta*log(s); s in {s_seasonal, local_drift}",
            "grid_c_log_s_local": [-0.05, -0.025, 0.0, 0.025, 0.05],
            "grid_beta_log_s_seasonal": [0.0, 0.25, 0.5, 0.75, 1.0],
            "note": "подбирать c/beta на pooled OOF всех фолдов; fold_end предикт калибруется последним",
        },
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"s_seasonal (twin vs prev month): {s_seasonal:.4f}")
    print(f"yoy_cohort_user:                 {yoy_cohort_user:.4f}")
    print(f"local_drift (pre-submit window): {local_drift:.4f}")
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
