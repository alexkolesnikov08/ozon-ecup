"""b03 step 2: feature cache on STL-normalised daily values (variant ii/iii).

Verbatim port of the exp02 feature formulas (56 base + 10 accepted extras =
the 66-feature config), evaluated on x~_d = x_d / m_hat_d where m_hat is the
CAUSAL per-anchor slice index from src/b03_index.py. Raw-value features are
additionally written for fold_00/fold_03 to validate the port against the
official read-only caches (exact-match assertion).

Output: data/v2/b03_plsi/<fold>/batch_NNNN.parquet
  columns: anchor_date, user_id, <66 normalised features>, target,
           (<66 raw features> suffixed _raw for fold_00/fold_03)

Design notes (inherited from exp02 unless stated):
- history is strictly event_date <= anchor; the STL index is the per-anchor
  causal slice fit, so normalisation itself cannot see post-anchor data;
- EWMA denominator = full 90-day calendar window weight sum (constant);
- conversions / due_ratio stay NaN where undefined (CatBoost handles NaN);
- ord_days_30 counted as number of days with to_ord>0 in the 30d window.
"""

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import time  # noqa: E402
from datetime import timedelta  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from b03_common import (  # noqa: E402
    ALLOWED_NAN, ANCHORS, BASE_VALUE_COLS, BASE_WINDOWS, EXTRA_ACCEPTED,
    HORIZON, INDEX_DIR, METRIC_COLS, OUT_DIR, PARTS_DIR, RECENCY_NONE,
    WINDOW_COLS, WINDOWS,
)

BATCH_SIZE = 50_000
RAW_CHECK_FOLDS = ("fold_00", "fold_03")

FEATURES_66_ORDER = [
    *(f"{c}_sum_{w}" for w in ("7d", "30d", "60d", "90d") for c in BASE_VALUE_COLS),
    *(f"{c}_max_{w}" for w in ("7d", "30d", "60d", "90d") for c in BASE_VALUE_COLS),
    *(f"{c}_mean_{w}" for w in ("7d", "30d", "60d", "90d") for c in BASE_VALUE_COLS),
    "active_days_30d", "recency_to_ord_days", "recency_searches_days", "tenure_days",
    "conv_to_ord_per_search_90d", "conv_to_cart_per_search_90d",
    "conv_to_ord_per_cart_90d", "gmv_per_order_90d",
    *EXTRA_ACCEPTED,
]
assert len(FEATURES_66_ORDER) == 66, len(FEATURES_66_ORDER)


def load_data() -> pl.DataFrame:
    need = ["event_date", "user_id", *METRIC_COLS]
    df = pl.read_parquet("data/train.parquet", columns=need)
    df = df.with_columns(
        pl.col("event_date").cast(pl.Date),
        *[pl.col(c).cast(pl.Float64) for c in METRIC_COLS],
    )
    assert df["gmv"].min() >= 0 and df["to_ord"].min() >= 0
    return df.select(need)


def partition_users(df: pl.DataFrame, n_parts: int) -> None:
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    users = df["user_id"].unique().sort()
    assert users.len() == n_parts * BATCH_SIZE
    for b in range(n_parts):
        path = PARTS_DIR / f"part_{b:04d}.parquet"
        if path.exists():
            print(f"part {b}: exists, skip", flush=True)
            continue
        ids = users[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
        part = df.filter(pl.col("user_id").is_in(ids))
        assert part["user_id"].n_unique() == BATCH_SIZE
        part.write_parquet(path)
        print(f"part {b}: {part.height:,} rows, {BATCH_SIZE} users", flush=True)


# ---------------- feature expressions (verbatim port of exp02) ----------------

def base_agg_exprs(anchor):
    """archive/exp01/src/features.py feature_exprs (window/recency aggregates)."""
    exprs = []
    for w_days, w_name in BASE_WINDOWS:
        mask = pl.col("event_date").is_between(anchor - timedelta(days=w_days - 1), anchor)
        for col in BASE_VALUE_COLS:
            c = pl.col(col)
            exprs.append(
                pl.when(mask).then(c).otherwise(0.0).sum().alias(f"{col}_sum_{w_name}"))
            exprs.append(
                pl.when(mask).then(c).otherwise(None).max().alias(f"{col}_max_{w_name}"))
            exprs.append(
                pl.when(mask).then(c).otherwise(None).mean().alias(f"{col}_mean_{w_name}"))
    m30 = pl.col("event_date").is_between(anchor - timedelta(days=29), anchor)
    exprs.append(
        pl.when(m30 & ((pl.col("gmv") > 0) | (pl.col("searches") > 0)))
        .then(1).otherwise(0).sum().cast(pl.Float64).alias("active_days_30d"))
    last_ord = pl.col("event_date").filter(pl.col("to_ord") > 0).max()
    last_srch = pl.col("event_date").filter(pl.col("searches") > 0).max()
    exprs += [
        pl.when(last_ord.is_null()).then(RECENCY_NONE)
        .otherwise((anchor - last_ord).dt.total_days()).cast(pl.Float64)
        .alias("recency_to_ord_days"),
        pl.when(last_srch.is_null()).then(RECENCY_NONE)
        .otherwise((anchor - last_srch).dt.total_days()).cast(pl.Float64)
        .alias("recency_searches_days"),
        (anchor - pl.col("event_date").min()).dt.total_days().cast(pl.Float64)
        .alias("tenure_days"),
    ]
    return exprs


def base_conv_exprs():
    """exp01 conv features, applied AFTER the window aggregation."""
    s7 = pl.col("searches_sum_90d")
    tc = pl.col("to_cart_sum_90d")
    to = pl.col("to_ord_sum_90d")
    g9 = pl.col("gmv_sum_90d")
    return [
        (to / s7.clip(lower_bound=1)).alias("conv_to_ord_per_search_90d"),
        (tc / s7.clip(lower_bound=1)).alias("conv_to_cart_per_search_90d"),
        (to / tc.clip(lower_bound=1)).alias("conv_to_ord_per_cart_90d"),
        (g9 / to.clip(lower_bound=1)).alias("gmv_per_order_90d"),
    ]


def extra_agg_exprs(anchor):
    """Window sums over the 9 value columns + ord_days_30 (exp02 block inputs)."""
    masks = {
        tag: pl.col("event_date").is_between(anchor - timedelta(days=d - 1), anchor)
        for tag, d in WINDOWS
    }
    exprs = []
    for col in WINDOW_COLS:
        for tag, _ in WINDOWS:
            exprs.append(
                pl.when(masks[tag]).then(pl.col(col)).otherwise(0.0).sum()
                .cast(pl.Float64).alias(f"{col}_sum_{tag}")
            )
    exprs.append(
        pl.when(masks["30d"] & (pl.col("to_ord") > 0)).then(1).otherwise(0)
        .sum().cast(pl.Float64).alias("ord_days_30")
    )
    return exprs


def extra_derived_exprs():
    """exp02 accepted extras: conversions, AOV, shares (+ trends)."""
    g7 = pl.col("gmv_sum_7d")
    g30 = pl.col("gmv_sum_30d")
    g90 = pl.col("gmv_sum_90d")
    out = [
        pl.when(pl.col("searches_sum_90d") > 0)
        .then(pl.col("search_to_ord_sum_90d") / pl.col("searches_sum_90d"))
        .otherwise(None).cast(pl.Float64).alias("conv_s2o"),
        pl.when(pl.col("cat_sum_90d") > 0)
        .then(pl.col("cat_to_ord_sum_90d") / pl.col("cat_sum_90d"))
        .otherwise(None).cast(pl.Float64).alias("conv_c2o"),
        pl.when(pl.col("to_cart_sum_90d") > 0)
        .then(pl.col("to_ord_sum_90d") / pl.col("to_cart_sum_90d"))
        .otherwise(None).cast(pl.Float64).alias("conv_o2c"),
        (g30 / pl.col("to_ord_sum_30d").clip(lower_bound=1)).cast(pl.Float64).alias("aov_30"),
    ]
    gs7 = pl.col("gmv_search_sum_7d")
    gs90 = pl.col("gmv_search_sum_90d")
    gc7 = pl.col("gmv_cat_sum_7d")
    gc90 = pl.col("gmv_cat_sum_90d")
    share_s7 = gs7 / g7.clip(lower_bound=1e-6)
    share_s90 = gs90 / g90.clip(lower_bound=1e-6)
    share_c7 = gc7 / g7.clip(lower_bound=1e-6)
    share_c90 = gc90 / g90.clip(lower_bound=1e-6)
    out += [
        share_s90.alias("share_gmv_search_90"),
        share_c90.alias("share_gmv_cat_90"),
        (share_s7 - share_s90).alias("share_gmv_search_trend"),
        (share_c7 - share_c90).alias("share_gmv_cat_trend"),
    ]
    return out


def due_ratio_frame(hist: pl.DataFrame, anchor):
    od = (
        hist.filter(pl.col("to_ord") > 0)
        .select("user_id", "event_date")
        .sort(["user_id", "event_date"])
    )
    if od.height == 0:
        return pl.DataFrame(schema={"user_id": pl.Int64, "due_ratio": pl.Float64})
    gaps = od.with_columns(
        pl.col("event_date").diff().over("user_id").dt.total_days()
        .cast(pl.Float64).alias("gap")
    )
    stat = gaps.group_by("user_id").agg(
        pl.col("gap").median().alias("_med_gap"),
        pl.len().alias("_n_ord_days"),
        pl.col("event_date").max().alias("_last_ord"),
    )
    return stat.select(
        "user_id",
        pl.when((pl.col("_n_ord_days") >= 2) & (pl.col("_med_gap") > 0))
        .then((pl.lit(anchor) - pl.col("_last_ord")).dt.total_days() / pl.col("_med_gap"))
        .otherwise(None)
        .cast(pl.Float64)
        .alias("due_ratio"),
    )


def build_features(frame: pl.DataFrame, anchor, full_users: pl.DataFrame) -> pl.DataFrame:
    hist = frame.filter(pl.col("event_date") <= anchor)
    f_base = (
        hist.group_by("user_id").agg(base_agg_exprs(anchor))
        .with_columns(base_conv_exprs())
    )
    f_extra = hist.group_by("user_id").agg(extra_agg_exprs(anchor))
    feats = (
        f_base.join(f_extra, on="user_id", how="inner")
        .with_columns(extra_derived_exprs())
        .join(due_ratio_frame(hist, anchor), on="user_id", how="left")
    )
    out = full_users.join(feats, on="user_id", how="left")
    fill = [c for c in FEATURES_66_ORDER if c not in ALLOWED_NAN]
    out = out.with_columns(*[pl.col(c).fill_null(0.0) for c in fill])
    return out.select(["user_id", *FEATURES_66_ORDER])


def target_frame(df: pl.DataFrame, anchor, with_target: bool) -> pl.DataFrame:
    if not with_target:
        return df.select("user_id").unique().with_columns(
            pl.lit(None, dtype=pl.Float64).alias("target"))
    t = (
        df.filter(pl.col("event_date").is_between(
            anchor + timedelta(days=1), anchor + timedelta(days=HORIZON)))
        .group_by("user_id")
        .agg(pl.col("gmv").sum().alias("target"))
    )
    return df.select("user_id").unique().join(t, on="user_id", how="left").with_columns(
        pl.col("target").fill_null(0.0).cast(pl.Float64))


def validate_batch(out: pl.DataFrame, raw_cols: list[str]) -> None:
    assert out.height == BATCH_SIZE
    assert out.schema["anchor_date"] == pl.Date and out.schema["user_id"] == pl.Int64
    assert out.schema["target"] == pl.Float64
    for c in FEATURES_66_ORDER:
        assert out.schema[c] == pl.Float64, f"{c}: {out.schema[c]}"
        if c not in ALLOWED_NAN:
            assert out[c].null_count() == 0, f"nulls in {c}"
            assert out[c].is_finite().all(), f"inf in {c}"
    for c in raw_cols:
        base = c[:-4]
        assert out.schema[c] == pl.Float64
        if base not in ALLOWED_NAN:
            assert out[c].null_count() == 0, f"nulls in {c}"
            assert out[c].is_finite().all(), f"inf in {c}"


def check_against_official(fold: str) -> dict:
    official_base_dir = Path(f"data/v2/features/{fold}")
    if not official_base_dir.exists():
        official_base_dir = Path(f"data/v2/features_exp02/{fold}_base")
    base = pl.read_parquet(str(official_base_dir / "batch_*.parquet"))
    extra = pl.read_parquet(f"data/v2/features_exp02/{fold}/batch_*.parquet")
    off = base.join(extra.select(["user_id", *EXTRA_ACCEPTED]), on="user_id", how="inner")
    mine = pl.read_parquet(OUT_DIR / fold / "batch_*.parquet")
    j = off.join(mine, on="user_id", how="inner", suffix="_mine")
    worst = 0.0
    nan_bad = 0
    for c in FEATURES_66_ORDER:
        a = j[c].to_numpy()
        b = j[f"{c}_raw"].to_numpy()  # plain-name cols of the b03 cache are normalised
        both_nan = np.isnan(a) & np.isnan(b)
        nan_bad += int((np.isnan(a) != np.isnan(b)).sum())
        diff = np.where(both_nan, 0.0, np.abs(np.nan_to_num(a) - np.nan_to_num(b)))
        scale = np.maximum(np.abs(np.nan_to_num(a)), 1.0)
        worst = max(worst, float((diff / scale).max()))
    ok = worst < 1e-9 and nan_bad == 0
    assert ok, f"{fold}: port mismatch worst_rel={worst:.3e} nan_mismatch={nan_bad}"
    return {"fold": fold, "rows_compared": j.height, "n_cols": len(FEATURES_66_ORDER),
            "worst_rel_diff": worst, "nan_pattern_mismatches": nan_bad, "ok": True}


def main() -> None:
    t0 = time.time()

    def p(msg: str) -> None:
        print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)

    summary = {"batches": {}, "port_check": {}, "timings_sec": {}}

    parts = sorted(PARTS_DIR.glob("part_*.parquet")) if PARTS_DIR.exists() else []
    if len(parts) != 5:
        p("partitioning users into 5 parts...")
        t1 = time.time()
        partition_users(load_data(), 5)
        parts = sorted(PARTS_DIR.glob("part_*.parquet"))
        summary["timings_sec"]["partition"] = round(time.time() - t1, 1)
    assert len(parts) == 5
    p(f"{len(parts)} user parts ready")

    for fold, anchor in ANCHORS.items():
        with_target = fold != "fold_end"
        idx = pl.read_parquet(INDEX_DIR / f"{fold}.parquet")
        inv = idx.select(
            pl.col("event_date"),
            (1.0 / pl.col("m_hat")).alias("_inv_m"),
        )
        raw_flag = fold in RAW_CHECK_FOLDS
        fold_dir = OUT_DIR / fold
        fold_dir.mkdir(parents=True, exist_ok=True)
        tf = time.time()
        built_now = 0
        for pi, part_path in enumerate(parts):
            out_path = fold_dir / f"batch_{pi:04d}.parquet"
            if out_path.exists():
                continue
            tb = time.time()
            df = pl.read_parquet(part_path)
            norm = (
                df.join(inv, on="event_date", how="left")
                .with_columns(*[(pl.col(c) * pl.col("_inv_m")).alias(c)
                                for c in METRIC_COLS])
                .drop("_inv_m")
            )
            # rows after the anchor keep raw values (only used for the target);
            # every feature expression filters to event_date <= anchor anyway
            full_users = df.select("user_id").unique().with_columns(
                pl.lit(anchor).alias("anchor_date"))
            feats_n = build_features(norm, anchor, full_users)
            tgt = target_frame(df, anchor, with_target)
            out = (
                feats_n.join(tgt, on="user_id", how="inner")
                .with_columns(pl.lit(anchor).alias("anchor_date"))
                .select(["anchor_date", "user_id", *FEATURES_66_ORDER, "target"])
            )
            raw_cols: list[str] = []
            if raw_flag:
                ren = {c: f"{c}_raw" for c in FEATURES_66_ORDER}
                feats_r = build_features(df, anchor, full_users).rename(ren)
                raw_cols = list(ren.values())
                out = out.join(feats_r, on="user_id", how="inner").select(
                    ["anchor_date", "user_id", *FEATURES_66_ORDER, "target", *raw_cols])
            validate_batch(out, raw_cols)
            out.write_parquet(out_path)
            built_now += 1
            p(f"  {fold} batch {pi}: {out.height} rows ({time.time() - tb:.1f}s)")
        summary["batches"][fold] = {"built_now": built_now}
        summary["timings_sec"][f"features_{fold}"] = round(time.time() - tf, 1)
        p(f"{fold} done in {time.time() - tf:.1f}s")

    for fold in RAW_CHECK_FOLDS:
        tp = time.time()
        rep = check_against_official(fold)
        summary["port_check"][fold] = rep
        p(f"port check vs official cache {fold}: worst_rel_diff="
          f"{rep['worst_rel_diff']:.3e}, nan_mismatch={rep['nan_pattern_mismatches']} "
          f"({time.time() - tp:.1f}s)")

    summary["timings_sec"]["total"] = round(time.time() - t0, 1)
    (OUT_DIR / "features_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    p(f"ALL DONE in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
