"""exp02 part A: extended feature construction (EWMA, trends, conversions,
frequency-x-check decomposition, due_ratio, GMV source shares).

Computes ONLY the new features from raw data/train.parquet using history
strictly up to and including the anchor. The 56 base features of exp01 are
reused read-only from data/v2/features/ and joined at train time
(src/exp02_train.py), which guarantees a clean ablation "base" vs "base+delta".

Output cache: data/v2/features_exp02/<fold_name>/batch_NNNN.parquet
columns: anchor_date, user_id, <17 new features>, target

Fold set = exp01 folds + fold_proxy (anchor 2025-02-14, YoY calendar twin of
the submission window, used for seasonal calibration) + fold_end (target NULL).

Documented design choices:
- EWMA normalisation denominator = sum of weights over the FULL 90-day calendar
  window ending at the anchor (a constant per half-life), not over observed
  rows only; days without activity contribute x=0 to the numerator.
- slope_loggmv_60d: closed-form OLS with x = 59 - day_age, i.e. positive
  slope = growth; missing days enter with y = log1p(0) = 0; n = 60 fixed;
  denominator n*Sxx - Sx^2 = 1079700 != 0 is a constant (the degenerate-case
  fallback slope=0 from the spec cannot trigger).
- conv_s2o / conv_c2o / conv_o2c / due_ratio remain NaN where undefined
  (zero denominator / fewer than 2 order-days); CatBoost handles NaN natively.
- pct_rank_gmv30 (spec block 7) needs the full fold population, so it is NOT
  computed here batch-wise; it is computed at train time on the assembled fold
  with a single polars rank expression (see src/exp02_train.py).
"""

import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

DATA_PATH = Path("data/train.parquet")
OUT_DIR = Path("data/v2/features_exp02")

BATCH_SIZE = 50_000
HORIZON_DAYS = 30

CV_ANCHORS = [
    date(2025, 12, 3),
    date(2025, 12, 17),
    date(2025, 12, 31),
    date(2026, 1, 14),
]
PROXY_ANCHOR = date(2025, 2, 14)
END_ANCHOR = date(2026, 2, 13)

FOLD_NAMES = ["fold_00", "fold_01", "fold_02", "fold_03", "fold_proxy", "fold_end"]
ANCHORS = [*CV_ANCHORS, PROXY_ANCHOR, END_ANCHOR]

WINDOW_COLS = [
    "gmv", "searches", "to_ord", "to_cart",
    "gmv_search", "gmv_cat", "search_to_ord", "cat_to_ord", "cat",
]
WINDOWS = [("7d", 7), ("30d", 30), ("90d", 90)]

HALF_LIVES = [(7, 7.0), (30, 30.0)]
EWMA_WINDOW = 90
SLOPE_N = 60
SLOPE_SX = sum(range(SLOPE_N))
SLOPE_DENOM = float(SLOPE_N * sum(i * i for i in range(SLOPE_N)) - SLOPE_SX**2)

ALLOWED_NAN = {"conv_s2o", "conv_c2o", "conv_o2c", "due_ratio"}

NEW_FEATURES = [
    "ewma_gmv_hl7", "ewma_gmv_hl30", "ewma_to_ord_hl7", "ewma_to_ord_hl30",
    "trend_gmv_7v30", "trend_gmv_30v90", "slope_loggmv_60d",
    "conv_s2o", "conv_c2o", "conv_o2c",
    "aov_30", "ord_days_30",
    "due_ratio",
    "share_gmv_search_90", "share_gmv_cat_90",
    "share_gmv_search_trend", "share_gmv_cat_trend",
]

BASE_VALUE_COLS = ["gmv", "searches", "to_ord", "to_cart"]
BASE_WINDOWS = [(7, "7d"), (30, "30d"), (60, "60d"), (90, "90d")]
RECENCY_NONE = 999


def ewma_denom(hl: float) -> float:
    return float(sum(0.5 ** (i / hl) for i in range(EWMA_WINDOW)))


def load_data() -> pl.DataFrame:
    df = pl.read_parquet(DATA_PATH)
    df = df.with_columns(
        pl.col("event_date").cast(pl.Date),
        pl.col("gmv").cast(pl.Float64),
        pl.col("gmv_search").cast(pl.Float64),
        pl.col("gmv_cat").cast(pl.Float64),
    )
    assert df["gmv"].min() >= 0 and df["to_ord"].min() >= 0
    print(f"loaded {df.shape[0]:,} rows x {df.shape[1]} cols, "
          f"dates {df['event_date'].min()} .. {df['event_date'].max()}", flush=True)
    return df


def window_masks(anchor: date) -> dict[str, pl.Expr]:
    return {
        tag: pl.col("event_date").is_between(anchor - timedelta(days=d - 1), anchor)
        for tag, d in WINDOWS
    }


def new_feature_exprs(anchor: date) -> tuple[list[pl.Expr], list[pl.Expr]]:
    masks = window_masks(anchor)
    age = (pl.lit(anchor) - pl.col("event_date")).dt.total_days()

    agg_exprs = []
    for col in WINDOW_COLS:
        for tag, _ in WINDOWS:
            agg_exprs.append(
                pl.when(masks[tag])
                .then(pl.col(col))
                .otherwise(0.0)
                .sum()
                .cast(pl.Float64)
                .alias(f"{col}_sum_{tag}")
            )

    m60 = pl.col("event_date").is_between(anchor - timedelta(days=SLOPE_N - 1), anchor)
    x = pl.lit(SLOPE_N - 1) - age
    lg = pl.col("gmv").log1p()
    agg_exprs.append(
        pl.when(m60).then(x * lg).otherwise(0.0).sum().cast(pl.Float64).alias("_sum_xy")
    )
    agg_exprs.append(
        pl.when(m60).then(lg).otherwise(0.0).sum().cast(pl.Float64).alias("_sum_y")
    )
    for src in ("gmv", "to_ord"):
        for hl_tag, hl in HALF_LIVES:
            w = pl.lit(0.5) ** (age / hl)
            agg_exprs.append(
                pl.when(masks["90d"]).then(w * pl.col(src)).otherwise(0.0)
                .sum().cast(pl.Float64).alias(f"_ewma_num_{src}_hl{hl_tag}")
            )

    derived = []
    for src in ("gmv", "to_ord"):
        for hl_tag, hl in HALF_LIVES:
            derived.append(
                (pl.col(f"_ewma_num_{src}_hl{hl_tag}") / ewma_denom(hl))
                .cast(pl.Float64)
                .alias(f"ewma_{src}_hl{hl_tag}")
            )

    g7 = pl.col("gmv_sum_7d")
    g30 = pl.col("gmv_sum_30d")
    g90 = pl.col("gmv_sum_90d")
    derived += [
        ((g7 / 7) / (g30 / 30).clip(lower_bound=1e-6)).alias("trend_gmv_7v30"),
        ((g30 / 30) / (g90 / 90).clip(lower_bound=1e-6)).alias("trend_gmv_30v90"),
        ((pl.lit(SLOPE_N) * pl.col("_sum_xy") - pl.lit(SLOPE_SX) * pl.col("_sum_y"))
         / pl.lit(SLOPE_DENOM)).cast(pl.Float64).alias("slope_loggmv_60d"),
    ]

    derived += [
        pl.when(pl.col("searches_sum_90d") > 0)
        .then(pl.col("search_to_ord_sum_90d") / pl.col("searches_sum_90d"))
        .otherwise(None).cast(pl.Float64).alias("conv_s2o"),
        pl.when(pl.col("cat_sum_90d") > 0)
        .then(pl.col("cat_to_ord_sum_90d") / pl.col("cat_sum_90d"))
        .otherwise(None).cast(pl.Float64).alias("conv_c2o"),
        pl.when(pl.col("to_cart_sum_90d") > 0)
        .then(pl.col("to_ord_sum_90d") / pl.col("to_cart_sum_90d"))
        .otherwise(None).cast(pl.Float64).alias("conv_o2c"),
    ]

    m30 = masks["30d"]
    agg_exprs.append(
        pl.when(m30 & (pl.col("to_ord") > 0)).then(1).otherwise(0)
        .sum().cast(pl.Float64).alias("ord_days_30")
    )
    derived += [
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
    derived += [
        share_s90.alias("share_gmv_search_90"),
        share_c90.alias("share_gmv_cat_90"),
        (share_s7 - share_s90).alias("share_gmv_search_trend"),
        (share_c7 - share_c90).alias("share_gmv_cat_trend"),
    ]

    return agg_exprs, derived


def due_ratio_frame(hist: pl.DataFrame, anchor: date) -> pl.DataFrame:
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


def build_new_batch(df_users: pl.DataFrame, anchor: date, with_target: bool) -> pl.DataFrame:
    hist = df_users.filter(pl.col("event_date") <= anchor)
    agg_exprs, derived = new_feature_exprs(anchor)

    feats = hist.group_by("user_id").agg(agg_exprs).with_columns(derived)
    feats = feats.join(due_ratio_frame(hist, anchor), on="user_id", how="left")

    idx = df_users.select("user_id").unique().with_columns(pl.lit(anchor).alias("anchor_date"))
    out = idx.join(feats, on="user_id", how="left")

    fill_cols = [c for c in NEW_FEATURES if c not in ALLOWED_NAN]
    out = out.with_columns(*[pl.col(c).fill_null(0.0) for c in fill_cols])

    if with_target:
        t = (
            df_users.filter(
                pl.col("event_date").is_between(
                    anchor + timedelta(days=1), anchor + timedelta(days=HORIZON_DAYS)
                )
            )
            .group_by("user_id")
            .agg(pl.col("gmv").sum().alias("target"))
        )
        out = out.join(t, on="user_id", how="left").with_columns(
            pl.col("target").fill_null(0.0).cast(pl.Float64)
        )
    else:
        out = out.with_columns(pl.lit(None, dtype=pl.Float64).alias("target"))

    return out.select(["anchor_date", "user_id", *NEW_FEATURES, "target"])


def base_feature_exprs(anchor: date) -> tuple[list[pl.Expr], list[pl.Expr]]:
    """Verbatim port of archive/exp01/src/features.py feature_exprs (56 feats)."""
    exprs = []
    for w_days, w_name in BASE_WINDOWS:
        w_start = anchor - timedelta(days=w_days - 1)
        mask = pl.col("event_date").is_between(w_start, anchor)
        for col in BASE_VALUE_COLS:
            c = pl.col(col)
            exprs.append(
                pl.when(mask).then(c).otherwise(0.0).sum().alias(f"{col}_sum_{w_name}")
            )
            exprs.append(
                pl.when(mask).then(c).otherwise(None).max().alias(f"{col}_max_{w_name}")
            )
            exprs.append(
                pl.when(mask).then(c).otherwise(None).mean().alias(f"{col}_mean_{w_name}")
            )

    m30 = pl.col("event_date").is_between(anchor - timedelta(days=29), anchor)
    exprs.append(
        pl.when(m30 & ((pl.col("gmv") > 0) | (pl.col("searches") > 0)))
        .then(1).otherwise(0).sum()
        .cast(pl.Float64)
        .alias("active_days_30d")
    )

    last_ord = pl.col("event_date").filter(pl.col("to_ord") > 0).max()
    last_srch = pl.col("event_date").filter(pl.col("searches") > 0).max()
    rec_ord = (
        pl.when(last_ord.is_null()).then(RECENCY_NONE)
        .otherwise((anchor - last_ord).dt.total_days())
    )
    rec_srch = (
        pl.when(last_srch.is_null()).then(RECENCY_NONE)
        .otherwise((anchor - last_srch).dt.total_days())
    )
    exprs += [
        rec_ord.cast(pl.Float64).alias("recency_to_ord_days"),
        rec_srch.cast(pl.Float64).alias("recency_searches_days"),
        (anchor - pl.col("event_date").min()).dt.total_days()
        .cast(pl.Float64)
        .alias("tenure_days"),
    ]

    s7 = pl.col("searches_sum_90d")
    tc = pl.col("to_cart_sum_90d")
    to = pl.col("to_ord_sum_90d")
    g9 = pl.col("gmv_sum_90d")
    conv_exprs = [
        (to / s7.clip(lower_bound=1)).alias("conv_to_ord_per_search_90d"),
        (tc / s7.clip(lower_bound=1)).alias("conv_to_cart_per_search_90d"),
        (to / tc.clip(lower_bound=1)).alias("conv_to_ord_per_cart_90d"),
        (g9 / to.clip(lower_bound=1)).alias("gmv_per_order_90d"),
    ]
    return exprs, conv_exprs


def build_base_batch(df_users: pl.DataFrame, anchor: date, with_target: bool) -> pl.DataFrame:
    """Verbatim port of archive/exp01/src/features.py build_batch (schema-compatible
    with data/v2/features/ caches; used only for fold_proxy which exp01 never built)."""
    parts = []
    hist = df_users.filter(pl.col("event_date") <= anchor)
    agg_exprs, conv_exprs = base_feature_exprs(anchor)
    feats = (
        hist.group_by("user_id")
        .agg(agg_exprs)
        .with_columns(conv_exprs)
    )
    idx = (
        df_users.select("user_id")
        .unique()
        .with_columns(pl.lit(anchor).alias("anchor_date"))
    )
    out = idx.join(feats, on=["user_id"], how="left")

    if with_target:
        t = (
            df_users.filter(
                pl.col("event_date").is_between(
                    anchor + timedelta(days=1), anchor + timedelta(days=HORIZON_DAYS))
            )
            .group_by("user_id")
            .agg(pl.col("gmv").sum().alias("target"))
        )
        out = out.join(t, on="user_id", how="left").with_columns(
            pl.col("target").fill_null(0.0)
        )
    else:
        out = out.with_columns(pl.lit(None, dtype=pl.Float64).alias("target"))

    feat_cols = [c for c in out.columns if c not in ("anchor_date", "user_id", "target")]
    out = out.with_columns(*[pl.col(c).fill_null(0.0) for c in feat_cols])
    parts.append(out.select(["anchor_date", "user_id", *feat_cols, "target"]))
    return pl.concat(parts)


def validate_batch(out: pl.DataFrame, expected_rows: int) -> None:
    assert out.height == expected_rows, f"rows {out.height} != {expected_rows}"
    assert out.schema["anchor_date"] == pl.Date and out.schema["user_id"] == pl.Int64
    for c in NEW_FEATURES:
        assert out.schema[c] == pl.Float64, f"{c}: {out.schema[c]}"
    inf_counts = out.select([pl.col(c).is_infinite().sum().alias(c) for c in NEW_FEATURES]).row(0)
    assert sum(inf_counts) == 0, f"inf found: {dict(zip(NEW_FEATURES, inf_counts))}"


def leakage_check(data: pl.DataFrame) -> None:
    users = sorted(data["user_id"].unique().to_list())
    ids = [users[0], users[len(users) // 2], users[-1]]
    anchors = [date(2025, 12, 31), date(2026, 2, 13)]
    sub_full = data.filter(pl.col("user_id").is_in(ids))
    for a in anchors:
        f_full = build_new_batch(sub_full, a, with_target=False)
        f_trunc = build_new_batch(sub_full.filter(pl.col("event_date") <= a), a, with_target=False)
        j = f_full.drop("anchor_date", "target").join(
            f_trunc.drop("anchor_date", "target"), on="user_id", suffix="_tr"
        )
        for c in NEW_FEATURES:
            x = np.nan_to_num(j[c].fill_null(float("nan")).to_numpy(), nan=np.nan)
            y = np.nan_to_num(j[f"{c}_tr"].fill_null(float("nan")).to_numpy(), nan=np.nan)
            same = np.isclose(x, y, rtol=1e-9, atol=1e-12, equal_nan=True)
            assert same.all(), f"LEAKAGE: {c} changed at anchor {a} for users {ids}"
    print(f"leakage check OK: users={ids}, anchors={anchors} "
          f"(features identical with post-anchor rows removed)", flush=True)


def main() -> None:
    t0 = time.time()
    data = load_data()

    leakage_check(data)

    user_ids = data["user_id"].unique().sort()
    n_users = len(user_ids)
    n_batches = (n_users + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"{n_users:,} users -> {n_batches} batches per fold, {len(FOLD_NAMES)} folds\n", flush=True)

    for fold_name, anchor in zip(FOLD_NAMES, ANCHORS):
        fold_dir = OUT_DIR / fold_name
        fold_dir.mkdir(parents=True, exist_ok=True)
        with_target = fold_name != "fold_end"
        tf = time.time()
        for b in range(n_batches):
            out_path = fold_dir / f"batch_{b:04d}.parquet"
            if out_path.exists():
                print(f"  {fold_name} batch {b + 1}/{n_batches}: exists, skip", flush=True)
                continue
            tb = time.time()
            ids_b = user_ids[b * BATCH_SIZE:(b + 1) * BATCH_SIZE].to_list()
            df_b = data.filter(pl.col("user_id").is_in(ids_b))
            out = build_new_batch(df_b, anchor, with_target=with_target)
            validate_batch(out, expected_rows=len(ids_b))
            out.write_parquet(out_path)
            print(f"  {fold_name} batch {b + 1}/{n_batches}: {out.height} rows "
                  f"({time.time() - tb:.1f}s)", flush=True)
        print(f"{fold_name} done (anchor {anchor}) in {time.time() - tf:.1f}s\n", flush=True)

    base_dir = OUT_DIR / "fold_proxy_base"
    if not (base_dir / f"batch_{n_batches - 1:04d}.parquet").exists():
        base_dir.mkdir(parents=True, exist_ok=True)
        print(f"fold_proxy_base: exp01-style 56 base features (no cache in data/v2/features)", flush=True)
        tf = time.time()
        for b in range(n_batches):
            out_path = base_dir / f"batch_{b:04d}.parquet"
            if out_path.exists():
                continue
            ids_b = user_ids[b * BATCH_SIZE:(b + 1) * BATCH_SIZE].to_list()
            df_b = data.filter(pl.col("user_id").is_in(ids_b))
            out = build_base_batch(df_b, PROXY_ANCHOR, with_target=True)
            assert out.height == len(ids_b)
            assert out.select(pl.all().null_count()).row(0) == (0,) * out.width
            out.write_parquet(out_path)
            print(f"  fold_proxy_base batch {b + 1}/{n_batches}: {out.height} rows", flush=True)
        print(f"fold_proxy_base done in {time.time() - tf:.1f}s\n", flush=True)
    else:
        print("fold_proxy_base: exists, skip\n", flush=True)

    print("=== final validation over all folds ===", flush=True)
    for fold_name in FOLD_NAMES:
        df = pl.read_parquet(OUT_DIR / fold_name / "batch_*.parquet")
        validate_batch(df, expected_rows=n_users)
        nulls = {c: int(df[c].is_null().sum()) for c in NEW_FEATURES}
        bad = {c: v for c, v in nulls.items() if c not in ALLOWED_NAN and v > 0}
        assert not bad, f"{fold_name}: unexpected nulls {bad}"
        tgt_nulls = int(df["target"].is_null().sum())
        print(f"{fold_name}: rows={df.height}, nulls(by design)={nulls}, "
              f"target_nulls={tgt_nulls}", flush=True)

    print(f"ALL DONE in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
