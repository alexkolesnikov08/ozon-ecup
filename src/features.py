"""Batched feature/target construction for LTV prediction (exp01).

Builds windowed aggregates + recency/tenure/conversion features for each
time-CV anchor and the final production anchor, writing parquet batches to
data/v2/features/<fold_name>/batch_NNNN.parquet.
"""

import time
from datetime import date, timedelta
from pathlib import Path

import polars as pl

DATA_PATH = Path("data/train.parquet")
OUT_DIR = Path("data/v2/features")

BATCH_SIZE = 50_000
HORIZON_DAYS = 30

CV_ANCHORS = [
    date(2025, 12, 3),
    date(2025, 12, 17),
    date(2025, 12, 31),
    date(2026, 1, 14),
]
END_ANCHOR = date(2026, 2, 13)

FOLD_NAMES = [f"fold_{i:02d}" for i in range(len(CV_ANCHORS))] + ["fold_end"]
ANCHORS = CV_ANCHORS + [END_ANCHOR]

VALUE_COLS = ["gmv", "searches", "to_ord", "to_cart"]
WINDOWS = [(7, "7d"), (30, "30d"), (60, "60d"), (90, "90d")]

RECENCY_NONE = 999


def load_data() -> pl.DataFrame:
    df = pl.read_parquet(DATA_PATH)
    df = df.with_columns(
        pl.col("event_date").cast(pl.Date),
        *[pl.col(c).cast(pl.UInt32) for c in VALUE_COLS if c != "gmv"],
        pl.col("gmv").cast(pl.Float64),
    )
    print(f"loaded {df.shape[0]:,} rows x {df.shape[1]} cols, "
          f"dates {df['event_date'].min()} .. {df['event_date'].max()}")
    return df


def feature_exprs(anchor: date) -> list[pl.Expr]:
    exprs = []
    for w_days, w_name in WINDOWS:
        w_start = anchor - timedelta(days=w_days - 1)
        mask = pl.col("event_date").is_between(w_start, anchor)
        for col in VALUE_COLS:
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


def build_batch(df_users: pl.DataFrame, anchors: list[date],
                with_target: bool) -> pl.DataFrame:
    parts = []
    for a in anchors:
        hist = df_users.filter(pl.col("event_date") <= a)
        agg_exprs, conv_exprs = feature_exprs(a)
        feats = (
            hist.group_by("user_id")
            .agg(agg_exprs)
            .with_columns(conv_exprs)
        )

        idx = (
            df_users.select("user_id")
            .unique()
            .with_columns(pl.lit(a).alias("anchor_date"))
        )

        out = idx.join(feats, on=["user_id"], how="left")

        if with_target:
            t = (
                df_users.filter(
                    pl.col("event_date").is_between(a + timedelta(days=1), a + timedelta(days=HORIZON_DAYS))
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


def main() -> None:
    t0 = time.time()
    data = load_data()

    user_ids = data["user_id"].unique().sort()
    n_users = len(user_ids)
    n_batches = (n_users + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"{n_users:,} users -> {n_batches} batches per fold\n")

    for fold_name, anchor in zip(FOLD_NAMES, ANCHORS):
        fold_dir = OUT_DIR / fold_name
        fold_dir.mkdir(parents=True, exist_ok=True)
        with_target = fold_name != "fold_end"
        tf = time.time()
        for b in range(n_batches):
            tb = time.time()
            ids_b = user_ids[b * BATCH_SIZE:(b + 1) * BATCH_SIZE].to_list()
            df_b = data.filter(pl.col("user_id").is_in(ids_b))
            out = build_batch(df_b, [anchor], with_target=with_target)
            out.write_parquet(fold_dir / f"batch_{b:04d}.parquet")
            print(f"  {fold_name} batch {b + 1}/{n_batches}: {out.height} rows "
                  f"({time.time() - tb:.1f}s)", flush=True)
        print(f"{fold_name} done (anchor {anchor}) in {time.time() - tf:.1f}s\n", flush=True)

    print(f"ALL DONE in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
