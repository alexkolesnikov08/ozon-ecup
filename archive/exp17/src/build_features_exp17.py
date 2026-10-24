import time
from datetime import date, timedelta
from pathlib import Path

import polars as pl

DATA_PATH = Path("data/train.parquet")
OUT_DIR = Path("data/v2/features_exp17")
BATCH_SIZE = 50_000

CV_ANCHORS = [
    date(2025, 12, 3),
    date(2025, 12, 17),
    date(2025, 12, 31),
    date(2026, 1, 14),
]
END_ANCHOR = date(2026, 2, 13)

FOLD_NAMES = [f"fold_{i:02d}" for i in range(len(CV_ANCHORS))] + ["fold_end"]
ANCHORS = CV_ANCHORS + [END_ANCHOR]


def load_data() -> pl.DataFrame:
    df = pl.read_parquet(DATA_PATH)
    df = df.with_columns(
        pl.col("event_date").cast(pl.Date),
        pl.col("gmv").cast(pl.Float64),
        pl.col("to_cart").cast(pl.UInt32),
        pl.col("to_ord").cast(pl.UInt32),
        pl.col("searches").cast(pl.UInt32),
    )
    return df


def feature_exprs(anchor: date) -> list[pl.Expr]:
    exprs = []

    m7 = pl.col("event_date").is_between(anchor - timedelta(days=6), anchor)
    m14 = pl.col("event_date").is_between(anchor - timedelta(days=13), anchor)
    m30 = pl.col("event_date").is_between(anchor - timedelta(days=29), anchor)
    m60 = pl.col("event_date").is_between(anchor - timedelta(days=59), anchor)

    gmv7 = pl.when(m7).then(pl.col("gmv")).otherwise(0.0).sum()
    gmv14 = pl.when(m14).then(pl.col("gmv")).otherwise(0.0).sum()
    gmv30 = pl.when(m30).then(pl.col("gmv")).otherwise(0.0).sum()
    gmv60 = pl.when(m60).then(pl.col("gmv")).otherwise(0.0).sum()

    tc14 = pl.when(m14).then(pl.col("to_cart")).otherwise(0.0).sum()
    to14 = pl.when(m14).then(pl.col("to_ord")).otherwise(0.0).sum()

    s30 = pl.when(m30).then(pl.col("searches")).otherwise(0.0).sum()
    act30 = pl.when(m30 & ((pl.col("gmv") > 0) | (pl.col("searches") > 0))).then(1).otherwise(0).sum().cast(pl.Float64)


    exprs.append((gmv7 / (gmv30 / 4.0).clip(lower_bound=1)).alias("trend_gmv_7d_vs_30d"))
    exprs.append((gmv14 / (gmv60 / 4.0).clip(lower_bound=1)).alias("trend_gmv_14d_vs_60d"))

    exprs.append((tc14 - to14).clip(lower_bound=0).cast(pl.Float64).alias("abandoned_cart_items_14d"))

    exprs.append((s30 / act30.clip(lower_bound=1)).alias("searches_per_active_day_30d"))

    last_ord = pl.col("event_date").filter(pl.col("to_ord") > 0).max()
    first_event = pl.col("event_date").min()

    recency = pl.when(last_ord.is_null()).then(999.0).otherwise((anchor - last_ord).dt.total_days().cast(pl.Float64))
    tenure = (anchor - first_event).dt.total_days().cast(pl.Float64)
    buy_days = pl.col("event_date").filter(pl.col("to_ord") > 0).n_unique().cast(pl.Float64)

    exprs.append(
        pl.when(buy_days > 1)
        .then((tenure - recency).clip(lower_bound=0) / (buy_days - 1))
        .otherwise(999.0)
        .alias("mean_inter_purchase_days")
    )

    return exprs


def build_batch(df_users: pl.DataFrame, anchor: date) -> pl.DataFrame:
    hist = df_users.filter(pl.col("event_date") <= anchor)
    agg_exprs = feature_exprs(anchor)

    feats = hist.group_by("user_id").agg(agg_exprs)
    out = df_users.select("user_id").unique().with_columns(pl.lit(anchor).alias("anchor_date"))
    out = out.join(feats, on="user_id", how="left")

    target_start = anchor + timedelta(days=1)
    target_end = anchor + timedelta(days=30)

    ny_season = date(2025, 12, 15) <= target_end and date(2025, 12, 31) >= target_start
    # 8 Марта / 23 Февраля (ажиотаж с 20 февраля по 8 марта)
    spring_season = date(2026, 2, 20) <= target_end and date(2026, 3, 8) >= target_start

    out = out.with_columns(
        pl.lit(1.0 if ny_season else 0.0).alias("target_covers_new_year"),
        pl.lit(1.0 if spring_season else 0.0).alias("target_covers_spring_holidays")
    )

    feat_cols = [c for c in out.columns if c not in ("anchor_date", "user_id")]
    out = out.with_columns(*[pl.col(c).fill_null(0.0) for c in feat_cols])

    return out.select(["anchor_date", "user_id"] + feat_cols)


def main() -> None:
    t0 = time.time()
    data = load_data()

    user_ids = data["user_id"].unique().sort()
    n_users = len(user_ids)
    n_batches = (n_users + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"Start generating exp17 features for {n_users:,} users...")

    for fold_name, anchor in zip(FOLD_NAMES, ANCHORS):
        fold_dir = OUT_DIR / fold_name
        fold_dir.mkdir(parents=True, exist_ok=True)
        tf = time.time()

        for b in range(n_batches):
            tb = time.time()
            ids_b = user_ids[b * BATCH_SIZE:(b + 1) * BATCH_SIZE].to_list()
            df_b = data.filter(pl.col("user_id").is_in(ids_b))

            out = build_batch(df_b, anchor)
            out.write_parquet(fold_dir / f"batch_{b:04d}.parquet")

            print(f"  {fold_name} batch {b + 1}/{n_batches}: {out.height} rows ({time.time() - tb:.1f}s)", flush=True)

        print(f"{fold_name} done in {time.time() - tf:.1f}s\n", flush=True)

    print(f"ALL DONE in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()