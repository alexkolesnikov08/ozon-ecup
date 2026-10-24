"""Build base time-series features directly from ``data/train.parquet``.

The script does not depend on data/v2 or data/v3.  For every time-CV anchor it
creates leakage-safe user features, a 26-week GMV sequence and the classical
intermittent-demand class:

    smooth / erratic / intermittent / lumpy

The classification uses the Syntetos-Boylan thresholds ADI=1.32 and CV^2=0.49
on positive weekly GMV observations.  Missing weeks are real zero-demand
periods and are therefore included in ADI.

Run from the repository root::

    python src/build_segmented_features.py

Outputs ``data/segmented_base/fold_{00..03,end}.parquet`` and ``manifest.json``.
Existing valid fold files are reused unless ``--overwrite`` is passed.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import date, timedelta
from pathlib import Path

import polars as pl


DATA_PATH = Path("data/train.parquet")
SAMPLE_PATH = Path("sample_submit.csv")
OUT_DIR = Path("data/segmented_base")

CV_ANCHORS = [
    date(2025, 12, 3),
    date(2025, 12, 17),
    date(2025, 12, 31),
    date(2026, 1, 14),
]
END_ANCHOR = date(2026, 2, 13)
FOLDS = [f"fold_{i:02d}" for i in range(4)] + ["fold_end"]
ANCHORS = CV_ANCHORS + [END_ANCHOR]

HORIZON_DAYS = 30
N_WEEKS = 26
WEEK_DAYS = 7
ADI_THRESHOLD = 1.32
CV2_THRESHOLD = 0.49
RECENCY_CAP = 365

# All columns below exist in the competition's base parquet.
SUM_METRICS = [
    "gmv",
    "searches",
    "to_ord",
    "to_cart",
    "search",
    "cat",
    "search_to_cart",
    "search_to_ord",
    "cat_to_cart",
    "cat_to_ord",
    "gmv_search",
    "gmv_cat",
]
CORE_METRICS = ["gmv", "searches", "to_ord", "to_cart"]
WINDOWS = [7, 30, 90, 180]
PROFILE_WINDOWS = [30, 90]
WEEKLY_CHANNELS = ["gmv", "to_ord", "searches", "to_cart"]
FEATURE_SCHEMA_VERSION = 2

CLASS_NAMES = {
    0: "smooth",
    1: "erratic",
    2: "intermittent",
    3: "lumpy",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build leakage-safe segmented LTV features from base parquet."
    )
    ap.add_argument("--data", type=Path, default=DATA_PATH)
    ap.add_argument("--sample", type=Path, default=SAMPLE_PATH)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--batch-size", type=int, default=50_000)
    ap.add_argument(
        "--limit-users",
        type=int,
        default=None,
        help="Smoke test only: build the first N users from sample_submit.csv.",
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.batch_size < 1:
        ap.error("--batch-size must be positive")
    if args.limit_users is not None and args.limit_users < 100:
        ap.error("--limit-users must be at least 100")
    return args


def validate_input(data_path: Path, sample_path: Path) -> None:
    if not data_path.exists():
        raise FileNotFoundError(
            f"Missing {data_path}. Put the base competition parquet there or pass --data."
        )
    if not sample_path.exists():
        raise FileNotFoundError(f"Missing {sample_path}")

    schema = pl.read_parquet_schema(data_path)
    required = {"event_date", "user_id", *SUM_METRICS}
    missing = sorted(required - set(schema))
    if missing:
        raise ValueError(f"Base parquet is missing columns: {missing}")

    sample_schema = pl.read_csv(sample_path, n_rows=0).schema
    if "user_id" not in sample_schema:
        raise ValueError(f"{sample_path} must contain user_id")


def load_base(data_path: Path) -> pl.DataFrame:
    df = pl.read_parquet(data_path).with_columns(
        pl.col("event_date").cast(pl.Date),
        pl.col("user_id").cast(pl.Int64),
        *[pl.col(c).cast(pl.Float64) for c in SUM_METRICS],
    )
    if df.select(pl.col("user_id").is_null().any()).item():
        raise ValueError("user_id contains NULL")
    return df


def conditional_sum(col: str, start: date, anchor: date, alias: str) -> pl.Expr:
    mask = pl.col("event_date").is_between(start, anchor)
    return pl.when(mask).then(pl.col(col)).otherwise(0.0).sum().alias(alias)


def aggregate_expressions(anchor: date) -> tuple[list[pl.Expr], list[str]]:
    exprs: list[pl.Expr] = []

    for days in WINDOWS:
        start = anchor - timedelta(days=days - 1)
        for col in SUM_METRICS:
            exprs.append(conditional_sum(col, start, anchor, f"{col}_sum_{days}d"))

    for days in PROFILE_WINDOWS:
        start = anchor - timedelta(days=days - 1)
        mask = pl.col("event_date").is_between(start, anchor)
        for col in CORE_METRICS:
            exprs.extend(
                [
                    pl.when(mask).then(pl.col(col)).otherwise(None).mean()
                    .alias(f"{col}_mean_{days}d"),
                    pl.when(mask).then(pl.col(col)).otherwise(None).max()
                    .alias(f"{col}_max_{days}d"),
                ]
            )
        exprs.extend(
            [
                pl.when(mask).then(1.0).otherwise(0.0).sum().alias(f"row_days_{days}d"),
                pl.when(mask & (pl.col("gmv") > 0)).then(1.0).otherwise(0.0).sum()
                .alias(f"gmv_days_{days}d"),
                pl.when(mask & (pl.col("to_ord") > 0)).then(1.0).otherwise(0.0).sum()
                .alias(f"order_days_{days}d"),
                pl.when(mask & (pl.col("searches") > 0)).then(1.0).otherwise(0.0).sum()
                .alias(f"search_days_{days}d"),
                pl.when(mask & (pl.col("to_cart") > 0)).then(1.0).otherwise(0.0).sum()
                .alias(f"cart_days_{days}d"),
            ]
        )

    for col, name in [
        ("gmv", "gmv"),
        ("to_ord", "order"),
        ("searches", "search"),
        ("to_cart", "cart"),
    ]:
        last_day = pl.col("event_date").filter(pl.col(col) > 0).max()
        exprs.extend(
            [
                pl.when(last_day.is_null())
                .then(float(RECENCY_CAP))
                .otherwise((anchor - last_day).dt.total_days().clip(0, RECENCY_CAP))
                .cast(pl.Float64)
                .alias(f"recency_{name}_days"),
                last_day.is_not_null().cast(pl.Float64).alias(f"has_{name}_history"),
            ]
        )

    exprs.extend(
        [
            (anchor - pl.col("event_date").min()).dt.total_days().clip(0, RECENCY_CAP)
            .cast(pl.Float64)
            .alias("tenure_days"),
            pl.len().cast(pl.Float64).alias("row_days_total"),
            pl.col("gmv").sum().alias("gmv_lifetime"),
            pl.col("to_ord").sum().cast(pl.Float64).alias("orders_lifetime"),
        ]
    )

    # 26 weekly GMV lags are both sequence-model features and the basis for
    # intermittent-demand classification. week_00 is the most recent week.
    weekly_gmv_cols: list[str] = []
    for col in WEEKLY_CHANNELS:
        for week in range(N_WEEKS):
            end = anchor - timedelta(days=week * WEEK_DAYS)
            start = end - timedelta(days=WEEK_DAYS - 1)
            name = f"week_{col}_{week:02d}"
            if col == "gmv":
                weekly_gmv_cols.append(name)
            exprs.append(conditional_sum(col, start, end, name))

    return exprs, weekly_gmv_cols


def add_derived_features(df: pl.DataFrame, weekly_gmv_cols: list[str]) -> pl.DataFrame:
    eps = 1e-6
    n_pos = pl.sum_horizontal(
        [(pl.col(c) > 0).cast(pl.Float64) for c in weekly_gmv_cols]
    )
    pos_sum = pl.sum_horizontal([pl.col(c) for c in weekly_gmv_cols])
    pos_sq_sum = pl.sum_horizontal([pl.col(c).pow(2) for c in weekly_gmv_cols])

    df = df.with_columns(
        n_pos.alias("demand_weeks_26w"),
        (1.0 - n_pos / float(N_WEEKS)).alias("zero_week_share_26w"),
        pl.when(n_pos > 0)
        .then(pos_sum / n_pos)
        .otherwise(0.0)
        .alias("positive_week_gmv_mean_26w"),
        (pos_sum / float(N_WEEKS)).alias("weekly_gmv_mean_26w"),
        (pl.max_horizontal([pl.col(c) for c in weekly_gmv_cols]))
        .alias("weekly_gmv_max_26w"),
    )

    n = pl.col("demand_weeks_26w")
    mean_pos = pl.col("positive_week_gmv_mean_26w")
    sample_var = pl.when(n > 1).then(
        ((pos_sq_sum - pos_sum.pow(2) / n) / (n - 1)).clip(lower_bound=0.0)
    ).otherwise(0.0)

    df = df.with_columns(
        pl.when(n > 0).then(float(N_WEEKS) / n).otherwise(float(N_WEEKS + 1))
        .alias("adi_26w"),
        pl.when((n > 1) & (mean_pos > 0))
        .then(sample_var / (mean_pos.pow(2) + eps))
        .otherwise(0.0)
        .clip(upper_bound=100.0)
        .alias("cv2_demand_26w"),
    )

    adi = pl.col("adi_26w")
    cv2 = pl.col("cv2_demand_26w")
    demand_class = (
        pl.when((adi < ADI_THRESHOLD) & (cv2 < CV2_THRESHOLD)).then(0)
        .when((adi < ADI_THRESHOLD) & (cv2 >= CV2_THRESHOLD)).then(1)
        .when((adi >= ADI_THRESHOLD) & (cv2 < CV2_THRESHOLD)).then(2)
        .otherwise(3)
        .cast(pl.Int8)
    )

    return df.with_columns(
        demand_class.alias("demand_class_id"),
        (pl.col("gmv_sum_7d") / (pl.col("gmv_sum_30d") + 1.0))
        .alias("gmv_recent_share_7_30"),
        (pl.col("gmv_sum_30d") / (pl.col("gmv_sum_90d") + 1.0))
        .alias("gmv_recent_share_30_90"),
        (pl.col("searches_sum_7d") / (pl.col("searches_sum_30d") + 1.0))
        .alias("search_recent_share_7_30"),
        (pl.col("to_ord_sum_90d") / (pl.col("searches_sum_90d") + 1.0))
        .alias("conv_order_per_search_90d"),
        (pl.col("to_cart_sum_90d") / (pl.col("searches_sum_90d") + 1.0))
        .alias("conv_cart_per_search_90d"),
        (pl.col("to_ord_sum_90d") / (pl.col("to_cart_sum_90d") + 1.0))
        .alias("conv_order_per_cart_90d"),
        (pl.col("gmv_sum_90d") / (pl.col("to_ord_sum_90d") + 1.0))
        .alias("gmv_per_order_90d"),
    )


def build_batch(
    base: pl.DataFrame,
    user_ids: pl.Series,
    anchor: date,
    with_target: bool,
) -> pl.DataFrame:
    user_frame = pl.DataFrame({"user_id": user_ids.cast(pl.Int64)})
    user_data = base.filter(pl.col("user_id").is_in(user_ids.implode()))
    history = user_data.filter(pl.col("event_date") <= anchor)

    exprs, weekly_cols = aggregate_expressions(anchor)
    features = history.group_by("user_id").agg(exprs)
    out = user_frame.join(features, on="user_id", how="left").with_columns(
        pl.lit(anchor, dtype=pl.Date).alias("anchor_date")
    )

    feature_cols = [c for c in out.columns if c not in {"user_id", "anchor_date"}]
    out = out.with_columns([pl.col(c).fill_null(0.0) for c in feature_cols])
    out = add_derived_features(out, weekly_cols)

    if with_target:
        target_end = anchor + timedelta(days=HORIZON_DAYS)
        targets = (
            user_data.filter(
                pl.col("event_date").is_between(anchor + timedelta(days=1), target_end)
            )
            .group_by("user_id")
            .agg(pl.col("gmv").sum().alias("target"))
        )
        out = out.join(targets, on="user_id", how="left").with_columns(
            pl.col("target").fill_null(0.0)
        )
    else:
        out = out.with_columns(pl.lit(None, dtype=pl.Float64).alias("target"))

    # Float32 materially lowers disk and RAM requirements during model fitting.
    cast_cols = [
        c
        for c, dtype in out.schema.items()
        if c not in {"user_id", "anchor_date", "demand_class_id"}
        and dtype.is_numeric()
    ]
    out = out.with_columns([pl.col(c).cast(pl.Float32) for c in cast_cols])
    ordered = [
        "anchor_date",
        "user_id",
        *[c for c in out.columns if c not in {"anchor_date", "user_id", "target"}],
        "target",
    ]
    return out.select(ordered)


def fold_is_valid(path: Path, expected_users: int) -> bool:
    if not path.exists():
        return False
    try:
        schema = pl.read_parquet_schema(path)
        required = {
            "anchor_date",
            "user_id",
            "target",
            "demand_class_id",
            *[f"week_{channel}_{N_WEEKS - 1:02d}" for channel in WEEKLY_CHANNELS],
        }
        if not required.issubset(schema):
            return False
        count = pl.scan_parquet(path).select(pl.len()).collect().item()
        return count == expected_users
    except Exception:
        return False


def validate_fold(path: Path, expected_users: int, with_target: bool) -> dict:
    df = pl.read_parquet(path)
    if df.height != expected_users:
        raise AssertionError(f"{path}: expected {expected_users} rows, got {df.height}")
    if df["user_id"].n_unique() != expected_users:
        raise AssertionError(f"{path}: duplicate user_id")

    feature_cols = [c for c in df.columns if c not in {"anchor_date", "user_id", "target"}]
    nulls = int(df.select(pl.sum_horizontal(pl.col(feature_cols).null_count())).item())
    if nulls:
        raise AssertionError(f"{path}: {nulls} feature NULLs")
    if with_target and df["target"].null_count():
        raise AssertionError(f"{path}: target contains NULL")
    if not with_target and df["target"].null_count() != df.height:
        raise AssertionError(f"{path}: fold_end target must be NULL")

    class_counts = {
        CLASS_NAMES[int(row[0])]: int(row[1])
        for row in df.group_by("demand_class_id").len().sort("demand_class_id").iter_rows()
    }
    return {
        "rows": df.height,
        "columns": df.width,
        "features": len(feature_cols),
        "class_counts": class_counts,
        "target_buy_rate": (
            round(float((df["target"] > 0).mean()), 6) if with_target else None
        ),
    }


def main() -> None:
    args = parse_args()
    validate_input(args.data, args.sample)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sample_users = pl.read_csv(args.sample, columns=["user_id"])["user_id"].cast(pl.Int64)
    if sample_users.n_unique() != len(sample_users):
        raise ValueError("sample_submit.csv contains duplicate user_id")
    if args.limit_users is not None:
        sample_users = sample_users.head(args.limit_users)
        smoke_sample_path = args.out_dir / "sample_submit_smoke.csv"
        pl.DataFrame(
            {"user_id": sample_users, "predict": pl.Series([0.0] * len(sample_users))}
        ).write_csv(smoke_sample_path)
        print(f"Smoke-test sample: {smoke_sample_path}")
    expected_users = len(sample_users)

    needed = [
        (fold, anchor)
        for fold, anchor in zip(FOLDS, ANCHORS)
        if args.overwrite or not fold_is_valid(args.out_dir / f"{fold}.parquet", expected_users)
    ]
    if needed:
        print(f"Loading base parquet {args.data} ...", flush=True)
        base = load_base(args.data)
        base = base.filter(pl.col("user_id").is_in(sample_users.implode()))
        print(
            f"Loaded {base.height:,} rows, {base['user_id'].n_unique():,} users, "
            f"dates {base['event_date'].min()}..{base['event_date'].max()}",
            flush=True,
        )
    else:
        base = pl.DataFrame()
        print("All fold files already exist and passed basic checks; reusing them.")

    manifest_folds: dict[str, dict] = {}
    for fold, anchor in zip(FOLDS, ANCHORS):
        path = args.out_dir / f"{fold}.parquet"
        with_target = fold != "fold_end"
        if fold_is_valid(path, expected_users) and not args.overwrite:
            print(f"{fold}: reuse {path}", flush=True)
        else:
            started = time.time()
            parts: list[pl.DataFrame] = []
            for start in range(0, expected_users, args.batch_size):
                ids = sample_users.slice(start, args.batch_size)
                part = build_batch(base, ids, anchor, with_target)
                parts.append(part)
                print(
                    f"  {fold}: users {start + 1:,}..{start + len(ids):,}",
                    flush=True,
                )
            result = pl.concat(parts, how="vertical")
            result.write_parquet(path, compression="zstd", statistics=True)
            print(f"{fold}: saved {path} in {time.time() - started:.1f}s", flush=True)
        manifest_folds[fold] = validate_fold(path, expected_users, with_target)
        print(f"  {manifest_folds[fold]}", flush=True)

    manifest = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "source": str(args.data),
        "sample": str(
            args.out_dir / "sample_submit_smoke.csv"
            if args.limit_users is not None
            else args.sample
        ),
        "n_users": expected_users,
        "anchors": {f: a.isoformat() for f, a in zip(FOLDS, ANCHORS)},
        "target_horizon_days": HORIZON_DAYS,
        "demand_classification": {
            "period": "weekly GMV over the last 26 weeks",
            "adi_threshold": ADI_THRESHOLD,
            "cv2_threshold": CV2_THRESHOLD,
            "classes": CLASS_NAMES,
        },
        "folds": manifest_folds,
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
