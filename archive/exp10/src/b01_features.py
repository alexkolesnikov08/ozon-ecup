"""b01 part A: features for the pseudo-anchor 2025-02-13.

The pseudo-anchor is the exact calendar twin of the submission anchor
(target window [2025-02-14..2025-03-15] mirrors [2026-02-14..2025-03-15] shifted
by one year; exp02 used 2025-02-14, we shift one day back per the b01 spec).

Computes the accepted 66-feature set = 56 base features of exp01 (verbatim port
of their build_batch, as done in archive/exp02/src/exp02_features.py for
fold_proxy_base) + the 10 accepted exp02 extras (conv/decomp/due/shares blocks).
Only history <= anchor is used (leakage-checked on a user sample).

Output cache: data/v2/b01_pseudo/pseudo_anchor/batch_NNNN.parquet
columns: anchor_date, user_id, <56 base feats>, <10 extra feats>, target

Idempotent: existing batches are skipped.
"""

import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

DATA_PATH = Path("data/train.parquet")
OUT_DIR = Path("data/v2/b01_pseudo/pseudo_anchor")

BATCH_SIZE = 50_000
HORIZON_DAYS = 30
PSEUDO_ANCHOR = date(2025, 2, 13)

BASE_VALUE_COLS = ["gmv", "searches", "to_ord", "to_cart"]
BASE_WINDOWS = [(7, "7d"), (30, "30d"), (60, "60d"), (90, "90d")]
RECENCY_NONE = 999

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

EXTRA_ACCEPTED = [
    "conv_s2o", "conv_c2o", "conv_o2c",
    "aov_30", "ord_days_30",
    "due_ratio",
    "share_gmv_search_90", "share_gmv_cat_90",
    "share_gmv_search_trend", "share_gmv_cat_trend",
]


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


def base_feature_exprs(anchor: date) -> tuple[list[pl.Expr], list[pl.Expr]]:
    """Verbatim port of exp01 feature_exprs via exp02_features.py."""
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


def extra_agg_and_derived(anchor: date) -> tuple[list[pl.Expr], list[pl.Expr]]:
    """Port of exp02 new_feature_exprs (all 17 computed; caller selects 10)."""
    masks = {
        tag: pl.col("event_date").is_between(anchor - timedelta(days=d - 1), anchor)
        for tag, d in WINDOWS
    }
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


def build_batch(df_users: pl.DataFrame, anchor: date, with_target: bool,
                base_cols: list[str]) -> pl.DataFrame:
    """56 base feats (exp01 semantics) joined with the 10 accepted extras."""
    hist = df_users.filter(pl.col("event_date") <= anchor)

    base_agg, base_conv = base_feature_exprs(anchor)
    base_feats = hist.group_by("user_id").agg(base_agg).with_columns(base_conv)

    extra_agg, extra_derived = extra_agg_and_derived(anchor)
    extra_feats = hist.group_by("user_id").agg(extra_agg).with_columns(extra_derived)
    extra_feats = extra_feats.join(due_ratio_frame(hist, anchor), on="user_id", how="left")

    out = base_feats.join(
        extra_feats.select(["user_id", *EXTRA_ACCEPTED]), on="user_id", how="inner"
    )
    idx = df_users.select("user_id").unique().with_columns(pl.lit(anchor).alias("anchor_date"))
    out = idx.join(out, on="user_id", how="left")

    feat_cols = [c for c in out.columns if c not in ("anchor_date", "user_id")]
    fill_zero = [c for c in feat_cols if c not in ALLOWED_NAN]
    out = out.with_columns(*[pl.col(c).fill_null(0.0) for c in fill_zero])

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
            pl.col("target").fill_null(0.0).cast(pl.Float64)
        )
    else:
        out = out.with_columns(pl.lit(None, dtype=pl.Float64).alias("target"))

    return out.select(["anchor_date", "user_id", *base_cols, *EXTRA_ACCEPTED, "target"])


def validate_batch(out: pl.DataFrame, expected_rows: int, base_cols: list[str]) -> None:
    assert out.height == expected_rows, f"rows {out.height} != {expected_rows}"
    assert out.schema["anchor_date"] == pl.Date and out.schema["user_id"] == pl.Int64
    for c in base_cols + EXTRA_ACCEPTED:
        assert out.schema[c] == pl.Float64, f"{c}: {out.schema[c]}"
    cols = base_cols + EXTRA_ACCEPTED
    inf_counts = out.select([pl.col(c).is_infinite().sum().alias(c) for c in cols]).row(0)
    assert sum(inf_counts) == 0, f"inf found: {dict(zip(cols, inf_counts))}"


def leakage_check(data: pl.DataFrame, base_cols: list[str]) -> None:
    users = sorted(data["user_id"].unique().to_list())
    ids = [users[0], users[len(users) // 2], users[-1]]
    sub_full = data.filter(pl.col("user_id").is_in(ids))
    f_full = build_batch(sub_full, PSEUDO_ANCHOR, with_target=True, base_cols=base_cols)
    f_trunc = build_batch(
        sub_full.filter(pl.col("event_date") <= PSEUDO_ANCHOR),
        PSEUDO_ANCHOR, with_target=False, base_cols=base_cols,
    )
    j = f_full.drop("target").join(
        f_trunc.drop("anchor_date", "target"), on=["user_id"], suffix="_tr"
    )
    for c in base_cols + EXTRA_ACCEPTED:
        x = np.asarray(j[c].fill_null(float("nan")).to_numpy(), dtype=float)
        y = np.asarray(j[f"{c}_tr"].fill_null(float("nan")).to_numpy(), dtype=float)
        same = np.isclose(x, y, rtol=1e-9, atol=1e-12, equal_nan=True)
        assert same.all(), f"LEAKAGE: {c} changed with post-anchor rows removed"
    print(f"leakage check OK: users={ids}, anchor={PSEUDO_ANCHOR}", flush=True)


def main() -> None:
    t0 = time.time()
    data = load_data()

    ref = pl.read_parquet("data/v2/features/fold_00/batch_0000.parquet")
    BASE_COLS = [c for c in ref.columns if c not in ("anchor_date", "user_id", "target")]
    assert len(BASE_COLS) == 56, len(BASE_COLS)

    leakage_check(data, BASE_COLS)

    user_ids = data["user_id"].unique().sort()
    n_batches = (len(user_ids) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"{len(user_ids):,} users -> {n_batches} batches, anchor {PSEUDO_ANCHOR}",
          flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for b in range(n_batches):
        out_path = OUT_DIR / f"batch_{b:04d}.parquet"
        if out_path.exists():
            print(f"  batch {b + 1}/{n_batches}: exists, skip", flush=True)
            continue
        tb = time.time()
        ids_b = user_ids[b * BATCH_SIZE:(b + 1) * BATCH_SIZE].to_list()
        df_b = data.filter(pl.col("user_id").is_in(ids_b))
        out = build_batch(df_b, PSEUDO_ANCHOR, with_target=True, base_cols=BASE_COLS)
        validate_batch(out, expected_rows=len(ids_b), base_cols=BASE_COLS)
        out.write_parquet(out_path)
        print(f"  batch {b + 1}/{n_batches}: {out.height} rows ({time.time() - tb:.1f}s)",
              flush=True)

    full = pl.read_parquet(str(OUT_DIR / "batch_*.parquet"))
    validate_batch(full, expected_rows=len(user_ids), base_cols=BASE_COLS)
    nulls = {c: int(full[c].is_null().sum()) for c in EXTRA_ACCEPTED}
    bad = {c: v for c, v in nulls.items() if c not in ALLOWED_NAN and v > 0}
    assert not bad, f"unexpected nulls {bad}"
    tgt_nulls = int(full["target"].is_null().sum())
    assert tgt_nulls == 0, f"pseudo target must be fully known, got {tgt_nulls} nulls"
    print(f"validation OK: rows={full.height}, extra nulls={nulls}, target_nulls={tgt_nulls}",
          flush=True)
    print(f"DONE in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
