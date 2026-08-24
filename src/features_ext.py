"""Extended feature construction (exp02-lite): exp01 base + intent/EWMA/trend/
frequency features from the exp02 card in EXPERIMENTS.md.

For every time-CV anchor and the production anchor writes parquet batches to
data/v2/features_e2/<fold_name>/batch_NNNN.parquet:

- base block identical to archive/exp01/src/features.py (nulls -> 0);
- x_* extended block:
    * intent: 14d sums (searches/cart/ord/gmv), cart-without-order days,
      visit-only days 30d, search-active days 14d;
    * source split: gmv_search/gmv_cat 30d + shares, search_to_ord/cat_to_ord/
      cat_to_cart 30d counts + conversion ratios;
    * dynamics from dense 56d panel of log1p daily values (numpy):
      EWMA halflife {7,30} for gmv/to_ord, OLS slope of log1p(gmv),
      momentum = ewma_h7 - ewma_h30;
    * trends/frequency: gmv_7/gmv_30, gmv_30/gmv_90 ratios, AOV_30,
      has_order_30d, due_ratio (recency vs mean inter-order pause);
- x_* ratios may stay NaN (CatBoost handles natively).

Run from repo root:  .venv/bin/python src/features_ext.py
"""

import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

DATA_PATH = Path("data/train.parquet")
OUT_DIR = Path("data/v2/features_e2")

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
WINDOWS = [(7, "7d"), (14, "14d"), (30, "30d"), (60, "60d"), (90, "90d")]

RECENCY_NONE = 999

PANEL_DAYS = 56


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


def base_exprs(anchor: date) -> tuple[list[pl.Expr], list[pl.Expr]]:
    exprs: list[pl.Expr] = []
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
        .then(1).otherwise(0).sum().cast(pl.Float64).alias("active_days_30d")
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
        .cast(pl.Float64).alias("tenure_days"),
        pl.when(pl.col("to_ord") > 0).then(1).otherwise(0).sum()
        .cast(pl.Float64).alias("x_order_days_total"),
        pl.len().cast(pl.Float64).alias("x_row_days_total"),
    ]

    s90 = pl.col("searches_sum_90d")
    tc = pl.col("to_cart_sum_90d")
    to = pl.col("to_ord_sum_90d")
    g9 = pl.col("gmv_sum_90d")
    conv_exprs = [
        (to / s90.clip(lower_bound=1)).alias("conv_to_ord_per_search_90d"),
        (tc / s90.clip(lower_bound=1)).alias("conv_to_cart_per_search_90d"),
        (to / tc.clip(lower_bound=1)).alias("conv_to_ord_per_cart_90d"),
        (g9 / to.clip(lower_bound=1)).alias("gmv_per_order_90d"),
    ]
    return exprs, conv_exprs


def ext_exprs(anchor: date) -> list[pl.Expr]:
    def m(days: int) -> pl.Expr:
        return pl.col("event_date").is_between(
            anchor - timedelta(days=days - 1), anchor
        )

    def wsum(days: int, col: str, name: str) -> pl.Expr:
        return pl.when(m(days)).then(pl.col(col)).otherwise(0.0).sum().alias(name)

    return [
        # intent signals
        wsum(14, "searches", "x_searches_sum_14d"),
        wsum(14, "to_cart", "x_cart_sum_14d"),
        wsum(14, "to_ord", "x_ord_sum_14d"),
        wsum(14, "gmv", "x_gmv_sum_14d"),
        pl.when(m(14) & (pl.col("to_cart") > 0) & (pl.col("to_ord") == 0))
        .then(1).otherwise(0).sum().cast(pl.Float64).alias("x_cart_no_ord_days_14d"),
        pl.when(m(30) & (pl.col("searches") == 0) & (pl.col("to_cart") == 0)
                & (pl.col("to_ord") == 0) & (pl.col("gmv") == 0))
        .then(1).otherwise(0).sum().cast(pl.Float64).alias("x_visit_only_days_30d"),
        pl.when(m(14) & (pl.col("searches") > 0))
        .then(1).otherwise(0).sum().cast(pl.Float64).alias("x_search_days_14d"),
        # source split
        wsum(30, "gmv_search", "x_gmv_search_sum_30d"),
        wsum(30, "gmv_cat", "x_gmv_cat_sum_30d"),
        wsum(30, "search_to_ord", "x_search_to_ord_30d"),
        wsum(30, "cat_to_ord", "x_cat_to_ord_30d"),
        wsum(30, "cat_to_cart", "x_cat_to_cart_30d"),
        wsum(30, "searches", "x_searches_sum_30d"),
        # frequency
        pl.when(pl.col("to_ord") > 0).then(pl.col("to_ord")).otherwise(None).sum()
        .cast(pl.Float64).alias("x_orders_total"),
    ]


def ext_ratio_exprs() -> list[pl.Expr]:
    eps = 1e-9
    def ratio(num: str, den: str, name: str) -> pl.Expr:
        return (
            pl.when(pl.col(den) > eps).then(pl.col(num) / pl.col(den))
            .otherwise(None).alias(name)
        )
    return [
        ratio("x_gmv_search_sum_30d", "gmv_sum_30d", "x_share_gmv_search_30d"),
        ratio("x_gmv_cat_sum_30d", "gmv_sum_30d", "x_share_gmv_cat_30d"),
        ratio("x_search_to_ord_30d", "x_searches_sum_30d", "x_conv_search_to_ord_30d"),
        ratio("x_cat_to_ord_30d", "x_cat_to_cart_30d", "x_conv_cat_to_ord_30d"),
        ratio("x_search_to_ord_30d", "x_cat_to_ord_30d", "x_search_vs_cat_ord"),
        ratio("gmv_sum_7d", "gmv_sum_30d", "x_trend_gmv_7v30"),
        ratio("gmv_sum_30d", "gmv_sum_90d", "x_trend_gmv_30v90"),
        ratio("to_ord_sum_7d", "to_ord_sum_30d", "x_trend_ord_7v30"),
        ratio("x_gmv_sum_14d", "gmv_sum_30d", "x_gmv_share_14_of_30"),
        pl.when(pl.col("to_ord_sum_30d") > 0)
        .then(pl.col("gmv_sum_30d") / pl.col("to_ord_sum_30d"))
        .otherwise(None).alias("x_aov_30d"),
        pl.when(pl.col("to_ord_sum_30d") > 0).then(1.0).otherwise(0.0)
        .cast(pl.Float64).alias("x_has_ord_30d"),
        (
            pl.when(pl.col("recency_to_ord_days") < RECENCY_NONE)
            .then(
                pl.col("recency_to_ord_days").clip(upper_bound=365)
                / ((pl.col("tenure_days") + 1.0) / (pl.col("x_order_days_total") + 1.0))
                .clip(lower_bound=1.0)
            )
            .otherwise(None)
            .alias("x_due_ratio")
        ),
    ]


def dense_panel(df_b: pl.DataFrame, anchor: date, ids: list[int],
                channels: list[str]) -> np.ndarray:
    """[n_users, PANEL_DAYS * len(channels)] of log1p daily values."""
    w_start = anchor - timedelta(days=PANEL_DAYS - 1)
    win = df_b.filter(pl.col("event_date").is_between(w_start, anchor)).with_columns(
        (anchor - pl.col("event_date")).dt.total_days().cast(pl.Int32).alias("_off")
    )
    day_cols = [f"d{i:02d}" for i in range(PANEL_DAYS)]
    grid = pl.DataFrame({"user_id": ids}).lazy()
    mats = []
    for ch in channels:
        piv = (
            win.pivot(on="_off", index="user_id", values=ch, aggregate_function="sum")
            .rename({str(o): day_cols[o] for o in range(PANEL_DAYS)})
        )
        piv = (
            grid.join(piv.lazy(), on="user_id", how="left")
            .with_columns([pl.col(c).fill_null(0.0) for c in day_cols])
            .select(["user_id", *day_cols])
            .collect()
        )
        mats.append(piv.drop("user_id").to_numpy().astype(np.float64))
    return np.log1p(np.concatenate(mats, axis=1))


def panel_derived(m_log: np.ndarray) -> dict[str, np.ndarray]:
    age = np.arange(PANEL_DAYS - 1, -1, -1, dtype=np.float64)
    t = age - age.mean()
    denom_sq = float((t ** 2).sum())
    out: dict[str, np.ndarray] = {}
    gmv = m_log[:, :PANEL_DAYS]
    ords = m_log[:, PANEL_DAYS:]
    for name, mat in (("gmv", gmv), ("ord", ords)):
        for h in (7.0, 30.0):
            w = np.power(0.5, age / h)
            out[f"x_{name}_ewma_h{int(h)}"] = mat @ w / w.sum()
        out[f"x_{name}_slope_56d"] = (mat @ t) / denom_sq
    out["x_momentum_gmv"] = out["x_gmv_ewma_h7"] - out["x_gmv_ewma_h30"]
    out["x_momentum_ord"] = out["x_ord_ewma_h7"] - out["x_ord_ewma_h30"]
    return out


def build_fold_table(df_b: pl.DataFrame, anchor: date, ids: list[int],
                     with_target: bool) -> pl.DataFrame:
    hist = df_b.filter(pl.col("event_date") <= anchor)
    bexprs, cexprs = base_exprs(anchor)
    feats = (
        hist.group_by("user_id")
        .agg([*bexprs, *ext_exprs(anchor)])
        .with_columns(cexprs)
        .with_columns(ext_ratio_exprs())
    )

    m_log = dense_panel(df_b, anchor, ids, ["gmv", "to_ord"])
    panel_df = pl.DataFrame(
        {"user_id": ids, **{k: v for k, v in panel_derived(m_log).items()}}
    )

    idx = pl.DataFrame({"user_id": ids})
    out = idx.join(feats, on="user_id", how="left").join(panel_df, on="user_id", how="left")

    if with_target:
        t = (
            df_b.filter(
                pl.col("event_date").is_between(
                    anchor + timedelta(days=1), anchor + timedelta(days=HORIZON_DAYS)
                )
            )
            .group_by("user_id")
            .agg(pl.col("gmv").sum().alias("target"))
        )
        out = out.join(t, on="user_id", how="left").with_columns(
            pl.col("target").fill_null(0.0)
        )
    else:
        out = out.with_columns(pl.lit(None, dtype=pl.Float64).alias("target"))

    keep = [c for c in out.columns if c != "user_id"]
    out = out.with_columns(*[pl.col(c).cast(pl.Float64) for c in keep if c != "target"])
    base_fill = [c for c in keep if c != "target" and not c.startswith(("x_",))]
    out = out.with_columns(*[pl.col(c).fill_null(0.0) for c in base_fill])
    return out.select(["user_id", *keep])


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
            out = build_fold_table(df_b, anchor, ids_b, with_target=with_target)
            out.write_parquet(fold_dir / f"batch_{b:04d}.parquet")
            print(f"  {fold_name} batch {b + 1}/{n_batches}: {out.height} rows x "
                  f"{out.width} cols ({time.time() - tb:.1f}s)", flush=True)
        print(f"{fold_name} done (anchor {anchor}) in {time.time() - tf:.1f}s\n", flush=True)

    print(f"ALL DONE in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
