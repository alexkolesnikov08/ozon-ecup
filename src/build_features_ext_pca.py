# Ported verbatim from teammate repo (github.com/Rafaildavar/ozon-ecup), 2026-08-24,
# for exp02.5 honest block ablation; paths are repo-root relative.
"""Feature construction v2: exp01 base + extended (intent/EWMA/trend/frequency)
+ PCA over dense daily panels (exp02 + exp07).

For every time-CV anchor and the production anchor writes parquet batches to
data/v2/features_ext/<fold_name>/batch_NNNN.parquet with column blocks:

- base (names as in archive/exp01): windowed sums/max/means for
  gmv/searches/to_ord/to_cart over 7/14/30/60/90d, recency/tenure, conv ratios;
- x_*: extended block — intent signals (searches/cart 14d, cart-without-order
  days, visit-only catalog days), source GMV shares, EWMA levels (halflife
  7/30d, log space), EWMA momentum, 56d OLS slope of log1p daily gmv/orders,
  frequency decomposition (order days, AOV, due ratio);
- pca_00..pca_31: IncrementalPCA components of the standardized dense daily
  panel [last 56 days] x [gmv, searches, to_ord, to_cart], log1p-transformed.
  Scaler + PCA are fitted ONLY on pre-anchor windows of train folds
  fold_00..02 (no leakage into fold_03 / fold_end).

Null policy: base block filled with 0 (exp01 parity); ratio-like x_* features
may stay NaN (CatBoost handles natively). Run from repo root:

    .venv/bin/python src/build_features_ext_pca.py            # full run
    .venv/bin/python src/build_features_ext_pca.py --smoke 4000
"""

import argparse
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path("data/train.parquet")
OUT_DIR = Path("data/v2/features_ext")

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
TRAIN_FOLDS = ["fold_00", "fold_01", "fold_02"]

VALUE_COLS = ["gmv", "searches", "to_ord", "to_cart"]
WINDOWS = [(7, "7d"), (14, "14d"), (30, "30d"), (60, "60d"), (90, "90d")]
RECENCY_NONE = 999.0

PANEL_DAYS = 56
PANEL_CHANNELS = ["gmv", "searches", "to_ord", "to_cart"]
N_COMPONENTS = 32
DAY_OFFSETS = [f"d{i:02d}" for i in range(PANEL_DAYS)]

NAN_OK = {
    "conv_to_ord_per_search_90d", "conv_to_cart_per_search_90d",
    "conv_to_ord_per_cart_90d", "gmv_per_order_90d",
    "x_share_gmv_search_30d", "x_share_gmv_cat_30d",
    "x_conv_s2o_30d", "x_conv_c2o_30d", "x_aov_30d", "x_gmv_share_14_of_30",
}


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


def win_mask(anchor: date, days: int) -> pl.Expr:
    return pl.col("event_date").is_between(
        anchor - timedelta(days=days - 1), anchor
    )


def wsum(anchor: date, days: int, col: str, name: str) -> pl.Expr:
    return (
        pl.when(win_mask(anchor, days)).then(pl.col(col)).otherwise(0.0)
        .sum().alias(name)
    )


def base_feature_exprs(anchor: date) -> tuple[list[pl.Expr], list[pl.Expr]]:
    exprs: list[pl.Expr] = []
    for w_days, w_name in WINDOWS:
        mask = win_mask(anchor, w_days)
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

    m30 = win_mask(anchor, 30)
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
        pl.when(m30 & ((pl.col("gmv") > 0) | (pl.col("searches") > 0)))
        .then(1).otherwise(0).sum().cast(pl.Float64).alias("active_days_30d"),
        rec_ord.cast(pl.Float64).alias("recency_to_ord_days"),
        rec_srch.cast(pl.Float64).alias("recency_searches_days"),
        (anchor - pl.col("event_date").min()).dt.total_days()
        .cast(pl.Float64).alias("tenure_days"),
        pl.when(pl.col("to_ord") > 0).then(1).otherwise(0).sum()
        .cast(pl.Float64).alias("order_days_total"),
        pl.len().cast(pl.Float64).alias("row_days_total"),
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


def ext_agg_exprs(anchor: date) -> list[pl.Expr]:
    m14 = win_mask(anchor, 14)
    return [
        wsum(anchor, 14, "searches", "x_searches_sum_14d"),
        wsum(anchor, 14, "to_cart", "x_cart_sum_14d"),
        wsum(anchor, 14, "to_ord", "x_ord_sum_14d"),
        wsum(anchor, 14, "gmv", "x_gmv_sum_14d"),
        wsum(anchor, 30, "gmv_search", "x_gmv_search_sum_30d"),
        wsum(anchor, 30, "gmv_cat", "x_gmv_cat_sum_30d"),
        wsum(anchor, 30, "search_to_ord", "x_search_to_ord_30d"),
        wsum(anchor, 30, "cat_to_ord", "x_cat_to_ord_30d"),
        wsum(anchor, 30, "cat_to_cart", "x_cat_to_cart_30d"),
        (
            pl.when(m14 & (pl.col("to_cart") > 0) & (pl.col("to_ord") == 0))
            .then(1).otherwise(0).sum().cast(pl.Float64).alias("x_cart_no_ord_days_14d")
        ),
        (
            pl.when(
                win_mask(anchor, 30) & (pl.col("searches") == 0)
                & (pl.col("to_cart") == 0) & (pl.col("to_ord") == 0)
                & (pl.col("gmv") == 0)
            ).then(1).otherwise(0).sum().cast(pl.Float64).alias("x_visit_only_days_30d")
        ),
        (
            pl.when(m14 & (pl.col("searches") > 0)).then(1).otherwise(0).sum()
            .cast(pl.Float64).alias("x_search_days_14d")
        ),
    ]


def ext_post_exprs() -> list[pl.Expr]:
    def ratio(num: str, den: str, name: str) -> pl.Expr:
        return (
            pl.when(pl.col(den) > 0).then(pl.col(num) / pl.col(den))
            .otherwise(None).alias(name)
        )

    return [
        ratio("x_gmv_search_sum_30d", "gmv_sum_30d", "x_share_gmv_search_30d"),
        ratio("x_gmv_cat_sum_30d", "gmv_sum_30d", "x_share_gmv_cat_30d"),
        ratio("x_search_to_ord_30d", "searches_sum_14d", "x_conv_s2o_14d"),
        ratio("x_cat_to_ord_30d", "x_cat_to_cart_30d", "x_conv_c2o_30d"),
        ratio("x_gmv_sum_14d", "gmv_sum_30d", "x_gmv_share_14_of_30"),
        (
            pl.when(pl.col("x_ord_sum_14d") == 0)
            .then(1.0).otherwise(0.0).alias("x_intent_no_ord_14d")
        ),
        (
            pl.when(pl.col("to_ord_sum_30d") > 0)
            .then(pl.col("gmv_sum_30d") / pl.col("to_ord_sum_30d"))
            .otherwise(None).alias("x_aov_30d")
        ),
        (
            (
                pl.col("recency_to_ord_days").clip(upper_bound=365)
                / (
                    (pl.col("tenure_days") + 1.0)
                    / (pl.col("order_days_total") + 1.0)
                ).clip(lower_bound=1.0)
            ).clip(upper_bound=60.0).alias("x_due_ratio")
        ),
    ]


def _ols_slope(age: np.ndarray, mat: np.ndarray) -> np.ndarray:
    t = age - age.mean()
    return (mat @ t) / float((t ** 2).sum())


def panel_derived(m_log: np.ndarray) -> dict[str, np.ndarray]:
    """EWMA levels / momentum / OLS slopes from the dense log-panel."""
    age = np.arange(PANEL_DAYS - 1, -1, -1, dtype=np.float64)
    out: dict[str, np.ndarray] = {}
    gmv = m_log[:, :PANEL_DAYS]
    ords = m_log[:, 2 * PANEL_DAYS:3 * PANEL_DAYS]
    for name, mat in (("gmv", gmv), ("ord", ords)):
        for h in (7.0, 30.0):
            w = np.power(0.5, age / h)
            out[f"x_{name}_ewma_h{int(h)}"] = mat @ w / w.sum()
        out[f"x_{name}_slope_56d"] = _ols_slope(age, mat)
    out["x_ewma_momentum_gmv"] = out["x_gmv_ewma_h7"] - out["x_gmv_ewma_h30"]
    out["x_ewma_momentum_ord"] = out["x_ord_ewma_h7"] - out["x_ord_ewma_h30"]
    return out


def dense_panel(df_b: pl.DataFrame, anchor: date, ids: list[int]) -> np.ndarray:
    """[n_users, PANEL_DAYS * len(PANEL_CHANNELS)] of log1p daily values."""
    w_start = anchor - timedelta(days=PANEL_DAYS - 1)
    win = df_b.filter(pl.col("event_date").is_between(w_start, anchor)).with_columns(
        (anchor - pl.col("event_date")).dt.total_days().cast(pl.Int64).alias("_off")
    )
    grid = pl.DataFrame({"user_id": ids}, schema={"user_id": pl.Int64})
    mats = []
    for ch in PANEL_CHANNELS:
        piv = win.pivot(on="_off", index="user_id", values=ch,
                        aggregate_function="sum")
        piv.columns = ["user_id"] + [f"d{int(c):02d}" for c in piv.columns[1:]]
        piv = (
            grid.join(piv.select(["user_id", *DAY_OFFSETS]), on="user_id", how="left")
            .with_columns([pl.col(c).fill_null(0.0) for c in DAY_OFFSETS])
        )
        mats.append(piv.select(DAY_OFFSETS).to_numpy().astype(np.float64))
    m = np.concatenate(mats, axis=1)
    return np.log1p(m)


def build_fold_table(df_b: pl.DataFrame, anchor: date, ids: list[int],
                     with_target: bool, scaler: StandardScaler,
                     ipca: IncrementalPCA) -> pl.DataFrame:
    hist = df_b.filter(pl.col("event_date") <= anchor)
    base_exprs, conv_exprs = base_feature_exprs(anchor)
    feats = (
        hist.group_by("user_id")
        .agg([*base_exprs, *ext_agg_exprs(anchor)])
        .with_columns(conv_exprs)
        .with_columns(ext_post_exprs())
    )

    m_log = dense_panel(df_b, anchor, ids)
    panel_df = pl.DataFrame({"user_id": ids, **panel_derived(m_log)})
    z = ipca.transform(scaler.transform(m_log))
    z_df = pl.DataFrame({"user_id": ids}).hstack(
        pl.DataFrame(z, schema={f"pca_{i:02d}": pl.Float64 for i in range(z.shape[1])})
    )

    idx = pl.DataFrame({"user_id": ids})
    out = (
        idx.join(feats, on="user_id", how="left")
        .join(panel_df, on="user_id", how="left")
        .join(z_df, on="user_id", how="left")
    )

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

    feat_cols = [c for c in out.columns if c not in ("user_id", "target")]
    fill_cols = [c for c in feat_cols if c not in NAN_OK]
    out = out.with_columns(*[pl.col(c).cast(pl.Float64) for c in feat_cols])
    out = out.with_columns(*[pl.col(c).fill_null(0.0) for c in fill_cols])
    return out.select([pl.lit(anchor).alias("anchor_date"), "user_id", *feat_cols, "target"])


def iter_user_batches(data: pl.DataFrame, smoke_n: int | None):
    user_ids = data["user_id"].unique().sort()
    if smoke_n:
        user_ids = user_ids.head(smoke_n)
    n_users = len(user_ids)
    bs = min(BATCH_SIZE, n_users)
    n_batches = (n_users + bs - 1) // bs
    for b in range(n_batches):
        yield b, n_batches, user_ids[b * bs:(b + 1) * bs].to_list()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=None, help="limit to first N users")
    args = ap.parse_args()

    t0 = time.time()
    data = load_data()

    print(f"\nPASS 1: fitting scaler + IncrementalPCA({N_COMPONENTS}) "
          f"on train folds {TRAIN_FOLDS} (pre-anchor windows only)")
    scaler, ipca = StandardScaler(), IncrementalPCA(n_components=N_COMPONENTS)
    for fold_name, anchor in zip(FOLD_NAMES, ANCHORS):
        if fold_name not in TRAIN_FOLDS:
            continue
        tf = time.time()
        for b, n_batches, ids in iter_user_batches(data, args.smoke):
            df_b = data.filter(pl.col("user_id").is_in(ids))
            m_log = dense_panel(df_b, anchor, ids)
            scaler.partial_fit(m_log)
            ipca.partial_fit(scaler.transform(m_log))
        print(f"  {fold_name}: cum explained variance = "
              f"{ipca.explained_variance_ratio_.sum():.3f} ({time.time() - tf:.1f}s)",
              flush=True)

    evr = ipca.explained_variance_ratio_
    print(f"\nPCA total explained variance: {evr.sum():.3f}")
    print("  " + " ".join(f"pca_{i:02d}:{v:.3f}" for i, v in enumerate(evr[:10])))

    print(f"\nPASS 2: building feature tables -> {OUT_DIR}")
    for fold_name, anchor in zip(FOLD_NAMES, ANCHORS):
        fold_dir = OUT_DIR / fold_name
        fold_dir.mkdir(parents=True, exist_ok=True)
        with_target = fold_name != "fold_end"
        tf = time.time()
        for b, n_batches, ids in iter_user_batches(data, args.smoke):
            tb = time.time()
            df_b = data.filter(pl.col("user_id").is_in(ids))
            out = build_fold_table(df_b, anchor, ids, with_target, scaler, ipca)
            out.write_parquet(fold_dir / f"batch_{b:04d}.parquet")
            print(f"  {fold_name} batch {b + 1}/{n_batches}: {out.height} rows x "
                  f"{out.width} cols ({time.time() - tb:.1f}s)", flush=True)
        print(f"{fold_name} done (anchor {anchor}) in {time.time() - tf:.1f}s\n",
              flush=True)

    print(f"ALL DONE in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
