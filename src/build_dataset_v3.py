"""Сборка обучающего датасета v3 по утверждённой схеме (2026-08-24).

Вход:  data/v2/features_ext/fold_*/batch_*.parquet + data/v2/features_bgnbd/<fold>.parquet
Выход: data/v3/fold_{00..03,end}.parquet (один файл на фолд, 88 колонок)

Решения схемы (голосование раунды 1-3):
- A: дубликаты x_{gmv,searches,to_ord,cart}_sum_14d ≡ base -> выкинуты x_*-копии
- B: слабые (corr_z<=0.03 или 43% NaN): slopes, shares, aov, gg_e_value, conv_c2o -> долой
- C: BTYD-входы T/tx/n_occasions/mon_freq/mbar дублируют base или слабые -> долой;
    носитель частоты eb_lambda_n30 вместо order_days_total (corr .983)
- D: лестница окон: ступень 60d выкинута целиком (зажата 30d/90d, corr .92-.98);
    max оставлен только на 30d/90d (max==sum у 91% юзеров на 7d)
- E: EWMA только gmv-ветка (чек плоский, corr веток .93-.97); x_search_to_ord_30d и
    x_gmv_share_14_of_30 выкинуты (дубли заказов/gmv_sum_14d>0)
- F: PCA урезан до 16 компонент (сигнал в 00-01, хвост 17..31 пустой)
- G: recency-сентинел 999 -> пара (флаг «был до якоря» + кап 90д)
- H: джойн BTYD inner (в fold_00 нет строк у 2071 юзеров — потеря 0.8% трейна)

Run:  .venv/bin/python src/build_dataset_v3.py [--smoke]
"""

import argparse
import json
import time
from pathlib import Path

import polars as pl

EXT_DIR = Path("data/v2/features_ext")
BTYD_DIR = Path("data/v2/features_bgnbd")
OUT_DIR = Path("data/v3")
REPORT_PATH = Path("reports/dataset_v3_report.json")

FOLDS = ["fold_00", "fold_01", "fold_02", "fold_03", "fold_end"]
SENTINEL = 999
CAP_DAYS = 90

METRICS = ["gmv", "searches", "to_ord", "to_cart"]

WINDOW_COLS = (
    [f"{m}_{s}_{w}" for m in METRICS for w in ("7d", "14d", "30d", "90d") for s in ("sum", "mean")]
    + [f"{m}_max_{w}" for m in METRICS for w in ("30d", "90d")]
)

PROFILE_COLS = [
    "active_days_30d",
    "tenure_days",
    "row_days_total",
    "has_ord_before_anchor",
    "recency_to_ord_capped90",
    "has_search_before_anchor",
    "recency_searches_capped90",
    "conv_to_ord_per_search_90d",
    "conv_to_cart_per_search_90d",
    "conv_to_ord_per_cart_90d",
    "gmv_per_order_90d",
]

X_COLS = [
    "x_cart_no_ord_days_14d",
    "x_visit_only_days_30d",
    "x_search_days_14d",
    "x_intent_no_ord_14d",
    "x_gmv_search_sum_30d",
    "x_gmv_cat_sum_30d",
    "x_cat_to_ord_30d",
    "x_cat_to_cart_30d",
    "x_conv_s2o_14d",
    "x_due_ratio",
    "x_gmv_ewma_h7",
    "x_gmv_ewma_h30",
    "x_ewma_momentum_gmv",
]

PCA_COLS = [f"pca_{i:02d}" for i in range(16)]

BTYD_COLS = ["bgnbd_p_alive", "bgnbd_en30", "eb_lambda_n30", "bgnbd_e_gmv30", "eb_e_gmv30"]

FEATURE_COLS = WINDOW_COLS + PROFILE_COLS + X_COLS + PCA_COLS + BTYD_COLS
SCHEMA_COLS = ["anchor_date", "user_id"] + FEATURE_COLS + ["target"]
N_FEATURES = len(FEATURE_COLS)


def load_ext(fold: str) -> pl.DataFrame:
    df = pl.read_parquet(EXT_DIR / fold / "batch_*.parquet")
    return df


def transform_ext(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        (pl.col("recency_to_ord_days") != SENTINEL).cast(pl.Int8).alias("has_ord_before_anchor"),
        pl.when(pl.col("recency_to_ord_days") == SENTINEL)
        .then(float(CAP_DAYS))
        .otherwise(pl.col("recency_to_ord_days").clip(0, CAP_DAYS))
        .alias("recency_to_ord_capped90"),
        (pl.col("recency_searches_days") != SENTINEL).cast(pl.Int8).alias("has_search_before_anchor"),
        pl.when(pl.col("recency_searches_days") == SENTINEL)
        .then(float(CAP_DAYS))
        .otherwise(pl.col("recency_searches_days").clip(0, CAP_DAYS))
        .alias("recency_searches_capped90"),
    )
    return df.select(
        ["anchor_date", "user_id"] + WINDOW_COLS + PROFILE_COLS + X_COLS + PCA_COLS
        + [c for c in df.columns if c == "target"]
    )


def build_fold(fold: str) -> dict:
    t0 = time.time()
    ext = transform_ext(load_ext(fold))
    n_ext = ext.height
    btyd = pl.read_parquet(BTYD_DIR / f"{fold}.parquet").select(
        ["anchor_date", "user_id"] + BTYD_COLS
    )
    df = ext.join(btyd, on=["anchor_date", "user_id"], how="inner")
    df = df.select(SCHEMA_COLS)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{fold}.parquet"
    df.write_parquet(out_path)

    feature_nulls = int(df.select(pl.sum_horizontal(pl.col(FEATURE_COLS).null_count())).item())
    dup_keys = df.group_by(["anchor_date", "user_id"]).len().filter(pl.col("len") > 1).height
    stats = {}
    if "target" in df.columns and df["target"].null_count() < df.height:
        z = pl.col("target").log1p()
        agg = df.select(
            pl.col("target").null_count().alias("target_nulls"),
            (z > 0).mean().alias("share_buyers"),
            z.mean().alias("z_mean"),
            z.std().alias("z_std"),
        ).row(0, named=True)
        stats = agg

    return {
        "fold": fold,
        "rows_ext": n_ext,
        "rows_out": df.height,
        "rows_dropped_inner": n_ext - df.height,
        "cols": df.width,
        "feature_nulls": feature_nulls,
        "target_nulls": int(df["target"].null_count()),
        "duplicate_keys": dup_keys,
        "target": stats,
        "seconds": round(time.time() - t0, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="только batch_0000 каждого фолда")
    args = ap.parse_args()

    global EXT_DIR
    results = []
    for fold in FOLDS:
        if args.smoke:
            src = sorted((EXT_DIR / fold).glob("batch_*.parquet"))[:1]
            tmp = pl.concat([pl.read_parquet(p) for p in src])
            ext_raw = tmp
            btyd = pl.read_parquet(BTYD_DIR / f"{fold}.parquet")
            ext_t = transform_ext(ext_raw)
            df = (
                ext_t.join(btyd.select(["anchor_date", "user_id"] + BTYD_COLS),
                           on=["anchor_date", "user_id"], how="inner")
                .select([c for c in SCHEMA_COLS if c in ext_t.columns or c in BTYD_COLS])
                .select(SCHEMA_COLS[: -1] if "target" not in ext_t.columns else SCHEMA_COLS)
            )
            info = {
                "fold": fold, "smoke": True, "rows_out": df.height,
                "cols": df.width, "nulls_total": int(df.select(pl.sum_horizontal(pl.all().null_count())).item()),
            }
        else:
            info = build_fold(fold)
        results.append(info)
        print(info)

    expected = len(SCHEMA_COLS)
    ok_schema = all(r.get("cols") == expected for r in results)
    ok_nulls = all(r.get("feature_nulls", 0) == 0 for r in results)
    ok_dup = all(r.get("duplicate_keys", 0) == 0 for r in results)
    print(f"\nfeatures={N_FEATURES} cols={expected} (ожидалось {expected})")
    print(f"schema_ok={ok_schema} nulls_ok={ok_nulls} keys_ok={ok_dup}")
    for r in results:
        if r.get("target"):
            print(
                f"{r['fold']}: rows={r['rows_out']} dropped_inner={r['rows_dropped_inner']} "
                f"P(buy)={r['target']['share_buyers']:.3f} "
                f"z_mean={r['target']['z_mean']:.3f} z_std={r['target']['z_std']:.3f}"
            )

    if not args.smoke:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps({
            "n_features": N_FEATURES,
            "columns": SCHEMA_COLS,
            "folds": results,
            "checks": {"schema_ok": ok_schema, "nulls_ok": ok_nulls, "keys_ok": ok_dup},
        }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
