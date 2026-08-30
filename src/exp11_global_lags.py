"""exp11 — Глобальная модель на лагах (M5-style pooled) со строгими time-series фолдами.

Одна pooled-модель «юзер × день» на лагах/роллингах дневной активности предсказывает
z = log1p(gmv за [t+1, t+30]) напрямую. Плотная сетка якорей даёт на порядки больше
обучающих окон, чем 4 фолда.

Time-series протокол (walk-forward, rolling origin):
  fold_k: train = все якоря t с t+30 <= anchor_k (таргеты заканчиваются строго до
          якоря — ни фичи, ни таргеты не заходят за якорь), eval = anchor_k.
  prod  : train = все якоря t <= 2026-01-14 (все таргеты внутри данных),
          предикт на fold_end (2026-02-13).

Фичи (нулевозаполненная дневная сетка юзера): лог-лаги gmv 1..28; роллинг-суммы
gmv/to_ord/searches/to_cart 7..90 (включительно до якоря); активные/заказные дни;
recency заказа/поиска; tenure; кумулятивы; тренд-отношения; календарь (DoW/день/месяц).

Артефакты: reports/exp11_global_lags.json, reports/figures/exp11_importance.png,
submissions/submission_exp11.csv.

Запуск из корня репо: .venv/bin/python src/exp11_global_lags.py
"""

from __future__ import annotations

import json
import time
import warnings
from datetime import date, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

DATA_PATH = Path("data/train.parquet")
SAMPLE_PATH = Path("sample_submit.csv")
OUT_PATH = Path("reports/exp11_global_lags.json")
FIG_DIR = Path("reports/figures")
SUB_PATH = Path("submissions/submission_exp11.csv")

H = 30
SEED = 42
BATCH_USERS = 50_000
ANCHOR_STEP = 10
GRID_START = date(2025, 4, 1)

EVAL_ANCHORS = {
    "fold_00": date(2025, 12, 3),
    "fold_01": date(2025, 12, 17),
    "fold_02": date(2025, 12, 31),
    "fold_03": date(2026, 1, 14),
}
PROD_ANCHOR = date(2026, 2, 13)
NAIVE_REF = {
    "fold_00": 2.22630,
    "fold_01": 2.21661,
    "fold_02": 2.22673,
    "fold_03": 2.19506,
}

LGB_PARAMS = dict(
    objective="regression",
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=200,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    n_jobs=-1,
    seed=SEED,
    verbosity=-1,
)
N_TREES = 400


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    zt = np.log1p(np.clip(y_true, 0, None))
    zp = np.log1p(np.clip(y_pred, 0, None))
    return float(np.sqrt(np.mean((zt - zp) ** 2)))


def anchor_grid(last_train_anchor: date) -> list[date]:
    anchors, d = [], GRID_START
    while d <= last_train_anchor:
        anchors.append(d)
        d += timedelta(days=ANCHOR_STEP)
    return sorted(set(anchors) | set(EVAL_ANCHORS.values()))


def build_features_stage1() -> list[pl.Expr]:
    """Лаги, роллинги, активность, recency, tenure, кумулятивы, календарь."""
    e: list[pl.Expr] = []
    u = "user_id"
    for k in (1, 2, 3, 7, 14, 28):
        e.append(
            pl.col("gmv").shift(k).over(u).log1p().alias(f"f_lag_gmv_{k}")
        )
    rolls = {
        "gmv": (7, 14, 30, 60, 90),
        "to_ord": (7, 30, 90),
        "searches": (7, 14, 30),
        "to_cart": (30,),
    }
    for col, ws in rolls.items():
        for w in ws:
            e.append(
                pl.col(col).rolling_sum(w).over(u).alias(f"f_{col}_{w}")
            )
    for w in (7, 30, 90):
        e.append(
            (pl.col("gmv") > 0).cast(pl.UInt8).rolling_sum(w).over(u).alias(f"f_act_{w}")
        )
    for w in (30, 90):
        e.append(
            (pl.col("to_ord") > 0)
            .cast(pl.UInt8)
            .rolling_sum(w)
            .over(u)
            .alias(f"f_ordd_{w}")
        )
    last_ord = (
        pl.when(pl.col("to_ord") > 0)
        .then(pl.col("event_date"))
        .otherwise(None)
        .forward_fill()
        .over(u)
    )
    last_src = (
        pl.when(pl.col("searches") > 0)
        .then(pl.col("event_date"))
        .otherwise(None)
        .forward_fill()
        .over(u)
    )
    e.append(
        (pl.col("event_date") - last_ord)
        .dt.total_days()
        .fill_null(999)
        .clip(0, 999)
        .alias("f_rec_ord")
    )
    e.append(
        (pl.col("event_date") - last_src)
        .dt.total_days()
        .fill_null(999)
        .clip(0, 999)
        .alias("f_rec_src")
    )
    first_act = (
        pl.when(pl.col("was_row"))
        .then(pl.col("event_date"))
        .otherwise(None)
        .min()
        .over(u)
    )
    e.append((pl.col("event_date") - first_act).dt.total_days().alias("f_tenure"))
    e.append(pl.col("gmv").cum_sum().over(u).log1p().alias("f_cum_gmv"))
    e.append(
        (pl.col("to_ord") > 0).cast(pl.UInt8).cum_sum().over(u).alias("f_cum_orddays")
    )
    e.append(pl.col("event_date").dt.weekday().alias("f_dow"))
    e.append(pl.col("event_date").dt.day().alias("f_dom"))
    e.append(pl.col("event_date").dt.month().alias("f_month"))
    return e


def build_features_stage2() -> list[pl.Expr]:
    """Тренд-отношения поверх роллингов stage1."""
    def ratio(num: str, den: str) -> pl.Expr:
        return (
            pl.when(pl.col(den) > 0)
            .then(pl.col(num) / pl.col(den))
            .otherwise(None)
            .alias(f"f_r_{num}_div_{den}")
        )

    return [
        ratio("f_gmv_7", "f_gmv_30"),
        ratio("f_gmv_30", "f_gmv_90"),
        ratio("f_to_ord_7", "f_to_ord_30"),
        ratio("f_to_ord_30", "f_to_ord_90"),
        ratio("f_searches_7", "f_searches_30"),
    ]


FEATURE_COLS = (
    [f"f_lag_gmv_{k}" for k in (1, 2, 3, 7, 14, 28)]
    + [f"f_{c}_{w}" for c, ws in {"gmv": (7, 14, 30, 60, 90), "to_ord": (7, 30, 90), "searches": (7, 14, 30), "to_cart": (30,)}.items() for w in ws]
    + ["f_act_7", "f_act_30", "f_act_90", "f_ordd_30", "f_ordd_90"]
    + ["f_rec_ord", "f_rec_src", "f_tenure", "f_cum_gmv", "f_cum_orddays"]
    + ["f_r_f_gmv_7_div_f_gmv_30", "f_r_f_gmv_30_div_f_gmv_90",
       "f_r_f_to_ord_7_div_f_to_ord_30", "f_r_f_to_ord_30_div_f_to_ord_90",
       "f_r_f_searches_7_div_f_searches_30"]
    + ["f_dow", "f_dom", "f_month"]
)


def build_batch(
    raw_lf: pl.LazyFrame,
    user_ids: list[int],
    cal_lf: pl.LazyFrame,
    anchor_dates: list[date],
) -> pl.DataFrame:
    ub = raw_lf.filter(pl.col("user_id").is_in(user_ids))
    grid = (
        pl.LazyFrame({"user_id": user_ids})
        .join(cal_lf, how="cross")
        .join(ub, on=["user_id", "event_date"], how="left")
        .with_columns(pl.col("gmv").is_not_null().alias("was_row"))
        .with_columns([pl.col(c).fill_null(0.0) for c in ("gmv", "searches", "to_ord", "to_cart")])
        .sort(["user_id", "event_date"])
    )
    p1 = grid.with_columns(build_features_stage1())
    p2 = p1.with_columns(build_features_stage2())
    tgt = (
        p2.sort(["user_id", "event_date"], descending=[False, True])
        .with_columns(pl.col("gmv").rolling_sum(H).shift(1).over("user_id").alias("y_next30"))
        .sort(["user_id", "event_date"])
    )
    return (
        tgt.filter(pl.col("event_date").is_in(anchor_dates))
        .with_columns(pl.col("y_next30").log1p().alias("z"))
        .select(["user_id", "event_date", "y_next30", "z"] + FEATURE_COLS)
        .rename({"event_date": "anchor_date"})
        .collect(streaming=True)
    )


def main() -> None:
    t0 = time.time()
    raw_lf = pl.scan_parquet(DATA_PATH)
    meta = raw_lf.select(
        pl.col("user_id").unique().sort().alias("user_id"),
        pl.col("event_date").min().alias("dmin"),
        pl.col("event_date").max().alias("dmax"),
    ).collect()
    all_users = meta["user_id"].to_list()
    dmin, dmax = meta["dmin"][0], meta["dmax"][0]
    cal = pl.LazyFrame(
        {"event_date": pl.date_range(dmin, dmax, "1d", eager=True)}
    )

    last_train = max(EVAL_ANCHORS.values())
    anchors = anchor_grid(last_train)
    print(f"anchors: {len(anchors)} ({anchors[0]} .. {anchors[-1]}), "
          f"users={len(all_users)}, calendar={dmin}..{dmax}", flush=True)

    parts = []
    for i in range(0, len(all_users), BATCH_USERS):
        chunk = all_users[i : i + BATCH_USERS]
        tb = time.time()
        part = build_batch(raw_lf, chunk, cal, anchors)
        parts.append(part)
        print(f"batch {i//BATCH_USERS}: {part.height} rows in {time.time()-tb:.1f}s", flush=True)
    panel = pl.concat(parts).with_columns(
        [pl.col(c).cast(pl.Float32) for c in FEATURE_COLS]
    ).sort(["anchor_date", "user_id"])
    print(f"panel: {panel.height} rows x {panel.width} cols, "
          f"build {time.time()-t0:.1f}s", flush=True)

    results = {
        "config": {
            "protocol": "time-series walk-forward: train anchors t+30<=anchor, eval=anchor",
            "n_trees": N_TREES,
            "lgb_params": {k: v for k, v in LGB_PARAMS.items()},
            "n_features": len(FEATURE_COLS),
            "anchor_step_days": ANCHOR_STEP,
            "grid_start": str(GRID_START),
        },
        "folds": {},
    }

    imp_agg: dict[str, float] = {}

    def fit_pool(tr: pl.DataFrame) -> lgb.Booster:
        X = tr.select(FEATURE_COLS).to_numpy().astype(np.float32)
        y = tr["z"].to_numpy().astype(np.float32)
        return lgb.train(LGB_PARAMS, lgb.Dataset(X, label=y), num_boost_round=N_TREES)

    for fold, anchor in EVAL_ANCHORS.items():
        tf = time.time()
        cutoff = anchor - timedelta(days=H)
        tr = panel.filter(
            (pl.col("anchor_date") <= cutoff) & pl.col("z").is_not_null()
        )
        ev = panel.filter(pl.col("anchor_date") == anchor)
        booster = fit_pool(tr)
        z_hat = booster.predict(ev.select(FEATURE_COLS).to_numpy().astype(np.float32))
        y_hat = np.clip(np.expm1(z_hat), 0, None)
        y_true = ev["y_next30"].to_numpy()
        naive = ev["f_gmv_30"].to_numpy().astype(np.float64)
        r_model = rmsle(y_true, y_hat)
        r_naive = rmsle(y_true, naive)
        zm, zn = np.log1p(np.clip(y_true, 0, None)) - z_hat, \
            np.log1p(np.clip(y_true, 0, None)) - np.log1p(np.clip(naive, 0, None))
        corr = float(np.corrcoef(zm, zn)[0, 1])
        results["folds"][fold] = {
            "anchor": str(anchor),
            "train_anchors": int(tr.select(pl.col("anchor_date").n_unique()).item()),
            "train_rows": tr.height,
            "rmsle_model": round(r_model, 5),
            "rmsle_naive": round(r_naive, 5),
            "naive_ref_eda": NAIVE_REF.get(fold),
            "resid_corr_vs_naive_z": round(corr, 4),
        }
        for fname, v in zip(FEATURE_COLS, booster.feature_importance("gain")):
            imp_agg[fname] = imp_agg.get(fname, 0.0) + float(v)
        print(f"{fold}: model={r_model:.5f} naive={r_naive:.5f} "
              f"(train {tr.height} rows, {time.time()-tf:.1f}s)", flush=True)

    prod_cut = last_train
    tr_prod = panel.filter(
        (pl.col("anchor_date") <= prod_cut) & pl.col("z").is_not_null()
    )
    ev_prod = panel.filter(pl.col("anchor_date") == PROD_ANCHOR)
    tp = time.time()
    booster = fit_pool(tr_prod)
    z_hat = booster.predict(ev_prod.select(FEATURE_COLS).to_numpy().astype(np.float32))
    y_hat = np.clip(np.expm1(z_hat), 0, None)
    print(f"prod model trained on {tr_prod.height} rows ({time.time()-tp:.1f}s)", flush=True)

    sub = pl.DataFrame({"user_id": ev_prod["user_id"], "predict": y_hat})
    sample = pl.read_csv(SAMPLE_PATH)
    sub = sample.select("user_id").join(sub, on="user_id", how="left").fill_null(0.0)
    SUB_PATH.parent.mkdir(parents=True, exist_ok=True)
    sub.write_csv(SUB_PATH)
    print(f"submission saved -> {SUB_PATH}", flush=True)

    top = sorted(imp_agg.items(), key=lambda kv: -kv[1])[:20]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([k for k, _ in top][::-1], [v for _, v in top][::-1])
    ax.set_title("exp11 LightGBM gain importance (summed over folds)")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "exp11_importance.png", dpi=130)

    results["feature_importance_top20"] = [[k, round(v, 1)] for k, v in top]
    results["total_seconds"] = round(time.time() - t0, 1)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
