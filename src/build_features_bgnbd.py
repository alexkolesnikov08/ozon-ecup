# Ported verbatim from teammate repo (github.com/Rafaildavar/ozon-ecup), 2026-08-24,
# for exp02.5 honest block ablation; paths are repo-root relative.
"""Probabilistic BTYD features per anchor (exp04): BG/NBD + Gamma-Gamma + EB.

Fits on TRAIN folds only (fold_00..02, pre-anchor windows), then produces
per-user features for every fold incl fold_end -> single parquet per fold:

    data/v2/features_bgnbd/<fold_name>.parquet
    keys: anchor_date, user_id

Feature blocks:

- inputs: bgnbd_x (repeat orders), bgnbd_tx, bgnbd_T, bgnbd_mon_freq, bgnbd_mbar
- bgnbd_p_alive     — P(alive | x, t_x, T)
- bgnbd_en30        — E[#orders next 30d] via hyp2f1 closed form (x>=1);
                       population prior for never-buyers
- eb_lambda_n30     — robust Gamma-Poisson empirical-Bayes E[#orders 30d]
                       (no dropout assumption; cold-start safe)
- gg_e_value        — Gamma-Gamma shrunk expected check per order
- bgnbd_e_gmv30     — bgnbd_en30 * gg_e_value
- eb_e_gmv30        — eb_lambda_n30 * gg_e_value

Run:  .venv/bin/python src/build_features_bgnbd.py [--smoke N]
"""

import argparse
import json
import time
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.special import gammaln, hyp2f1

DATA_PATH = Path("data/train.parquet")
OUT_DIR = Path("data/v2/features_bgnbd")

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
FIT_SAMPLE = 300_000


def rfm_per_anchor(data: pl.DataFrame, anchor: date) -> pl.DataFrame:
    hist = data.filter(pl.col("event_date") <= anchor)
    return (
        hist.group_by("user_id")
        .agg(
            (anchor - pl.col("event_date").min()).dt.total_days().cast(pl.Float64).alias("_T"),
            (anchor - pl.col("event_date").filter(pl.col("to_ord") > 0).min())
            .dt.total_days().cast(pl.Float64).alias("_first_ord"),
            (anchor - pl.col("event_date").filter(pl.col("to_ord") > 0).max())
            .dt.total_days().cast(pl.Float64).alias("_last_ord"),
            pl.when(pl.col("to_ord") > 0).then(1).otherwise(0).sum().cast(pl.Float64).alias("n_occ"),
            pl.when((pl.col("to_ord") > 0) & (pl.col("gmv") > 0))
            .then(pl.col("gmv")).otherwise(None).alias("_g"),
            pl.col("gmv").sum().alias("_life_gmv"),
        )
        .with_columns(
            (pl.col("_T") + 1.0).alias("bgnbd_T"),
            pl.when(pl.col("_first_ord").is_null())
            .then(None).otherwise(pl.col("_first_ord") - pl.col("_last_ord") + 1.0)
            .alias("bgnbd_tx"),
            pl.col("n_occ").alias("bgnbd_n_occasions"),
            pl.col("_g").list.mean().alias("bgnbd_mbar"),
            pl.col("_g").list.len().cast(pl.Float64).alias("bgnbd_mon_freq"),
        )
        .select(["user_id", "bgnbd_T", "bgnbd_tx", "bgnbd_n_occasions",
                 "bgnbd_mon_freq", "bgnbd_mbar"])
    )


def _bgnbd_nll(log_params: np.ndarray, x: np.ndarray, tx: np.ndarray, T: np.ndarray) -> float:
    r, alpha, a, b = np.exp(log_params)
    A1 = gammaln(r + x) - gammaln(r) + r * np.log(alpha)
    A2 = gammaln(a + b) + gammaln(b + x) - gammaln(b) - gammaln(a + b + x)
    A3 = -(r + x) * np.log(alpha + T)
    A4 = np.log(a) - np.log(b + np.maximum(x, 1.0) - 1.0) - (r + x) * np.log(alpha + tx)
    mx = np.maximum(A3, A4)
    ll = A1 + A2 + mx + np.log(np.exp(A3 - mx) + np.exp(A4 - mx))
    if not np.all(np.isfinite(ll)):
        return 1e12
    return -float(ll.mean())


def fit_bgnbd(df: pl.DataFrame) -> dict[str, float]:
    d = df.filter(pl.col("bgnbd_n_occasions") >= 1)
    if len(d) > FIT_SAMPLE:
        d = d.sample(FIT_SAMPLE, seed=42)
    x = (d["bgnbd_n_occasions"] - 1).to_numpy()
    tx = d["bgnbd_tx"].to_numpy()
    T = d["bgnbd_T"].to_numpy()
    res = minimize(
        _bgnbd_nll, x0=np.log([1.0, 100.0, 1.5, 3.0]), args=(x, tx, T),
        method="L-BFGS-B", bounds=[(-5, 5), (-5, 12), (-5, 5), (-5, 8)],
    )
    r, alpha, a, b = np.exp(res.x)
    print(f"  BG/NBD: r={r:.3f} alpha={alpha:.1f} a={a:.3f} b={b:.3f} "
          f"(NLL/user={res.fun:.4f}, conv={res.success})")
    return {"r": r, "alpha": alpha, "a": a, "b": b}


def _gg_nll(log_params: np.ndarray, x: np.ndarray, m: np.ndarray) -> float:
    p, q, v = np.exp(log_params)
    ll = (
        gammaln(p * x + q) - gammaln(p * x) - gammaln(q) + q * np.log(v)
        + (p * x - 1) * np.log(m) + (p * x) * np.log(x)
        - (p * x + q) * np.log(x * m + v)
    )
    return -float(ll.mean())


def fit_gamma_gamma(df: pl.DataFrame) -> dict[str, float]:
    d = df.filter(pl.col("bgnbd_mon_freq") >= 2)
    if len(d) > FIT_SAMPLE:
        d = d.sample(FIT_SAMPLE, seed=42)
    x = d["bgnbd_mon_freq"].to_numpy()
    m = np.clip(np.nan_to_num(d["bgnbd_mbar"].to_numpy(), nan=0.0), 0.01, None)
    res = minimize(_gg_nll, x0=np.log([6.0, 4.0, 1500.0]), args=(x, m),
                   method="L-BFGS-B", bounds=[(-5, 8), (-5, 8), (-2, 15)])
    p, q, v = np.exp(res.x)
    pop_mean = v * p / (q - 1)
    print(f"  Gamma-Gamma: p={p:.3f} q={q:.3f} v={v:.1f} "
          f"(pop mean check={pop_mean:.1f}, conv={res.success})")
    return {"p": p, "q": q, "v": v, "pop_mean_check": pop_mean}


def add_predictions(df: pl.DataFrame, bg: dict[str, float],
                    gg: dict[str, float]) -> pl.DataFrame:
    r, alpha, a, b = bg["r"], bg["alpha"], bg["a"], bg["b"]
    HORIZON = 30.0
    x = df["bgnbd_n_occasions"].to_numpy() - 1.0
    tx = df["bgnbd_tx"].to_numpy().astype(np.float64)
    T = df["bgnbd_T"].to_numpy().astype(np.float64)
    has_buy = (x >= 0) & ~np.isnan(tx)

    xx, txx, TT = x.copy(), tx.copy(), T.copy()
    xx[~has_buy] = 0.0
    txx[~has_buy] = 1.0
    TT[~has_buy] = 1.0

    log_div = (r + xx) * np.log(alpha + TT) - (r + xx) * np.log(alpha + txx) \
        + np.log(a / (b + np.clip(xx - 1, 0, None)))
    p_alive = np.where(has_buy, 1.0 / (1.0 + np.exp(np.clip(log_div, -500, 500))), 1.0)

    ft = (a + b + xx - 1.0) / (a - 1.0)
    st = 1.0 - ((alpha + TT) / (alpha + TT + HORIZON)) ** (r + xx) * hyp2f1(
        r + xx, b + xx, a + b + xx - 1.0, HORIZON / (alpha + TT + HORIZON)
    )
    den = 1.0 + has_buy * (a / (b + np.clip(xx, 1, None) - 1.0)) \
        * ((alpha + TT) / (alpha + txx)) ** (r + xx)
    en30 = np.where(has_buy, ft * st / den, np.nan)

    lam_pop = r / alpha
    n_occ = df["bgnbd_n_occasions"].to_numpy()
    Td = df["bgnbd_T"].to_numpy()
    eb_lambda = (n_occ + r) / (Td + alpha)
    eb_lambda_n30 = np.clip(eb_lambda, 0, None) * HORIZON
    en30_filled = np.where(np.isfinite(en30), en30,
                           np.where(has_buy, eb_lambda_n30 / HORIZON, lam_pop) * HORIZON)

    p_, q_, v_ = gg["p"], gg["q"], gg["v"]
    pop_check = gg["pop_mean_check"]
    xf = df["bgnbd_mon_freq"].to_numpy()
    mb = np.nan_to_num(df["bgnbd_mbar"].to_numpy(), nan=pop_check)
    w = np.where(xf >= 2, np.clip((p_ * xf - 1.0) / (p_ * xf - 1.0 + q_), 0.0, 1.0), 0.0)
    personal = np.where(xf >= 1, (v_ + xf * mb) / np.maximum(xf, 1.0), pop_check)
    e_value = (1.0 - w) * pop_check + w * personal

    out = df.with_columns([
        pl.Series("bgnbd_p_alive", p_alive),
        pl.Series("bgnbd_en30", np.clip(en30_filled, 0, None)),
        pl.Series("eb_lambda_n30", eb_lambda_n30),
        pl.Series("gg_e_value", e_value),
    ])
    return out.with_columns([
        (pl.col("bgnbd_en30") * pl.col("gg_e_value")).alias("bgnbd_e_gmv30"),
        (pl.col("eb_lambda_n30") * pl.col("gg_e_value")).alias("eb_e_gmv30"),
    ])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=None)
    args = ap.parse_args()

    t0 = time.time()
    data = pl.read_parquet(DATA_PATH).with_columns(pl.col("event_date").cast(pl.Date))

    rfm_folds: dict[str, pl.DataFrame] = {}
    for fold_name, anchor in zip(FOLD_NAMES, ANCHORS):
        rf = rfm_per_anchor(data, anchor)
        if args.smoke:
            rf = rf.head(args.smoke)
        rfm_folds[fold_name] = rf
        print(f"{fold_name}: {rf.height:,} users")

    pooled = pl.concat(
        [rfm_folds[f].filter(pl.col("bgnbd_n_occasions") >= 1) for f in TRAIN_FOLDS],
        how="vertical",
    )
    print(f"\nfitting on pooled train folds: {pooled.height:,} buyer-rows")
    bg = fit_bgnbd(pooled)
    gg = fit_gamma_gamma(pooled)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for fold_name, anchor in zip(FOLD_NAMES, ANCHORS):
        out = add_predictions(rfm_folds[fold_name], bg, gg)
        out = out.with_columns(pl.lit(anchor).alias("anchor_date"))
        out = out.select(["anchor_date"] + [c for c in out.columns if c != "anchor_date"])
        out.write_parquet(OUT_DIR / f"{fold_name}.parquet")
        ev = out["bgnbd_e_gmv30"]
        print(f"{fold_name}: saved, E[gmv30] mean={ev.mean():.1f} p99={ev.quantile(0.99):.1f}")

    meta_path = OUT_DIR / "fit_params.json"
    meta_path.write_text(json.dumps({"bgnbd": bg, "gamma_gamma": gg,
                                     "fitted_on": TRAIN_FOLDS}, indent=2),
                         encoding="utf-8")
    print(f"\nparams -> {meta_path}")
    print(f"ALL DONE in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
