"""BTYD (Buy-Till-You-Die) pipeline: BG/NBD + Gamma-Gamma meta-features -> LightGBM -> submission.

Leak-free time-based design ("last 30 days = local target"):

    DATA_END = max(event_date)                      # 2026-02-13
    VAL_ANCHOR  = DATA_END - 30d                    # local holdout target: next 30d after VAL
    FIT_ANCHORS = [VAL-56d, VAL-28d]                # earlier anchors, targets fully observed

Per anchor A every feature is computed ONLY from rows with event_date <= A;
target y_A = sum(gmv) over (A, A+30].

Models:
    m_es    : LGBM trained on FIT[0], early-stopped on FIT[1] -> best_iteration
    m_eval  : refit on FIT[0]+FIT[1] with frozen rounds -> honest RMSLE on VAL_ANCHOR
    m_final : refit on FIT[0]+FIT[1]+VAL                -> predict END_ANCHOR -> submission

BTYD block (lifetimes):
    frequency  = repeated purchase days (gmv > 0), i.e. buy_days - 1
    recency    = last_purchase_day - first_purchase_day
    T          = age since first purchase (+1, calendar days); never-buyers use activity age
    monetary   = mean gmv over purchase days
    outputs    : p_alive, E[purchases 30d], E[check | Gamma-Gamma], E[GMV 30d]

Outputs:
    submissions/submission_btyd.csv          LGBM on log1p-target (RMSLE-optimal z-space)
    submissions/submission_btyd_direct.csv   pure BTYD E[gmv30] baseline
    reports/btyd_fit_params.json             per-anchor BG/NBD & GG params + val metrics

Run:
    .venv/bin/python src/train_btyd.py [--smoke N] [--tag NAME]
"""

from __future__ import annotations

import argparse
import gc
import json
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter

DATA_PATH = Path("data/train.parquet")
SAMPLE_PATH = Path("sample_submit.csv")
OUT_DIR = Path("submissions")
REPORT_DIR = Path("reports")

DATE_COL_CANDIDATES = ("event_date", "date")
CART_COL_CANDIDATES = ("to_cart", "cart_adds")
SEARCH_COL_CANDIDATES = ("searches",)
GMV_COL = "gmv"
USER_COL = "user_id"

HORIZON = 30
FIT_SAMPLE_CAP = 400_000
PENALIZERS = (0.001, 0.01, 0.1, 1.0)
SEED = 42

LGB_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 300,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l2": 5.0,
    "n_jobs": -1,
    "seed": SEED,
    "verbosity": -1,
}

FEATURES = [
    "bt_buy_days", "bt_recency_days", "bt_T_days", "bt_mon_mean",
    "bt_tenure_act", "bt_act_days", "d_last_buy", "d_last_act",
    "s_7", "s_14", "s_30", "c_7", "c_14", "c_30",
    "g_7", "g_14", "g_30", "act_30",
    "bt_p_alive", "bt_pred_purch_30", "bt_exp_value", "bt_exp_gmv30",
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def resolve_column(df: pd.DataFrame, candidates: tuple[str, ...], what: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"column for '{what}' not found; tried {candidates}, have {list(df.columns)}")


def load_data(smoke_users: np.ndarray | None) -> tuple[pd.DataFrame, str, str]:
    df = pd.read_parquet(DATA_PATH)
    date_col = resolve_column(df, DATE_COL_CANDIDATES, "date")
    cart_col = resolve_column(df, CART_COL_CANDIDATES, "cart adds")
    search_col = resolve_column(df, SEARCH_COL_CANDIDATES, "searches")
    df = df[[USER_COL, date_col, GMV_COL, cart_col, search_col]]
    df.columns = [USER_COL, "event_date", "gmv", "cart_adds", "searches"]

    df[USER_COL] = df[USER_COL].astype(np.int32)
    df["gmv"] = df["gmv"].astype(np.float32)
    df["cart_adds"] = df["cart_adds"].astype(np.int32)
    df["searches"] = df["searches"].astype(np.int32)
    df["event_date"] = pd.to_datetime(df["event_date"])
    df = df.sort_values(["event_date", USER_COL], kind="stable", ignore_index=True)

    if smoke_users is not None:
        df = df[df[USER_COL].isin(smoke_users)].reset_index(drop=True)
    gc.collect()
    return df, cart_col, search_col


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    z = np.log1p(np.clip(y_pred, 0, None))
    return float(np.sqrt(np.mean((np.log1p(y_true) - z) ** 2)))


def window_sums(hist: pd.DataFrame, anchor: pd.Timestamp, days: int) -> pd.DataFrame:
    lo = anchor - pd.Timedelta(days=days - 1)
    w = hist[hist["event_date"] >= lo]
    out = w.groupby(USER_COL, sort=False).agg(
        s=("searches", "sum"), c=("cart_adds", "sum"),
        g=("gmv", "sum"), d=("event_date", "size"),
    )
    return out.astype(np.float64)


def build_features(hist: pd.DataFrame, anchor: pd.Timestamp, user_index: pd.Index,
                   cart_note: str = "") -> pd.DataFrame:
    ts = anchor
    life = hist.groupby(USER_COL, sort=False).agg(
        first_act=("event_date", "min"), last_act=("event_date", "max"),
        act_days=("event_date", "size"),
    )
    buys = hist[hist["gmv"] > 0]
    buy = buys.groupby(USER_COL, sort=False).agg(
        first_buy=("event_date", "min"), last_buy=("event_date", "max"),
        buy_days=("event_date", "size"), mon_sum=("gmv", "sum"),
    )

    X = pd.DataFrame(index=user_index)
    life = life.reindex(user_index)
    buy = buy.reindex(user_index)

    first_act = life["first_act"]
    first_buy = buy["first_buy"]
    has_buy = first_buy.notna()

    X["bt_act_days"] = life["act_days"].fillna(0.0)
    X["bt_tenure_act"] = (ts - first_act).dt.days.fillna(0).clip(lower=0) + 1.0
    X["bt_buy_days"] = buy["buy_days"].fillna(0.0)
    X["bt_recency_days"] = ((buy["last_buy"] - first_buy).dt.days).fillna(0.0).clip(lower=0)
    t_buy = ((ts - first_buy).dt.days.clip(lower=0) + 1.0).where(has_buy)
    X["bt_T_days"] = t_buy.fillna(X["bt_tenure_act"])
    X["bt_mon_mean"] = (buy["mon_sum"] / buy["buy_days"]).fillna(0.0)
    X["d_last_buy"] = ((ts - buy["last_buy"]).dt.days).fillna(9999.0).clip(lower=0)
    X["d_last_act"] = ((ts - life["last_act"]).dt.days).fillna(9999.0).clip(lower=0)

    for days in (7, 14, 30):
        agg = window_sums(hist, ts, days).reindex(user_index)
        X[f"s_{days}"] = agg["s"].fillna(0.0)
        X[f"c_{days}"] = agg["c"].fillna(0.0)
        X[f"g_{days}"] = agg["g"].fillna(0.0)
        if days == 30:
            X["act_30"] = agg["d"].fillna(0.0)
    return X.astype(np.float64)


def build_target(df: pd.DataFrame, anchor: pd.Timestamp, user_index: pd.Index) -> np.ndarray:
    hi = anchor + pd.Timedelta(days=HORIZON)
    tgt = df[(df["event_date"] > anchor) & (df["event_date"] <= hi)]
    y = tgt.groupby(USER_COL, sort=False)["gmv"].sum()
    return y.reindex(user_index).fillna(0.0).to_numpy(dtype=np.float64)


def _finite(params) -> bool:
    return bool(np.all(np.isfinite(list(params.values()))))


def fit_bgnbd(summary: pd.DataFrame) -> BetaGeoFitter:
    d = summary[summary["frequency"] >= 0]
    if len(d) > FIT_SAMPLE_CAP:
        d = d.sample(FIT_SAMPLE_CAP, random_state=SEED)
    last_err: Exception | None = None
    for pen in PENALIZERS:
        try:
            f = BetaGeoFitter(penalizer_coef=pen)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                f.fit(d["frequency"], d["recency"], d["T"])
            if _finite(f.params_):
                log(f"  BG/NBD pen={pen}: {dict(f.params_.round(4))}")
                return f
        except Exception as e:
            last_err = e
            log(f"  BG/NBD failed (pen={pen}): {e}")
    raise RuntimeError(f"BG/NBD did not converge: {last_err}")


def fit_gamma_gamma(summary: pd.DataFrame) -> tuple[GammaGammaFitter | None, float]:
    d = summary[(summary["frequency"] >= 1) & (summary["monetary_value"] > 0)]
    pop_mean = float(d["monetary_value"].median()) if len(d) else 0.0
    if len(d) < 100:
        log(f"  Gamma-Gamma skipped: only {len(d)} repeat-buyers -> pop mean={pop_mean:.1f}")
        return None, pop_mean
    if len(d) > FIT_SAMPLE_CAP:
        d = d.sample(FIT_SAMPLE_CAP, random_state=SEED)
    last_err: Exception | None = None
    for pen in PENALIZERS:
        try:
            f = GammaGammaFitter(penalizer_coef=pen)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                f.fit(d["frequency"], d["monetary_value"], q_constraint=True)
            if _finite(f.params_) and f.params_["q"] > 1.0:
                log(f"  Gamma-Gamma pen={pen}: {dict(f.params_.round(4))} pop_check={pop_mean:.1f}")
                return f, pop_mean
        except Exception as e:
            last_err = e
            log(f"  Gamma-Gamma failed (pen={pen}): {e}")
    log(f"  Gamma-Gamma fallback -> empirical pop mean={pop_mean:.1f} ({last_err})")
    return None, pop_mean


def btyd_scores(summary: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    bg = fit_bgnbd(summary)
    gg, pop_mean = fit_gamma_gamma(summary)

    freq = summary["frequency"].to_numpy(dtype=np.float64)
    rec = summary["recency"].to_numpy(dtype=np.float64)
    t = summary["T"].to_numpy(dtype=np.float64)
    mon = summary["monetary_value"].to_numpy(dtype=np.float64)
    has_buy = summary["has_buy"].to_numpy(dtype=bool)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with np.errstate(all="ignore"):
            p_alive = bg.conditional_probability_alive(freq, rec, t)
            purch = bg.conditional_expected_number_of_purchases_up_to_time(
                float(HORIZON), freq, rec, t)
            if gg is not None:
                value = gg.conditional_expected_average_profit(
                    summary["frequency"], summary["monetary_value"]).to_numpy(dtype=np.float64)
            else:
                value = np.full(len(summary), pop_mean, dtype=np.float64)

    p_alive = np.nan_to_num(p_alive, nan=0.0)
    purch = np.nan_to_num(purch, nan=0.0)
    value = np.where(has_buy, np.nan_to_num(value, nan=pop_mean), pop_mean)
    p_alive = np.clip(p_alive, 0.0, 1.0)
    purch = np.clip(purch, 0.0, None)
    value = np.clip(value, 0.0, None)
    gmv = purch * value
    params = {"bgnbd": dict(bg.params_), "gg": {} if gg is None else dict(gg.params_),
              "gg_pop_mean": pop_mean}
    return p_alive, purch, value, gmv, params


def make_dataset(df: pd.DataFrame, anchor_ts: pd.Timestamp, user_index: pd.Index,
                 with_target: bool, tag: str) -> tuple[np.ndarray, np.ndarray | None, dict]:
    log(f"dataset[{tag}] anchor={anchor_ts.date()} hist_rows<={(len(df)):,} users={len(user_index):,}")
    hist = df[df["event_date"] <= anchor_ts]
    X = build_features(hist, anchor_ts, user_index)

    summary = pd.DataFrame({
        "frequency": X["bt_buy_days"].to_numpy() - 1.0,
        "recency": X["bt_recency_days"].to_numpy(),
        "T": X["bt_T_days"].to_numpy(),
        "monetary_value": X["bt_mon_mean"].to_numpy(),
        "has_buy": X["bt_buy_days"].to_numpy() >= 1,
    })
    summary["T"] = np.maximum(summary["T"], 1.0)
    summary.loc[~summary["has_buy"], ["frequency", "recency", "monetary_value"]] = 0.0

    p_alive, purch, value, gmv, params = btyd_scores(summary)
    X["bt_p_alive"] = p_alive
    X["bt_pred_purch_30"] = purch
    X["bt_exp_value"] = value
    X["bt_exp_gmv30"] = gmv

    X = X[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = build_target(df, anchor_ts, user_index) if with_target else None
    if y is not None:
        log(f"dataset[{tag}] y: mean={y.mean():.1f} nonzero={(y > 0).mean():.3f} "
            f"direct-BTYD RMSLE={rmsle(y, X['bt_exp_gmv30'].to_numpy()):.4f}")
    del hist
    gc.collect()
    return X.to_numpy(dtype=np.float32), y, params


def train_lgbm(X: np.ndarray, y_log: np.ndarray, X_es: np.ndarray | None,
               y_es_log: np.ndarray | None, rounds: int) -> lgb.Booster:
    train_set = lgb.Dataset(X, label=y_log, free_raw_data=False)
    if X_es is not None:
        es_set = lgb.Dataset(X_es, label=y_es_log, reference=train_set)
        booster = lgb.train(
            LGB_PARAMS, train_set, num_boost_round=rounds, valid_sets=[es_set],
            callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
        log(f"  LGBM ES: best_iter={booster.best_iteration}")
        return booster
    return lgb.train(LGB_PARAMS, train_set, num_boost_round=rounds,
                     callbacks=[lgb.log_evaluation(0)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=None, help="use first N users of sample_submit")
    ap.add_argument("--tag", default="", help="suffix for output files")
    args = ap.parse_args()
    suffix = f"_{args.tag}" if args.tag else ""

    sample = pd.read_csv(SAMPLE_PATH, usecols=["user_id"])
    smoke_users = sample["user_id"].to_numpy()[: args.smoke] if args.smoke else None
    if smoke_users is not None:
        sample = sample[sample["user_id"].isin(set(smoke_users.tolist()))]

    df, _, _ = load_data(smoke_users)
    data_end = pd.Timestamp(df["event_date"].max())
    val_anchor = data_end - pd.Timedelta(days=HORIZON)
    fit_anchors = [val_anchor - pd.Timedelta(days=d) for d in (56, 28)]
    log(f"data: {df['event_date'].min().date()}..{data_end.date()} "
        f"rows={len(df):,} users={df[USER_COL].nunique():,}")
    log(f"anchors: fit={[a.date() for a in fit_anchors]} val={val_anchor.date()} end={data_end.date()}")

    all_users = pd.Index(np.sort(df[USER_COL].unique()))
    datasets: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}
    fit_params: dict[str, dict] = {}
    for i, a in enumerate(fit_anchors):
        X, y, p = make_dataset(df, a, all_users, True, f"fit{i}")
        datasets[f"fit{i}"] = (X, np.log1p(y) if y is not None else None)
        fit_params[str(a.date())] = p
    X_val, y_val, p = make_dataset(df, val_anchor, all_users, True, "val")
    datasets["val"] = (X_val, np.log1p(y_val))
    fit_params[str(val_anchor.date())] = p
    val_raw_y = y_val

    X_end, _, p = make_dataset(df, data_end, all_users, False, "end")
    fit_params[str(data_end.date())] = p

    log("training: ES model")
    m_es = train_lgbm(datasets["fit0"][0], datasets["fit0"][1],
                      datasets["fit1"][0], datasets["fit1"][1], 4000)
    rounds = min(max(int(m_es.best_iteration * 1.05) + 50, 400), 4000)

    log(f"training: eval model ({rounds} rounds)")
    X_tr = np.concatenate([datasets["fit0"][0], datasets["fit1"][0]])
    y_tr = np.concatenate([datasets["fit0"][1], datasets["fit1"][1]])
    m_eval = train_lgbm(X_tr, y_tr, None, None, rounds)
    pz = m_eval.predict(X_val)
    val_rmsle = rmsle(val_raw_y, np.expm1(pz))
    direct_rmsle = rmsle(val_raw_y, X_val[:, FEATURES.index("bt_exp_gmv30")])
    log(f"VALIDATION @ {val_anchor.date()}: LGBM RMSLE={val_rmsle:.5f} | "
        f"direct BTYD RMSLE={direct_rmsle:.5f}")

    log(f"training: final model ({rounds} rounds on fit0+fit1+val)")
    X_tr_full = np.concatenate([X_tr, datasets["val"][0]])
    y_tr_full = np.concatenate([y_tr, datasets["val"][1]])
    m_fin = train_lgbm(X_tr_full, y_tr_full, None, None, rounds)

    pred_z = m_fin.predict(X_end)
    pred = np.clip(np.expm1(pred_z), 0, None)
    direct = np.clip(X_end[:, FEATURES.index("bt_exp_gmv30")], 0, None)
    log(f"final preds: mean={pred.mean():.1f} median={np.median(pred):.1f} "
        f"p99={np.percentile(pred, 99):.1f} share_zero={(pred < 1).mean():.3f}")

    OUT_DIR.mkdir(exist_ok=True)
    pred_by_user = pd.Series(pred, index=all_users)
    direct_by_user = pd.Series(direct, index=all_users)
    sub = pd.DataFrame({"user_id": sample["user_id"].to_numpy(np.int64)})
    sub["predict"] = pred_by_user.reindex(sub["user_id"]).fillna(0.0).to_numpy()
    sub_path = OUT_DIR / f"submission_btyd{suffix}.csv"
    sub.to_csv(sub_path, index=False)

    sub_d = sub.copy()
    sub_d["predict"] = direct_by_user.reindex(sub_d["user_id"]).fillna(0.0).to_numpy()
    sub_d_path = OUT_DIR / f"submission_btyd_direct{suffix}.csv"
    sub_d.to_csv(sub_d_path, index=False)

    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / f"btyd_fit_params{suffix}.json").write_text(json.dumps({
        "anchors": {"fit": [str(a.date()) for a in fit_anchors],
                    "val": str(val_anchor.date()), "end": str(data_end.date())},
        "horizon_days": HORIZON,
        "fit_params": fit_params,
        "lgbm_rounds": rounds,
        "val_rmsle_lgbm": val_rmsle,
        "val_rmsle_direct_btyd": direct_rmsle,
    }, indent=2, default=str), encoding="utf-8")

    log(f"saved: {sub_path}\nsaved: {sub_d_path}\nDONE")


if __name__ == "__main__":
    main()
