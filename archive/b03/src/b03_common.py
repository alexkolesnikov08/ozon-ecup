"""b03 (PLSI-CAL V1') shared constants and helpers.

Platform-level STL seasonality index m_hat_t = y_t / trend_t (multiplicative,
ratio-to-smooth-trend), estimated CAUSALLY per anchor: STL is fitted only on
daily platform sums with event_date <= anchor. Target-window normalisation
factor M_window(anchor) is a deterministic function of data <= anchor:
    M(a) = mean_{k=1..30} G(a,k) * base(d)
where
    G(a,k)  = g_yoy ** (k/365) - causal annual trend growth, g_yoy =
              mean(T trailing-L)/mean(T same dates -365), longest L<=30 with
              L>=14 (else 1.0: first-year anchors carry no separable annual
              info; a short-window ratio would project the January V-shape
              of the STL trend forward - measured T(91d)-ratio 0.913 at the
              fold_03 anchor vs true +46%/yr);
    base(d) = m_hat_{d-365} * wp(dow(d))/wp(dow(d-365))   [date-exact prior-year
              index with day-of-week correction, if d-365 within data]
            = wp(dow(d))                                   [fallback: trailing
              weekly profile], wp = trailing 70d mean of m_hat by dow, mean-1.
mode="assist" replaces base(d) by the REALISED full-sample m_hat_d (oracle;
used only in the proxy-fold diagnostic, symmetric to exp02's realised s_hat).
"""

import sys

sys.dont_write_bytecode = True

from datetime import date, timedelta  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from statsmodels.tsa.seasonal import STL  # noqa: E402

SEED = 42

DATA_PATH = Path("data/train.parquet")
OUT_DIR = Path("data/v2/b03_plsi")
INDEX_DIR = OUT_DIR / "index"
PARTS_DIR = OUT_DIR / "parts"
FIG_DIR = Path("reports/b03_figures")
METRICS_PATH = Path("reports/b03_metrics.json")
SUB_PATH = Path("submissions/b03_submission_plsi.csv")

HORIZON = 30
STL_PERIOD = 7
STL_ROBUST = True
GROWTH_CLIP = (0.85, 1.8)
WEEKLY_TRAILING = 70
LOOKBACK = 365  # date-exact prior-year mapping (weekday corrected separately)

CV_FOLDS = ["fold_00", "fold_01", "fold_02", "fold_03"]
TRAIN_FOLDS = CV_FOLDS[:3]
ANCHORS = {
    "fold_00": date(2025, 12, 3),
    "fold_01": date(2025, 12, 17),
    "fold_02": date(2025, 12, 31),
    "fold_03": date(2026, 1, 14),
    "fold_proxy": date(2025, 2, 14),
    "fold_end": date(2026, 2, 13),
}
PROXY_ANCHOR = ANCHORS["fold_proxy"]
END_ANCHOR = ANCHORS["fold_end"]

METRIC_COLS = [
    "gmv", "searches", "to_ord", "to_cart",
    "gmv_search", "gmv_cat", "search_to_ord", "cat_to_ord", "cat",
]

BASE_VALUE_COLS = ["gmv", "searches", "to_ord", "to_cart"]
BASE_WINDOWS = [(7, "7d"), (30, "30d"), (60, "60d"), (90, "90d")]
RECENCY_NONE = 999

WINDOW_COLS = [
    "gmv", "searches", "to_ord", "to_cart",
    "gmv_search", "gmv_cat", "search_to_ord", "cat_to_ord", "cat",
]
WINDOWS = [("7d", 7), ("30d", 30), ("90d", 90)]

EXTRA_ACCEPTED = [
    "conv_s2o", "conv_c2o", "conv_o2c",
    "aov_30", "ord_days_30",
    "due_ratio",
    "share_gmv_search_90", "share_gmv_cat_90",
    "share_gmv_search_trend", "share_gmv_cat_trend",
]
ALLOWED_NAN = {"conv_s2o", "conv_c2o", "conv_o2c", "due_ratio"}

DROP_COLS = ["anchor_date", "user_id", "target"]


def platform_daily() -> pl.DataFrame:
    """Daily platform totals: event_date, y (sum gmv), n_users."""
    df = pl.read_parquet(DATA_PATH, columns=["event_date", "user_id", "gmv"])
    d = (
        df.group_by("event_date")
        .agg(pl.col("gmv").sum().alias("y"), pl.col("user_id").len().alias("n_users"))
        .sort("event_date")
    )
    assert d.height > 400 and d["y"].is_finite().all() and (d["y"] > 0).all()
    return d


def stl_trend(y: np.ndarray) -> np.ndarray:
    res = STL(y, period=STL_PERIOD, robust=STL_ROBUST).fit()
    return np.asarray(res.trend, dtype=float)


def index_from_frame(daily: pl.DataFrame) -> pl.DataFrame:
    """STL on the given (already causally sliced) daily frame -> index table."""
    dates = daily["event_date"].to_list()
    y = daily["y"].to_numpy()
    trend = stl_trend(y)
    assert (trend > 0).all(), "non-positive STL trend"
    return pl.DataFrame({
        "event_date": dates,
        "y": y,
        "trend": trend,
        "m_hat": y / trend,
    })


def yoy_growth(idx: pl.DataFrame, anchor: date) -> tuple[float, str, int]:
    """Causal annual growth g = mean(T trailing-L) / mean(T same dates -365).

    L is the longest trailing window <= 30 whose prior-year copy lies inside
    the data; requires L >= 14, else returns 1.0 (no separable annual info -
    first-year anchors). A short-window ratio (e.g. 91d) is NOT usable because
    the STL trend carries annual seasonality (Q4 ramp / New-Year dip), which
    would project the January V-shape forward (measured: T_ratio(91d) at
    fold_03 anchor = 0.913 while true YoY growth is ~+46%/yr).
    """
    dates = idx["event_date"].to_list()
    pos = {d: i for i, d in enumerate(dates)}
    ia = pos[anchor]
    trend = idx["trend"].to_numpy()
    for L in range(min(30, ia + 1), 13, -1):
        src0 = anchor - timedelta(days=L - 1) - timedelta(days=LOOKBACK)
        src1 = anchor - timedelta(days=LOOKBACK)
        if src0 in pos and src1 in pos:
            i0, i1 = pos[src0], pos[src1]
            assert i1 - i0 == L - 1
            cur = float(trend[ia - L + 1:ia + 1].mean())
            past = float(trend[i0:i1 + 1].mean())
            g = float(np.clip(cur / past, *GROWTH_CLIP))
            return g, "yoy_trend_-365", L
    return 1.0, "insufficient_history_fallback_g1", 0


def weekly_profile(idx: pl.DataFrame, trailing: int = WEEKLY_TRAILING) -> dict[int, float]:
    tail = idx.tail(trailing)
    dow = np.array([d.weekday() for d in tail["event_date"].to_list()])
    mh = tail["m_hat"].to_numpy()
    prof = {w: float(mh[dow == w].mean()) for w in range(7)}
    m = float(np.mean(list(prof.values())))
    prof = {w: max(v / m, 0.5) for w, v in prof.items()}
    return prof


def m_window(
    anchor: date,
    idx: pl.DataFrame,
    mode: str = "strict",
    realized_idx: pl.DataFrame | None = None,
) -> dict:
    """Deterministic causal window factor M(anchor) + component breakdown."""
    assert mode in ("strict", "assist")
    dates = idx["event_date"].to_list()
    pos = {d: i for i, d in enumerate(dates)}
    assert anchor in pos, f"anchor {anchor} outside index slice"
    ia = pos[anchor]
    mh = idx["m_hat"].to_numpy()

    g, g_method, L_used = yoy_growth(idx, anchor)

    wp = weekly_profile(idx)
    rpos = None
    if mode == "assist":
        assert realized_idx is not None, "assist mode requires realised full index"
        rdates = realized_idx["event_date"].to_list()
        rpos = {d: i for i, d in enumerate(rdates)}
        rmh = realized_idx["m_hat"].to_numpy()

    terms = []
    for k in range(1, HORIZON + 1):
        d = anchor + timedelta(days=k)
        gk = g ** (k / 365.0) if L_used >= 14 else 1.0
        if mode == "assist":
            assert d in rpos, f"realised index missing {d}"
            base, src_kind = float(rmh[rpos[d]]), "realised_mhat"
        else:
            src = d - timedelta(days=LOOKBACK)
            if src in pos:
                wc = wp[d.weekday()] / wp[src.weekday()]
                base, src_kind = float(mh[pos[src]]) * wc, "prior_year_index+dowcorr"
            else:
                base, src_kind = wp[d.weekday()], "weekly_profile_fallback"
        terms.append({"day": d.isoformat(), "k": k, "growth": round(gk, 6),
                      "base": round(base, 6), "source": src_kind})

    M = float(np.mean([t["base"] * t["growth"] for t in terms]))
    cov_prior = float(np.mean([t["source"] == "prior_year_index+dowcorr" for t in terms]))
    return {
        "anchor": anchor.isoformat(),
        "mode": mode,
        "M": round(M, 6),
        "log_M": round(float(np.log(M)), 6),
        "growth_annual": round(g, 6),
        "growth_method": g_method,
        "growth_window_days": int(L_used),
        "prior_year_coverage_frac": round(cov_prior, 4),
        "weekly_profile": {k: round(v, 5) for k, v in wp.items()},
        "terms": terms,
    }


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    lt = np.log1p(np.clip(y_true, 0, None))
    lp = np.log1p(np.clip(y_pred, 0, None))
    return float(np.sqrt(np.mean((lt - lp) ** 2)))
