"""exp10 - Классическая декомпозиция: платформенный ряд x доля юзера.

Идея: GMV юзера за окно = (прогноз платформенного ряда на окно) x (доля юзера).
Платформенный дневной ряд суммарного GMV прогнозируется классическими методами
временных рядов, затем распределяется по юзерам пропорционально их недавней доле
с опциональным шринкейджем к равномерной доле по активности.

Модели платформенного уровня:
  holt_damped : ETS, аддитивный затухающий тренд (Holt)
  hw_add7     : Holt-Winters + недельная сезонность (add, period=7)
  sarima_w    : SARIMA (1,1,1)x(1,0,1)[7]
  drift_ratio : эмпирический дрейф - сумма последних 30д x медиана отношений
                соседних выровненных 30д-блоков истории

Варианты распределения:
  naive              : carry-forward past30 (референс)
  <model>__preserve  : y_u = G_hat * (past30_u / G_prev30)
  <model>__shrinkK   : шринкейдж доли псевдо-каунтами K по числу активных дней
  oracle_preserve    : то же с фактическим уровнем окна (потолок семейства)

Оценка: RMSLE на fold_00..03 (все 250k юзеров, нули в outer join). fold_end -
только прогнозы платформы (таргета нет). Артефакты: reports/exp10_decomposition.json,
reports/figures/exp10_<fold>.png.

Запуск из корня репо: .venv/bin/python src/exp10_decomposition.py
"""

from __future__ import annotations

import json
import warnings
from datetime import date, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

warnings.filterwarnings("ignore")

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

DATA_PATH = Path("data/train.parquet")
REPORT_DIR = Path("reports")
FIG_DIR = REPORT_DIR / "figures"
OUT_PATH = REPORT_DIR / "exp10_decomposition.json"

H = 30
ANCHORS = {
    "fold_00": date(2025, 12, 3),
    "fold_01": date(2025, 12, 17),
    "fold_02": date(2025, 12, 31),
    "fold_03": date(2026, 1, 14),
    "fold_end": date(2026, 2, 13),
}
CV_FOLDS = ["fold_00", "fold_01", "fold_02", "fold_03"]
PLATFORM_MODELS = ["holt_damped", "hw_add7", "sarima_w", "drift_ratio"]
SHRINK_K = [5, 25]
NAIVE_REF = {  # из EDA для sanity-check джойнов
    "fold_00": 2.22630,
    "fold_01": 2.21661,
    "fold_02": 2.22673,
    "fold_03": 2.19506,
}


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    zt = np.log1p(y_true)
    zp = np.log1p(np.clip(y_pred, 0, None))
    return float(np.sqrt(np.mean((zt - zp) ** 2)))


def fit_platform_forecast(hist_pd: pd.Series, model: str) -> np.ndarray:
    """Дневной прогноз H дней вперёд по платформенному ряду."""
    if model == "holt_damped":
        fit = ExponentialSmoothing(
            hist_pd, trend="add", damped_trend=True, initialization_method="estimated"
        ).fit(optimized=True)
        return np.clip(fit.forecast(H).to_numpy(), 0, None)
    if model == "hw_add7":
        fit = ExponentialSmoothing(
            hist_pd,
            trend="add",
            damped_trend=True,
            seasonal="add",
            seasonal_periods=7,
            initialization_method="estimated",
        ).fit(optimized=True)
        return np.clip(fit.forecast(H).to_numpy(), 0, None)
    if model == "sarima_w":
        fit = SARIMAX(
            hist_pd,
            order=(1, 1, 1),
            seasonal_order=(1, 0, 1, 7),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        return np.clip(fit.forecast(H).to_numpy(), 0, None)
    raise ValueError(model)


def drift_ratio_total(daily_hist: pl.DataFrame, anchor: date, k_blocks: int = 4):
    """Эмпирический дрейф: медиана отношений соседних 30д-блоков x последний блок."""
    blocks = []
    for j in range(k_blocks + 1):
        hi = anchor - timedelta(days=30 * j)
        lo = hi - timedelta(days=29)
        blocks.append(
            float(daily_hist.filter(pl.col("event_date").is_between(lo, hi))["gmv"].sum())
        )
    ratios = np.array([b / b2 for b, b2 in zip(blocks[:-1], blocks[1:]) if b2 > 0])
    total = blocks[0] * float(np.median(ratios)) if len(ratios) else float(blocks[0])
    total = max(total, 0.0)
    daily = np.full(H, total / H)
    return total, daily


def allocate(past30: np.ndarray, n30: np.ndarray, ghat: float, gprev: float, k=None):
    """Распределить прогноз платформы по долям юзеров (с шринкейджем k)."""
    num = past30.astype(float).copy()
    if k is not None:
        w = n30.astype(float) / (n30.astype(float) + k)
        num = w * num + (1 - w) * gprev / len(num)
    tot = num.sum()
    if tot <= 0:
        return np.zeros_like(num)
    return ghat * num / tot


def main() -> None:
    raw = pl.read_parquet(DATA_PATH)
    users = raw.select(pl.col("user_id").unique().sort()).to_series()

    daily_full = (
        raw.group_by("event_date")
        .agg(pl.col("gmv").sum().alias("gmv"))
        .sort("event_date")
    )
    cal = pl.DataFrame(
        {
            "event_date": pl.date_range(
                daily_full["event_date"].min(),
                daily_full["event_date"].max(),
                "1d",
                eager=True,
            )
        }
    )
    daily_full = cal.join(daily_full, on="event_date", how="left").fill_null(0.0)

    results = {
        "config": {
            "H": H,
            "platform_models": PLATFORM_MODELS,
            "shrink_k": SHRINK_K,
            "allocation": "share preserve / pseudo-count shrink",
        },
        "folds": {},
    }

    for fold, anchor in ANCHORS.items():
        print(f"=== {fold} anchor={anchor} ===", flush=True)
        hist = daily_full.filter(pl.col("event_date") <= anchor)
        idx = pd.date_range(hist["event_date"].min(), hist["event_date"].max(), freq="D")
        hist_pd = pd.Series(hist["gmv"].to_numpy(), index=idx)

        lo_past = anchor - timedelta(days=H - 1)
        past = (
            raw.filter(pl.col("event_date").is_between(lo_past, anchor))
            .group_by("user_id")
            .agg(pl.col("gmv").sum().alias("past30"), pl.len().alias("n30"))
        )
        base = (
            pl.DataFrame({"user_id": users})
            .join(past, on="user_id", how="left")
            .fill_null(0)
        )

        tgt = None
        if fold in CV_FOLDS:
            fut = (
                raw.filter(
                    pl.col("event_date").is_between(
                        anchor + timedelta(days=1), anchor + timedelta(days=H)
                    )
                )
                .group_by("user_id")
                .agg(pl.col("gmv").sum().alias("y"))
            )
            base = base.join(fut, on="user_id", how="left").fill_null(0)
            tgt = base["y"].to_numpy()

        past30 = base["past30"].to_numpy()
        n30 = base["n30"].to_numpy()
        gprev = float(past30.sum())

        fc_daily: dict[str, np.ndarray] = {}
        totals: dict[str, float] = {}
        for m in PLATFORM_MODELS:
            if m == "drift_ratio":
                t, d = drift_ratio_total(hist, anchor)
            else:
                d = fit_platform_forecast(hist_pd, m)
                t = float(d.sum())
            fc_daily[m], totals[m] = d, t
            print(f"  {m}: total30={t/1e6:.1f}M", flush=True)

        preds = {"naive": past30.copy()}
        for m in PLATFORM_MODELS:
            preds[f"{m}__preserve"] = allocate(past30, n30, totals[m], gprev, None)
            for k in SHRINK_K:
                preds[f"{m}__shrink{k}"] = allocate(past30, n30, totals[m], gprev, k)
        if tgt is not None:
            gnext = float(tgt.sum())
            preds["oracle_preserve"] = allocate(past30, n30, gnext, gprev, None)

        fold_res = {"anchor": str(anchor), "platform_prev30": gprev}
        if tgt is not None:
            fold_res["platform_next30_actual"] = float(tgt.sum())
            fold_res["naive_rmsle_ref_eda"] = NAIVE_REF.get(fold)
            fold_res["rmsle"] = {name: rmsle(tgt, p) for name, p in preds.items()}
            fold_res["platform_forecast_total"] = totals
            if fold == "fold_end":
                pass
        else:
            fold_res["platform_forecast_total"] = totals

        # график: история 180д + прогнозы + факт (для CV фолдов)
        fig, ax = plt.subplots(figsize=(11, 4))
        tail = hist_pd.tail(180) / 1e6
        ax.plot(tail.index, tail.values, color="black", lw=0.9, label="history")
        fut_idx = pd.date_range(anchor + timedelta(days=1), periods=H, freq="D")
        for m in PLATFORM_MODELS:
            ax.plot(fut_idx, fc_daily[m] / 1e6, lw=1.0, alpha=0.85, label=m)
        if tgt is not None:
            actual_d = (
                daily_full.filter(
                    pl.col("event_date").is_between(
                        anchor + timedelta(days=1), anchor + timedelta(days=H)
                    )
                )["gmv"].to_numpy() / 1e6
            )
            ax.plot(fut_idx, actual_d, color="tab:red", lw=1.4, ls="--", label="actual")
        ax.axvline(pd.Timestamp(anchor), color="gray", ls=":", lw=0.8)
        ax.set_title(f"{fold} anchor={anchor}: platform daily GMV, M")
        ax.legend(fontsize=8, ncol=5)
        fig.tight_layout()
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG_DIR / f"exp10_{fold}.png", dpi=130)
        plt.close(fig)

        results["folds"][fold] = fold_res

    # сводная таблица по CV фолдам
    variants = sorted({v for f in CV_FOLDS for v in results["folds"][f]["rmsle"]})
    lines = ["| Вариант | " + " | ".join(CV_FOLDS) + " |", "|---|" + "---|" * len(CV_FOLDS)]
    for v in variants:
        row = [results["folds"][f]["rmsle"][v] for f in CV_FOLDS]
        best = min(row)
        cells = [f"{x:.5f}" + (" **" if abs(x - best) < 1e-12 else "") for x in row]
        lines.append(f"| {v} | " + " | ".join(cells) + " |")
    table_md = "\n".join(lines)
    results["summary_table_md"] = table_md
    print(table_md)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
