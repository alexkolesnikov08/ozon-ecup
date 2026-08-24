"""exp02 parts B, B2, C: CatBoost training, feature ablation, loss ablation,
leave-one-fold-out stability and YoY seasonal calibration.

Protocol of exp01 is reused exactly: train = fold_00+01+02, validation =
fold_03; CatBoost RMSE on z = log1p(target); lr=0.05, depth=8, l2_leaf_reg=3,
seed=42. Prediction inverse-transform expm1 once, clip >= 0.

Feature sets: the 56 base features of exp01 (read-only cache) joined by
user_id with the new exp02 features (cache data/v2/features_exp02/), plus
pct_rank_gmv30 computed HERE on the assembled full fold with a single polars
rank expression (spec block 7 - population statistic, not batch-computable):
    pct_rank_gmv30 = (rank_average(gmv_sum_30d) - 1) / (n_users - 1) in [0,1].

Writes reports/exp02_metrics.json and figures to reports/exp02_figures/.
The chosen best configuration ("decision" block of the json) is consumed by
src/exp02_submit.py.
"""

import json
import platform
import time
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from catboost import CatBoostRegressor, Pool

SEED = 42
DROP_COLS = ["anchor_date", "user_id", "target"]
FIG_DIR = Path("reports/exp02_figures")
METRICS_PATH = Path("reports/exp02_metrics.json")

BASE_REF_RMSLE = 1.70261
BASE_REF_TOL = 0.002

BLOCKS = {
    "ewma": ["ewma_gmv_hl7", "ewma_gmv_hl30", "ewma_to_ord_hl7", "ewma_to_ord_hl30"],
    "trend": ["trend_gmv_7v30", "trend_gmv_30v90", "slope_loggmv_60d"],
    "conv": ["conv_s2o", "conv_c2o", "conv_o2c"],
    "decomp": ["aov_30", "ord_days_30"],
    "due": ["due_ratio"],
    "shares": [
        "share_gmv_search_90", "share_gmv_cat_90",
        "share_gmv_search_trend", "share_gmv_cat_trend",
    ],
    "pop": ["pct_rank_gmv30"],
}

CONFIGS = {
    "base": [],
    "base_ewma_trend": ["ewma", "trend"],
    "base_conv_decomp_due_shares": ["conv", "decomp", "due", "shares"],
    "full": ["ewma", "trend", "conv", "decomp", "due", "shares", "pop"],
}

CV_FOLDS = ["fold_00", "fold_01", "fold_02", "fold_03"]

METRICS: dict = {}


def dump_metrics() -> None:
    METRICS_PATH.write_text(json.dumps(METRICS, indent=2, ensure_ascii=False))


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    lt = np.log1p(np.clip(y_true, 0, None))
    lp = np.log1p(np.clip(y_pred, 0, None))
    return float(np.sqrt(np.mean((lt - lp) ** 2)))


EXTRA_ALL = [f for b, cols in BLOCKS.items() if b != "pop" for f in cols]


def join_fold(name: str) -> pl.DataFrame:
    base_dir = Path(f"data/v2/features/{name}")
    if not base_dir.exists():
        base_dir = Path(f"data/v2/features_exp02/{name}_base")
        assert base_dir.exists(), f"no base cache for {name}"
    base = pl.read_parquet(str(base_dir / "batch_*.parquet"))
    extra = pl.read_parquet(f"data/v2/features_exp02/{name}/batch_*.parquet")
    assert base.height == extra.height == 250_000
    j = base.join(
        extra.select(["user_id", "anchor_date", *EXTRA_ALL]),
        on="user_id", how="inner", suffix="_x2",
    )
    assert j.height == base.height
    assert (j["anchor_date"] == j["anchor_date_x2"]).all(), f"{name}: anchor mismatch"
    j = j.drop("anchor_date_x2")
    n = j.height
    j = j.with_columns(
        ((pl.col("gmv_sum_30d").rank(method="average") - 1) / (n - 1))
        .cast(pl.Float64)
        .alias("pct_rank_gmv30")
    )
    return j


class View:
    def __init__(self, name: str, df: pl.DataFrame):
        self.name = name
        self.df = df
        self.y_raw = np.clip(df["target"].to_numpy(), 0, None)
        self.has_target = bool((~np.isnan(self.y_raw)).all())

    def X(self, config: str) -> np.ndarray:
        cols = config_cols(config)
        return self.df.select(cols).to_numpy()

    def y_log(self) -> np.ndarray:
        return np.log1p(self.y_raw)


def config_cols(config: str) -> list[str]:
    assert config in CONFIGS, config
    return BASE_FEATURES + [f for b in CONFIGS[config] for f in BLOCKS[b]]


def fit_catboost(X_tr, y_tr, feat_names, n_estimators, loss_function="RMSE",
                 X_val=None, y_val=None):
    model = CatBoostRegressor(
        loss_function=loss_function,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3,
        n_estimators=n_estimators,
        thread_count=-1,
        random_seed=SEED,
        verbose=0,
        allow_writing_files=False,
    )
    train_pool = Pool(X_tr, label=y_tr, feature_names=feat_names)
    eval_pool = None
    if X_val is not None:
        eval_pool = Pool(X_val, label=y_val, feature_names=feat_names)
    t0 = time.time()
    model.fit(train_pool, eval_set=eval_pool)
    return model, time.time() - t0


def predict_raw(model, X) -> np.ndarray:
    return np.clip(np.expm1(model.predict(X)), 0, None)


def main() -> None:
    global BASE_FEATURES
    t_start = time.time()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    base_check = pl.read_parquet("data/v2/features/fold_00/batch_0000.parquet")
    BASE_FEATURES = [c for c in base_check.columns if c not in DROP_COLS]
    assert len(BASE_FEATURES) == 56, len(BASE_FEATURES)

    METRICS["versions"] = {
        "python": platform.python_version(),
        "polars": pl.__version__,
        "catboost": __import__("catboost").__version__,
        "numpy": np.__version__,
        "scikit_learn": __import__("sklearn").__version__,
        "platform": platform.platform(),
    }
    METRICS["features"] = {
        "blocks": BLOCKS,
        "configs": {c: {"blocks": bs, "n_features": len(config_cols(c))}
                    for c, bs in CONFIGS.items()},
        "pct_rank_gmv30_note": (
            "computed at train time on assembled fold: "
            "(rank_average(gmv_sum_30d)-1)/(n_users-1), single polars expression"
        ),
    }

    print("loading and joining folds...", flush=True)
    views = {}
    for name in CV_FOLDS + ["fold_proxy", "fold_end"]:
        views[name] = View(name, join_fold(name))
        print(f"  {name}: {views[name].df.shape}, target_ok={views[name].has_target}",
              flush=True)
    assert all(
        tuple(views[f].df.select(BASE_FEATURES).null_count().row(0)) == (0,) * 56
        for f in CV_FOLDS
    )
    METRICS["features"]["base_n"] = len(BASE_FEATURES)

    train_views = [views[f] for f in CV_FOLDS[:3]]
    val_view = views["fold_03"]

    def concat_train(names):
        return pl.concat([views[n].df for n in names], how="vertical")

    # ---------- Stage A: feature ablation on fold_03 ----------
    ablation = []
    print("\n=== stage A: feature ablation (train fold_00..02, val fold_03) ===", flush=True)
    runs = [("base", 1000), ("base_ewma_trend", 1000),
            ("base_conv_decomp_due_shares", 1000),
            ("full", 500), ("full", 1000), ("full", 1500)]
    tr_df = concat_train(CV_FOLDS[:3])
    for cfg, n_est in runs:
        feat_cols = config_cols(cfg)
        X_tr = tr_df.select(feat_cols).to_numpy()
        y_tr = np.log1p(np.clip(tr_df["target"].to_numpy(), 0, None))
        X_va = val_view.X(cfg)
        model, ft = fit_catboost(X_tr, y_tr, feat_cols, n_est,
                                 X_val=X_va, y_val=val_view.y_log())
        score = rmsle(val_view.y_raw, predict_raw(model, X_va))
        ablation.append({"config": cfg, "n_estimators": n_est,
                         "n_features": len(feat_cols),
                         "rmsle_fold03": round(score, 5),
                         "fit_time_sec": round(ft, 1)})
        print(f"  {cfg:>30} x{n_est:>5}: RMSLE={score:.5f} ({ft:.1f}s)", flush=True)
    METRICS["ablation_fold03"] = ablation
    dump_metrics()

    base_row = next(r for r in ablation if r["config"] == "base" and r["n_estimators"] == 1000)
    dev = abs(base_row["rmsle_fold03"] - BASE_REF_RMSLE)
    METRICS["baseline_reproduction"] = {
        "exp01_reference": BASE_REF_RMSLE,
        "reproduced": base_row["rmsle_fold03"],
        "abs_deviation": round(dev, 5),
        "within_tol": dev <= BASE_REF_TOL,
    }
    if dev > BASE_REF_TOL:
        print(f"WARNING: base reproduction deviates from exp01 reference "
              f"by {dev:.5f} (> {BASE_REF_TOL})", flush=True)
    else:
        print(f"base reproduction OK: {base_row['rmsle_fold03']} (dev {dev:.5f})",
              flush=True)

    best_row = min(ablation, key=lambda r: r["rmsle_fold03"])
    best_cfg, best_n = best_row["config"], best_row["n_estimators"]
    print(f"\nbest so far: {best_cfg} x{best_n} RMSLE={best_row['rmsle_fold03']}", flush=True)

    # ---------- Stage B: leave-one-fold-out stability ----------
    print("\n=== stage B: LOFO stability (base vs best) ===", flush=True)
    lofo = {"base": [], "best": []}
    for i in range(4):
        hold = CV_FOLDS[i]
        rest = [f for j, f in enumerate(CV_FOLDS) if j != i]
        tr_df_i = concat_train(rest)
        va = views[hold]
        for tag, cfg, n_est in (("base", "base", 1000), ("best", best_cfg, best_n)):
            feat_cols = config_cols(cfg)
            model, ft = fit_catboost(
                tr_df_i.select(feat_cols).to_numpy(),
                np.log1p(np.clip(tr_df_i["target"].to_numpy(), 0, None)),
                feat_cols, n_est,
            )
            score = rmsle(va.y_raw, predict_raw(model, va.X(cfg)))
            lofo[tag].append({"fold": hold, "rmsle": round(score, 5),
                              "fit_time_sec": round(ft, 1)})
            print(f"  holdout {hold} [{tag:>4}]: RMSLE={score:.5f} ({ft:.1f}s)", flush=True)
    METRICS["lofo"] = {
        "description": f"train on other 3 folds, validate held-out; configs: base(56) vs {best_cfg}(x{best_n})",
        "base": lofo["base"], "best": lofo["best"],
    }
    dump_metrics()

    # ---------- Stage B2: loss ablation ----------
    print("\n=== stage B2: loss ablation (best features, 1000 iters, fold_03) ===", flush=True)
    losses = ["RMSE", "MAE", "Tweedie:variance_power=1.1",
              "Tweedie:variance_power=1.5", "Tweedie:variance_power=1.9"]
    loss_rows = []
    loss_models = {}
    feat_cols = config_cols(best_cfg)
    X_tr = tr_df.select(feat_cols).to_numpy()
    y_tr = np.log1p(np.clip(tr_df["target"].to_numpy(), 0, None))
    X_va = val_view.X(best_cfg)
    for loss in losses:
        model, ft = fit_catboost(X_tr, y_tr, feat_cols, 1000, loss_function=loss)
        score = rmsle(val_view.y_raw, predict_raw(model, X_va))
        loss_rows.append({"loss": loss, "rmsle_fold03": round(score, 5),
                          "fit_time_sec": round(ft, 1)})
        loss_models[loss] = model
        print(f"  {loss:>28}: RMSLE={score:.5f} ({ft:.1f}s)", flush=True)
    METRICS["loss_ablation_fold03"] = loss_rows
    best_loss_row = min(loss_rows, key=lambda r: r["rmsle_fold03"])
    best_loss = best_loss_row["loss"]
    print(f"best loss: {best_loss}", flush=True)

    # ---------- Stage C: YoY calibration on proxy fold ----------
    print("\n=== stage C: YoY seasonal calibration ===", flush=True)
    raw = pl.read_parquet("data/train.parquet", columns=["event_date", "gmv"])
    num = raw.filter(pl.col("event_date").is_between(
        date(2025, 2, 14), date(2025, 3, 15)))["gmv"].sum()
    den = raw.filter(pl.col("event_date").is_between(
        date(2025, 1, 15), date(2025, 2, 13)))["gmv"].sum()
    s_hat = float(num / den)
    ln_s = float(np.log(s_hat))
    print(f"s_hat = {s_hat:.6f} (ln={ln_s:.6f})", flush=True)

    proxy = views["fold_proxy"]
    X_px = proxy.X(best_cfg)
    model_best, _ = fit_catboost(X_tr, y_tr, feat_cols, best_n,
                                 loss_function=best_loss)
    z_pred = model_best.predict(X_px)
    z_true = np.log1p(proxy.y_raw)

    beta_grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    grid_scores = []
    for b in beta_grid:
        s = float(np.sqrt(np.mean((z_true - (z_pred + b * ln_s)) ** 2)))
        grid_scores.append(round(s, 5))
        print(f"  beta={b:.2f}: proxy RMSLE={s:.5f}", flush=True)
    beta_hat = float(beta_grid[int(np.argmin(grid_scores))])

    fine = np.arange(0.0, 1.0001, 0.02)
    fine_scores = [float(np.sqrt(np.mean((z_true - (z_pred + b * ln_s)) ** 2)))
                   for b in fine]

    combo_score_f03 = rmsle(val_view.y_raw, predict_raw(model_best, X_va))
    print(f"best combo ({best_cfg} x{best_n} {best_loss}) on fold_03: "
          f"RMSLE={combo_score_f03:.5f}", flush=True)

    METRICS["yoy"] = {
        "platform_index_definition": "s_hat = sum(gmv[2025-02-14..2025-03-15]) / sum(gmv[2025-01-15..2025-02-13]), all users",
        "s_hat": round(s_hat, 6),
        "proxy_anchor": "2025-02-14",
        "beta_grid": beta_grid,
        "rmsle_by_beta": grid_scores,
        "beta_hat": beta_hat,
        "caveats": [
            "proxy users have only 45 days of history vs production anchors",
            "weekday alignment differs between 2025 and 2026 windows",
            "holidays shift relative to weekdays year-to-year",
            "beta is a directional estimate picked on the proxy, not a guarantee",
        ],
    }
    METRICS["decision"] = {
        "feature_config": best_cfg,
        "blocks": CONFIGS[best_cfg],
        "n_features": len(feat_cols),
        "n_estimators": best_n,
        "loss": best_loss,
        "beta": beta_hat,
        "s_hat": round(s_hat, 6),
    }
    METRICS["best_combo_fold03_rmsle"] = round(combo_score_f03, 5)
    METRICS["runtime_train_sec"] = round(time.time() - t_start, 1)
    dump_metrics()

    # ---------- figures ----------
    imp = model_best.get_feature_importance()
    order = np.argsort(imp)[::-1][:20][::-1]
    names = [feat_cols[i] for i in order]
    plt.figure(figsize=(9, 7))
    plt.barh(names, [imp[i] for i in order], color="steelblue")
    plt.xlabel("feature importance, %")
    plt.title(f"exp02: top-20 features ({best_cfg}, x{best_n}, {best_loss.split(':')[0]})")
    plt.grid(True, axis="x", alpha=0.35)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "exp02_feature_importance.png", dpi=150)
    plt.close()

    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(val_view.y_raw), size=20_000, replace=False)
    pred_sample = predict_raw(model_best, X_va[idx])
    pred_log = np.log1p(pred_sample)
    actual_log = np.log1p(val_view.y_raw[idx])
    lims = [min(actual_log.min(), pred_log.min()),
            max(actual_log.max(), pred_log.max())]
    plt.figure(figsize=(7, 7))
    plt.scatter(actual_log, pred_log, s=4, alpha=0.25, color="darkorange",
                edgecolors="none")
    plt.plot(lims, lims, "k--", lw=1, label="y = x")
    plt.xlabel("actual log1p(gmv next 30d)")
    plt.ylabel("predicted log1p(gmv)")
    plt.title(f"exp02: pred vs actual, fold_03 sample 20k "
              f"({best_cfg}, x{best_n}, {best_loss.split(':')[0]})")
    plt.legend()
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "exp02_pred_vs_actual.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(fine, fine_scores, lw=2, color="seagreen", label="fine grid")
    plt.scatter(beta_grid, grid_scores, s=55, color="darkred", zorder=3,
                label="coarse grid")
    plt.axvline(beta_hat, ls="--", lw=1, color="gray",
                label=f"beta_hat={beta_hat}")
    plt.xlabel("beta (calibration exponent, y_cal = y * s_hat^beta)")
    plt.ylabel("RMSLE on proxy fold (anchor 2025-02-14)")
    plt.title(f"exp02: YoY calibration, s_hat={s_hat:.4f}")
    plt.legend()
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "exp02_beta_calibration.png", dpi=150)
    plt.close()

    print(f"\nTRAIN DONE in {(time.time() - t_start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
