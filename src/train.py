"""CatBoost training series on windowed features (exp01).

Train = fold_00..02 concat, validation = fold_03. Target log1p-transformed,
loss RMSE -> val-RMSE equals RMSLE in original space. Final RMSLE computed
after expm1 inverse transform.
"""

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from catboost import CatBoostRegressor, Pool

DROP_COLS = ["anchor_date", "user_id", "target"]
SEED = 42
N_ESTIMATORS_SERIES = [100, 300, 500, 1000]

FIG_DIR = Path("reports/figures")


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    lt = np.log1p(np.clip(y_true, 0, None))
    lp = np.log1p(np.clip(y_pred, 0, None))
    return float(np.sqrt(np.mean((lt - lp) ** 2)))


def load_fold(name: str) -> pl.DataFrame:
    return pl.read_parquet(f"data/v2/features/{name}/batch_*.parquet")


def xy(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = df.drop(DROP_COLS).to_numpy()
    y = np.log1p(np.clip(df["target"].to_numpy(), 0, None))
    return X, y


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("loading folds...", flush=True)
    train_df = pl.concat(
        [load_fold(f"fold_{i:02d}") for i in range(3)], how="vertical"
    )
    val_df = load_fold("fold_03")
    feature_names = [c for c in train_df.columns if c not in DROP_COLS]
    print(f"train: {train_df.shape}, val: {val_df.shape}, features: {len(feature_names)}")

    X_tr, y_tr = xy(train_df)
    X_val, y_val_log = xy(val_df)
    y_val_raw = np.clip(val_df["target"].to_numpy(), 0, None)

    train_pool = Pool(X_tr, label=y_tr, feature_names=feature_names)
    val_pool = Pool(X_val, label=y_val_log, feature_names=feature_names)

    results = {}
    curves = {}
    best = {"rmsle": float("inf"), "n_estimators": None, "model": None}

    for n_est in N_ESTIMATORS_SERIES:
        model = CatBoostRegressor(
            loss_function="RMSE",
            learning_rate=0.05,
            depth=8,
            l2_leaf_reg=3,
            n_estimators=n_est,
            thread_count=-1,
            random_seed=SEED,
            verbose=0,
        )
        t0 = time.time()
        model.fit(train_pool, eval_set=val_pool)
        fit_time = time.time() - t0

        evals = model.get_evals_result()
        learn_rmse = evals["learn"]["RMSE"]
        val_rmse = evals["validation"]["RMSE"]
        iters = np.arange(1, len(val_rmse) + 1)
        curves[n_est] = (iters, learn_rmse, val_rmse)

        pred_raw = np.clip(np.expm1(model.predict(X_val)), 0, None)
        score = rmsle(y_val_raw, pred_raw)
        results[str(n_est)] = {
            "rmsle_fold03": round(score, 5),
            "fit_time_sec": round(fit_time, 1),
            "best_iteration": len(val_rmse),
            "final_val_rmse_logspace": round(float(val_rmse[-1]), 5),
        }
        print(
            f"n_estimators={n_est:>5}: RMSLE={score:.5f} "
            f"(val RMSE(log)={val_rmse[-1]:.5f}) fit {fit_time:.1f}s",
            flush=True,
        )
        if score < best["rmsle"]:
            best.update(rmsle=score, n_estimators=n_est, model=model)

    print(f"\nBEST: n_estimators={best['n_estimators']} RMSLE={best['rmsle']:.5f}")

    with open("reports/exp01_metrics.json", "w") as fh:
        json.dump(
            {
                "config": {
                    "loss": "RMSE on log1p(target)",
                    "learning_rate": 0.05,
                    "depth": 8,
                    "l2_leaf_reg": 3,
                    "random_seed": SEED,
                    "train": "fold_00+fold_01+fold_02",
                    "validation": "fold_03",
                },
                "results": results,
                "best": {"n_estimators": best["n_estimators"], "rmsle_fold03": round(best["rmsle"], 5)},
                "baselines_exp00": json.load(open("reports/exp00_baselines.json"))
                if Path("reports/exp00_baselines.json").exists()
                else None,
            },
            fh,
            indent=2,
        )

    # --- plot 1: loss curves ---
    plt.figure(figsize=(10, 6))
    for n_est, (iters, learn_rmse, val_rmse) in curves.items():
        color = plt.cm.viridis(N_ESTIMATORS_SERIES.index(n_est) / 3)
        plt.plot(iters, val_rmse, color=color, lw=2.2, label=f"val RMSLE, {n_est} iters")
        plt.plot(iters, learn_rmse, color=color, lw=1.0, alpha=0.55, linestyle="--")
    plt.xlabel("iteration")
    plt.ylabel("RMSE in log1p space (= RMSLE)")
    plt.title("exp01: CatBoost series — learn (thin dashed) vs val (thick) RMSLE, fold_03")
    plt.legend()
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "exp01_loss_curves.png", dpi=150)
    plt.close()

    # --- plot 2: top-20 feature importance of best model ---
    imp = best["model"].get_feature_importance()
    order = np.argsort(imp)[::-1][:20][::-1]
    names = [feature_names[i] for i in order]
    plt.figure(figsize=(9, 7))
    plt.barh(names, [imp[i] for i in order], color="steelblue")
    plt.xlabel("feature importance, %")
    plt.title(f"exp01: top-20 features (best model, {best['n_estimators']} iters)")
    plt.grid(True, axis="x", alpha=0.35)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "exp01_feature_importance.png", dpi=150)
    plt.close()

    # --- plot 3: pred vs actual scatter (log1p space), 20k sample ---
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(y_val_raw), size=20_000, replace=False)
    pred_raw_sample = np.clip(np.expm1(best["model"].predict(X_val[idx])), 0, None)
    pred_log = np.log1p(pred_raw_sample)
    actual_log = np.log1p(y_val_raw[idx])
    lims = [
        min(actual_log.min(), pred_log.min()),
        max(actual_log.max(), pred_log.max()),
    ]
    plt.figure(figsize=(7, 7))
    plt.scatter(actual_log, pred_log, s=4, alpha=0.25, color="darkorange", edgecolors="none")
    plt.plot(lims, lims, "k--", lw=1, label="y = x")
    plt.xlabel("actual log1p(gmv next 30d)")
    plt.ylabel("predicted log1p(gmv)")
    plt.title(f"exp01: pred vs actual, fold_03 sample 20k ({best['n_estimators']} iters)")
    plt.legend()
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "exp01_pred_vs_actual.png", dpi=150)
    plt.close()

    best["model"].save_model("reports/exp01_best_model.cbm")
    print("plots and metrics saved")


if __name__ == "__main__":
    main()
