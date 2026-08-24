"""b03 step 4: final refit on all 4 CV folds -> fold_end -> submission.

Gated by the "decision" block of reports/b03_metrics.json (written by
src/b03_train.py). Only runs when decision["adopted"] is true and the variant
is one of ii/iii; otherwise exits without writing a submission.

Variant iii: targets normalised per anchor with strict causal M(anchor)
(z = log1p(target/M)), fold_end prediction denormalised: y = expm1(z)*M_end.
Variant ii: raw targets, raw predictions. Config fixed as exp02: RMSE on z,
lr=0.05, depth=8, l2_leaf_reg=3, seed=42, 1000 iters.

Submission asserts mirror archive/exp02/src/exp02_submit.py.
"""

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from catboost import CatBoostRegressor  # noqa: E402

from b03_common import (  # noqa: E402
    CV_FOLDS, METRICS_PATH, OUT_DIR, SEED, SUB_PATH,
)
from b03_features import FEATURES_66_ORDER  # noqa: E402
from b03_train import (  # noqa: E402
    FEAT_NAMES, Xtr_df, Ytr_log1p, Ytr_norm, add_pct_rank, join_ref,
)

METRICS = json.loads(METRICS_PATH.read_text())


def join_b03(name: str) -> pl.DataFrame:
    j = pl.read_parquet(str(OUT_DIR / name / "batch_*.parquet")).select(
        ["anchor_date", "user_id", *FEATURES_66_ORDER, "target"])
    assert j.height == 250_000
    return add_pct_rank(j)


def main() -> None:
    t0 = time.time()
    dec = METRICS.get("decision", {})
    if not dec.get("adopted"):
        print(f"decision not adopted ({dec.get('reason')}): "
              f"no b03 submission written", flush=True)
        return
    kind = dec["variant"]
    assert kind in ("ii", "iii"), f"unsupported variant {kind}"
    print(f"decision adopted: variant={kind} ({dec['reason']})", flush=True)

    idx_summary = json.loads((OUT_DIR / "index_summary.json").read_text())
    M_strict = {f: idx_summary["M"][f]["strict"]["log_M"]
                for f in [*CV_FOLDS, "fold_end"]}

    train_df = pl.concat([join_b03(f) for f in CV_FOLDS], how="vertical")
    end_df = join_b03("fold_end")
    X_tr = Xtr_df(train_df)
    y_tr = (Ytr_norm(train_df, {k: v for k, v in M_strict.items() if k != "fold_end"})
            if kind == "iii" else Ytr_log1p(train_df))
    X_end = end_df.select(FEAT_NAMES).to_numpy()
    print(f"train: {train_df.shape}, fold_end: {end_df.shape}", flush=True)

    model = CatBoostRegressor(
        loss_function="RMSE",
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3,
        n_estimators=1000,
        thread_count=-1,
        random_seed=SEED,
        verbose=0,
        allow_writing_files=False,
    )
    model.fit(X_tr, y_tr)
    fit_time = time.time() - t0
    print(f"final refit ({kind}, 4 folds, 1M rows) done in {fit_time:.1f}s", flush=True)

    # honest check of the refit model on fold_03 (protocol of exp01/exp02)
    val_df = join_b03("fold_03")
    zv = model.predict(val_df.select(FEAT_NAMES).to_numpy())
    if kind == "iii":
        pred_val = np.clip(np.expm1(zv) * float(np.exp(M_strict["fold_03"])), 0, None)
    else:
        pred_val = np.clip(np.expm1(zv), 0, None)
    lt = np.log1p(np.clip(val_df["target"].to_numpy(), 0, None))
    lp = np.log1p(pred_val)
    refit_score = float(np.sqrt(np.mean((lt - lp) ** 2)))
    print(f"refit-model RMSLE on fold_03: {refit_score:.5f}", flush=True)

    zend = model.predict(X_end)
    if kind == "iii":
        log_M_end = M_strict["fold_end"]
        pred = np.clip(np.expm1(zend + 0.0) * float(np.exp(log_M_end)), 0, None)
        print(f"fold_end denormalised with strict M: log_M={log_M_end:.6f} "
              f"(M={np.exp(log_M_end):.6f})", flush=True)
    else:
        pred = np.clip(np.expm1(zend), 0, None)
    pred = pred.astype(np.float64)
    assert np.isfinite(pred).all() and (pred >= 0).all()

    sample = pl.read_csv("sample_submit.csv")
    sub = pl.DataFrame({"user_id": end_df["user_id"], "predict": pred}).join(
        sample.select("user_id"), on="user_id", how="semi")
    assert set(sub["user_id"]) == set(sample["user_id"]), "user_id set mismatch"
    sub = sub.join(sample.select("user_id").with_row_index("__ord"), on="user_id")
    sub = sub.sort("__ord").drop("__ord")
    assert sub.height == 250_000 and sub.width == 2

    SUB_PATH.parent.mkdir(exist_ok=True)
    sub.write_csv(SUB_PATH)

    chk = pl.read_csv(SUB_PATH)
    assert chk.height == 250_000 and chk.width == 2
    assert chk.columns == ["user_id", "predict"]
    assert chk.schema["predict"] == pl.Float64
    assert chk["predict"].is_finite().all() and (chk["predict"] >= 0).all()
    print(f"submission saved: {SUB_PATH} ({chk.height} rows, "
          f"mean={chk['predict'].mean():.2f}, median={chk['predict'].median():.4f})",
          flush=True)

    METRICS["submission"] = {
        "path": str(SUB_PATH), "variant": kind,
        "n_features": len(FEAT_NAMES), "n_estimators": 1000, "loss": "RMSE(log1p)",
        **({"log_M_fold_end": M_strict["fold_end"],
            "M_fold_end": round(float(np.exp(M_strict["fold_end"])), 6)}
           if kind == "iii" else {}),
        "fit_time_sec": round(fit_time, 1),
        "refit_rmsle_fold03": round(refit_score, 5),
        "exp02_refit_rmsle_fold03": 1.67165,
        "pred_mean": round(float(chk["predict"].mean()), 4),
        "pred_median": round(float(chk["predict"].median()), 4),
        "pred_min": round(float(chk["predict"].min()), 6),
        "pred_max": round(float(chk["predict"].max()), 2),
        "rows": chk.height, "columns": chk.columns,
        "runtime_total_sec": round(time.time() - t0, 1),
    }
    METRICS_PATH.write_text(json.dumps(METRICS, indent=2, ensure_ascii=False))
    print("metrics updated", flush=True)


if __name__ == "__main__":
    main()
