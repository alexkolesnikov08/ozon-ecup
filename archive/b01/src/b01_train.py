"""b01 part B: honest OOF (LOFO) predictions with the accepted exp02 config
(66 features = 56 base + conv/decomp/due/shares extras), plus pseudo-anchor
predictions and the final refit predictions.

Protocol = exp02 LOFO exactly: for each CV fold, CatBoost (RMSE on
z=log1p(target), lr=0.05, depth=8, l2_leaf_reg=3, seed=42, 1000 trees) is
trained on the other three folds and predicts the held-out fold. The model
trained on fold_00..02 additionally predicts the pseudo-anchor 2025-02-13
(features from data/v1/b01_pseudo/pseudo_anchor, built by src/b01_features.py).
Finally a refit on all four folds predicts fold_end and fold_03 (in-sample).

Artifacts (data/v2/b01_pseudo/):
- oof_pool.parquet        fold, anchor_date, user_id, z_pred, target   (LOFO)
- pseudo_pool.parquet     anchor_date, user_id, z_pred, target           (pseudo)
- fold_end_pred.parquet   user_id, z_pred                                (refit model)
- refit_fold03.parquet    user_id, z_pred, target                        (refit, in-sample)
- train_times.json        per-stage timings

Idempotent: stages with existing artifacts are skipped (models are retrained
only if a downstream stage needs them and they are not in memory).
"""

import json
import time
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from catboost import CatBoostRegressor

SEED = 42
N_ESTIMATORS = 1000
DROP_COLS = ["anchor_date", "user_id", "target"]
CV_FOLDS = ["fold_00", "fold_01", "fold_02", "fold_03"]

EXTRA_ACCEPTED = [
    "conv_s2o", "conv_c2o", "conv_o2c",
    "aov_30", "ord_days_30",
    "due_ratio",
    "share_gmv_search_90", "share_gmv_cat_90",
    "share_gmv_search_trend", "share_gmv_cat_trend",
]

ART_DIR = Path("data/v2/b01_pseudo")
PSEUDO_ANCHOR = date(2025, 2, 13)

REF_LOFO = {
    "fold_00": 1.81545,
    "fold_01": 1.77420,
    "fold_02": 1.71090,
    "fold_03": 1.69277,
}


def rmsle_z(y_true: np.ndarray, z_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - z_pred) ** 2)))


def join_fold(name: str) -> pl.DataFrame:
    base = pl.read_parquet(f"data/v2/features/{name}/batch_*.parquet")
    extra = pl.read_parquet(
        f"data/v2/features_exp02/{name}/batch_*.parquet",
        columns=["user_id", "anchor_date", *EXTRA_ACCEPTED],
    )
    assert base.height == extra.height == 250_000, name
    j = base.join(extra, on="user_id", how="inner", suffix="_x2")
    assert j.height == base.height
    assert (j["anchor_date"] == j["anchor_date_x2"]).all(), f"{name}: anchor mismatch"
    return j.drop("anchor_date_x2")


def new_model() -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function="RMSE",
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3,
        n_estimators=N_ESTIMATORS,
        thread_count=-1,
        random_seed=SEED,
        verbose=0,
        allow_writing_files=False,
    )


def fit_predict(X_tr, y_tr, X_va) -> tuple[CatBoostRegressor, np.ndarray, float]:
    m = new_model()
    t0 = time.time()
    m.fit(X_tr, y_tr)
    dt = time.time() - t0
    return m, m.predict(X_va), dt


def main() -> None:
    t_start = time.time()
    ART_DIR.mkdir(parents=True, exist_ok=True)
    times: dict = {}

    print("loading folds...", flush=True)
    feats = {f: join_fold(f) for f in CV_FOLDS}
    # joined frame already holds all 66 features (56 base + 10 accepted extras)
    FEAT_COLS = [c for c in feats["fold_00"].columns if c not in DROP_COLS]
    assert len(FEAT_COLS) == 66, len(FEAT_COLS)
    assert set(EXTRA_ACCEPTED).issubset(FEAT_COLS)
    for f in CV_FOLDS:
        nulls = feats[f].select(
            [pl.col(c).is_null().sum().alias(c) for c in FEAT_COLS]
        ).row(0)
        allowed = {"conv_s2o", "conv_c2o", "conv_o2c", "due_ratio"}
        bad = {c: v for c, v in zip(FEAT_COLS, nulls) if v > 0 and c not in allowed}
        assert not bad, f"{f}: unexpected nulls {bad}"
    print(f"66 features assembled; anchors: "
          f"{ {f: str(feats[f]['anchor_date'][0]) for f in CV_FOLDS} }", flush=True)

    oof_path = ART_DIR / "oof_pool.parquet"
    pseudo_path = ART_DIR / "pseudo_pool.parquet"
    fend_path = ART_DIR / "fold_end_pred.parquet"
    rf03_path = ART_DIR / "refit_fold03.parquet"

    model_m3: CatBoostRegressor | None = None

    # ---------- stage 1: LOFO OOF ----------
    if oof_path.exists():
        print("stage 1 (LOFO OOF): exists, skip", flush=True)
    else:
        pools = []
        for i, hold in enumerate(CV_FOLDS):
            rest = [f for f in CV_FOLDS if f != hold]
            tr_df = pl.concat([feats[f] for f in rest], how="vertical")
            va_df = feats[hold]
            X_tr = tr_df.select(FEAT_COLS).to_numpy()
            y_tr = np.log1p(np.clip(tr_df["target"].to_numpy(), 0, None))
            X_va = va_df.select(FEAT_COLS).to_numpy()
            m, z_va, dt = fit_predict(X_tr, y_tr, X_va)
            if hold == "fold_03":
                model_m3 = m
            z_true = np.log1p(np.clip(va_df["target"].to_numpy(), 0, None))
            s = rmsle_z(z_true, z_va)
            ref = REF_LOFO[hold]
            print(f"  holdout {hold}: RMSLE={s:.5f} "
                  f"(exp02 ref {ref:.5f}, dev {abs(s - ref):.5f}) [{dt:.1f}s]", flush=True)
            times[f"fit_lofo_{hold}_sec"] = round(dt, 1)
            pools.append(pl.DataFrame({
                "fold": pl.Series([hold] * va_df.height, dtype=pl.String),
                "anchor_date": va_df["anchor_date"],
                "user_id": va_df["user_id"],
                "z_pred": z_va.astype(np.float64),
                "target": va_df["target"].to_numpy().astype(np.float64),
            }))
        oof = pl.concat(pools, how="vertical")
        oof.write_parquet(oof_path)
        mean_s = float(np.mean([
            rmsle_z(
                np.log1p(np.clip(oof.filter(pl.col("fold") == f)["target"].to_numpy(), 0, None)),
                oof.filter(pl.col("fold") == f)["z_pred"].to_numpy(),
            ) for f in CV_FOLDS
        ]))
        ref_mean = float(np.mean(list(REF_LOFO.values())))
        print(f"stage 1 done: mean LOFO RMSLE={mean_s:.5f} (ref {ref_mean:.5f})", flush=True)
        times["lofo_mean_rmsle_reproduced"] = round(mean_s, 5)

    # ---------- stage 2: pseudo-anchor predictions (model on fold_00..02) ----------
    if pseudo_path.exists():
        print("stage 2 (pseudo predict): exists, skip", flush=True)
    else:
        pseudo_feats = pl.read_parquet(str(ART_DIR / "pseudo_anchor" / "batch_*.parquet"))
        assert pseudo_feats.height == 250_000
        assert (pseudo_feats["anchor_date"][0] == PSEUDO_ANCHOR)
        X_px = pseudo_feats.select(FEAT_COLS).to_numpy()
        nulls = pseudo_feats.select(
            [pl.col(c).is_null().sum().alias(c) for c in FEAT_COLS]
        ).row(0)
        allowed = {"conv_s2o", "conv_c2o", "conv_o2c", "due_ratio"}
        bad = {c: v for c, v in zip(FEAT_COLS, nulls) if v > 0 and c not in allowed}
        assert not bad, f"pseudo: unexpected nulls {bad}"
        if model_m3 is None:
            tr_df = pl.concat([feats[f] for f in CV_FOLDS[:3]], how="vertical")
            X_tr = tr_df.select(FEAT_COLS).to_numpy()
            y_tr = np.log1p(np.clip(tr_df["target"].to_numpy(), 0, None))
            model_m3, _, dt = fit_predict(X_tr, y_tr, X_px)
            times["fit_refold_m3_sec"] = round(dt, 1)
            print(f"  m3 retrained ({dt:.1f}s)", flush=True)
        z_px = model_m3.predict(X_px).astype(np.float64)
        pl.DataFrame({
            "anchor_date": pseudo_feats["anchor_date"],
            "user_id": pseudo_feats["user_id"],
            "z_pred": z_px,
            "target": pseudo_feats["target"].to_numpy().astype(np.float64),
        }).write_parquet(pseudo_path)
        s_px = rmsle_z(np.log1p(np.clip(pseudo_feats["target"].to_numpy(), 0, None)), z_px)
        print(f"stage 2 done: pseudo RMSLE={s_px:.5f}", flush=True)
        times["pseudo_raw_rmsle"] = round(s_px, 5)

    # ---------- stage 3: refit on all 4 folds -> fold_end + fold_03(in-sample) ----------
    if fend_path.exists() and rf03_path.exists():
        print("stage 3 (refit): exists, skip", flush=True)
    else:
        tr_all = pl.concat([feats[f] for f in CV_FOLDS], how="vertical")
        X_tr = tr_all.select(FEAT_COLS).to_numpy()
        y_tr = np.log1p(np.clip(tr_all["target"].to_numpy(), 0, None))
        end_df = join_fold("fold_end")
        X_end = end_df.select(FEAT_COLS).to_numpy()
        val_df = feats["fold_03"]
        X_v3 = val_df.select(FEAT_COLS).to_numpy()
        m, z_end, dt = fit_predict(X_tr, y_tr, np.vstack([X_end, X_v3]))
        z_end, z_v3 = z_end[:X_end.shape[0]], z_end[X_end.shape[0]:]
        s_insample = rmsle_z(
            np.log1p(np.clip(val_df["target"].to_numpy(), 0, None)), z_v3
        )
        print(f"stage 3 done: refit fit {dt:.1f}s, fold_03 in-sample RMSLE={s_insample:.5f}",
              flush=True)
        times["fit_refit_sec"] = round(dt, 1)
        times["refit_fold03_insample_rmsle_raw"] = round(s_insample, 5)
        pl.DataFrame({"user_id": end_df["user_id"], "z_pred": z_end.astype(np.float64)}
                     ).write_parquet(fend_path)
        pl.DataFrame({
            "user_id": val_df["user_id"],
            "z_pred": z_v3.astype(np.float64),
            "target": val_df["target"].to_numpy().astype(np.float64),
        }).write_parquet(rf03_path)

    times["total_sec"] = round(time.time() - t_start, 1)
    (ART_DIR / "train_times.json").write_text(json.dumps(times, indent=2))
    print(f"TRAIN DONE in {(time.time() - t_start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
