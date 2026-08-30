"""exp02.5: honest attribution of teammate's feature blocks under our protocol.

Teammate repo (github.com/Rafaildavar/ozon-ecup) ships a richer feature matrix;
his own block ablation was an in-sample refit (fold_03 inside train), so gains
are not attributable. This script re-scores his blocks on OUR exp02 protocol:
train fold_00..02 -> holdout fold_03, CatBoost RMSE on z=log1p(target),
lr=0.05, depth=8, l2_leaf_reg=3, seed=42, 1000 iters.

Configs:
- ours66            : our exp02 matrix (control, must reproduce ~1.69277)
- theirs141         : his full matrix as-is (base70 + x_* + pca32 + btyd)
- ours+pca_btyd     : ours66 + his pca_* + BTYD blocks (no semantic overlap)
- ours+all_new      : ours66 + pca/btyd + deduped x_*/base extras (stoplist of
                      semantic duplicates of our exp02 blocks removed)

Writes reports/exp025_ablation.json. Run from repo root:
    .venv/bin/python src/exp025_ablation.py
"""

import json
import time
from pathlib import Path

import numpy as np
import polars as pl
from catboost import CatBoostRegressor

SEED = 42
CV_FOLDS = ["fold_00", "fold_01", "fold_02", "fold_03"]
TRAIN_FOLDS = CV_FOLDS[:3]
KEYS = {"anchor_date", "user_id", "target"}
REPORT = Path("reports/exp025_ablation.json")

OUR_CONV_DECOMP_DUE_SHARES = [
    "conv_s2o", "conv_c2o", "conv_o2c",
    "aov_30", "ord_days_30",
    "due_ratio",
    "share_gmv_search_90", "share_gmv_cat_90",
    "share_gmv_search_trend", "share_gmv_cat_trend",
]

BTYD_FILL_ZERO = {
    "bgnbd_tx", "bgnbd_en30", "eb_lambda_n30", "bgnbd_e_gmv30", "eb_e_gmv30",
}

STOPLIST_SEMANTIC_DUPS = {
    "x_aov_30d", "x_due_ratio",
    "x_share_gmv_search_30d", "x_share_gmv_cat_30d",
}


def rmsle(y_raw: np.ndarray, z_pred: np.ndarray) -> float:
    lt = np.log1p(np.clip(y_raw, 0, None))
    lp = np.clip(z_pred, None, 30.0)
    return float(np.sqrt(np.mean((lt - lp) ** 2)))


def load_ours(name: str) -> pl.DataFrame:
    base = pl.read_parquet(f"data/v2/features/{name}/batch_*.parquet")
    extra = pl.read_parquet(f"data/v2/features_exp02/{name}/batch_*.parquet")
    j = base.join(
        extra.select(["user_id", *OUR_CONV_DECOMP_DUE_SHARES]),
        on="user_id", how="inner", suffix="_x2",
    )
    assert j.height == 250_000
    return j


def load_theirs(name: str) -> pl.DataFrame:
    feats = pl.read_parquet(f"data/v2/features_ext/{name}/batch_*.parquet")
    btyd = pl.read_parquet(f"data/v2/features_bgnbd/{name}.parquet")
    j = feats.join(btyd, on=["anchor_date", "user_id"], how="left")
    return j.with_columns([
        pl.col(c).fill_null(0.0) if c in BTYD_FILL_ZERO else pl.col(c).fill_null(-1.0)
        for c in btyd.columns if c not in KEYS
    ])


def fit_score(X_tr, y_tr, X_va, y_va_raw):
    model = CatBoostRegressor(
        loss_function="RMSE", learning_rate=0.05, depth=8, l2_leaf_reg=3,
        n_estimators=1000, thread_count=-1, random_seed=SEED, verbose=0,
        allow_writing_files=False,
    )
    t0 = time.time()
    model.fit(X_tr, y_tr)
    z_va = model.predict(X_va)
    return rmsle(y_va_raw, z_va), time.time() - t0


def main() -> None:
    t00 = time.time()
    print("loading fold tables...", flush=True)
    ours = {f: load_ours(f) for f in CV_FOLDS}
    theirs = {f: load_theirs(f) for f in CV_FOLDS}
    for f in CV_FOLDS:
        assert ours[f]["user_id"].sort().equals(theirs[f]["user_id"].sort()), f

    ours_feats = [c for c in ours["fold_00"].columns if c not in KEYS]
    theirs_feats = [c for c in theirs["fold_00"].columns if c not in KEYS]
    pca_btyd = [c for c in theirs_feats
                if c.startswith(("pca_", "bgnbd_", "eb_", "gg_"))]
    dup_names = set(ours_feats)
    new_x = [c for c in theirs_feats
             if c not in set(pca_btyd)
             and c not in STOPLIST_SEMANTIC_DUPS
             and c not in dup_names]
    print(f"ours={len(ours_feats)} theirs={len(theirs_feats)} "
          f"pca_btyd={len(pca_btyd)} new_extra={len(new_x)}", flush=True)

    def mats(cols_from_ours, cols_from_theirs):
        X, Y = {}, {}
        for f in CV_FOLDS:
            o = ours[f].select(["user_id", *cols_from_ours, "target"])
            t = theirs[f].select(["user_id", *cols_from_theirs])
            j = o.join(t, on="user_id", how="inner")
            assert j.height == 250_000, (f, j.height)
            feat_cols = cols_from_ours + cols_from_theirs
            X[f] = j.select(feat_cols).to_numpy().astype(np.float64)
            Y[f] = np.clip(j["target"].to_numpy(), 0, None)
        return X, Y

    configs = {
        "ours66": (ours_feats, []),
        "ours_pca_btyd": (ours_feats, pca_btyd),
        "ours_all_new": (ours_feats, pca_btyd + new_x),
        "theirs_all": ([], theirs_feats),
    }

    results = {}
    for cfg, (cols_o, cols_t) in configs.items():
        X, Y = mats(cols_o, list(dict.fromkeys(cols_t)))
        Xtr = np.concatenate([X[f] for f in TRAIN_FOLDS])
        ytr = np.log1p(np.concatenate([Y[f] for f in TRAIN_FOLDS]))
        score, ft = fit_score(Xtr, ytr, X["fold_03"], Y["fold_03"])
        results[cfg] = {
            "n_features": int(X["fold_03"].shape[1]),
            "rmsle_fold03": round(score, 5),
            "fit_time_sec": round(ft, 1),
        }
        print(f"  {cfg:>16} ({results[cfg]['n_features']:>3} feats): "
              f"RMSLE={score:.5f} ({ft:.0f}s)", flush=True)

    ref = 1.69277
    print(f"\ncontrol check: ours66 vs exp02 reference {ref}: "
          f"delta {results['ours66']['rmsle_fold03'] - ref:+.5f}", flush=True)
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps({
        "protocol": "train fold_00..02 -> fold_03 holdout; catboost rmse-z "
                    "lr0.05 d8 l2_3 seed42 x1000 (exp02 parity)",
        "exp02_reference_fold03": ref,
        "stoplist_semantic_dups": sorted(STOPLIST_SEMANTIC_DUPS),
        "results": results,
        "runtime_min": round((time.time() - t00) / 60, 1),
    }, indent=2, ensure_ascii=False))
    print(f"report -> {REPORT}", flush=True)


if __name__ == "__main__":
    main()
