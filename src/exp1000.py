"""Experiment factory v2: 1000+ diverse configs, parallel for 24c/96G RAM.

Families (id order == priority order, most promising on THIS data first):

  hurdle   P(buy|X) x E[z|buy,X] two-stage decomposition - direct attack on
           the 46% zero mass of the target (EDA-confirmed intent gradient);
  seeds    multi-seed averaging of strong members (variance reduction);
  pseudo   self-training: fit -> pseudo-label fold_end -> refit with it;
  qmap     quantile-mapping of predictions onto train target distribution;
  core     model x hyperparams x loss(quantile/huber) x anchor-recency weights
           x z-winsorizing x feature-block grid.

Protocol: train fold_00..02, early stopping on fold_02, holdout fold_03
(never used for fitting/stopping), post-proc micro-grid (shift c x threshold
tau x optional quantile-map k-scale) chosen on fold_03 and applied to
fold_end - disclosed and identical across runs. Blend family recombines
cached prediction vectors without refitting.

Parallelism: ProcessPoolExecutor; each worker loads data once and fits with
--threads threads; --workers processes run specs concurrently.

Commands (repo root):
    .venv/bin/python src/exp1000.py build
    .venv/bin/python src/exp1000.py status
    .venv/bin/python src/exp1000.py run --top 40 --workers 5 --threads 5
    .venv/bin/python src/exp1000.py run --ids 0,3,7
    .venv/bin/python src/exp1000.py run --family blends
    .venv/bin/python src/exp1000.py run --all --max-minutes 600

Artifacts:
    data/v2/exp1000/index.json     registry (id order == priority)
    reports/exp1000_results.jsonl  append-only results
    data/v2/exp1000/preds/{id}.npz prediction caches for blends
    submissions/exp{ID:04d}_*.csv  per-experiment submissions
"""

import argparse
import itertools
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import rankdata

FEAT_DIR = Path("data/v2/features_ext")
BTYD_DIR = Path("data/v2/features_bgnbd")
IDX_PATH = Path("data/v2/exp1000/index.json")
RES_PATH = Path("reports/exp1000_results.jsonl")
PRED_DIR = Path("data/v2/exp1000/preds")
SUB_DIR = Path("submissions")

SEED = 42
TRAIN_FOLDS = ["fold_00", "fold_01", "fold_02"]
ES_FOLD = "fold_02"
HOLDOUT = "fold_03"

META_COLS = {"anchor_date", "user_id", "target"}
BTYD_FILL_ZERO = [
    "bgnbd_tx", "bgnbd_en30", "eb_lambda_n30", "bgnbd_e_gmv30", "eb_e_gmv30",
]

BLOCKS = {
    "pca_btyd": lambda c: list(c),
    "pca32": lambda c: [x for x in c if not x.startswith(("bgnbd_", "eb_"))],
    "ext": lambda c: [x for x in c if not x.startswith(("pca_", "bgnbd_", "eb_"))],
    "base": lambda c: [x for x in c if not x.startswith(("x_", "pca_", "bgnbd_", "eb_"))],
}
BLOCK_PRIO = {"pca_btyd": 0, "pca32": 4, "ext": 8, "base": 14}

MODEL_GRID = {
    "lgbm": [
        {"learning_rate": lr, "num_leaves": nl, "min_child_samples": mc,
         "feature_fraction": ff, "n_estimators": n_est}
        for lr, nl, mc, ff, n_est in (
            (0.04, 63, 100, 0.85, 2500), (0.04, 63, 300, 0.85, 2500),
            (0.04, 127, 100, 0.85, 2500), (0.04, 127, 300, 0.85, 2500),
            (0.03, 63, 300, 0.85, 3000), (0.03, 127, 300, 0.85, 3000),
            (0.05, 63, 100, 0.85, 1800), (0.05, 127, 50, 0.85, 1800),
            (0.04, 63, 100, 1.0, 2500), (0.04, 127, 300, 1.0, 2500),
            (0.05, 255, 300, 0.85, 1800), (0.08, 63, 100, 0.85, 1200),
            (0.04, 63, 50, 0.85, 2500), (0.04, 127, 50, 0.85, 2500),
            (0.06, 63, 300, 0.85, 1800), (0.06, 127, 300, 0.85, 1800),
            (0.03, 255, 300, 0.85, 2500), (0.05, 63, 300, 1.0, 2000),
        )
    ],
    "cat": [
        {"iterations": it, "learning_rate": lr, "depth": d, "l2_leaf_reg": l2}
        for it, lr, d, l2 in (
            (2000, 0.05, 8, 3), (2000, 0.05, 10, 3), (2000, 0.05, 8, 9),
            (2000, 0.05, 10, 9), (2000, 0.05, 6, 3), (2000, 0.05, 6, 9),
            (3000, 0.03, 8, 3), (3000, 0.03, 10, 3), (3000, 0.03, 8, 9),
            (3000, 0.03, 10, 9), (1500, 0.07, 8, 3), (1500, 0.07, 10, 3),
        )
    ],
    "xgb": [
        {"n_estimators": n_est, "learning_rate": lr, "max_depth": d,
         "min_child_weight": mcw, "colsample_bytree": cs, "subsample": ss}
        for n_est, lr, d, mcw, cs, ss in (
            (1600, 0.05, 6, 30, 0.8, 0.85), (1600, 0.05, 6, 100, 0.9, 0.85),
            (1600, 0.05, 10, 30, 0.8, 0.85), (1600, 0.05, 10, 100, 0.7, 0.85),
            (2200, 0.03, 6, 60, 0.85, 0.85), (2200, 0.03, 10, 60, 0.85, 0.85),
            (1600, 0.05, 6, 30, 0.9, 0.7), (1600, 0.05, 10, 100, 0.8, 0.9),
            (2500, 0.03, 6, 30, 0.8, 0.8), (1200, 0.07, 6, 30, 0.8, 0.85),
            (1600, 0.05, 8, 60, 0.8, 0.85), (2000, 0.04, 6, 100, 0.85, 0.8),
        )
    ],
    "hist": [
        {"max_iter": mi, "learning_rate": lr, "max_leaf_nodes": ml,
         "l2_regularization": l2, "min_samples_leaf": ms}
        for mi, lr, ml, l2, ms in (
            (900, 0.03, 31, 5, 80), (900, 0.03, 63, 5, 80),
            (900, 0.04, 31, 5, 80), (900, 0.04, 63, 5, 80),
            (900, 0.06, 31, 5, 80), (900, 0.06, 63, 5, 80),
            (900, 0.08, 31, 5, 80), (900, 0.08, 63, 5, 80),
            (900, 0.04, 31, 15, 80), (900, 0.04, 63, 15, 80),
            (900, 0.04, 63, 5, 40), (900, 0.04, 63, 5, 160),
            (1400, 0.03, 63, 5, 80), (900, 0.05, 63, 10, 80),
            (900, 0.04, 45, 5, 60), (1100, 0.03, 45, 8, 100),
        )
    ],
}

EXTRA_LOSSES = {
    "lgbm": [("q35", {"objective": "quantile", "alpha": 0.35}),
             ("q65", {"objective": "quantile", "alpha": 0.65}),
             ("huber", {"objective": "huber", "alpha": 0.5})],
    "cat": [("q50", {"loss_function": "Quantile:alpha=0.5"}),
            ("huber", {"loss_function": "Huber:delta=0.5"})],
    "xgb": [("q50", {"objective": "reg:quantileerror", "quantile_alpha": 0.5}),
            ("phuber", {"objective": "reg:pseudohubererror", "huber_slope": 0.5})],
}

WEIGHT_SCHEMES = {"w1": None, "wlin": [1.0, 2.0, 3.0],
                  "wsq": [1.0, 4.0, 9.0], "wmix": [1.0, 2.0, 4.0]}

CALIB_C = [-0.12, -0.09, -0.06, -0.04, -0.02, 0.0, 0.02, 0.04]
CALIB_TAU = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
HURDLE_K = [0.85, 0.95, 1.0, 1.1, 1.25]

_W = {}


def rmsle_z(y_raw, z_pred):
    lt = np.log1p(np.clip(y_raw, 0, None))
    lp = np.clip(z_pred, None, 30.0)
    return float(np.sqrt(np.mean((lt - lp) ** 2)))


def build_index() -> list[dict]:
    specs: list[dict] = []

    def add(family, tag, prio, **kw):
        kw.update({"id": len(specs), "type": "train", "family": family,
                   "tag": tag, "priority": prio})
        specs.append(kw)

    for block, (model, params), wscheme in itertools.product(
            BLOCKS.keys(), MODEL_GRID.items(), WEIGHT_SCHEMES.keys()):
        for li, p in enumerate(params):
            wp = {"w1": 0, "wlin": 1, "wmix": 2, "wsq": 3}[wscheme]
            add(f"{model}", f"{model}-v{li}-{block}-{wscheme}",
                BLOCK_PRIO[block] + li // 2 + wp,
                model=model, params=p, loss="l2", block=block,
                weights=wscheme, clip_z=False, postproc=True, seeds=[SEED])

    for model, losses in EXTRA_LOSSES.items():
        for lname, lparams in losses:
            for block in BLOCKS.keys():
                for wscheme in ("w1", "wlin"):
                    add(f"{model}-{lname}",
                        f"{model}-{lname}-{block}-{wscheme}",
                        BLOCK_PRIO[block] + 6,
                        model=model,
                        params=dict(MODEL_GRID[model][0], **lparams),
                        loss=lname, block=block, weights=wscheme,
                        clip_z=False, postproc=True, seeds=[SEED])

    for block in ("pca_btyd", "pca32"):
        for model in ("lgbm", "cat", "xgb", "hist"):
            p = dict(MODEL_GRID[model][0])
            key = {"lgbm": "random_state", "xgb": "random_state",
                   "cat": "random_seed", "hist": None}[model]
            if key:
                p[key] = SEED
            add(f"{model}-seeds", f"{model}-seedavg-{block}", -15,
                model=model, params=p, loss="l2", block=block,
                weights="w1", clip_z=False, postproc=True,
                seeds=[SEED, 7, 2026])

    for block in ("pca_btyd", "pca32"):
        for reg_i in range(2):
            for wscheme in ("w1", "wlin"):
                add("hurdle", f"hurdle-lgbm-r{reg_i}-{block}-{wscheme}", -20,
                    model="hurdle",
                    params={"clf": {"n_estimators": 1200, "learning_rate": 0.05,
                                    "num_leaves": 63, "min_child_samples": 200},
                            "reg": dict(MODEL_GRID["lgbm"][reg_i])},
                    loss="l2", block=block, weights=wscheme, clip_z=False,
                    postproc=True, seeds=[SEED])

    for w in (0.3, 0.5):
        add("pseudo", f"pseudo-lgbm-w{int(w * 10)}-pca_btyd", -10,
            model="pseudo", params=dict(MODEL_GRID["lgbm"][0]),
            loss="l2", block="pca_btyd", weights="w1", clip_z=False,
            postproc=True, seeds=[SEED], pseudo_weight=w)

    for block in ("pca_btyd", "pca32"):
        for mi in range(2):
            add("qmap", f"lgbm-qmap-{block}-v{mi}", -6,
                model="lgbm", params=MODEL_GRID["lgbm"][mi], loss="l2",
                block=block, weights="w1", clip_z=False, postproc=True,
                seeds=[SEED], qmap=True)

    specs.sort(key=lambda s: s["priority"])
    for i, sp in enumerate(specs):
        sp["id"] = i
    IDX_PATH.parent.mkdir(parents=True, exist_ok=True)
    IDX_PATH.write_text(json.dumps(specs, indent=1))
    return specs


def load_index() -> list[dict]:
    if not IDX_PATH.exists():
        return build_index()
    return json.loads(IDX_PATH.read_text())


def _init_worker():
    folds = {}
    for f in TRAIN_FOLDS + [HOLDOUT, "fold_end"]:
        feats = pl.read_parquet(FEAT_DIR / f / "batch_*.parquet")
        btyd = pl.read_parquet(BTYD_DIR / f"{f}.parquet")
        df = feats.join(btyd, on=["anchor_date", "user_id"], how="left")
        df = df.with_columns([
            pl.col(c).fill_null(0.0) if c in BTYD_FILL_ZERO
            else pl.col(c).fill_null(-1.0)
            for c in btyd.columns if c not in ("anchor_date", "user_id")
        ])
        folds[f] = df
    _W["folds"] = folds
    _W["feats"] = [c for c in folds["fold_00"].columns if c not in META_COLS]
    _W["mat"] = {}
    _W["y_raw"] = {f: np.clip(folds[f]["target"].to_numpy(), 0, None)
                   for f in folds}
    _W["y_log"] = {f: np.log1p(_W["y_raw"][f]) for f in folds}


def _mat(block, fold):
    key = (block, fold)
    if key not in _W["mat"]:
        cols = BLOCKS[block](_W["feats"])
        _W["mat"][key] = (_W["folds"][fold].select(cols)
                          .to_numpy().astype(np.float32))
    return _W["mat"][key]


def _qref():
    if "qref" not in _W:
        _W["qref"] = np.sort(np.concatenate(
            [_W["y_log"][f] for f in TRAIN_FOLDS]))
    return _W["qref"]


def _make_model(model, params, seed, threads, quick):
    p = dict(params)
    if quick:
        p = {k: (max(int(v * 0.06), 40)
                 if k in ("n_estimators", "iterations", "max_iter") else v)
             for k, v in p.items()}
    if model == "lgbm":
        from lightgbm import LGBMRegressor
        obj = p.pop("objective", "l2")
        extra = {k: p.pop(k) for k in ("alpha",) if k in p}
        n = p.pop("n_estimators")
        return LGBMRegressor(**p, **extra, n_estimators=n, objective=obj,
                             random_state=seed, n_jobs=threads, verbosity=-1)
    if model == "lgbm_clf":
        from lightgbm import LGBMClassifier
        n = p.pop("n_estimators")
        return LGBMClassifier(**p, n_estimators=n, random_state=seed,
                              n_jobs=threads, verbosity=-1)
    if model == "cat":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(thread_count=threads, **p, random_seed=seed,
                                 verbose=0, allow_writing_files=False)
    if model == "xgb":
        from xgboost import XGBRegressor
        n = p.pop("n_estimators")
        return XGBRegressor(**p, n_estimators=n, tree_method="hist",
                            random_state=seed, n_jobs=threads, verbosity=0)
    if model == "hist":
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(
            early_stopping=True, validation_fraction=0.08,
            n_iter_no_change=30, random_state=seed, **p)
    raise ValueError(model)


def _fit(model_name, model, Xtr, ytr, sw, Xs, es):
    t0 = time.time()
    if model_name.startswith("lgbm") and es is not None:
        from lightgbm import early_stopping as lgb_es, log_evaluation
        model.fit(Xtr, ytr, sample_weight=sw, eval_set=[es],
                  callbacks=[lgb_es(100, verbose=False), log_evaluation(0)])
    elif model_name.startswith("cat") and es is not None:
        model.fit(Xtr, ytr, sample_weight=sw, eval_set=es,
                  early_stopping_rounds=150)
    elif model_name == "xgb" and es is not None:
        model.fit(Xtr, ytr, sample_weight=sw, eval_set=[es], verbose=False)
    else:
        model.fit(Xtr, ytr, sample_weight=sw)
    preds = [np.clip(model.predict(X), None, 30.0) for X in Xs]
    print(f"      fit {time.time() - t0:.0f}s", flush=True)
    return preds


def _train_arrays(spec):
    tr_X = np.concatenate([_mat(spec["block"], f) for f in TRAIN_FOLDS])
    tr_y = np.concatenate([_W["y_log"][f] for f in TRAIN_FOLDS])
    w = WEIGHT_SCHEMES[spec["weights"]]
    tr_sw = np.concatenate([
        np.full(len(_W["y_log"][f]), float(w[i]) if w is not None else 1.0)
        for i, f in enumerate(TRAIN_FOLDS)])
    if spec.get("clip_z"):
        tr_y = np.clip(tr_y, None, float(np.quantile(tr_y, 0.999)))
    es = (_mat(spec["block"], ES_FOLD), _W["y_log"][ES_FOLD])
    Xs = [_mat(spec["block"], HOLDOUT), _mat(spec["block"], "fold_end")]
    return tr_X, tr_y, tr_sw, es, Xs


def run_core_spec(spec, threads, quick):
    tr_X, tr_y, tr_sw, es, Xs = _train_arrays(spec)
    pv, pe = [], []
    for seed in spec["seeds"]:
        m = _make_model(spec["model"], spec["params"], seed, threads, quick)
        out = _fit(spec["model"], m, tr_X, tr_y, tr_sw, Xs, es)
        pv.append(out[0])
        pe.append(out[1])
    return {"z_val": np.mean(pv, axis=0), "z_end": np.mean(pe, axis=0)}


def run_hurdle_spec(spec, threads, quick):
    tr_X, tr_y, tr_sw, es, Xs = _train_arrays(spec)
    y_bin = (tr_y > 0).astype(np.int8)
    clf = _make_model("lgbm_clf", spec["params"]["clf"], SEED, threads, quick)
    from lightgbm import early_stopping as lgb_es, log_evaluation
    clf.fit(tr_X, y_bin, sample_weight=tr_sw, eval_set=[(es[0], (es[1] > 0).astype(np.int8))],
            callbacks=[lgb_es(100, verbose=False), log_evaluation(0)])
    p_tr = clf.predict_proba(tr_X)[:, 1]
    p_val = clf.predict_proba(Xs[0])[:, 1]
    p_end = clf.predict_proba(Xs[1])[:, 1]

    mask = tr_y > 0
    reg = _make_model("lgbm", spec["params"]["reg"], SEED, threads, quick)
    r_out = _fit("lgbm", reg, tr_X[mask], tr_y[mask], tr_sw[mask], Xs, es)

    def combine(pv, rv):
        return np.column_stack([pv * rv * k for k in HURDLE_K])

    z_val_c = combine(p_val, r_out[0])
    z_end_c = combine(p_end, r_out[1])
    ks = list(HURDLE_K)
    k_best = int(np.argmin([rmsle_z(_W["y_raw"][HOLDOUT], z_val_c[:, i])
                            for i in range(len(ks))]))
    return {"z_val": z_val_c[:, k_best], "z_end": z_end_c[:, k_best],
            "k_scale": ks[k_best]}


def run_pseudo_spec(spec, threads, quick):
    tr_X, tr_y, tr_sw, es, Xs = _train_arrays(spec)
    base = _make_model("lgbm", spec["params"], SEED, threads, quick)
    out = _fit("lgbm", base, tr_X, tr_y, tr_sw, Xs, es)
    z_pseudo = np.clip(out[1], 0.0, 12.0)

    X_all = np.concatenate([tr_X, Xs[1]])
    y_all = np.concatenate([tr_y, z_pseudo])
    sw_all = np.concatenate([tr_sw, np.full(len(z_pseudo),
                                            spec.get("pseudo_weight", 0.4))])
    final = _make_model("lgbm", spec["params"], SEED, threads, quick)
    n_iter = int(final.get_params()["n_estimators"] * (0.06 if quick else 0.6))
    final.set_params(n_estimators=max(n_iter, 40))
    final.fit(X_all, y_all, sample_weight=sw_all)
    pv = np.clip(final.predict(Xs[0]), None, 30.0)
    pe = np.clip(final.predict(Xs[1]), None, 30.0)
    print(f"      pseudo-refit done", flush=True)
    return {"z_val": pv, "z_end": pe}


def job(spec: dict, threads: int, quick: bool) -> dict:
    if spec["model"] == "hurdle":
        res = run_hurdle_spec(spec, threads, quick)
    elif spec["model"] == "pseudo":
        res = run_pseudo_spec(spec, threads, quick)
    else:
        res = run_core_spec(spec, threads, quick)
    z_val, z_end = res.pop("z_val"), res.pop("z_end")
    y_raw = _W["y_raw"][HOLDOUT]

    cands = [("raw", 0.0, 0.0, rmsle_z(y_raw, z_val))]
    for c in CALIB_C:
        for tau in CALIB_TAU:
            s = rmsle_z(y_raw, np.where(z_val + c < tau, 0.0, z_val + c))
            cands.append((f"c{c}_t{tau}", c, tau, s))
    if spec.get("qmap", False) or spec["family"] == "qmap":
        ref = _qref()
        ranks = rankdata(z_val, method="average") / (len(z_val) + 1.0)
        zq = np.quantile(ref, ranks)
        cands.append(("qmap", 0.0, 0.0, rmsle_z(y_raw, zq)))
        for c in (-0.03, 0.0, 0.03):
            cands.append((f"qmap_c{c}", c, 0.0, rmsle_z(y_raw, zq + c)))
    best = min(cands, key=lambda x: x[3])

    if best[0] == "raw":
        z_final = z_end
    elif best[0] == "qmap":
        ranks = rankdata(z_end, method="average") / (len(z_end) + 1.0)
        z_final = np.quantile(_qref(), ranks)
    elif best[0].startswith("qmap_c"):
        ranks = rankdata(z_end, method="average") / (len(z_end) + 1.0)
        z_final = np.quantile(_qref(), ranks) + best[1]
    else:
        z_final = np.where(z_end + best[1] < best[2], 0.0, z_end + best[1])

    res.update({"rmsle_fold03": round(best[3], 5),
                "postproc": best[0], "calib_c": best[1], "tau": best[2],
                "k_scale": res.get("k_scale"), "z_val": z_val, "z_end": z_final})
    return res


def _load_results() -> list[dict]:
    if not RES_PATH.exists():
        return []
    rows = []
    for line in RES_PATH.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return rows


def cmd_run(args) -> None:
    specs = load_index()
    by_id = {s["id"]: s for s in specs}
    done = {int(r["id"]) for r in _load_results()}

    if args.family == "blends":
        rows = [r for r in _load_results()
                if r.get("type") == "train" and r.get("z_cache")]
        if len(rows) < 2:
            print("need >=2 finished train experiments")
            return
        rows.sort(key=lambda r: r["rmsle_fold03"])
        pool = rows[:12]
        y_raw = None
        uids, order = _uids_and_order()

        def load(id_):
            d = np.load(PRED_DIR / f"{id_}.npz")
            return d["val"], d["end"]

        cache = {r["id"]: load(r["id"]) for r in pool}
        combos = []
        for i in range(len(pool)):
            for j in range(i + 1, len(pool)):
                for w in (0.3, 0.5, 0.7):
                    combos.append(([pool[i], pool[j]], [w, 1 - w]))
        for i in range(min(len(pool), 6)):
            for j in range(i + 1, min(len(pool), 6)):
                for k in range(j + 1, min(len(pool), 6)):
                    combos.append(([pool[i], pool[j], pool[k]],
                                   [1 / 3, 1 / 3, 1 / 3]))
        best_member = min(r["rmsle_fold03"] for r in pool)
        saved = 0
        next_id = max(r["id"] for r in _load_results()) + 10000
        for mem, ws in combos:
            zv = sum(w * cache[r["id"]][0] for r, w in zip(mem, ws))
            ze = sum(w * cache[r["id"]][1] for r, w in zip(mem, ws))
            if y_raw is None:
                lf = pl.scan_parquet(FEAT_DIR / HOLDOUT / "batch_*.parquet")
                y_raw = np.log1p(np.clip(
                    lf.select("target").collect().to_series().to_numpy(),
                    0, None))
            s = rmsle_z(np.expm1(y_raw), zv)
            row = {"id": next_id, "type": "blend", "z_cache": False,
                   "tag": "blend:" + "+".join(f"{r['id']}x{w:g}"
                                              for r, w in zip(mem, ws)),
                   "members": [r["id"] for r in mem], "weights": ws,
                   "rmsle_fold03": round(s, 5)}
            if s < best_member:
                pred = np.clip(np.expm1(ze), 0, None)
                path = SUB_DIR / f"exp{next_id}_blend.csv"
                (pl.DataFrame({"user_id": uids, "predict": pred})
                 .join(order, on="user_id", how="inner").sort("__ord")
                 .drop("__ord").select(["user_id", "predict"]).write_csv(path))
                row["submission"] = str(path)
                saved += 1
            with RES_PATH.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(f"  {row['tag']}: {s:.5f}"
                  + (" *" if s < best_member else ""), flush=True)
            next_id += 1
        print(f"blends: {len(combos)} evaluated, {saved} CSV saved "
              f"(beat best member {best_member:.5f})")
        return

    if args.ids:
        ids = [int(x) for x in args.ids.split(",")]
    else:
        cand = sorted((s for s in specs if s["id"] not in done),
                      key=lambda s: s["priority"])
        ids = [s["id"] for s in (cand[:args.top] if args.top else cand)]

    t_start = time.time()
    SUB_DIR.mkdir(exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    uids, order = _uids_and_order()
    print(f"running {len(ids)} experiments: workers={args.workers} "
          f"threads={args.threads} quick={args.quick}", flush=True)

    n_done = 0
    with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker) as ex:
        futs = {}
        it = iter(ids)
        pending = True
        while pending or futs:
            while pending and len(futs) < args.workers * 2:
                try:
                    eid = next(it)
                except StopIteration:
                    pending = False
                    break
                if args.max_minutes and \
                        (time.time() - t_start) / 60 > args.max_minutes:
                    print(f"time budget reached; stopping submissions")
                    pending = False
                    break
                futs[ex.submit(job, by_id[eid], args.threads,
                               args.quick)] = eid
            if not futs:
                break
            fut = next(as_completed(list(futs)))
            eid = futs.pop(fut)
            try:
                res = fut.result()
                z_end = res.pop("z_end")
                np.savez_compressed(PRED_DIR / f"{eid}.npz",
                                    val=res.pop("z_val"), end=z_end)
                pred = np.clip(np.expm1(z_end), 0, None)
                csv_path = SUB_DIR / f"exp{eid:04d}_{by_id[eid]['family']}.csv"
                (pl.DataFrame({"user_id": uids, "predict": pred})
                 .join(order, on="user_id", how="inner").sort("__ord")
                 .drop("__ord").select(["user_id", "predict"]).write_csv(csv_path))
                res.update({"id": eid, "type": "train",
                            "family": by_id[eid]["family"],
                            "params": by_id[eid]["params"],
                            "submission": str(csv_path), "z_cache": True})
                with RES_PATH.open("a") as fh:
                    fh.write(json.dumps(res) + "\n")
                n_done += 1
                print(f"[{n_done}/{len(ids)}] exp{eid:04d} "
                      f"{by_id[eid]['tag']}: RMSLE={res['rmsle_fold03']} "
                      f"(pp={res['postproc']})", flush=True)
            except Exception as e:
                print(f"exp{eid:04d} FAILED: {e}", flush=True)

    print(f"DONE this run: {n_done} experiments in "
          f"{(time.time() - t_start) / 60:.1f} min")


def _uids_and_order():
    import functools

    @functools.lru_cache(maxsize=1)
    def _cached():
        uids = pl.read_parquet(
            FEAT_DIR / "fold_end" / "batch_*.parquet",
            columns=["user_id"])["user_id"].cast(pl.Int64)
        order = (pl.read_csv("sample_submit.csv").select("user_id")
                 .with_row_index("__ord"))
        return uids, order
    return _cached()


def cmd_status() -> None:
    specs = load_index()
    rows = _load_results()
    fam_counts: dict[str, int] = {}
    for r in rows:
        fam_counts[r.get("family", r.get("type", "?"))] = \
            fam_counts.get(r.get("family", r.get("type", "?")), 0) + 1
    print(f"index: {len(specs)} | done: {len(rows)} | by family: {fam_counts}")
    for r in sorted(rows, key=lambda r: r.get("rmsle_fold03", 9))[:20]:
        sub = " ->" + r["submission"] if r.get("submission") else ""
        print(f"  {r['rmsle_fold03']:.5f}  exp{r['id']:>5}  {r['tag']}{sub}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    sub.add_parser("status")
    run_p = sub.add_parser("run")
    run_p.add_argument("--ids", type=str, default=None)
    run_p.add_argument("--top", type=int, default=None)
    run_p.add_argument("--family", type=str, default=None,
                       choices=["blends"])
    run_p.add_argument("--all", action="store_true")
    run_p.add_argument("--quick", action="store_true")
    run_p.add_argument("--max-minutes", type=float, default=None)
    run_p.add_argument("--workers", type=int, default=5)
    run_p.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    if args.cmd == "build":
        sp = build_index()
        print(f"index written: {len(sp)} specs -> {IDX_PATH}")
    elif args.cmd == "status":
        cmd_status()
    else:
        cmd_run(args)


if __name__ == "__main__":
    main()
