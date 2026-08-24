"""B5 stage 1: behavioural trajectory segmentation.

Builds compact standardized user profiles from the accepted exp02 66-feature
scheme (history <= anchor only, target NEVER used), clusters them per anchor,
aligns segment labels across anchors with Hungarian assignment on centroid
distances, and caches segments for b05_train.py.

Method notes
------------
- Profile: 18 dims = log1p magnitudes (gmv/search/to_cart sums 30/90d, aov_30)
  + raw rates/recencies (recency_to_ord_days, recency_searches_days,
  tenure_days, active_days_30d, ord_days_30, conv_s2o/c2o/o2c,
  share_gmv_search_90, share_gmv_cat_90).
- NaN policy: conv_* are undefined when the denominator event never happened;
  filled with 0.0 ("no conversion happened"). Other profile columns are
  asserted null-free. Sentinel recency=999 (no event ever) kept as-is under
  log1p - semantically "very far".
- StandardScaler is fit SEPARATELY on each anchor's 250k profiles (per-anchor
  standardization, per hypothesis spec).
- Model choice KMeans vs GMM: both evaluated on fold_03 subsample for
  k=4..10 with unified criterion (silhouette on labels), plus stability
  (ARI over seeds {42,1,2}) and GMM BIC. Constraint: minimal segment >= 5%
  (checked on full-anchor fits for finalists). Winner -> refit on every
  anchor (full 250k, seed=42).
- Label alignment: adjacent anchors (chronological) matched by Hungarian
  assignment (scipy.optimize.linear_sum_assignment) on the Euclidean distance
  matrix of centroids expressed in the COMMON raw-log space (each anchor's
  centroids inverse_transformed out of its own scaler). ARI / exact-match on
  the user intersection (=all 250k users) reported as diagnostics.

Writes data/v2/b05_seg/<fold>.parquet (user_id, seg_raw, segment_id),
data/v2/b05_seg/meta.json, figures to reports/b05_figures/.
"""

import json
import platform
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

SEED = 42
FOLDS = ["fold_00", "fold_01", "fold_02", "fold_03", "fold_end"]
OUT_DIR = Path("data/v2/b05_seg")
FIG_DIR = Path("reports/b05_figures")
LOG_PATH = Path("reports/b05_cluster.log")

KS = list(range(4, 11))
MIN_SEG_SHARE = 0.05
SUBSAMPLE_N = 50_000
SIL_SAMPLE_N = 20_000
STABILITY_SEEDS = [42, 1, 2]

BASE_COLS = [
    "gmv_sum_30d", "gmv_sum_90d",
    "searches_sum_30d", "searches_sum_90d",
    "to_ord_sum_30d", "to_ord_sum_90d",
    "to_cart_sum_30d",
    "recency_to_ord_days", "recency_searches_days", "tenure_days",
    "active_days_30d",
]
EXTRA_COLS = ["aov_30", "ord_days_30", "conv_s2o", "conv_c2o", "conv_o2c",
              "share_gmv_search_90", "share_gmv_cat_90"]
# log1p-transformed magnitudes (heavy right tail)
LOG_COLS = BASE_COLS[:7] + ["aov_30"]
# kept raw (bounded or discrete)
RAW_COLS = ["recency_to_ord_days", "recency_searches_days", "tenure_days",
            "active_days_30d", "ord_days_30",
            "conv_s2o", "conv_c2o", "conv_o2c",
            "share_gmv_search_90", "share_gmv_cat_90"]
PROFILE_COLS = LOG_COLS + RAW_COLS

BASE_DROP = ["anchor_date", "user_id", "target"]
CONV_FILL = {"conv_s2o": 0.0, "conv_c2o": 0.0, "conv_o2c": 0.0}

M: dict = {}


def log(msg: str) -> None:
    print(msg, flush=True)


def load_profile(fold: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (profile_matrix float64 [n x d], user_id u32[n]) for one anchor."""
    base_dir = Path(f"data/v2/features/{fold}")
    if not base_dir.exists():
        base_dir = Path(f"data/v2/features_exp02/{fold}_base")
    base = pl.read_parquet(str(base_dir / "batch_*.parquet")).select(
        ["user_id"] + BASE_COLS)
    extra = pl.read_parquet(f"data/v2/features_exp02/{fold}/batch_*.parquet").select(
        ["user_id", *EXTRA_COLS])
    df = base.join(extra, on="user_id", how="inner", validate="1:1")
    assert df.height == base.height
    df = df.with_columns([pl.col(c).fill_null(v) for c, v in CONV_FILL.items()])
    nulls = {c: df[c].null_count() for c in PROFILE_COLS}
    bad = {c: n for c, n in nulls.items() if n > 0}
    assert not bad, f"{fold}: unexpected nulls {bad}"
    X = df.select(PROFILE_COLS).to_numpy().astype(np.float64)
    X[:, : len(LOG_COLS)] = np.log1p(X[:, : len(LOG_COLS)])
    return np.ascontiguousarray(X), df["user_id"].to_numpy()


def fit_labels(method: str, k: int, X: np.ndarray, seed: int, n_init: int = 1):
    if method == "kmeans":
        km = KMeans(n_clusters=k, n_init=n_init, random_state=seed)
        return km.fit_predict(X)
    gm = GaussianMixture(
        n_components=k, covariance_type="full", random_state=seed,
        n_init=n_init, reg_covar=1e-5,
    )
    return gm.fit_predict(X)


def seg_shares(labels: np.ndarray, k: int) -> np.ndarray:
    return np.bincount(labels, minlength=k) / len(labels)


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    M["meta"] = {
        "seed": SEED,
        "profile_cols": {"log1p": LOG_COLS, "raw": RAW_COLS},
        "nan_policy": {
            "conv_s2o/conv_c2o/conv_o2c": "fill 0.0 (denominator event absent)",
            "others": "asserted null-free",
            "recency_sentinel": "999 (no event ever) kept, log1p applied",
        },
        "versions": {
            "python": platform.python_version(),
            "polars": pl.__version__,
            "sklearn": __import__("sklearn").__version__,
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
        },
        "constraint_min_segment_share": MIN_SEG_SHARE,
    }

    # ---- load profiles -------------------------------------------------
    log("loading profiles...")
    prof: dict[str, np.ndarray] = {}
    scalers: dict[str, StandardScaler] = {}
    user_ids: dict[str, np.ndarray] = {}
    for f in FOLDS:
        t = time.time()
        X, uid = load_profile(f)
        sc = StandardScaler().fit(X)
        prof[f] = sc.transform(X)
        scalers[f] = sc
        user_ids[f] = uid
        log(f"  {f}: {X.shape} scaled ({time.time() - t:.1f}s)")
    ref_set = set(user_ids[FOLDS[0]].tolist())
    assert all(set(user_ids[f].tolist()) == ref_set for f in FOLDS), \
        "user_id sets differ across anchors"
    log(f"user intersection across anchors: {len(ref_set)} users")
    M["profiles"] = {
        f: {"n": int(prof[f].shape[0]), "d": int(prof[f].shape[1])} for f in FOLDS
    }

    rng = np.random.default_rng(SEED)
    sel_fold = "fold_03"
    idx_sub = rng.choice(prof[sel_fold].shape[0], SUBSAMPLE_N, replace=False)
    Xsub = prof[sel_fold][idx_sub]
    sil_idx = rng.choice(SUBSAMPLE_N, SIL_SAMPLE_N, replace=False)

    # ---- model/k selection on fold_03 ----------------------------------
    log(f"\nmodel/k selection on {sel_fold} (subsample {SUBSAMPLE_N})...")
    table = []
    n_init_sel = {"kmeans": 10, "gmm": 1}
    for method in ["kmeans", "gmm"]:
        for k in KS:
            t = time.time()
            lab_main = fit_labels(method, k, Xsub, SEED, n_init=n_init_sel[method])
            sil = float(silhouette_score(Xsub[sil_idx], lab_main[sil_idx]))
            labs_other = [fit_labels(method, k, Xsub, s, n_init=n_init_sel[method])
                          for s in STABILITY_SEEDS[1:]]
            ari_stab = float(np.mean([adjusted_rand_score(lab_main, l) for l in labs_other]))
            shares = seg_shares(lab_main, k)
            bic = None
            if method == "gmm":
                gm = GaussianMixture(
                    n_components=k, covariance_type="full", random_state=SEED,
                    n_init=1, reg_covar=1e-5,
                ).fit(Xsub)
                bic = float(gm.bic(Xsub))
            row = {
                "method": method, "k": k,
                "silhouette": round(sil, 4),
                "stability_ari_seeds": round(ari_stab, 4),
                "min_segment_share_subsample": round(float(shares.min()), 4),
                "bic_subsample": None if bic is None else round(bic, 1),
                "sec": round(time.time() - t, 1),
            }
            table.append(row)
            log(f"  {method:>6} k={k}: sil={sil:.4f} stab={ari_stab:.3f} "
                f"minseg={shares.min():.3f} {'bic=%.0f' % bic if bic else ''} ({row['sec']}s)")
    M["selection_table_fold03"] = table

    # valid candidates: stability + subsample min-share, pick best silhouette
    def valid(r):
        return (r["stability_ari_seeds"] >= 0.5
                and r["min_segment_share_subsample"] >= MIN_SEG_SHARE * 0.95)

    cands = sorted([r for r in table if valid(r)],
                   key=lambda r: (-r["silhouette"], r["k"]))
    assert cands, "no valid (method,k) candidate"
    n_init_fit = {"kmeans": 10, "gmm": 2}

    chosen = None
    labels_raw: dict[str, np.ndarray] = {}
    for cand in cands[:3]:  # verify min-share >=5% on FULL anchor fits
        method, k = cand["method"], cand["k"]
        full_shares = {}
        ok = True
        labs_full: dict[str, np.ndarray] = {}
        for f in FOLDS:
            lab = fit_labels(method, k, prof[f], SEED, n_init=n_init_fit[method])
            labs_full[f] = lab
            sh = seg_shares(lab, k).min()
            full_shares[f] = round(float(sh), 4)
            if sh < MIN_SEG_SHARE:
                ok = False
                break
        if ok:
            chosen = (method, k)
            labels_raw = labs_full  # reuse the verified full-fit labels
            M["full_min_segment_share"] = full_shares
            break
        log(f"  candidate {method} k={k} rejected on full fits: {full_shares}")
    assert chosen, "no candidate satisfied >=5% on full fits"
    method_chosen, k_chosen = chosen
    M["chosen"] = {
        "method": method_chosen, "k": k_chosen,
        "rationale": (
            f"max silhouette among candidates with stability ARI>=0.5 and "
            f"min segment >={MIN_SEG_SHARE:.0%} verified on full-anchor fits; "
            f"GMM BIC reported as cross-check"
        ),
    }
    log(f"\nCHOSEN: {method_chosen}, k={k_chosen}")

    # ---- Hungarian alignment across chronological anchors ---------------
    # Primary: distribution-based Hungarian on the user-overlap contingency
    # (the SAME 250k users exist under every anchor; cost = -log overlap,
    # equivalent to maximizing total preserved membership).
    # Secondary (diagnostic only): centroid-distance Hungarian in the common
    # raw-log space.
    log("\nHungarian label alignment...")
    maps = {}
    align_stats = []
    aligned = {FOLDS[0]: labels_raw[FOLDS[0]].copy()}
    for prev_f, f in zip(FOLDS[:-1], FOLDS[1:]):
        lab_p = aligned[prev_f]
        lab_c_raw = labels_raw[f]
        ord_p = np.argsort(user_ids[prev_f])
        ord_c = np.argsort(user_ids[f])

        # primary: contingency-overlap assignment
        cont = np.zeros((k_chosen, k_chosen), dtype=np.int64)
        np.add.at(cont, (lab_p[ord_p], lab_c_raw[ord_c]), 1)
        row_ind, col_ind = linear_sum_assignment(-cont)
        mapping = np.empty(k_chosen, dtype=np.int64)
        mapping[col_ind] = row_ind  # raw label j of anchor f -> prev vocabulary
        mapped = mapping[lab_c_raw]

        # secondary diagnostic: centroid distance assignment
        c_prev = scalers[prev_f].inverse_transform(
            _centroids(method_chosen, k_chosen, prev_f, prof, labels_raw))
        c_cur = scalers[f].inverse_transform(
            _centroids(method_chosen, k_chosen, f, prof, labels_raw))
        dist = np.linalg.norm(c_prev[:, None, :] - c_cur[None, :, :], axis=-1)
        r2, c2 = linear_sum_assignment(dist)
        map_cent = np.empty(k_chosen, dtype=np.int64)
        map_cent[c2] = r2

        aligned[f] = mapped
        maps[f"{prev_f}->{f}"] = {
            "overlap_based": mapping.tolist(),
            "centroid_based": map_cent.tolist(),
        }
        lab_p_sorted = lab_p[ord_p]
        lab_c_sorted = mapped[ord_c]
        agree = float((lab_p_sorted == lab_c_sorted).mean())
        ari = float(adjusted_rand_score(lab_p_sorted, lab_c_sorted))
        cent_agree = float((lab_p_sorted == map_cent[lab_c_raw][ord_c]).mean())
        align_stats.append({
            "pair": f"{prev_f}->{f}",
            "match_rate_all_users": round(agree, 4),
            "ari_after_mapping": round(ari, 4),
            "match_rate_if_centroid_mapping": round(cent_agree, 4),
            "max_centroid_dist": round(float(dist.max()), 3),
        })
        log(f"  {prev_f}->{f}: match={agree:.3f} ari={ari:.3f} "
            f"(centroid-mapping would give {cent_agree:.3f})")
    M["alignment"] = {
        "space": "primary: Hungarian (linear_sum_assignment) on user-overlap "
                 "contingency between adjacent anchors (maximizes preserved "
                 "membership); secondary diagnostic: centroid distances in "
                 "common raw-log space",
        "pairs": align_stats,
        "maps": maps,
    }

    # consistency test: mapped labels must be consistent on user intersection
    for st in align_stats:
        assert st["match_rate_all_users"] > 0.5, \
            f"label consistency test failed: {st}"

    # ---- persist --------------------------------------------------------
    for f in FOLDS:
        pl.DataFrame({
            "user_id": user_ids[f],
            "seg_raw": labels_raw[f].astype(np.int32),
            "segment_id": aligned[f].astype(np.int32),
        }).write_parquet(OUT_DIR / f"{f}.parquet")

    shift = [st["match_rate_all_users"] for st in align_stats]
    M["segment_shift_between_adjacent_anchors_share"] = [round(1 - s, 4) for s in shift]
    M["segment_sizes"] = {
        f: {str(i): round(float(s), 5)
            for i, s in enumerate(seg_shares(aligned[f], k_chosen))}
        for f in FOLDS
    }

    # ---- figures ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for meth, marker, color in [("kmeans", "o-", "tab:blue"), ("gmm", "s--", "tab:orange")]:
        rows = [r for r in table if r["method"] == meth]
        axes[0].plot([r["k"] for r in rows], [r["silhouette"] for r in rows],
                     marker, color=color, label=meth)
        crit = [(r["bic_subsample"] if r["bic_subsample"] is not None else np.nan)
                for r in rows]
        axes[1].plot([r["k"] for r in rows], crit, marker, color=color, label=meth)
    axes[0].axvline(k_chosen, ls=":", color="gray")
    axes[0].set(xlabel="k", ylabel="silhouette (fold_03 subsample)",
                title="B5 k-selection: silhouette")
    axes[0].legend()
    axes[0].grid(alpha=0.35)
    axes[1].set(xlabel="k", ylabel="BIC (lower=better)", title="B5 k-selection: GMM BIC")
    axes[1].legend()
    axes[1].grid(alpha=0.35)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "b05_k_selection.png", dpi=150)
    plt.close(fig)

    sizes = {f: seg_shares(aligned[f], k_chosen) for f in FOLDS}
    x = np.arange(len(FOLDS))
    plt.figure(figsize=(9, 5))
    bottom = np.zeros(len(FOLDS))
    cmap = plt.get_cmap("tab10")
    for s in range(k_chosen):
        vals = np.array([sizes[f][s] for f in FOLDS])
        plt.bar(x, vals, bottom=bottom, color=cmap(s % 10), label=f"seg {s}")
        for xi, (v, b) in enumerate(zip(vals, bottom)):
            if v > 0.07:
                plt.text(xi, b + v / 2, f"{v:.2f}", ha="center", va="center",
                         fontsize=8, color="white")
        bottom += vals
    plt.xticks(x, FOLDS)
    plt.ylabel("share of users")
    plt.title(f"B5 segment sizes by anchor ({method_chosen}, k={k_chosen}, "
              f"Hungarian-aligned)")
    plt.legend(ncol=2, fontsize=8)
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "b05_segment_sizes.png", dpi=150)
    plt.close()

    M["runtime_cluster_sec"] = round(time.time() - t0, 1)
    (OUT_DIR / "meta.json").write_text(json.dumps(M, indent=2, ensure_ascii=False))
    log(f"\nCLUSTER DONE in {(time.time() - t0) / 60:.1f} min")


def _centroids(method: str, k: int, fold: str,
               prof: dict, labels: dict) -> np.ndarray:
    """Centroid of each cluster in the anchor's STANDARDIZED space."""
    Xs, lab = prof[fold], labels[fold]
    return np.vstack([Xs[lab == j].mean(axis=0) for j in range(k)])


if __name__ == "__main__":
    main()
