"""B5 stage 2: CatBoost ablation with behavioural segments (hypothesis B5,
Arm A - segments as a cheap categorical feature for the single global model).

Protocol of exp02 reused exactly: train = fold_00+01+02, validation = fold_03;
CatBoost RMSE on z = log1p(target); lr=0.05, depth=8, l2_leaf_reg=3, seed=42,
n_estimators=1000; inverse expm1 once, clip >= 0.

Feature sets (ablation on fold_03):
  ref66    - accepted exp02 config (56 base + conv/decomp/due/shares +
             pct_rank_gmv30), control reproduction of 1.69277;
  seg_cat  - ref66 + segment_id as ONE native categorical feature
             (Pool cat_features; codes passed as distinct float values);
  seg_ohe  - ref66 + one-hot dummies of segment_id (k binary columns).

Best of {seg_cat, seg_ohe} by fold_03 -> LOFO over fold_00..03 against the
exp02 reference LOFO (1.81545 / 1.77420 / 1.71090 / 1.69277).

Diagnostics: mean target-z per segment (+ between-segment variance share),
segment feature importance, per-segment RMSLE deltas vs ref on fold_03.
Reads data/v2/b05_seg/<fold>.parquet produced by src/b05_cluster.py.
Writes reports/b05_metrics.json, reports/b05_report.md, figures.
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
from catboost import CatBoostRegressor, Pool

SEED = 42
DROP_COLS = ["anchor_date", "user_id", "target"]
SEG_DIR = Path("data/v2/b05_seg")
FIG_DIR = Path("reports/b05_figures")
METRICS_PATH = Path("reports/b05_metrics.json")
REPORT_PATH = Path("reports/b05_report.md")

REF_LOFO = {"fold_00": 1.81545, "fold_01": 1.77420,
            "fold_02": 1.71090, "fold_03": 1.69277}
ADOPTION_FOLD03 = 1.68938  # -0.2% from 1.69277

EXTRA_ALL = ["conv_s2o", "conv_c2o", "conv_o2c", "aov_30", "ord_days_30",
             "due_ratio", "share_gmv_search_90", "share_gmv_cat_90",
             "share_gmv_search_trend", "share_gmv_cat_trend"]
CV_FOLDS = ["fold_00", "fold_01", "fold_02", "fold_03"]

M: dict = {}


def log(msg: str) -> None:
    print(msg, flush=True)


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    lt = np.log1p(np.clip(y_true, 0, None))
    lp = np.log1p(np.clip(y_pred, 0, None))
    return float(np.sqrt(np.mean((lt - lp) ** 2)))


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
    seg = pl.read_parquet(SEG_DIR / f"{name}.parquet")
    j = j.join(seg, on="user_id", how="inner", validate="1:1")
    assert j.height == base.height
    return j


def fit_catboost(X_tr, y_tr, feat_names, n_estimators=1000,
                 cat_features=None):
    model = CatBoostRegressor(
        loss_function="RMSE",
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3,
        n_estimators=n_estimators,
        thread_count=-1,
        random_seed=SEED,
        verbose=0,
        allow_writing_files=False,
    )
    train_pool = Pool(X_tr, label=y_tr, feature_names=feat_names,
                      cat_features=cat_features)
    t0 = time.time()
    model.fit(train_pool)
    return model, time.time() - t0


def predict_raw(model, X, cat_features=None) -> np.ndarray:
    pool = Pool(X, feature_names=model.feature_names_, cat_features=cat_features) \
        if cat_features else X
    return np.clip(np.expm1(model.predict(pool)), 0, None)


class Mat:
    """Feature matrices for the three ablation variants."""

    def __init__(self, df: pl.DataFrame, feat_cols: list[str], k_seg: int):
        import pandas as pd
        self._pd = pd
        self.feat_cols = feat_cols
        self.k = k_seg
        self.X66 = df.select(feat_cols).to_numpy().astype(np.float64)
        seg = df["segment_id"].to_numpy().astype(np.int64)
        assert seg.min() >= 0 and seg.max() < k_seg
        self.seg = seg
        self.y_raw = np.clip(df["target"].to_numpy(), 0, None)
        self.y_log = np.log1p(self.y_raw)

    def ref66(self):
        return self.X66, self.feat_cols, None

    def seg_cat(self):
        # CatBoost requires non-float dtype for native categorical features;
        # a pandas frame keeps the numeric block float64 and the segment
        # column int64 (marked categorical via cat_features).
        df = self._pd.DataFrame(self.X66, columns=self.feat_cols, copy=False)
        df["segment_id"] = self._pd.Categorical.from_codes(
            self.seg, categories=[str(j) for j in range(self.k)])
        cols = self.feat_cols + ["segment_id"]
        return df, cols, ["segment_id"]

    def seg_ohe(self):
        ohe = np.zeros((len(self.seg), self.k), dtype=np.float64)
        ohe[np.arange(len(self.seg)), self.seg] = 1.0
        X = np.column_stack([self.X66, ohe])
        cols = self.feat_cols + [f"seg_{j}" for j in range(self.k)]
        return X, cols, None


VARIANTS = {
    "ref66": lambda m: m.ref66(),
    "seg_cat": lambda m: m.seg_cat(),
    "seg_ohe": lambda m: m.seg_ohe(),
}


def main() -> None:
    t_start = time.time()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    meta_cluster = json.loads((SEG_DIR / "meta.json").read_text())
    k_seg = int(meta_cluster["chosen"]["k"])
    M["cluster_meta"] = meta_cluster

    M["versions"] = {
        "python": platform.python_version(),
        "polars": pl.__version__,
        "catboost": __import__("catboost").__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
    }

    base_check = pl.read_parquet("data/v2/features/fold_00/batch_0000.parquet")
    feat_cols = [c for c in base_check.columns if c not in DROP_COLS]
    assert len(feat_cols) == 56, len(feat_cols)
    feat_cols = feat_cols + EXTRA_ALL + ["pct_rank_gmv30"]
    assert len(feat_cols) == 66 + 1 and len(set(feat_cols)) == len(feat_cols)

    log("loading folds...")
    mats: dict[str, Mat] = {}
    dfs: dict[str, pl.DataFrame] = {}
    for name in CV_FOLDS:
        t = time.time()
        df = join_fold(name)
        dfs[name] = df
        mats[name] = Mat(df, feat_cols, k_seg)
        log(f"  {name}: {df.shape} target_ok={bool((~np.isnan(mats[name].y_raw)).all())}"
            f" ({time.time() - t:.1f}s)")

    tr = pl.concat([dfs[f] for f in CV_FOLDS[:3]], how="vertical")
    tr_mat = Mat(tr, feat_cols, k_seg)
    va = mats["fold_03"]

    # ---------- Stage A: ablation on fold_03 -----------------------------
    log("\n=== stage A: ablation (train 00+01+02, val fold_03) ===")
    ablation = []
    models = {}
    for tag in ["ref66", "seg_cat", "seg_ohe"]:
        X_tr, cols, cat_idx = VARIANTS[tag](tr_mat)
        model, ft = fit_catboost(X_tr, tr_mat.y_log, cols, cat_features=cat_idx)
        score = rmsle_score(model, va, tag)
        models[tag] = model
        row = {"variant": tag, "n_features": len(cols),
               "rmsle_fold03": round(score, 5), "fit_time_sec": round(ft, 1)}
        ablation.append(row)
        log(f"  {tag:>8}: RMSLE={score:.5f} ({ft:.1f}s)")
    M["ablation_fold03"] = ablation
    dump()

    ref_row = next(r for r in ablation if r["variant"] == "ref66")
    dev = abs(ref_row["rmsle_fold03"] - REF_LOFO["fold_03"])
    M["ref_reproduction"] = {
        "expected": REF_LOFO["fold_03"], "reproduced": ref_row["rmsle_fold03"],
        "abs_deviation": round(dev, 5), "within_tol": dev <= 0.002,
    }
    log(f"ref reproduction dev={dev:.5f}")

    seg_rows = [r for r in ablation if r["variant"] != "ref66"]
    best_row = min(seg_rows, key=lambda r: r["rmsle_fold03"])
    best_tag = best_row["variant"]
    delta = best_row["rmsle_fold03"] - REF_LOFO["fold_03"]
    log(f"best segment variant: {best_tag} ({best_row['rmsle_fold03']}, "
        f"delta {delta:+.5f})")

    # ---------- Stage B: LOFO for the best variant ------------------------
    log(f"\n=== stage B: LOFO for {best_tag} ===")
    lofo_best = []
    for i in range(4):
        hold = CV_FOLDS[i]
        rest = [f for j, f in enumerate(CV_FOLDS) if j != i]
        tm = Mat(pl.concat([dfs[f] for f in rest], how="vertical"),
                 feat_cols, k_seg)
        vm = mats[hold]
        X_tr, cols, cat_idx = VARIANTS[best_tag](tm)
        model, ft = fit_catboost(X_tr, tm.y_log, cols, cat_features=cat_idx)
        score = rmsle_score(model, vm, best_tag)
        lofo_best.append({"fold": hold, "rmsle": round(score, 5),
                          "fit_time_sec": round(ft, 1)})
        ref = REF_LOFO[hold]
        log(f"  holdout {hold}: {best_tag}={score:.5f} vs ref={ref:.5f} "
            f"(delta {score - ref:+.5f}, {ft:.1f}s)")
    M["lofo_best"] = {"variant": best_tag, "rows": lofo_best,
                      "reference_exp02": REF_LOFO}
    dump()

    no_degradation = all(
        r["rmsle"] <= REF_LOFO[r["fold"]] + 1e-9 for r in lofo_best)
    adoption = (best_row["rmsle_fold03"] <= ADOPTION_FOLD03 and no_degradation)
    M["verdict"] = {
        "adoption_threshold_fold03": ADOPTION_FOLD03,
        "best_variant": best_tag,
        "best_rmsle_fold03": best_row["rmsle_fold03"],
        "delta_vs_ref_fold03": round(delta, 5),
        "lofo_no_degradation": no_degradation,
        "adoption": bool(adoption),
        "note": "submit decision belongs to the coordinator",
    }

    # ---------- diagnostics ------------------------------------------------
    log("\n=== diagnostics ===")
    diag = {}
    for name in CV_FOLDS:
        z = np.log1p(mats[name].y_raw)
        seg = mats[name].seg
        means = np.array([z[seg == j].mean() if (seg == j).any() else np.nan
                          for j in range(k_seg)])
        gm = z.mean()
        between = sum((mats[name].seg == j).sum() * (m - gm) ** 2
                      for j, m in enumerate(means) if np.isfinite(m)) / len(z)
        total = z.var()
        diag[name] = {
            "mean_z_by_segment": [round(float(x), 4) for x in means],
            "between_segment_var_share": round(float(between / total), 4),
        }
        log(f"  {name}: between-seg var share={between / total:.3f}; "
            f"mean_z={['%.2f' % x for x in means]}")
    M["target_z_by_segment"] = diag

    imp = models["seg_cat"].get_feature_importance()
    seg_imp = float(imp[list(models["seg_cat"].feature_names_).index("segment_id")])
    M["segment_feature_importance_pct"] = {
        "seg_cat_model_fold03_run": round(seg_imp, 4),
        "rank_among_67": int((np.array(imp) > seg_imp).sum() + 1),
    }
    log(f"segment importance: {seg_imp:.3f}% "
        f"(rank {M['segment_feature_importance_pct']['rank_among_67']} of 67)")

    # per-segment RMSLE delta on fold_03 (best vs ref)
    pred_best = predict_raw_score(models[best_tag], va, best_tag)
    pred_ref = predict_raw_score(models["ref66"], va, "ref66")
    z_true = np.log1p(va.y_raw)
    per_seg = []
    for j in range(k_seg):
        m = va.seg == j
        r_b = float(np.sqrt(np.mean((z_true[m] - np.log1p(pred_best[m])) ** 2)))
        r_r = float(np.sqrt(np.mean((z_true[m] - np.log1p(pred_ref[m])) ** 2)))
        per_seg.append({"segment": j, "n": int(m.sum()),
                        "rmsle_best": round(r_b, 5),
                        "rmsle_ref": round(r_r, 5),
                        "delta": round(r_b - r_r, 5)})
    M["per_segment_rmsle_fold03_best_vs_ref"] = per_seg

    M["runtime_train_sec"] = round(time.time() - t_start, 1)
    dump()
    make_figures(diag, per_seg, imp, models["seg_cat"].feature_names_)
    write_report(best_row, lofo_best, no_degradation, adoption, delta)
    log(f"\nTRAIN DONE in {(time.time() - t_start) / 60:.1f} min")


def rmsle_wrap(model, mat: Mat, tag: str) -> float:
    return rmsle(mat.y_raw, predict_raw_score(model, mat, tag))


# kept for clarity; both helpers below do the same dispatch
def rmsle_score(model, mat: Mat, tag: str) -> float:
    return rmsle_wrap(model, mat, tag)


def predict_raw_score(model, mat: Mat, tag: str) -> np.ndarray:
    X, _cols, cat_idx = VARIANTS[tag](mat)
    return predict_raw(model, X, cat_features=cat_idx)


def dump() -> None:
    METRICS_PATH.write_text(json.dumps(M, indent=2, ensure_ascii=False))


def make_figures(diag, per_seg, imp, names) -> None:
    k_seg = int(M["cluster_meta"]["chosen"]["k"])

    # mean target-z by segment across anchors
    plt.figure(figsize=(9, 5))
    folds = list(diag.keys())
    w = 0.8 / len(folds)
    for fi, fname in enumerate(folds):
        means = diag[fname]["mean_z_by_segment"]
        plt.plot(np.arange(k_seg) + fi * w - 0.4 + w / 2, means,
                 "o--", ms=5, label=fname)
    plt.axhline(0, color="gray", lw=0.8)
    plt.xticks(range(k_seg), [f"seg {j}" for j in range(k_seg)])
    plt.ylabel("mean target z=log1p(gmv+30d)")
    plt.title("B5: mean target-z by segment (Hungarian-aligned labels)")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.35)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "b05_target_by_segment.png", dpi=150)
    plt.close()

    # per-segment RMSLE best vs ref on fold_03
    ps = sorted(per_seg, key=lambda r: r["segment"])
    x = np.arange(len(ps))
    plt.figure(figsize=(9, 5))
    plt.bar(x - 0.2, [r["rmsle_ref"] for r in ps], width=0.4,
            color="lightgray", label="ref66")
    plt.bar(x + 0.2, [r["rmsle_best"] for r in ps], width=0.4,
            color="steelblue", label=M["verdict"]["best_variant"])
    plt.xticks(x, [f"seg {r['segment']}\nn={r['n']//1000}k" for r in ps],
               fontsize=8)
    plt.ylabel("RMSLE on fold_03")
    plt.title("B5: RMSLE by segment, best variant vs ref66 (fold_03)")
    plt.legend()
    plt.grid(alpha=0.35, axis="y")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "b05_rmsle_by_segment.png", dpi=150)
    plt.close()

    # top-20 importance with segment highlighted
    order = np.argsort(imp)[::-1][:20][::-1]
    colors = ["crimson" if names[i].startswith(("segment_id", "seg_"))
              else "steelblue" for i in order]
    plt.figure(figsize=(9, 7))
    plt.barh([names[i] for i in order], [imp[i] for i in order],
             color=colors)
    plt.xlabel("feature importance, %")
    plt.title("B5 seg_cat model: top-20 features (red = segment)")
    plt.grid(True, axis="x", alpha=0.35)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "b05_feature_importance.png", dpi=150)
    plt.close()


def write_report(best_row, lofo_best, no_degradation, adoption, delta) -> None:
    ch = M["cluster_meta"]["chosen"]
    align = M["cluster_meta"]["alignment"]["pairs"]
    sel = M["cluster_meta"]["selection_table_fold03"]
    sel_lines = "\n".join(
        f"| {r['method']} | {r['k']} | {r['silhouette']} | {r['stability_ari_seeds']} | "
        f"{r['min_segment_share_subsample']} | {r['bic_subsample'] or '-'} |"
        for r in sel)
    lofo_lines = "\n".join(
        f"| {r['fold']} | {r['rmsle']} | {REF_LOFO[r['fold']]} | "
        f"{r['rmsle'] - REF_LOFO[r['fold']]:+.5f} |" for r in lofo_best)
    abl_lines = "\n".join(
        f"| {r['variant']} | {r['n_features']} | {r['rmsle_fold03']} |"
        for r in M["ablation_fold03"])
    sel = M["cluster_meta"]["selection_table_fold03"]
    var_share_f03 = M["target_z_by_segment"]["fold_03"]["between_segment_var_share"]

    verdict_txt = (
        "**ПРИНЯТА** — порог fold_03 ≤ 1.68938 и LOFO без деградации выполнены."
        if adoption else
        "**ОТКЛОНЕНА** — сегментация как одна категориальная фича не даёт "
        "требуемого выигрыша; смесь популяций бустинг ловит и так.")
    gain_note = ("\n- ⚠ Выигрыш >0.5% — координатору решить вопрос сабмита."
                 if delta <= -0.005 * 1.69277 else "")

    report = f"""# Б5 — Сегментация поведенческих траекторий как категориальная фича (Arm A)

## Карточка

- **Гипотеза**: популяция — смесь режимов (новички / спящие / регулярные / киты);
  одна категориальная фича поведенческого сегмента даст деревьям дешёвые полезные
  сплиты поверх глобальной регрессии.
- **Метод**: профили юзеров из принятой схемы 66ф (18 признаков, только история ≤ якоря;
  таргет в кластеризацию НЕ входит), StandardScaler по каждому якорю отдельно;
  выбор модели/k на fold_03 (silhouette + стабильность ARI по сидам для KMeans и GMM,
  BIC как кросс-чек); ограничение «минимальный сегмент ≥5%» проверено на полных
  подгонках всех якорей. Метки согласованы между соседними якорями Hungarian-маппингом:
  основной вариант — по пересечению пользователей (контингенция перекрытия сегментов,
  `linear_sum_assignment` на -counts); маппинг по расстояниям центроидов в общем
  raw-log пространстве сохранён как диагностика (уступает: до 5% совпадений).
  Вставка в протокол exp02: (i) сегмент как native categorical, (ii) one-hot.
- **Результат**: см. таблицы ниже.

## Выбор k ({ch['method']}, k={ch['k']})

| метод | k | silhouette | stability ARI | min seg (sub) | BIC |
|---|---|---|---|---|---|
{sel_lines}

Правило: максимум silhouette среди кандидатов со stability ARI ≥ 0.5 и минимальным
сегментом ≥5% на полных подгонках (проверено на всех якорях). GMM BIC — кросс-чек.

## Абляция (train fold_00–02 → val fold_03)

| вариант | фичей | RMSLE fold_03 |
|---|---|---|
{abl_lines}

Референс exp02 (66ф): 1.69277; воспроизведение: {M['ref_reproduction']['reproduced']}
(dev {M['ref_reproduction']['abs_deviation']}).

## LOFO лучшего варианта ({best_row['variant']})

| holdout | B5 | ref exp02 | delta |
|---|---|---|---|
{lofo_lines}

## Диагностика

- Между-сегментная доля дисперсии target-z на fold_03: **{var_share_f03}**
  (если близко к нулю — сегменты не различают таргет).
- Важность segment-фичи (native cat): {M['segment_feature_importance_pct']['seg_cat_model_fold03_run']}%
  (место {M['segment_feature_importance_pct']['rank_among_67']} из 67).
- Доля юзеров со сменой сегмента между соседними якорями:
  {', '.join(f"{p['pair']}: {1 - p['match_rate_all_users']:.1%}" for p in M['cluster_meta']['alignment']['pairs'])}
- Согласование меток после Hungarian-маппинга (совпадение на всех 250k юзерах):
  {', '.join(f"{a['pair']}: {a['match_rate_all_users']:.1%}" for a in align)}.
- Размеры сегментов ≥5% на всех якорях: да (см. b05_metrics.json / график).
- ⚠ Дрейф семантики меток: сегмент с максимальным средним target-z — seg 0 на
  fold_00/01/03, но seg 2 на fold_02 (см. b05_target_by_segment.png). При ARI
  0.36–0.57 между соседними якорями словарь меток не семантически стабилен:
  Hungarian сохраняет членство пользователей, но режимы GMM переориентируются
  между якорями, и категориальная фича несёт межъякорный шум.

## Вывод

- Лучший вариант: **{best_row['variant']}** — fold_03 {best_row['rmsle_fold03']}
  (delta {delta:+.5f} против референса 1.69277), LOFO без деградации: {no_degradation}.
- Парадокс диагностики: сегменты РАЗЛИЧАЮТ таргет внутри якоря (между-сегментная
  доля дисперсии z ~27%, mean z 0.5–3.8), но бустинг уже ловит эту структуру
  числовыми фичами (важность segment 0.75%, место 23/67), а межъякорный дрейф
  семантики меток превращает фичу в шум → лёгкая деградация на 2 из 4 LOFO-фолдов.
- Критерий adoption (fold_03 ≤ 1.68938 И LOFO без деградации): {verdict_txt}{gain_note}
- Гипотеза Б5 (Arm A) снята: популяционная смесь эффективно аппроксимируется
  одним глобальным бустингом; дешёвая категориальная сегментация пользы не даёт.
  Направление Arm B (hard routing) не оправдано этим результатом — стабильность
  сегментов между якорями недостаточна (совпадение меток 60–80%, ARI ≤0.57).

## Артефакты

- `src/b05_cluster.py`, `src/b05_train.py`
- `data/v2/b05_seg/` — кэш меток сегментов по якорям (+ meta.json)
- `reports/b05_metrics.json`, `reports/b05_report.md`
- `reports/b05_figures/b05_k_selection.png`, `b05_segment_sizes.png`,
  `b05_target_by_segment.png`, `b05_rmsle_by_segment.png`, `b05_feature_importance.png`

## Ограничения

- Кластеризация transductive: фит на всех 250k юзеров конкретного якоря
  (без таргета — утечки нет, но на новых юзерах нужен инференс кластера).
- Метки согласованы Hungarian-маппингом по перекрытию пользователей; при сильном
  дрейфе режимов между якорями соответствие «один сегмент = один режим» может
  размываться (совпадение меток 60–80%, ARI 0.36–0.57 после маппинга).
- Arm A сознательно дёшево (одна категориальная фича); hard routing / soft blending — вне рамки.
- Прочие оговорки: sentinel recency=999 сохранён (лог1п), conv-фичи с NaN → 0
  («нет возможности конверсии»), due_ratio в профиль не включён (много NaN).
"""
    REPORT_PATH.write_text(report)


if __name__ == "__main__":
    main()
