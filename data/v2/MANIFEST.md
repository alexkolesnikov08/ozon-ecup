# Манифест артефактов данных (handoff для обучения)

Дата: 2026-08-24. Автор: этап данных (EDA + трансформации). Обучение и сборка бейзлайна —
следующий этап. Все пути от корня репо. Окружение: `.venv` (polars, sklearn, scipy, catboost).

## Карта экспериментов → артефакты

| Эксперимент | Артефакт | Генератор |
|---|---|---|
| exp02 — расширенные фичи | `data/v2/features_ext/fold_*/batch_*.parquet` (блок `x_*`) | `src/build_features_ext_pca.py` |
| exp07 — PCA дневных панелей | тот же файл (блок `pca_00..31`) | там же |
| exp04 — BTYD фичи | `data/v2/features_bgnbd/<fold>.parquet` + `fit_params.json` | `src/build_features_bgnbd.py` |
| exp03 — hurdle приоры | `reports/exp03_hurdle_priors.json` | `src/exp03_hurdle_priors.py` |
| exp06 — калибровка уровня | `reports/exp06_calibration.json` | `src/exp06_calibration_indices.py` |

## Схема `features_ext` (250k строк × 133 колонки, по фолду fold_00..03 + fold_end)

- Ключи: `anchor_date`, `user_id`, таргет `target` (= сумма gmv за [anchor+1, anchor+30];
  в `fold_end` — NULL).
- **base (70 кол.)**: оконные sum/max/mean по gmv/searches/to_ord/to_cart за 7/14/30/60/90d,
  active_days_30d, recency_to_ord/searches_days (999 = никогда), tenure_days, order_days_total,
  row_days_total, conv-ratios 90d. NaN → 0 (паритет с exp01, кроме 14d окна — оно новое).
- **x_* (28 кол.)** — интент/EWMA/тренды/частота:
  интент: x_searches_sum_14d, x_cart_sum_14d, x_ord_sum_14d, x_gmv_sum_14d, x_cart_no_ord_days_14d,
  x_visit_only_days_30d (визиты каталога без действий), x_search_days_14d, x_intent_no_ord_14d;
  источники: x_gmv_search/cat_sum_30d, доли x_share_gmv_search/cat_30d, конверсии x_conv_s2o/c2o;
  EWMA (log-space): x_gmv/ord_ewma_h7/h30, моментумы x_ewma_momentum_gmv/ord;
  тренды: x_gmv_slope_56d, x_ord_slope_56d, x_gmv_share_14_of_30;
  частота: order_days_total, x_aov_30d, x_due_ratio.
  Ratio-фичи могут быть NaN (нулевой знаменатель) — CatBoost нативно; для других моделей заполнить.
- **pca_00..pca_31**: IncrementalPCA(32) поверх стандартизованной плотной панели
  [56 дней × log1p(gmv, searches, to_ord, to_cart)] = 224 dims → 32 компоненты (EVR = 0.514).
  Scaler+PCA фитились ТОЛЬКО на пре-якорных окнах train-фолдов fold_00..02 (утечки нет);
  компоненты для fold_03/fold_end — чистый transform.

## Схема `features_bgnbd` (по фолду, один parquet)

Ключи: `anchor_date`, `user_id`. Колонки: входы (bgnbd_T, bgnbd_tx, bgnbd_n_occasions,
bgnbd_mon_freq, bgnbd_mbar) + предикты: bgnbd_p_alive, bgnbd_en30 (BG/NBD hyp2f1-формула),
eb_lambda_n30 (Gamma-Poisson EB, устойчиво на холодных), gg_e_value (Gamma-Gamma шринкейдж чека),
произведения bgnbd_e_gmv30 / eb_e_gmv30.
Параметры MLE на pooled train-фолдах: r=1.251, α=19.2, a=0.135, b=1.238; GG: p=0.071, q=3.02,
v=1546 (⚠️ p<1 ⇒ веса почти всегда 0 → gg_e_value ≈ популяционному среднему 54; использовать
как слабую фичу/константу шринкейжа). ⚠️ В fold_00 отсутствуют юзеры, впервые появившиеся после
якоря (247,929 из 250k) — джойнить как left + fill_null или inner.

## Как собрать обучающую таблицу

```python
import polars as pl
feats = pl.read_parquet("data/v2/features_ext/fold_03/batch_*.parquet")
btyd  = pl.read_parquet("data/v2/features_bgnbd/fold_03.parquet")
df = feats.join(btyd, on=["anchor_date", "user_id"], how="left")
```

Train = concat(fold_00..02), val = fold_03, прод-инференс = fold_end. Лосс: RMSE на z=log1p(target),
финальный предикт expm1. Референсные точки fold_03: zero=3.20364, median=2.28900, naive=2.19506,
CatBoost(exp01)=1.70261.

## Калибровка после обучения (exp06)

`z' = z + c + β·log(s)`; индексы в reports/exp06_calibration.json:
s_seasonal=1.1628 (сезонный аплифт тестового окна), yoy_cohort_user=1.3067 (структурный рост),
local_drift=0.8094 (локальный даунтафт перед якорем сабмита — декабрьский пик нормализуется).
Сетки c/β — внутри JSON. Приоры хёрдла (P(buy), E[z|buy] по сегментам рекенси×интент) —
в reports/exp03_hurdle_priors.json.

## Воспроизведение

```bash
.venv/bin/python src/build_features_ext_pca.py      # ~4 мин
.venv/bin/python src/build_features_bgnbd.py        # ~1 мин
.venv/bin/python src/exp03_hurdle_priors.py         # ~1 мин
.venv/bin/python src/exp06_calibration_indices.py   # ~1 мин
```
