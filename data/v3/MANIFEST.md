# Манифест датасета v3 (утверждён голосованием 2026-08-24)

Генератор: `src/build_dataset_v3.py` → `data/v3/fold_{00..03,end}.parquet`
Отчёт сборки: `reports/dataset_v3_report.json`. Окружение: `.venv`.

## Схема (85 фичей + ключи + таргет = 88 колонок на фолд)

| Блок | Кол. | Состав |
|---|---|---|
| Ключи | 2 | `anchor_date`, `user_id` |
| Окна base | 40 | sum/mean × {gmv, searches, to_ord, to_cart} × {7d,14d,30d,90d} + max × {30d,90d} |
| Профиль | 11 | active_days_30d, tenure_days, row_days_total, has_ord/search_before_anchor, recency_*_capped90, conv×3_90d, gmv_per_order_90d |
| x_* интент/EWMA | 13 | корзины-без-заказа, визиты-без-действий, search_days, intent_no_ord, gmv_search/cat_sum_30d, cat_to_ord/cart_30d, conv_s2o_14d, due_ratio, gmv-EWMA h7/h30, momentum_gmv |
| PCA | 16 | pca_00..15 |
| BTYD | 5 | bgnbd_p_alive, bgnbd_en30, eb_lambda_n30, bgnbd_e_gmv30, eb_e_gmv30 |
| Таргет | 1 | сумма gmv за [anchor+1, anchor+30]; в `fold_end` — NULL |

Плотность: **0 NULL во всех фичах** всех фолдов; дубликатов ключей нет.

## Лог решений (раунды голосования)

1. Дубликаты `x_{gmv,searches,to_ord,cart}_sum_14d ≡ base` (100% совпадение) → выкинуты x_-копии.
2. Слабые (corr_z ≤ .03 и/или 43% NaN): slopes×2, share_gmv_search/cat, x_aov_30d,
   gg_e_value, x_conv_c2o_30d → выкинуты.
3. BTYD-входы bgnbd_T/n_occasions/mon_freq/mbar (копии base или corr_z≈.06) → выкинуты;
   носитель частоты — `eb_lambda_n30` вместо order_days_total (corr .983, EB-шринкейдж гранулярнее).
4. Ступень окон 60d выкинута целиком (зажата 30d/90d, corr .92–.98);
   max оставлен только на 30/90d (max==sum у 91%/82% юзеров на 7/14d).
5. EWMA только gmv-ветка (чек плоский ⇒ ветки дубль, corr .93–.97);
   `x_search_to_ord_30d` (corr .992 с to_ord_sum_30d) и `x_gmv_share_14_of_30` → долой.
6. PCA 32→16 (сигнал в pca_00–01; хвост 17..31 corr≈0).
7. recency-сентинел 999 («никогда», 13.5% юзеров) → пара
   (`has_ord_before_anchor` Int8 + `recency_to_ord_capped90` = min(days, 90)); то же для searches.
8. Джойн BTYD inner: fold_00 теряет 2071 строку (юзеры, появившиеся после якоря), остальные фолды полные.

## Использование

```python
import polars as pl
train = pl.read_parquet("data/v3/fold_00.parquet")   # + fold_01, fold_02
val   = pl.read_parquet("data/v3/fold_03.parquet")   # холдаут, основной критерий
prod  = pl.read_parquet("data/v3/fold_end.parquet")  # инференс сабмита (target=NULL)
```

Протокол как в exp01/exp1000: трейн fold_00..02 (+fold_03 при рефите), лосс RMSE на
z=log1p(target), предикт expm1. Референсные точки fold_03: zero 3.20364, median 2.28900,
naive 2.19506, CatBoost exp01 1.70261, стек v2 1.67103.

Калибровка уровня поверх предиктов — `reports/exp06_calibration.json`,
hurdle-приоры — `reports/exp03_hurdle_priors.json`.
