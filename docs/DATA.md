# Данные

## Исходные данные

`data/train.parquet` — дневные агрегаты поведения пользователей.
**~172 МБ, в git не попадает** (лимит GitHub 100 МБ) — файл нужно положить
вручную.

- 30 631 006 строк × 18 колонок, 250 000 пользователей
- Период: 2025-01-01 → 2026-02-13, пропусков нет, дублей `(user_id, event_date)` нет

| Группа | Колонки | Смысл |
|---|---|---|
| Ключи | `event_date`, `user_id` | дата и пользователь |
| Воронка поиска | `search`, `search_to_cart`, `search_to_ord`, `has_search_to_cart`, `has_search_to_ord` | события через поиск |
| Воронка категорий | `cat`, `cat_to_cart`, `cat_to_ord`, `has_cat_to_cart`, `has_cat_to_ord` | события через категории |
| Итоги дня | `to_cart`, `to_ord`, `gmv`, `searches` | агрегаты по пользователю |
| Выручка | `gmv_search`, `gmv_cat` | GMV в разрезе источника |

## Таргет

В трене таргета нет — он конструируется:

```
target(anchor) = sum(gmv) за [anchor+1, anchor+30]
```

Окно всегда 30 дней, оканчивается за день до следующего якоря. В `fold_end`
таргет `NULL` — это прод-инференс.

## Feature store

Тяжёлые датасеты в git не попадают, но их **схема зафиксирована** в
манифестах — они и в репозитории:

| Хранилище | Что внутри | Манифест |
|---|---|---|
| `data/v2/features_ext/` | base70 + x_\*28 + pca32, по фолдам `fold_00..03` + `fold_end`, батчами по 50k | [data/v2/MANIFEST.md](../data/v2/MANIFEST.md) |
| `data/v2/features_bgnbd/` | BTYD-фичи (по фолду один parquet) + `fit_params.json` | там же |
| `data/v2/features_exp02/` | кэш exp02 (алиас, дублирует `features_ext`) | там же |
| `data/v2/b01_pseudo/`, `b03_plsi/`, `b05_seg/` | артефакты закрытых экспериментов | там же |
| `data/v3/` | очищенный датасет: 85 фичей + ключи + таргет | [data/v3/MANIFEST.md](../data/v3/MANIFEST.md) |
| `data/segmented_base/` | 205 признаков, 26-недельные ряды, ADI/CV² классы (exp26) | `manifest.json` внутри каталога |

### Сборка обучающей таблицы

```python
import polars as pl

feats = pl.read_parquet("data/v2/features_ext/fold_03/batch_*.parquet")
btyd  = pl.read_parquet("data/v2/features_bgnbd/fold_03.parquet")
df = feats.join(btyd, on=["anchor_date", "user_id"], how="left")
```

Train = concat(`fold_00..02`), val = `fold_03`, прод-инференс = `fold_end`.
Лосс — RMSE на `z = log1p(target)`, финальный предикт `expm1`.

## Что пересобирать и как

```bash
.venv/bin/python src/build_features_ext_pca.py      # ~4 мин → data/v2/features_ext
.venv/bin/python src/build_features_bgnbd.py        # ~1 мин → data/v2/features_bgnbd
```

Сегментированный feature store строится отдельно, из сырого parquet в обход
`data/v2|v3`:

```bash
bash archive/exp26/scripts/run_segmented_pipeline.sh features
```

## Известные особенности данных

- **Прореженные ряды**: 18.2% строк — «пустые» дни активности (все с `cat=1`),
  интерпретируются как слабый интент-сигнал.
- **Расхождения сплита**: `gmv ≠ gmv_search + gmv_cat` в 0.25% строк.
  Решение об исключении из таргета не принято — см. `notes/backlog.md`.
- **Сезонность**: февральско-мартовское окно сабмита приподнято —
  `s_seasonal = 1.1628` (YoY к календарному близнецу 2025). Учтено в сабмите
  exp02 через `β = 1`.
- **Дрейф**: платформа растёт ~46% YoY; локальный даунтафт перед якорем
  сабмита (`local_drift = 0.8094`) — следствие нормализации декабрьского пика.

## Почему данные не в git

| Артефакт | Размер | Регенерируется |
|---|---|---|
| `data/train.parquet` | ~172 МБ | нельзя, только из организаторов |
| `data/v2/features_ext/` | десятки МБ | `src/build_features_ext_pca.py`, ~4 мин |
| `data/v3/` | ~100 МБ | `archive/exp19/src/build_dataset_v3.py` |
| `data/segmented_base/` | ~1 ГБ | `archive/exp26/scripts/run_segmented_pipeline.sh features` |
| `submissions/*.csv` | ~6 МБ каждый | скриптами эксперимента |
