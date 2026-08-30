# Ozon Search LTV (ecup)

Предсказание **LTV пользователей Ozon**: суммарный GMV каждого пользователя за 30 дней после
конца обучающего периода (**2026-02-14 → 2026-03-15**), по истории его поискового и
покупательского поведения.

Метрика — **RMSLE**:

```
RMSLE = sqrt( mean( (log1p(y_true) - log1p(y_pred))^2 ) )
```

Лог-метрика ⇒ оптимален предикт условного среднего в z-пространстве `z = log1p(y)`
(квадратичный лосс на z ≡ RMSLE), инверсия `expm1` один раз в конце.

## Данные

`data/train.parquet` (~172 МБ, **не в гите** из-за лимита GitHub 100 МБ): дневные агрегаты
поведения, 30 631 006 строк × 18 колонок, 250 000 пользователей, период 2025-01-01 → 2026-02-13,
пропусков нет.

| Группа | Колонки | Смысл |
|---|---|---|
| Ключи | `event_date`, `user_id` | дата и пользователь |
| Воронка поиска | `search`, `search_to_cart`, `search_to_ord`, `has_search_to_cart`, `has_search_to_ord` | события через поиск |
| Воронка категорий | `cat`, `cat_to_cart`, `cat_to_ord`, `has_cat_to_cart`, `has_cat_to_ord` | события через категории |
| Итоги дня | `to_cart`, `to_ord`, `gmv`, `searches` | агрегаты по пользователю |
| Выручка | `gmv_search`, `gmv_cat` | GMV в разрезе источника |

Таргета в трейне нет — конструируется как сумма `gmv` за окно `[anchor+1, anchor+30]`.
Сабмит: `sample_submit.csv` (`user_id,predict`), один предикт на каждого из 250k юзеров.
EDA сырых данных — `reports/eda_2026_08_24.md`; схема фичевых датасетов —
[data/v2/MANIFEST.md](data/v2/MANIFEST.md).

## Валидация (единый протокол всех экспериментов)

Time-CV: якоря fold_00..03 = `2025-12-03 / 12-17 / 12-31 / 2026-01-14` с шагом 14 дней;
обучение fold_00..02, холдаут **fold_03** — основной критерий лидерборда; прод-якорь
сабмита — fold_end (`2026-02-13`). Обучение — RMSE на `z = log1p(target)`, сид 42,
никакой информации после якоря.

Справочные точки fold_03: zero 3.20364 · median 2.28900 · naive 2.19506.

## Текущее состояние

Чемпион по LB: **exp15 — стек v2 + калибровка, LB 1.66035** (CV fold_03 1.66908). Лучший CV — **exp23 per-user CatBoost n=3 it20, CV 1.62463** (LB 1.686 — переобучение). Полный журнал — в [EXPERIMENTS.md](EXPERIMENTS.md).

| # | Эксперимент | RMSLE (fold_03) | Статус |
|---|---|---|---|
| 0 | Наивный автогресс | 2.19506 | done |
| 1 | CatBoost на оконных агрегатах | 1.70261 | done |
| 2 | GBDT: расширенные фичи + YoY | 1.69277 | done |
| 8 | Стек v1 LGBM+Hist+Cat+Ridge | 1.67103 | done |
| 13 | Абляция фичевых блоков (соло CatBoost, 141f) | 1.67027 | done |
| **15** | **Стек v2 + OOF-калибровка** | **1.66908** | **LB champion 1.660** |
| 17 | Тренды / брошенные корзины / сезонность | 1.66847 | done |
| **23** | **Per-user CatBoost n=3 it20 (47 мин)** | **1.62463** | **CV champion (LB 1.686)** |
| 24 | Per-user n=5 it25 dense (1k pilot) | 1.63498 | done |
| 25 | Heavy 6h per-user n=3 it50 (planned) | 1.638 (1k) | planned |
| 26 | Segmented LTV + Transformer blend (PR #5) | 1.675 (quick) | done |
| 26p | Pilots per-user/dense/lag/transformer | — | archive |

Дальше: стабилизация per-user (blend 0.5 → LB 1.66) + segmented ансамбль на CPU/GPU.

## Структура репозитория

```
ozon/
├── README.md                      # этот файл
├── EXPERIMENTS.md                 # журнал exp00–26: лидерборд + карточки
├── docs/SEGMENTED_PIPELINE_RUNBOOK.md # запуск сегментированного пайплайна (PR #5)
├── baseline-search-ltv.ipynb      # исходный бейзлайн организаторов
├── sample_submit.csv              # формат сабмита
├── data/                          # НЕ в гите:
│   ├── train.parquet              #   исходные данные (~172 МБ)
│   ├── v2/                        #   фичевые датасеты по фолдам (см. v2/MANIFEST.md)
│   ├── v3/                        #   очищенный датасет 85f (exp19, см. v3/MANIFEST.md)
│   └── segmented_base/            #   кэш сегментированного пайплайна (игнор)
├── reports/                       # живые отчёты: EDA + приоры + per-user/segm
│   ├── figures/                   #   графики (per_user_*.png, exp10_fold_*.png)
│   └── *.json                     #   dataset_v3, exp10/14, per_user_* итд (ckpt_*/ *.pt игнор)
├── src/                           # активные инструменты:
│   ├── eda.py
│   ├── build_features_ext_pca.py  #   x_* + pca → data/v2/features_ext/
│   ├── build_features_bgnbd.py    #   BTYD → data/v2/features_bgnbd/
│   ├── build_dataset_v3.py        #   очистка 141f → 85f → data/v3/
│   ├── build_segmented_features.py#   segmented: 26 недель + ADI/CV² классы
│   ├── train_segmented_submit.py  #   иерархия global/class/cluster + blend
│   ├── train_weekly_transformer.py#   weekly Transformer (26x4) branch
│   ├── blend_segmented_transformer.py
│   ├── per_user_full.py           #   exp23 n=3 it20 (1.624)
│   ├── per_user_full_n5_it25.py   #   exp24 n=5 it25
│   ├── exp1000.py                 #   фабрика 1006 конфигов (exp09)
│   └── requirements_*.txt         #   segmented / transformer deps
└── archive/                       # закрытые эксперименты — история:
    ├── exp01/                     #   оконные агрегаты + CatBoost (1.70261)
    ├── exp02/                     #   расширенные фичи + YoY (1.69277)
    ├── exp08/                     #   стек v1 (1.67103)
    ├── exp10..exp12/              #   wave1 b01/b03/b05 — rejected
    ├── exp13/                     #   абляция блоков (1.67027)
    ├── exp14/                     #   CatBoost гиперы (1.70124)
    ├── exp15/                     #   стек v2 + калибровка — LB CHAMPION
    ├── exp16/                     #   диагностика accuracy
    ├── exp23/                     #   per-user 6h heavy spec (planned)
    └── exp26_pilots/              #   диагностика per-user/dense/lag/transformer
```

Каждый закрытый эксперимент лежит в `archive/expNN/{src,reports,submissions}`.
Исторические имена (`b01`, `e2`, `stack`, `exp025`) — алиасы, указаны в карточках журнала.

Правила размещения:

- Новый эксперимент работает в `src/`; после закрытия переносится в `archive/expNN/`
  вместе с отчётами и сабмитом, получает следующее свободное имя `expNN`.
- Скрипты используют относительные пути и **запускаются из корня репо**: читают
  `data/v2/...`, пишут `reports/` и `submissions/` в корне.
- Тяжёлые артефакты (паркеты фичей, сырые данные, CSV-сабмиты) в гит не попадают.

## Как воспроизвести

Окружение — `.venv` (Python 3.12: polars, scikit-learn, catboost, lightgbm, xgboost).
Запуск скриптов — из корня репо: `.venv/bin/python <скрипт>`. Пакеты добавлять:
`uv pip install --python .venv/bin/python <pkg>`.

```bash
# 0) положить train.parquet в data/ (в гите его нет)

# 1) фичевые датасеты полной таблицы (base + x_* + pca32 + btyd)
.venv/bin/python src/build_features_ext_pca.py      # ~4 мин
.venv/bin/python src/build_features_bgnbd.py        # ~1 мин

# 2) чемпионский стек + сабмит -> submissions/submission_stack_v2*.csv
.venv/bin/python archive/exp15/src/train_stack_v2.py

# 3) сегментированный пайплайн (CPU) -> submissions/submission_segmented.csv
bash scripts/run_segmented_pipeline.sh all           # классика, 16 потоков
# полный ансамбль с Transformer (GPU):
.venv/bin/python -m pip install -r requirements_transformer.txt
bash scripts/run_full_segmented_ensemble.sh          # -> submission_segmented_transformer.csv

# 4) per-user персоналка (47 мин на M1, 1.624 CV)
.venv/bin/python src/per_user_full.py                # n=3 it20
.venv/bin/python src/per_user_full_n5_it25.py        # n=5 it25

# пример одиночного эксперимента (exp01):
#   .venv/bin/python archive/exp01/src/features.py
#   .venv/bin/python archive/exp01/src/train.py
```

Специального железа не требуется: тяжёлые шаги батчируются по 50k юзеров, обучение
на CPU занимает минуты. Для NN-экспериментов (exp05) может понадобиться GPU (MPS/CUDA).
Matplotlib в скриптах — только с бэкендом `Agg`.

## Правила работы

1. **Один эксперимент = одна запись `expNN` в [EXPERIMENTS.md](EXPERIMENTS.md)**:
   карточка (гипотеза → метод → результат → вывод → артефакты) + строка в лидерборде.
   Исторические имена сохраняются как алиасы. Статусы:
   `planned → running → done / rejected` (+ `data ready` / `priors ok` / `infra`
   для подготовительных записей).
2. **Код и изменения льём в git** (`origin` =
   https://github.com/alexkolesnikov08/ozon-ecup): пушим после каждого значимого результата.
3. Честность валидации: никакой информации после якоря в фичах; сравнение конфигураций —
   только на одном наборе фолдов; сиды фиксируем; критерий adoption формулируем до запуска.
4. Перед коммитом проверять `git status`: в репо не должно быть мусора
   (`__pycache__`, `.DS_Store`, `catboost_info/`, логи, временные файлы).
