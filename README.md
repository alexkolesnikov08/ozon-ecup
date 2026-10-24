# Ozon Search LTV (ecup)

Предсказание **LTV пользователей Ozon**: суммарный GMV каждого пользователя за
30 дней после конца обучающего периода (**2026-02-14 → 2026-03-15**), по истории
его поискового и покупательского поведения.

Метрика — **RMSLE**:

```
RMSLE = sqrt( mean( (log1p(y_true) - log1p(y_pred))^2 ) )
```

Лог-метрика ⇒ оптимален предикт условного среднего в z-пространстве
`z = log1p(y)` (квадратичный лосс на z ≡ RMSLE), инверсия `expm1` один раз в
конце.

Хакатон завершён. Репозиторий закрыт на лучшем результате и приведён в порядок:
результаты всех экспериментов — в журнале, код — рядом с своим экспериментом,
правила — в `docs/`.

## Точка входа

| Хочу | Открыть |
|---|---|
| Что уже пробовали и что выиграло | [EXPERIMENTS.md](EXPERIMENTS.md) |
| Протокол валидации, метрика, правила честности | [docs/PROTOCOL.md](docs/PROTOCOL.md) |
| Данные, таргет, feature store | [docs/DATA.md](docs/DATA.md) |
| Прогнать сегментированный пайплайн | [docs/SEGMENTED_PIPELINE_RUNBOOK.md](docs/SEGMENTED_PIPELINE_RUNBOOK.md) |
| Настроить удалённый сервер | [docs/SERVER.md](docs/SERVER.md) |
| Что пробовать дальше | [notes/backlog.md](notes/backlog.md) |
| Карта всей документации | [docs/INDEX.md](docs/INDEX.md) |
| Как устроен код | [src/README.md](src/README.md) |
| Как вести репозиторий | [CONTRIBUTING.md](CONTRIBUTING.md) |

## Результат

**Чемпион по лидерборду — стек v2 + OOF-калибровка (exp15), LB 1.66035**,
локальный `fold_03` 1.66908.

**Лучший локальный CV — пер-юзерная персонализация (exp23), 1.62463**, но на
лидерборде она переобучилась (LB 1.6862); бленд 50/50 с чемпионом даёт ~1.66.

Прод-база — сегментированный классический ансамбль (exp26), `fold_03` 1.66916.

Полный журнал: [EXPERIMENTS.md](EXPERIMENTS.md) · Справочные точки `fold_03`:
zero 3.20364 · median 2.28900 · naive 2.19506.

## Данные

`data/train.parquet` (~172 МБ, **в git не попадает** — лимит GitHub 100 МБ):
дневные агрегаты поведения, 30 631 006 строк × 18 колонок, 250 000 пользователей,
период 2025-01-01 → 2026-02-13, пропусков нет.

| Группа | Колонки | Смысл |
|---|---|---|
| Ключи | `event_date`, `user_id` | дата и пользователь |
| Воронка поиска | `search`, `search_to_cart`, `search_to_ord`, `has_search_to_cart`, `has_search_to_ord` | события через поиск |
| Воронка категорий | `cat`, `cat_to_cart`, `cat_to_ord`, `has_cat_to_cart`, `has_cat_to_ord` | события через категории |
| Итоги дня | `to_cart`, `to_ord`, `gmv`, `searches` | агрегаты по пользователю |
| Выручка | `gmv_search`, `gmv_cat` | GMV в разрезе источника |

Таргета в трейне нет — он конструируется как сумма `gmv` за окно
`[anchor+1, anchor+30]`. Сабмит: `sample_submit.csv` (`user_id,predict`), один
предикт на каждого из 250k юзеров.

Схема фичевых датасетов — [data/v2/MANIFEST.md](data/v2/MANIFEST.md) и
[data/v3/MANIFEST.md](data/v3/MANIFEST.md).

## Быстрый старт

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements/base.txt

# 0) положить train.parquet в data/ (в гите его нет)
ls -lh data/train.parquet

# 1) фичевые датасеты (base + x_* + pca32 + btyd), ~5 мин
.venv/bin/python src/build_features_ext_pca.py
.venv/bin/python src/build_features_bgnbd.py

# 2) чемпионский стек + сабмит
.venv/bin/python archive/exp15/src/train_stack_v2.py

# 3) сегментированный пайплайн (CPU) — база прод-сабмита
bash archive/exp26/scripts/run_segmented_pipeline.sh all

# 4) полный ансамбль с Transformer (нужен torch)
.venv/bin/python -m pip install -r requirements/transformer.txt
bash archive/exp26/scripts/run_full_segmented_ensemble.sh
```

Подробности — [docs/SEGMENTED_PIPELINE_RUNBOOK.md](docs/SEGMENTED_PIPELINE_RUNBOOK.md).

Скрипты в `src/` запускаются **из любого каталога** (пути строятся от корня
репозитория через `src/common/paths.py`). Скрипты в `archive/` — **из корня
репозитория**, они используют относительные пути.

Специального железа не требуется: тяжёлые шаги батчируются по 50k юзеров,
обучение на CPU занимает минуты. Для NN-экспериментов (exp05) может
понадобиться GPU (MPS/CUDA). Matplotlib в скриптах — только с бэкендом `Agg`.

## Структура репозитория

```
ozon-ecup/
├── README.md                     # этот файл — точка входа
├── EXPERIMENTS.md                # журнал: лидерборд + карточки exp00–exp26
├── CONTRIBUTING.md               # как вести репозиторий, как заводить эксперимент
│
├── docs/                         # стабильные правила
│   ├── INDEX.md                  # карта документации
│   ├── PROTOCOL.md               # метрика, фолды, правила честности
│   ├── DATA.md                   # данные, таргет, feature store
│   ├── SERVER.md                 # удалённый сервер, чек-листы
│   └── SEGMENTED_PIPELINE_RUNBOOK.md
│
├── notes/                        # живое и незакрытое
│   ├── README.md
│   ├── backlog.md                # открытые гипотезы
│   ├── reviews/2026-08-25.md     # разбор внешних гипотез
│   └── incidents/                # инцидент с серверными метриками
│
├── src/                          # активный код (см. src/README.md)
│   ├── README.md
│   ├── common/paths.py           # единая точка правды для путей
│   ├── eda.py                    # EDA сырых данных
│   ├── build_features_ext_pca.py # → data/v2/features_ext
│   ├── build_features_bgnbd.py   # → data/v2/features_bgnbd
│   ├── exp03_hurdle_priors.py    # приоры хёрдла (exp03, не завершён)
│   ├── exp1000.py                # фабрика экспериментов (exp09)
│   ├── exp11_global_lags.py      # не заведён в журнал
│   └── train_btyd.py             # соло-диагностика BTYD (exp04)
│
├── archive/                      # закрытые эксперименты, каждый со своей карточкой
│   ├── exp01/ … exp26p/          # expNN/{README.md,src,reports,submissions}
│
├── data/                         # НЕ в гите (в репо только манифесты)
│   ├── train.parquet             #   исходные данные (~172 МБ)
│   ├── v2/MANIFEST.md            #   схема фичевых датасетов v2
│   └── v3/MANIFEST.md            #   схема очищенного датасета v3
│
├── reports/                      # отчёты активных экспериментов
│   ├── eda_2026_08_24.md         #   снимок EDA
│   └── exp03_hurdle_priors.json  #   приоры хёрдла
│
├── requirements/                 # base.txt + transformer.txt
├── submissions/                  # черновики ручных сабмитов (в гите не попадают);
│                                 # сабмиты экспериментов — в archive/expNN/submissions/
├── sample_submit.csv             # формат сабмита
├── baseline-search-ltv.ipynb     # исходный бейзлайн организаторов
├── pyproject.toml                # конфигурация ruff и basedpyright
└── .pre-commit-config.yaml
```

## Правила размещения

- **Новый эксперимент** живёт в `src/`. После закрытия (`done` / `rejected`)
  переносится в `archive/expNN/` вместе с отчётами и сабмитом, получает
  следующее свободное имя `expNN` и карточку в журнале.
- **Каждый эксперимент владеет своими артефактами**: отчёт закрытого
  эксперимента лежит в `archive/expNN/reports/`, корень `reports/` — только для
  активных.
- **Один эксперимент — одна карточка** в [EXPERIMENTS.md](EXPERIMENTS.md):
  гипотеза → метод → результат → вывод → артефакты. Исторические имена
  (`b01`, `b03`, `b05`, `e2`, `stack`, `exp025`) сохранены как алиасы.
- **Критерий adoption формулируется до запуска**; сид 42; сравнение
  конфигураций — только на одном наборе фолдов; никакой информации после
  якоря. Подробности — [docs/PROTOCOL.md](docs/PROTOCOL.md).
- **Тяжёлые артефакты** (паркеты фичей, сырые данные, CSV-сабмиты, чекпоинты)
  в git не попадают и регенерируются скриптами.

Подробности — [CONTRIBUTING.md](CONTRIBUTING.md).
