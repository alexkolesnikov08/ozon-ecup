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

Сабмит: `sample_submit.csv` (`user_id,predict`), по одному предикту на каждого из 250k юзеров.

## Валидация

Time-CV из бейзлайна: якоря с шагом 14 дней, таргет = GMV за следующие 30 дней.

- Основной критерий — **CV RMSLE на fold_03** (якорь `2026-01-14`, последний с полным таргетом).
- Фолды: fold_00..fold_03 = якоря `2025-12-03 / 2025-12-17 / 2025-12-31 / 2026-01-14`.
- Финальный якорь для сабмита — `2026-02-13` (fold_end).
- ⚠️ Окно предсказания содержит праздники (23.02, 08.03) — календарная специфика учитывается
  отдельно (см. эксперимент 2 в [EXPERIMENTS.md](EXPERIMENTS.md)).

## Структура репозитория

```
ozon/
├── README.md                      # этот файл
├── EXPERIMENTS.md                 # журнал экспериментов: лидерборд + карточки (гипотеза→метод→результат→вывод)
├── baseline-seacrh-ltv.ipynb      # исходный бейзлайн организаторов (референс CV-схемы и фичей)
├── sample_submit.csv              # формат сабмита
├── src/                           # текущий этап данных:
│   ├── eda.py                     #   EDA сырых данных -> reports/eda_*.md
│   ├── build_features_ext_pca.py  #   exp02+exp07: base + x_* + pca_* -> data/v2/features_ext/
│   ├── build_features_bgnbd.py    #   exp04: BTYD-фичи -> data/v2/features_bgnbd/
│   ├── exp03_hurdle_priors.py     #   exp03: сегментные приоры P(buy)/E[z|buy] -> reports/
│   └── exp06_calibration_indices.py  # exp06: сезонный/YoY/локальный индексы -> reports/
├── data/                          # НЕ в гите:
│   ├── train.parquet              #   исходные данные (~172 МБ)
│   ├── v2/MANIFEST.md             #   схема артефактов для этапа обучения
│   ├── v2/features/               #   exp01-фичи по фолдам (fold_00..03, fold_end)
│   ├── v2/features_ext/           #   base+x_+pca фичи по фолдам
│   └── v2/features_bgnbd/         #   BTYD-фичи по фолдам
└── archive/                       # закрытые эксперименты — только история
    └── exp01/                     #   эксп. 1: оконные агрегаты + CatBoost (RMSLE 1.70261)
        ├── src/                   #   код: features.py → baselines.py → sanity_check.py → train.py → submit.py
        ├── reports/               #   метрики (*.json) и графики (figures/)
        └── submissions/           #   submission_exp01.csv
```

Правила размещения:

- Новый эксперимент работает в `src/` (создаётся под задачу); после закрытия переносится
  в `archive/expNN/` вместе с отчётами (`reports/`) и сабмитом (`submissions/`).
- Скрипты используют относительные пути и **запускаются из корня репо**: читают
  `data/v2/features/...`, пишут `reports/` и `submissions/` в корне; при архивации
  эти папки переезжают внутрь `archive/expNN/`.
- Тяжёлые артефакты (паркеты фичей, сырые данные) живут в `data/` и игнорируются гитом.

## Как воспроизвести (пример: exp01)

Каждый скрипт автономен (`if __name__ == "__main__"`), порядок конвейера:

```bash
# 0) положить train.parquet в data/ (в гите его нет)

# 1) извлечение фичей по юзерам -> data/v2/features/fold_*/batch_*.parquet
.venv/bin/python archive/exp01/src/features.py

# 2) sanity-чек: схемы, дат, юзеров по фолдам
.venv/bin/python archive/exp01/src/sanity_check.py

# 3) наивные референсы (zero/median/carry-forward) -> reports/exp00_baselines.json
.venv/bin/python archive/exp01/src/baselines.py

# 4) серия CatBoost по n_estimators + графики/метрики -> reports/
.venv/bin/python archive/exp01/src/train.py

# 5) финальная модель на всех 4 фолдах -> submissions/submission_exp01.csv
.venv/bin/python archive/exp01/src/submit.py
```

## Текущее состояние

| # | Эксперимент | RMSLE (fold_03) | Статус |
|---|---|---|---|
| 0 | Наивный автогресс (gmv за 30д) | 2.19506 | done |
| 1 | CatBoost на оконных агрегатах, 1000 iters | 1.70261 | done |
| 2 | GBDT: расширенные фичи (интент, EWMA, тренды) | — | data ready |
| 7 | PCA-компоненты дневных панелей → GBDT | — | data ready |
| 4 | BG/NBD×Gamma-Gamma → фичи для бустинга | — | data ready |
| 6 | Калибровка уровня: z-shift × сезонный β | — | priors ok |
| 3 | Hurdle в z-пространстве: P(buy)×E[z\|buy] | — | priors ok |
| 5 | Двухмасштабный TCN, бленд с GBDT | — | planned |

`data ready` = датасет фичей собран (`data/v2/MANIFEST.md` — схема, джойны, утечки),
обучение — следующий этап. Порядок: базовая таблица (2+7+4) → калибровка (6) → hurdle (3).

## Окружение

- **Каноническое окружение — `.venv` (Python 3.12)**: polars, scikit-learn, catboost.
  Запуск: `.venv/bin/python <скрипт>`. Добавление пакетов:
  `uv pip install --python .venv/bin/python <pkg>`.
- Специального железа не требуется: тяжёлые шаги (извлечение фичей) батчируются по
  50k юзеров; обучение CatBoost на полном трейне занимает минуты на CPU.
- Для будущих NN-экспериментов может понадобиться GPU (MPS/CUDA) — пакеты ставятся
  в `.venv` под конкретный эксперимент.
- Matplotlib в скриптах — только с бэкендом `Agg`.

## Правила работы

1. **Один эксперимент = одна запись в [EXPERIMENTS.md](EXPERIMENTS.md)**:
   карточка (гипотеза → метод → результат → вывод) + строка в лидерборде.
   Статусы: `planned → running → done / rejected`.
2. **Код и изменения льём в git** (`origin` =
   https://github.com/alexkolesnikov08/ozon-ecup): пушим после каждого значимого результата;
   данные, фичевые паркеты и мусор в гит не попадают (см. `.gitignore`).
3. Честность валидации: никакой информации после якоря в фичах; сравнение конфигураций —
   только на одном и том же наборе фолдов; случайные сиды фиксируем.
4. Перед коммитом проверять `git status`: в репо не должно быть мусора
   (`__pycache__`, `.DS_Store`, `catboost_info/`, логи, временные файлы).
