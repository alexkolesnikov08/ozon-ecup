# exp01 — CatBoost на оконных агрегатах

| | |
|---|---|
| **RMSLE (fold_03)** | `1.70261` |
| **Статус** | done · adopted |
| **Дата** | 2026-08-24 |
| **Карточка в журнале** | [EXPERIMENTS.md](../../EXPERIMENTS.md) |

56 оконных агрегатов (gmv / searches / to_ord / to_cart за 7–90 дней, recency, tenure, конверсии) → CatBoost.

**Вывод.** Первая ML-модель: −0.49 RMSLE к наиву (2.19506 → 1.70261). Заказная история важнее сырого gmv; кривая обучения ещё снижалась на 1000 итерациях.

## Состав каталога

- `src/` — код эксперимента: `baselines.py`, `features.py`, `sanity_check.py`, `submit.py`, `train.py`
- `reports/` — отчёты, метрики, графики: `exp00_baselines.json`, `exp01_best_model.cbm`, `exp01_metrics.json`
- `submissions/` — CSV-сабмиты: `submission_exp01.csv`

## Как запустить

Архивные скрипты используют относительные пути и запускаются **из корня репозитория**:

```bash
.venv/bin/python archive/exp01/src/train.py
```

Крупные артефакты (parquet-фичи, CSV-сабмиты, чекпоинты) в git не попадают —
регенерируются этим же скриптом. Протокол валидации: [docs/PROTOCOL.md](../../docs/PROTOCOL.md).
