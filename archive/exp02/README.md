# exp02 — GBDT: расширенные фичи + YoY

| | |
|---|---|
| **RMSLE (fold_03)** | `1.69277` |
| **Статус** | done · adopted |
| **Дата** | 2026-08-24 |
| **Карточка в журнале** | [EXPERIMENTS.md](../../EXPERIMENTS.md) |

Поверх exp01 добавлены интент-сигналы, конверсии по источникам, декомпозиция «частота × чек», due_ratio, доли gmv_search/cat и YoY-калибровка.

**Вывод.** Прирост дали конверсии источников, декомпозиция и due_ratio; EWMA/тренды не вошли. adopted (порог ≤ 1.69920).

## Состав каталога

- `src/` — код эксперимента: `exp02_beta_check.py`, `exp02_features.py`, `exp02_submit.py`, `exp02_train.py`
- `reports/` — отчёты, метрики, графики: `exp02_metrics.json`, `exp02_report.md`
- `submissions/` — CSV-сабмиты: `submission_exp02.csv`

## Как запустить

Архивные скрипты используют относительные пути и запускаются **из корня репозитория**:

```bash
.venv/bin/python archive/exp02/src/exp02_train.py
```

Крупные артефакты (parquet-фичи, CSV-сабмиты, чекпоинты) в git не попадают —
регенерируются этим же скриптом. Протокол валидации: [docs/PROTOCOL.md](../../docs/PROTOCOL.md).
