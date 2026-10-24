# exp08 — Стек v1: LGBM + HistGBR + CatBoost + Ridge

| | |
|---|---|
| **RMSLE (fold_03)** | `1.67103` |
| **Статус** | done |
| **Дата** | 2026-08-24 |
| **Карточка в журнале** | [EXPERIMENTS.md](../../EXPERIMENTS.md) |

Ансамбль четырёх GBDT на 141 признаке с мета-моделью (RidgeCV / OLS-positive) на честных OOF.

**Вывод.** Стек не превзошёл лучший соло-CatBoost: GBDT-члены сильно коррелированы. Ценность — робастность и встроенная калибровка уровня.

## Состав каталога

- `src/` — код эксперимента: `submit_baseline_blocks.py`, `train_stack.py`
- `reports/` — отчёты, метрики, графики: `bl_blocks.json`, `stack_search.json`

## Как запустить

Архивные скрипты используют относительные пути и запускаются **из корня репозитория**:

```bash
.venv/bin/python archive/exp08/src/train_stack.py
```

Крупные артефакты (parquet-фичи, CSV-сабмиты, чекпоинты) в git не попадают —
регенерируются этим же скриптом. Протокол валидации: [docs/PROTOCOL.md](../../docs/PROTOCOL.md).
