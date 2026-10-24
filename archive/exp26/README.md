# exp26 — Segmented LTV pipeline с Transformer blend

| | |
|---|---|
| **RMSLE (fold_03)** | `1.66916` |
| **Статус** | done |
| **Дата** | 2026-08-30 |
| **Карточка в журнале** | [EXPERIMENTS.md](../../EXPERIMENTS.md) |

205 признаков напрямую из train.parquet, 26-недельные ряды, классы спроса ADI/CV², иерархия global → class → cluster, ветка Transformer.

**Вывод.** Классическая ветка на уровне чемпиона exp15 и повторяет SOTA. Transformer пока не добавляет выигрыша. База прод-сабмита.

## Состав каталога

- `src/` — код эксперимента: `blend_segmented_transformer.py`, `build_segmented_features.py`, `train_segmented_submit.py`, `train_weekly_transformer.py`
- `reports/` — отчёты, метрики, графики: `segmented_pipeline.json`, `segmented_transformer_blend.json`, `weekly_transformer.json`
- `scripts/` — лаунчеры пайплайна: `run_full_segmented_ensemble.sh`, `run_segmented_pipeline.sh`

## Как запустить

Архивные скрипты используют относительные пути и запускаются **из корня репозитория**:

```bash
.venv/bin/python archive/exp26/src/train_segmented_submit.py
```

Крупные артефакты (parquet-фичи, CSV-сабмиты, чекпоинты) в git не попадают —
регенерируются этим же скриптом. Протокол валидации: [docs/PROTOCOL.md](../../docs/PROTOCOL.md).
