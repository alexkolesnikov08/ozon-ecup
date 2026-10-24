# exp14 — CatBoost: серия гиперпараметров на base56

| | |
|---|---|
| **RMSLE (fold_03)** | `1.70124` |
| **Статус** | done |
| **Дата** | 2026-08-24 |
| **Карточка в журнале** | [EXPERIMENTS.md](../../EXPERIMENTS.md) |

Сетка по n_estimators / lr / depth / l2_leaf_reg / bootstrap с early stopping.

**Вывод.** Весь тюнинг даёт ≤ 0.0017 против parity — уровень шума. Depth 8 оптимален, дальнейшее удлинение обучения нерентабельно.

## Состав каталога

- `src/` — код эксперимента: `train_ablation.py`
- `reports/` — отчёты, метрики, графики: `train_ablation.json`

## Как запустить

Архивные скрипты используют относительные пути и запускаются **из корня репозитория**:

```bash
.venv/bin/python archive/exp14/src/train_ablation.py
```

Крупные артефакты (parquet-фичи, CSV-сабмиты, чекпоинты) в git не попадают —
регенерируются этим же скриптом. Протокол валидации: [docs/PROTOCOL.md](../../docs/PROTOCOL.md).
