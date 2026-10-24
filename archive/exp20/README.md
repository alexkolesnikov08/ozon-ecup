# exp20 — Смешанный стек: 9 семейств моделей

| | |
|---|---|
| **RMSLE (fold_03)** | `1.68608` |
| **Статус** | done · rejected |
| **Дата** | 2026-08-25 |
| **Карточка в журнале** | [EXPERIMENTS.md](../../EXPERIMENTS.md) |

lgbm/cat/xgb/hist + ridge + LinearSVR + RandomForest + kNN по PCA + hurdle_lin, на блоке base+ext+pca.

**Вывод.** rejected: мета-модель работает, только если члены сопоставимы по качеству; линейные/kNN/hurdle-члены тянут смесь вниз.

## Состав каталога

- `src/` — код эксперимента: `train_stack_mixed.py`

## Как запустить

Архивные скрипты используют относительные пути и запускаются **из корня репозитория**:

```bash
.venv/bin/python archive/exp20/src/train_stack_mixed.py
```

Крупные артефакты (parquet-фичи, CSV-сабмиты, чекпоинты) в git не попадают —
регенерируются этим же скриптом. Протокол валидации: [docs/PROTOCOL.md](../../docs/PROTOCOL.md).
