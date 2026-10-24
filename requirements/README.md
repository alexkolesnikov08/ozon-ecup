# Зависимости

| Файл | Что ставит | Нужен для |
|---|---|---|
| [base.txt](base.txt) | numpy, polars, pandas, scipy, scikit-learn, catboost, lightgbm, xgboost, lifetimes, matplotlib, statsmodels | всё, кроме ветки с Transformer |
| [transformer.txt](transformer.txt) | base + torch | полный ансамбль сегментированного пайплайна (exp26) |

## Установка

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements/base.txt
```

Для ветки с Transformer:

```bash
.venv/bin/python -m pip install -r requirements/transformer.txt
```

На CUDA-сервере PyTorch лучше ставить версию под его CUDA — по инструкции
[pytorch.org](https://pytorch.org); `torch>=2.2` из файла подходит для CPU/MPS.

## Замечания

- **`lifetimes`** нужен только для `src/train_btyd.py` (соло-диагностика BTYD,
  exp04). Сама генерация BTYD-фичей (`src/build_features_bgnbd.py`) реализована
  на scipy и пакет не требует.
- **`statsmodels`** нужен только для SARIMA в `archive/exp18/src/exp10_decomposition.py`.
- **`torch`** — только ветка Transformer в `archive/exp26/src/train_weekly_transformer.py`.
- На сервере без sudo venv недоступен: `pip install --user`, и обязателен
  `LD_LIBRARY_PATH` на `libgomp` — см. [../docs/SERVER.md](../docs/SERVER.md).
