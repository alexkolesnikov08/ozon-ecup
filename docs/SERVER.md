# Удалённый сервер

Сервер используется для долгих прогонов: полных стеков, фабрики exp1000,
пер-юзерной персонализации на полном объёме (250k юзеров).

## Характеристики

| Параметр | Значение |
|---|---|
| CPU / RAM | 24 ядра / 96 ГБ (тяжёлые варианты считались на 50c / 47 ГБ) |
| Python | системный 3.12 (`pip install --user`) |
| venv | **недоступен без sudo** — зависимости ставятся в `--user` |
| Доступ | ssh-ключом, нестандартный порт, алиас `ssh main` |
| `libgomp` | распакован из deb в `~/local-libs` — обязателен для lightgbm/xgboost/catboost |

## Подготовка окружения

```bash
ssh main
mkdir -p ~/local-libs && tar xf libgomp*.deb -C ~/local-libs --strip-components=1
export LD_LIBRARY_PATH=$HOME/local-libs/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

python3 -m pip install --user -r requirements/base.txt
```

`LD_LIBRARY_PATH` обязателен — без него падает импорт lightgbm/xgboost/catboost.

## Порядок прогона

```bash
# 1) залить репозиторий (данные — отдельно, только train.parquet)
rsync -av --exclude 'data/' --exclude '.git/' ./ main:<remote>/ozon-ecup/
rsync -av --progress data/train.parquet main:<remote>/ozon-ecup/data/

# 2) собрать фичи
cd <remote>/ozon-ecup
python3 src/build_features_ext_pca.py
python3 src/build_features_bgnbd.py

# 3) долгий прогон — строго в tmux
tmux new -s ozon-ltv
python3 archive/exp15/src/train_stack_v2_remote.py
# отсоединиться: Ctrl-b, d   вернуться: tmux attach -t ozon-ltv
```

Долгие прогоны **обязаны** идти в tmux: обрыв SSH убивает процесс.

Полезно: `src/exp1000.py` — фабрика экспериментов с параллельным пулом,
рассчитанная именно на 24 ядра / 96 ГБ.

## Чек-лист перед отправкой сабмита

Обязательная проверка: локальные и серверные прогоны одного и того же кода
должны давать совпадающие метрики на `fold_03`. Расхождение означает
проблему данных, а не модели.

- [ ] `md5sum data/train.parquet` совпадает с локальным
- [ ] `data/v2/features_ext` полон: 5 фолдов (`fold_00..03` + `fold_end`) × батчи
- [ ] отчёт серверного прогона на `fold_03` совпадает с локальным
- [ ] CSV сабмита: число строк, среднее, порядок `user_id` vs `sample_submit.csv`

Полная проверка сабмита:

```python
import polars as pl

sample = pl.read_csv("sample_submit.csv")
sub = pl.read_csv("submissions/submission_segmented.csv")
assert sub.columns == ["user_id", "predict"]
assert sub.height == sample.height
assert sub["user_id"].to_list() == sample["user_id"].to_list()
assert sub["predict"].is_finite().all()
assert (sub["predict"] >= 0).all()
```

## Известный инцидент

Серверные прогоны стеков v2/v3 дали LB **1.73** и **2.80** при локальных
`fold_03` 1.669/1.672 — расхождение на порядки. Расследование не завершено;
версии и чек-лист — в
[notes/incidents/2026-08-25-server-lb-anomaly.md](../notes/incidents/2026-08-25-server-lb-anomaly.md).

Обходной путь, который сработал: заливка готового датасета напрямую
(минуя сборку фичей на сервере) — так был получен корректный артефакт
exp21 (`data/v3` залит байт-в-байт, порядок строк ассертен).
