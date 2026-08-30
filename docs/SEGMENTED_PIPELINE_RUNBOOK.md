# Серверный запуск сегментированного LTV-пайплайна

Пайплайн читает только базовые файлы:

- `data/train.parquet` — исходная история;
- `sample_submit.csv` — список и порядок пользователей в сабмите.

Готовые датасеты `data/v2` и `data/v3` не используются.

## Что делает пайплайн

1. Для якорей `fold_00..fold_03` и финального `fold_end` строит признаки только по истории
   до соответствующего якоря. Таргет обучающих фолдов — GMV следующих 30 дней.
2. Строит 26-недельный ряд GMV каждого пользователя.
3. По ADI и CV² относит ряд к одному из четырёх стандартных классов:
   `smooth`, `erratic`, `intermittent`, `lumpy`.
4. Внутри каждого класса запускает `MiniBatchKMeans` (по умолчанию до трёх кластеров).
5. Обучает глобальную модель, модели классов и модели кластеров. Для больших групп
   используется CatBoost, для средних — HistGradientBoosting, для маленьких — Ridge;
   слишком маленькие группы автоматически откатываются к модели верхнего уровня.
6. Подбирает веса бленда на `fold_02`, честно оценивает варианты на `fold_03`, затем
   переобучает выбранный вариант на `fold_00..fold_03` и предсказывает `fold_end`.
7. Проверяет число строк, порядок `user_id`, отсутствие NULL/NaN и отрицательных прогнозов.

## 1. Подготовка сервера

Рекомендуется Python 3.11 или 3.12, 16 CPU-ядер и от 32 ГБ RAM. Для классической
ветки GPU не нужен; для Transformer желательно иметь CUDA GPU с 8+ ГБ VRAM.

Из корня репозитория:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements_segmented.txt
```

Для полного ансамбля с Transformer дополнительно установите PyTorch. На CUDA-сервере
лучше взять команду установки под его версию CUDA с сайта PyTorch; универсальный вариант:

```bash
.venv/bin/python -m pip install -r requirements_transformer.txt
```

Проверьте наличие входов:

```bash
ls -lh data/train.parquet sample_submit.csv
```

Ожидаемая схема `train.parquet` содержит `event_date`, `user_id`, `gmv`, `searches`,
`to_ord`, `to_cart` и каналы поисковой/категорийной воронки. Скрипт проверит схему сам.

## 2. Короткая проверка перед полным обучением

Она прогоняет весь код на первых 10 000 пользователях и пишет артефакты в `/tmp`, не
затрагивая полные признаки:

```bash
.venv/bin/python src/build_segmented_features.py \
  --out-dir /tmp/segmented_smoke \
  --limit-users 10000 \
  --batch-size 5000 \
  --overwrite

.venv/bin/python src/train_segmented_submit.py \
  --features-dir /tmp/segmented_smoke \
  --sample /tmp/segmented_smoke/sample_submit_smoke.csv \
  --submission /tmp/segmented_smoke/submission.csv \
  --report /tmp/segmented_smoke/report.json \
  --quick \
  --threads 8 \
  --min-local-rows 100 \
  --min-hist-rows 500 \
  --min-cat-rows 3000
```

В конце должны появиться строки `Submission:` и `Report:` без traceback.

## 3. Классическая ветка одной командой

```bash
bash scripts/run_segmented_pipeline.sh all
```

По умолчанию используются 16 потоков и батчи по 50 000 пользователей. Настройка для
сервера, например на 24 ядра:

```bash
SEGMENTED_THREADS=24 SEGMENTED_BATCH_SIZE=50000 \
  bash scripts/run_segmented_pipeline.sh all
```

Этот вариант уже создаёт готовый `submissions/submission_segmented.csv` и подходит для
CPU-сервера.

## 4. Полный ансамбль с Transformer

После установки `requirements_transformer.txt`:

```bash
SEGMENTED_THREADS=24 SEGMENTED_DEVICE=cuda SEGMENTED_EPOCHS=12 \
  bash scripts/run_full_segmented_ensemble.sh
```

Скрипт последовательно строит признаки, обучает классическую и Transformer-ветки,
подбирает их мета-бленд на `fold_02`, проверяет на `fold_03` и создаёт:

```text
submissions/submission_segmented_transformer.csv
```

Если GPU отсутствует, используйте `SEGMENTED_DEVICE=cpu`; это существенно дольше. Для
проверки всей цепочки с двумя эпохами:

```bash
SEGMENTED_QUICK=1 SEGMENTED_DEVICE=cuda \
  bash scripts/run_full_segmented_ensemble.sh
```

Если процесс запускается по SSH, используйте `tmux`, чтобы он пережил разрыв соединения:

```bash
tmux new -s ozon-ltv
SEGMENTED_THREADS=24 bash scripts/run_segmented_pipeline.sh all
```

Отсоединиться от `tmux`: `Ctrl-b`, затем `d`. Вернуться: `tmux attach -t ozon-ltv`.

## 5. Раздельный и повторный запуск

Только построение признаков:

```bash
bash scripts/run_segmented_pipeline.sh features
```

Только обучение, если `data/segmented_base/fold_*.parquet` уже готовы:

```bash
bash scripts/run_segmented_pipeline.sh train
```

Готовые валидные фолды автоматически переиспользуются. Чтобы принудительно перестроить их:

```bash
SEGMENTED_OVERWRITE=1 bash scripts/run_segmented_pipeline.sh features
```

Быстрый полный прогон с уменьшенным числом итераций моделей:

```bash
SEGMENTED_QUICK=1 bash scripts/run_segmented_pipeline.sh all
```

`SEGMENTED_QUICK=1` нужен для проверки инфраструктуры, а не для финальной отправки.

## 6. Результаты

После полного запуска:

- `submissions/submission_segmented.csv` — готовый файл для отправки;
- `submissions/submission_segmented_transformer.csv` — итог полного ансамбля; внутри
  автоматически выбран лучший по `fold_03` вариант: classical, Transformer или blend;
- `reports/segmented_pipeline.json` — RMSLE моделей на временных фолдах, выбранный
  вариант, веса бленда, размеры классов/кластеров и тип модели каждой группы;
- `reports/weekly_transformer.json` и `reports/segmented_transformer_blend.json` —
  метрики sequence-модели и финального мета-бленда;
- `data/segmented_base/manifest.json` — схема построенных данных и распределение четырёх
  типов временных рядов;
- `data/segmented_base/fold_*.parquet` — кэш признаков для повторного обучения.

Финальная ручная проверка:

```bash
.venv/bin/python - <<'PY'
import polars as pl

sample = pl.read_csv("sample_submit.csv")
# Для классической ветки замените имя на submission_segmented.csv.
sub = pl.read_csv("submissions/submission_segmented_transformer.csv")
assert sub.columns == ["user_id", "predict"]
assert sub.height == sample.height
assert sub["user_id"].to_list() == sample["user_id"].to_list()
assert sub["predict"].is_finite().all()
assert (sub["predict"] >= 0).all()
print(sub.shape)
print(sub.select("predict").describe())
PY
```

После полного запуска используйте `submissions/submission_segmented_transformer.csv`.
После запуска только классической ветки — `submissions/submission_segmented.csv`.

## 7. Полезные параметры

Прямой запуск обучающего скрипта позволяет менять вычислительный бюджет:

```bash
.venv/bin/python src/train_segmented_submit.py \
  --threads 24 \
  --clusters-per-class 3 \
  --global-iters 1000 \
  --local-iters 500 \
  --hist-iters 300
```

Если RAM мало, снизьте `--batch-size` при построении признаков. Если времени мало,
уменьшите `--global-iters`, `--local-iters` и `--hist-iters`; это влияет на качество,
но не меняет формат сабмита.
