# exp23 — Heavy 6h Per-User CatBoost (n=3 anchors, 50 iters per user)

**Статус:** `planned` — запуск вручную на `ssh main` (50c, 47GB), 5.9ч wall, выделен отдельно по просьбе.
**Связанные:** exp23 в `EXPERIMENTS.md` (planned), пилоты `reports/per_user_pilot.json` (n=3 it20 1.624, 47мин), `reports/per_user_dense_pilot.json` (1k, n=3/10/50 x 5/20/30/40/50), `reports/test_n5_it25.json` (n5_it25 1.634).

## 1. Гипотеза
Глобальный CatBoost `1.674` на `fold_03` знает среднего юзера, но теряет индивидуальный уровень. Доучивание `50` деревьев на `3` окнах конкретного юзера (`init_model=глобал`) даст `1.638` на `1k` пилоте и `~1.62` на `250k`, уложится в `6ч` на `50c` серваке (`85мс/юзера`). Тяжелее `n=10 50ит` дает `1.618` но `9ч` на серваке — не влезает.

## 2. Данные и протокол
- **Фичи:** `data/v2/features_ext` 130 колонок (base 70 + x_* 28 + pca 32, без btyd), `NaN->0` для base, `NaN` остается для ratio (CatBoost нативно).
- **Якоря трейна (n=3):** `fold_00 2025-12-03`, `fold_01 2025-12-17`, `fold_02 2025-12-31` — 3 строки на юзера, `750k` строк total.
- **Холдаут:** `fold_03 2026-01-14` — 250k строк, метрика `RMSLE = RMSE(log1p)`, `z=log1p(y)`, `pred=expm1(z)`.
- **Прод:** `fold_end 2026-02-13` — 250k строк, таргет `NULL`, те же 130 фич.
- **Сид:** 42, `depth=8, lr=0.05, l2=3`, `loss=RMSE` на `z`.

## 3. Метод (ровно как в валидации, но 50ит)
1. **Глобал:** `CatBoostRegressor(iterations=500, depth=8, lr=0.05)` на `750k` rows (`00..02`), `~29с` на M1, `~101с` на 50c. Сохранить `RSMLE 1.674` на `03`.
2. **Пер-юзер:** для каждого `user_id` из `250k` взять его `3` строки трейна, `init_model=глобал`, доучить `50` итераций на этих `3` строках (`depth/lr` те же). Если `3` таргета одинаковые (`All train targets are equal`, ~32% юзеров) — скип, оставить глобал. Среднее `16.12мс` на `1k` пилоте (M1), `85мс` на серваке (50c, thread=1 оптимально).
3. **Инференс:** предсказать `z` для `fold_03` (валидация) и `fold_end` (сабмит), `y=expm1(z)`, `clip 0`.
4. **Чекпоинт:** каждые `5000` юзеров `reports/ckpt_6h/preds.npy`, `times.npy`, `progress.json` — resume при обрыве.

## 4. Ожидаемые результаты
- **1k пилот (M1, 130f):** `n=3 it20 1.66271`, `it50 1.63803` (`global 1.69284`), mean `16.12мс`, `316` скипов.
- **250k full (M1, 130f, it20):** `1.62463` vs `1.67446` (`-0.049`), `11.45мс`, `47.7мин` — уже доказано `reports/per_user_full.json`.
- **250k full (50c, it50, n=3):** ожидается `~1.62-1.63` на `03`, `LB` `~1.66` (blend 50/50 с глобалом дает `1.66` vs `per 1.686` overfit, см. `submission_hurdle_blend.csv`).
- **Время:** `250k *85мс = 5.9ч` + глобал `101с` = `6.1ч` на серваке, `250k*16мс=1.1ч` на M1 (но M1 занят).

## 5. Как запустить на серваке (строго в tmux)
```bash
ssh main
tmux new -s peruser6h
cd ~/ozon
# repo уже залит (main cd19569), data/v2/features_ext на месте (554M) + train.parquet (172M)
# если нет - rsync: rsync -avz -e ssh data/v2/features_ext main:~/ozon/data/v2/features_ext
.venv/bin/python src/per_user_full.py  # ITER_PER_USER=50, n=3 (поменять в файле 20->50)
# или: .venv/bin/python archive/exp23/src/train_6h.py
# логи: tail -f reports/ckpt_per_user_full/log.txt
# чекпоинт: ls reports/ckpt_per_user_full/
# по окончании: ls -lh submissions/submission_*.csv reports/per_user_full.json
tmux detach # Ctrl-b d
```

## 6. Артефакты
- **Код:** `src/per_user_full.py` (n=3 it20, 47мин, 1.624) — база, `src/per_user_full_n5_it25.py` (n=5 it25, 56мин, LB 1.686), `archive/exp23/src/train_6h.py` — заглушка 6ч.
- **Отчеты:** `reports/per_user_full.json`, `reports/per_user_dense_pilot.json` (1k ablation), `reports/test_n5_it25.json`.
- **Графики:** `reports/figures/per_user_*.png` + `~/Desktop/per_user_*.png` (скорость, метрика vs итерации, heatmap, cumulative).
- **Сабмиты:** `~/Desktop/submission_hurdle_blend.csv` (blend 50/50, 1.66 LB) — текущий лучший, `submissions/submission_n5_it25.csv` (1.686).

## 7. Риски и митигация
- **Переобучение на январь (fold_03 яма):** `per 1.624 -> LB 1.686`, `blend 1.66` лучше. Для 6ч добавить `shrinkage` или `blend 0.5`.
- **32% скипов (3 одинаковых таргета):** нормально, fallback на глобал.
- **Сервер медленнее в 3-12x на 3 строках:** использовать `thread_count=1` для пер-юзер (34мс vs 94мс на 50 потоках), батчи 5k.
- **Память:** `1.25M` rows x130f ~1.3GB, влезает в 47GB.

## 8. Критерий прохождения
`fold_03 RMSLE <1.64` и `LB <1.66` (лучше `1.66035` чемпиона). Если `LB >1.66` — откат к `blend` или `n=3 it20`.

## 9. Срочность
Надо СРОЧНО прогнать `n=3 it50` на фулле `250k` на серваке, снять `fold_03` и `LB`, сравнить с `n=3 it20` и `n=5 it25`, выбрать финальный сабмит до дедлайна. Код готов, данные на месте, запуск — одна команда в `tmux`.
