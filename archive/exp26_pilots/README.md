# exp26 — Per-user & Transformer pilots (diagnostics)

Архив пилотных прогонов для персонализации CatBoost и Transformer, не вошедших в основные эксперименты exp23–25.

## Что внутри

- `src/grid_dense_fixed.py` — 1k pilot n=3/5/10, 95 cols, поиск dense якорей (7d vs 14d шаг)
- `src/grid_dense_opt.py` — аналогично n=3/5 с 141f (ext+btyd), тест btyd attach
- `src/grid_per_user_hypers.py` — grid depth/lr/l2/it для n=3 на 1k (depth 4/6/8 x lr 0.03/0.05/0.10 x l2 1/3/9 x it 10/20/30/50)
- `src/grid_per_user_n.py` — валидация n=3/5/10 с best hypers (depth4 lr0.1 l2=1)
- `src/lag_4ch.py` — 4ch lag 30d/60d (gmv/searches/to_ord/to_cart) vs aggregates, CatBoost 500it
- `src/lag_baseline.py` — 30d lag baseline (gmv only), 1k/10k
- `src/per_user_n7_full.py` — n=7 full 250k (7 якорей 26.11-07.01), broken prototype
- `src/per_user_n7_submit.py` / `per_user_n7_submit_v2.py` — n=7 it30 d4 lr0.05, 1.67 RMSLE, checkpointed Pool
- `src/transformer_full.py` — full Transformer 90d x4ch, 1.75M rows, 15 epochs, MPS, 90seq
- `src/transformer_pilot.py` — pilot 1k Transformer, 90d x4ch, 30 epochs
- `reports/grid_per_user_hypers.json` — результаты grid_per_user_hypers (best d4 lr0.1 l2=1 it50 RMSLE 1.581)
- `reports/per_user_n7_full.json` — n=7 it30 pilot (global 1.671, per 1.37 на 1k - overfit indicator)

## Связь с основным треком

- Основой остались `src/per_user_full.py` (exp23 n=3 it20, 1.624) и `src/per_user_full_n5_it25.py` (exp24 n=5 it25)
- Dense grids показали прирост n=5 vs n=3 ~0.02 на 1k, но full LB переобучается (1.686)
- n=7 не дал выигрыша vs n=5, 24ms/юзера, skipped 70k
- Transformer pilot RMSLE ~1.8 vs agg 1.67 — пока хуже, нужен статический контекст

Все скрипты запускаются из корня `python archive/exp26_pilots/src/<script>.py`
