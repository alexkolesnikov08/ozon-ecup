"""Общие пути репозитория.

Модуль существует, чтобы скрипты не зависели от текущего рабочего каталога:
все пути строятся от корня репозитория, а не от `cwd`.

Использование из скрипта в `src/`:

    from common.paths import DATA_DIR, REPO_ROOT

`src/` попадает в `sys.path` автоматически, когда скрипт запущен как
`python src/<скрипт>.py`, поэтому отдельного `sys.path`-хака не требуется.
Скрипты в `archive/` — замороженная история, они работают на относительных
путях и запускаются из корня репозитория.
"""

from common.paths import (
    BTYD_DIR,
    DATA_DIR,
    EXP1000_DIR,
    FEATURES_EXT_DIR,
    REPO_ROOT,
    REPORTS_DIR,
    SAMPLE_SUBMIT,
    SUBMISSIONS_DIR,
    TRAIN_PARQUET,
)

__all__ = [
    "BTYD_DIR",
    "DATA_DIR",
    "EXP1000_DIR",
    "FEATURES_EXT_DIR",
    "REPO_ROOT",
    "REPORTS_DIR",
    "SAMPLE_SUBMIT",
    "SUBMISSIONS_DIR",
    "TRAIN_PARQUET",
]
