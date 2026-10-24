"""Пути репозитория — единая точка правды для всех скриптов в `src/`.

Пути считаются от корня репозитория (`REPO_ROOT`), поэтому скрипт можно
запускать из любого каталога:

    .venv/bin/python /path/to/ozon-ecup/src/eda.py

Крупные артефакты (parquet-фичи, сырые данные, CSV-сабмиты) в git не
попадают — см. `.gitignore`. Каталоги создаются на лету, если их нет.
"""

from __future__ import annotations

from pathlib import Path

# src/common/paths.py -> src/common -> src -> корень репозитория
REPO_ROOT = Path(__file__).resolve().parents[2]

# Исходные данные и производные датасеты (в git не коммитятся).
DATA_DIR = REPO_ROOT / "data"
TRAIN_PARQUET = DATA_DIR / "train.parquet"
SAMPLE_SUBMIT = REPO_ROOT / "sample_submit.csv"

# Фичевые хранилища.
FEATURES_EXT_DIR = DATA_DIR / "v2" / "features_ext"
BTYD_DIR = DATA_DIR / "v2" / "features_bgnbd"
EXP1000_DIR = DATA_DIR / "v2" / "exp1000"

# Отчёты активных экспериментов и сабмиты.
REPORTS_DIR = REPO_ROOT / "reports"
SUBMISSIONS_DIR = REPO_ROOT / "submissions"


def ensure_dir(path: Path) -> Path:
    """Создать каталог (вместе с родителями) и вернуть его."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def repo_path(*parts: str) -> Path:
    """Собрать путь внутри репозитория: `repo_path("data", "v2")`."""
    return REPO_ROOT.joinpath(*parts)
