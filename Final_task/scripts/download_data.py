import glob
import os
import shutil
import sys
import zipfile

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data")
RAW_DIR = os.path.join(DATA_DIR, "_raw")
TARGET_PATH = os.path.join(DATA_DIR, "train.csv")

COMPETITION = "mlurfuflat"

REQUIRED_COLS = [
    "full_sq", "life_sq", "floor", "max_floor", "build_year",
    "num_room", "kitch_sq", "sub_area", "price_doc",
]


def _download_raw() -> None:
    # Импорт внутри функции: пакет kaggle нужен только пайплайну,
    # а не самому веб-приложению.
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    os.makedirs(RAW_DIR, exist_ok=True)
    print(f"Скачиваю файлы соревнования '{COMPETITION}'...")
    api.competition_download_files(COMPETITION, path=RAW_DIR, quiet=False)

    for archive in glob.glob(os.path.join(RAW_DIR, "*.zip")):
        print(f"Распаковываю {os.path.basename(archive)}")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(RAW_DIR)


def _find_train_csv() -> str:
    candidates = glob.glob(os.path.join(RAW_DIR, "**", "*.csv"), recursive=True)
    if not candidates:
        raise FileNotFoundError(f"В {RAW_DIR} не найдено ни одного csv после скачивания.")

    for path in candidates:
        try:
            header = pd.read_csv(path, nrows=0).columns
        except Exception:
            continue
        if "price_doc" in header:
            print(f"Обучающий файл: {os.path.basename(path)}")
            return path

    # На крайний случай берём самый большой csv.
    biggest = max(candidates, key=os.path.getsize)
    print(f"price_doc не найден явно, беру самый большой csv: {os.path.basename(biggest)}")
    return biggest


def _normalize(src: str) -> None:
    df = pd.read_csv(src)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"В скачанном датасете не хватает колонок: {missing}.\n"
            f"Доступные колонки: {list(df.columns)}"
        )

    df = df[REQUIRED_COLS]
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(TARGET_PATH, index=False)
    print(f"Сохранено: {TARGET_PATH}  (строк: {len(df):,})")


def main() -> int:
    try:
        _download_raw()
        src = _find_train_csv()
        _normalize(src)
    except Exception as exc:  # noqa: BLE001 — хотим понятное сообщение в логе Jenkins
        print(f"[download_data] ОШИБКА: {exc}", file=sys.stderr)
        return 1
    finally:
        # Чистим временную папку с архивами, чтобы не мусорить в репозитории.
        if os.path.isdir(RAW_DIR):
            shutil.rmtree(RAW_DIR, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
