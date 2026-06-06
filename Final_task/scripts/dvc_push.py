"""
dvc_push.py — вспомогательный скрипт для DVC-операций внутри Jenkins-пайплайна.

Команды:
    python scripts/dvc_push.py pull   — скачать актуальные данные из remote
    python scripts/dvc_push.py add    — зарегистрировать обновлённые данные в DVC
    python scripts/dvc_push.py push   — загрузить новую версию данных в remote
    python scripts/dvc_push.py status — показать статус отслеживаемых файлов

Переменные окружения для S3:
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
    (или GOOGLE_APPLICATION_CREDENTIALS для GCS, AZURE_* для Azure)
"""

import subprocess
import sys

DATA_FILES = [
    "data/train.csv",
    "data/reference.csv",
]

DVC_FILES = [f"{f}.dvc" for f in DATA_FILES]


def run(cmd: list[str]) -> int:
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        print(f"[dvc_push] ОШИБКА: команда завершилась с кодом {result.returncode}", file=sys.stderr)
    return result.returncode


def pull() -> int:
    """Получить данные из remote-хранилища."""
    return run(["dvc", "pull", "--force"] + DATA_FILES)


def add() -> int:
    """Добавить/обновить отслеживание файлов данных."""
    rc = run(["dvc", "add"] + DATA_FILES)
    if rc != 0:
        return rc
    # После `dvc add` обновляются .dvc-файлы — их нужно закоммитить в Git.
    rc = run(["git", "add"] + DVC_FILES + ["data/.gitignore"])
    if rc != 0:
        return rc
    return run([
        "git", "commit", "-m",
        "chore(data): update DVC pointers after dataset refresh [skip ci]",
    ])


def push() -> int:
    """Загрузить данные в remote-хранилище."""
    return run(["dvc", "push"] + DATA_FILES)


def status() -> int:
    """Показать статус отслеживаемых файлов."""
    return run(["dvc", "status"])


COMMANDS = {"pull": pull, "add": add, "push": push, "status": status}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(f"Использование: python {sys.argv[0]} <{'|'.join(COMMANDS)}>")
        return 1
    return COMMANDS[sys.argv[1]]()


if __name__ == "__main__":
    raise SystemExit(main())
