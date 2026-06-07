"""
conftest.py — общие фикстуры для всех тестов.

Главная идея: модульные тесты не должны зависеть ни от обученной «настоящей»
модели (models/apartment_model.joblib), ни от данных из DVC. Поэтому фикстура
`dummy_model` обучает крошечную модель на синтетических данных, кладёт её по тем
путям, откуда её читает app/model.py, и подменяет (monkeypatch) пути на время
теста. Это делает тесты быстрыми, детерминированными и не требующими сети.
"""
import os
import sys

import numpy as np
import pandas as pd
import joblib
import pytest

# Корень репозитория = на уровень выше каталога tests/.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Где лежат CSV. По умолчанию data/ в репозитории; в CI можно переопределить
# переменной окружения DATA_DIR (например, чтобы указать на путь после dvc pull).
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(REPO_ROOT, "data"))


@pytest.fixture(scope="session")
def repo_root():
    return REPO_ROOT


@pytest.fixture(scope="session")
def data_dir():
    return DATA_DIR


def _train_tiny_model(tmp_path):
    """Обучаем игрушечную модель в формате, который ждёт app/model.py."""
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import LabelEncoder
    from app.schemas import DISTRICTS

    encoder = LabelEncoder().fit(DISTRICTS)

    rng = np.random.default_rng(42)
    n = 200
    sub_enc = encoder.transform(rng.choice(DISTRICTS, size=n))
    X = np.column_stack([
        rng.uniform(20, 120, n),    # full_sq
        rng.uniform(10, 80, n),     # life_sq
        rng.integers(1, 30, n),     # floor
        rng.integers(1, 30, n),     # max_floor
        rng.integers(1950, 2020, n),  # build_year
        rng.integers(1, 5, n),      # num_room
        rng.uniform(4, 20, n),      # kitch_sq
        sub_enc,                    # sub_area_enc
    ])
    y = X[:, 0] * 100_000 + rng.normal(0, 50_000, n)

    model = GradientBoostingRegressor(n_estimators=20, max_depth=3, random_state=42)
    model.fit(X, y)

    model_path = os.path.join(tmp_path, "apartment_model.joblib")
    encoder_path = os.path.join(tmp_path, "district_encoder.joblib")
    joblib.dump(model, model_path)
    joblib.dump(encoder, encoder_path)
    return model_path, encoder_path


@pytest.fixture()
def dummy_model(tmp_path, monkeypatch):
    """
    Подменяет пути к модели/кодировщику в app.model на игрушечные артефакты
    и сбрасывает закэшированные глобальные объекты, чтобы load_model() их
    подхватил. Возвращает (model_path, encoder_path).
    """
    import app.model as model_mod

    model_path, encoder_path = _train_tiny_model(str(tmp_path))
    monkeypatch.setattr(model_mod, "MODEL_PATH", model_path, raising=False)
    monkeypatch.setattr(model_mod, "ENCODER_PATH", encoder_path, raising=False)
    monkeypatch.setattr(model_mod, "_model", None, raising=False)
    monkeypatch.setattr(model_mod, "_encoder", None, raising=False)
    model_mod.load_model()
    yield model_path, encoder_path


@pytest.fixture()
def client(dummy_model):
    """FastAPI TestClient с уже загруженной игрушечной моделью."""
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as c:
        yield c


def _read_csv_or_skip(path):
    if not os.path.exists(path):
        pytest.skip(
            f"Файл данных не найден: {path}. "
            f"В CI он появляется после стадии 'DVC pull'. "
            f"Локально укажите DATA_DIR или выполните dvc pull."
        )
    return pd.read_csv(path)
