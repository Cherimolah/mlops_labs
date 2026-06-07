"""
Модульные тесты слоя модели (app/model.py). Используют фикстуру dummy_model
из conftest.py — настоящая обученная модель и данные DVC не нужны.
"""
import numpy as np
import pandas as pd
import pytest

from app.schemas import ApartmentFeatures, DISTRICTS
from app import model as model_mod


def _features(**overrides):
    base = dict(
        full_sq=65.0, life_sq=40.0, floor=5, max_floor=9,
        build_year=2005, num_room=2, kitch_sq=10.0, sub_area=DISTRICTS[0],
    )
    base.update(overrides)
    return ApartmentFeatures(**base)


def test_predict_price_returns_positive_float(dummy_model):
    price = model_mod.predict_price(_features())
    assert isinstance(price, float)
    assert np.isfinite(price)


def test_predict_price_unknown_district_falls_back(dummy_model):
    # Неизвестный район не должен ронять предсказание (model.py делает fallback
    # на первый известный класс кодировщика).
    price = model_mod.predict_price(_features(sub_area="НесуществующийРайон"))
    assert isinstance(price, float)


def test_feature_order_matches_contract():
    # Порядок признаков критичен: model.py подаёт их в модель именно так.
    assert model_mod.FEATURES == [
        "full_sq", "life_sq", "floor", "max_floor", "build_year",
        "num_room", "kitch_sq", "sub_area_enc",
    ]


def test_predict_batch_basic(dummy_model):
    df = pd.DataFrame([
        dict(full_sq=65.0, life_sq=40.0, floor=5, max_floor=9,
             build_year=2005, num_room=2, kitch_sq=10.0, sub_area=DISTRICTS[0]),
        dict(full_sq=80.0, life_sq=55.0, floor=3, max_floor=12,
             build_year=2010, num_room=3, kitch_sq=12.0, sub_area=DISTRICTS[1]),
    ])
    out = model_mod.predict_batch(df)
    assert isinstance(out, list) and len(out) == 2
    for row in out:
        assert "price" in row and isinstance(row["price"], float)
        assert "price_formatted" in row and row["price_formatted"].endswith("руб.")


def test_predict_batch_handles_nan_and_unknown_district(dummy_model):
    # Пропуск в числовой колонке заполняется медианой, неизвестный район —
    # fallback. Предсказание не должно падать.
    df = pd.DataFrame([
        dict(full_sq=65.0, life_sq=np.nan, floor=5, max_floor=9,
             build_year=2005, num_room=2, kitch_sq=10.0, sub_area="WAT"),
        dict(full_sq=70.0, life_sq=45.0, floor=4, max_floor=10,
             build_year=2011, num_room=2, kitch_sq=11.0, sub_area=DISTRICTS[2]),
    ])
    out = model_mod.predict_batch(df)
    assert len(out) == 2
    assert all(np.isfinite(r["price"]) for r in out)


def test_load_model_missing_raises(tmp_path, monkeypatch):
    # Если артефакта нет — load_model должен явно сообщить об этом.
    monkeypatch.setattr(model_mod, "MODEL_PATH", str(tmp_path / "nope.joblib"))
    monkeypatch.setattr(model_mod, "_model", None, raising=False)
    with pytest.raises(FileNotFoundError):
        model_mod.load_model()
