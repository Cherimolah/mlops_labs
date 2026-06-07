"""
Модульные тесты Pydantic-схем (app/schemas.py): проверяем границы полей и то,
что справочник районов корректен. Это «чистые» модульные тесты — без модели,
без сети, без данных.
"""
import pytest
from pydantic import ValidationError

from app.schemas import (
    ApartmentFeatures, PredictionResult, BatchPredictionResult, DISTRICTS,
)


def _valid_payload(**overrides):
    base = dict(
        full_sq=65.0, life_sq=40.0, floor=5, max_floor=9,
        build_year=2005, num_room=2, kitch_sq=10.0, sub_area=DISTRICTS[0],
    )
    base.update(overrides)
    return base


def test_valid_features_construct():
    f = ApartmentFeatures(**_valid_payload())
    assert f.full_sq == 65.0
    assert f.sub_area == DISTRICTS[0]


@pytest.mark.parametrize("field,value", [
    ("full_sq", 0),       # gt=0
    ("full_sq", 1001),    # le=1000
    ("life_sq", 0),       # gt=0
    ("floor", 0),         # ge=1
    ("floor", 78),        # le=77
    ("max_floor", 78),    # le=77
    ("build_year", 1899), # ge=1900
    ("build_year", 2031), # le=2030
    ("num_room", 0),      # ge=1
    ("num_room", 20),     # le=19
    ("kitch_sq", 0),      # gt=0
    ("kitch_sq", 201),    # le=200
])
def test_out_of_range_rejected(field, value):
    with pytest.raises(ValidationError):
        ApartmentFeatures(**_valid_payload(**{field: value}))


def test_missing_field_rejected():
    payload = _valid_payload()
    payload.pop("sub_area")
    with pytest.raises(ValidationError):
        ApartmentFeatures(**payload)


def test_districts_catalog_is_clean():
    assert isinstance(DISTRICTS, list)
    assert len(DISTRICTS) > 0
    assert all(isinstance(d, str) and d.strip() for d in DISTRICTS)
    assert len(DISTRICTS) == len(set(DISTRICTS)), "В справочнике районов есть дубликаты"


def test_result_models_construct():
    f = ApartmentFeatures(**_valid_payload())
    pr = PredictionResult(price=1_000_000.0, price_formatted="1 000 000 руб.", features=f)
    assert pr.price == 1_000_000.0
    bp = BatchPredictionResult(predictions=[{"price": 1.0}], count=1)
    assert bp.count == 1
