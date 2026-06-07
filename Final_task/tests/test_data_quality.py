"""
Тесты КАЧЕСТВА ДАННЫХ (Great Expectations-стиль, но на чистом pytest).

Проверяют, что свежий датасет удовлетворяет «контракту данных» (data_contract.py):
нужные колонки, валидный таргет, допустимая доля пропусков, согласованность
схемы train.csv и reference.csv. Все тесты помечены маркером `data_quality`,
поэтому в Jenkins они запускаются отдельной стадией от модульных тестов:

    pytest tests -m "not data_quality"   # стадия Unit tests
    pytest tests -m "data_quality"       # стадия Data quality tests
"""
import os

import pandas as pd
import pytest

from tests import data_contract as dc
from tests.conftest import DATA_DIR, _read_csv_or_skip

pytestmark = pytest.mark.data_quality

# Датасеты с «боевой» схемой, на которых работает пайплайн.
PIPELINE_DATASETS = ["train.csv", "reference.csv"]


@pytest.fixture(params=PIPELINE_DATASETS)
def dataset(request):
    path = os.path.join(DATA_DIR, request.param)
    df = _read_csv_or_skip(path)
    return request.param, df


# --------------------------- STRICT: ломает пайплайн ----------------------- #

def test_required_columns_present(dataset):
    name, df = dataset
    missing = set(dc.REQUIRED_COLUMNS) - set(df.columns)
    assert not missing, f"{name}: отсутствуют колонки {missing}"


def test_min_rows(dataset):
    name, df = dataset
    assert len(df) >= dc.MIN_ROWS, f"{name}: слишком мало строк ({len(df)} < {dc.MIN_ROWS})"


def test_target_present_nonnull_positive(dataset):
    name, df = dataset
    assert dc.TARGET in df.columns, f"{name}: нет таргета {dc.TARGET}"
    target = pd.to_numeric(df[dc.TARGET], errors="coerce")
    assert target.notna().all(), f"{name}: в таргете есть пропуски"
    assert (target > 0).all(), f"{name}: таргет содержит неположительные значения"


def test_non_null_columns(dataset):
    name, df = dataset
    for col in dc.NON_NULL_COLUMNS:
        assert df[col].notna().all(), f"{name}: в колонке {col} не должно быть пропусков"


def test_no_fully_duplicated_dataset(dataset):
    name, df = dataset
    dup_frac = df.duplicated().mean()
    assert dup_frac < 0.5, f"{name}: подозрительно много дублей строк ({dup_frac:.0%})"


# --------------------------- SOFT: ловит битый файл ------------------------ #

def test_soft_null_fraction_within_ceiling(dataset):
    name, df = dataset
    for col in dc.SOFT_NULLABLE_COLUMNS:
        frac = df[col].isna().mean()
        assert frac <= dc.MAX_NULL_FRACTION, (
            f"{name}: доля пропусков в {col} = {frac:.0%} превышает потолок "
            f"{dc.MAX_NULL_FRACTION:.0%} — вероятно, скачался битый датасет"
        )


def test_full_sq_median_sane(dataset):
    name, df = dataset
    median = pd.to_numeric(df["full_sq"], errors="coerce").median()
    lo, hi = dc.FULL_SQ_MEDIAN_RANGE
    assert lo <= median <= hi, (
        f"{name}: медиана full_sq = {median} вне диапазона {dc.FULL_SQ_MEDIAN_RANGE} "
        f"— возможна смена единиц измерения или сдвиг колонок"
    )


# --------------------- Согласованность train.csv / reference.csv ----------- #

def test_train_and_reference_schema_match():
    train_path = os.path.join(DATA_DIR, "train.csv")
    ref_path = os.path.join(DATA_DIR, "reference.csv")
    train = _read_csv_or_skip(train_path)
    ref = _read_csv_or_skip(ref_path)
    assert list(train.columns) == list(ref.columns), (
        "Схемы train.csv и reference.csv не совпадают — drift_detection.py "
        "не сможет сравнивать признаки по одному и тому же набору колонок"
    )
