import os
import joblib
import numpy as np
import pandas as pd
from app.schemas import ApartmentFeatures

_BASE = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(_BASE, "models", "apartment_model.joblib")
ENCODER_PATH = os.path.join(_BASE, "models", "district_encoder.joblib")

FEATURES = ["full_sq", "life_sq", "floor", "max_floor", "build_year",
            "num_room", "kitch_sq", "sub_area_enc"]

_model = None
_encoder = None


def load_model():
    global _model, _encoder
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. Run train.py first."
        )
    _model = joblib.load(MODEL_PATH)
    _encoder = joblib.load(ENCODER_PATH)


def get_model():
    if _model is None:
        load_model()
    return _model, _encoder


def predict_price(features: ApartmentFeatures) -> float:
    model, encoder = get_model()
    known_classes = list(encoder.classes_)
    sub_area = features.sub_area if features.sub_area in known_classes else known_classes[0]
    sub_area_enc = encoder.transform([sub_area])[0]
    X = np.array([[
        features.full_sq, features.life_sq, features.floor, features.max_floor,
        features.build_year, features.num_room, features.kitch_sq, sub_area_enc
    ]])
    return float(model.predict(X)[0])


def predict_batch(df: pd.DataFrame) -> list:
    model, encoder = get_model()
    known_classes = list(encoder.classes_)

    df = df.copy()
    for col in ["life_sq", "floor", "max_floor", "build_year", "num_room", "kitch_sq"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(df[col].median())

    df["sub_area"] = df["sub_area"].apply(
        lambda d: d if d in known_classes else known_classes[0]
    )
    df["sub_area_enc"] = encoder.transform(df["sub_area"])

    X = df[FEATURES]
    predictions = model.predict(X)

    results = []
    for idx, (i, row) in enumerate(df.iterrows()):
        price = float(predictions[idx])
        results.append({
            "full_sq": row["full_sq"],
            "life_sq": row["life_sq"],
            "floor": int(row["floor"]),
            "max_floor": int(row["max_floor"]),
            "build_year": int(row["build_year"]),
            "num_room": int(row["num_room"]),
            "kitch_sq": row["kitch_sq"],
            "sub_area": row["sub_area"],
            "price": price,
            "price_formatted": f"{price:,.0f}".replace(",", " ") + " руб.",
        })
    return results
