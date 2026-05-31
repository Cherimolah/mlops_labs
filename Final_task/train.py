import os
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "train.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "apartment_model.joblib")
ENCODER_PATH = os.path.join(os.path.dirname(__file__), "models", "district_encoder.joblib")

FEATURES = ["full_sq", "life_sq", "floor", "max_floor", "build_year",
            "num_room", "kitch_sq", "sub_area_enc"]
TARGET = "price_doc"


def train():
    df = pd.read_csv(DATA_PATH)

    # Удаляем строки без цены
    df = df[df[TARGET] > 0].dropna(subset=[TARGET]).copy()

    # Приводим числовые колонки к float и заполняем пропуски медианой
    for col in ["life_sq", "floor", "max_floor", "build_year", "num_room", "kitch_sq"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    # Кодируем район
    le = LabelEncoder()
    df["sub_area_enc"] = le.fit_transform(df["sub_area"])

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MAE: {mae:,.0f} rub.  |  R2: {r2:.3f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)
    print(f"Model saved: {MODEL_PATH}")
    print(f"Encoder saved: {ENCODER_PATH}")


if __name__ == "__main__":
    train()