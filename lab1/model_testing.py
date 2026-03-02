import os
import pandas as pd
import pickle
from sklearn.metrics import mean_absolute_error


def main():
    # Загружаем модель
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Загружаем тестовые данные
    test_files = [f for f in os.listdir('test') if f.startswith('prep_')]
    df_test_all = pd.concat([pd.read_csv(os.path.join('test', f)) for f in test_files])

    X_test = df_test_all[['area_sqm', 'distance_to_center_km']]
    y_test = df_test_all['rent_price_rub']

    # Делаем предсказание
    predictions = model.predict(X_test)

    # Считаем среднюю абсолютную ошибку
    mae = mean_absolute_error(y_test, predictions)

    # Формируем итоговый вывод
    print(f"Model Mean Absolute Error (MAE) is {mae:.2f} RUB")


if __name__ == "__main__":
    main()
