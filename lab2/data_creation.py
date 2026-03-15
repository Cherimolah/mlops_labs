import os
import numpy as np
import pandas as pd


def create_ekb_rent_data(num_samples, noise_scale):
    """Генерация данных по стоимости аренды недвижимости в Екатеринбурге."""
    # Площадь от 20 до 200 кв.м.
    area_sqm = np.random.uniform(20, 200, num_samples)

    # Расстояние от Площади 1905 года (от 0 до 15 км — Уралмаш, Химмаш и т.д.)
    distance_to_center_km = np.random.uniform(0, 15, num_samples)

    # Базовая ставка за квадратный метр (убывает по мере удаления от центра)
    price_per_sqm = 1500 - (distance_to_center_km * 60)

    # Итоговая цена + случайный шум (погрешности рынка)
    rent_price_rub = (area_sqm * price_per_sqm) + np.random.normal(0, noise_scale, num_samples)

    # Ограничиваем минимальную цену, чтобы не было отрицательных аномалий
    rent_price_rub = np.maximum(rent_price_rub, 10000)

    return pd.DataFrame({
        'area_sqm': area_sqm,
        'distance_to_center_km': distance_to_center_km,
        'rent_price_rub': rent_price_rub
    })


def main():
    os.makedirs('train', exist_ok=True)
    os.makedirs('test', exist_ok=True)

    # Обучающая выборка (3 набора с нарастающим шумом)
    for i in range(1, 4):
        df_train = create_ekb_rent_data(500, noise_scale=15000 * i)
        df_train.to_csv(f'train/ekb_rent_train_{i}.csv', index=False)

    # Тестовая выборка
    for i in range(1, 3):
        df_test = create_ekb_rent_data(150, noise_scale=20000 * i)
        df_test.to_csv(f'test/ekb_rent_test_{i}.csv', index=False)


if __name__ == "__main__":
    main()