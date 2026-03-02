import os
import pandas as pd
from sklearn.preprocessing import StandardScaler


def process_directory(directory, scaler, fit=False):
    files = [f for f in os.listdir(directory) if f.endswith('.csv') and not f.startswith('prep_')]
    features = ['area_sqm', 'distance_to_center_km']

    # Обучаем скейлер только на тренировочной выборке
    if fit:
        all_data = pd.concat([pd.read_csv(os.path.join(directory, f)) for f in files])
        scaler.fit(all_data[features])

    # Трансформируем данные и сохраняем
    for f in files:
        df = pd.read_csv(os.path.join(directory, f))
        df[features] = scaler.transform(df[features])
        df.to_csv(os.path.join(directory, f'prep_{f}'), index=False)


def main():
    scaler = StandardScaler()
    process_directory('train', scaler, fit=True)
    process_directory('test', scaler, fit=False)


if __name__ == "__main__":
    main()