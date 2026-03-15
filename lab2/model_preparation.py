import os
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor

def main():
    # Загружаем все подготовленные тренировочные данные
    train_files = [f for f in os.listdir('train') if f.startswith('prep_')]
    df_train_all = pd.concat([pd.read_csv(os.path.join('train', f)) for f in train_files])

    X_train = df_train_all[['area_sqm', 'distance_to_center_km']]
    y_train = df_train_all['rent_price_rub']

    # Обучаем модель
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Сохраняем готовую модель
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    main()
    