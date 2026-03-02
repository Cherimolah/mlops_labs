#!/bin/bash

# Устанавливаем зависимости
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Запускаем конвейер
python3 data_creation.py
python3 data_preprocessing.py
python3 model_preparation.py
python3 model_testing.py