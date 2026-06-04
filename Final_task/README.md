# Apartment Price Predictor

Веб-приложение на **FastAPI** для предсказания стоимости квартиры по её параметрам. Использует предобученную ML-модель (GradientBoostingRegressor) и предоставляет REST API + готовый веб-интерфейс.

Проект выполнен в рамках учебного задания «Final task» (ML Pipeline).

---

## Стек

| Компонент | Технология |
|-----------|-----------|
| Бэкенд | Python 3.11, FastAPI 0.115 |
| ML | scikit-learn 1.5 (GradientBoostingRegressor) |
| Шаблоны | Jinja2 |
| Данные | pandas 2.2 |
| Контейнер | Docker |

---

## Структура проекта

```
.
├── app/
│   ├── __init__.py
│   ├── main.py            # FastAPI: endpoints, история предсказаний
│   ├── model.py           # загрузка модели, predict_price, predict_batch
│   ├── schemas.py         # Pydantic-схемы входных и выходных данных
│   ├── templates/
│   │   └── index.html     # HTML-интерфейс (Jinja2-шаблон)
│   └── static/
│       ├── style.css      # стили
│       └── app.js         # клиентская логика (fetch, вкладки, CSV-загрузка)
├── data/
│   └── sample.csv         # stub-датасет (100 квартир); заменяется через DVC
├── models/
│   ├── apartment_model.joblib    # сериализованная модель (генерируется train.py)
│   └── district_encoder.joblib  # LabelEncoder для районов
├── scripts/
│   ├── download_data.py   # скачивание датасета mlurfuflat с Kaggle
│   └── drift_detection.py # метрики сдвига данных (PSI / KS / Jensen-Shannon)
├── train.py               # обучение модели + сохранение эталона и метрик
├── Jenkinsfile            # ежедневный пайплайн: данные -> дрейф -> переобучение
├── requirements.txt
├── requirements-pipeline.txt  # доп. зависимости пайплайна (kaggle, scipy)
├── Dockerfile
└── .gitignore
```

---

## Быстрый старт

### 1. Клонировать репозиторий

```bash
git clone <url-репозитория>
cd <папка-проекта>
```

### 2. Установить зависимости

```bash
pip install -r requirements.txt
```

### 3. Обучить модель

```bash
python train.py
```

Выведет метрики и сохранит `models/apartment_model.joblib` и `models/district_encoder.joblib`.

### 4. Запустить приложение

```bash
uvicorn app.main:app --reload
```

Открыть в браузере: [http://localhost:8000](http://localhost:8000)

---

## API Endpoints

| Метод | Путь | Описание |
|-------|------|---------|
| `GET` | `/` | Веб-интерфейс (форма + история) |
| `POST` | `/predict` | Предсказание цены для одной квартиры (JSON) |
| `POST` | `/predict/batch` | Пакетное предсказание из CSV-файла |
| `GET` | `/health` | Healthcheck (`{"status": "ok"}`) |

Интерактивная документация API доступна по адресу [http://localhost:8000/docs](http://localhost:8000/docs).

---

## Формат данных

### Запрос `POST /predict`

```json
{
  "area": 65.0,
  "rooms": 2,
  "floor": 5,
  "total_floors": 9,
  "district": "Центральный",
  "year_built": 2005
}
```

| Поле | Тип | Описание | Ограничения |
|------|-----|---------|-------------|
| `area` | float | Площадь, м² | 0 < area ≤ 1000 |
| `rooms` | int | Количество комнат | 1–10 |
| `floor` | int | Этаж | 1–50, ≤ total_floors |
| `total_floors` | int | Этажность дома | 1–50 |
| `district` | string | Район | см. список ниже |
| `year_built` | int | Год постройки | 1900–2030 |

**Доступные районы:** Верх-Исетский, Железнодорожный, Кировский, Ленинский, Октябрьский, Орджоникидзевский, Центральный, Чкаловский.

### Ответ

```json
{
  "price": 6099191.07,
  "price_formatted": "6 099 191 руб.",
  "features": { ... }
}
```

### Формат CSV для `POST /predict/batch`

Файл должен содержать колонки: `area`, `rooms`, `floor`, `total_floors`, `district`, `year_built`.

---

## Docker

### Сборка образа

```bash
docker build -t apartment-predictor .
```

При сборке автоматически запускается `train.py` — модель обучается внутри образа.

### Запуск контейнера

```bash
docker run -p 8000:8000 apartment-predictor
```

Приложение доступно на [http://localhost:8000](http://localhost:8000).

---

## Модель

- **Алгоритм:** GradientBoostingRegressor (scikit-learn)
- **Признаки:** площадь, количество комнат, этаж, этажность, район (label-encoded), год постройки
- **Датасет:** stub-данные 100 квартир (Екатеринбург)
- **Метрики на тестовой выборке:** MAE ≈ 141 787 руб., **R² = 0.997**

Когда команда подготовит реальный датасет — достаточно заменить `data/sample.csv` и перезапустить `python train.py`.


---

## Пайплайн данных и переобучение (MLOps)

Датасет берётся из соревнования Kaggle [mlurfuflat](https://www.kaggle.com/competitions/mlurfuflat).
Ежедневный Jenkins-пайплайн поддерживает модель в актуальном состоянии и
переобучает её **только при сдвиге данных** (data drift).

### 1. Скачивание данных

```bash
python scripts/download_data.py
```

Авторизация в Kaggle — через переменные окружения `KAGGLE_USERNAME` / `KAGGLE_KEY`
(или файл `~/.kaggle/kaggle.json`). Скрипт качает соревнование, находит обучающий
файл и нормализует его в `data/train.csv`.

### 2. Детекция сдвига данных

```bash
python scripts/drift_detection.py
```

Свежий `data/train.csv` сравнивается с эталоном `data/reference.csv` (срез данных,
на котором обучена текущая модель — сохраняется в `train.py`). Считаются метрики:

| Метрика | Что показывает | Порог |
|---------|----------------|-------|
| **PSI** (Population Stability Index) | насколько сдвинулось распределение признака/таргета | <0.1 стабильно · 0.1–0.25 умеренно · >0.25 значимо |
| **KS-тест** (Колмогоров–Смирнов) | статистическая значимость различия распределений | p-value < 0.05 — значимо |
| **Jensen–Shannon distance** | расхождение распределения цены `price_doc` | 0 (совпадают) … 1 |

**Переобучение запускается, если** PSI таргета > 0.2, **или** хотя бы у одного
признака PSI > 0.25, **или** не менее половины признаков имеют PSI > 0.1.

Результат: человекочитаемый `drift_report.json`; при сдвиге создаётся файл-маркер
`RETRAIN_REQUIRED`. Код возврата: `0` — сдвига нет, `10` — сдвиг, `1` — ошибка.

### 3. Jenkins (раз в день)

`Jenkinsfile` описывает пайплайн с триггером `cron('H 3 * * *')`:

1. **Setup** — venv + установка `requirements.txt` и `requirements-pipeline.txt`.
2. **Download data** — `scripts/download_data.py`.
3. **Detect drift** — `scripts/drift_detection.py`.
4. **Retrain** — `train.py` (выполняется только при наличии `RETRAIN_REQUIRED`).
5. **Publish model** — архивирование `models/*.joblib`, `metrics.json`, `reference.csv`.

Нужен Jenkins-credential типа *Username with password* с id **`kaggle-api`**
(username = Kaggle username, password = Kaggle API key). `drift_report.json`
архивируется в каждом запуске.

> Реальный датасет на Kaggle меняется редко, поэтому переобучение в норме не
> срабатывает — пайплайн демонстрирует механику автообновления и реакции на дрейф.

---

## Веб-интерфейс

- **Вкладка «Одна квартира»** — форма ввода параметров, мгновенный расчёт цены

- **История предсказаний** — все запросы текущей сессии отображаются внизу страницы
