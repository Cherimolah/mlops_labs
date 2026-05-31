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
├── train.py               # обучение модели и сохранение артефактов
├── requirements.txt
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

## Интеграция с командой

| Участник | Задача | Точка интеграции |
|----------|--------|-----------------|
| **Руслан** | DVC | версионирует `data/sample.csv` и `models/*.joblib` |
| **Илья** | Тесты | покрывает `predict_price()` и `predict_batch()` в `app/model.py` |
| **Никита / Маша** | Docker + CI/CD | собирают образ по `Dockerfile`; `GET /health` используется для healthcheck в pipeline |

---

## Веб-интерфейс

- **Вкладка «Одна квартира»** — форма ввода параметров, мгновенный расчёт цены
- **Вкладка «Загрузить CSV»** — drag-and-drop или выбор файла, таблица с результатами
- **История предсказаний** — все запросы текущей сессии отображаются внизу страницы
