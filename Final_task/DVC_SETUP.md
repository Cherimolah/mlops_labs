# DVC — Версионирование датасетов

Этот документ описывает, как настроить и использовать [DVC](https://dvc.org)
для версионирования датасетов проекта и их синхронизации с удалённым хранилищем.

---

## Что версионируется

| Файл | Описание | DVC-указатель |
|---|---|---|
| `data/train.csv` | Основной обучающий датасет | `data/train.csv.dvc` |
| `data/reference.csv` | Эталонный снапшот (для drift detection) | `data/reference.csv.dvc` |

Сами CSV-файлы **не попадают в Git** — только их `.dvc`-указатели.
Фактические данные хранятся в S3 (или другом remote-хранилище).

---

## Первичная настройка (один раз)

### 1. Установить зависимости пайплайна

```bash
pip install -r requirements-pipeline.txt
```

### 2. Инициализировать DVC в репозитории

```bash
# Если DVC ещё не инициализирован (делается один раз на репозиторий)
dvc init
git add .dvc .dvcignore
git commit -m "chore: init DVC"
```

### 3. Настроить remote-хранилище

#### S3 (AWS / MinIO)

```bash
dvc remote add -d myremote s3://my-mlops-bucket/dvc-store
dvc remote modify myremote region us-east-1

# Для MinIO или другого S3-совместимого хранилища:
# dvc remote modify myremote endpointurl http://minio:9000
# dvc remote modify myremote access_key_id     YOUR_KEY
# dvc remote modify myremote secret_access_key  YOUR_SECRET

git add .dvc/config
git commit -m "chore(dvc): configure S3 remote"
```

#### Google Cloud Storage

```bash
dvc remote add -d myremote gs://my-bucket/dvc-store
# Авторизация через: gcloud auth application-default login
```

#### Azure Blob Storage

```bash
dvc remote add -d myremote azure://mycontainer/dvc-store
dvc remote modify myremote connection_string "DefaultEndpointsProtocol=..."
```

#### Локальная папка (быстрый тест без облака)

```bash
dvc remote add -d myremote /tmp/dvc-remote
```

---

## Повседневные команды

### Первый раз: добавить существующие данные

```bash
dvc add data/train.csv data/reference.csv
git add data/train.csv.dvc data/reference.csv.dvc data/.gitignore
git commit -m "chore(data): track datasets with DVC"
dvc push
```

### Скачать данные на новой машине / в CI

```bash
git clone <repo-url>
cd Final_task
pip install -r requirements-pipeline.txt
dvc pull          # скачает train.csv и reference.csv из remote
```

Или через вспомогательный скрипт:

```bash
python scripts/dvc_push.py pull
```

### После обновления данных (вручную или через download_data.py)

```bash
python scripts/dvc_push.py add    # dvc add + git commit .dvc-файлов
python scripts/dvc_push.py push   # dvc push в remote
```

### Посмотреть статус

```bash
dvc status               # что изменилось локально
dvc remote status        # что не синхронизировано с remote
python scripts/dvc_push.py status
```

### Откатиться к предыдущей версии данных

```bash
# 1. Найти нужный коммит
git log --oneline -- data/train.csv.dvc

# 2. Переключиться на нужный указатель
git checkout <commit-hash> -- data/train.csv.dvc

# 3. Скачать соответствующие данные
dvc pull data/train.csv
```

---

## Jenkins: новые credentials

В Jenkinsfile добавлены две новые стадии — **DVC pull** и **DVC add & push**.
Для их работы нужен credential `aws-s3-dvc` в Jenkins:

1. Jenkins → **Manage Jenkins** → **Credentials** → **System** → **Global** → **Add Credentials**
2. Тип: **Username with password**
3. ID: `aws-s3-dvc`
4. Username: `AWS_ACCESS_KEY_ID`
5. Password: `AWS_SECRET_ACCESS_KEY`

Для GCS/Azure замените переменные окружения в Jenkinsfile на соответствующие.

---

## Архитектура пайплайна с DVC

```
Git-репозиторий         DVC remote (S3)
┌─────────────────┐     ┌──────────────────────────┐
│ train.csv.dvc   │────▶│ ab/cd1234...  (train.csv) │
│ reference.csv   │     │ ef/gh5678...  (ref.csv)   │
│ .dvc/config     │     └──────────────────────────┘
└─────────────────┘
        │
        ▼
  Jenkins pipeline
  ┌──────────────────────────────────────┐
  │ 1. Setup          (pip install)      │
  │ 2. DVC pull       (актуальные данные)│  ← NEW
  │ 3. Download data  (kaggle → train.csv│
  │ 4. DVC add&push   (версия в remote)  │  ← NEW
  │ 5. Detect drift                      │
  │ 6. Retrain        (если нужно)       │
  │ 7. DVC push ref   (новый эталон)     │  ← NEW
  │ 8. Publish model                     │
  └──────────────────────────────────────┘
```

---

## Структура новых файлов

```
Final_task/
├── .dvc/
│   ├── config           # remote URL и настройки
│   └── .gitignore       # исключает кэш из Git
├── data/
│   ├── .gitignore       # исключает train.csv и reference.csv из Git
│   ├── train.csv.dvc    # DVC-указатель (md5 + размер)
│   └── reference.csv.dvc
├── scripts/
│   └── dvc_push.py      # pull / add / push / status
├── requirements-pipeline.txt   # dvc, dvc-s3, kaggle, scipy
└── .gitignore           # обновлён: data/*.csv + .dvc/cache/
```
