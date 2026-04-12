FROM python:3.12-slim

EXPOSE 8000

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir
COPY . .

CMD ["python3", "main.py"]