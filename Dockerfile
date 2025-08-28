# Python slim + ffmpeg
FROM python:3.11-slim

# Устанавливаем ffmpeg (для аудио/видео → WAV)
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Зависимости
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Код
COPY bot_webhook.py /app/

# Порт для Uvicorn
EXPOSE 10000

# Запуск FastAPI
CMD ["uvicorn", "bot_webhook:app", "--host", "0.0.0.0", "--port", "10000"]
