FROM python:3.11-slim

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY bot_webhook.py /app/

EXPOSE 10000
CMD ["uvicorn", "bot_webhook:app", "--host", "0.0.0.0", "--port", "10000"]

