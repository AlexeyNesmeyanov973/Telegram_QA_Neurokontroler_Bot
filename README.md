# Neuro-QC Bot (Webhook на Render)

Aiogram v3 + FastAPI + OpenAI (Audio + Chat). Обрабатывает аудио/видео/файлы/текст и присылает отчёт в Markdown.

## Быстрый старт (Render)

1) Скопируйте файлы в новый Git-репозиторий.  
2) На https://render.com → **New → Blueprint** (используя `render.yaml`) или **New → Web Service** (Docker).  
3) В **Environment** добавьте переменные:
   - `OPENAI_API_KEY` = `sk-...`
   - `TELEGRAM_BOT_TOKEN` = токен вашего бота от @BotFather
   - `TELEGRAM_WEBHOOK_SECRET` = длинная случайная строка (напр. сгенерируйте GUID)
   - *(опционально)* `PUBLIC_BASE_URL` — если у вас свой домен; иначе Render передаст `RENDER_EXTERNAL_URL`
4) Деплойте → после запуска в логах увидите `Webhook set: https://.../webhook/<секрет>`.
5) В Telegram напишите боту `/start`, пришлите файл/аудио/видео/текст.

## Как это работает
- При `startup` приложение вызывает `bot.set_webhook` на свой публичный URL Render.
- Telegram шлёт апдейты на `POST /webhook/<секрет>`.
- FastAPI передаёт апдейты в `dp.feed_update(...)`.
- Для аудио/видео используется `ffmpeg` (Dockerfile уже включает его).

## Примечания
- Порт 10000 публикуется через Uvicorn (см. Dockerfile).
- Если ошиблись с вебхуком — перезапустите сервис, он обновится.
- Можно переопределить модели через переменные:
  - `OPENAI_MODEL_CHAT` (по умолчанию `gpt-4o-mini`)
  - `OPENAI_MODEL_TRANSCRIBE` (по умолчанию `gpt-4o-transcribe`, fallback: `whisper-1`)
