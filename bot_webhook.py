# === Aiogram v3 + FastAPI (guided flow) ===
from aiogram import Bot, Dispatcher, F, Router
from aiogram.types import Update, Message, BufferedInputFile, InlineKeyboardButton, InlineKeyboardMarkup, CallbackQuery
from aiogram.filters import CommandStart, Command

# FSM
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.fsm.context import FSMContext

bot = Bot(TELEGRAM_BOT_TOKEN)
dp  = Dispatcher(storage=MemoryStorage())
rt  = Router()
dp.include_router(rt)

MIN_TEXT_CHARS = int(os.getenv("MIN_TEXT_CHARS", 120))

HELP = (
    "Я — Нейро-контролёр качества.\n"
    "Шаги:\n"
    "1) Выберите формат (текст или файл)\n"
    "2) Пришлите материал\n"
    "3) Подтвердите начало анализа — и я пришлю отчёт (Markdown/DOCX/PDF)\n\n"
    "Команды:\n"
    "/start — главное меню\n"
    "/help — справка\n"
    "/cancel — отмена текущего шага"
)

class Flow(StatesGroup):
    menu       = State()  # показ меню
    waiting_text = State()
    waiting_file = State()
    confirm      = State()  # подтверждение перед анализом

def kb_main_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🔤 Анализировать текст", callback_data="flow:text")],
        [InlineKeyboardButton(text="📁 Загрузить аудио/видео/файл", callback_data="flow:file")],
    ])

def kb_confirm() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="✅ Начать анализ", callback_data="flow:go")],
        [InlineKeyboardButton(text="↩️ Отмена",        callback_data="flow:cancel")],
    ])

def kb_cancel() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="↩️ Отмена", callback_data="flow:cancel")],
    ])

# -------- Команды --------
@rt.message(CommandStart())
async def on_start(message: Message, state: FSMContext):
    await state.clear()
    await state.set_state(Flow.menu)
    await message.answer(
        "Привет! Давай подготовим материал к анализу.\nВыбери вариант ниже:",
        reply_markup=kb_main_menu()
    )

@rt.message(Command("help"))
async def on_help(message: Message):
    await message.answer(HELP, reply_markup=kb_main_menu())

@rt.message(Command("cancel"))
async def on_cancel_cmd(message: Message, state: FSMContext):
    await state.clear()
    await state.set_state(Flow.menu)
    await message.answer("Ок, вернулись в главное меню.", reply_markup=kb_main_menu())

# -------- Навигация меню --------
@rt.callback_query(F.data == "flow:text")
async def cb_flow_text(cb: CallbackQuery, state: FSMContext):
    await state.set_state(Flow.waiting_text)
    await cb.message.edit_text(
        f"Пришли текст (транскрипт/диалог). Минимум {MIN_TEXT_CHARS} символов.\n"
        "Когда будет готово — я попрошу подтвердить анализ.",
        reply_markup=kb_cancel()
    )
    await cb.answer()

@rt.callback_query(F.data == "flow:file")
async def cb_flow_file(cb: CallbackQuery, state: FSMContext):
    await state.set_state(Flow.waiting_file)
    await cb.message.edit_text(
        "Пришли файл: 🎧 аудио / 🎥 видео / 📄 документ (txt/md/csv/rtf).\n"
        "После загрузки попрошу подтвердить анализ.",
        reply_markup=kb_cancel()
    )
    await cb.answer()

@rt.callback_query(F.data == "flow:cancel")
async def cb_flow_cancel(cb: CallbackQuery, state: FSMContext):
    await state.clear()
    await state.set_state(Flow.menu)
    await cb.message.edit_text("Отменено. Выбери, что делать дальше:", reply_markup=kb_main_menu())
    await cb.answer()

# -------- Приём текста --------
@rt.message(Flow.waiting_text, F.text)
async def on_text_in_state(message: Message, state: FSMContext):
    txt = (message.text or "").strip()
    if len(txt) < MIN_TEXT_CHARS:
        await message.answer(
            f"Текста маловато ({len(txt)} симв.). Пришли ≥ {MIN_TEXT_CHARS} символов "
            "или вернись в меню через /cancel.",
            reply_markup=kb_cancel()
        )
        return
    await state.update_data(src_type="text", text=txt, src_name="text.txt")
    await state.set_state(Flow.confirm)
    await message.answer(
        "Принято ✅\nНажми «Начать анализ», чтобы запустить обработку.",
        reply_markup=kb_confirm()
    )

# -------- Приём файлов --------
async def _accept_media_and_ask_confirm(path: Path, message: Message, state: FSMContext, orig_name: str):
    await state.update_data(src_type="file", file_path=str(path), src_name=orig_name)
    await state.set_state(Flow.confirm)
    await message.answer(
        f"Файл получен: *{orig_name}*.\nНажми «Начать анализ», чтобы запустить обработку.",
        reply_markup=kb_confirm(),
        parse_mode="Markdown"
    )

@rt.message(Flow.waiting_file, F.voice | F.audio)
async def on_audio(message: Message, state: FSMContext):
    try:
        file = message.audio or message.voice
        fname = (file.file_name if message.audio else "voice.ogg") or "audio.ogg"
        dst = INBOX / safe_filename(fname)
        await bot.download(file, destination=dst)
        await _accept_media_and_ask_confirm(dst, message, state, orig_name=fname)
    except Exception as e:
        await message.answer(f"Ошибка аудио: {e}", reply_markup=kb_cancel())

@rt.message(Flow.waiting_file, F.video | F.video_note)
async def on_video(message: Message, state: FSMContext):
    try:
        file = message.video or message.video_note
        fname = (message.video.file_name if message.video else "video_note.mp4") or "video.mp4"
        dst = INBOX / safe_filename(fname)
        await bot.download(file, destination=dst)
        await _accept_media_and_ask_confirm(dst, message, state, orig_name=fname)
    except Exception as e:
        await message.answer(f"Ошибка видео: {e}", reply_markup=kb_cancel())

@rt.message(Flow.waiting_file, F.document)
async def on_document(message: Message, state: FSMContext):
    try:
        doc = message.document
        fname = safe_filename(doc.file_name or f"file_{doc.file_unique_id}")
        dst = INBOX / fname
        await bot.download(doc, destination=dst)
        # принимаем любой документ — аудио/видео/текст определим дальше при анализе
        await _accept_media_and_ask_confirm(dst, message, state, orig_name=fname)
    except Exception as e:
        await message.answer(f"Ошибка документа: {e}", reply_markup=kb_cancel())

# -------- Подтверждение и запуск анализа --------
@rt.callback_query(Flow.confirm, F.data == "flow:go")
async def cb_go_analyze(cb: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    src_type = data.get("src_type")
    src_name = data.get("src_name") or "source"
    await cb.message.edit_text("🧠 Анализирую качество...")

    try:
        if src_type == "text":
            text = data.get("text", "")
            await handle_text_and_reply(text, cb.message, src_name=src_name)
        elif src_type == "file":
            path = Path(data.get("file_path"))
            # определим тип: если текст — читаем; иначе конвертим и транскрибируем
            if is_text(path.name):
                text = read_text_file(path)
                await handle_text_and_reply(text, cb.message, src_name=src_name)
            else:
                await process_media_file(path, cb.message, orig_name=src_name)
        else:
            await cb.message.answer("Не понял тип источника. Попробуй снова из меню.", reply_markup=kb_main_menu())
            await state.clear()
            return
        await state.clear()
        await state.set_state(Flow.menu)
    except Exception as e:
        await cb.message.answer(f"Ошибка при анализе: {e}", reply_markup=kb_main_menu())
        await state.clear()
        await state.set_state(Flow.menu)

    await cb.answer()

# -------- Текст вне потока: показываем меню, не анализируем сразу --------
@rt.message(F.text)
async def on_text_default(message: Message, state: FSMContext):
    s = await state.get_state()
    if s in {Flow.waiting_text.state, Flow.waiting_file.state, Flow.confirm.state}:
        # уже в процессе — хендлеры выше разрулят
        return
    await state.set_state(Flow.menu)
    await message.answer(
        "Выбери, что будем анализировать:",
        reply_markup=kb_main_menu()
    )

# === FastAPI app + webhook ===
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse

app = FastAPI()

@app.get("/", response_class=PlainTextResponse)
async def root():
    return "OK: neuro-qc-bot webhook"

def build_webhook_url() -> str:
    base = PUBLIC_BASE_URL or RENDER_EXTERNAL_URL
    if not base:
        raise RuntimeError("No PUBLIC_BASE_URL or RENDER_EXTERNAL_URL provided by Render.")
    base = base.rstrip("/")
    return f"{base}/webhook/{TELEGRAM_WEBHOOK_SECRET}"

@app.on_event("startup")
async def on_startup():
    url = build_webhook_url()
    await bot.set_webhook(
        url,
        secret_token=TELEGRAM_WEBHOOK_SECRET,
        drop_pending_updates=True,
        allowed_updates=["message","edited_message","callback_query"]
    )
    print("Webhook set:", url)

@app.on_event("shutdown")
async def on_shutdown():
    try:
        await bot.delete_webhook(drop_pending_updates=False)
    except Exception:
        pass

@app.post("/webhook/{secret}")
async def telegram_webhook(secret: str, request: Request):
    if secret != TELEGRAM_WEBHOOK_SECRET:
        raise HTTPException(status_code=403, detail="forbidden")
    data = await request.json()
    try:
        update = Update.model_validate(data)
    except Exception:
        raise HTTPException(status_code=400, detail="bad update")
    await dp.feed_update(bot, update)
    return JSONResponse({"ok": True})
