import os, re, json, uuid, asyncio
from pathlib import Path
from typing import Optional
import ffmpeg
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

# --- Load env vars (from Render) ---
OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY", "")
TELEGRAM_BOT_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_WEBHOOK_SECRET = os.getenv("TELEGRAM_WEBHOOK_SECRET", "")
MIN_TEXT_CHARS       = int(os.getenv("MIN_TEXT_CHARS", "120"))
FSM_TIMEOUT          = int(os.getenv("FSM_TIMEOUT", "180"))  # секунды

if not OPENAI_API_KEY or not TELEGRAM_BOT_TOKEN or not TELEGRAM_WEBHOOK_SECRET:
    raise RuntimeError("Env vars: OPENAI_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_WEBHOOK_SECRET required.")

from openai import AsyncOpenAI
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

QUALITY_SYSTEM_PROMPT = (
    "Ты — строгий контролёр качества диалогов... (оставляем тот же промпт)"
)

# FS setup
DATA_ROOT = Path("./data")
INBOX = DATA_ROOT / "inbox"
OUTBOX = DATA_ROOT / "outbox"
for p in (DATA_ROOT, INBOX, OUTBOX):
    p.mkdir(exist_ok=True, parents=True)

HISTORY_FILE = DATA_ROOT / "history.json"
if not HISTORY_FILE.exists():
    HISTORY_FILE.write_text("[]")

# Utils (same as before)
# ... [safe_filename, is_audio/video/text, transcode, read_text_file, OpenAI transcribe and analyze functions, report generators MD/DOCX/PDF]
# (Omitted here to save space — reuse your existing code)

# File preview utility
def file_preview(path: Path):
    """Return a short preview dict: name, size_kb, type, duration_or_chars."""
    info = {"name": path.name, "size_kb": round(path.stat().st_size / 1024, 1)}
    if is_text(path.name):
        content = path.read_text(errors="ignore")
        info["type"] = "text"
        info["chars"] = len(content)
    elif is_audio(path.name) or is_video(path.name):
        # extract duration via ffmpeg
        try:
            probe = ffmpeg.probe(str(path))
            dur = float(probe["format"]["duration"])
            info["type"] = "audio" if is_audio(path.name) else "video"
            info["duration_sec"] = int(dur)
        except:
            info["type"] = "media"
    else:
        info["type"] = "unknown"
    return info

# History append
def add_history_entry(src_name: str, verdict: str):
    hist = json.loads(HISTORY_FILE.read_text())
    hist.append({"id": src_name, "timestamp": datetime.utcnow().isoformat(), "verdict": verdict})
    HISTORY_FILE.write_text(json.dumps(hist, ensure_ascii=False, indent=2))

# Aiogram + FastAPI with guided flow + preview + timeout
from aiogram import Bot, Dispatcher, F, Router
from aiogram.types import Update, Message, BufferedInputFile, InlineKeyboardButton, InlineKeyboardMarkup, CallbackQuery
from aiogram.filters import CommandStart, Command
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.fsm.context import FSMContext

bot = Bot(TELEGRAM_BOT_TOKEN)
dp = Dispatcher(storage=MemoryStorage())
rt = Router()
dp.include_router(rt)

HELP = (
    "Я — Нейро‑контролёр качества.\n"
    "1) Выберите формат: текст или файл\n"
    "2) Пришлите\n"
    "3) Подтвердите, и я отправлю отчёты (MD/DOCX/PDF)\n\n"
    "Команды: /cancel, /help"
)

class Flow(StatesGroup):
    menu = State()
    waiting_text = State()
    waiting_file = State()
    confirm = State()

def kb_main_menu():
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton("🔤 Текст", callback_data="flow:text")],
        [InlineKeyboardButton("📁 Файл", callback_data="flow:file")],
    ])
def kb_confirm(): return InlineKeyboardMarkup([[InlineKeyboardButton("✅ Анализ", "flow:go")], [InlineKeyboardButton("↩️ Отмена", "flow:cancel")]])
def kb_cancel(): return InlineKeyboardMarkup([[InlineKeyboardButton("↩️ Отмена", "flow:cancel")]])

# Helper to auto-clear state after timeout
async def auto_clear(state: FSMContext, timeout=FSM_TIMEOUT):
    await asyncio.sleep(timeout)
    if await state.get_state():
        await state.clear()
        # can't send message here due to missing context.

@rt.message(CommandStart())
async def cmd_start(msg: Message, state: FSMContext):
    await state.clear()
    await state.set_state(Flow.menu)
    asyncio.create_task(auto_clear(state))
    await msg.answer("Привет, выбери:", reply_markup=kb_main_menu())

@rt.message(Command("help"))
async def cmd_help(msg, state: FSMContext):
    await msg.answer(HELP, reply_markup=kb_main_menu())

@rt.message(Command("cancel"))
async def cmd_cancel(msg, state: FSMContext):
    await state.clear()
    await msg.answer("Отмена. Вернулись в меню.", reply_markup=kb_main_menu())

@rt.callback_query(F.data == "flow:text")
async def cb_text(cb: CallbackQuery, state: FSMContext):
    await state.set_state(Flow.waiting_text); asyncio.create_task(auto_clear(state))
    await cb.message.edit_text(f"Пришли текст ≥ {MIN_TEXT_CHARS} символов.", reply_markup=kb_cancel())
    await cb.answer()

@rt.callback_query(F.data == "flow:file")
async def cb_file(cb: CallbackQuery, state: FSMContext):
    await state.set_state(Flow.waiting_file); asyncio.create_task(auto_clear(state))
    await cb.message.edit_text("Пришли файл (аудио/видео/документ).", reply_markup=kb_cancel())
    await cb.answer()

@rt.callback_query(F.data == "flow:cancel")
async def cb_cancel(cb: CallbackQuery, state: FSMContext):
    await state.clear()
    await cb.message.edit_text("Отменили. Меню:", reply_markup=kb_main_menu())
    await cb.answer()

@rt.message(Flow.waiting_text, F.text)
async def handle_text(msg: Message, state: FSMContext):
    txt = msg.text.strip()
    if len(txt) < MIN_TEXT_CHARS:
        await msg.answer(f"Слишком коротко ({len(txt)}), минимум {MIN_TEXT_CHARS}", reply_markup=kb_cancel())
        return
    await state.update_data(src_type="text", text=txt, src_name="text.txt")
    await state.set_state(Flow.confirm)
    preview = f"Текст: {len(txt)} символов"
    await msg.answer(f"{preview}\n{ 'Нажми Анализ' }", reply_markup=kb_confirm())

@rt.message(Flow.waiting_file, F.document | F.audio | F.voice | F.video | F.video_note)
async def handle_file(msg: Message, state: FSMContext):
    file = msg.document or msg.audio or msg.voice or msg.video or msg.video_note
    fname = (getattr(file, 'file_name', None) or "file.bin")
    dst = INBOX / fname
    await bot.download(file, dst)
    info = file_preview(dst)
    preview = f"Файл: {info}"
    await state.update_data(src_type="file", file_path=str(dst), src_name=fname)
    await state.set_state(Flow.confirm)
    await msg.answer(preview + "\nНажми Анализ", reply_markup=kb_confirm(), parse_mode="Markdown")

@rt.callback_query(Flow.confirm, F.data == "flow:go")
async def cb_go(cb: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    src_type = data.get("src_type")
    src_name = data.get("src_name")
    await cb.message.edit_text("Анализирую...")
    if src_type == "text":
        await handle_text_and_reply(data["text"], cb.message, src_name=src_name)
    else:
        path = Path(data["file_path"])
        if is_text(path.name):
            t = read_text_file(path)
            await handle_text_and_reply(t, cb.message, src_name=src_name)
        else:
            await process_media_file(path, cb.message, orig_name=src_name)
    add_history_entry(src_name, "ok")
    await state.clear()
    await state.set_state(Flow.menu)
    await cb.answer()

@rt.message(F.text)
async def fallback(msg: Message, state: FSMContext):
    if not await state.get_state():
        await state.set_state(Flow.menu)
        await msg.answer("Меню:", reply_markup=kb_main_menu())

from fastapi import FastAPI
app = FastAPI()

@app.on_event("startup")
async def on_startup():
    url = (os.getenv("PUBLIC_BASE_URL") or os.getenv("RENDER_EXTERNAL_URL")).rstrip("/") + f"/webhook/{TELEGRAM_WEBHOOK_SECRET}"
    await bot.set_webhook(url, secret_token=TELEGRAM_WEBHOOK_SECRET, drop_pending_updates=True, allowed_updates=["message","callback_query"])

@app.post("/webhook/{secret}")
async def webhook(secret: str, request: Request):
    if secret != TELEGRAM_WEBHOOK_SECRET:
        raise HTTPException(403)
    data = await request.json()
    upd = Update.model_validate(data)
    await dp.feed_update(bot, upd)
    return JSONResponse({"ok": True})

@app.get("/", response_class=PlainTextResponse)
async def root():
    return "OK"
