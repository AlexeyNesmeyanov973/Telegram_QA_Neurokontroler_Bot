import os, re, json, uuid, asyncio
from pathlib import Path
from typing import Optional

import ffmpeg
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse
from datetime import datetime

# === ENV ===
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_WEBHOOK_SECRET = os.getenv("TELEGRAM_WEBHOOK_SECRET", "")  # придумайте случайную длинную строку
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL")
RENDER_EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL")

if not OPENAI_API_KEY or not TELEGRAM_BOT_TOKEN or not TELEGRAM_WEBHOOK_SECRET:
    raise RuntimeError("Set env vars: OPENAI_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_WEBHOOK_SECRET")

# === OpenAI (async) ===
from openai import AsyncOpenAI
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

OPENAI_MODEL_CHAT = os.getenv("OPENAI_MODEL_CHAT", "gpt-4o-mini")
OPENAI_MODEL_TRANSCRIBE_PREF = os.getenv("OPENAI_MODEL_TRANSCRIBE", "gpt-4o-transcribe")
OPENAI_MODEL_TRANSCRIBE_FALLBACK = "whisper-1"

QUALITY_SYSTEM_PROMPT = (
    "Ты — строгий контролёр качества диалогов. Оцени запись/транскрипт и особенно работу с возражением про время/длительность/темп. "
    "Верни строго валидный JSON:\n"
    "{\n"
    '  "scores": {\n'
    '    "greeting": 0-5,\n'
    '    "needs_analysis": 0-5,\n'
    '    "value_presentation": 0-5,\n'
    '    "objection_time_detected": 0-1,\n'
    '    "objection_time_handling": 0-5,\n'
    '    "next_steps": 0-5,\n'
    '    "empathy_tone": 0-5,\n'
    '    "structure": 0-5,\n'
    '    "overall": 0-100\n'
    "  },\n"
    '  "findings": {\n'
    '    "time_objection_quotes": ["..."],\n'
    '    "key_strengths": ["..."],\n'
    '    "key_issues": ["..."]\n'
    "  },\n"
    '  "actions": ["..."],\n'
    '  "verdict": "1–2 предложения"\n'
    "}\n"
    "Если пункт не оценим — ставь 0 и укажи в issues. Пиши только JSON."
)

# === Файловая система ===
DATA_ROOT = Path("./data")
INBOX = DATA_ROOT / "inbox"
OUTBOX = DATA_ROOT / "outbox"
for p in (DATA_ROOT, INBOX, OUTBOX):
    p.mkdir(parents=True, exist_ok=True)

AUDIO_EXT = {".mp3",".wav",".m4a",".aac",".ogg",".oga",".flac",".webm"}
VIDEO_EXT = {".mp4",".mov",".mkv",".avi",".webm"}
TEXT_EXT  = {".txt",".md",".csv",".rtf"}

def safe_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.\-]+", "_", name).strip("._") or "file"

def is_audio(name: str) -> bool: return Path(name).suffix.lower() in AUDIO_EXT
def is_video(name: str) -> bool: return Path(name).suffix.lower() in VIDEO_EXT
def is_text(name: str)  -> bool: return Path(name).suffix.lower() in TEXT_EXT

def transcode_to_wav(src_path: Path, sr: int = 16000) -> Path:
    dst_path = src_path.with_suffix(".wav")
    (
      ffmpeg.input(str(src_path))
            .output(str(dst_path), ac=1, ar=sr, format="wav")
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
    )
    return dst_path

def read_text_file(path: Path, limit_mb=10.0) -> str:
    data = path.read_bytes()[:int(limit_mb*1024*1024)]
    return data.decode("utf-8", errors="ignore")

async def openai_transcribe(path: Path) -> str:
    try:
        with open(path, "rb") as f:
            r = await client.audio.transcriptions.create(model=OPENAI_MODEL_TRANSCRIBE_PREF, file=f)
        return getattr(r, "text", "") or (r.get("text","") if isinstance(r, dict) else "")
    except Exception:
        with open(path, "rb") as f:
            r = await client.audio.transcriptions.create(model=OPENAI_MODEL_TRANSCRIBE_FALLBACK, file=f)
        return getattr(r, "text", "") or (r.get("text","") if isinstance(r, dict) else "")

async def openai_quality_json(transcript_text: str) -> dict:
    msgs = [{"role":"system","content":QUALITY_SYSTEM_PROMPT},
            {"role":"user","content": transcript_text[:20000]}]
    resp = await client.chat.completions.create(model=OPENAI_MODEL_CHAT, messages=msgs, temperature=0.0)
    content = resp.choices[0].message.content
    m = re.search(r"\{[\s\S]*\}", content)
    raw = m.group(0) if m else content
    try:
        return json.loads(raw)
    except Exception:
        try:
            return json.loads(raw.replace("'", '"'))
        except Exception as e:
            return {"raw": content, "error": f"JSON parse failed: {e}"}

# --- Markdown отчёт ---
def md_report(analysis: dict, src_name: str) -> str:
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    def get(path, default=None):
        cur = analysis
        for k in path:
            if isinstance(cur, dict) and k in cur: cur = cur[k]
            else: return default
        return cur
    lines = [f"# Отчёт по контролю качества ({src_name})", f"_Создано: {ts}_", ""]
    scores = get(["scores"], {})
    if isinstance(scores, dict):
        lines.append("## Оценки")
        for k,v in scores.items(): lines.append(f"- **{k}**: {v}")
        lines.append("")
    findings = get(["findings"], {})
    if isinstance(findings, dict):
        to = findings.get("time_objection_quotes") or []
        if to:
            lines.append("## Фрагменты про возражение времени")
            for q in to[:8]: lines.append(f"> {q}")
            lines.append("")
        ks = findings.get("key_strengths") or []
        if ks:
            lines.append("## Сильные стороны")
            for it in ks[:8]: lines.append(f"- {it}")
            lines.append("")
        ki = findings.get("key_issues") or []
        if ki:
            lines.append("## Проблемы")
            for it in ki[:10]: lines.append(f"- {it}")
            lines.append("")
    actions = get(["actions"], [])
    if actions:
        lines.append("## Рекомендации менеджеру")
        for a in actions[:10]: lines.append(f"- {a}")
        lines.append("")
    verdict = get(["verdict"], "")
    if verdict:
        lines.append("## Вывод"); lines.append(verdict); lines.append("")
    if "error" in analysis:
        lines.append("> ⚠️ JSON не распарсился — приложен сырой ответ в конце.")
        lines.append("")
    return "\n".join(lines)

# --- DOCX & PDF отчёты ---
from docx import Document
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm

def docx_report(analysis: dict, src_name: str, out_path: Path):
    doc = Document()
    doc.add_heading(f"Отчёт по контролю качества ({src_name})", 0)
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    doc.add_paragraph(f"Создано: {ts}")

    scores = analysis.get("scores", {})
    if isinstance(scores, dict):
        doc.add_heading("Оценки", level=1)
        for k, v in scores.items():
            p = doc.add_paragraph()
            p.add_run(f"{k}: ").bold = True
            p.add_run(str(v))

    findings = analysis.get("findings", {})
    if isinstance(findings, dict):
        to = findings.get("time_objection_quotes") or []
        if to:
            doc.add_heading("Фрагменты про возражение времени", level=1)
            for q in to[:8]:
                doc.add_paragraph(q)

        ks = findings.get("key_strengths") or []
        if ks:
            doc.add_heading("Сильные стороны", level=1)
            for it in ks[:8]:
                doc.add_paragraph(f"• {it}")

        ki = findings.get("key_issues") or []
        if ki:
            doc.add_heading("Проблемы", level=1)
            for it in ki[:10]:
                doc.add_paragraph(f"• {it}")

    actions = analysis.get("actions") or []
    if actions:
        doc.add_heading("Рекомендации менеджеру", level=1)
        for a in actions[:10]:
            doc.add_paragraph(f"• {a}")

    verdict = analysis.get("verdict")
    if verdict:
        doc.add_heading("Вывод", level=1)
        doc.add_paragraph(verdict)

    if "error" in analysis:
        doc.add_paragraph("⚠️ JSON не распарсился — приложен сырой ответ в конце.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out_path))

def pdf_report(analysis: dict, src_name: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(out_path), pagesize=A4)
    width, height = A4
    margin = 15 * mm
    x = margin
    y = height - margin

    def writeln(text: str, leading=14):
        nonlocal y
        max_width = width - 2*margin
        from reportlab.pdfbase.pdfmetrics import stringWidth
        words = text.split()
        line = ""
        for w in words:
            cand = (line + " " + w).strip()
            if stringWidth(cand, "Helvetica", 11) <= max_width:
                line = cand
            else:
                c.setFont("Helvetica", 11); c.drawString(x, y, line); y -= leading
                if y < margin: c.showPage(); y = height - margin
                line = w
        if line:
            c.setFont("Helvetica", 11); c.drawString(x, y, line); y -= leading
            if y < margin: c.showPage(); y = height - margin

    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    c.setFont("Helvetica-Bold", 14); c.drawString(x, y, f"Отчёт по контролю качества ({src_name})"); y -= 18
    c.setFont("Helvetica", 10); c.drawString(x, y, f"Создано: {ts}"); y -= 16

    scores = analysis.get("scores", {})
    if isinstance(scores, dict) and scores:
        c.setFont("Helvetica-Bold", 12); c.drawString(x, y, "Оценки"); y -= 16
        for k, v in scores.items(): writeln(f"- {k}: {v}")

    findings = analysis.get("findings", {})
    if isinstance(findings, dict):
        to = findings.get("time_objection_quotes") or []
        if to:
            c.setFont("Helvetica-Bold", 12); c.drawString(x, y, "Фрагменты про возражение времени"); y -= 16
            for q in to[:8]: writeln(f"» {q}")

        ks = findings.get("key_strengths") or []
        if ks:
            c.setFont("Helvetica-Bold", 12); c.drawString(x, y, "Сильные стороны"); y -= 16
            for it in ks[:8]: writeln(f"• {it}")

        ki = findings.get("key_issues") or []
        if ki:
            c.setFont("Helvetica-Bold", 12); c.drawString(x, y, "Проблемы"); y -= 16
            for it in ki[:10]: writeln(f"• {it}")

    actions = analysis.get("actions") or []
    if actions:
        c.setFont("Helvetica-Bold", 12); c.drawString(x, y, "Рекомендации менеджеру"); y -= 16
        for a in actions[:10]: writeln(f"- {a}")

    verdict = analysis.get("verdict")
    if verdict:
        c.setFont("Helvetica-Bold", 12); c.drawString(x, y, "Вывод"); y -= 16
        writeln(verdict)

    if "error" in analysis:
        writeln("⚠️ JSON не распарсился — приложен сырой ответ в конце.")

    c.showPage()
    c.save()

# === Aiogram v3 + FastAPI ===
from aiogram import Bot, Dispatcher, F, Router
from aiogram.types import Update, Message, BufferedInputFile
from aiogram.filters import CommandStart, Command

bot = Bot(TELEGRAM_BOT_TOKEN)
dp = Dispatcher()
rt = Router()
dp.include_router(rt)

HELP = ("Я — Нейро-контролёр качества. Пришлите аудио/видео/файл c разговором или текст — пришлю оценку и рекомендации.\n\n"
        "Команды:\n/start — приветствие\n/help — справка")

@rt.message(CommandStart())
async def on_start(message: Message):
    await message.answer("Привет! " + HELP)

@rt.message(Command("help"))
async def on_help(message: Message):
    await message.answer(HELP)

async def handle_text_and_reply(text: str, message: Message, src_name: str):
    await message.answer("🧠 Анализирую качество...")
    analysis = await openai_quality_json(text)

    # Markdown
    out_base = f"report_{uuid.uuid4().hex[:8]}"
    md_path  = OUTBOX / f"{out_base}.md"
    report_md = md_report(analysis, src_name=src_name)
    with open(md_path, "w", encoding="utf-8") as f: f.write(report_md)

    # DOCX
    docx_path = OUTBOX / f"{out_base}.docx"
    try:
        docx_report(analysis, src_name=src_name, out_path=docx_path)
    except Exception:
        docx_path = None

    # PDF
    pdf_path = OUTBOX / f"{out_base}.pdf"
    try:
        pdf_report(analysis, src_name=src_name, out_path=pdf_path)
    except Exception:
        pdf_path = None

    verdict = analysis.get("verdict") or "Готов отчёт."
    await message.answer(f"✅ Готово: {verdict}")

    # Отправляем все доступные форматы
    with open(md_path, "rb") as f:
        await message.answer_document(BufferedInputFile(f.read(), filename=md_path.name), caption="Отчёт (Markdown)")
    if docx_path and docx_path.exists():
        with open(docx_path, "rb") as f:
            await message.answer_document(BufferedInputFile(f.read(), filename=docx_path.name), caption="Отчёт (DOCX)")
    if pdf_path and pdf_path.exists():
        with open(pdf_path, "rb") as f:
            await message.answer_document(BufferedInputFile(f.read(), filename=pdf_path.name), caption="Отчёт (PDF)")

async def process_media_file(path: Path, message: Message, orig_name: str):
    await message.answer("🎧 Конвертирую и транскрибирую...")
    wav = transcode_to_wav(path)
    text = await openai_transcribe(wav)
    if not (text and text.strip()):
        await message.answer("⚠️ Не удалось получить транскрипт."); return
    await handle_text_and_reply(text, message, src_name=orig_name)

@rt.message(F.voice | F.audio)
async def on_audio(message: Message):
    try:
        file = message.audio or message.voice
        fname = (file.file_name if message.audio else "voice.ogg") or "audio.ogg"
        dst = INBOX / safe_filename(fname)
        await bot.download(file, destination=dst)
        await process_media_file(dst, message, orig_name=fname)
    except Exception as e:
        await message.answer(f"Ошибка аудио: {e}")

@rt.message(F.video | F.video_note)
async def on_video(message: Message):
    try:
        file = message.video or message.video_note
        fname = (message.video.file_name if message.video else "video_note.mp4") or "video.mp4"
        dst = INBOX / safe_filename(fname)
        await bot.download(file, destination=dst)
        await process_media_file(dst, message, orig_name=fname)
    except Exception as e:
        await message.answer(f"Ошибка видео: {e}")

@rt.message(F.document)
async def on_document(message: Message):
    try:
        doc = message.document
        fname = safe_filename(doc.file_name or f"file_{doc.file_unique_id}")
        dst = INBOX / fname
        await bot.download(doc, destination=dst)
        if is_audio(fname) or is_video(fname):
            await process_media_file(dst, message, orig_name=fname)
        elif is_text(fname):
            text = read_text_file(dst)
            await handle_text_and_reply(text, message, src_name=fname)
        else:
            try:
                await process_media_file(dst, message, orig_name=fname)
            except Exception:
                await message.answer("Формат файла не распознан как аудио/видео/текст.")
    except Exception as e:
        await message.answer(f"Ошибка документа: {e}")

@rt.message(F.text)
async def on_text(message: Message):
    txt = message.text.strip()
    if txt: await handle_text_and_reply(txt, message, src_name="text")

# === FastAPI app + webhook ===
from aiogram.types import Update

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
        allowed_updates=["message","edited_message"]
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
