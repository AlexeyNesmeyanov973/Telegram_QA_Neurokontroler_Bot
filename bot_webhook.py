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
                doc.add_paragraph(f"• {_
