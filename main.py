"""
P1 Q5 — Data Analyst Telegram Bot

A Telegram bot that receives a plain-text data-analysis question, works out
the answer with an LLM tool-calling loop (the model can run pandas code and
fetch public URLs), and replies with exactly one JSON object:

    {"answer": <shaped as the question asks>, "log_url": "https://.../logs/run.jsonl"}

Run it:

    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    cp .env.example .env   # fill in TELEGRAM_BOT_TOKEN, GEMINI_API_KEY, PUBLIC_BASE_URL, ...
    uvicorn main:app --host 0.0.0.0 --port 8009

The bot reaches OUT to Telegram (long-polling via getUpdates) rather than
requiring Telegram to reach IN via a webhook — no domain, TLS cert, or
inbound firewall rule is needed for message delivery itself. On startup the
app deletes any previously-registered webhook (harmless if there wasn't
one) and starts a background poll loop. PUBLIC_BASE_URL is still used for
one thing: building the log_url this bot reports, since GET /logs/run.jsonl
still needs to be a public URL per the assignment — see README.md for why
that means you still want a real domain + HTTPS in front of this app even
though Telegram delivery no longer requires it.

Every step of every run (incoming message, tool calls, tool output, final
answer) is appended as one JSON object per line to run.jsonl next to this
file, and served back out at GET /logs/run.jsonl — that's the log_url this
bot reports.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse, PlainTextResponse

import sandbox

APP_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(APP_DIR, ".env"))

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "").strip().rstrip("/")
# Google AI Studio (Gemini) via its OpenAI-compatible endpoint is the primary
# provider — a plain API key that doesn't expire on a timer (unlike AI Pipe's
# 7-day JWTs), so it survives an unknown grading date. Tried in order, first
# model that answers wins, so a quota/outage on one doesn't sink the run.
LLM_BASE_URL = os.environ.get(
    "LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai"
).rstrip("/")
LLM_MODELS = [
    # Ordered by capability (Pro tier excluded — it doesn't appear on this
    # project's free-tier quota dashboard, so it's assumed unusable there),
    # falling through to progressively cheaper/higher-quota Flash-Lite
    # models so a low daily cap on one model doesn't stall a whole run.
    # NOTE: gemini-2.5-flash was previously observed 404ing on this
    # OpenAI-compatible /chat/completions route on this project even though
    # it's a listed model — if you see repeated 404s for it in run.jsonl,
    # that's why; the loop just falls through to the next model either way.
    m.strip() for m in os.environ.get(
        "LLM_MODELS",
        "gemini-3.6-flash,gemini-3.5-flash,gemini-3-flash-preview,"
        "gemini-flash-latest,gemini-2.5-flash,gemini-2.0-flash,"
        "gemini-3.5-flash-lite,gemini-3.1-flash-lite,gemini-2.5-flash-lite,"
        "gemini-flash-lite-latest,gemini-2.0-flash-lite",
    ).split(",") if m.strip()
]
ALLOWED_USER_IDS = {
    int(x) for x in os.environ.get("ALLOWED_TELEGRAM_USER_IDS", "").split(",") if x.strip()
}
AGENT_MAX_STEPS = int(os.environ.get("AGENT_MAX_STEPS", "8"))
AGENT_STEP_TIMEOUT = int(os.environ.get("AGENT_STEP_TIMEOUT", "20"))
# Wall-clock budget for the whole tool-calling loop, independent of step
# count: a run that's spending its steps on slow model calls/retries (see
# the 429/503 cascade in a real deployed run) can burn AGENT_MAX_STEPS
# without ever exceeding this, and vice versa. When it's exceeded, the next
# call drops run_python from the offered tools and adds a "time's up"
# nudge, so the model is pushed to answer with whatever it has rather than
# silently exhausting the step budget into a null reply.
AGENT_TIME_BUDGET = int(os.environ.get("AGENT_TIME_BUDGET_SECONDS", "90"))
HISTORY_TURNS = int(os.environ.get("HISTORY_TURNS", "12"))

DB_PATH = os.environ.get("BOT_DB_PATH", os.path.join(APP_DIR, "bot_agent.db"))
LOG_PATH = os.environ.get("RUN_LOG_PATH", os.path.join(APP_DIR, "run.jsonl"))

TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("p1q5")


def llm_key() -> str:
    """Resolved per call so a refreshed key is picked up without a restart.
    LLM_API_KEY is a generic override (set it if LLM_BASE_URL points at a
    different OpenAI-compatible provider); GEMINI_API_KEY / GOOGLE_API_KEY
    are the Google AI Studio key names."""
    for name in ("LLM_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return ""


# --------------------------------------------------------------------------
# Storage: chat history (for multi-turn questions) + seen Telegram update ids
# (a defense-in-depth backstop against reprocessing — see poll_loop() for
# why getUpdates' own offset is normally enough on its own) + the JSONL run
# log this bot's log_url points at.
# --------------------------------------------------------------------------

def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    with _conn() as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS messages ("
            " chat_id INTEGER, role TEXT, content TEXT, ts REAL)"
        )
        conn.execute("CREATE TABLE IF NOT EXISTS seen_updates (update_id INTEGER PRIMARY KEY)")


def remember(chat_id: int, role: str, content: str) -> None:
    with _conn() as conn:
        conn.execute(
            "INSERT INTO messages (chat_id, role, content, ts) VALUES (?, ?, ?, ?)",
            (chat_id, role, content, time.time()),
        )


def history(chat_id: int, limit: int) -> list[dict[str, str]]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE chat_id = ? ORDER BY ts DESC LIMIT ?",
            (chat_id, limit),
        ).fetchall()
    return [{"role": role, "content": content} for role, content in reversed(rows)]


def is_new_update(update_id: int | None) -> bool:
    if update_id is None:
        return True
    with _conn() as conn:
        cur = conn.execute(
            "INSERT OR IGNORE INTO seen_updates (update_id) VALUES (?)", (update_id,)
        )
        return cur.rowcount == 1


_log_lock = threading.Lock()


def log_event(run_id: str, chat_id: Any, kind: str, **fields: Any) -> None:
    record = {"ts": time.time(), "run_id": run_id, "chat_id": chat_id, "kind": kind, **fields}
    line = json.dumps(record, ensure_ascii=False, default=str)
    with _log_lock:
        with open(LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    log.info("%s", line[:500])


# --------------------------------------------------------------------------
# Telegram
# --------------------------------------------------------------------------

async def tg_call(method: str, **params: Any) -> Any:
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(f"{TELEGRAM_API}/{method}", json=params)
    data = resp.json()
    if not data.get("ok"):
        raise RuntimeError(f"telegram {method} failed: {data}")
    return data["result"]


async def send_message(chat_id: int, text: str) -> None:
    await tg_call("sendMessage", chat_id=chat_id, text=text)


# --------------------------------------------------------------------------
# The agent: one LLM tool-calling loop per incoming message.
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a meticulous data analyst answering one question at a time from \
a Telegram message. The question may embed its data directly in the text, or point at a \
public dataset (MOSPI or similar) that you must fetch and compute over yourself — never \
guess a number you could compute.

If the latest message is not actually a data-analysis question — a bare /start, a greeting \
like "hi", small talk, or anything with no question to compute an answer to — do NOT call \
run_python at all. There is nothing to fetch or explore for a greeting; your own source code \
and database are not the subject of any question and reading them wastes tool calls for no \
reason. Just call final_answer directly with a short friendly reply (e.g. answer_json \
'"Hi! Send me a data-analysis question and I\\'ll work out the answer."'), or with an \
`answer_json` of `null` if the message gives no JSON template to match.

You have two tools:
- run_python(code): execute Python. `pd` (pandas) and `np` (numpy) are preloaded. Use \
  fetch(url) to download a public URL as text, or fetch_df(url) to parse it straight into \
  a DataFrame (or list of DataFrames for a multi-table HTML page) — picked automatically \
  from the URL's extension (csv, json, xlsx, html, parquet). Variables persist between \
  calls in this conversation, so fetch once and explore over several calls. Anything you \
  print() is returned to you as output; nothing else is.
- final_answer(answer_json, reasoning): call this exactly once, when you are done. \
  `answer_json` is a STRING containing ONLY the JSON-encoded VALUE that belongs INSIDE the \
  message's "answer" key — never the surrounding {"answer": ..., "log_url": ...} wrapper \
  itself. E.g. if the template is {"answer": {"state": "..."}, "log_url": "..."}, pass the \
  string '{"state": "Assam"}' (just that inner object) — NOT '{"answer": {"state": \
  "Assam"}, "log_url": "..."}'. If the template's answer is a plain number, pass the string \
  '42'; if a plain string, pass '"some text"' (quoted, so it parses as JSON). Never include \
  the literal words "answer" or "log_url" as keys inside answer_json — those belong to the \
  outer wrapper this system builds automatically, even if you've seen a real log_url earlier \
  in this conversation; do not copy it into answer_json. `reasoning` is one or two sentences \
  for the audit log, not shown to anyone.

Read the message carefully for the exact JSON shape it wants back, and for any earlier \
messages in this chat that supply context or data for the current question — answer only \
the most recent question, using earlier messages purely as background. If the data can't \
be found or the question is ambiguous, still call final_answer with your best-supported \
guess rather than leaving the conversation without a reply."""

RUN_PYTHON_TOOL = {
    "type": "function",
    "function": {
        "name": "run_python",
        "description": (
            "Execute Python for data analysis. pandas as pd, numpy as np are "
            "preloaded. fetch(url) downloads a public URL as text; fetch_df(url) "
            "parses it straight into a DataFrame. State persists between calls."
        ),
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "Python source to run."}},
            "required": ["code"],
        },
    },
}

FINAL_ANSWER_TOOL = {
    "type": "function",
    "function": {
        "name": "final_answer",
        "description": "Submit the final answer. Call exactly once, when done.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer_json": {
                    "type": "string",
                    "description": (
                        "The answer as a JSON-encoded string, shaped exactly as the "
                        "question's own JSON template requests, e.g. '{\"state\": "
                        "\"Assam\"}' or '42' or '\"some text\"'."
                    ),
                },
                "reasoning": {"type": "string", "description": "One or two sentences, for the log only."},
            },
            "required": ["answer_json"],
        },
    },
}

TOOLS = [RUN_PYTHON_TOOL, FINAL_ANSWER_TOOL]
FINAL_ANSWER_ONLY_TOOLS = [FINAL_ANSWER_TOOL]


async def llm_chat(messages: list[dict], tools: list[dict] = TOOLS) -> dict:
    key = llm_key()
    if not key:
        raise RuntimeError("no LLM key configured (set GEMINI_API_KEY, GOOGLE_API_KEY, or LLM_API_KEY)")
    errors = []
    async with httpx.AsyncClient(timeout=90) as client:
        for model in LLM_MODELS:
            resp = await client.post(
                f"{LLM_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={"model": model, "messages": messages, "tools": tools, "temperature": 0},
            )
            if resp.status_code < 400:
                return resp.json()["choices"][0]["message"]
            errors.append(f"{model}: {resp.status_code} {resp.text[:200]}")
    raise RuntimeError("every model in LLM_MODELS failed — " + " | ".join(errors))


_MISSING = object()


def extract_final_answer(raw: str) -> Any:
    """Parse final_answer's answer_json, defensively unwrapping a common
    model mistake: copying the ENTIRE {"answer": ..., "log_url": ...}
    template into answer_json instead of just the inner value (seen live —
    the model had a real log_url sitting in its own chat history from an
    earlier reply and mimicked the whole shape). No legitimate answer's
    real content would ever itself be a dict with a "log_url" key, so
    unwrapping on sight of one is always correct, never a false positive."""
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw  # model forgot to quote a plain string — use it as-is
    if isinstance(parsed, dict) and "log_url" in parsed and "answer" in parsed:
        return parsed["answer"]
    return parsed


async def run_agent(chat_id: int, run_id: str) -> Any:
    """Runs the tool-calling loop for the latest message in `chat_id`'s
    history (already stored by the caller). Returns the answer value, or
    None if no final_answer was produced."""
    if not llm_key():
        log_event(run_id, chat_id, "no_llm_key")
        return None

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += history(chat_id, HISTORY_TURNS)
    state: dict = {}
    deadline = time.monotonic() + AGENT_TIME_BUDGET
    forced_final = False

    for step in range(AGENT_MAX_STEPS):
        tools = TOOLS
        if not forced_final and time.monotonic() > deadline:
            # Wall-clock budget is up: stop offering run_python (a model
            # can't keep "thinking" forever) and nudge it to answer with
            # whatever it has, instead of silently running out the step
            # count into a null reply - independent of AGENT_MAX_STEPS,
            # since a run can burn its time budget on slow model calls or
            # 429/503 retries well before hitting the step cap (or vice
            # versa: a fast multi-step run should never be cut off early
            # just because it used many steps quickly).
            forced_final = True
            tools = FINAL_ANSWER_ONLY_TOOLS
            messages.append({
                "role": "user",
                "content": "Time is up. Call final_answer NOW with your best answer so far.",
            })
            log_event(run_id, chat_id, "time_budget_exceeded", step=step)
        try:
            msg = await llm_chat(messages, tools=tools)
        except Exception as exc:
            log_event(run_id, chat_id, "llm_error", step=step, error=str(exc))
            return None
        messages.append(msg)
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            log_event(run_id, chat_id, "model_text_without_tool_call",
                      step=step, content=msg.get("content"))
            break

        final = _MISSING
        for call in tool_calls:
            name = call["function"]["name"]
            try:
                args = json.loads(call["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}

            if name == "final_answer":
                final = extract_final_answer(args.get("answer_json", ""))
                log_event(run_id, chat_id, "final_answer", step=step,
                          answer=final, reasoning=args.get("reasoning"))
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": "recorded"})
                continue

            if name == "run_python":
                code = args.get("code", "")
                log_event(run_id, chat_id, "tool_call", step=step, tool="run_python", code=code)
                loop = asyncio.get_running_loop()
                output, ok, state = await loop.run_in_executor(
                    None, sandbox.run_python, code, state, AGENT_STEP_TIMEOUT
                )
                log_event(run_id, chat_id, "tool_result", step=step, tool="run_python",
                          ok=ok, output=output[:4000])
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": output[:8000]})
                continue

            messages.append({"role": "tool", "tool_call_id": call["id"],
                             "content": f"unknown tool {name!r}"})

        if final is not _MISSING:
            return final

    log_event(run_id, chat_id, "step_budget_exhausted")
    return None


_chat_locks: dict[int, asyncio.Lock] = {}


async def handle_message_serialized(chat_id: int, user_id: int, text: str) -> None:
    """Runs handle_message() for this chat one at a time. Without this,
    a short burst of messages to the same chat (exactly the multi-turn case
    the assignment describes) get dispatched as independent concurrent
    tasks — observed live: a reply to an earlier, ambiguous message
    finished (and was sent) *after* the reply to a later, real question
    had already gone out, so the last message Telegram shows in the chat
    was the earlier one's non-answer, not the actual answer. Serializing
    per chat_id guarantees replies go out in the same order the questions
    arrived, and that each message's history() snapshot includes every
    prior turn fully completed — not a partial, racing one. Different
    chats still run fully concurrently; only same-chat delivery order
    interacts with the lock, which is the whole point."""
    lock = _chat_locks.setdefault(chat_id, asyncio.Lock())
    async with lock:
        await handle_message(chat_id, user_id, text)


async def handle_message(chat_id: int, user_id: int, text: str) -> None:
    run_id = uuid.uuid4().hex
    log_event(run_id, chat_id, "incoming", user_id=user_id, text=text)
    remember(chat_id, "user", text)

    try:
        answer = await run_agent(chat_id, run_id)
    except Exception as exc:  # never let a bug swallow the reply
        log_event(run_id, chat_id, "agent_crashed", error=repr(exc))
        answer = None

    log_url = f"{PUBLIC_BASE_URL}/logs/run.jsonl" if PUBLIC_BASE_URL else "log_url_not_configured"
    reply_obj = {"answer": answer, "log_url": log_url}
    reply_text = json.dumps(reply_obj, ensure_ascii=False)
    log_event(run_id, chat_id, "reply", reply=reply_obj)
    remember(chat_id, "assistant", reply_text)

    try:
        await send_message(chat_id, reply_text)
    except Exception as exc:
        log_event(run_id, chat_id, "send_failed", error=repr(exc))


# --------------------------------------------------------------------------
# Long-polling: the bot reaches out to Telegram instead of Telegram reaching
# in, so message delivery needs no domain, TLS cert, or inbound firewall
# rule at all. getUpdates' offset is the dedupe mechanism Telegram itself
# uses (an update below the offset you last sent is never redelivered), so
# is_new_update()'s SQLite table is a defense-in-depth backstop rather than
# the primary guard here — it only matters if this process crashes and
# restarts with a stale in-memory offset while an already-seen update is
# still in Telegram's queue.
# --------------------------------------------------------------------------

async def poll_updates_once(client: httpx.AsyncClient, offset: int) -> int:
    """Fetch one batch via long-poll and dispatch each message. Returns the
    offset to pass on the next call."""
    resp = await client.get(f"{TELEGRAM_API}/getUpdates",
                            params={"offset": offset, "timeout": 50})
    data = resp.json()
    if not data.get("ok"):
        raise RuntimeError(f"getUpdates failed: {data}")

    for update in data.get("result", []):
        offset = update["update_id"] + 1
        message = update.get("message") or update.get("edited_message")
        if not message or "text" not in message or not is_new_update(update.get("update_id")):
            continue

        chat_id = message["chat"]["id"]
        user_id = message.get("from", {}).get("id")
        text = message["text"]

        if ALLOWED_USER_IDS and user_id not in ALLOWED_USER_IDS:
            log_event("blocked", chat_id, "blocked_sender", user_id=user_id)
            continue

        asyncio.create_task(handle_message_serialized(chat_id, user_id, text))
    return offset


async def poll_loop() -> None:
    log.info("starting Telegram long-poll loop")
    try:
        await tg_call("deleteWebhook", drop_pending_updates=False)
    except Exception as exc:
        log.warning("deleteWebhook failed (harmless if none was set): %s", exc)

    offset = 0
    async with httpx.AsyncClient(timeout=65) as client:
        while True:
            try:
                offset = await poll_updates_once(client, offset)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("poll loop error, retrying in 5s: %s", exc)
                await asyncio.sleep(5)


# --------------------------------------------------------------------------
# HTTP app — now only serves the public log (and health/debug endpoints).
# Telegram delivery goes through poll_loop() above, not an HTTP route.
# --------------------------------------------------------------------------

app = FastAPI(title="P1 Q5 Data Analyst Telegram Bot")


@app.on_event("startup")
async def _startup() -> None:
    init_db()
    if BOT_TOKEN:
        app.state.poll_task = asyncio.create_task(poll_loop())
    else:
        log.warning("TELEGRAM_BOT_TOKEN not set — polling not started")


@app.on_event("shutdown")
async def _shutdown() -> None:
    task = getattr(app.state, "poll_task", None)
    if task:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


@app.get("/")
def health() -> dict:
    return {
        "status": "ok",
        "service": "p1-q5-telegram-data-analyst-bot",
        "llm_configured": bool(llm_key()),
        "polling_active": bool(BOT_TOKEN),
        "log_url_configured": bool(PUBLIC_BASE_URL),
    }


@app.get("/logs/run.jsonl")
def get_log():
    if not os.path.exists(LOG_PATH):
        return PlainTextResponse("", media_type="application/x-ndjson")
    return FileResponse(LOG_PATH, media_type="application/x-ndjson", filename="run.jsonl")


@app.get("/telegram/debug")
async def telegram_debug() -> Any:
    """Should show no webhook URL once polling has taken over — useful to
    confirm deleteWebhook actually ran if messages aren't arriving."""
    return await tg_call("getWebhookInfo")
