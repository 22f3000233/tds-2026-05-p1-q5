# Data Analyst Telegram Bot

A Telegram bot that answers data-analysis questions. Message it a question — with data inline or a link to a public dataset — and it replies with a single JSON object containing the answer and a link to its run log.

```json
{"answer": <shaped as your question asks>, "log_url": "https://your-host/logs/run.jsonl"}
```

## How it works

The bot hands each question to an LLM (Google Gemini) with two tools:

- **`run_python`** — executes Python in a sandboxed subprocess, with `pandas`/`numpy` preloaded and `fetch()`/`fetch_df()` helpers for downloading public datasets (CSV, JSON, Excel, HTML tables, Parquet).
- **`final_answer`** — submits the final answer once the model is confident, shaped exactly as the question's own JSON template asks for.

The model can call `run_python` as many times as it needs (download data, inspect it, compute, verify) before calling `final_answer`. Every step — the incoming message, code executed, its output, and the final answer — is logged as JSONL and served publicly so the reasoning behind any answer can be audited.

Messages are delivered via long-polling (the bot reaches out to Telegram, rather than requiring an inbound webhook), and chat history is persisted so multi-turn conversations (a short sequence of messages building up to one question) retain context across turns.

## Project layout

```
main.py             FastAPI app: Telegram long-polling loop, the LLM tool-calling
                     loop, chat history, run logging, and the log-serving endpoint
sandbox.py           Sandboxed Python execution for the run_python tool
test_agent.py         Offline test suite (no live Telegram/LLM key required)
requirements.txt      Python dependencies
.env.example          Configuration template
```

## Setup

```bash
git clone <this-repo-url>
cd <this-repo>
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env`:

| Variable | Description |
|---|---|
| `TELEGRAM_BOT_TOKEN` | From [@BotFather](https://t.me/BotFather) on Telegram. |
| `GEMINI_API_KEY` | A key from [Google AI Studio](https://aistudio.google.com/apikey). The underlying Google Cloud project needs a billing account linked — projects without one get a zero free-tier quota. |
| `PUBLIC_BASE_URL` | The public HTTPS URL this app is reachable at once deployed. Used to build `log_url` in replies. |
| `LLM_MODELS` | Comma-separated fallback list of Gemini models, tried in order. |
| `ALLOWED_TELEGRAM_USER_IDS` | Optional comma-separated Telegram user IDs allowed to message the bot. Leave empty during testing. |

See `.env.example` for the full list, including tuning options for the agent loop's step and time budgets.

## Running

```bash
uvicorn main:app --host 0.0.0.0 --port 8009
```

No public URL, tunnel, or TLS setup is required just to message the bot locally — it reaches out to Telegram itself. `PUBLIC_BASE_URL` only matters once you deploy, since the run log at `/logs/run.jsonl` needs to be publicly reachable.

## Testing

```bash
python test_agent.py
```

Runs entirely offline — no live bot or LLM key needed. Covers the sandboxed execution, the tool-calling loop's core logic, and the message-handling plumbing.

## Deployment

Requires an always-on host with a public HTTPS URL (not a serverless function that sleeps), since the run log must stay reachable at any time. Use a persistent disk/volume for the SQLite chat history and run log if your host's filesystem is otherwise ephemeral across restarts. Run exactly one process — Telegram's long-polling isn't designed for multiple concurrent pollers on the same bot token, and the log/history files aren't safe for concurrent writers.

## Security

This bot executes model-generated code, and is reachable by anyone who messages it on Telegram — not only its intended user. `run_python` runs in an isolated, time-limited subprocess with a scrubbed environment and restricted network/process access. These measures reduce the attack surface substantially but don't make arbitrary code execution fully safe in the way a purpose-built sandbox (gVisor, a VM, etc.) would.

Set `ALLOWED_TELEGRAM_USER_IDS` once you know who should legitimately be using the bot — every other sender is then silently ignored. You can read a sender's numeric Telegram ID out of the run log (each `incoming` entry records `user_id`) after they've messaged the bot once, or look it up via [@userinfobot](https://t.me/userinfobot).
