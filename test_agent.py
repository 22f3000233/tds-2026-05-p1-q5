"""Offline tests — no real Telegram account and no real LLM key needed.

Covers the parts that don't need the network: the sandboxed run_python tool
(execution, state persistence, the security hardening), the reply's JSON
shape, and the long-polling update handler's plumbing (dedupe, allowlist)
with the agent loop and Telegram calls stubbed out.

Run: python test_agent.py
"""
import asyncio
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sandbox  # noqa: E402


def test_sandbox_runs_and_persists_state():
    out, ok, state = sandbox.run_python("x = 1 + 1\nprint(x)", {})
    assert ok, out
    assert out.strip() == "2"
    assert state["x"] == 2

    out2, ok2, state2 = sandbox.run_python("print(x * 10)", state)
    assert ok2, out2
    assert out2.strip() == "20"
    print("test_sandbox_runs_and_persists_state: OK")


def test_sandbox_imported_module_persists_across_calls():
    # Modules aren't picklable, so a naive state-persistence scheme drops
    # them silently — regression test for the NameError this caused live
    # (see run.jsonl from the first deployed run: 'import os' in one
    # run_python call, then NameError: name 'os' is not defined two calls
    # later, because the module object itself can't cross the process
    # boundary that sandbox.run_python uses for each call).
    out, ok, state = sandbox.run_python("import os\nprint(os.getcwd() != '')", {})
    assert ok, out
    assert out.strip() == "True"

    out2, ok2, state2 = sandbox.run_python("print(os.path.join('a', 'b'))", state)
    assert ok2, out2
    assert out2.strip() == "a/b"
    print("test_sandbox_imported_module_persists_across_calls: OK")


def test_sandbox_reports_errors_without_crashing():
    out, ok, _ = sandbox.run_python("1 / 0", {})
    assert not ok
    assert "ZeroDivisionError" in out
    print("test_sandbox_reports_errors_without_crashing: OK")


def test_sandbox_times_out():
    out, ok, _ = sandbox.run_python("while True: pass", {}, timeout=2)
    assert not ok
    assert "timed out" in out
    print("test_sandbox_times_out: OK")


def test_ssrf_guard_blocks_local_and_metadata_targets():
    for url in ("http://127.0.0.1/", "http://localhost/", "http://169.254.169.254/latest",
               "http://10.0.0.5/", "ftp://example.com/file"):
        try:
            sandbox._check_public_host(url)
        except ValueError:
            continue
        raise AssertionError(f"expected {url} to be rejected")
    # A real public host should pass the check (no network call is made here).
    sandbox._check_public_host("https://example.com/data.csv")
    print("test_ssrf_guard_blocks_local_and_metadata_targets: OK")


def test_sandbox_scrubs_environment_variables():
    # multiprocessing's spawn start method otherwise hands the child the
    # full parent environment - regression test for a stray
    # `import os; print(os.environ)` being able to read back the bot's own
    # Telegram/LLM tokens (found while comparing against a colleague's
    # implementation that had this exact gap, then confirmed it was true
    # of this sandbox too before _harden_child_process() was added).
    os.environ["P1Q5_TEST_SECRET_SHOULD_NOT_LEAK"] = "super-secret-value"
    try:
        out, ok, _ = sandbox.run_python("import os\nprint(dict(os.environ))", {})
    finally:
        os.environ.pop("P1Q5_TEST_SECRET_SHOULD_NOT_LEAK", None)
    assert ok, out
    assert "super-secret-value" not in out
    assert out.strip() == "{}"
    print("test_sandbox_scrubs_environment_variables: OK")


def test_sandbox_blocks_raw_network_and_process_imports():
    for module in ("socket", "requests", "subprocess", "ctypes"):
        out, ok, _ = sandbox.run_python(f"import {module}", {})
        assert not ok, f"expected importing {module} to be blocked"
        assert "disabled in this sandbox" in out, out
    print("test_sandbox_blocks_raw_network_and_process_imports: OK")


def test_sandbox_disables_os_system():
    out, ok, _ = sandbox.run_python("import os\nos.system('echo pwned')", {})
    assert not ok, out
    assert "PermissionError" in out
    # harmless os.* attributes must still work for pandas/numpy's own use
    out2, ok2, _ = sandbox.run_python("import os\nprint(os.getcwd() != '')", {})
    assert ok2, out2
    assert out2.strip() == "True"
    print("test_sandbox_disables_os_system: OK")


def test_sandbox_dataframe_roundtrip():
    code = (
        "import pandas as pd\n"
        "df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})\n"
        "print(int(df['a'].sum()))\n"
    )
    out, ok, state = sandbox.run_python(code, {})
    assert ok, out
    assert out.strip() == "6"
    assert "df" in state and list(state["df"].columns) == ["a", "b"]
    print("test_sandbox_dataframe_roundtrip: OK")


def _fresh_main(tmp_path, monkeypatch):
    """Import main.py against a scratch DB/log so tests don't touch the
    real bot_agent.db / run.jsonl next to the source files."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://example-host.test")
    monkeypatch.setenv("BOT_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("RUN_LOG_PATH", str(tmp_path / "run.jsonl"))
    sys.modules.pop("main", None)
    import main  # noqa: PLC0415
    # main.py's load_dotenv() just re-read the real .env next to the source
    # files during that import — undo it so tests never depend on whatever
    # real secrets happen to be sitting in the developer's local .env.
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "LLM_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    main.init_db()
    return main


def test_reply_shape_and_dedupe(tmp_path, monkeypatch):
    main = _fresh_main(tmp_path, monkeypatch)

    async def fake_run_agent(chat_id, run_id):
        return {"state": "Assam"}

    sent = []

    async def fake_send_message(chat_id, text):
        sent.append((chat_id, text))

    monkeypatch.setattr(main, "run_agent", fake_run_agent)
    monkeypatch.setattr(main, "send_message", fake_send_message)

    asyncio.run(main.handle_message(123, 456, "Which state has the highest MMR?"))

    assert len(sent) == 1
    chat_id, text = sent[0]
    assert chat_id == 123
    parsed = json.loads(text)
    assert set(parsed.keys()) == {"answer", "log_url"}
    assert parsed["answer"] == {"state": "Assam"}
    assert parsed["log_url"] == "https://example-host.test/logs/run.jsonl"

    hist = main.history(123, 10)
    assert hist[0] == {"role": "user", "content": "Which state has the highest MMR?"}
    assert hist[1]["role"] == "assistant"

    assert main.is_new_update(999) is True
    assert main.is_new_update(999) is False
    print("test_reply_shape_and_dedupe: OK")


def test_no_llm_key_still_replies_valid_json(tmp_path, monkeypatch):
    main = _fresh_main(tmp_path, monkeypatch)  # clears LLM key env vars itself
    assert main.llm_key() == ""

    sent = []

    async def fake_send_message(chat_id, text):
        sent.append(text)

    monkeypatch.setattr(main, "send_message", fake_send_message)
    asyncio.run(main.handle_message(1, 2, "What is 2+2?"))
    parsed = json.loads(sent[0])
    assert parsed["answer"] is None
    assert parsed["log_url"].endswith("/logs/run.jsonl")
    print("test_no_llm_key_still_replies_valid_json: OK")


def test_extract_final_answer_unwraps_double_wrapped_shape(tmp_path, monkeypatch):
    # Regression test for a real failure seen live: the model had a real
    # log_url sitting in its own chat history (from an earlier reply) and
    # copied the WHOLE {"answer": ..., "log_url": ...} template into
    # answer_json instead of just the inner value. Left unhandled, that
    # produces a doubly-nested, wrong-shaped final reply to the grader
    # despite the underlying computed answer being correct.
    main = _fresh_main(tmp_path, monkeypatch)

    wrapped = json.dumps({"answer": {"species": "versicolor"}, "log_url": "https://example.test/logs/run.jsonl"})
    assert main.extract_final_answer(wrapped) == {"species": "versicolor"}

    # A normal, correctly-shaped answer_json must pass through unchanged.
    assert main.extract_final_answer('{"species": "versicolor"}') == {"species": "versicolor"}
    assert main.extract_final_answer("42") == 42
    assert main.extract_final_answer('"Assam"') == "Assam"

    # An unquoted plain string (model forgot to JSON-quote it) is used as-is
    # rather than raising, since json.loads("Assam") is invalid JSON.
    assert main.extract_final_answer("Assam") == "Assam"
    print("test_extract_final_answer_unwraps_double_wrapped_shape: OK")


def test_messages_for_same_chat_are_serialized(tmp_path, monkeypatch):
    # Regression test for a real ordering bug seen live: three near-
    # simultaneous messages to one chat were dispatched as independent
    # concurrent tasks, so a reply to an earlier, ambiguous message could
    # finish (and be sent) AFTER the reply to a later, real question -
    # meaning the last message Telegram showed in the chat was a non-answer
    # even though the real question had already been correctly answered.
    main = _fresh_main(tmp_path, monkeypatch)
    events = []

    async def fake_handle_message(chat_id, user_id, text):
        events.append(("start", chat_id, text))
        # The first message deliberately takes longer, so a race would
        # show up as the second message's ("start", ...)/("end", ...) pair
        # interleaving with the first's instead of coming after it.
        await asyncio.sleep(0.05 if text == "slow" else 0)
        events.append(("end", chat_id, text))

    monkeypatch.setattr(main, "handle_message", fake_handle_message)

    async def run():
        await asyncio.gather(
            main.handle_message_serialized(1, 10, "slow"),
            main.handle_message_serialized(1, 10, "fast"),
            main.handle_message_serialized(2, 20, "other-chat"),
        )

    asyncio.run(run())

    chat1_events = [e for e in events if e[1] == 1]
    assert chat1_events == [
        ("start", 1, "slow"), ("end", 1, "slow"),
        ("start", 1, "fast"), ("end", 1, "fast"),
    ], chat1_events

    # A different chat must not wait behind chat 1's slow message.
    other_start_index = events.index(("start", 2, "other-chat"))
    chat1_slow_end_index = events.index(("end", 1, "slow"))
    assert other_start_index < chat1_slow_end_index
    print("test_messages_for_same_chat_are_serialized: OK")


def test_health_and_log_endpoints(tmp_path, monkeypatch):
    main = _fresh_main(tmp_path, monkeypatch)
    from fastapi.testclient import TestClient

    # _startup() schedules poll_loop() as a background task, which would
    # otherwise make real HTTP calls to api.telegram.org with a fake token
    # for the duration of the TestClient context - poll_loop's own network
    # behavior is covered separately (test_poll_updates_*), so stub it here
    # to keep this test fully offline.
    async def fake_poll_loop():
        pass

    monkeypatch.setattr(main, "poll_loop", fake_poll_loop)

    with TestClient(main.app) as client:
        resp = client.get("/logs/run.jsonl")
        assert resp.status_code == 200

        resp = client.get("/")
        body = resp.json()
        assert body["status"] == "ok"
        assert body["polling_active"] is True
    print("test_health_and_log_endpoints: OK")


class _FakeTelegramClient:
    def __init__(self, payload):
        self._payload = payload

    async def get(self, url, params=None):
        class _Resp:
            def __init__(self_inner, payload):
                self_inner._payload = payload

            def json(self_inner):
                return self_inner._payload

        return _Resp(self._payload)


def test_poll_updates_dispatches_and_dedupes(tmp_path, monkeypatch):
    main = _fresh_main(tmp_path, monkeypatch)
    calls = []

    async def fake_handle_message(chat_id, user_id, text):
        calls.append((chat_id, user_id, text))

    monkeypatch.setattr(main, "handle_message", fake_handle_message)

    payload = {"ok": True, "result": [
        {"update_id": 10, "message": {"chat": {"id": 1}, "from": {"id": 2}, "text": "hi"}},
    ]}
    client = _FakeTelegramClient(payload)

    async def run_once():
        offset = await main.poll_updates_once(client, 0)
        await asyncio.sleep(0)  # let the dispatched task actually run
        return offset

    offset = asyncio.run(run_once())
    assert offset == 11
    assert calls == [(1, 2, "hi")]

    # Telegram re-delivering the same update_id (e.g. after a restart with
    # a stale offset) must not dispatch it a second time.
    calls.clear()
    asyncio.run(run_once())
    assert calls == []
    print("test_poll_updates_dispatches_and_dedupes: OK")


def test_poll_updates_allowlist_blocks_unknown_senders(tmp_path, monkeypatch):
    monkeypatch.setenv("ALLOWED_TELEGRAM_USER_IDS", "42")
    main = _fresh_main(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(main, "handle_message", lambda *a: calls.append(a))

    payload = {"ok": True, "result": [
        {"update_id": 5, "message": {"chat": {"id": 1}, "from": {"id": 999}, "text": "hi"}},
    ]}
    client = _FakeTelegramClient(payload)

    async def run_once():
        offset = await main.poll_updates_once(client, 0)
        await asyncio.sleep(0)
        return offset

    offset = asyncio.run(run_once())
    assert offset == 6
    assert calls == []
    print("test_poll_updates_allowlist_blocks_unknown_senders: OK")


if __name__ == "__main__":
    import types

    test_sandbox_runs_and_persists_state()
    test_sandbox_imported_module_persists_across_calls()
    test_sandbox_reports_errors_without_crashing()
    test_sandbox_times_out()
    test_ssrf_guard_blocks_local_and_metadata_targets()
    test_sandbox_scrubs_environment_variables()
    test_sandbox_blocks_raw_network_and_process_imports()
    test_sandbox_disables_os_system()
    test_sandbox_dataframe_roundtrip()

    class _Monkeypatch:
        """Tiny stand-in for pytest's monkeypatch fixture so this file also
        runs with plain `python test_agent.py`, no pytest required."""

        def __init__(self):
            self._sets: list[tuple[dict, str, object, bool]] = []

        def setenv(self, key, value):
            self._sets.append((os.environ, key, os.environ.get(key), key in os.environ))
            os.environ[key] = value

        def delenv(self, key, raising=False):
            had = key in os.environ
            self._sets.append((os.environ, key, os.environ.get(key), had))
            os.environ.pop(key, None)

        def setattr(self, obj, name, value):
            old = getattr(obj, name)
            self._sets.append((obj, name, old, True))
            setattr(obj, name, value)

        def undo(self):
            for target, key, old, had in reversed(self._sets):
                if isinstance(target, dict):
                    if had:
                        target[key] = old
                    else:
                        target.pop(key, None)
                else:
                    setattr(target, key, old)
            self._sets.clear()

    for fn in (test_reply_shape_and_dedupe, test_no_llm_key_still_replies_valid_json,
              test_extract_final_answer_unwraps_double_wrapped_shape,
              test_messages_for_same_chat_are_serialized,
              test_health_and_log_endpoints,
              test_poll_updates_dispatches_and_dedupes,
              test_poll_updates_allowlist_blocks_unknown_senders):
        with tempfile.TemporaryDirectory() as tmp:
            import pathlib
            mp = _Monkeypatch()
            try:
                fn(pathlib.Path(tmp), mp)
            finally:
                mp.undo()

    print("ALL TESTS PASSED")
