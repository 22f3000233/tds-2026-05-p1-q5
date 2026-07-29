"""Sandboxed Python execution for the data-analyst agent's `run_python` tool.

Code the model writes runs in its own child process (`multiprocessing`,
spawned fresh each call) so a runaway loop or crash can be killed on a
timeout instead of taking the whole bot down. Variables the model sets
persist across calls within one conversation turn: the picklable subset of
the exec() globals is sent back to the parent and re-seeded into the next
call's globals.

`fetch`/`fetch_df` are the intended way sandboxed code reaches the network.
Both resolve the host and refuse anything that isn't a public address
(blocks localhost, private ranges, link-local, and the cloud metadata IP)
— this bot is reachable by anyone who messages it on Telegram, not just
the grader, so "fetch a URL" must not become "probe my own host's
network". Redirects are followed by hand so every hop gets the same check.

Before running the model's code, `_harden_child_process()` also:
  - wipes this process's environment variables, since `multiprocessing`'s
    spawn start method otherwise hands the child the full parent
    environment — including the bot's Telegram/LLM tokens — for a plain
    `import os; print(os.environ)` to read straight back out;
  - blocks importing raw network modules (`socket`, `requests`, `urllib`,
    etc.) and process-execution modules (`subprocess`, `ctypes`, ...), so
    the model can't bypass the fetch()/fetch_df() guard above by just not
    using it;
  - disables os.system/popen/exec*/spawn*/fork on the *already-loaded*
    `os` module object, rather than blocking `import os` outright (pandas
    and numpy use harmless os.path/os.getcwd internally and blocking the
    whole module risks breaking them unpredictably).

Be clear-eyed about what this is and isn't: it meaningfully raises the bar
against casual or accidental abuse — the obvious `import socket` /
`os.system(...)` attempts fail with a clear error — but this is defense in
depth, not a hardened, escape-proof sandbox. Pure-Python import/attribute
restrictions are well known to be incomplete against a sufficiently
determined attacker with Python-internals knowledge. Fully closing that
gap needs OS-level isolation (a separate unprivileged user with outbound
network rules restricting that user specifically) — infrastructure work
outside what a Python-only fix can guarantee, and isn't set up here.
"""
from __future__ import annotations

import contextlib
import importlib
import io
import ipaddress
import multiprocessing
import os
import pickle
import socket
import sys
import traceback
import types
import urllib.parse

import numpy as np
import pandas as pd
import requests

MAX_FETCH_BYTES = 5_000_000
FETCH_TIMEOUT = 20
MAX_REDIRECTS = 5


def _ip_is_forbidden(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return (
        ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
        or ip.is_multicast or ip.is_unspecified
        or (ip.version == 4 and ip in ipaddress.ip_network("100.64.0.0/10"))
    )


def _check_public_host(url: str) -> None:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("only http/https URLs are allowed")
    if parsed.username or parsed.password:
        raise ValueError("URLs with embedded credentials are not allowed")
    host = (parsed.hostname or "").rstrip(".")
    if not host:
        raise ValueError("URL has no host")
    try:
        infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror as exc:
        raise ValueError(f"host does not resolve: {exc}") from None
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if _ip_is_forbidden(ip):
            raise ValueError(f"host resolves to a non-public address ({ip}); refusing to fetch it")


def _fetch_bytes(url: str, **kwargs) -> bytes:
    """Guarded GET, downloaded ourselves (not handed to pandas/openpyxl to
    fetch) so every redirect hop is re-validated against the same guard."""
    current = url
    for _ in range(MAX_REDIRECTS):
        _check_public_host(current)
        resp = requests.get(current, timeout=FETCH_TIMEOUT, allow_redirects=False, **kwargs)
        if resp.status_code in (301, 302, 303, 307, 308) and resp.headers.get("location"):
            current = urllib.parse.urljoin(current, resp.headers["location"])
            continue
        resp.raise_for_status()
        return resp.content[:MAX_FETCH_BYTES]
    raise ValueError("too many redirects")


def fetch(url: str, **kwargs) -> str:
    """Download a public URL as text (truncated to a few MB)."""
    return _fetch_bytes(url, **kwargs).decode("utf-8", errors="replace")


def fetch_df(url: str, **kwargs):
    """Download a public URL and parse it into a DataFrame (or a list of
    them for multi-table HTML pages). Picks the parser from the URL's file
    extension, defaulting to CSV."""
    raw = _fetch_bytes(url)
    buf = io.BytesIO(raw)
    lower = url.lower().split("?")[0]
    if lower.endswith((".xlsx", ".xls")):
        return pd.read_excel(buf, **kwargs)
    if lower.endswith(".json"):
        return pd.read_json(buf, **kwargs)
    if lower.endswith((".html", ".htm")):
        tables = pd.read_html(io.StringIO(raw.decode("utf-8", errors="replace")), **kwargs)
        return tables[0] if len(tables) == 1 else tables
    if lower.endswith(".parquet"):
        return pd.read_parquet(buf, **kwargs)
    return pd.read_csv(buf, **kwargs)


_SANDBOX_GLOBALS = {"pd": pd, "np": np, "fetch": fetch, "fetch_df": fetch_df}

# Modules that provide raw network access (bypassing the fetch()/fetch_df()
# host guard) or process/shell execution. Blocked for freshly-imported
# names; see _harden_child_process() for why already-cached ones need
# evicting first, and the module docstring for what this does not cover.
_BLOCKED_MODULES = {
    "socket", "ssl", "http", "urllib", "requests", "httpx", "aiohttp",
    "ftplib", "smtplib", "telnetlib", "poplib", "imaplib", "socketserver",
    "subprocess", "ctypes", "pty", "multiprocessing",
}

# os.* functions that start a process, run a shell command, or send a
# signal — everything else on os (path, getcwd, environ, ...) is left
# alone since pandas/numpy use it internally.
_DANGEROUS_OS_ATTRS = (
    "system", "popen", "popen2", "popen3", "popen4",
    "exec", "execl", "execle", "execlp", "execlpe", "execv", "execve",
    "execvp", "execvpe", "spawnl", "spawnle", "spawnlp", "spawnlpe",
    "spawnv", "spawnve", "spawnvp", "spawnvpe",
    "fork", "forkpty", "kill", "killpg", "posix_spawn", "posix_spawnp",
)


class _BlockedImport:
    """Meta-path finder installed at the front of sys.meta_path in each
    sandboxed child: turns `import <blocked>` into a clear ImportError
    instead of silently succeeding."""

    def find_spec(self, fullname, path, target=None):
        if fullname.split(".")[0] in _BLOCKED_MODULES:
            raise ImportError(
                f"{fullname!r} is disabled in this sandbox for security; "
                f"use fetch()/fetch_df() for network access instead"
            )
        return None


def _harden_child_process() -> None:
    """Runs once at the top of _worker, before any model-written code. See
    the module docstring for what this does and does not close off."""
    os.environ.clear()

    # sys.modules already has socket/ssl/http.client/etc. cached from our
    # own `import requests` above — without evicting them, `import socket`
    # in user code would be served straight from cache, never reaching the
    # meta-path finder below.
    for name in list(sys.modules):
        if name.split(".")[0] in _BLOCKED_MODULES:
            sys.modules.pop(name, None)
    sys.meta_path.insert(0, _BlockedImport())

    def _disabled(*_args, **_kwargs):
        raise PermissionError("disabled in this sandbox for security")

    for attr in _DANGEROUS_OS_ATTRS:
        if hasattr(os, attr):
            try:
                setattr(os, attr, _disabled)
            except AttributeError:
                pass


class _ModuleRef:
    """Stands in for an imported module across the process boundary — a
    live module object isn't picklable, so `import os` in one run_python
    call would otherwise silently vanish by the next call, leaving the
    model's own code raising NameError on a name it thinks it already
    has. Re-imported by name when state is loaded back into a worker."""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name


def _picklable(value) -> bool:
    try:
        pickle.dumps(value)
        return True
    except Exception:
        return False


def _worker(code: str, state: dict, conn) -> None:
    _harden_child_process()
    g = dict(_SANDBOX_GLOBALS)
    for key, value in state.items():
        g[key] = importlib.import_module(value.name) if isinstance(value, _ModuleRef) else value
    out = io.StringIO()
    error = None
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            exec(compile(code, "<run_python>", "exec"), g)
    except Exception:
        error = traceback.format_exc(limit=6)
    new_state = {}
    for key, value in g.items():
        if key in _SANDBOX_GLOBALS or key.startswith("__"):
            continue
        if isinstance(value, types.ModuleType):
            new_state[key] = _ModuleRef(value.__name__)
        elif _picklable(value):
            new_state[key] = value
    conn.send((out.getvalue()[-20_000:], error, new_state))
    conn.close()


def run_python(code: str, state: dict, timeout: int = 20) -> tuple[str, bool, dict]:
    """Execute `code` in a fresh child process seeded with `state` from the
    previous call. Returns (output_or_error, ok, new_state). A runaway
    process is killed after `timeout` seconds."""
    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()
    proc = ctx.Process(target=_worker, args=(code, state, child_conn), daemon=True)
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join(2)
        return f"execution timed out after {timeout}s", False, state
    try:
        stdout, error, new_state = parent_conn.recv()
    except EOFError:
        return "the sandbox process crashed without producing output", False, state
    merged_state = {**state, **new_state}
    if error:
        text = (stdout + "\n" if stdout else "") + error
        return text, False, merged_state
    return stdout or "(no output; use print(...) to see a value)", True, merged_state
