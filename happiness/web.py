"""The web chat: one page, a streaming chat endpoint and file uploads, on the standard library alone.

Put it behind HTTPS (a reverse proxy such as Caddy, or your host's TLS) when it is reachable from
the internet: the password travels with every request.
"""

from __future__ import annotations

import base64
import hmac
import json
import logging
import re
import secrets
import sys
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlsplit

import anthropic

from .agent import EFFORTS, Agent, Settings, describe_error

log = logging.getLogger("happiness.web")

MAX_CONVERSATIONS = 32
MAX_REQUEST = 256 * 1024  # bytes of JSON in one chat message
MAX_UPLOAD = 25 * 1024 * 1024
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


class Conversations:
    """The open conversations, by id. When there are too many, the least recently used is forgotten."""

    def __init__(self, make_agent: Callable[[], Agent], limit: int = MAX_CONVERSATIONS):
        self._make = make_agent
        self._limit = limit
        self._items: OrderedDict[str, tuple[Agent, threading.Lock]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, conversation: str) -> tuple[Agent, threading.Lock]:
        with self._lock:
            if conversation in self._items:
                self._items.move_to_end(conversation)
            else:
                self._items[conversation] = (self._make(), threading.Lock())
                while len(self._items) > self._limit:
                    self._items.popitem(last=False)
            return self._items[conversation]


def safe_filename(name: str) -> str:
    name = Path(name.replace("\\", "/")).name
    name = re.sub(r"[^\w.\- ]+", "_", name).strip(" .")
    return name[:120]


def unused_path(path: Path) -> Path:
    candidate, n = path, 1
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}-{n}{path.suffix}")
        n += 1
    return candidate


class Handler(BaseHTTPRequestHandler):
    server: HappinessServer
    server_version = "Happiness"

    def log_message(self, format: str, *args: Any) -> None:
        log.info("%s %s", self.address_string(), format % args)

    # ---------------------------------------------------------------- routing

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/healthz":
            return self._send(200, b"ok\n", "text/plain")
        if not self._authorized():
            return
        if path in ("/", "/index.html"):
            return self._page()
        self._send(404, b"not found\n", "text/plain")

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        if not self._authorized():
            return
        if self.headers.get("X-Happiness") != "1":
            # Only the page's own scripts send this header; forms and other sites can't.
            return self._json(403, {"error": "missing X-Happiness header"})
        if path == "/api/chat":
            return self._chat()
        if path == "/api/upload":
            return self._upload()
        self._json(404, {"error": "not found"})

    # ---------------------------------------------------------------- helpers

    def _authorized(self) -> bool:
        password = self.server.password
        if not password:
            return True
        scheme, _, token = self.headers.get("Authorization", "").partition(" ")
        if scheme.lower() == "basic":
            try:
                given = base64.b64decode(token, validate=True).decode("utf-8").partition(":")[2]
            except ValueError:
                given = ""
            if hmac.compare_digest(given.encode(), password.encode()):
                return True
            time.sleep(1)  # slow down password guessing
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="Happiness", charset="UTF-8"')
        self.send_header("Content-Length", "0")
        self.end_headers()
        return False

    def _send(self, status: int, body: bytes, content_type: str, headers: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, data: dict[str, Any]) -> None:
        self._send(status, json.dumps(data).encode(), "application/json")

    def _body(self, limit: int) -> bytes | None:
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self._json(411, {"error": "Content-Length is required"})
            return None
        if not 0 <= length <= limit:
            self._json(413, {"error": f"too large (the limit is {limit // (1024 * 1024) or 1} MB)"})
            return None
        return self.rfile.read(length)

    # ---------------------------------------------------------------- endpoints

    def _page(self) -> None:
        nonce = secrets.token_urlsafe(16)
        policy = (
            f"default-src 'none'; script-src 'nonce-{nonce}'; style-src 'unsafe-inline'; connect-src 'self'; "
            "img-src 'self' data:; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
        )
        headers = {"Content-Security-Policy": policy, "Referrer-Policy": "no-referrer", "Cache-Control": "no-store"}
        self._send(200, PAGE.replace("__NONCE__", nonce).encode(), "text/html; charset=utf-8", headers)

    def _chat(self) -> None:
        body = self._body(MAX_REQUEST)
        if body is None:
            return
        try:
            data = json.loads(body)
            conversation, text = str(data["conversation"]), str(data["message"]).strip()
        except (ValueError, KeyError, TypeError):
            return self._json(400, {"error": 'send JSON: {"conversation": "<id>", "message": "<text>"}'})
        if not text or not 0 < len(conversation) <= 100:
            return self._json(400, {"error": "empty message or bad conversation id"})

        agent, busy = self.server.conversations.get(conversation)
        if not busy.acquire(blocking=False):
            return self._json(409, {"error": "Still working on your previous message."})
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")  # tell proxies not to hold the stream back
            self.end_headers()
            listening = True

            def emit(kind: str, payload: str) -> None:
                nonlocal listening
                if not listening:
                    return
                try:
                    self.wfile.write(f"event: {kind}\ndata: {json.dumps(payload)}\n\n".encode())
                    self.wfile.flush()
                except OSError:
                    listening = False  # the browser left; the turn still finishes, so the conversation stays whole

            try:
                agent.ask(text, on_event=emit)
                emit("done", "")
            except Exception as error:
                message = describe_error(error)
                if message is None:
                    log.exception("chat request failed")
                    message = "Something went wrong on the server; its log has the details."
                else:
                    log.warning("%s", message)
                emit("error", message)
        finally:
            busy.release()

    def _upload(self) -> None:
        name = safe_filename(parse_qs(urlsplit(self.path).query).get("name", [""])[0])
        if not name:
            return self._json(400, {"error": "name the file: /api/upload?name=<file name>"})
        body = self._body(MAX_UPLOAD)
        if body is None:
            return
        folder = self.server.settings.workspace_dir / "uploads"
        folder.mkdir(parents=True, exist_ok=True)
        target = unused_path(folder / name)
        target.write_bytes(body)
        self._json(200, {"path": f"uploads/{target.name}"})


class HappinessServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], settings: Settings, password: str | None, client: anthropic.Anthropic):
        super().__init__(address, Handler)
        self.settings = settings
        self.password = password or None
        self.conversations = Conversations(lambda: Agent(settings, client=client))


def make_server(
    settings: Settings, host: str, port: int, *, password: str | None, client: anthropic.Anthropic | None = None
) -> HappinessServer:
    settings.shell = False  # no shell in the browser: nobody would be there to approve the commands
    return HappinessServer((host, port), settings, password, client or anthropic.Anthropic())


def serve(settings: Settings, host: str, port: int, *, password: str | None) -> int:
    app_log = logging.getLogger("happiness")
    if not app_log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        app_log.addHandler(handler)
        app_log.setLevel(logging.INFO)
    if settings.shell:
        log.warning("The shell tool is only available in the terminal; the web chat runs without it.")
    if settings.effort not in EFFORTS:
        print(f"HAPPINESS_EFFORT must be one of {', '.join(EFFORTS)}, not {settings.effort!r}.", file=sys.stderr)
        return 2
    if not password and host not in LOCAL_HOSTS:
        print(f"Refusing to serve on {host} without a password. Set HAPPINESS_PASSWORD.", file=sys.stderr)
        return 2
    server = make_server(settings, host, port, password=password)
    shown = "localhost" if host in ("127.0.0.1", "::1") else host
    log.info("Happiness (%s) is listening on http://%s:%d/", settings.model, shown, server.server_address[1])
    log.info("Workspace: %s", settings.workspace_dir)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Happiness</title>
<style>
  :root {
    --bg: #fbf8f3; --fg: #2b2620; --muted: #857b6e; --line: #e7dfd3; --accent: #c8741f;
    --on-accent: #fff; --bubble: #f1eae0; --code: #efe7da; --warn: #9a5800; --err: #b3261e;
    color-scheme: light dark;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #1b1916; --fg: #ece5da; --muted: #a29785; --line: #35302a; --accent: #f0a04b;
      --on-accent: #1b1916; --bubble: #2a2621; --code: #302a23; --warn: #f2b35a; --err: #ff8a80;
    }
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; margin: 0; }
  body {
    background: var(--bg); color: var(--fg); display: flex; flex-direction: column;
    font: 16px/1.55 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  }
  header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 10px 16px; border-bottom: 1px solid var(--line);
  }
  h1 { font-size: 17px; font-weight: 600; margin: 0; display: flex; align-items: center; gap: 10px; }
  h1::before {
    content: ""; width: 11px; height: 11px; border-radius: 50%; background: var(--accent);
    box-shadow: 0 0 0 4px color-mix(in srgb, var(--accent) 22%, transparent);
  }
  button, .attach {
    font: inherit; font-size: 15px; color: inherit; background: none; cursor: pointer;
    border: 1px solid var(--line); border-radius: 8px; padding: 6px 12px; line-height: 1.4;
  }
  button:hover, .attach:hover { border-color: var(--muted); }
  button:disabled { opacity: .5; cursor: default; }
  #send { background: var(--accent); border-color: var(--accent); color: var(--on-accent); font-weight: 600; }
  main { flex: 1; overflow-y: auto; }
  #log { max-width: 760px; margin: 0 auto; padding: 20px 16px; }
  .welcome { color: var(--muted); }
  .msg { margin: 0 0 20px; overflow-wrap: anywhere; }
  .user { background: var(--bubble); border-radius: 12px; padding: 9px 14px; margin-left: 12%; white-space: pre-wrap; }
  .bot p { margin: 0 0 .6em; }
  .bot ul, .bot ol { margin: 0 0 .6em; padding-left: 1.4em; }
  .bot pre { background: var(--code); padding: 10px 12px; border-radius: 8px; overflow-x: auto; font-size: 14px; }
  .bot code { background: var(--code); padding: 1px 4px; border-radius: 4px; font-size: .92em; }
  .bot pre code { background: none; padding: 0; }
  .bot a { color: var(--accent); }
  .status { color: var(--muted); font-size: 14px; margin: 2px 0 6px; }
  .notice { color: var(--warn); font-size: 14px; margin: 6px 0; }
  .error { color: var(--err); font-size: 14px; margin: 6px 0; }
  .waiting::after { content: " …"; animation: blink 1.2s infinite; }
  @keyframes blink { 50% { opacity: .2; } }
  footer { border-top: 1px solid var(--line); padding: 10px 16px 14px; }
  #composer { max-width: 760px; margin: 0 auto; display: flex; gap: 8px; align-items: flex-end; }
  #chips { max-width: 760px; margin: 0 auto 6px; font-size: 14px; color: var(--muted); }
  #chips:empty { display: none; }
  textarea {
    flex: 1; resize: none; font: inherit; color: inherit; background: var(--bubble); max-height: 40vh;
    border: 1px solid var(--line); border-radius: 10px; padding: 8px 12px;
  }
  textarea:focus { outline: 2px solid color-mix(in srgb, var(--accent) 45%, transparent); outline-offset: 0; }
</style>
</head>
<body>
<header>
  <h1>Happiness</h1>
  <button id="new" type="button">New chat</button>
</header>
<main id="scroller">
  <div id="log">
    <p class="welcome">Hello, I'm Happiness. Ask me to look something up, read or draft something, keep a note,
      or find the right word for a particular kind of joy. Attach files with +.</p>
  </div>
</main>
<footer>
  <div id="chips"></div>
  <form id="composer">
    <label class="attach" title="Attach files">+<input id="file" type="file" multiple hidden></label>
    <textarea id="input" rows="1" placeholder="Message Happiness" autofocus></textarea>
    <button id="send" type="submit">Send</button>
  </form>
</footer>
<script nonce="__NONCE__">
"use strict";
const $ = (id) => document.getElementById(id);
const log = $("log"), scroller = $("scroller"), input = $("input"), send = $("send"), chips = $("chips");
let conversation = newId(), busy = false, uploads = [];

function newId() {
  if (window.crypto && crypto.randomUUID) return crypto.randomUUID();
  return Date.now().toString(36) + Math.random().toString(36).slice(2);
}

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

function nearBottom() { return scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 120; }
function follow(stick) { if (stick) scroller.scrollTop = scroller.scrollHeight; }

// A small, safe Markdown renderer: everything is escaped first, then a few patterns become tags.
function esc(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}
function inline(s) {
  return s.split(/(`[^`\n]+`)/).map((part, i) => i % 2 ? "<code>" + part.slice(1, -1) + "</code>" : part
    .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[\s(])\*([^*\s][^*\n]*?)\*(?=[\s).,;:!?]|$)/g, "$1<em>$2</em>")
    .replace(/\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>')
  ).join("");
}
function blocks(text) {
  let html = "", list = null;
  for (const line of esc(text).split("\n")) {
    const item = line.match(/^\s*([-*•]|\d+[.)])\s+(.*)$/);
    if (item) {
      const kind = /\d/.test(item[1]) ? "ol" : "ul";
      if (list !== kind) { if (list) html += "</" + list + ">"; html += "<" + kind + ">"; list = kind; }
      html += "<li>" + inline(item[2]) + "</li>";
      continue;
    }
    if (list) { html += "</" + list + ">"; list = null; }
    const heading = line.match(/^#{1,6}\s+(.*)$/);
    if (heading) html += "<p><strong>" + inline(heading[1]) + "</strong></p>";
    else if (line.trim()) html += "<p>" + inline(line) + "</p>";
  }
  return list ? html + "</" + list + ">" : html;
}
function markdown(src) {
  return src.split(/^```.*$/m).map((part, i) =>
    i % 2 ? "<pre><code>" + esc(part.replace(/^\n/, "").replace(/\n$/, "")) + "</code></pre>" : blocks(part)
  ).join("");
}

function addUser(text) {
  const welcome = log.querySelector(".welcome");
  if (welcome) welcome.remove();
  log.append(el("div", "msg user", text));
}

function reply() {
  const box = el("div", "msg bot");
  const waiting = el("div", "status waiting", "thinking");
  box.append(waiting);
  log.append(box);
  let segment = null, text = "";
  const settle = () => { if (waiting.isConnected) waiting.remove(); };
  return {
    text(chunk) {
      settle();
      if (!segment) { segment = el("div"); box.append(segment); text = ""; }
      text += chunk;
      segment.innerHTML = markdown(text);
    },
    line(cls, words) { settle(); box.append(el("div", cls, words)); segment = null; },
    done: settle,
  };
}

function handle(r, chunk) {
  let kind = "message", data = "";
  for (const line of chunk.split("\n")) {
    if (line.startsWith("event: ")) kind = line.slice(7);
    else if (line.startsWith("data: ")) data += line.slice(6);
  }
  const payload = data ? JSON.parse(data) : "";
  if (kind === "text") r.text(payload);
  else if (kind === "tool") r.line("status", "· " + payload);
  else if (kind === "notice") r.line("notice", payload);
  else if (kind === "error") r.line("error", payload);
  else if (kind === "done") r.done();
}

async function ask(message) {
  busy = true; send.disabled = true;
  addUser(message);
  const r = reply();
  follow(true);
  try {
    const res = await fetch("api/chat", {
      method: "POST",
      headers: {"Content-Type": "application/json", "X-Happiness": "1"},
      body: JSON.stringify({conversation, message}),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({error: res.status + " " + res.statusText}));
      r.line("error", body.error || "Request failed.");
      return;
    }
    const reader = res.body.getReader(), decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const {value, done} = await reader.read();
      if (done) break;
      const stickNow = nearBottom();
      buffer += decoder.decode(value, {stream: true});
      let cut;
      while ((cut = buffer.indexOf("\n\n")) >= 0) {
        handle(r, buffer.slice(0, cut));
        buffer = buffer.slice(cut + 2);
      }
      follow(stickNow);
    }
  } catch (error) {
    r.line("error", "Lost the connection: " + error.message);
  } finally {
    r.done();
    busy = false; send.disabled = false;
    input.focus();
  }
}

function showUploads() {
  chips.textContent = uploads.length ? "Attached: " + uploads.join(", ") : "";
}

$("file").addEventListener("change", async (event) => {
  for (const file of event.target.files) {
    chips.textContent = "Uploading " + file.name + "…";
    try {
      const res = await fetch("api/upload?name=" + encodeURIComponent(file.name), {
        method: "POST", headers: {"X-Happiness": "1", "Content-Type": "application/octet-stream"}, body: file,
      });
      const body = await res.json();
      if (res.ok) uploads.push(body.path);
      else log.append(el("div", "error", file.name + ": " + body.error));
    } catch (error) {
      log.append(el("div", "error", file.name + ": " + error.message));
    }
  }
  event.target.value = "";
  showUploads();
});

$("composer").addEventListener("submit", (event) => {
  event.preventDefault();
  let message = input.value.trim();
  if (busy || (!message && !uploads.length)) return;
  if (uploads.length) {
    message = "(I uploaded " + uploads.join(", ") + " to the workspace.)\n\n" + (message || "Please have a look.");
    uploads = [];
    showUploads();
  }
  input.value = "";
  input.style.height = "auto";
  ask(message);
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    $("composer").requestSubmit();
  }
});
input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = input.scrollHeight + "px";
});

$("new").addEventListener("click", () => {
  conversation = newId();
  log.replaceChildren(el("p", "welcome", "New conversation. What shall we do?"));
  input.focus();
});
</script>
</body>
</html>
"""
