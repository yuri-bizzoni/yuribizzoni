"""The web chat: authentication, the page, streaming replies and uploads, over real HTTP."""

from __future__ import annotations

import base64
import contextlib
import io
import json
import logging
import re
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from happiness.agent import Settings
from happiness.tests import fakeapi as api
from happiness.web import make_server, safe_filename, serve

AUTH = "Basic " + base64.b64encode(b"me:open sesame").decode()
logging.getLogger("happiness").addHandler(logging.NullHandler())  # keep the server's request log out of test output


class WebTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = Path(tmp.name)
        self.script = api.ScriptedAPI(
            api.reply(api.tool_use("srvtoolu_1", "web_search", {"query": "gioia"}, server=True), api.search_results("srvtoolu_1"), api.text("**Gioia** is bright.")),
        )
        settings = Settings(home=self.home, shell=True)
        self.server = make_server(settings, "127.0.0.1", 0, password="open sesame", client=self.script.client())
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def fetch(self, path: str, *, data: bytes | None = None, headers: dict[str, str] | None = None) -> tuple[int, dict[str, str], bytes]:
        request = urllib.request.Request(self.base + path, data=data, headers=headers or {}, method="POST" if data is not None else "GET")
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status, dict(response.headers), response.read()
        except urllib.error.HTTPError as error:
            return error.code, dict(error.headers), error.read()

    def chat(self, message: str, headers: dict[str, str] | None = None) -> tuple[int, dict[str, str], bytes]:
        body = json.dumps({"conversation": "c1", "message": message}).encode()
        base = {"Authorization": AUTH, "Content-Type": "application/json", "X-Happiness": "1"}
        return self.fetch("/api/chat", data=body, headers={**base, **(headers or {})})

    def test_the_page_needs_the_password(self) -> None:
        status, headers, _ = self.fetch("/")
        self.assertEqual(status, 401)
        self.assertIn("Basic", headers["WWW-Authenticate"])
        self.assertEqual(self.fetch("/healthz")[0], 200)

        status, headers, body = self.fetch("/", headers={"Authorization": AUTH})
        self.assertEqual(status, 200)
        nonce = re.search(r"'nonce-([^']+)'", headers["Content-Security-Policy"]).group(1)
        self.assertIn(f'<script nonce="{nonce}">', body.decode())

    def test_chat_streams_tool_calls_and_text(self) -> None:
        status, headers, body = self.chat("What is gioia?")
        self.assertEqual(status, 200)
        self.assertTrue(headers["Content-Type"].startswith("text/event-stream"))
        events = [(m.group(1), json.loads(m.group(2))) for m in re.finditer(r"event: (\w+)\ndata: (.*)\n\n", body.decode())]
        self.assertEqual(
            events,
            [("tool", "searching the web for “gioia”"), ("text", "**Gioia** is bright."), ("done", "")],
        )
        self.assertEqual(self.script.requests[0]["messages"], [{"role": "user", "content": "What is gioia?"}])

    def test_the_shell_is_never_offered_in_the_browser(self) -> None:
        self.chat("Hi")
        self.assertNotIn("bash", [tool["name"] for tool in self.script.requests[0]["tools"]])

    def test_api_calls_need_the_page_header(self) -> None:
        status, _, _ = self.chat("Hi", headers={"X-Happiness": ""})
        self.assertEqual(status, 403)
        self.assertEqual(self.script.requests, [])

    def test_api_errors_are_reported_in_the_stream(self) -> None:
        self.chat("Hi")  # uses up the scripted reply; the next request gets a 500
        status, _, body = self.chat("Again")
        self.assertEqual(status, 200)
        self.assertIn("event: error", body.decode())
        self.assertIn("500", body.decode())

    def test_uploads_land_in_the_workspace(self) -> None:
        headers = {"Authorization": AUTH, "X-Happiness": "1"}
        status, _, body = self.fetch("/api/upload?name=..%2F..%2Fnotes.txt", data=b"hello", headers=headers)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), {"path": "uploads/notes.txt"})
        status, _, body = self.fetch("/api/upload?name=notes.txt", data=b"again", headers=headers)
        self.assertEqual(json.loads(body), {"path": "uploads/notes-1.txt"})
        self.assertEqual((self.home / "workspace/uploads/notes.txt").read_text(), "hello")

    def test_safe_filenames(self) -> None:
        self.assertEqual(safe_filename("../../etc/passwd"), "passwd")
        self.assertEqual(safe_filename("C:\\Users\\me\\Tesi finale.pdf"), "Tesi finale.pdf")
        self.assertEqual(safe_filename(".."), "")


class ServeTests(unittest.TestCase):
    def refusal(self, settings: Settings, host: str) -> str:
        with contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(serve(settings, host, 0, password=None), 2)
        return err.getvalue()

    def test_refuses_to_serve_publicly_without_a_password(self) -> None:
        with tempfile.TemporaryDirectory() as home:
            self.assertIn("HAPPINESS_PASSWORD", self.refusal(Settings(home=Path(home)), "0.0.0.0"))

    def test_refuses_an_unknown_effort(self) -> None:
        with tempfile.TemporaryDirectory() as home:
            self.assertIn("HAPPINESS_EFFORT", self.refusal(Settings(home=Path(home), effort="lots"), "127.0.0.1"))


if __name__ == "__main__":
    unittest.main()
