"""Where analysis scripts run: Docker without network, or the user's Python after approval."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from anthropic.lib.tools import ToolError

from happiness import sandbox
from happiness.sandbox import DockerSandbox, LocalSandbox, clip, make_sandbox


class TempDirTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.corpus = self.tmp / "corpus"
        self.corpus.mkdir()
        self.work = self.tmp / "work"
        (self.work / "scripts").mkdir(parents=True)

    def script(self, code: str) -> Path:
        path = self.work / "scripts" / "01-test.py"
        path.write_text(code, encoding="utf-8")
        return path


class LocalSandboxTests(TempDirTest):
    def test_nothing_runs_without_approval(self) -> None:
        box = LocalSandbox(self.corpus, approve=lambda code, path: False)
        with self.assertRaises(ToolError):
            box.run(self.script("open('ran', 'w')"), self.work)
        self.assertFalse((self.work / "ran").exists())

    def test_runs_in_the_lab_folder_without_your_keys(self) -> None:
        box = LocalSandbox(self.corpus, approve=lambda code, path: True)
        code = "import os\nprint(os.getcwd()); print(os.environ.get('ANTHROPIC_API_KEY')); print(os.environ['HAPPINESS_CORPUS'])"
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-secret"}):
            run = box.run(self.script(code), self.work)
        self.assertEqual(run.exit_code, 0)
        self.assertEqual(run.output.split(), [str(self.work), "None", str(self.corpus)])

    def test_long_scripts_are_stopped(self) -> None:
        box = LocalSandbox(self.corpus, approve=None)
        with mock.patch.object(sandbox, "TIMEOUT", 1):
            run = box.run(self.script("import time\nprint('started', flush=True)\ntime.sleep(30)"), self.work)
        self.assertTrue(run.timed_out)
        self.assertIn("started", run.output)


class DockerSandboxTests(TempDirTest):
    def calls(self, inspect_code: int = 0, build_code: int = 0) -> tuple[list[list[str]], mock._patch]:
        seen: list[list[str]] = []

        def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
            seen.append(command)
            code = {"inspect": inspect_code, "build": build_code}.get(command[1] if command[1] != "image" else "inspect", 0)
            return subprocess.CompletedProcess(command, code, stdout="out\n", stderr="")

        return seen, mock.patch.object(subprocess, "run", side_effect=fake_run)

    def test_scripts_run_without_network_and_with_the_corpus_read_only(self) -> None:
        seen, patch = self.calls()
        with patch:
            run = DockerSandbox(self.corpus).run(self.script("print(1)"), self.work)
        self.assertEqual(run.output, "out\n")
        command = seen[-1]
        joined = " ".join(command)
        self.assertIn("--network none", joined)
        self.assertIn("--read-only", command)
        self.assertIn(f"{self.corpus}:/corpus:ro", command)
        self.assertIn(f"{self.work}:/work", command)
        self.assertEqual(command[-2:], ["python", "scripts/01-test.py"])
        self.assertTrue(command[command.index("--network") - 1].startswith("happiness-run-"))

    def test_the_image_is_built_once_and_failures_explained(self) -> None:
        seen, patch = self.calls(inspect_code=1)
        notices: list[str] = []
        with patch:
            DockerSandbox(self.corpus).run(self.script("print(1)"), self.work, lambda kind, text: notices.append(text))
        self.assertEqual([c[1] for c in seen], ["image", "build", "run"])
        self.assertIn("Building the analysis sandbox", notices[0])
        seen, patch = self.calls(inspect_code=1, build_code=1)
        with patch, self.assertRaises(ToolError) as caught:
            DockerSandbox(self.corpus).run(self.script("print(1)"), self.work)
        self.assertIn("--sandbox local", str(caught.exception))


class ChoiceTests(TempDirTest):
    def test_make_sandbox(self) -> None:
        self.assertIsInstance(make_sandbox("local", self.corpus, None), LocalSandbox)
        self.assertIsInstance(make_sandbox("docker", self.corpus, None), DockerSandbox)
        with mock.patch.object(sandbox, "docker_available", return_value=False):
            self.assertIsInstance(make_sandbox("auto", self.corpus, None), LocalSandbox)
        with mock.patch.object(sandbox, "docker_available", return_value=True):
            self.assertIsInstance(make_sandbox("auto", self.corpus, None), DockerSandbox)
        with self.assertRaises(ValueError):
            make_sandbox("cloud", self.corpus, None)

    def test_clip_keeps_the_end(self) -> None:
        clipped = clip("a" * 5000 + "TRACEBACK", limit=300)
        self.assertTrue(clipped.endswith("TRACEBACK"))
        self.assertIn("characters left out", clipped)


if __name__ == "__main__":
    unittest.main()
