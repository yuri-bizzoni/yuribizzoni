"""Running the analysis code Claude writes.

With Docker, scripts run in a container with no network, the corpus mounted read-only and only this
conversation's lab folder writable. Without Docker they run with your own Python, after you approve
each one.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from anthropic.lib.tools import ToolError

IMAGE_DIR = Path(__file__).resolve().parent / "sandbox_image"
TIMEOUT = 300  # seconds per script
MAX_OUTPUT = 6_000  # characters of output returned to the model

Notify = Callable[[str, str], None]
ScriptApprover = Callable[[str, Path], bool]
"""Gets a script's code and path before it runs, and says whether it may run."""


def ask_before_running(code: str, path: Path) -> bool:
    """The default without Docker: show the script and ask (works in a terminal and in Jupyter)."""
    lines = code.splitlines()
    print(f"Happiness wants to run {path.name} with your own Python (there is no sandbox without Docker):\n")
    print("\n".join(f"    {line}" for line in lines[:60]) + (f"\n    … {len(lines) - 60} more lines in {path}" if len(lines) > 60 else ""))
    try:
        return input("Run it? [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def run_without_asking(code: str, path: Path) -> bool:
    return True


@dataclass
class Run:
    output: str
    exit_code: int
    seconds: float
    timed_out: bool = False


def clip(output: str, limit: int = MAX_OUTPUT) -> str:
    output = output.strip()
    if len(output) <= limit:
        return output
    head = limit // 3
    return f"{output[:head]}\n… {len(output) - limit:,} characters left out …\n{output[-(limit - head):]}"


def _ignore(kind: str, text: str) -> None:
    pass


def docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"], capture_output=True, timeout=15).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


class DockerSandbox:
    kind = "docker"

    def __init__(self, corpus_root: Path):
        self.corpus_root = corpus_root
        digest = hashlib.sha256((IMAGE_DIR / "Dockerfile").read_bytes()).hexdigest()[:12]
        self.image = f"happiness-sandbox:{digest}"

    def describe(self) -> str:
        return (
            "Python 3.12 in a Docker container on the user's machine, with no network: numpy, pandas, scipy, "
            "scikit-learn, statsmodels and matplotlib. The corpus is read-only at /corpus."
        )

    def ensure_image(self, notify: Notify) -> None:
        if subprocess.run(["docker", "image", "inspect", self.image], capture_output=True).returncode == 0:
            return
        notify("notice", "Building the analysis sandbox (first time only; it takes a few minutes).")
        build = subprocess.run(["docker", "build", "-t", self.image, str(IMAGE_DIR)], capture_output=True, text=True, timeout=1800)
        if build.returncode:
            raise ToolError(
                "Could not build the Docker sandbox:\n" + clip(build.stdout + build.stderr, 2000)
                + "\nThe user can run Happiness with --sandbox local instead (they approve each script)."
            )

    def run(self, script: Path, workdir: Path, notify: Notify = _ignore) -> Run:
        self.ensure_image(notify)
        name = f"happiness-run-{uuid.uuid4().hex[:10]}"
        user = [] if not hasattr(os, "getuid") else ["--user", f"{os.getuid()}:{os.getgid()}"]
        command = [
            "docker", "run", "--rm", "--name", name, "--network", "none",
            "--memory", "4g", "--cpus", "2", "--pids-limit", "256", "--read-only", "--tmpfs", "/tmp:rw,size=1g",
            *user, "-v", f"{self.corpus_root}:/corpus:ro", "-v", f"{workdir}:/work", "-w", "/work",
            "-e", "HOME=/tmp", "-e", "MPLCONFIGDIR=/tmp/matplotlib", "-e", "MPLBACKEND=Agg", "-e", "PYTHONPATH=/work",
            "-e", "HAPPINESS_CORPUS=/corpus", self.image, "python", script.relative_to(workdir).as_posix(),
        ]
        started = time.monotonic()
        try:
            done = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=TIMEOUT)
        except subprocess.TimeoutExpired as expired:
            subprocess.run(["docker", "kill", name], capture_output=True)
            output = (expired.stdout or b"").decode("utf-8", "replace") if isinstance(expired.stdout, bytes) else (expired.stdout or "")
            return Run(output, -1, time.monotonic() - started, timed_out=True)
        return Run(done.stdout + done.stderr, done.returncode, time.monotonic() - started)


class LocalSandbox:
    kind = "local"

    def __init__(self, corpus_root: Path, approve: ScriptApprover | None):
        self.corpus_root = corpus_root
        self.approve = approve

    def describe(self) -> str:
        return (
            "the user's own Python (whatever libraries they have installed). It is not isolated, so the user "
            f"approves each script before it runs. The corpus is at {self.corpus_root}: read it, never write to it."
        )

    def run(self, script: Path, workdir: Path, notify: Notify = _ignore) -> Run:
        if self.approve is not None and not self.approve(script.read_text(encoding="utf-8"), script):
            raise ToolError("The user declined to run this script.")
        env = {k: v for k, v in os.environ.items() if not k.startswith(("ANTHROPIC_", "HAPPINESS_"))}
        env.update(HAPPINESS_CORPUS=str(self.corpus_root), PYTHONPATH=str(workdir), MPLBACKEND="Agg")
        started = time.monotonic()
        process = subprocess.Popen(
            [sys.executable, str(script)], cwd=workdir, env=env, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
            start_new_session=True,
        )
        try:
            output, _ = process.communicate(timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate()
            return Run(output, -1, time.monotonic() - started, timed_out=True)
        return Run(output, process.returncode, time.monotonic() - started)


Sandbox = DockerSandbox | LocalSandbox


def make_sandbox(mode: str, corpus_root: Path, approve: ScriptApprover | None) -> Sandbox:
    """`docker`, `local`, or `auto` (Docker when it is running, otherwise local with approval)."""
    if mode not in ("auto", "docker", "local"):
        raise ValueError(f"sandbox must be auto, docker or local, not {mode!r}")
    if mode == "docker" or (mode == "auto" and docker_available()):
        return DockerSandbox(corpus_root)
    return LocalSandbox(corpus_root, approve)
