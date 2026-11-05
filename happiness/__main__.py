"""Command line.

    python -m happiness --corpus ~/corpora/novels                 talk about your corpus
    python -m happiness --corpus ~/corpora/novels "a question"    answer once and exit (stdin is added, if piped)
    python -m happiness index --corpus ~/corpora/novels           index the corpus and describe it
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Callable, TextIO

from .agent import DEFAULT_MODEL, EFFORTS, SANDBOXES, Agent, Settings, describe_error
from .corpus import Corpus, CorpusError
from .sandbox import ScriptApprover, run_without_asking

try:
    import readline  # noqa: F401  # line editing and history for input()
except ImportError:
    pass

HELP = "/new starts afresh · /lab shows the notebook folder · /budget the passages left · /exit leaves · Ctrl-C stops a reply"


class Printer:
    """Shows the agent's progress in a terminal: the reply as it streams, tool calls as dim lines."""

    def __init__(self, out: TextIO = sys.stdout):
        self.out = out
        self.color = out.isatty() and not os.environ.get("NO_COLOR")
        self.mid_line = False
        self.started = False

    def style(self, code: str, text: str, prompt: bool = False) -> str:
        if not self.color:
            return text
        if prompt:  # tell readline which characters take no room
            return f"\001\033[{code}m\002{text}\001\033[0m\002"
        return f"\033[{code}m{text}\033[0m"

    def line(self, text: str, code: str = "2") -> None:
        if self.mid_line:
            self.out.write("\n")
        self.out.write(self.style(code, text) + "\n")
        self.out.flush()
        self.mid_line = False

    def begin(self) -> None:
        self.started = False

    def end(self) -> None:
        if self.mid_line:
            self.out.write("\n")
            self.mid_line = False
        self.out.flush()

    def __call__(self, kind: str, text: str) -> None:
        if kind == "thinking":
            if not self.started:
                self.line("  · thinking…")
        elif kind == "text":
            self.out.write(text)
            self.out.flush()
            self.mid_line = not text.endswith("\n")
        elif kind == "tool":
            self.line(f"  · {text}")
        elif kind == "notice":
            self.line(f"  ! {text}", "33")
        self.started = True


def script_approver(printer: Printer) -> ScriptApprover:
    """Show each script before it runs with the user's own Python, and ask."""
    always = False

    def approve(code: str, path: Path) -> bool:
        nonlocal always
        if always:
            return True
        lines = code.splitlines()
        printer.line(f"  ? Happiness wants to run {path.name} with your Python (there is no sandbox without Docker):", "36")
        printer.out.write(printer.style("2", "\n".join(f"    {line}" for line in lines[:40])) + "\n")
        if len(lines) > 40:
            printer.line(f"    … {len(lines) - 40} more lines in {path}")
        try:
            answer = input("    run it? [y]es, [n]o, [a]lways in this session: ").strip().lower()
        except EOFError:
            printer.out.write("\n")
            return False
        always = answer in ("a", "always")
        return always or answer in ("y", "yes")

    return approve


def add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--corpus", type=Path, help="the folder of your texts: .txt files, searched recursively (default: $HAPPINESS_CORPUS)")
    parser.add_argument("--metadata", type=Path, help="a CSV with one row per text (default: metadata.csv in the corpus folder)")
    parser.add_argument("--home", type=Path, help="the index, memory and skills (default: $HAPPINESS_HOME or ~/.happiness)")


def settings_from(args: argparse.Namespace) -> Settings:
    settings = Settings.from_env()
    for key in ("corpus", "metadata", "home", "lab"):
        if getattr(args, key, None):
            setattr(settings, key, getattr(args, key).expanduser())
    for key in ("model", "effort", "sandbox"):
        if getattr(args, key, None):
            setattr(settings, key, getattr(args, key))
    if getattr(args, "passages", None) is not None:
        settings.passages = args.passages
    if getattr(args, "web", False):
        settings.web = True
    return settings


def progress_printer(out: TextIO) -> Callable[[int, int, str], None]:
    tty = out.isatty()
    last = [0.0]

    def progress(done: int, total: int, name: str) -> None:
        if tty:
            out.write(f"\r\033[2K  indexing {done:,}/{total:,} {name[:60]}" if done < total else "\r\033[2K")
            out.flush()
        elif done == total or time.monotonic() - last[0] > 10:
            out.write(f"  indexing {done:,}/{total:,}\n")
            last[0] = time.monotonic()

    return progress


def open_corpus(settings: Settings, printer: Printer) -> Corpus:
    started = time.monotonic()
    corpus = settings.open_corpus()
    report = corpus.update(progress_printer(printer.out))
    if report["indexed"] or report["removed"]:
        printer.line(
            f"  · indexed {report['indexed']:,} texts" + (f", dropped {report['removed']:,}" if report["removed"] else "")
            + f" ({time.monotonic() - started:.0f} s)"
        )
    return corpus


def answer(agent: Agent, printer: Printer, text: str) -> bool:
    """Run one request, streaming to the terminal. False if it failed or was stopped."""
    printer.begin()
    try:
        agent.ask(text, on_event=printer)
    except KeyboardInterrupt:
        printer.line("  · stopped")
        return False
    except Exception as error:
        message = describe_error(error)
        if message is None:
            raise
        printer.line(message, "31")
        return False
    finally:
        printer.end()
    return True


def chat(agent: Agent, printer: Printer) -> int:
    s, texts = agent.settings, agent.corpus.texts()
    sandbox = "Docker, no network" if agent.toolbox.sandbox.kind == "docker" else "your Python, you approve each script"
    printer.out.write(printer.style("1", "Happiness") + f" · {s.model} · {len(texts):,} texts, "
                      f"{sum(t['words'] or 0 for t in texts):,} words · analysis: {sandbox}\n")
    printer.line(f"lab: {agent.lab.root}\n{HELP}\n")
    you = printer.style("1", "you ›", prompt=True) + " "
    while True:
        try:
            text = input(you).strip()
        except (EOFError, KeyboardInterrupt):
            printer.out.write("\n")
            return 0
        if not text:
            continue
        if text in ("/exit", "/quit"):
            return 0
        if text == "/new":
            agent.new_conversation()
            printer.line("  · fresh conversation (and a fresh passage budget)")
            continue
        if text == "/lab":
            printer.line(f"  · {agent.lab.session or 'no folder yet: it is made at your first question'}")
            continue
        if text == "/budget":
            budget = agent.toolbox.budget
            printer.line(f"  · {budget.left:,} of {budget.limit:,} characters of passages left in this conversation")
            continue
        if text in ("/help", "/?"):
            printer.line(HELP)
            continue
        printer.out.write("\n")
        answer(agent, printer, text)
        printer.out.write("\n")


def index_command(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m happiness index", description="Index a corpus (or bring its index up to date) and describe it.")
    add_common_options(parser)
    args = parser.parse_args(argv)
    settings = settings_from(args)
    if settings.corpus is None:
        parser.error("give the folder of your texts with --corpus (or HAPPINESS_CORPUS)")
    printer = Printer()
    try:
        corpus = open_corpus(settings, printer)
    except CorpusError as error:
        parser.error(str(error))
    printer.out.write(corpus.describe() + f"\nIndex: {corpus.index_path}\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["index"]:
        return index_command(argv[1:])

    parser = argparse.ArgumentParser(
        prog="python -m happiness",
        description="Happiness: a research assistant for your own corpus, on your own machine.",
        epilog="Index without chatting: python -m happiness index --corpus PATH",
    )
    add_common_options(parser)
    parser.add_argument("--lab", type=Path, help="notebooks, tables, scripts and figures (default: <home>/lab)")
    parser.add_argument("--model", help=f"Claude model (default: $HAPPINESS_MODEL or {DEFAULT_MODEL})")
    parser.add_argument("--effort", choices=EFFORTS, help="how hard to think (default: $HAPPINESS_EFFORT or medium)")
    parser.add_argument("--sandbox", choices=SANDBOXES, help="where analysis scripts run (default: auto, i.e. Docker if it is running)")
    parser.add_argument("--yes", action="store_true", help="with --sandbox local: run scripts without asking first")
    parser.add_argument("--passages", type=int, help="characters of corpus text that may go to Claude per conversation (default 20000)")
    parser.add_argument("--web", action="store_true", help="also let Happiness search the web (for facts about authors and works)")
    parser.add_argument("request", nargs="*", help="answer this once and exit; leave out to chat")
    args = parser.parse_args(argv)

    settings = settings_from(args)
    try:
        settings.check()
    except ValueError as error:
        parser.error(str(error))
    printer = Printer()
    try:
        corpus = open_corpus(settings, printer)
    except CorpusError as error:
        parser.error(str(error))
    agent = Agent(settings, corpus=corpus, approve=run_without_asking if args.yes else script_approver(printer))

    request = " ".join(args.request).strip()
    if not sys.stdin.isatty():
        piped = sys.stdin.read().strip()
        request = f"{request}\n\n{piped}".strip() if piped else request
        if not request:
            parser.error("nothing to do: give a request, or run in a terminal to chat")
    if request:
        return 0 if answer(agent, printer, request) else 1
    return chat(agent, printer)


if __name__ == "__main__":
    sys.exit(main())
