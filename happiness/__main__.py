"""Command line.

    python -m happiness                     chat in the terminal
    python -m happiness "a request"         answer once and exit (stdin is appended, if piped)
    python -m happiness serve               the web chat
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import TextIO

from .agent import DEFAULT_MODEL, EFFORTS, Agent, Settings, describe_error
from .tools import Approver

try:
    import readline  # noqa: F401  (line editing and history for input())
except ImportError:
    pass

HELP = "/new starts a fresh conversation · /exit leaves · Ctrl-C stops a reply"


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


def approver(printer: Printer) -> Approver:
    def approve(command: str) -> bool:
        printer.line(f"  ? Happiness wants to run:  {command}", "36")
        try:
            answer = input("    allow? [y/N] ")
        except EOFError:
            printer.out.write("\n")
            return False
        return answer.strip().lower() in ("y", "yes")

    return approve


def add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", help=f"Claude model (default: $HAPPINESS_MODEL or {DEFAULT_MODEL})")
    parser.add_argument("--effort", choices=EFFORTS, help="how hard to think (default: $HAPPINESS_EFFORT or medium)")
    parser.add_argument("--home", type=Path, help="memory, skills and workspace (default: $HAPPINESS_HOME or ~/.happiness)")
    parser.add_argument("--workspace", type=Path, help="the folder Happiness may read and write (default: <home>/workspace)")
    parser.add_argument("--no-web", action="store_true", help="turn off web search and web fetch")


def settings_from(args: argparse.Namespace) -> Settings:
    settings = Settings.from_env()
    if args.model:
        settings.model = args.model
    if args.effort:
        settings.effort = args.effort
    if args.home:
        settings.home = args.home.expanduser()
    if args.workspace:
        settings.workspace = args.workspace.expanduser()
    if args.no_web:
        settings.web = False
    return settings


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
    s = agent.settings
    tools = "web, memory, workspace, skills" + (", shell" if s.shell else "")
    printer.out.write(printer.style("1", "Happiness") + f" · {s.model} · {tools}\n")
    printer.line(f"workspace: {agent.toolbox.workspace.root}\n{HELP}\n")
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
            printer.line("  · fresh conversation")
            continue
        if text in ("/help", "/?"):
            printer.line(HELP)
            continue
        printer.out.write("\n")
        answer(agent, printer, text)
        printer.out.write("\n")


def serve_command(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m happiness serve",
        description="Serve the Happiness web chat. Set HAPPINESS_PASSWORD to serve beyond this computer.",
    )
    add_common_options(parser)
    parser.add_argument("--host", default=os.environ.get("HAPPINESS_HOST", "127.0.0.1"), help="default: 127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("HAPPINESS_PORT", "8000")), help="default: 8000")
    args = parser.parse_args(argv)

    from .web import serve

    return serve(settings_from(args), args.host, args.port, password=os.environ.get("HAPPINESS_PASSWORD"))


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["serve"]:
        return serve_command(argv[1:])

    parser = argparse.ArgumentParser(
        prog="python -m happiness",
        description="Happiness: a small agent you run yourself.",
        epilog="The web chat: python -m happiness serve --help",
    )
    add_common_options(parser)
    parser.add_argument("--shell", action="store_true", help="let Happiness run shell commands in the workspace (you approve each)")
    parser.add_argument("--yes", action="store_true", help="with --shell: run commands without asking first")
    parser.add_argument("request", nargs="*", help="answer this once and exit; leave out to chat")
    args = parser.parse_args(argv)

    settings = settings_from(args)
    settings.shell = settings.shell or args.shell
    printer = Printer()
    try:
        agent = Agent(settings, approve=None if args.yes else approver(printer))
    except ValueError as error:  # e.g. HAPPINESS_EFFORT set to something unknown
        parser.error(str(error))

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
