"""The tools Happiness can use, each one confined to a folder you choose.

Web search and web fetch are server tools: Anthropic runs them, so this module only
declares them. Memory, the workspace editor, skills and the optional shell run here,
on your machine.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from anthropic.lib.tools import ToolError
from anthropic.tools.memory import BetaLocalFilesystemMemoryTool

WEB_TOOLS: list[dict[str, Any]] = [
    {"type": "web_search_20260209", "name": "web_search", "max_uses": 8},
    {"type": "web_fetch_20260209", "name": "web_fetch", "max_uses": 8},
]
EDITOR: dict[str, Any] = {"type": "text_editor_20250728", "name": "str_replace_based_edit_tool"}
BASH: dict[str, Any] = {"type": "bash_20250124", "name": "bash"}
LOAD_SKILL: dict[str, Any] = {
    "name": "load_skill",
    "description": (
        "Load the full instructions of one of your skills. Call it before any task a skill covers "
        "(see 'Your skills' in your instructions), then follow what it says."
    ),
    "eager_input_streaming": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "The skill's name, as listed in your instructions."},
            "file": {
                "type": "string",
                "description": "Another file inside the skill's folder, when its instructions point to one. "
                "Defaults to SKILL.md.",
            },
        },
        "required": ["name"],
        "additionalProperties": False,
    },
}

MAX_TEXT = 100_000  # characters returned when viewing a text file
MAX_LISTING = 400  # entries returned when viewing a folder
MAX_PDF = 20 * 1024 * 1024
MAX_IMAGE = 5 * 1024 * 1024
IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif", ".webp": "image/webp"}
SHELL_TIMEOUT = 120  # seconds
MAX_SHELL_OUTPUT = 30_000  # characters

Approver = Callable[[str], bool]
"""Gets a shell command before it runs and says whether it may run."""


class InvalidInput(Exception):
    """A tool call whose arguments don't match the tool's schema."""


# --------------------------------------------------------------------------- skills


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    folder: Path


def read_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Split a SKILL.md into its `---` header fields (flat `key: value` lines) and its body."""
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    fields: dict[str, str] = {}
    for line in text[3:end].splitlines():
        key, sep, value = line.partition(":")
        if not sep or not key.strip() or line[:1].isspace():
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] == '"':
            try:
                value = json.loads(value)
            except ValueError:
                value = value[1:-1]
        elif len(value) >= 2 and value[0] == value[-1] == "'":
            value = value[1:-1].replace("''", "'")
        fields[key.strip()] = value
    return fields, text[end + 4 :].lstrip("\n")


def discover_skills(folders: Iterable[Path]) -> dict[str, Skill]:
    """Find `<folder>/<skill>/SKILL.md` files. A skill in a later folder replaces one of the same name."""
    skills: dict[str, Skill] = {}
    for folder in folders:
        if not folder.is_dir():
            continue
        for skill_md in sorted(folder.glob("*/SKILL.md")):
            try:
                fields, _ = read_frontmatter(skill_md.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError):
                continue
            name = fields.get("name") or skill_md.parent.name
            skills[name] = Skill(name, " ".join(fields.get("description", "").split()), skill_md.parent)
    return skills


def read_skill(skills: dict[str, Skill], name: str, file: str = "SKILL.md") -> str:
    skill = skills.get(name)
    if skill is None:
        raise ToolError(f"No skill called {name!r}. Available: {', '.join(sorted(skills)) or 'none'}.")
    folder = skill.folder.resolve()
    target = (folder / file).resolve()
    if not target.is_relative_to(folder) or not target.is_file():
        raise ToolError(f"{file} is not a file of the {name} skill.")
    return target.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- workspace


def _size(n: float) -> str:
    if n < 1024:
        return f"{int(n)} B"
    for unit in ("KB", "MB"):
        n /= 1024
        if n < 1024:
            break
    return f"{n:.1f} {unit}"


def _children(folder: Path) -> list[os.DirEntry[str]]:
    try:
        with os.scandir(folder) as entries:
            return sorted((e for e in entries if not e.name.startswith(".")), key=lambda e: e.name.lower())
    except OSError:
        return []


class Workspace:
    """Anthropic's text editor tool, confined to one folder: view, create, str_replace, insert."""

    def __init__(self, root: Path):
        root.mkdir(parents=True, exist_ok=True)
        self.root = root.resolve()

    def resolve(self, path: object) -> Path:
        """Turn a path from the model into a real path inside the workspace, or refuse it."""
        if not isinstance(path, str) or not path.strip():
            raise ToolError("Give a path inside the workspace, such as `notes.md`, or `.` for the workspace itself.")
        raw = Path(path.strip())
        target = (raw if raw.is_absolute() else self.root / raw).resolve()
        if raw.is_absolute() and not target.is_relative_to(self.root):
            # The model may write `/notes.md` for the workspace's own notes.md.
            target = (self.root / raw.relative_to(raw.anchor)).resolve()
        if not target.is_relative_to(self.root):
            raise ToolError(f"{path} is outside the workspace.")
        return target

    def name(self, target: Path) -> str:
        return target.relative_to(self.root).as_posix() or "."

    def __call__(self, args: dict[str, Any]) -> str | list[dict[str, Any]]:
        command = args.get("command")
        target = self.resolve(args.get("path"))
        if command == "view":
            return self.view(target, args.get("view_range"))
        if command == "create":
            return self.create(target, args.get("file_text"))
        if command == "str_replace":
            return self.str_replace(target, args.get("old_str"), args.get("new_str", ""))
        if command == "insert":
            return self.insert(target, args.get("insert_line"), args.get("insert_text", args.get("new_str")))
        raise ToolError(f"Unknown command {command!r}: use view, create, str_replace or insert.")

    def view(self, target: Path, view_range: object = None) -> str | list[dict[str, Any]]:
        if target.is_dir():
            return self._listing(target)
        if not target.is_file():
            raise ToolError(f"{self.name(target)} does not exist.")
        size = target.stat().st_size
        suffix = target.suffix.lower()
        if suffix == ".pdf" or suffix in IMAGE_TYPES:
            limit = MAX_PDF if suffix == ".pdf" else MAX_IMAGE
            if size > limit:
                raise ToolError(f"{self.name(target)} is {_size(size)}; the limit for this kind of file is {_size(limit)}.")
            data = base64.standard_b64encode(target.read_bytes()).decode("ascii")
            kind = "document" if suffix == ".pdf" else "image"
            media_type = "application/pdf" if suffix == ".pdf" else IMAGE_TYPES[suffix]
            return [
                {"type": "text", "text": f"{self.name(target)} ({_size(size)}):"},
                {"type": kind, "source": {"type": "base64", "media_type": media_type, "data": data}},
            ]
        lines = self._read_text(target).splitlines()
        first, last = 1, len(lines)
        if view_range is not None:
            if not (isinstance(view_range, list) and len(view_range) == 2 and all(type(n) is int for n in view_range)):
                raise ToolError("view_range must be [first_line, last_line]; use -1 as last_line for the end of the file.")
            first = max(1, view_range[0])
            last = len(lines) if view_range[1] == -1 else min(view_range[1], len(lines))
        text = "\n".join(f"{n:6}\t{lines[n - 1]}" for n in range(first, last + 1))
        if len(text) > MAX_TEXT:
            text = text[:MAX_TEXT] + f"\n… cut at {MAX_TEXT:,} characters; use view_range to read the rest."
        return text or "(empty file)"

    def _listing(self, folder: Path) -> str:
        lines: list[str] = []
        for entry in _children(folder):
            for item in [entry, *(_children(Path(entry.path)) if entry.is_dir() else [])]:
                try:
                    detail = "/" if item.is_dir() else f"  {_size(item.stat().st_size)}"
                except OSError:  # a broken link, say
                    continue
                lines.append(self.name(Path(item.path)) + detail)
                if len(lines) >= MAX_LISTING:
                    return "\n".join(lines) + f"\n… more than {MAX_LISTING} entries; view a subfolder to see the rest."
        return "\n".join(lines) or f"{self.name(folder)} is empty."

    def _read_text(self, target: Path) -> str:
        if not target.is_file():
            raise ToolError(f"{self.name(target)} does not exist.")
        try:
            return target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise ToolError(f"{self.name(target)} is not a UTF-8 text file.") from None

    def create(self, target: Path, file_text: object) -> str:
        if not isinstance(file_text, str):
            raise ToolError("create needs file_text.")
        if target.is_dir():
            raise ToolError(f"{self.name(target)} is a folder.")
        target.parent.mkdir(parents=True, exist_ok=True)
        note = ""
        if target.exists():
            backup = self.root / ".backups" / self.name(target)
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, backup)
            note = f" (the previous version is in .backups/{self.name(target)})"
        target.write_text(file_text, encoding="utf-8")
        return f"Wrote {self.name(target)}{note}."

    def str_replace(self, target: Path, old: object, new: object) -> str:
        text = self._read_text(target)
        if not isinstance(old, str) or not old:
            raise ToolError("str_replace needs old_str.")
        if not isinstance(new, str):
            raise ToolError("new_str must be text.")
        count = text.count(old)
        if count != 1:
            raise ToolError(
                f"old_str was not found in {self.name(target)}."
                if count == 0
                else f"old_str appears {count} times in {self.name(target)}; include more context so it matches once."
            )
        target.write_text(text.replace(old, new, 1), encoding="utf-8")
        return f"Edited {self.name(target)}."

    def insert(self, target: Path, line: object, text: object) -> str:
        lines = self._read_text(target).splitlines(keepends=True)
        if type(line) is not int or not 0 <= line <= len(lines):
            raise ToolError(f"insert_line must be a line number from 0 (the top) to {len(lines)}.")
        if not isinstance(text, str):
            raise ToolError("insert needs insert_text.")
        if lines and line == len(lines) and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        lines.insert(line, text if text.endswith("\n") else text + "\n")
        target.write_text("".join(lines), encoding="utf-8")
        return f"Inserted the text after line {line} of {self.name(target)}."


# --------------------------------------------------------------------------- shell


def _clip(output: str) -> str:
    output = output.strip()
    if len(output) <= MAX_SHELL_OUTPUT:
        return output
    half = MAX_SHELL_OUTPUT // 2
    return f"{output[:half]}\n… {len(output) - MAX_SHELL_OUTPUT:,} characters left out …\n{output[-half:]}"


class Shell:
    """Anthropic's bash tool. Each command runs in a fresh shell in the workspace, without your API keys."""

    def __init__(self, cwd: Path, approve: Approver | None = None):
        self.cwd = cwd
        self.approve = approve

    def __call__(self, args: dict[str, Any]) -> str:
        if args.get("restart"):
            return "Ready. Each command runs in a fresh shell that starts in the workspace."
        command = args.get("command")
        if not isinstance(command, str) or not command.strip():
            raise ToolError("Give a command to run.")
        if self.approve is not None and not self.approve(command):
            raise ToolError("The user declined to run this command.")
        env = {k: v for k, v in os.environ.items() if not k.startswith(("ANTHROPIC_", "HAPPINESS_"))}
        process = subprocess.Popen(
            ["bash", "-c", command],
            cwd=self.cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            start_new_session=True,
        )
        try:
            output, _ = process.communicate(timeout=SHELL_TIMEOUT)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate()
            raise ToolError(f"Stopped after {SHELL_TIMEOUT} seconds.\n{_clip(output)}") from None
        output = _clip(output) or "(no output)"
        return output if process.returncode == 0 else f"{output}\n(exit code {process.returncode})"


# --------------------------------------------------------------------------- toolbox


def describe(name: str, args: object) -> str:
    """A one-line, human-readable account of a tool call, for the chat window."""
    a = args if isinstance(args, dict) else {}
    if name == "web_search":
        return f"searching the web for “{a.get('query', '')}”"
    if name == "web_fetch":
        return f"reading {a.get('url', 'a web page')}"
    if name == "memory":
        return f"memory: {a.get('command', '')} {a.get('path', '')}".strip()
    if name == "str_replace_based_edit_tool":
        verbs = {"view": "reading", "create": "writing", "str_replace": "editing", "insert": "editing"}
        return f"{verbs.get(str(a.get('command')), 'using')} {a.get('path', 'the workspace')}"
    if name == "load_skill":
        return f"loading the {a.get('name', '')} skill"
    if name == "bash":
        return f"$ {a.get('command', 'restart')}"
    return name.replace("_", " ")


class Toolbox:
    """Everything Happiness can call on your machine, and the definitions the model sees."""

    def __init__(
        self,
        *,
        workspace: Path,
        memory: Path,
        skill_folders: Iterable[Path],
        web: bool = True,
        shell: bool = False,
        approve: Approver | None = None,
    ):
        self.workspace = Workspace(workspace)
        self.memory = BetaLocalFilesystemMemoryTool(base_path=str(memory))
        self.skills = discover_skills(skill_folders)
        self.web = web
        self.shell = Shell(self.workspace.root, approve) if shell else None

    def definitions(self) -> list[dict[str, Any]]:
        """The tool list for a conversation. It stays the same from the first request to the last."""
        tools = [*WEB_TOOLS] if self.web else []
        tools += [dict(self.memory.to_dict()), EDITOR, LOAD_SKILL]
        if self.shell:
            tools.append(BASH)
        return tools

    def load_skill(self, args: dict[str, Any]) -> str:
        name, file = args.get("name"), args.get("file", "SKILL.md")
        if not isinstance(name, str) or not isinstance(file, str) or set(args) - {"name", "file"}:
            raise InvalidInput(args)
        return read_skill(self.skills, name, file)

    def run(self, name: str, args: object) -> tuple[Any, bool]:
        """Run one tool call. Returns the result content and whether it is an error."""
        handlers: dict[str, Callable[[dict[str, Any]], Any]] = {
            "memory": self.memory.call,
            "str_replace_based_edit_tool": self.workspace,
            "load_skill": self.load_skill,
        }
        if self.shell:
            handlers["bash"] = self.shell
        handler = handlers.get(name)
        if handler is None:
            return f"There is no tool called {name}.", True
        try:
            if not isinstance(args, dict):
                raise InvalidInput(args)
            return handler(args), False
        except InvalidInput as error:
            # Arguments that don't fit the schema (possible with eager input streaming): give them back.
            return json.dumps({"INVALID_JSON": json.dumps(error.args[0], ensure_ascii=False)}), True
        except ToolError as error:
            return error.content, True
        except Exception as error:  # a failed tool is reported to the model, which can adapt
            return f"{type(error).__name__}: {error}", True
