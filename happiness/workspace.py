"""A folder Claude may read and write through Anthropic's text editor tool, and nothing outside it."""

from __future__ import annotations

import base64
import os
import shutil
from pathlib import Path
from typing import Any

from anthropic.lib.tools import ToolError

MAX_TEXT = 100_000  # characters returned when viewing a text file
MAX_LISTING = 400  # entries returned when viewing a folder
MAX_PDF = 20 * 1024 * 1024
MAX_IMAGE = 5 * 1024 * 1024
IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif", ".webp": "image/webp"}


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


