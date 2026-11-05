"""Skills (folders of instructions Claude loads when a task calls for them) and lexicons (CSV lists of terms)."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from anthropic.lib.tools import ToolError

from .corpus import decode

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




# --------------------------------------------------------------------------- lexicons

HEADER_WORDS = {
    "term", "terms", "form", "forms", "forma", "forme", "word", "words", "parola", "parole", "lemma", "category",
    "categories", "categoria", "family", "famiglia", "kategori", "ord", "wort", "mot", "catégorie",
}


@dataclass(frozen=True)
class Lexicon:
    name: str
    path: Path
    terms: list[str]
    categories: list[str] | None

    def summary(self) -> str:
        if not self.categories:
            return f"{len(self.terms)} terms"
        unique = list(dict.fromkeys(self.categories))
        shown = ", ".join(unique[:8]) + (", …" if len(unique) > 8 else "")
        return f"{len(self.terms)} terms in {len(unique)} categories ({shown})"


def read_lexicon(path: Path, name: str | None = None) -> Lexicon:
    """A lexicon from a CSV file: `category,term` per row (or just `term`); a header row is fine."""
    text = decode(path.read_bytes())
    try:
        dialect = csv.Sniffer().sniff(text[:4000], delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    rows = [[cell.strip() for cell in row] for row in csv.reader(io.StringIO(text), dialect) if any(c.strip() for c in row)]
    if rows and any(cell.lower() in HEADER_WORDS for cell in rows[0]):
        rows = rows[1:]
    pairs = [(cells[0], cells[1]) if len(cells) > 1 else ("", cells[0]) for cells in ([c for c in r if c] for r in rows) if cells]
    if not pairs:
        raise ToolError(f"The lexicon {path.name} has no terms.")
    categories = [category or term for category, term in pairs] if any(c for c, _ in pairs) else None
    return Lexicon(name or path.stem, path, [term for _, term in pairs], categories)


def discover_lexicons(folders: Iterable[Path]) -> dict[str, Path]:
    """Lexicons by name (the file name without .csv). A lexicon in a later folder replaces one of the same name."""
    found: dict[str, Path] = {}
    for folder in folders:
        if folder.is_dir():
            for path in sorted([*folder.glob("*.csv"), *folder.glob("*.tsv")]):
                found[path.stem] = path
    return found
