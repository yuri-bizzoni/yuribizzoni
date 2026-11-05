"""The lab: a folder per conversation, holding a notebook of what was asked and done, and the tables,
figures and scripts behind every answer. Corpus text is not copied into it: passages are logged by
position only, so the notebook says what was read without repeating it."""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

from .corpus import Corpus

HELPER = '''"""Helpers for analysis scripts: the corpus catalogue and the texts, read-only.

    from happiness_corpus import catalogue, text, words, sentences
    cat = catalogue()                      # a pandas DataFrame, one row per text
    body = text("collodi_pinocchio")       # the text, without header or licence
"""
import csv
import os
import re
from pathlib import Path

CORPUS = Path(os.environ.get("HAPPINESS_CORPUS", "/corpus"))
HERE = Path(__file__).resolve().parent
WORD = re.compile(r"[^\\W\\d_]+")
SENTENCE = re.compile(r"(?<=[.!?…])[\\"'»”’)\\]]*\\s+")
_START = re.compile(r"^\\*{3}\\s*START OF (?:THE|THIS) PROJECT GUTENBERG.*$", re.I | re.M)
_END = re.compile(r"^\\*{3}\\s*END OF (?:THE|THIS) PROJECT GUTENBERG.*$", re.I | re.M)


def catalogue():
    """One row per text: name, metadata, statistics and path. A DataFrame if pandas is there, else a list of dicts."""
    try:
        import pandas as pd
        return pd.read_csv(HERE / "catalogue.csv")
    except ImportError:
        with open(HERE / "catalogue.csv", newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))


def text(name):
    """The body of a text, as Happiness indexed it (header lines and Gutenberg licence removed)."""
    rows = catalogue()
    paths = dict(zip(rows["name"], rows["path"])) if hasattr(rows, "columns") else {r["name"]: r["path"] for r in rows}
    data = (CORPUS / paths[name]).read_bytes()
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            raw = data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    raw = raw.replace("\\r\\n", "\\n").replace("\\r", "\\n")
    start, end = _START.search(raw), _END.search(raw)
    if start and end and end.start() > start.end():
        raw = raw[start.end():end.start()]
    lines = raw.lstrip("\\n").split("\\n")
    n = 0
    while n < len(lines) and lines[n].startswith("#"):
        n += 1
    return "\\n".join(lines[n:]).strip()


def words(body):
    """The words of a text, lowercased (letters only, as in Happiness's counts)."""
    return [w.lower() for w in WORD.findall(body)]


def sentences(body):
    """The sentences of a text, split roughly at final punctuation."""
    return [s.strip() for s in SENTENCE.split(body) if s.strip()]
'''


def slug(text: str, limit: int = 40) -> str:
    words = re.findall(r"\w+", text.lower())
    return "-".join(words)[:limit].strip("-") or "session"


class Lab:
    """The lab folder (default ~/.happiness/lab), with one folder per conversation inside."""

    def __init__(self, root: Path, corpus: Corpus, model: str = ""):
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "lexicons").mkdir(exist_ok=True)
        self.corpus = corpus
        self.model = model
        self.session: Path | None = None
        self._numbers: dict[str, int] = {}

    def new_conversation(self) -> None:
        self.session = None
        self._numbers = {}

    def begin(self, question: str) -> Path:
        """The folder of this conversation, made at its first question."""
        if self.session is None:
            now = dt.datetime.now()
            stamp = now.strftime("%Y-%m-%d_%H%M")
            folder = self.root / f"{stamp}-{slug(question)}"
            n = 2
            while folder.exists():
                folder = self.root / f"{stamp}-{slug(question)}-{n}"
                n += 1
            for sub in ("tables", "figures", "scripts"):
                (folder / sub).mkdir(parents=True)
            self.corpus.export_catalogue(folder / "catalogue.csv")
            (folder / "happiness_corpus.py").write_text(HELPER, encoding="utf-8")
            texts = self.corpus.texts()
            words = sum(t["words"] or 0 for t in texts)
            (folder / "notebook.md").write_text(
                f"# Lab notebook · {now.day} {now:%B %Y, %H:%M}\n\n"
                f"Corpus: `{self.corpus.root}` ({len(texts):,} texts, {words:,} words)"
                + (f" · model {self.model}" if self.model else "")
                + "\n\nFiles: `tables/`, `figures/`, `scripts/`; `catalogue.csv` is the corpus catalogue as it was.\n",
                encoding="utf-8",
            )
            self.session = folder
        return self.session

    def path(self, kind: str, stem: str, suffix: str) -> Path:
        """A new numbered file in tables/, figures/ or scripts/, e.g. tables/count-03.csv."""
        assert self.session is not None
        number = self._numbers[kind] = self._numbers.get(kind, 0) + 1
        return self.session / kind / f"{number:02d}-{slug(stem)}{suffix}"

    def note(self, markdown: str) -> None:
        if self.session is not None:
            with (self.session / "notebook.md").open("a", encoding="utf-8") as handle:
                handle.write("\n" + markdown.rstrip() + "\n")

    def relative(self, path: Path) -> str:
        assert self.session is not None
        return path.relative_to(self.session).as_posix()
