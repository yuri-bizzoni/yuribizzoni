"""Your corpus: plain-text files in a folder, a catalogue of them, and a local index to search them.

Everything here runs on your machine. The index is a SQLite database among your other Happiness
files (with full-text search when your SQLite has FTS5); the texts themselves are never modified.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import re
import sqlite3
import statistics
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

SCHEMA = "1"
WORD = re.compile(r"[^\W\d_]+")
SENTENCE_END = re.compile(r"[.!?…]+[\"'»”’)\]]*(?=\s|$)")
APOSTROPHE = re.compile("['’]")
BOUNDARY = r"(?<!\w)"
CHUNK_WORDS = 1000
METADATA_FILES = ("metadata.csv", "metadata.tsv", "meta.csv", "catalogue.csv", "catalog.csv")
LANGUAGE_COLUMNS = ("language", "lang", "lingua", "sprog", "sprache", "langue")
LANGUAGE_NAMES = {
    "english": "en", "inglese": "en", "engelsk": "en", "englisch": "en", "anglais": "en",
    "italian": "it", "italiano": "it", "italiensk": "it", "italienisch": "it", "italien": "it",
    "danish": "da", "dansk": "da", "danese": "da", "dänisch": "da", "danois": "da",
    "german": "de", "deutsch": "de", "tedesco": "de", "tysk": "de", "allemand": "de",
    "french": "fr", "français": "fr", "francese": "fr", "fransk": "fr", "französisch": "fr",
}
GUTENBERG_START = re.compile(r"^\*{3}\s*START OF (?:THE|THIS) PROJECT GUTENBERG.*$", re.IGNORECASE | re.MULTILINE)
GUTENBERG_END = re.compile(r"^\*{3}\s*END OF (?:THE|THIS) PROJECT GUTENBERG.*$", re.IGNORECASE | re.MULTILINE)

# The commonest function words of each language: enough to tell the five apart.
STOPWORDS = {
    "en": "the and of to a in that it was he i his you with for had is not her she but as at on be my have him this which",
    "it": "di e che il la a in non per un una si con è le del della lo mi ma i gli da al come più se era ha anche",
    "fr": "de la le et les à des il un une que en du je est pas elle pour qui dans ne se au sur plus mais avec ce son lui",
    "de": "der die und in den von zu das mit sich des auf für ist im dem nicht ein eine als auch es an er hat aus bei sie nach wie",
    "da": "og i at det en den til er som på de med han af for ikke der var mig sig men et har om vi min havde ham hun nu",
}
STOPWORD_SETS = {language: set(words.split()) for language, words in STOPWORDS.items()}
STATS = ("words", "sentences", "words_per_sentence", "letters_per_word", "sttr")
BASE_COLUMNS = ("language", "detected_language", *STATS, "chars", "path")


class CorpusError(Exception):
    """Something about the corpus or a query that the user (or the model) can fix."""


# --------------------------------------------------------------------------- reading texts


def decode(data: bytes) -> str:
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            pass
    return data.decode("latin-1")


def column_name(name: str) -> str:
    name = re.sub(r"\W+", "_", name.strip().lower()).strip("_")
    return name if name and not name[0].isdigit() else f"c_{name}"


def load_text(path: Path) -> tuple[str, dict[str, str]]:
    """The body of a text and the fields of its header.

    A Project Gutenberg licence around the text is dropped. Leading lines that start with `#` are a header:
    `# key: value` lines become fields, and a `# Title — Author` line gives the title and the author.
    """
    raw = decode(path.read_bytes()).replace("\r\n", "\n").replace("\r", "\n")
    start, end = GUTENBERG_START.search(raw), GUTENBERG_END.search(raw)
    if start and end and end.start() > start.end():
        raw = raw[start.end() : end.start()]
    lines = raw.lstrip("\n").split("\n")
    header: dict[str, str] = {}
    n = 0
    while n < len(lines) and lines[n].startswith("#"):
        line = lines[n].lstrip("#").strip()
        key, sep, value = line.partition(":")
        if sep and 0 < len(key.strip()) <= 30 and value.strip() and not value.startswith("//"):
            header.setdefault(column_name(key), value.strip())
        elif " — " in line and "title" not in header:
            title, _, author = line.partition(" — ")
            header["title"], header["author"] = title.strip(), author.strip()
        n += 1
    return "\n".join(lines[n:]).strip(), header


def detect_language(words: list[str]) -> str:
    counts = Counter(words)
    scores = {language: sum(counts[w] for w in stop) for language, stop in STOPWORD_SETS.items()}
    best = max(scores, key=scores.__getitem__)
    return best if words and scores[best] / len(words) >= 0.08 else "unknown"


def text_stats(body: str) -> dict[str, Any]:
    """Length, sentence length, word length, standardised type-token ratio and language (all approximate)."""
    words = WORD.findall(body)
    n = len(words)
    lower = [w.lower() for w in words]
    sentences = len(SENTENCE_END.findall(body)) or (1 if n else 0)
    windows = [lower[i : i + 1000] for i in range(0, min(n, 50_000), 1000)]
    windows = [w for w in windows if len(w) == 1000] or ([lower] if lower else [])
    return {
        "words": n,
        "sentences": sentences,
        "words_per_sentence": round(n / sentences, 2) if sentences else None,
        "letters_per_word": round(sum(map(len, words)) / n, 3) if n else None,
        "sttr": round(statistics.fmean(len(set(w)) / len(w) for w in windows), 4) if windows else None,
        "detected_language": detect_language(lower[:30_000]),
    }


def chunk_spans(body: str, target: int = CHUNK_WORDS) -> list[tuple[int, int]]:
    """Cut a text into consecutive pieces of about `target` words, at paragraph breaks where possible."""
    spans: list[tuple[int, int]] = []
    start, n = 0, len(body)
    while start < n:
        end = None
        for count, match in enumerate(WORD.finditer(body, start), 1):
            if count == target:
                end = match.end()
                break
        if end is None:
            spans.append((start, n))
            break
        paragraph = body.find("\n\n", end, end + 4000)
        line = body.find("\n", end, end + 1000)
        end = paragraph if paragraph != -1 else line if line != -1 else end
        spans.append((start, end))
        start = end
    return spans


# --------------------------------------------------------------------------- terms


def term_pattern(term: str) -> str:
    """A regular expression for a word (`gioia`), a phrase (`la gioia`) or a prefix (`felic*`)."""
    text = term.strip()
    prefix = text.endswith("*")
    words = text.rstrip("*").split()
    if not words or not any(WORD.search(w) for w in words):
        raise CorpusError(f"{term!r} is not a word, a phrase or a prefix.")
    parts = [APOSTROPHE.sub("['’]", re.escape(w)) for w in words]
    return BOUNDARY + r"\s+".join(parts) + (r"\w*" if prefix else r"(?!\w)")


def fts_query(terms: Iterable[str]) -> str:
    """The same terms as a full-text query. It finds a superset of the matches; the regex decides."""
    parts = []
    for term in terms:
        text = term.strip()
        quoted = '"' + text.rstrip("*").strip().replace('"', '""') + '"'
        parts.append(quoted + (" *" if text.endswith("*") else ""))
    return " OR ".join(parts)


def compile_terms(terms: list[str]) -> re.Pattern[str]:
    """One case-insensitive regex for all the terms; the group `t<i>` says which term matched.

    The word boundary they share is checked once, then the first letter, before trying each term:
    on a long lexicon that is many times faster than an alternation of whole patterns.
    """
    alternatives = [f"(?P<t{i}>{term_pattern(t)[len(BOUNDARY):]})" for i, t in enumerate(terms)]
    first: set[str] = set()
    for term in terms:
        c = term.strip()[0]
        first |= {"'", "’"} if c in "'’" else {c.lower()}
    letters = "".join(re.escape(c) for c in sorted(first))
    return re.compile(f"{BOUNDARY}(?=[{letters}])(?:{'|'.join(alternatives)})", re.IGNORECASE)


@dataclass
class TextCount:
    id: int
    name: str
    words: int
    chars: int
    hits: int = 0
    by_category: Counter = field(default_factory=Counter)
    segments: list[int] = field(default_factory=list)


@dataclass
class CountResult:
    texts: list[TextCount]
    by_term: Counter
    segments: int | None


@dataclass
class Line:
    text: str
    offset: int
    position: float
    left: str
    match: str
    right: str


def _flat(text: str) -> str:
    return " ".join(text.split())


def _texts(n: int) -> str:
    return f"{n:,} text" + ("" if n == 1 else "s")


# --------------------------------------------------------------------------- the corpus


def _has_fts5(db: sqlite3.Connection) -> bool:
    try:
        db.execute("CREATE VIRTUAL TABLE temp.fts5_probe USING fts5(x)")
        db.execute("DROP TABLE temp.fts5_probe")
        return True
    except sqlite3.OperationalError:
        return False


class Corpus:
    """A folder of plain-text files (searched recursively for *.txt) and its index."""

    def __init__(self, root: Path, index_dir: Path, metadata: Path | None = None):
        self.root = root.expanduser().resolve()
        if not self.root.is_dir():
            raise CorpusError(f"{root} is not a folder.")
        self.metadata_path = metadata.expanduser().resolve() if metadata else self._find_metadata()
        if self.metadata_path and not self.metadata_path.is_file():
            raise CorpusError(f"{metadata} is not a file.")
        index_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha1(str(self.root).encode()).hexdigest()[:10]
        self.index_path = index_dir / f"{column_name(self.root.name)}-{digest}.sqlite"
        self.db = sqlite3.connect(self.index_path, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.fts = _has_fts5(self.db)
        self._readonly: sqlite3.Connection | None = None
        self._setup()

    def _find_metadata(self) -> Path | None:
        return next((self.root / n for n in METADATA_FILES if (self.root / n).is_file()), None)

    def close(self) -> None:
        if self._readonly is not None:
            self._readonly.close()
            self._readonly = None
        self.db.close()

    def _setup(self) -> None:
        try:
            info = dict(self.db.execute("SELECT key, value FROM info"))
        except sqlite3.OperationalError:
            info = {}
        if info.get("schema") != SCHEMA or info.get("fts") != str(self.fts):
            for table in ("texts", "chunk_text", "chunk", "docs", "info"):
                self.db.execute(f"DROP TABLE IF EXISTS {table}")
        text_table = (
            "CREATE VIRTUAL TABLE IF NOT EXISTS chunk_text USING fts5(body)"
            if self.fts
            else "CREATE TABLE IF NOT EXISTS chunk_text(rowid INTEGER PRIMARY KEY, body TEXT)"
        )
        with self.db:
            self.db.executescript(
                f"""
                CREATE TABLE IF NOT EXISTS info(key TEXT PRIMARY KEY, value TEXT);
                CREATE TABLE IF NOT EXISTS docs(id INTEGER PRIMARY KEY, name TEXT UNIQUE, path TEXT, size INTEGER,
                    mtime REAL, chars INTEGER, header TEXT, stats TEXT);
                CREATE TABLE IF NOT EXISTS chunk(id INTEGER PRIMARY KEY, doc INTEGER, seq INTEGER, start INTEGER);
                CREATE INDEX IF NOT EXISTS chunk_doc ON chunk(doc);
                {text_table};
                CREATE TEMP TABLE IF NOT EXISTS sel(id INTEGER PRIMARY KEY);
                """
            )
            self.db.executemany("INSERT OR REPLACE INTO info VALUES (?, ?)", [("schema", SCHEMA), ("fts", str(self.fts))])
        self._rebuild_catalogue()

    # ---------------------------------------------------------------- indexing

    def files(self) -> dict[str, Path]:
        found = {}
        for path in sorted(self.root.rglob("*.txt")):
            relative = path.relative_to(self.root)
            if any(part.startswith(".") for part in relative.parts) or not path.is_file():
                continue
            found[relative.with_suffix("").as_posix()] = path
        return found

    def update(self, progress: Callable[[int, int, str], None] | None = None) -> dict[str, int]:
        """Bring the index up to date with the folder: new and changed texts are indexed, missing ones dropped."""
        files = self.files()
        known = {name: (doc, size, mtime) for doc, name, size, mtime in self.db.execute("SELECT id, name, size, mtime FROM docs")}
        removed = [doc for name, (doc, _, _) in known.items() if name not in files]
        changed = []
        for name, path in files.items():
            stat = path.stat()
            if name not in known or known[name][1:] != (stat.st_size, stat.st_mtime):
                changed.append(name)
        for doc in removed:
            with self.db:
                self._delete(doc)
        for done, name in enumerate(changed):
            if progress:
                progress(done, len(changed), name)
            with self.db:  # one transaction per text: an interrupted run leaves a usable index
                self._index(name, files[name])
        if progress and changed:
            progress(len(changed), len(changed), "")
        self._rebuild_catalogue()
        return {"texts": len(files), "indexed": len(changed), "removed": len(removed)}

    def _delete(self, doc: int) -> None:
        self.db.execute("DELETE FROM chunk_text WHERE rowid IN (SELECT id FROM chunk WHERE doc = ?)", (doc,))
        self.db.execute("DELETE FROM chunk WHERE doc = ?", (doc,))
        self.db.execute("DELETE FROM docs WHERE id = ?", (doc,))

    def _index(self, name: str, path: Path) -> None:
        stat = path.stat()
        body, header = load_text(path)
        old = self.db.execute("SELECT id FROM docs WHERE name = ?", (name,)).fetchone()
        if old:
            self._delete(old[0])
        doc = self.db.execute(
            "INSERT INTO docs(name, path, size, mtime, chars, header, stats) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (name, path.relative_to(self.root).as_posix(), stat.st_size, stat.st_mtime, len(body),
             json.dumps(header, ensure_ascii=False), json.dumps(text_stats(body))),
        ).lastrowid
        for seq, (start, end) in enumerate(chunk_spans(body)):
            chunk = self.db.execute("INSERT INTO chunk(doc, seq, start) VALUES (?, ?, ?)", (doc, seq, start)).lastrowid
            self.db.execute("INSERT INTO chunk_text(rowid, body) VALUES (?, ?)", (chunk, body[start:end]))

    # ---------------------------------------------------------------- the catalogue

    def _read_metadata(self, names: list[str]) -> tuple[dict[str, dict[str, str]], list[str]]:
        """Rows of the metadata table, keyed by text name. The key column is whichever matches the most names."""
        if not self.metadata_path:
            return {}, []
        text = decode(self.metadata_path.read_bytes())
        try:
            dialect: Any = csv.Sniffer().sniff(text[:8000], delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel_tab if self.metadata_path.suffix == ".tsv" else csv.excel
        rows = list(csv.DictReader(io.StringIO(text), dialect=dialect))
        if not rows:
            return {}, []
        exact = set(names)
        by_basename: dict[str, list[str]] = {}
        for name in names:
            by_basename.setdefault(name.rsplit("/", 1)[-1], []).append(name)

        def match(value: str | None) -> str | None:
            value = (value or "").strip().replace("\\", "/").removeprefix("./")
            value = value[:-4] if value.lower().endswith(".txt") else value
            if value in exact:
                return value
            candidates = by_basename.get(value.rsplit("/", 1)[-1], [])
            return candidates[0] if len(candidates) == 1 else None

        fields = [f for f in rows[0] if f]
        key = max(fields, key=lambda f: sum(match(r.get(f)) is not None for r in rows))
        found = {}
        for row in rows:
            name = match(row.get(key))
            if name:
                found[name] = {f: (row.get(f) or "").strip() for f in fields if f != key}
        return found, [f for f in fields if f != key]

    def _rebuild_catalogue(self) -> None:
        """The `texts` table: one row per text, with its metadata and statistics. Rebuilt whenever the index changes."""
        docs = self.db.execute("SELECT id, name, path, chars, header, stats FROM docs ORDER BY name").fetchall()
        metadata, self.csv_fields = self._read_metadata([d[1] for d in docs])
        columns: list[str] = []
        records = []
        for doc, name, path, chars, header, stats in docs:
            fields: dict[str, Any] = {}
            for key, value in [*json.loads(header).items(), *metadata.get(name, {}).items()]:
                column = column_name(key)
                if column in LANGUAGE_COLUMNS:
                    column = "language"
                    value = LANGUAGE_NAMES.get(value.strip().lower(), value.strip())
                elif column in BASE_COLUMNS or column in ("id", "name"):
                    column = f"meta_{column}"
                if value not in ("", None):
                    fields[column] = value
                if column != "language" and column not in columns:
                    columns.append(column)
            stats = json.loads(stats)
            records.append({
                "id": doc, "name": name, **fields,
                "language": fields.get("language") or stats["detected_language"],
                "detected_language": stats["detected_language"],
                **{k: stats[k] for k in STATS},
                "chars": chars, "path": path,
            })
        for column in columns:  # numbers stored as numbers, so that `year < 1900` works
            having = [r for r in records if r.get(column) not in (None, "")]
            for kind in (int, float):
                try:
                    converted = [kind(r[column]) for r in having]
                except ValueError:
                    continue
                for record, value in zip(having, converted):
                    record[column] = value
                break
        self.columns = ["id", "name", *columns, *BASE_COLUMNS]
        self.metadata_columns = columns
        quoted = [f'"{c}"' for c in self.columns]
        with self.db:
            self.db.execute("DROP TABLE IF EXISTS texts")
            self.db.execute(f"CREATE TABLE texts ({quoted[0]} INTEGER PRIMARY KEY, {', '.join(quoted[1:])})")
            self.db.executemany(
                f"INSERT INTO texts ({', '.join(quoted)}) VALUES ({', '.join('?' * len(quoted))})",
                [[r.get(c) for c in self.columns] for r in records],
            )

    # ---------------------------------------------------------------- read-only SQL over the catalogue

    def _readonly_db(self) -> sqlite3.Connection:
        if self._readonly is None:
            db = sqlite3.connect(f"file:{self.index_path}?mode=ro", uri=True, check_same_thread=False)
            db.execute("PRAGMA query_only = ON")
            allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_FUNCTION, getattr(sqlite3, "SQLITE_RECURSIVE", -1)}

            def authorize(action: int, table: str | None, *_: object) -> int:
                if action in allowed or (action == sqlite3.SQLITE_READ and table == "texts"):
                    return sqlite3.SQLITE_OK
                return sqlite3.SQLITE_DENY

            db.set_authorizer(authorize)
            self._readonly = db
        return self._readonly

    def query(self, sql: str, limit: int = 100, seconds: float = 10.0) -> tuple[list[str], list[tuple[Any, ...]], bool]:
        """Run one read-only SQL statement, which may read the table `texts` and nothing else."""
        if not re.match(r"\s*(select|with)\b", sql, re.IGNORECASE):
            raise CorpusError("Only SELECT (or WITH … SELECT) queries over the table `texts` are allowed.")
        db = self._readonly_db()
        deadline = time.monotonic() + seconds
        db.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
        try:
            cursor = db.execute(sql)
            columns = [d[0] for d in cursor.description or []]
            rows = cursor.fetchmany(limit + 1)
        except (sqlite3.DatabaseError, sqlite3.Warning) as error:
            message = "the query took too long" if "interrupt" in str(error) else str(error)
            if any(s in message for s in ("not authorized", "prohibited", "vtable constructor failed")):
                message = "only the table `texts` can be read"
            raise CorpusError(f"SQL error: {message}") from None
        return columns, rows[:limit], len(rows) > limit

    def select(self, where: str | None = None) -> list[int] | None:
        """The ids of the texts that meet an SQL condition on `texts`; None (all texts) without a condition."""
        if not where or not where.strip():
            return None
        _, rows, _ = self.query(f"SELECT id FROM texts WHERE ({where})", limit=10**9)
        return [row[0] for row in rows]

    def texts(self, ids: Iterable[int] | None = None) -> list[dict[str, Any]]:
        cursor = self.db.execute("SELECT * FROM texts ORDER BY name")
        columns = [d[0] for d in cursor.description]
        wanted = set(ids) if ids is not None else None
        return [dict(zip(columns, row)) for row in cursor if wanted is None or row[0] in wanted]

    def find(self, name: str) -> dict[str, Any]:
        """A text by its name, or by a part of its name that picks out a single text."""
        rows = self.texts()
        exact = [r for r in rows if r["name"] == name]
        if exact:
            return exact[0]
        needle = name.strip().lower()
        found = [r for r in rows if needle and needle in r["name"].lower()]
        if len(found) == 1:
            return found[0]
        if not found:
            raise CorpusError(f"No text is called {name!r}. The catalogue has the names.")
        names = ", ".join(r["name"] for r in found[:10])
        raise CorpusError(f"{name!r} fits {len(found)} texts ({names}{', …' if len(found) > 10 else ''}): be more specific.")

    # ---------------------------------------------------------------- searching

    def matches(self, terms: list[str], ids: list[int] | None = None) -> Iterator[tuple[int, int, str, re.Match[str]]]:
        """Every occurrence of the terms in the texts `ids` (all texts if None), in corpus order, as
        (text id, start of its chunk, chunk text, match). A match's group name is `t<index of the term>`."""
        pattern = compile_terms(terms)
        self.db.execute("DELETE FROM sel")
        if ids is not None:
            self.db.executemany("INSERT OR IGNORE INTO sel(id) VALUES (?)", [(i,) for i in ids])
        subset = "" if ids is None else " AND c.doc IN (SELECT id FROM sel)"
        scan = f"SELECT c.doc, c.start, t.body FROM chunk c JOIN chunk_text t ON t.rowid = c.id WHERE 1{subset} ORDER BY c.id"
        cursor = None
        if self.fts:
            try:
                cursor = self.db.execute(
                    "SELECT c.doc, c.start, t.body FROM chunk_text t JOIN chunk c ON c.id = t.rowid "
                    f"WHERE chunk_text MATCH ?{subset} ORDER BY c.id",
                    (fts_query(terms),),
                )
            except sqlite3.OperationalError:  # something the full-text syntax can't express: read everything
                cursor = None
        for doc, start, body in cursor if cursor is not None else self.db.execute(scan):
            for match in pattern.finditer(body):
                yield doc, start, body, match

    def count(self, terms: list[str], categories: list[str] | None = None, ids: list[int] | None = None,
              segments: int | None = None) -> CountResult:
        """Occurrences of the terms per text, per category of term, and (optionally) per segment of each text."""
        counts = {
            t["id"]: TextCount(t["id"], t["name"], t["words"] or 0, t["chars"] or 0, segments=[0] * (segments or 0))
            for t in self.texts(ids)
        }
        by_term: Counter = Counter()
        for doc, start, _, match in self.matches(terms, ids):
            entry = counts[doc]
            index = int(str(match.lastgroup)[1:])
            entry.hits += 1
            by_term[terms[index]] += 1
            if categories:
                entry.by_category[categories[index]] += 1
            if segments:
                offset = start + match.start()
                entry.segments[min(segments - 1, offset * segments // max(1, entry.chars))] += 1
        return CountResult(list(counts.values()), by_term, segments)

    def concordance(self, terms: list[str], ids: list[int] | None = None, limit: int = 20,
                    context: int = 60) -> tuple[int, list[Line]]:
        """How often the terms occur, and up to `limit` occurrences spread evenly over all of them."""
        names = {t["id"]: (t["name"], t["chars"] or 1) for t in self.texts(ids)}
        rng = random.Random(0)
        total, sample = 0, []
        for doc, start, body, match in self.matches(terms, ids):
            total += 1
            slot = len(sample) if len(sample) < limit else rng.randrange(total)
            if slot >= limit:
                continue
            name, chars = names[doc]
            line = Line(
                name, start + match.start(), 100 * (start + match.start()) / chars,
                _flat(body[max(0, match.start() - context) : match.start()]), _flat(match.group()),
                _flat(body[match.end() : match.end() + context]),
            )
            if slot == len(sample):
                sample.append(line)
            else:
                sample[slot] = line
        return total, sorted(sample, key=lambda line: (line.text, line.offset))

    def passage(self, name: str, start: int | None = None, position: float | None = None, length: int = 800) -> dict[str, Any]:
        """A passage of a text, from a character offset or a position in percent, cut at word boundaries."""
        text = self.find(name)
        body, _ = load_text(self.root / text["path"])
        if start is None:
            start = int(len(body) * max(0.0, min(100.0, float(position or 0))) / 100)
        start = max(0, min(int(start), len(body)))
        back = start
        while back > 0 and start - back < 60 and not body[back - 1].isspace():
            back -= 1
        if back == 0 or body[back - 1].isspace():
            start = back
        end = min(len(body), start + max(1, length))
        cut = end
        while start < cut < len(body) and not body[cut].isspace():
            cut -= 1
        if cut > start:
            end = cut
        return {"name": text["name"], "start": start, "end": end, "chars": len(body), "text": body[start:end].strip()}

    # ---------------------------------------------------------------- description

    def describe(self) -> str:
        """A short account of the corpus, for the model's instructions."""
        texts = self.texts()
        words = sum(t["words"] or 0 for t in texts)
        lines = [f"{self.root}: {_texts(len(texts))}, {words:,} words."]
        by_language: dict[str, list[int]] = {}
        for t in texts:
            by_language.setdefault(t["language"] or "unknown", []).append(t["words"] or 0)
        if by_language:
            parts = sorted(by_language.items(), key=lambda kv: -sum(kv[1]))
            lines.append("Languages: " + "; ".join(f"{lang} {_texts(len(w))} ({sum(w):,} words)" for lang, w in parts) + ".")
        if self.metadata_path and self.csv_fields:
            lines.append(f"Metadata from {self.metadata_path.name}: {', '.join(self.csv_fields)}.")
        lines.append("Columns of `texts`: " + ", ".join(self.columns) + ".")
        for column in self.metadata_columns[:12]:
            values = [t[column] for t in texts if t.get(column) not in (None, "")]
            if not values:
                continue
            if all(isinstance(v, (int, float)) for v in values):
                lines.append(f"- {column}: {min(values)}–{max(values)} ({_texts(len(values))})")
            else:
                common = Counter(map(str, values)).most_common(4)
                examples = "; ".join(v if len(v) <= 40 else v[:39] + "…" for v, _ in common)
                distinct = len(set(map(str, values)))
                lines.append(f"- {column}: {distinct:,} distinct value{'s' if distinct != 1 else ''} in {_texts(len(values))}, e.g. {examples}")
        return "\n".join(lines)

    def export_catalogue(self, target: Path) -> None:
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.columns)
            writer.writeheader()
            writer.writerows(self.texts())
