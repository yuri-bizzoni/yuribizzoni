"""The tools Happiness uses on your corpus. They all run on your machine; what they return is what
Claude sees, and the corpus text in it is held to a passage budget."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any, Callable, Iterable

from anthropic.lib.tools import ToolError
from anthropic.tools.memory import BetaLocalFilesystemMemoryTool

from .corpus import Corpus, CorpusError, CountResult, term_pattern
from .lab import Lab
from .resources import Lexicon, discover_lexicons, discover_skills, read_lexicon, read_skill
from .sandbox import Sandbox, clip
from .workspace import Workspace

Notify = Callable[[str, str], None]

MAX_READ = 1500
DEFAULT_READ = 800
MAX_LINES = 40
MAX_CONTEXT = 100
MAX_ROWS = 100

WEB_TOOLS: list[dict[str, Any]] = [
    {"type": "web_search_20260209", "name": "web_search", "max_uses": 8},
    {"type": "web_fetch_20260209", "name": "web_fetch", "max_uses": 8},
]
EDITOR: dict[str, Any] = {"type": "text_editor_20250728", "name": "str_replace_based_edit_tool"}


def _tool(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "eager_input_streaming": True,
        "input_schema": {"type": "object", "properties": properties, "required": required, "additionalProperties": False},
    }


TERMS = {
    "type": "array",
    "items": {"type": "string"},
    "description": "Words (gioia), phrases (la gioia) or prefixes (felic*), matched as whole words, ignoring case.",
}
LEXICON = {
    "type": "string",
    "description": "Instead of terms: a lexicon by name (see Lexicons in your instructions), or a CSV path in the lab folder.",
}
WHERE = {
    "type": "string",
    "description": "An SQL condition on the columns of `texts` choosing the texts, e.g. language = 'it' AND year < 1900. "
    "Leave it out for the whole corpus.",
}
CATALOGUE = _tool(
    "catalogue",
    "Query the corpus catalogue with SQL (SQLite). The table `texts` has one row per text with its metadata and statistics "
    "(the columns are listed in your instructions). Use it to see what the corpus holds, to choose subsets, and to compare "
    f"texts by metadata or statistics. Returns at most {MAX_ROWS} rows and no corpus text.",
    {"sql": {"type": "string", "description": "One SELECT statement over `texts`."}},
    ["sql"],
)
COUNT = _tool(
    "count",
    "Count words, phrases, prefixes or a lexicon in the corpus or part of it: hits per text, normalised per 10,000 words, "
    "and optionally per segment of each text (for arcs across a narrative) and per group of a catalogue column. Saves the "
    "full table (one row per text, with its metadata) in the lab for run_python, and returns a summary. No corpus text.",
    {
        "terms": TERMS,
        "lexicon": LEXICON,
        "where": WHERE,
        "segments": {"type": "integer", "description": "Cut each text into this many equal parts (2–100) and count in each; 10 gives deciles."},
        "group_by": {"type": "string", "description": "A column of `texts` to summarise by, such as author or year."},
    },
    [],
)
CONCORDANCE = _tool(
    "concordance",
    "Show occurrences in context (keyword in context), sampled evenly from all the matches: to see how words are used and "
    "whether a count measures what you think. Each line draws on the passage budget.",
    {
        "terms": TERMS,
        "lexicon": LEXICON,
        "where": WHERE,
        "limit": {"type": "integer", "description": f"How many lines, at most {MAX_LINES} (default 15)."},
        "context": {"type": "integer", "description": f"Characters of context on each side, at most {MAX_CONTEXT} (default 60)."},
    },
    [],
)
READ = _tool(
    "read",
    f"Read a short passage of a text, from a character offset (concordance shows them) or a position in percent. At most "
    f"{MAX_READ} characters, drawn from the passage budget: read what you need to check or to quote, no more.",
    {
        "text": {"type": "string", "description": "The text's name, or a part of it that picks out one text."},
        "start": {"type": "integer", "description": "The character offset where the passage starts."},
        "position": {"type": "number", "description": "Or where it starts, in percent of the text (0–100)."},
        "length": {"type": "integer", "description": f"How many characters (default {DEFAULT_READ}, at most {MAX_READ})."},
    },
    ["text"],
)
RUN_PYTHON = _tool(
    "run_python",
    "Run a Python script for what the other tools can't do: statistics, models, plots, features of your own. It runs on the "
    "user's machine (see Your sandbox) inside this conversation's lab folder, where count's tables are in tables/. "
    "`from happiness_corpus import catalogue, text, words, sentences` gives the catalogue as a DataFrame and texts by name. "
    "Save figures in figures/ and tables in tables/. Print numbers and short summaries, never corpus text; the output is cut "
    "at 6,000 characters.",
    {
        "title": {"type": "string", "description": "A few words on what the script does; it names the file."},
        "code": {"type": "string", "description": "The whole script."},
    },
    ["title", "code"],
)
LOAD_SKILL = _tool(
    "load_skill",
    "Load the full instructions of one of your skills. Call it before any task a skill covers (see Your skills), then follow them.",
    {
        "name": {"type": "string", "description": "The skill's name."},
        "file": {"type": "string", "description": "Another file in the skill's folder, if its instructions point to one."},
    },
    ["name"],
)


class InvalidInput(Exception):
    """A tool call whose arguments don't fit the tool's schema."""


class PassageBudget:
    """How many characters of corpus text may still go to the API in this conversation."""

    def __init__(self, limit: int):
        self.limit = limit
        self.used = 0

    @property
    def left(self) -> int:
        return max(0, self.limit - self.used)


def _text_arg(args: dict[str, Any], key: str, required: bool = False) -> str | None:
    value = args.get(key)
    if value is None and not required:
        return None
    if not isinstance(value, str) or (required and not value.strip()):
        raise InvalidInput(args)
    return value.strip()


def _int_arg(args: dict[str, Any], key: str) -> int | None:
    value = args.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
        raise InvalidInput(args)
    return int(value)


def _check_keys(args: dict[str, Any], allowed: Iterable[str]) -> None:
    if set(args) - set(allowed):
        raise InvalidInput(args)


def _rate(hits: int, words: int) -> float:
    return 10_000 * hits / words if words else 0.0


def _fmt(value: float) -> str:
    return f"{value:,.2f}" if abs(value) < 100 else f"{value:,.1f}"


def describe(name: str, args: object) -> str:
    """A one-line account of a tool call, for the terminal."""
    a = args if isinstance(args, dict) else {}
    what = a.get("lexicon") or ", ".join(str(t) for t in (a.get("terms") or [])[:5])
    where = f" where {a['where']}" if a.get("where") else ""
    if name == "catalogue":
        sql = " ".join(str(a.get("sql", "")).split())
        return f"catalogue: {sql[:90]}{'…' if len(sql) > 90 else ''}"
    if name == "count":
        extra = (f", {a['segments']} segments" if a.get("segments") else "") + (f", by {a['group_by']}" if a.get("group_by") else "")
        return f"counting {what}{where}{extra}"
    if name == "concordance":
        return f"concordance: {what}{where}"
    if name == "read":
        at = f"{a['position']}%" if a.get("position") is not None else f"character {a.get('start', 0)}"
        return f"reading {a.get('text', '')} at {at}"
    if name == "run_python":
        return f"running a script: {a.get('title', '')}"
    if name == "memory":
        return f"memory: {a.get('command', '')} {a.get('path', '')}".strip()
    if name == "str_replace_based_edit_tool":
        verbs = {"view": "looking at", "create": "writing", "str_replace": "editing", "insert": "editing"}
        return f"{verbs.get(str(a.get('command')), 'using')} {a.get('path', 'the lab')}"
    if name == "load_skill":
        return f"loading the {a.get('name', '')} skill"
    if name == "web_search":
        return f"searching the web for “{a.get('query', '')}”"
    if name == "web_fetch":
        return f"reading {a.get('url', 'a web page')}"
    return name.replace("_", " ")


def _ignore(kind: str, text: str) -> None:
    pass


class Toolbox:
    """Everything Happiness can call on your machine, and the definitions the model sees."""

    def __init__(
        self,
        *,
        corpus: Corpus,
        lab: Lab,
        sandbox: Sandbox,
        memory_dir: Path,
        skill_folders: Iterable[Path],
        lexicon_folders: Iterable[Path],
        web: bool = False,
        passages: int = 20_000,
    ):
        self.corpus = corpus
        self.lab = lab
        self.sandbox = sandbox
        self.memory = BetaLocalFilesystemMemoryTool(base_path=str(memory_dir))
        self.workspace = Workspace(lab.root)
        self.skills = discover_skills(skill_folders)
        self.lexicon_folders = list(lexicon_folders)
        self.web = web
        self.budget = PassageBudget(passages)

    def new_conversation(self) -> None:
        self.budget = PassageBudget(self.budget.limit)

    def definitions(self) -> list[dict[str, Any]]:
        """The tool list for a conversation; it stays the same from the first request to the last."""
        tools = [*WEB_TOOLS] if self.web else []
        return tools + [CATALOGUE, COUNT, CONCORDANCE, READ, RUN_PYTHON, dict(self.memory.to_dict()), EDITOR, LOAD_SKILL]

    def lexicons(self) -> dict[str, Lexicon]:
        found = {}
        for name, path in discover_lexicons(self.lexicon_folders).items():
            try:
                found[name] = read_lexicon(path, name)
            except (ToolError, OSError, UnicodeDecodeError):
                continue
        return found

    # ---------------------------------------------------------------- dispatch

    def run(self, name: str, args: object, notify: Notify = _ignore) -> tuple[Any, bool]:
        """Run one tool call and note it in the lab notebook. Returns the result and whether it is an error."""
        handlers: dict[str, Callable[[dict[str, Any], Notify], Any]] = {
            "catalogue": self._catalogue,
            "count": self._count,
            "concordance": self._concordance,
            "read": self._read,
            "run_python": self._run_python,
            "memory": lambda a, _: self.memory.call(a),
            "str_replace_based_edit_tool": lambda a, _: self.workspace(a),
            "load_skill": self._load_skill,
        }
        handler = handlers.get(name)
        if handler is None:
            return f"There is no tool called {name}.", True
        try:
            if not isinstance(args, dict):
                raise InvalidInput(args)
            return handler(args, notify), False
        except InvalidInput as error:
            # Arguments that don't fit the schema (possible with eager input streaming): give them back.
            self.lab.note(f"- **{name}**: malformed arguments, returned to Claude")
            return json.dumps({"INVALID_JSON": json.dumps(error.args[0], ensure_ascii=False)}), True
        except (ToolError, CorpusError) as error:
            content = error.content if isinstance(error, ToolError) else str(error)
            self.lab.note(f"- **{name}** failed: {str(content).splitlines()[0][:300] if content else ''}")
            return content, True
        except Exception as error:  # a failed tool is reported to the model, which can adapt
            self.lab.note(f"- **{name}** failed: {type(error).__name__}: {error}")
            return f"{type(error).__name__}: {error}", True

    # ---------------------------------------------------------------- shared arguments

    def _terms(self, args: dict[str, Any]) -> tuple[list[str], list[str] | None, str]:
        """The terms to look for, their categories if they come from a lexicon, and a label for them."""
        lexicon_name = _text_arg(args, "lexicon")
        if lexicon_name:
            lexicon = self.lexicons().get(lexicon_name)
            if lexicon is None:
                path = self.workspace.resolve(lexicon_name)
                if not path.is_file():
                    names = ", ".join(sorted(self.lexicons())) or "none yet"
                    raise ToolError(f"No lexicon called {lexicon_name!r}. Available: {names}.")
                lexicon = read_lexicon(path)
            for term in lexicon.terms:
                term_pattern(term)
            return lexicon.terms, lexicon.categories, f"the {lexicon.name} lexicon"
        terms = args.get("terms")
        if not isinstance(terms, list) or not terms or not all(isinstance(t, str) and t.strip() for t in terms):
            if terms is None:
                raise ToolError("Give some terms, or a lexicon.")
            raise InvalidInput(args)
        terms = list(dict.fromkeys(t.strip() for t in terms))
        for term in terms:
            term_pattern(term)
        label = ", ".join(terms[:6]) + (f" and {len(terms) - 6} more" if len(terms) > 6 else "")
        return terms, None, label

    # ---------------------------------------------------------------- the corpus tools

    def _catalogue(self, args: dict[str, Any], notify: Notify) -> str:
        _check_keys(args, ["sql"])
        sql = _text_arg(args, "sql", required=True) or ""
        columns, rows, more = self.corpus.query(sql, limit=MAX_ROWS)
        cell = lambda v: "" if v is None else (str(v) if len(str(v)) <= 60 else str(v)[:59] + "…")  # noqa: E731
        table = "\n".join(["\t".join(columns), *("\t".join(cell(v) for v in row) for row in rows)])
        self.lab.note(f"- **catalogue** `{' '.join(sql.split())}` → {len(rows)}{'+' if more else ''} rows")
        tail = f"\n(first {MAX_ROWS} rows; there are more)" if more else ""
        return (table + tail) if rows else "No rows."

    def _count(self, args: dict[str, Any], notify: Notify) -> str:
        _check_keys(args, ["terms", "lexicon", "where", "segments", "group_by"])
        terms, categories, label = self._terms(args)
        where = _text_arg(args, "where")
        segments = _int_arg(args, "segments")
        if segments is not None and not 2 <= segments <= 100:
            raise ToolError("segments must be between 2 and 100.")
        group_by = _text_arg(args, "group_by")
        if group_by and group_by not in self.corpus.columns:
            raise ToolError(f"There is no column {group_by!r}; the columns are {', '.join(self.corpus.columns)}.")
        ids = self.corpus.select(where)
        result = self.corpus.count(terms, categories, ids, segments)
        rows = {t["id"]: t for t in self.corpus.texts(ids)}
        table = self.lab.path("tables", f"count {label}", ".csv")
        self._write_count(table, result, rows, categories)
        summary = self._summarise(result, rows, categories, label, where, group_by)
        summary += f"\nTable: {self.lab.relative(table)}"
        self.lab.note(
            f"- **count** {label}" + (f" where `{where}`" if where else "") + (f", {segments} segments" if segments else "")
            + (f", by {group_by}" if group_by else "") + f" → [{self.lab.relative(table)}]({self.lab.relative(table)})\n"
            + "\n".join(f"  > {line}" for line in summary.splitlines()[:-1])
        )
        return summary

    def _write_count(self, path: Path, result: CountResult, rows: dict[int, dict[str, Any]], categories: list[str] | None) -> None:
        keep = [c for c in self.corpus.columns if c not in ("id", "path", "chars")]
        groups = list(dict.fromkeys(categories)) if categories else []
        segments = result.segments or 0
        header = [*keep, "hits", "per_10k", *(f"hits_{g}" for g in groups), *(f"seg_{i + 1:02d}" for i in range(segments))]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            for entry in sorted(result.texts, key=lambda e: e.name):
                row = rows[entry.id]
                per_segment = entry.words / segments if segments else 0
                writer.writerow([
                    *(row.get(c) for c in keep), entry.hits, round(_rate(entry.hits, entry.words), 4),
                    *(entry.by_category.get(g, 0) for g in groups),
                    *(round(_rate(n, per_segment), 4) if per_segment else 0 for n in entry.segments),
                ])

    def _label(self, row: dict[str, Any]) -> str:
        details = [str(row[c]) for c in ("title", "author", "year") if row.get(c) not in (None, "")]
        return row["name"] + (f" ({', '.join(details)})" if details else "")

    def _summarise(self, result: CountResult, rows: dict[int, dict[str, Any]], categories: list[str] | None,
                   label: str, where: str | None, group_by: str | None) -> str:
        texts = [e for e in result.texts if e.words]
        if not texts:
            return f"No texts{' match ' + where if where else ''}."
        words = sum(e.words for e in texts)
        hits = sum(e.hits for e in texts)
        rates = sorted(((_rate(e.hits, e.words), e) for e in texts), key=lambda pair: (pair[0], pair[1].name))
        lines = [
            f"Counted {label} in {len(texts):,} texts ({words:,} words){' where ' + where if where else ''}: "
            f"{hits:,} hits; {sum(1 for e in texts if e.hits):,} texts have at least one.",
            f"Per 10,000 words: {_fmt(_rate(hits, words))} overall; per text, median {_fmt(statistics.median(r for r, _ in rates))}, "
            f"range {_fmt(rates[0][0])}–{_fmt(rates[-1][0])}.",
        ]
        if categories:
            totals: dict[str, int] = {}
            for entry in texts:
                for category, n in entry.by_category.items():
                    totals[category] = totals.get(category, 0) + n
            ordered = sorted(totals.items(), key=lambda kv: -kv[1])
            lines.append("By category (hits): " + " · ".join(f"{c} {n:,}" for c, n in ordered))
        if len(result.by_term) > 1:
            top = result.by_term.most_common(12)
            lines.append("Commonest terms: " + " · ".join(f"{t} {n:,}" for t, n in top) + (" · …" if len(result.by_term) > 12 else ""))
        if len(texts) > 1:
            lines.append("Highest rates: " + " · ".join(f"{self._label(rows[e.id])} {_fmt(r)}" for r, e in rates[::-1][:8]))
        if len(texts) > 8:
            lines.append("Lowest rates: " + " · ".join(f"{self._label(rows[e.id])} {_fmt(r)}" for r, e in rates[:3]))
        if result.segments:
            means = [statistics.fmean(_rate(e.segments[i], e.words / result.segments) for e in texts) for i in range(result.segments)]
            lines.append(f"Per segment ({result.segments} equal parts of each text; mean of the per-text rates per 10,000 words): "
                         + " · ".join(f"{i + 1}: {_fmt(m)}" for i, m in enumerate(means)))
        if group_by:
            groups: dict[Any, list[Any]] = {}
            for entry in texts:
                groups.setdefault(rows[entry.id].get(group_by), []).append(entry)
            order = sorted(groups, key=lambda k: (k is None, k if isinstance(k, (int, float)) else str(k)))
            if len(order) > 30:
                order = sorted(groups, key=lambda k: -len(groups[k]))[:30]
            parts = []
            for key in order:
                members = groups[key]
                pooled = _rate(sum(e.hits for e in members), sum(e.words for e in members))
                mean = statistics.fmean(_rate(e.hits, e.words) for e in members)
                parts.append(f"{key}: {_fmt(pooled)} pooled, {_fmt(mean)} mean ({len(members)} text{'' if len(members) == 1 else 's'})")
            more = f" (the 30 largest of {len(groups)} groups)" if len(groups) > 30 else ""
            lines.append(f"By {group_by}{more}: " + " · ".join(parts))
        return "\n".join(lines)

    def _concordance(self, args: dict[str, Any], notify: Notify) -> str:
        _check_keys(args, ["terms", "lexicon", "where", "limit", "context"])
        terms, _, label = self._terms(args)
        where = _text_arg(args, "where")
        limit = max(1, min(MAX_LINES, _int_arg(args, "limit") or 15))
        context = max(10, min(MAX_CONTEXT, _int_arg(args, "context") or 60))
        ids = self.corpus.select(where)
        wanted = limit if self.budget.left else 0
        total, lines = self.corpus.concordance(terms, ids, wanted, context)
        shown, used = [], 0
        for line in lines:
            cost = len(line.left) + len(line.match) + len(line.right)
            if cost > self.budget.left - used:
                break
            used += cost
            shown.append(f"{line.text} {line.position:.0f}% [{line.offset}]: …{line.left} «{line.match}» {line.right}…")
        self.budget.used += used
        self.lab.note(
            f"- **concordance** {label}" + (f" where `{where}`" if where else "")
            + f" → {total:,} matches, {len(shown)} lines shown ({used:,} characters sent)"
        )
        head = f"{total:,} matches for {label}" + (f" where {where}" if where else "") + "."
        if not shown:
            reason = "The passage budget is used up, so no lines are shown." if total else ""
            return f"{head} {reason}".strip()
        return (
            f"{head} {len(shown)} lines, spread over all of them ({used:,} characters; "
            f"{self.budget.left:,} left in the passage budget):\n" + "\n".join(shown)
        )

    def _read(self, args: dict[str, Any], notify: Notify) -> str:
        _check_keys(args, ["text", "start", "position", "length"])
        name = _text_arg(args, "text", required=True) or ""
        start = _int_arg(args, "start")
        position = args.get("position")
        if position is not None and (isinstance(position, bool) or not isinstance(position, (int, float))):
            raise InvalidInput(args)
        length = max(50, min(MAX_READ, _int_arg(args, "length") or DEFAULT_READ))
        if not self.budget.left:
            raise ToolError(
                f"The passage budget for this conversation ({self.budget.limit:,} characters) is used up. Work from counts "
                "and tables, or tell the user: they can start a new conversation or raise the budget with --passages."
            )
        passage = self.corpus.passage(name, start, position, min(length, self.budget.left))
        self.budget.used += len(passage["text"])
        where = 100 * passage["start"] / max(1, passage["chars"])
        self.lab.note(
            f"- **read** {passage['name']}, characters {passage['start']:,}–{passage['end']:,} ({where:.0f}%; "
            f"{len(passage['text']):,} characters sent)"
        )
        return (
            f"{passage['name']} · characters {passage['start']:,}–{passage['end']:,} of {passage['chars']:,} ({where:.0f}%) · "
            f"{self.budget.left:,} characters left in the passage budget\n\n{passage['text']}"
        )

    def _run_python(self, args: dict[str, Any], notify: Notify) -> str:
        _check_keys(args, ["title", "code"])
        title = _text_arg(args, "title", required=True) or "analysis"
        code = args.get("code")
        if not isinstance(code, str) or not code.strip():
            raise InvalidInput(args)
        session = self.lab.session
        if session is None:
            raise ToolError("No lab folder is open.")
        script = self.lab.path("scripts", title, ".py")
        script.write_text(code, encoding="utf-8")

        def snapshot() -> dict[Path, float]:
            return {p: p.stat().st_mtime for p in session.rglob("*") if p.is_file() and "scripts" not in p.relative_to(session).parts}

        before = snapshot()
        run = self.sandbox.run(script, session, notify)
        after = snapshot()
        written = sorted(self.lab.relative(p) for p, mtime in after.items() if before.get(p) != mtime and "__pycache__" not in p.parts)
        status = f"stopped after {run.seconds:.0f} s (too long)" if run.timed_out else f"exit code {run.exit_code}, {run.seconds:.1f} s"
        output = clip(run.output)
        report = (
            f"{self.lab.relative(script)} ran ({status})."
            + (f"\nFiles written: {', '.join(written)}" if written else "")
            + (f"\nOutput:\n{output}" if output else "\nNo output.")
        )
        self.lab.note(
            f"- **run_python** [{self.lab.relative(script)}]({self.lab.relative(script)}) ({status})"
            + (f"; wrote {', '.join(written)}" if written else "")
            + (f"\n\n```text\n{clip(run.output, 2000)}\n```" if output else "")
        )
        if run.timed_out or run.exit_code != 0:
            raise ToolError(report)
        return report

    def _load_skill(self, args: dict[str, Any], notify: Notify) -> str:
        _check_keys(args, ["name", "file"])
        name = _text_arg(args, "name", required=True) or ""
        file = _text_arg(args, "file") or "SKILL.md"
        text = read_skill(self.skills, name, file)
        self.lab.note(f"- **load_skill** {name}")
        return text
