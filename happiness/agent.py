"""The agent itself: settings, instructions, and the loop that talks to Claude and runs the tools."""

from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import anthropic

from .corpus import Corpus
from .lab import Lab
from .sandbox import Sandbox, ScriptApprover, ask_before_running, make_sandbox
from .tools import Toolbox, describe

DEFAULT_MODEL = "claude-opus-5-5"
EFFORTS = ("low", "medium", "high", "xhigh", "max")
SANDBOXES = ("auto", "docker", "local")
FALLBACK_BETA = "server-side-fallback-2026-07-01"
PACKAGE = Path(__file__).resolve().parent
BUNDLED_SKILLS = PACKAGE / "skills"
BUNDLED_LEXICONS = PACKAGE / "lexicons"

Handler = Callable[[str, str], None]
"""Receives progress while the agent works: ("text", chunk), ("tool", what it is doing),
("thinking", ""), or ("notice", something the user should know)."""


def _ignore(kind: str, text: str) -> None:
    pass


def _flag(value: str | None, default: bool) -> bool:
    if value is None or not value.strip():
        return default
    return value.strip().lower() not in ("0", "false", "no", "off")


def _path(value: str | None) -> Path | None:
    return Path(value).expanduser() if value else None


@dataclass
class Settings:
    corpus: Path | None = None
    metadata: Path | None = None
    model: str = DEFAULT_MODEL
    effort: str = "medium"
    home: Path = field(default_factory=lambda: Path("~/.happiness").expanduser())
    lab: Path | None = None
    sandbox: str = "auto"
    passages: int = 20_000
    web: bool = False
    max_tokens: int = 64_000
    max_steps: int = 60

    @classmethod
    def from_env(cls) -> Settings:
        env = os.environ
        return cls(
            corpus=_path(env.get("HAPPINESS_CORPUS")),
            metadata=_path(env.get("HAPPINESS_METADATA")),
            model=env.get("HAPPINESS_MODEL") or DEFAULT_MODEL,
            effort=env.get("HAPPINESS_EFFORT") or "medium",
            home=_path(env.get("HAPPINESS_HOME")) or Path("~/.happiness").expanduser(),
            lab=_path(env.get("HAPPINESS_LAB")),
            sandbox=env.get("HAPPINESS_SANDBOX") or "auto",
            passages=int(env.get("HAPPINESS_PASSAGES") or 20_000),
            web=_flag(env.get("HAPPINESS_WEB"), False),
        )

    def check(self) -> None:
        if self.corpus is None:
            raise ValueError("no corpus: give the folder of your texts with --corpus (or HAPPINESS_CORPUS)")
        if self.effort not in EFFORTS:
            raise ValueError(f"effort must be one of {', '.join(EFFORTS)}, not {self.effort!r}")
        if self.sandbox not in SANDBOXES:
            raise ValueError(f"sandbox must be one of {', '.join(SANDBOXES)}, not {self.sandbox!r}")
        if self.passages < 0:
            raise ValueError("the passage budget can't be negative")

    @property
    def lab_dir(self) -> Path:
        return self.lab or self.home / "lab"

    @property
    def index_dir(self) -> Path:
        return self.home / "index"

    def open_corpus(self) -> Corpus:
        """The corpus and its index (made or brought up to date by `Corpus.update`)."""
        assert self.corpus is not None
        return Corpus(self.corpus, self.index_dir, self.metadata)


def system_prompt(toolbox: Toolbox, now: dt.datetime) -> str:
    lexicons = toolbox.lexicons()
    skills = sorted(toolbox.skills.values(), key=lambda s: s.name)
    web = (
        ["- web_search and web_fetch: facts about authors, works, editions and methods. Never put corpus text in a query."]
        if toolbox.web
        else []
    )
    return "\n".join(
        [
            "You are Happiness, a research assistant for work on a corpus of texts. You run on the user's own computer, "
            "next to their corpus, and you study it through the tools below, which all run there. The user studies "
            "literature with computational methods: treat them as a colleague.",
            "",
            "## The corpus",
            toolbox.corpus.describe(),
            "",
            "## Your tools",
            "- catalogue: SQL over the table `texts` (one row per text, with metadata and statistics). Use it to see what "
            "is there and to choose subsets; every `where` argument is a condition on this table.",
            "- count: occurrences of words, phrases, prefixes or a lexicon per text, normalised per 10,000 words; per "
            "segment of each text (arcs) and per group when you ask. It saves the full table in the lab for run_python.",
            "- concordance: occurrences in context, sampled evenly from all the matches.",
            "- read: a short passage, from an offset or a position.",
            f"- run_python: a script for everything else (statistics, models, plots). Your sandbox: {toolbox.sandbox.describe()} "
            "Scripts start in this conversation's lab folder, where count's tables are; `from happiness_corpus import "
            "catalogue, text, words, sentences` loads the catalogue and the texts. Save figures in figures/ and tables in "
            "tables/; you can look at your figures with the editor tool.",
            "- str_replace_based_edit_tool: the lab folder, with this conversation's folder and the earlier ones (their "
            "notebook.md, tables, scripts and figures), and lexicons/, where you can save lexicons to reuse.",
            "- memory: notes that last between conversations: what you've learned about this corpus, the user's "
            "preferences, findings worth keeping. Never store passages there.",
            "- load_skill: loads one of your skills; load a skill before any task it covers.",
            *web,
            "",
            "## Passages and the budget",
            "Whatever a tool returns goes to Claude's API. The user allows short passages of the corpus to leave their "
            "machine, but not many: concordance lines and read passages draw on a budget of "
            f"{toolbox.budget.limit:,} characters per conversation, and each result says how much is left. Work from "
            "counts and tables first, and spend passages on checking and quoting what matters. In run_python, print "
            "numbers, never passages.",
            "",
            "## Lexicons",
            *(f"- {lex.name}: {lex.summary()}" for lex in lexicons.values()),
            "A lexicon is a CSV of `category,term` rows; a term ending in * is a prefix. Write new ones to lexicons/ "
            "(as lexicons/<name>.csv) when a question needs them, and check them with concordance before trusting "
            "their counts.",
            "",
            "## Your skills",
            *([f"- {s.name}: {s.description}" for s in skills] or ["(none installed)"]),
            "",
            "## How to work",
            "- Measure rather than guess: every number you report comes from a tool result in this conversation, and "
            "you say which table, script or figure it comes from.",
            "- Check your instruments: before trusting a count, look at some concordance lines, and say what the terms "
            "catch that they shouldn't (other senses, names, negation, irony) and what they miss (inflections, "
            "spelling variants, OCR errors).",
            "- Compare like with like: normalise by length, look at how many texts carry an effect rather than only at "
            "averages, and watch for one author or one long book dominating a group.",
            "- Keep methods simple and transparent, and say what they cannot show. Keep exploratory patterns apart from "
            "tested differences; give effect sizes and uncertainty, not only p-values.",
            "- For a question that takes several steps, say in one line what you'll do, then do it, and end with what "
            "you found, how sure you are, and which files hold the details.",
            "- Be warm and direct. Answer in the language the user writes in.",
            "",
            "## Happiness",
            "You carry a curated lexicon of words for happiness and its neighbours across languages (the "
            "happiness-lexicon skill) and the felicita-it lexicon of Italian forms in six families. Load the skill when "
            "the user asks about such words or wants a happiness lexicon for another language. Its cross-index maps the "
            "stages of a research project to the kind of happiness each runs on, a gentle diagnostic when a project "
            "stalls. Don't push it into unrelated tasks.",
            "",
            f"Today is {now:%A} {now.day} {now:%B %Y} (time zone: {now.tzname() or 'local'}).",
        ]
    )


def echoable(content: list[Any]) -> list[Any]:
    """The blocks of a reply to send back on the next request.

    When the model declined partway through and a fallback model took over (server-side fallback),
    only the text the first model wrote before the switch goes back; its other blocks don't.
    """
    switches = [i for i, block in enumerate(content) if block.type == "fallback"]
    if not switches:
        return list(content)
    last = switches[-1]
    return [block for block in content[:last] if block.type == "text"] + list(content[last + 1 :])


def reply_text(content: list[Any]) -> str:
    return "".join(block.text for block in content if block.type == "text")


def describe_error(error: BaseException) -> str | None:
    """Say in plain words what went wrong talking to Claude, or return None if `error` is something else."""
    if isinstance(error, TypeError) and "authentication method" in str(error):
        return "No Claude API key found. Set ANTHROPIC_API_KEY (create a key at https://console.anthropic.com)."
    if isinstance(error, anthropic.AuthenticationError):
        return "Claude rejected the API key. Check ANTHROPIC_API_KEY."
    if isinstance(error, anthropic.PermissionDeniedError):
        return f"This API key isn't allowed to do that: {error.message}"
    if isinstance(error, anthropic.NotFoundError):
        return f"Not found: {error.message} (is the model name right?)"
    if isinstance(error, anthropic.RateLimitError):
        return "The Claude API is rate-limiting this key. Wait a little and try again."
    if isinstance(error, anthropic.BadRequestError) and "web" in error.message.lower():
        return (
            f"The Claude API refused the request: {error.message}\nIf web search isn't enabled for your "
            "organization, enable it in the Claude Console or run without --web."
        )
    if isinstance(error, anthropic.APIStatusError):
        return f"The Claude API returned an error ({error.status_code}): {error.message}"
    if isinstance(error, anthropic.APIConnectionError):
        return "Could not reach the Claude API. Check the network connection."
    if isinstance(error, anthropic.AnthropicError):
        return str(error)
    return None


def refusal_note(message: Any) -> str:
    category = getattr(getattr(message, "stop_details", None), "category", None)
    reason = f" (category: {category})" if category else ""
    return (
        f"That request was declined by Claude's safety checks{reason}. "
        "I left it out of our conversation, so you can rephrase and go on."
    )


class Agent:
    """One conversation with Happiness about a corpus. Give each conversation its own Agent, or call
    new_conversation() between them."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: anthropic.Anthropic | None = None,
        corpus: Corpus | None = None,
        sandbox: Sandbox | None = None,
        approve: ScriptApprover = ask_before_running,
    ):
        self.settings = settings or Settings.from_env()
        self.settings.check()
        self.client = client or anthropic.Anthropic()
        if corpus is None:
            corpus = self.settings.open_corpus()
            corpus.update()
        self.corpus = corpus
        self.lab = Lab(self.settings.lab_dir, corpus, self.settings.model)
        self.toolbox = Toolbox(
            corpus=corpus,
            lab=self.lab,
            sandbox=sandbox or make_sandbox(self.settings.sandbox, corpus.root, approve),
            memory_dir=self.settings.home / "memory",
            skill_folders=[BUNDLED_SKILLS, self.settings.home / "skills"],
            lexicon_folders=[BUNDLED_LEXICONS, self.lab.root / "lexicons"],
            web=self.settings.web,
            passages=self.settings.passages,
        )
        self.new_conversation()

    def new_conversation(self) -> None:
        """Start afresh. The instructions and tools are fixed here and never change mid-conversation,
        and the history only ever grows: that keeps the prompt cache warm and every earlier turn valid."""
        self.lab.new_conversation()
        self.toolbox.new_conversation()
        self.system = system_prompt(self.toolbox, dt.datetime.now().astimezone())
        self.tools = self.toolbox.definitions()
        self.messages: list[dict[str, Any]] = []

    def request(self) -> dict[str, Any]:
        return {
            "model": self.settings.model,
            "max_tokens": self.settings.max_tokens,
            "system": [{"type": "text", "text": self.system, "cache_control": {"type": "ephemeral"}}],
            "messages": self.messages,
            "tools": self.tools,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": self.settings.effort},
            "cache_control": {"type": "ephemeral"},  # also cache the growing conversation
            "betas": [FALLBACK_BETA],
            "fallbacks": "default",  # if the model declines on safety grounds, a fallback model answers
        }

    def ask(self, text: str, on_event: Handler = _ignore) -> str:
        """Send one message, let Happiness work until it is done, and return its reply."""
        self.lab.begin(text)
        self.lab.note(f"\n## {dt.datetime.now():%H:%M} · You asked\n\n{text}\n")

        def handler(kind: str, payload: str) -> None:
            if kind == "notice":
                self.lab.note(f"> {payload}")
            on_event(kind, payload)

        start = len(self.messages)
        self.messages.append({"role": "user", "content": text})
        try:
            answer = self._work(start, handler)
        except BaseException as error:
            # An error or Ctrl-C: drop this exchange, so the conversation is as it was and can go on.
            del self.messages[start:]
            self.lab.note(f"> Stopped: {describe_error(error) or type(error).__name__}")
            raise
        budget = self.toolbox.budget
        self.lab.note(
            f"\n## {dt.datetime.now():%H:%M} · Happiness answered\n\n{answer or '(no answer)'}\n\n"
            f"_Passages sent so far: {budget.used:,} of {budget.limit:,} characters._"
        )
        return answer

    def _work(self, start: int, on_event: Handler) -> str:
        garbled = 0
        for _ in range(self.settings.max_steps):
            try:
                message = self._turn(on_event)
            except ValueError:
                # A tool call whose JSON could not be parsed at all (inputs stream eagerly):
                # there is no call to answer, so ask again, a couple of times at most.
                garbled += 1
                if garbled > 2:
                    raise
                on_event("notice", "A tool call came through garbled; trying that step again.")
                continue
            garbled = 0

            if message.stop_reason == "refusal":
                del self.messages[start:]
                on_event("notice", refusal_note(message))
                return ""

            content = echoable(message.content)
            self.messages.append({"role": "assistant", "content": content})
            if message.stop_reason == "pause_turn":
                continue  # a long server-side tool turn: send it back and the server picks it up again

            calls = [block for block in content if block.type == "tool_use"]
            if not calls:
                if message.stop_reason == "max_tokens":
                    on_event("notice", "The reply reached the output limit and was cut short.")
                return reply_text(content)

            cut_short = message.stop_reason == "max_tokens"
            results = [self._result(call, cut_short, on_event) for call in calls]
            self.messages.append({"role": "user", "content": results})

        on_event("notice", f"I stopped after {self.settings.max_steps} steps. Say “go on” and I'll continue.")
        return ""

    def _turn(self, on_event: Handler) -> Any:
        with self.client.beta.messages.stream(**self.request()) as stream:
            for event in stream:
                if event.type == "text":
                    on_event("text", event.text)
                elif event.type == "content_block_start" and event.content_block.type == "thinking":
                    on_event("thinking", "")
                elif event.type == "content_block_stop" and event.content_block.type in ("tool_use", "server_tool_use"):
                    on_event("tool", describe(event.content_block.name, event.content_block.input))
            return stream.get_final_message()

    def _result(self, call: Any, cut_short: bool, on_event: Handler) -> dict[str, Any]:
        if cut_short:
            # The reply hit max_tokens, so this call's input may be incomplete: don't run it.
            content: Any = "Not run: the reply reached the output limit before this call was complete. Try smaller steps."
            is_error = True
        else:
            content, is_error = self.toolbox.run(call.name, call.input, on_event)
        result = {"type": "tool_result", "tool_use_id": call.id, "content": content}
        if is_error:
            result["is_error"] = True
        return result
