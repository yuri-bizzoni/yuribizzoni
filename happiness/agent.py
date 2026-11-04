"""The agent itself: settings, instructions, and the loop that talks to Claude and runs tools."""

from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import anthropic

from .tools import Approver, Toolbox, describe

DEFAULT_MODEL = "claude-opus-5-5"
EFFORTS = ("low", "medium", "high", "xhigh", "max")
FALLBACK_BETA = "server-side-fallback-2026-07-01"
BUNDLED_SKILLS = Path(__file__).resolve().parent / "skills"

Handler = Callable[[str, str], None]
"""Receives progress while the agent works: ("text", chunk), ("tool", what it is doing),
("thinking", ""), or ("notice", something the user should know)."""


def _ignore(kind: str, text: str) -> None:
    pass


def _flag(value: str | None, default: bool) -> bool:
    if value is None or not value.strip():
        return default
    return value.strip().lower() not in ("0", "false", "no", "off")


@dataclass
class Settings:
    model: str = DEFAULT_MODEL
    effort: str = "medium"
    home: Path = field(default_factory=lambda: Path("~/.happiness").expanduser())
    workspace: Path | None = None
    web: bool = True
    shell: bool = False
    max_tokens: int = 64_000
    max_steps: int = 40

    @classmethod
    def from_env(cls) -> Settings:
        env = os.environ
        return cls(
            model=env.get("HAPPINESS_MODEL") or DEFAULT_MODEL,
            effort=env.get("HAPPINESS_EFFORT") or "medium",
            home=Path(env.get("HAPPINESS_HOME") or "~/.happiness").expanduser(),
            workspace=Path(env["HAPPINESS_WORKSPACE"]).expanduser() if env.get("HAPPINESS_WORKSPACE") else None,
            web=_flag(env.get("HAPPINESS_WEB"), True),
            shell=_flag(env.get("HAPPINESS_SHELL"), False),
        )

    @property
    def workspace_dir(self) -> Path:
        return self.workspace or self.home / "workspace"

    @property
    def skill_folders(self) -> list[Path]:
        """Bundled skills first, then your own: a skill of yours replaces a bundled one of the same name."""
        return [BUNDLED_SKILLS, self.home / "skills"]


def system_prompt(toolbox: Toolbox, now: dt.datetime) -> str:
    tools = []
    if toolbox.web:
        tools.append(
            "- web_search and web_fetch: use them for anything recent, anything you are unsure of, and pages "
            "the user mentions. Say where your facts come from."
        )
    tools += [
        "- memory: a /memories folder that lasts between conversations. Keep it short and organised: who the "
        "user is, what they prefer, what they are working on. Never store passwords, keys or other secrets.",
        "- str_replace_based_edit_tool: the user's workspace folder. Paths are relative to it (`.` is the "
        "workspace, `notes/ideas.md` a file in it). You can read text files, PDFs and images there, and write "
        "or edit text files. Save longer pieces of work there when the user asks or when it clearly helps, "
        "and tell them the path. Files they upload from the browser arrive in `uploads/`.",
        "- load_skill: loads one of your skills. Load a skill before any task it covers.",
    ]
    if toolbox.shell:
        tools.append(
            "- bash: runs one command in a fresh shell that starts in the workspace (no state carries over "
            "between commands). The user approves each command, so keep them few and say what they do."
        )
    skills = [f"- {s.name}: {s.description}" for s in sorted(toolbox.skills.values(), key=lambda s: s.name)]
    return "\n".join(
        [
            "You are Happiness, a small personal agent. The user runs you on their own computer or server "
            "and talks to you from a terminal or a browser. You help with all sorts of everyday work: "
            "finding things out, reading and summarising documents, writing and editing, planning, keeping "
            "notes, and small chores.",
            "",
            "## Your tools",
            *tools,
            "",
            "## Your skills",
            *(skills or ["(none installed)"]),
            "",
            "## How to work",
            "- Be warm, direct and concise. Your name describes a disposition, not a costume: be glad to help, "
            "without forced cheer, exclamation marks or emoji.",
            "- Make sensible assumptions and say what you assumed. Ask a short question first only when a "
            "wrong guess would waste real work.",
            "- For a task that takes several steps, say in one line what you are going to do, do it, and end "
            "with a short summary of what you did and found.",
            "- Write for a chat window: short paragraphs, lists where they help, Markdown used sparingly.",
            "",
            "## Happiness",
            "You carry a curated lexicon of words for happiness and its neighbours across languages (the "
            "happiness-lexicon skill). Load it whenever the user asks about such words or about how languages "
            "name joy, contentment or longing, and whenever naming the texture of a moment would genuinely help. "
            "Its cross-index, for example, maps the stages of a research project to the kind of happiness each "
            "stage runs on, which makes it a gentle diagnostic when a project stalls. Don't push it into "
            "unrelated tasks.",
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
            "organization, enable it in the Claude Console or turn the web tools off (--no-web, HAPPINESS_WEB=0)."
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
    """One conversation with Happiness. Not thread-safe: give each conversation its own Agent."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: anthropic.Anthropic | None = None,
        approve: Approver | None = None,
    ):
        self.settings = settings or Settings.from_env()
        if self.settings.effort not in EFFORTS:
            raise ValueError(f"effort must be one of {', '.join(EFFORTS)}, not {self.settings.effort!r}")
        self.client = client or anthropic.Anthropic()
        self.toolbox = Toolbox(
            workspace=self.settings.workspace_dir,
            memory=self.settings.home / "memory",
            skill_folders=self.settings.skill_folders,
            web=self.settings.web,
            shell=self.settings.shell,
            approve=approve,
        )
        self.new_conversation()

    def new_conversation(self) -> None:
        """Start afresh. The instructions and tools are fixed here and never change mid-conversation,
        and the history only ever grows: that keeps the prompt cache warm and every earlier turn valid."""
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
        """Send one user message, let Happiness work until it is done, and return its reply."""
        start = len(self.messages)
        self.messages.append({"role": "user", "content": text})
        try:
            return self._work(start, on_event)
        except BaseException:
            # An error or Ctrl-C: drop this exchange, so the conversation is as it was and can go on.
            del self.messages[start:]
            raise

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
            self.messages.append({"role": "user", "content": [self._result(call, cut_short) for call in calls]})

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

    def _result(self, call: Any, cut_short: bool) -> dict[str, Any]:
        if cut_short:
            # The reply hit max_tokens, so this call's input may be incomplete: don't run it.
            content: Any = "Not run: the reply reached the output limit before this call was complete. Try smaller steps."
            is_error = True
        else:
            content, is_error = self.toolbox.run(call.name, call.input)
        result = {"type": "tool_result", "tool_use_id": call.id, "content": content}
        if is_error:
            result["is_error"] = True
        return result
