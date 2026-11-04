"""The agent loop, end to end through the real SDK and a scripted API. Run from the repository root:

    python -m unittest discover -s happiness/tests -t .
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

import anthropic

from happiness.agent import FALLBACK_BETA, Agent, Settings, describe_error
from happiness.tests import fakeapi as api


class AgentTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = Path(tmp.name)
        self.settings = Settings(home=self.home)
        self.events: list[tuple[str, str]] = []

    def agent(self, *replies: bytes, **kwargs: Any) -> tuple[Agent, api.ScriptedAPI]:
        script = api.ScriptedAPI(*replies)
        return Agent(self.settings, client=script.client(), **kwargs), script

    def ask(self, agent: Agent, text: str) -> str:
        return agent.ask(text, on_event=lambda kind, payload: self.events.append((kind, payload)))

    def test_answers_and_sends_a_well_formed_request(self) -> None:
        agent, script = self.agent(api.reply(api.thinking(), api.text("Hello there.")))
        self.assertEqual(self.ask(agent, "Hi"), "Hello there.")
        self.assertIn(("thinking", ""), self.events)
        self.assertIn(("text", "Hello there."), self.events)

        sent = script.requests[0]
        self.assertEqual(sent["model"], "claude-opus-5-5")
        self.assertEqual(sent["thinking"], {"type": "adaptive"})
        self.assertEqual(sent["output_config"], {"effort": "medium"})
        self.assertEqual(sent["fallbacks"], "default")
        self.assertEqual(sent["cache_control"], {"type": "ephemeral"})
        self.assertEqual(sent["system"][0]["cache_control"], {"type": "ephemeral"})
        self.assertIn(FALLBACK_BETA, script.headers[0]["anthropic-beta"])
        self.assertEqual(
            [tool["name"] for tool in sent["tools"]],
            ["web_search", "web_fetch", "memory", "str_replace_based_edit_tool", "load_skill"],
        )
        self.assertTrue(sent["tools"][-1]["eager_input_streaming"])
        self.assertIn("- happiness-lexicon: A cross-linguistic lexicon", sent["system"][0]["text"])
        self.assertEqual([m["role"] for m in agent.messages], ["user", "assistant"])

    def test_runs_a_tool_and_keeps_the_history_append_only(self) -> None:
        agent, script = self.agent(
            api.reply(api.thinking(), api.tool_use("toolu_1", "load_skill", {"name": "happiness-lexicon"}), stop_reason="tool_use"),
            api.reply(api.text("Try *Vorfreude*.")),
        )
        self.assertEqual(self.ask(agent, "A word for looking forward to something?"), "Try *Vorfreude*.")
        self.assertIn(("tool", "loading the happiness-lexicon skill"), self.events)

        first, second = script.requests
        self.assertEqual([m["role"] for m in second["messages"]], ["user", "assistant", "user"])
        self.assertEqual(second["messages"][: len(first["messages"])], first["messages"])
        self.assertEqual((first["system"], first["tools"]), (second["system"], second["tools"]))
        thinking = second["messages"][1]["content"][0]
        self.assertEqual((thinking["type"], thinking["signature"]), ("thinking", "c2lnbmF0dXJl"))
        result = second["messages"][2]["content"][0]
        self.assertEqual(result["tool_use_id"], "toolu_1")
        self.assertIn("Vorfreude", result["content"])
        self.assertNotIn("is_error", result)

    def test_edits_files_in_the_workspace(self) -> None:
        agent, _ = self.agent(
            api.reply(api.tool_use("toolu_1", "str_replace_based_edit_tool", {"command": "create", "path": "notes/todo.md", "file_text": "- buy bread\n"}), stop_reason="tool_use"),
            api.reply(api.tool_use("toolu_2", "str_replace_based_edit_tool", {"command": "str_replace", "path": "notes/todo.md", "old_str": "bread", "new_str": "flowers"}), stop_reason="tool_use"),
            api.reply(api.text("Noted.")),
        )
        self.ask(agent, "Remind me to buy bread. No, flowers.")
        self.assertEqual((self.home / "workspace/notes/todo.md").read_text(), "- buy flowers\n")
        self.assertIn(("tool", "writing notes/todo.md"), self.events)

    def test_remembers_across_conversations(self) -> None:
        agent, _ = self.agent(
            api.reply(api.tool_use("toolu_1", "memory", {"command": "create", "path": "/memories/user.md", "file_text": "Teaches on Tuesdays.\n"}), stop_reason="tool_use"),
            api.reply(api.text("I'll remember.")),
        )
        self.ask(agent, "I teach on Tuesdays.")
        self.assertEqual((self.home / "memory/memories/user.md").read_text(), "Teaches on Tuesdays.\n")

    def test_resumes_a_paused_server_tool_turn(self) -> None:
        agent, script = self.agent(
            api.reply(
                api.tool_use("srvtoolu_1", "web_search", {"query": "felicità"}, server=True),
                api.search_results("srvtoolu_1"),
                stop_reason="pause_turn",
            ),
            api.reply(api.text("Found it.")),
        )
        self.assertEqual(self.ask(agent, "Search for felicità"), "Found it.")
        self.assertIn(("tool", "searching the web for “felicità”"), self.events)
        second = script.requests[1]["messages"]
        self.assertEqual([m["role"] for m in second], ["user", "assistant"])
        self.assertEqual([b["type"] for b in second[1]["content"]], ["server_tool_use", "web_search_tool_result"])

    def test_a_refusal_leaves_the_conversation_as_it_was(self) -> None:
        agent, script = self.agent(
            api.reply(api.text("Hello.")),
            api.reply(api.text("Part"), stop_reason="refusal", stop_details={"type": "refusal", "category": "cyber", "explanation": None}),
            api.reply(api.text("Sure.")),
        )
        self.ask(agent, "Hi")
        self.assertEqual(self.ask(agent, "Something declined"), "")
        self.assertEqual(len(agent.messages), 2)
        notices = [payload for kind, payload in self.events if kind == "notice"]
        self.assertIn("category: cyber", notices[0])
        self.ask(agent, "Something else")
        third = script.requests[2]["messages"]
        self.assertEqual([m["content"] for m in third if m["role"] == "user"], ["Hi", "Something else"])

    def test_a_tool_call_cut_off_by_max_tokens_is_not_run(self) -> None:
        agent, script = self.agent(
            api.reply(api.tool_use("toolu_1", "str_replace_based_edit_tool", {"command": "create", "path": "long.md", "file_text": "Once upon"}), stop_reason="max_tokens"),
            api.reply(api.text("I'll write it in parts.")),
        )
        self.ask(agent, "Write a long story to long.md")
        self.assertFalse((self.home / "workspace/long.md").exists())
        result = script.requests[1]["messages"][2]["content"][0]
        self.assertTrue(result["is_error"])

    def test_after_a_fallback_only_text_from_before_the_switch_goes_back(self) -> None:
        agent, script = self.agent(
            api.reply(api.thinking(), api.text("Part one. "), api.fallback(), api.text("Part two.")),
            api.reply(api.text("Next.")),
        )
        self.assertEqual(self.ask(agent, "Hi"), "Part one. Part two.")
        self.ask(agent, "And then?")
        echoed = script.requests[1]["messages"][1]["content"]
        self.assertEqual([(b["type"], b["text"]) for b in echoed], [("text", "Part one. "), ("text", "Part two.")])

    def test_arguments_that_do_not_fit_the_schema_come_back_as_invalid_json(self) -> None:
        agent, script = self.agent(
            api.reply(api.tool_use("toolu_1", "load_skill", {"name": 42}), stop_reason="tool_use"),
            api.reply(api.text("Let me try again.")),
        )
        self.ask(agent, "Load a skill")
        result = script.requests[1]["messages"][2]["content"][0]
        self.assertTrue(result["is_error"])
        self.assertIn("INVALID_JSON", result["content"])

    def test_unparseable_tool_json_is_asked_again(self) -> None:
        agent, script = self.agent(
            api.reply(api.tool_use("toolu_1", "load_skill", {}, raw="not json at all"), stop_reason="tool_use"),
            api.reply(api.text("Here you are.")),
        )
        self.assertEqual(self.ask(agent, "Load a skill"), "Here you are.")
        self.assertEqual(script.requests[0]["messages"], script.requests[1]["messages"])
        self.assertTrue(any(kind == "notice" for kind, _ in self.events))

    def test_an_api_error_drops_the_exchange(self) -> None:
        agent, _ = self.agent()  # nothing scripted: the API answers 500
        with self.assertRaises(anthropic.InternalServerError) as caught:
            self.ask(agent, "Hi")
        self.assertEqual(agent.messages, [])
        self.assertIn("500", describe_error(caught.exception) or "")

    def test_settings_shape_the_tools_and_request(self) -> None:
        self.settings.web = False
        self.settings.effort = "high"
        agent, script = self.agent(api.reply(api.text("Ok.")))
        self.ask(agent, "Hi")
        sent = script.requests[0]
        self.assertEqual([t["name"] for t in sent["tools"]], ["memory", "str_replace_based_edit_tool", "load_skill"])
        self.assertEqual(sent["output_config"], {"effort": "high"})
        self.assertNotIn("web_search", sent["system"][0]["text"])

    def test_new_conversation_starts_empty(self) -> None:
        agent, _ = self.agent(api.reply(api.text("Ok.")))
        self.ask(agent, "Hi")
        agent.new_conversation()
        self.assertEqual(agent.messages, [])

    def test_rejects_an_unknown_effort(self) -> None:
        self.settings.effort = "enormous"
        with self.assertRaises(ValueError):
            self.agent()


class ErrorTests(unittest.TestCase):
    def test_missing_credentials_are_explained(self) -> None:
        error = TypeError("Could not resolve authentication method. Expected one of api_key, ...")
        self.assertIn("ANTHROPIC_API_KEY", describe_error(error) or "")
        self.assertIsNone(describe_error(TypeError("something else")))


if __name__ == "__main__":
    unittest.main()
