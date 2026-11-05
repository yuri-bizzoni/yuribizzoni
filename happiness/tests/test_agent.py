"""The agent loop, end to end through the real SDK and a scripted API. Run from the repository root:

    python -m unittest discover -s happiness/tests -t .
"""

from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path

import anthropic

from happiness.agent import FALLBACK_BETA, Agent, Settings, describe_error
from happiness.corpus import Corpus
from happiness.sandbox import LocalSandbox, ask_before_running
from happiness.tests import fakeapi as api
from happiness.tests.corpusfix import make_corpus

CORPUS_TOOLS = ["catalogue", "count", "concordance", "read", "run_python", "memory", "str_replace_based_edit_tool", "load_skill"]


class AgentTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = Path(tmp.name)
        root = make_corpus(self.home)
        self.settings = Settings(corpus=root, home=self.home / "happiness", passages=2_000)
        self.corpus = Corpus(root, self.settings.index_dir)
        self.addCleanup(self.corpus.close)
        self.corpus.update()
        self.events: list[tuple[str, str]] = []

    def agent(self, *replies: bytes) -> tuple[Agent, api.ScriptedAPI]:
        script = api.ScriptedAPI(*replies)
        sandbox = LocalSandbox(self.corpus.root, approve=None)
        return Agent(self.settings, client=script.client(), corpus=self.corpus, sandbox=sandbox), script

    def ask(self, agent: Agent, text: str) -> str:
        return agent.ask(text, on_event=lambda kind, payload: self.events.append((kind, payload)))

    def notebook(self, agent: Agent) -> str:
        assert agent.lab.session is not None
        return (agent.lab.session / "notebook.md").read_text(encoding="utf-8")

    def test_answers_and_sends_a_well_formed_request(self) -> None:
        agent, script = self.agent(api.reply(api.thinking(), api.text("Five texts, five languages.")))
        self.assertEqual(self.ask(agent, "What is in my corpus?"), "Five texts, five languages.")
        self.assertIn(("thinking", ""), self.events)

        sent = script.requests[0]
        self.assertEqual(sent["model"], "claude-opus-5-5")
        self.assertEqual(sent["thinking"], {"type": "adaptive"})
        self.assertEqual(sent["output_config"], {"effort": "medium"})
        self.assertEqual(sent["fallbacks"], "default")
        self.assertEqual(sent["cache_control"], {"type": "ephemeral"})
        self.assertEqual(sent["system"][0]["cache_control"], {"type": "ephemeral"})
        self.assertIn(FALLBACK_BETA, script.headers[0]["anthropic-beta"])
        self.assertEqual([tool["name"] for tool in sent["tools"]], CORPUS_TOOLS)
        system = sent["system"][0]["text"]
        for expected in ("5 texts", "Columns of `texts`: id, name, author, year, prize, title, anno, language", "- felicita-it: 35 terms in 6 categories",
                         "- happiness-lexicon: A cross-linguistic lexicon", "a budget of 2,000 characters per conversation"):
            self.assertIn(expected, system)
        notebook = self.notebook(agent)
        self.assertIn("You asked\n\nWhat is in my corpus?", notebook)
        self.assertIn("Happiness answered\n\nFive texts, five languages.", notebook)

    def test_counts_and_keeps_the_history_append_only(self) -> None:
        agent, script = self.agent(
            api.reply(api.thinking(), api.tool_use("toolu_1", "count", {"lexicon": "felicita-it", "segments": 10}), stop_reason="tool_use"),
            api.reply(api.text("The novella carries almost all of it.")),
        )
        self.assertEqual(self.ask(agent, "Where is the happiness?"), "The novella carries almost all of it.")
        self.assertIn(("tool", "counting felicita-it, 10 segments"), self.events)
        first, second = script.requests
        self.assertEqual(second["messages"][: len(first["messages"])], first["messages"])
        self.assertEqual((first["system"], first["tools"]), (second["system"], second["tools"]))
        result = second["messages"][2]["content"][0]
        self.assertEqual(result["tool_use_id"], "toolu_1")
        self.assertIn("Counted the felicita-it lexicon in 5 texts", result["content"])
        assert agent.lab.session is not None
        self.assertTrue((agent.lab.session / "tables" / "01-count-the-felicita-it-lexicon.csv").is_file())
        self.assertIn("**count** the felicita-it lexicon", self.notebook(agent))

    def test_the_passage_budget_lasts_a_conversation(self) -> None:
        read = {"text": "novella", "position": 20, "length": 1500}
        agent, script = self.agent(
            api.reply(api.tool_use("toolu_1", "read", read), stop_reason="tool_use"),
            api.reply(api.text("Read.")),
            api.reply(api.tool_use("toolu_2", "read", read), stop_reason="tool_use"),
            api.reply(api.text("Read again.")),
            api.reply(api.tool_use("toolu_3", "read", read), stop_reason="tool_use"),
            api.reply(api.text("Fresh budget.")),
        )
        self.ask(agent, "Read me something.")
        self.ask(agent, "And more.")
        second = script.requests[3]["messages"][-1]["content"][0]["content"]
        left = int(re.search(r"([\d,]+) characters left in the passage budget", second).group(1).replace(",", ""))
        self.assertLess(left, 50)  # 1,500 + about 500 of 2,000: passages end at word boundaries
        first_session = agent.lab.session
        agent.new_conversation()
        self.ask(agent, "Start again.")
        third = script.requests[5]["messages"][-1]["content"][0]["content"]
        self.assertIn("characters left in the passage budget", third)
        self.assertNotIn("is used up", third)
        self.assertNotEqual(agent.lab.session, first_session)

    def test_resumes_a_paused_server_tool_turn(self) -> None:
        self.settings.web = True
        agent, script = self.agent(
            api.reply(api.tool_use("srvtoolu_1", "web_search", {"query": "Grazia Deledda"}, server=True), api.search_results("srvtoolu_1"), stop_reason="pause_turn"),
            api.reply(api.text("Nobel 1926.")),
        )
        self.assertEqual(self.ask(agent, "When did Deledda win the Nobel?"), "Nobel 1926.")
        self.assertEqual([t["name"] for t in script.requests[0]["tools"]][:2], ["web_search", "web_fetch"])
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
        self.assertIn("category: cyber", [p for k, p in self.events if k == "notice"][0])
        self.ask(agent, "Something else")
        self.assertEqual([m["content"] for m in script.requests[2]["messages"] if m["role"] == "user"], ["Hi", "Something else"])
        self.assertIn("> That request was declined", self.notebook(agent))

    def test_a_tool_call_cut_off_by_max_tokens_is_not_run(self) -> None:
        agent, script = self.agent(
            api.reply(api.tool_use("toolu_1", "run_python", {"title": "x", "code": "open('ran', 'w')"}), stop_reason="max_tokens"),
            api.reply(api.text("I'll do it in parts.")),
        )
        self.ask(agent, "Analyse")
        assert agent.lab.session is not None
        self.assertFalse((agent.lab.session / "ran").exists())
        self.assertTrue(script.requests[1]["messages"][2]["content"][0]["is_error"])

    def test_after_a_fallback_only_text_from_before_the_switch_goes_back(self) -> None:
        agent, script = self.agent(
            api.reply(api.thinking(), api.text("Part one. "), api.fallback(), api.text("Part two.")),
            api.reply(api.text("Next.")),
        )
        self.assertEqual(self.ask(agent, "Hi"), "Part one. Part two.")
        self.ask(agent, "And then?")
        echoed = script.requests[1]["messages"][1]["content"]
        self.assertEqual([(b["type"], b["text"]) for b in echoed], [("text", "Part one. "), ("text", "Part two.")])

    def test_bad_arguments_and_garbled_json(self) -> None:
        agent, script = self.agent(
            api.reply(api.tool_use("toolu_1", "count", {"terms": "joy"}), stop_reason="tool_use"),
            api.reply(api.tool_use("toolu_2", "catalogue", {}, raw="not json at all"), stop_reason="tool_use"),
            api.reply(api.text("Done.")),
        )
        self.assertEqual(self.ask(agent, "Count"), "Done.")
        result = script.requests[1]["messages"][2]["content"][0]
        self.assertTrue(result["is_error"])
        self.assertIn("INVALID_JSON", result["content"])
        self.assertEqual(script.requests[1]["messages"], script.requests[2]["messages"])  # the garbled turn was asked again
        self.assertTrue(any(kind == "notice" for kind, _ in self.events))

    def test_an_api_error_drops_the_exchange(self) -> None:
        agent, _ = self.agent()  # nothing scripted: the API answers 500
        with self.assertRaises(anthropic.InternalServerError) as caught:
            self.ask(agent, "Hi")
        self.assertEqual(agent.messages, [])
        self.assertIn("500", describe_error(caught.exception) or "")
        self.assertIn("> Stopped: The Claude API returned an error (500)", self.notebook(agent))

    def test_without_docker_scripts_wait_for_approval_by_default(self) -> None:
        self.settings.sandbox = "local"
        agent = Agent(self.settings, client=api.ScriptedAPI().client(), corpus=self.corpus)
        self.assertIs(agent.toolbox.sandbox.approve, ask_before_running)

    def test_missing_credentials_are_explained(self) -> None:
        error = TypeError("Could not resolve authentication method. Expected one of api_key, ...")
        self.assertIn("ANTHROPIC_API_KEY", describe_error(error) or "")
        self.assertIsNone(describe_error(TypeError("something else")))


if __name__ == "__main__":
    unittest.main()
