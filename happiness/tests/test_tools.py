"""The tools: the corpus tools and their passage budget, the lab notebook, the workspace, skills and lexicons."""

from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from anthropic.lib.tools import ToolError

from happiness.agent import BUNDLED_LEXICONS, BUNDLED_SKILLS, Settings
from happiness.corpus import Corpus, load_text
from happiness.lab import Lab
from happiness.resources import discover_skills, read_frontmatter, read_lexicon, read_skill
from happiness.sandbox import LocalSandbox
from happiness.tests.corpusfix import make_corpus
from happiness.tools import Toolbox, describe
from happiness.workspace import Workspace


class TempDirTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)


class ToolboxTests(TempDirTest):
    def setUp(self) -> None:
        super().setUp()
        self.corpus = Corpus(make_corpus(self.tmp), self.tmp / "index")
        self.addCleanup(self.corpus.close)
        self.corpus.update()
        self.lab = Lab(self.tmp / "lab", self.corpus, "claude-opus-5-5")
        self.approvals: list[str] = []
        self.box = Toolbox(
            corpus=self.corpus,
            lab=self.lab,
            sandbox=LocalSandbox(self.corpus.root, approve=lambda code, path: self.approvals.append(path.name) or True),
            memory_dir=self.tmp / "memory",
            skill_folders=[BUNDLED_SKILLS],
            lexicon_folders=[BUNDLED_LEXICONS, self.lab.root / "lexicons"],
            passages=600,
        )
        self.lab.begin("How happy are these texts?")

    def run_tool(self, name: str, args: object) -> tuple[str, bool]:
        content, is_error = self.box.run(name, args)
        return content if isinstance(content, str) else json.dumps(content), is_error

    def notebook(self) -> str:
        assert self.lab.session is not None
        return (self.lab.session / "notebook.md").read_text(encoding="utf-8")

    def test_the_lab_folder_is_ready_for_scripts(self) -> None:
        session = self.lab.session
        assert session is not None
        self.assertTrue(session.name.endswith("how-happy-are-these-texts"))
        self.assertTrue((session / "happiness_corpus.py").is_file())
        with open(session / "catalogue.csv", encoding="utf-8") as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 5)
        self.assertIn("5 texts", self.notebook())

    def test_catalogue(self) -> None:
        content, is_error = self.run_tool("catalogue", {"sql": "SELECT name, author, year FROM texts WHERE year IS NOT NULL ORDER BY year"})
        self.assertFalse(is_error)
        self.assertEqual(content.splitlines()[:2], ["name\tauthor\tyear", "it/novella\tAutrice Prova\t1890"])
        content, is_error = self.run_tool("catalogue", {"sql": "SELECT body FROM chunk_text"})
        self.assertTrue(is_error)
        self.assertIn("only the table `texts`", content)

    def test_count_saves_a_table_and_summarises_it(self) -> None:
        content, is_error = self.run_tool("count", {"lexicon": "felicita-it", "segments": 10, "group_by": "prize"})
        self.assertFalse(is_error, content)
        self.assertIn("Counted the felicita-it lexicon in 5 texts", content)
        self.assertIn("By category (hits): felicità 81 · gioia 40", content)
        self.assertIn("By prize: no: ", content)
        self.assertIn("Per segment (10 equal parts", content)
        table = content.splitlines()[-1].removeprefix("Table: ")
        assert self.lab.session is not None
        with open(self.lab.session / table, encoding="utf-8") as handle:
            rows = {row["name"]: row for row in csv.DictReader(handle)}
        self.assertEqual(rows["it/novella"]["hits"], "162")
        self.assertEqual(rows["it/novella"]["hits_letizia"], "1")
        self.assertIn("seg_10", rows["it/novella"])
        self.assertIn("**count** the felicita-it lexicon", self.notebook())

    def test_count_checks_its_arguments(self) -> None:
        self.assertTrue(self.run_tool("count", {"terms": ["joy"], "segments": 1})[1])
        self.assertIn("no column", self.run_tool("count", {"terms": ["joy"], "group_by": "colour"})[0])
        self.assertIn("No lexicon called", self.run_tool("count", {"lexicon": "nope"})[0])
        self.assertIn("Give some terms", self.run_tool("count", {})[0])
        content, is_error = self.run_tool("count", {"terms": "joy"})
        self.assertTrue(is_error)
        self.assertEqual(json.loads(content), {"INVALID_JSON": '{"terms": "joy"}'})
        self.assertIn("INVALID_JSON", self.run_tool("count", {"terms": ["joy"], "colour": "red"})[0])
        self.assertIn("SQL error", self.run_tool("count", {"terms": ["joy"], "where": "colour = 1"})[0])

    def test_a_lexicon_written_in_the_lab_can_be_used(self) -> None:
        (self.lab.root / "lexicons" / "lykke-da.csv").write_text("kategori,ord\nlykke,lykke*\nglæde,glæde*\nglæde,glad\n", encoding="utf-8")
        content, _ = self.run_tool("count", {"lexicon": "lykke-da", "where": "language = 'da'"})
        self.assertIn("By category (hits): lykke 8 · glæde 8", content)

    def test_passages_are_held_to_the_budget(self) -> None:
        content, is_error = self.run_tool("concordance", {"terms": ["felice"], "limit": 5, "context": 30})
        self.assertFalse(is_error)
        self.assertIn("40 matches for felice", content)
        self.assertIn("«felice»", content)
        spent = self.box.budget.used
        self.assertTrue(0 < spent <= 5 * 66)
        content, _ = self.run_tool("read", {"text": "novella", "position": 50, "length": 1500})
        passage = content.split("\n\n", 1)[1]
        self.assertGreater(len(passage), 200)
        self.assertLessEqual(len(passage), 600 - spent)  # 600 characters in all
        self.assertEqual(self.box.budget.used, spent + len(passage))
        self.box.budget.used = 600
        content, is_error = self.run_tool("read", {"text": "novella", "position": 10})
        self.assertTrue(is_error)
        self.assertIn("passage budget for this conversation (600 characters) is used up", content)
        content, _ = self.run_tool("concordance", {"terms": ["joy"]})
        self.assertEqual(content, "4 matches for joy. The passage budget is used up, so no lines are shown.")
        self.box.new_conversation()
        self.assertEqual(self.box.budget.left, 600)

    def test_the_notebook_logs_passages_without_their_text(self) -> None:
        offset = next(line.offset for line in self.corpus.concordance(["letizia"])[1])
        content, _ = self.run_tool("read", {"text": "novella", "start": offset, "length": 200})
        self.assertIn("letizia tornò", content)
        notebook = self.notebook()
        self.assertIn("**read** it/novella, characters", notebook)
        self.assertNotIn("letizia tornò", notebook)

    def test_run_python_writes_tables_and_figures(self) -> None:
        self.run_tool("count", {"terms": ["felice"]})
        code = (
            "import csv, pathlib\n"
            "from happiness_corpus import catalogue, text, words\n"
            "rows = list(csv.DictReader(open('tables/01-count-felice.csv')))\n"
            "print('texts', len(rows), 'catalogue', len(catalogue()))\n"
            "print('words in the novella', len(words(text('it/novella'))))\n"
            "pathlib.Path('figures/plot.svg').write_text('<svg/>')\n"
        )
        content, is_error = self.run_tool("run_python", {"title": "Check the table", "code": code})
        self.assertFalse(is_error, content)
        self.assertIn("scripts/01-check-the-table.py ran (exit code 0", content)
        self.assertIn("Files written: figures/plot.svg", content)
        self.assertIn("texts 5 catalogue 5", content)
        self.assertEqual(self.approvals, ["01-check-the-table.py"])
        content, is_error = self.run_tool("run_python", {"title": "fail", "code": "raise SystemExit('nope')"})
        self.assertTrue(is_error)
        self.assertIn("exit code 1", content)
        self.assertIn("**run_python** [scripts/02-fail.py]", self.notebook())

    def test_the_helper_reads_texts_as_the_index_does(self) -> None:
        assert self.lab.session is not None
        code = "from happiness_corpus import text\nimport sys\nsys.stdout.write(text('en/story'))"
        content, is_error = self.run_tool("run_python", {"title": "story", "code": code})
        self.assertFalse(is_error, content)
        body, _ = load_text(self.corpus.root / "en" / "story.txt")
        self.assertIn(body[:200], content)
        self.assertNotIn("Licence", content)

    def test_skills_memory_and_the_lab_editor(self) -> None:
        content, is_error = self.run_tool("load_skill", {"name": "happiness-lexicon"})
        self.assertFalse(is_error)
        self.assertIn("Vorfreude", content)
        self.assertFalse(self.run_tool("memory", {"command": "create", "path": "/memories/corpus.md", "file_text": "Five languages.\n"})[1])
        self.assertTrue((self.tmp / "memory" / "memories" / "corpus.md").is_file())
        content, is_error = self.run_tool("str_replace_based_edit_tool", {"command": "view", "path": "."})
        self.assertFalse(is_error)
        self.assertIn("lexicons/", content)
        content, is_error = self.run_tool("str_replace_based_edit_tool", {"command": "view", "path": "../corpus/en/story.txt"})
        self.assertTrue(is_error)
        self.assertEqual(self.run_tool("nope", {}), ("There is no tool called nope.", True))

    def test_definitions(self) -> None:
        names = [tool["name"] for tool in self.box.definitions()]
        self.assertEqual(names, ["catalogue", "count", "concordance", "read", "run_python", "memory", "str_replace_based_edit_tool", "load_skill"])
        self.box.web = True
        self.assertEqual([tool["name"] for tool in self.box.definitions()][:2], ["web_search", "web_fetch"])


class LexiconTests(TempDirTest):
    def test_the_course_lexicon(self) -> None:
        lexicon = read_lexicon(BUNDLED_LEXICONS / "felicita-it.csv")
        self.assertEqual(len(lexicon.terms), 35)
        self.assertEqual(lexicon.summary(), "35 terms in 6 categories (felicità, gioia, allegria, contentezza, letizia, beatitudine)")

    def test_other_shapes(self) -> None:
        (self.tmp / "plain.csv").write_text("joy\nhappiness\nglad*\n", encoding="utf-8")
        self.assertEqual(read_lexicon(self.tmp / "plain.csv").categories, None)
        (self.tmp / "semi.csv").write_text("joie;joie*\nbonheur;bonheur\n", encoding="utf-8")
        self.assertEqual(read_lexicon(self.tmp / "semi.csv").terms, ["joie*", "bonheur"])
        (self.tmp / "empty.csv").write_text("term\n", encoding="utf-8")
        with self.assertRaises(ToolError):
            read_lexicon(self.tmp / "empty.csv")


class WorkspaceTests(TempDirTest):
    def setUp(self) -> None:
        super().setUp()
        self.ws = Workspace(self.tmp / "ws")

    def test_paths_stay_inside_the_workspace(self) -> None:
        (self.tmp / "secret.txt").write_text("secret")
        for path in ("../secret.txt", "notes/../../secret.txt", str(self.tmp / "secret.txt")):
            with self.assertRaises(ToolError, msg=path):
                self.ws({"command": "view", "path": path})
        with self.assertRaises(ToolError) as caught:
            self.ws({"command": "view", "path": "/etc/hostname"})
        self.assertIn("does not exist", str(caught.exception))
        (self.ws.root / "link.txt").symlink_to(self.tmp / "secret.txt")
        with self.assertRaises(ToolError):
            self.ws({"command": "view", "path": "link.txt"})

    def test_create_view_edit_insert(self) -> None:
        self.ws({"command": "create", "path": "a/b.md", "file_text": "one\ntwo\n"})
        self.assertEqual(self.ws({"command": "view", "path": "a/b.md", "view_range": [2, -1]}), "     2\ttwo")
        self.ws({"command": "str_replace", "path": "a/b.md", "old_str": "two", "new_str": "deux"})
        self.ws({"command": "insert", "path": "a/b.md", "insert_line": 0, "insert_text": "zero"})
        self.assertEqual((self.ws.root / "a/b.md").read_text(), "zero\none\ndeux\n")
        self.assertIn(".backups/a/b.md", self.ws({"command": "create", "path": "a/b.md", "file_text": "new"}))
        with self.assertRaises(ToolError):
            self.ws({"command": "str_replace", "path": "a/b.md", "old_str": "absent", "new_str": "x"})

    def test_images_come_back_as_content_blocks(self) -> None:
        (self.ws.root / "plot.png").write_bytes(b"\x89PNG tiny")
        self.assertEqual(self.ws({"command": "view", "path": "plot.png"})[1]["type"], "image")


class SkillTests(TempDirTest):
    def test_the_lexicon_skill_ships_with_happiness(self) -> None:
        skills = discover_skills([BUNDLED_SKILLS])
        self.assertTrue(skills["happiness-lexicon"].description.startswith("A cross-linguistic lexicon"))
        self.assertIn("Cross-index", read_skill(skills, "happiness-lexicon"))
        with self.assertRaises(ToolError):
            read_skill(skills, "happiness-lexicon", "../../agent.py")

    def test_frontmatter(self) -> None:
        fields, body = read_frontmatter('---\nname: x\ndescription: "say \\"hi\\""\n---\n\nBody')
        self.assertEqual((fields, body), ({"name": "x", "description": 'say "hi"'}, "Body"))


class DescribeTests(unittest.TestCase):
    def test_descriptions(self) -> None:
        self.assertEqual(describe("count", {"lexicon": "felicita-it", "where": "language = 'it'", "segments": 10}),
                         "counting felicita-it where language = 'it', 10 segments")
        self.assertEqual(describe("read", {"text": "novella", "position": 35}), "reading novella at 35%")
        self.assertEqual(describe("run_python", {"title": "arcs"}), "running a script: arcs")
        self.assertEqual(describe("catalogue", {"sql": "SELECT  name\nFROM texts"}), "catalogue: SELECT name FROM texts")


class SettingsTests(unittest.TestCase):
    def test_from_env(self) -> None:
        env = {"HAPPINESS_CORPUS": "/srv/novels", "HAPPINESS_WEB": "1", "HAPPINESS_PASSAGES": "5000", "HAPPINESS_HOME": "/srv/h"}
        with mock.patch.dict(os.environ, env, clear=True):
            settings = Settings.from_env()
        self.assertEqual((settings.corpus, settings.web, settings.passages), (Path("/srv/novels"), True, 5000))
        self.assertEqual((settings.lab_dir, settings.index_dir), (Path("/srv/h/lab"), Path("/srv/h/index")))
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(Settings.from_env().web)

    def test_check(self) -> None:
        for settings in (Settings(), Settings(corpus=Path("."), effort="lots"), Settings(corpus=Path("."), sandbox="cloud")):
            with self.assertRaises(ValueError):
                settings.check()


if __name__ == "__main__":
    unittest.main()
