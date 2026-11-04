"""The local tools: workspace confinement, skills, the shell, and how calls are described."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from anthropic.lib.tools import ToolError

from happiness.agent import BUNDLED_SKILLS, Settings
from happiness.tools import Shell, Toolbox, Workspace, describe, discover_skills, read_frontmatter, read_skill


class TempDirTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)


class WorkspaceTests(TempDirTest):
    def setUp(self) -> None:
        super().setUp()
        self.ws = Workspace(self.tmp / "ws")

    def test_paths_stay_inside_the_workspace(self) -> None:
        (self.tmp / "secret.txt").write_text("secret")
        for path in ("../secret.txt", "notes/../../secret.txt", str(self.tmp / "secret.txt")):
            with self.assertRaises(ToolError, msg=path):
                self.ws({"command": "view", "path": path})
        # An absolute path outside means the same path inside the workspace, never the real file.
        with self.assertRaises(ToolError) as caught:
            self.ws({"command": "view", "path": "/etc/hostname"})
        self.assertIn("does not exist", str(caught.exception))
        self.assertEqual(self.ws.resolve("/notes.md"), self.ws.root / "notes.md")

    def test_symlinks_out_of_the_workspace_are_refused(self) -> None:
        (self.tmp / "secret.txt").write_text("secret")
        (self.ws.root / "link.txt").symlink_to(self.tmp / "secret.txt")
        with self.assertRaises(ToolError):
            self.ws({"command": "view", "path": "link.txt"})

    def test_create_view_edit_insert(self) -> None:
        self.assertEqual(self.ws({"command": "create", "path": "a/b.md", "file_text": "one\ntwo\n"}), "Wrote a/b.md.")
        self.assertEqual(self.ws({"command": "view", "path": "a/b.md"}), "     1\tone\n     2\ttwo")
        self.assertEqual(self.ws({"command": "view", "path": "a/b.md", "view_range": [2, -1]}), "     2\ttwo")
        self.ws({"command": "str_replace", "path": "a/b.md", "old_str": "two", "new_str": "deux"})
        self.ws({"command": "insert", "path": "a/b.md", "insert_line": 0, "insert_text": "zero"})
        self.ws({"command": "insert", "path": "a/b.md", "insert_line": 3, "insert_text": "three"})
        self.assertEqual((self.ws.root / "a/b.md").read_text(), "zero\none\ndeux\nthree\n")
        self.assertEqual(self.ws({"command": "view", "path": "."}), "a/\na/b.md  20 B")

    def test_overwriting_keeps_a_backup(self) -> None:
        self.ws({"command": "create", "path": "n.md", "file_text": "first"})
        self.assertIn(".backups/n.md", self.ws({"command": "create", "path": "n.md", "file_text": "second"}))
        self.assertEqual((self.ws.root / ".backups/n.md").read_text(), "first")
        self.assertNotIn(".backups", self.ws({"command": "view", "path": "."}))

    def test_ambiguous_or_missing_replacements_fail(self) -> None:
        self.ws({"command": "create", "path": "n.md", "file_text": "la la"})
        with self.assertRaises(ToolError):
            self.ws({"command": "str_replace", "path": "n.md", "old_str": "la", "new_str": "lo"})
        with self.assertRaises(ToolError):
            self.ws({"command": "str_replace", "path": "n.md", "old_str": "do", "new_str": "re"})
        with self.assertRaises(ToolError):
            self.ws({"command": "undo_edit", "path": "n.md"})

    def test_pdfs_and_images_come_back_as_content_blocks(self) -> None:
        (self.ws.root / "paper.pdf").write_bytes(b"%PDF-1.4 tiny")
        (self.ws.root / "photo.png").write_bytes(b"\x89PNG tiny")
        pdf = self.ws({"command": "view", "path": "paper.pdf"})
        self.assertEqual(pdf[1]["type"], "document")
        self.assertEqual(pdf[1]["source"]["media_type"], "application/pdf")
        self.assertEqual(self.ws({"command": "view", "path": "photo.png"})[1]["type"], "image")
        (self.ws.root / "blob.bin").write_bytes(b"\xff\xfe\x00")
        with self.assertRaises(ToolError):
            self.ws({"command": "view", "path": "blob.bin"})


class SkillTests(TempDirTest):
    def test_the_lexicon_ships_with_happiness(self) -> None:
        skills = discover_skills([BUNDLED_SKILLS])
        lexicon = skills["happiness-lexicon"]
        self.assertTrue(lexicon.description.startswith("A cross-linguistic lexicon"))
        self.assertIn('"what\'s the word for X kind of happiness"', lexicon.description)
        text = read_skill(skills, "happiness-lexicon")
        for word in ("felicità", "Vorfreude", "hygge", "saudade", "Cross-index"):
            self.assertIn(word, text)

    def test_your_skills_replace_bundled_ones_and_stay_in_their_folder(self) -> None:
        mine = self.tmp / "skills" / "lexicon"
        mine.mkdir(parents=True)
        (mine / "SKILL.md").write_text("---\nname: happiness-lexicon\ndescription: 'My own, it''s better'\n---\nMine.\n")
        (mine / "extra.md").write_text("More.")
        skills = discover_skills([BUNDLED_SKILLS, self.tmp / "skills"])
        self.assertEqual(skills["happiness-lexicon"].description, "My own, it's better")
        self.assertEqual(read_skill(skills, "happiness-lexicon", "extra.md"), "More.")
        for bad in ("../../SKILL.md", "/etc/hostname", "missing.md"):
            with self.assertRaises(ToolError):
                read_skill(skills, "happiness-lexicon", bad)
        with self.assertRaises(ToolError):
            read_skill(skills, "nope")

    def test_frontmatter(self) -> None:
        fields, body = read_frontmatter('---\nname: x\ndescription: "say \\"hi\\""\n---\n\nBody')
        self.assertEqual(fields, {"name": "x", "description": 'say "hi"'})
        self.assertEqual(body, "Body")
        self.assertEqual(read_frontmatter("No header"), ({}, "No header"))


class ShellTests(TempDirTest):
    def test_commands_need_approval(self) -> None:
        asked: list[str] = []
        shell = Shell(self.tmp, approve=lambda command: asked.append(command) or False)
        with self.assertRaises(ToolError):
            shell({"command": "touch made.txt"})
        self.assertEqual(asked, ["touch made.txt"])
        self.assertFalse((self.tmp / "made.txt").exists())

    def test_runs_in_the_workspace_without_your_keys(self) -> None:
        shell = Shell(self.tmp, approve=lambda command: True)
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-secret", "HAPPINESS_PASSWORD": "pw"}):
            output = shell({"command": 'pwd; echo "[$ANTHROPIC_API_KEY$HAPPINESS_PASSWORD]"; exit 3'})
        self.assertIn(str(self.tmp.resolve()), output)
        self.assertIn("[]", output)
        self.assertIn("(exit code 3)", output)


class ToolboxTests(TempDirTest):
    def toolbox(self, **kwargs: object) -> Toolbox:
        settings = Settings(home=self.tmp)
        return Toolbox(workspace=settings.workspace_dir, memory=self.tmp / "memory", skill_folders=settings.skill_folders, **kwargs)

    def test_definitions_follow_the_settings(self) -> None:
        names = lambda box: [tool["name"] for tool in box.definitions()]  # noqa: E731
        self.assertEqual(names(self.toolbox()), ["web_search", "web_fetch", "memory", "str_replace_based_edit_tool", "load_skill"])
        self.assertEqual(names(self.toolbox(web=False, shell=True)), ["memory", "str_replace_based_edit_tool", "load_skill", "bash"])

    def test_errors_go_back_to_the_model(self) -> None:
        box = self.toolbox()
        self.assertEqual(box.run("bash", {"command": "ls"}), ("There is no tool called bash.", True))
        content, is_error = box.run("str_replace_based_edit_tool", {"command": "view", "path": "../x"})
        self.assertTrue(is_error)
        self.assertIn("outside the workspace", content)
        content, is_error = box.run("load_skill", "not a dict")
        self.assertEqual(json.loads(content), {"INVALID_JSON": '"not a dict"'})
        content, is_error = box.run("memory", {"command": "view", "path": "/etc"})
        self.assertTrue(is_error)

    def test_descriptions(self) -> None:
        self.assertEqual(describe("web_fetch", {"url": "https://example.org"}), "reading https://example.org")
        self.assertEqual(describe("memory", {"command": "view", "path": "/memories"}), "memory: view /memories")
        self.assertEqual(describe("bash", {"command": "ls"}), "$ ls")
        self.assertEqual(describe("code_execution", None), "code execution")


class SettingsTests(unittest.TestCase):
    def test_from_env(self) -> None:
        env = {"HAPPINESS_MODEL": "claude-sonnet-5-5", "HAPPINESS_WEB": "off", "HAPPINESS_SHELL": "1", "HAPPINESS_HOME": "/srv/h"}
        with mock.patch.dict(os.environ, env, clear=True):
            settings = Settings.from_env()
        self.assertEqual((settings.model, settings.web, settings.shell), ("claude-sonnet-5-5", False, True))
        self.assertEqual(settings.workspace_dir, Path("/srv/h/workspace"))
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual((Settings.from_env().web, Settings.from_env().shell), (True, False))


if __name__ == "__main__":
    unittest.main()
