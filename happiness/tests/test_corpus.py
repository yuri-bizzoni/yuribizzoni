"""The corpus: reading texts, the catalogue, the index and searching it. Run from the repository root:

    python -m unittest discover -s happiness/tests -t .
"""

from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from happiness import corpus as corpus_module
from happiness.corpus import Corpus, CorpusError, chunk_spans, load_text, term_pattern
from happiness.tests.corpusfix import CLOSING, ITALIAN, make_corpus


class CorpusTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.root = make_corpus(self.tmp)
        self.corpus = self.open(self.tmp / "index")
        self.report = self.corpus.update()

    def open(self, index: Path) -> Corpus:
        corpus = Corpus(self.root, index)
        self.addCleanup(corpus.close)
        return corpus

    def row(self, name: str) -> dict:
        return self.corpus.find(name)

    def test_indexes_every_visible_text(self) -> None:
        self.assertEqual(self.report, {"texts": 5, "indexed": 5, "removed": 0})
        names = [t["name"] for t in self.corpus.texts()]
        self.assertEqual(names, ["da/fortaelling", "de/erzaehlung", "en/story", "fr/conte", "it/novella"])

    def test_detects_languages_and_reads_metadata(self) -> None:
        detected = {t["name"]: t["detected_language"] for t in self.corpus.texts()}
        self.assertEqual(detected, {"da/fortaelling": "da", "de/erzaehlung": "de", "en/story": "en", "fr/conte": "fr", "it/novella": "it"})
        self.assertEqual(self.row("story")["language"], "en")  # "English" in the metadata
        novella = self.row("novella")
        self.assertEqual((novella["author"], novella["year"], novella["prize"]), ("Autrice Prova", 1890, "yes"))
        self.assertEqual((novella["title"], novella["anno"]), ("Una novella", 1890))  # from its header
        self.assertEqual(self.row("fortaelling")["author"], "Forfatter Test")  # matched by file name alone
        self.assertIsNone(self.row("conte")["author"])  # no metadata row
        self.assertEqual(self.corpus.query("SELECT name FROM texts WHERE year < 1905 ORDER BY year")[1], [("it/novella",), ("de/erzaehlung",)])

    def test_reads_cp1252_and_drops_the_gutenberg_licence(self) -> None:
        body, _ = load_text(self.root / "da" / "fortaelling.txt")
        self.assertIn("lykkelig", body)
        body, _ = load_text(self.root / "en" / "story.txt")
        self.assertTrue(body.startswith("It was a long winter"))
        self.assertNotIn("Licence", body)
        result = self.corpus.count(["joy"])
        self.assertEqual(sum(t.hits for t in result.texts), 4)  # the licence's joy is not counted

    def test_counts_words_phrases_and_prefixes(self) -> None:
        result = self.corpus.count(["felic*", "la gioia", "gioia", "letizia"], ["happy", "joy", "joy", "gladness"])
        novella = next(t for t in result.texts if t.name == "it/novella")
        self.assertEqual(result.by_term["felic*"], 81)  # felicità and felice 40 times each, felici once
        self.assertEqual(result.by_term["letizia"], 1)
        self.assertEqual(novella.by_category["joy"], 40)
        self.assertEqual(novella.hits, 122)
        apostrophes = self.corpus.count(["l'amour", "c’était"])
        self.assertEqual(apostrophes.by_term, {"l'amour": 4, "c’était": 4})
        self.assertEqual(self.corpus.count(["GLÜCK*"]).by_term["GLÜCK*"], 8)  # Glück and glücklich, any case

    def test_segments_and_subsets(self) -> None:
        italian = self.corpus.select("language = 'it'")
        result = self.corpus.count(["letizia"], ids=italian, segments=10)
        self.assertEqual([t.name for t in result.texts], ["it/novella"])
        self.assertEqual(result.texts[0].segments, [0] * 9 + [1])  # the closing line is in the last tenth
        everything = self.corpus.count(["felice"], segments=4)
        self.assertEqual(sum(next(t for t in everything.texts if t.name == "it/novella").segments), 40)

    def test_the_same_counts_without_full_text_search(self) -> None:
        with mock.patch.object(corpus_module, "_has_fts5", return_value=False):
            plain = self.open(self.tmp / "plain-index")
        plain.update()
        self.assertFalse(plain.fts)
        for terms in (["felic*", "gioia"], ["glück*"], ["l'amour"], ["happiness", "joy"]):
            self.assertEqual(plain.count(terms).by_term, self.corpus.count(terms).by_term, terms)

    def test_concordance_spreads_its_sample(self) -> None:
        total, lines = self.corpus.concordance(["felice"], limit=5, context=20)
        self.assertEqual((total, len(lines)), (40, 5))
        self.assertEqual(sorted(lines, key=lambda l: (l.text, l.offset)), lines)
        self.assertGreater(lines[-1].position - lines[0].position, 30)  # not just the first five
        self.assertTrue(all(line.match.lower() == "felice" and len(line.left) <= 20 for line in lines))
        total, lines = self.corpus.concordance(["felice"], limit=0)
        self.assertEqual((total, lines), (40, []))

    def test_passages_are_cut_at_word_boundaries(self) -> None:
        offset = next(line.offset for line in self.corpus.concordance(["letizia"])[1])
        passage = self.corpus.passage("novella", start=offset + 3, length=40)
        body, _ = load_text(self.root / "it" / "novella.txt")
        self.assertTrue(passage["text"].startswith("letizia"))
        self.assertTrue(passage["end"] == len(body) or body[passage["end"]].isspace())
        self.assertIn(passage["text"], CLOSING)
        self.assertLessEqual(passage["end"] - passage["start"], 40)
        end = self.corpus.passage("novella", position=100, length=100)
        self.assertEqual(end["end"], end["chars"])
        with self.assertRaises(CorpusError):
            self.corpus.passage("t")  # matches several texts
        with self.assertRaises(CorpusError):
            self.corpus.passage("nothing like this")

    def test_sql_reads_the_catalogue_and_nothing_else(self) -> None:
        columns, rows, more = self.corpus.query("SELECT name, words FROM texts ORDER BY words DESC", limit=2)
        self.assertEqual((columns, rows[0][0], more), (["name", "words"], "it/novella", True))
        for sql in ("SELECT body FROM chunk_text", "SELECT * FROM docs", "DELETE FROM texts", "SELECT 1; DROP TABLE texts",
                    "ATTACH DATABASE 'x.db' AS x", "SELECT * FROM sqlite_master"):
            with self.assertRaises(CorpusError, msg=sql):
                self.corpus.query(sql)
        self.assertEqual(self.corpus.query("WITH t AS (SELECT language FROM texts) SELECT count(*) FROM t")[1], [(5,)])
        self.assertIsNone(self.corpus.select(None))
        with self.assertRaises(CorpusError):
            self.corpus.select("nonexistent_column = 1")

    def test_updates_only_what_changed(self) -> None:
        self.assertEqual(self.open(self.tmp / "index").update()["indexed"], 0)
        story = self.root / "en" / "story.txt"
        time.sleep(0.01)
        story.write_text("A new story full of joy and joy.", encoding="utf-8")
        os.utime(story, (time.time() + 5, time.time() + 5))
        (self.root / "fr" / "conte.txt").unlink()
        (self.root / "it" / "nuova.txt").write_text(ITALIAN, encoding="utf-8")
        report = self.corpus.update()
        self.assertEqual(report, {"texts": 5, "indexed": 2, "removed": 1})
        self.assertEqual(self.corpus.count(["joy"]).by_term["joy"], 2)
        self.assertNotIn("fr/conte", [t["name"] for t in self.corpus.texts()])

    def test_chunks_cover_the_text(self) -> None:
        body, _ = load_text(self.root / "it" / "novella.txt")
        spans = chunk_spans(body, target=300)
        self.assertGreater(len(spans), 3)
        self.assertEqual(spans[0][0], 0)
        self.assertEqual(spans[-1][1], len(body))
        self.assertTrue(all(a[1] == b[0] for a, b in zip(spans, spans[1:])))

    def test_bad_terms_are_refused(self) -> None:
        for term in ("", "*", "!!!"):
            with self.assertRaises(CorpusError):
                term_pattern(term)

    def test_describes_itself(self) -> None:
        text = self.corpus.describe()
        self.assertIn("5 texts", text)
        self.assertIn("Metadata from metadata.csv: author, year, prize, language.", text)
        self.assertIn("- year: 1890–1923 (4 texts)", text)
        self.assertIn("Languages: it 1 text (3,055 words)", text)


if __name__ == "__main__":
    unittest.main()
