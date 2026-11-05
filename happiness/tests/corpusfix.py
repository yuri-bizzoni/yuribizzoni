"""A small corpus in five languages for the tests, with a metadata table, a header and a Gutenberg licence."""

from __future__ import annotations

from pathlib import Path

ITALIAN = (
    "Era una mattina di primavera e la casa era piena di luce. Il vecchio maestro non aveva mai visto una gioia "
    "così grande negli occhi della bambina, che correva per il giardino con il cane. Per lui la felicità era una "
    "cosa semplice: un libro, una tazza di caffè e la voce della moglie che cantava in cucina. Ma quella mattina "
    "si sentì felice come non era stato da anni, e anche il cane sembrava allegro."
)
ENGLISH = (
    "It was a long winter, and the house was quiet. She had not felt such joy since she was a child, and she said "
    "so to her brother, who laughed at her. He was not a man who believed in happiness; he believed in work and in "
    "the weather. But that evening, as the snow fell on the garden, he too was happy, and he did not know why."
)
GERMAN = (
    "Es war ein kalter Morgen, und die Stadt lag still unter dem Schnee. Der alte Mann hatte das Glück nie gesucht, "
    "aber er fand es in den kleinen Dingen: in dem Brot, in der Stimme seiner Tochter und in dem Licht auf dem Fluss. "
    "Sie war glücklich, als sie ihn sah, und die Freude war in ihren Augen. Er sagte nichts, aber er war froh."
)
DANISH = (
    "Det var en stille aften, og huset var fyldt med lys. Den gamle mand havde aldrig søgt lykken, men han fandt den "
    "i de små ting: i brødet, i sin datters stemme og i lyset over havet. Hun var lykkelig, da hun så ham, og glæden "
    "var i hendes øjne. Han sagde ikke noget til hende, men han var glad, og det var nok for ham."
)
FRENCH = (
    "C'était un soir d'hiver, et la maison était pleine de lumière. Le vieil homme n'avait jamais cherché le bonheur, "
    "mais il le trouvait dans les petites choses : dans le pain, dans la voix de sa fille et dans la lumière sur la "
    "mer. Elle était heureuse quand elle le vit, et la joie était dans ses yeux. Il ne disait rien, mais il était "
    "content de l'amour qu'il voyait."
)
CLOSING = "Alla fine del libro, la letizia tornò nella casa e tutti furono felici per sempre."

METADATA = """file,author,year,prize,language
it/novella.txt,Autrice Prova,1890,yes,
story,Writer Test,1912,no,English
de/erzaehlung,Autor Test,1901,no,
fortaelling.txt,Forfatter Test,1923,yes,
"""


def make_corpus(root: Path) -> Path:
    """Write the corpus under root/corpus and return its folder."""
    corpus = root / "corpus"
    for folder in ("it", "en", "de", "da", "fr"):
        (corpus / folder).mkdir(parents=True)
    (corpus / "it" / "novella.txt").write_text(
        "# Una novella — Autrice Prova\n# anno: 1890\n\n" + "\n\n".join([ITALIAN] * 40) + "\n\n" + CLOSING + "\n",
        encoding="utf-8",
    )
    (corpus / "en" / "story.txt").write_text(
        "The Project Gutenberg eBook of A Story. This joy is not part of the text.\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK A STORY ***\n\n" + "\n\n".join([ENGLISH] * 4)
        + "\n\n*** END OF THE PROJECT GUTENBERG EBOOK A STORY ***\nLicence: joy, happiness and other words.\n",
        encoding="utf-8",
    )
    (corpus / "de" / "erzaehlung.txt").write_text("\n\n".join([GERMAN] * 4), encoding="utf-8")
    (corpus / "da" / "fortaelling.txt").write_text("\n\n".join([DANISH] * 4), encoding="cp1252")  # not UTF-8 on purpose
    (corpus / "fr" / "conte.txt").write_text("\n\n".join([FRENCH] * 4), encoding="utf-8")
    (corpus / ".hidden").mkdir()
    (corpus / ".hidden" / "skip.txt").write_text("hidden", encoding="utf-8")
    (corpus / "metadata.csv").write_text(METADATA, encoding="utf-8")
    return corpus
