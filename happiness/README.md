# Happiness

A research assistant for your own corpus, running on your own machine, with a lexicon of happiness built in.

Point it at a folder of plain-text files and ask questions in plain language. Do happiness words rise towards
the end of the Italian novels? Which authors write the longest sentences, and has that changed since 1850?
How does Deledda use *letizia*? Happiness (Claude underneath) plans the analysis, measures with tools that run
on your machine, checks a few passages, and answers with numbers you can trace back to a table, a script or a figure.

What it adds to Claude in a chat window:

- **Corpora of any size.** Texts are indexed locally (SQLite full-text search), so a question about thousands of
  novels doesn't need anything uploaded.
- **The texts stay on your disk.** Claude sees counts, tables, figures and short passages. Passages draw on a budget
  you set, 20,000 characters per conversation by default.
- **Real analysis.** It writes and runs Python (pandas, scipy, scikit-learn, statsmodels, matplotlib) in a Docker
  sandbox with no network and the corpus read-only.
- **A lab notebook.** Each conversation gets a folder with every question, every tool call, the tables, the scripts
  and the figures. Passages are logged by position, never copied.
- **Happiness built in.** It has the `felicita-it` lexicon from the *Machine Learning for the Humanities* course,
  and your happiness-lexicon skill for words in other languages. It can also build and check new lexicons, and save
  them for next time.

## Start

From the root of this repository, with Python 3.10 or newer:

```bash
pip install -r happiness/requirements.txt
```

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

```bash
python -m happiness --corpus ~/corpora/novels
```

The first run indexes the corpus; later runs only re-index files that changed. Inside the chat, `/new` starts
afresh, `/lab` shows the conversation's folder, `/budget` the passages left, and `/exit` leaves. Ctrl-C stops a reply.

Answering a single request works too:

```bash
python -m happiness --corpus ~/corpora/novels "Compare sentence length across decades, by language"
```

With [Docker](https://docs.docker.com/get-docker/) running, analysis scripts go in the sandbox. Happiness builds its
image the first time it needs it, which takes a few minutes. Without Docker, scripts run with your own Python, and
Happiness shows you each one and asks first.

## Your corpus

- **Texts:** `.txt` files anywhere under the folder (hidden folders are skipped). UTF-8, or Windows-1252/Latin-1
  for older files. A Project Gutenberg licence around a text is removed.
- **Metadata:** a `metadata.csv` (or `.tsv`) in the corpus folder, or any file given with `--metadata`, with one row
  per text. Happiness finds the column holding the file names by itself (`it/novella.txt`, `it/novella` or
  `novella` all work). Numbers are stored as numbers, so `year < 1900` works.
- **Headers:** leading lines starting with `#` are read as a header: `# key: value` lines become metadata, and
  `# Title — Author` gives the title and the author, as in the course texts.
- **Statistics:** every text gets `language` (detected for Italian, English, Danish, German and French unless the
  metadata says otherwise), `words`, `sentences`, `words_per_sentence`, `letters_per_word` and `sttr`
  (standardised type-token ratio over 1,000-word windows). They are approximate: sentences are split at final
  punctuation.

`python -m happiness index --corpus PATH` indexes without starting a conversation and shows what Happiness found.

## What it can do

| Tool | What it does | Corpus text sent |
|---|---|---|
| `catalogue` | SQL over the catalogue table `texts`: metadata and statistics, one row per text | none |
| `count` | Words, phrases, prefixes (`felic*`) or a lexicon, per text, per 10,000 words, per segment (arcs) and per group. Saves the full table | none |
| `concordance` | Keyword in context, sampled evenly from all matches | each line, from the budget |
| `read` | A passage of up to 1,500 characters | the passage, from the budget |
| `run_python` | A script in the sandbox, with the catalogue, the texts and the tables at hand | what it prints (told to print numbers only) |
| memory, lab editor, skills | Notes across conversations, the lab folder, the happiness lexicon skill | none |

`--web` adds web search and fetch, for facts about authors and editions. Corpus text never goes into a query.

## The lab

```
~/.happiness/
├── lab/
│   ├── 2026-11-05_1432-does-happiness-rise-towards-the-end/
│   │   ├── notebook.md        what you asked, every tool call, the answer
│   │   ├── tables/            count tables and the tables scripts write
│   │   ├── scripts/           every script that ran
│   │   ├── figures/
│   │   └── catalogue.csv      the catalogue as it was
│   └── lexicons/              lexicons you or Happiness made (category,term per row)
├── memory/                    what Happiness remembers between conversations
├── skills/                    your own skills, if you add any
└── index/                     the corpus indexes
```

Lexicons are CSV files with `category,term` per row (a header row is fine; a term ending in `*` is a prefix). Put
yours in `lab/lexicons/` and ask for them by name.

## Settings

| Flag | Environment variable | Default | |
|---|---|---|---|
| `--corpus` | `HAPPINESS_CORPUS` | | The folder of your texts (required) |
| `--metadata` | `HAPPINESS_METADATA` | `metadata.csv` in the corpus | One row per text |
| `--passages` | `HAPPINESS_PASSAGES` | `20000` | Characters of corpus text that may go to Claude per conversation |
| `--sandbox` | `HAPPINESS_SANDBOX` | `auto` | `docker`, `local` (your Python, with approval) or `auto` |
| `--yes` | | | With `local`: run scripts without asking |
| `--model` | `HAPPINESS_MODEL` | `claude-opus-5-5` | `claude-sonnet-5-5` or `claude-haiku-5-5` cost less |
| `--effort` | `HAPPINESS_EFFORT` | `medium` | `low`, `medium`, `high`, `xhigh` or `max` |
| `--web` | `HAPPINESS_WEB` | off | Web search and fetch |
| `--lab` | `HAPPINESS_LAB` | `~/.happiness/lab` | Notebooks, tables, scripts, figures |
| `--home` | `HAPPINESS_HOME` | `~/.happiness` | Index, memory and skills |

## From Python or a notebook

```python
from pathlib import Path
from happiness import Agent, Settings

agent = Agent(Settings(corpus=Path("~/corpora/novels").expanduser()))
answer = agent.ask("Which decade has the most happiness words per 10,000 words?")
```

`agent.lab.session` is the folder with the tables and figures behind the answer. Without Docker, it shows you each
script and asks before running it, as in the terminal; `Agent(..., approve=run_without_asking)` (from
`happiness.sandbox`) skips that.

## What to keep in mind

- **The passage budget covers `concordance` and `read`.** What an analysis script prints is cut at 6,000
  characters per run. Happiness is told to print numbers, not text, and every script is saved so you can check.
- **Counting is lexical.** Each occurrence is counted once, as whole words, ignoring case. Negation, irony and
  other senses are not handled, which is why Happiness checks counts with concordances. Segments are equal slices
  of a text by length.
- **Indexing speed.** On the machine this was built on, 300 texts (18 million words) indexed in about 20 seconds,
  and counts over all of them took one to three seconds.
- **Costs.** Every request is billed to your API key. Conversations are prompt-cached, which makes long sessions
  much cheaper.
- **Refusal fallback is on.** Requests carry `fallbacks: "default"` (beta `server-side-fallback-2026-07-01`), so if
  the model declines on safety grounds, the API retries on the fallback model Anthropic recommends, in the same
  call. Remove those two lines in `Agent.request()` if you'd rather not.

## How it works

- [`corpus.py`](corpus.py): reading texts, the catalogue, the index (SQLite with FTS5 when available, a plain scan
  otherwise), counts, concordances and passages.
- [`tools.py`](tools.py): the tools Claude calls, the passage budget, and the notebook entries.
- [`sandbox.py`](sandbox.py) and [`sandbox_image/`](sandbox_image/Dockerfile): where scripts run.
- [`lab.py`](lab.py): the lab folders and notebooks.
- [`agent.py`](agent.py): the loop. Each step streams a reply from the Messages API with adaptive thinking, runs
  the tools it asks for and sends the results back. The instructions and tools are fixed for a conversation and
  the history only grows, which keeps the prompt cache warm.
- [`__main__.py`](__main__.py): the command line.

## Tests

From the repository root:

```bash
python -m unittest discover -s happiness/tests -t .
```

They run offline on a small corpus in five languages. The real SDK talks to a scripted stand-in for the API, so
indexing, counting, the budget, the sandbox, the notebook and the agent loop are all exercised without a key.
