# Happiness

A small agent you run yourself, for all sorts of everyday tasks, with a lexicon of happiness built in.

Happiness talks to Claude (Opus 5.5 by default) and can:

- **search the web and read pages**, saying where its facts come from;
- **remember you** between conversations, in a folder of plain notes on your machine;
- **read your files** (text, PDFs, images) and **write or edit text files**, all inside one workspace folder;
- **load skills**, folders of instructions it reads when a task calls for them. It ships with one: the
  **happiness lexicon**, words for joy, contentment and longing in about twenty languages, with register,
  etymology and literary loci, plus a cross-index that maps the stages of a research project to the kind of
  happiness each one runs on (a gentle diagnostic when a project stalls);
- **run shell commands**, if you allow it: terminal only, and you approve each command.

You talk to it in a terminal, or in a small web chat that also works on a phone. It is a few short Python files
with a single dependency, the Anthropic Python SDK.

## Start

From the root of this repository, with Python 3.10 or newer:

```bash
pip install -r happiness/requirements.txt
```

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

```bash
python -m happiness
```

That opens a conversation. `/new` starts afresh, `/exit` leaves, Ctrl-C stops a reply. Get an API key from the
[Claude Console](https://console.anthropic.com).

One request at a time works too. Text piped in is added to the request:

```bash
python -m happiness "Is there a word for the joy of watching students finally get it?"
```

```bash
cat chapter.txt | python -m happiness "Summarise this in five lines, in Italian"
```

```bash
python -m happiness --workspace ~/Documents/thesis "Read draft.pdf and list the open questions"
```

```bash
python -m happiness --shell "How many words are in each .txt file here?"
```

## The web chat

```bash
python -m happiness serve
```

Then open <http://localhost:8000>. Attach files with **+**; they are saved in the workspace's `uploads/` folder,
where Happiness can read them. To serve beyond your own computer, set a password first: the server refuses to
listen on another address without one.

```bash
HAPPINESS_PASSWORD='a long passphrase' python -m happiness serve --host 0.0.0.0
```

The browser asks for it (any user name works). Anywhere outside your home network, put HTTPS in front, as below:
the password travels with every request.

## Deploy it

### With Docker, on any machine

```bash
cd happiness
```

```bash
docker build -t happiness .
```

```bash
docker run -d --name happiness --restart unless-stopped -p 8000:8000 \
  -e ANTHROPIC_API_KEY=sk-ant-... -e HAPPINESS_PASSWORD='a long passphrase' -e TZ=Europe/Copenhagen \
  -v happiness-data:/data happiness
```

Memory, skills and the workspace live in the `happiness-data` volume, so they survive upgrades.

### On a server with a domain name, with HTTPS

[`compose.yaml`](compose.yaml) runs Happiness behind [Caddy](https://caddyserver.com), which gets and renews
the certificate by itself. Point the domain at the server, open ports 80 and 443, and create a `.env` file next
to `compose.yaml` (it is git-ignored):

```
ANTHROPIC_API_KEY=sk-ant-...
HAPPINESS_PASSWORD=a long passphrase
HAPPINESS_DOMAIN=happiness.example.org
TZ=Europe/Copenhagen
```

```bash
docker compose up -d --build
```

## Settings

| Environment variable | Flag | Default | |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | | | Your Claude API key (required) |
| `HAPPINESS_MODEL` | `--model` | `claude-opus-5-5` | Any Claude model; `claude-sonnet-5-5` and `claude-haiku-5-5` cost less |
| `HAPPINESS_EFFORT` | `--effort` | `medium` | How hard it thinks: `low`, `medium`, `high`, `xhigh`, `max` |
| `HAPPINESS_HOME` | `--home` | `~/.happiness` | Where memory, your skills and the workspace live |
| `HAPPINESS_WORKSPACE` | `--workspace` | `<home>/workspace` | The only folder it may read and write |
| `HAPPINESS_WEB` | `--no-web` | on | Web search and web fetch |
| `HAPPINESS_SHELL` | `--shell`, `--yes` | off | Shell commands, terminal only; `--yes` skips the approval |
| `HAPPINESS_PASSWORD` | | | Required to serve the web chat beyond localhost |
| `HAPPINESS_HOST`, `HAPPINESS_PORT` | `--host`, `--port` | `127.0.0.1`, `8000` | Where the web chat listens |
| `TZ` | | the system's | The time zone behind "today" |

Web search must be enabled for your organization in the Claude Console; if it isn't, turn the web tools off with
`--no-web`. Each search is billed on top of the tokens.

## Where your things live

```
~/.happiness/
├── memory/memories/   what Happiness remembers about you: plain files you can read, edit or delete
├── skills/            your own skills
└── workspace/         the files it may read and write; uploads go to workspace/uploads/
```

## Skills

A skill is a folder with a `SKILL.md` file: a short header, then instructions. Happiness sees each skill's name
and description, and loads the whole file when a task calls for it. Add your own as
`~/.happiness/skills/<name>/SKILL.md`:

```markdown
---
name: course-emails
description: Emails to the students of Machine Learning for the Humanities. Use for any message to them.
---
Write in Italian, warmly and briefly. Sign as Yuri. Put dates as "mercoledì 12 novembre".
```

A skill of yours with the same name as a bundled one replaces it. The bundled lexicon is
[`skills/happiness-lexicon/SKILL.md`](skills/happiness-lexicon/SKILL.md): edit it to extend the lexicon.

## Safety

- **Files.** The workspace tool cannot leave the workspace, through `..`, absolute paths or symbolic links.
  Memory stays in its folder, and skills are read-only. Overwriting a file keeps the old version in
  `workspace/.backups/`.
- **Shell.** Off unless you ask for it, never available in the web chat, and each command needs your approval
  unless you pass `--yes`. Commands run in the workspace without the `ANTHROPIC_*` and `HAPPINESS_*` variables, so
  your keys stay out of reach. It is not a sandbox: an approved command can do anything your user can.
- **Web chat.** It needs a password beyond localhost, and the page's API calls carry a header that other sites
  cannot send. A strict Content Security Policy is set, and model output is escaped before it is rendered.
- **What it reads.** Web pages and files can contain instructions aimed at an AI. Happiness treats them as
  material, not orders, but stay aware of it, especially with the shell on.
- **Costs.** Every request is billed to your API key. The conversation is prompt-cached, which makes long chats
  much cheaper; `/new` (or **New chat**) starts a short one.

## How it works

- [`agent.py`](agent.py) holds the loop. Each step streams a reply from the Messages API
  (`client.beta.messages.stream`) with adaptive thinking and your effort level. If the reply asks for tools,
  Happiness runs them and sends the results back, until Claude is done. Long server-side searches that pause are
  resumed, a tool call cut off by the output limit is not run, and if Claude's safety checks decline a request, the
  exchange is dropped so the conversation can go on.
- **Refusal fallback is on**: requests carry `fallbacks: "default"` (beta `server-side-fallback-2026-07-01`), so if
  the model declines on safety grounds, the API retries on the fallback model Anthropic recommends, within the same
  call. Remove those two lines in `Agent.request()` if you'd rather not.
- The instructions and the tool list are fixed when a conversation starts, and the history only ever grows. That
  keeps the prompt cache warm and every earlier turn valid.
- [`tools.py`](tools.py) holds the local tools: the workspace (Anthropic's text editor tool), memory (the SDK's
  local memory tool), `load_skill`, and the optional shell (Anthropic's bash tool). Web search and fetch run on
  Anthropic's side.
- [`web.py`](web.py) is the web chat (standard library only; replies stream as server-sent events), and
  [`__main__.py`](__main__.py) is the command line.

## Tests

From the repository root:

```bash
python -m unittest discover -s happiness/tests -t .
```

They run offline: the real SDK talks to a scripted stand-in for the API, so requests, streaming, tools, refusals,
fallbacks, the web server and uploads are all exercised without a key.
