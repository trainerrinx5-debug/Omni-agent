# Omni Agent

A small, local-first **tool-using agent scaffold** built from the Omni Agent brief. It runs as a CLI, connects to an OpenAI-compatible Chat Completions endpoint, can inspect and edit a dedicated workspace, and follows a bounded plan → tool → observe loop.

This is a working starter project, not a claim of unlimited autonomy. It does not run shell commands, access accounts, make purchases, schedule future work, or read files outside its configured workspace.

## Quick start

Requirements: Python 3.10+ and a model endpoint that supports OpenAI-style function/tool calling.

### Local Ollama

From this project directory:

```bash
ollama pull qwen2.5:7b
# Start Ollama if it is not already running, then in another terminal:
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
python -m pip install -e .
python -m omni_agent
```

The defaults are `http://localhost:11434/v1` and `qwen2.5:7b`. The first launch creates `./workspace` and `./.omni`. The default model must be available in your Ollama installation; change `OMNI_MODEL` if you use another tool-capable model.

### Another OpenAI-compatible provider

Set the endpoint, model, and (if required) API key in your shell before launching:

```bash
export OMNI_BASE_URL="https://your-provider.example/v1"
export OMNI_MODEL="your-tool-capable-model"
export OMNI_API_KEY="..."
python -m omni_agent
```

PowerShell equivalents use `$env:OMNI_BASE_URL = "..."`, etc. `.env.example` documents the supported variables; it is a reference file and is **not** automatically loaded. Remote endpoints must use HTTPS. Plain HTTP is accepted only for loopback/local model servers.

You can also install the `omni-agent` command with `python -m pip install -e .`.

## Use

```text
you> Calculate the total for 3 items at 19.50 each, then write a short receipt to receipt.txt.
```

The calculator is read-only. A proposed file write triggers an approval prompt. To allow every workspace write for a run, use `--yes` deliberately:

```bash
python -m omni_agent --yes
```

Use `--yes` only with a trusted model/provider and a disposable or backed-up workspace. It disables the write confirmation; it does not expand the workspace boundary.

Useful commands:

- `/help` — show CLI commands
- `/tools` — list enabled tools
- `/memory` — list saved notes
- `/remember TEXT` — save a note after an additional confirmation
- `/forget ID` — remove one note
- `/clear` — clear in-process conversation history
- `/exit` — quit

One-shot request:

```bash
python -m omni_agent --prompt "Summarize the files in my workspace"
```

Diagnostics that do not contact a provider:

```bash
python -m omni_agent --doctor
python -m omni_agent --list-tools
```

## Included capabilities

- **Bounded tool loop:** up to 8 rounds by default (configurable from 1 to 12), with a per-turn tool-call cap and trimmed conversation history.
- **Calculator:** arithmetic-only AST evaluator; no Python `eval`, names, imports, or function calls.
- **Workspace listing and reading:** relative paths only, UTF-8 text only, with size limits.
- **Workspace writing:** new or existing files only under the workspace root; confirmation per write unless the explicit `--yes` flag is used.
- **Optional web search:** set `BRAVE_SEARCH_API_KEY` to enable `web_search`. It returns a small set of Brave result titles, URLs, and snippets; it does not fetch arbitrary URLs. Search queries are sent to Brave.
- **Explicit local memory:** notes are created/deleted only by the user's CLI commands, never automatically. Notes are included as context in later model requests.

## Safety and privacy boundaries

- Tools are allowlisted; there is **no shell, code execution, arbitrary URL fetch, account integration, payment, or background scheduling**.
- File tools resolve paths inside the configured workspace and reject traversal and symlink escapes. Do not put credentials or files you would not want the model to see in that workspace.
- Each file write is approval-gated by default. In non-interactive mode, writes are denied unless `--yes` is explicitly supplied.
- Prompts, chat history, tool outputs, and saved memory notes are sent to the configured model provider. If web search is enabled, the query is also sent to Brave. Use only providers and data you trust.
- Memory is local in `.omni/memory.json` with restrictive file permissions where supported, but it is **not encrypted**. Do not store secrets, passwords, tokens, or sensitive personal data.
- File/path checks are an application boundary, not an OS sandbox. Keep the workspace narrow and back up important files.
- Web pages, search snippets, and workspace file contents are untrusted data; the agent is instructed not to follow embedded instructions from them.

## Configuration

| Variable / flag | Default | Purpose |
|---|---|---|
| `OMNI_BASE_URL` / `--base-url` | `http://localhost:11434/v1` | OpenAI-compatible API base URL |
| `OMNI_MODEL` / `--model` | `qwen2.5:7b` | Model name |
| `OMNI_API_KEY` | unset | Optional provider key; never printed by diagnostics |
| `OMNI_WORKSPACE` / `--workspace` | `./workspace` | Root allowed for file tools |
| `OMNI_STATE_DIR` / `--state-dir` | `./.omni` | Local memory location |
| `OMNI_MAX_STEPS` / `--max-steps` | `8` | Maximum tool rounds per request (1–12) |
| `OMNI_TIMEOUT_SECONDS` | `90` | Model request timeout |
| `BRAVE_SEARCH_API_KEY` | unset | Enables optional web search |
| `--yes` | off | Explicitly auto-approve workspace writes |

## Tests

The project has no runtime third-party dependencies. Run its standard-library test suite with:

```bash
python -m unittest discover -s tests -v
```
