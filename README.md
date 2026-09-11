<p align="center">
  <img src="docs/logo.gif" alt="cici" width="440">
</p>

<p align="center">
  <sub>a small, transparent coding agent — you can see right through it</sub>
</p>

<h1 align="center">cici</h1>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white" alt="Python 3.12">
  <img src="https://img.shields.io/badge/license-MIT-black" alt="MIT">
  <img src="https://img.shields.io/badge/model-claude--opus--5-D97757" alt="Claude Opus 5">
</p>

<p align="center">
  The whole harness is ~1,650 lines of Python. No framework.<br>
  Six tools, a streaming loop, per-turn telemetry — small enough to read end to end in an afternoon.
</p>

<p align="center">
  <img src="docs/demo.gif" alt="cici answering a question about its own source" width="880">
</p>

## Quickstart

```bash
uv sync                            # creates .venv, installs deps + the project
cp .env.local.example .env.local   # add ANTHROPIC_API_KEY
uv run cici                        # start it in whatever project you want to work on
```

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). `.python-version`
pins the interpreter and `uv.lock` pins the dependencies, so the environment is
reproducible.

cici is meant to be launched **inside the project you are working on** — that
directory becomes its workspace, and the file tools cannot reach outside it.
The API key is resolved against cici's own repo root, not the working
directory, so it keeps working wherever you start it.

`/exit` or Ctrl-C to quit. `CICI_DEBUG=1` dumps the message array each turn;
`CICI_YOLO=1` skips the confirmation prompt on shell commands.

## Tools

Six tools with deliberately non-overlapping jobs. Two tools that can do the
same thing is a reliable way to make a model pick the wrong one.

| tool | job | parameters |
|---|---|---|
| `ls` | find files by name | `path`, `depth` |
| `grep` | find files by content | `pattern`, `path`, `glob`, `output_mode` |
| `read` | view a file, with line numbers | `path`, `offset`, `limit` |
| `write` | create a file, or replace one | `path`, `content` |
| `edit` | change part of a file | `command`, `path`, `old_str`, `new_str`, `insert_line` |
| `bash` | everything else | `command`, `timeout` |

Every schema is hand-written, one description per parameter. Those strings are
the entire interface the model sees — it never sees the code — so they are
prompt engineering, not type declarations, and generating them from type hints
would throw away the part that determines whether a tool gets called correctly.

## How the loop works

```
your input
   └─ Agent.run(query)
        ├─ _process_query      → append to the session as a user turn
        ├─ registry.schemas()  → the tool JSON sent with every request
        ├─ provider.stream()   → render text and reasoning as they arrive
        ├─ session.add_assistant(content blocks, verbatim)
        ├─ stop_reason != "tool_use" ? → done
        └─ registry.run_all()  → execute, return tool_results, loop
```

Up to 20 turns. Each one prints a telemetry line — stop reason, tool count,
latency, tokens — because when an agent misbehaves the conversation record is
the only evidence there is.

### Layers

| module | responsibility |
|---|---|
| `llm.py` | the only module that imports the Anthropic SDK. Streaming, unwrapping responses, closing the pool |
| `agent.py` | the turn loop, stream rendering, tool dispatch |
| `session.py` | conversation state. Stores content, never provider objects |
| `tools/base.py` | the `Tool` contract, `SyncTool`, and the registry that contains failures |
| `tools/workspace.py` | path confinement, undo snapshots, the shared walk |
| `telemetry.py` | per-turn record |
| `tui.py` | spinner, banner, dimmed reasoning, TTY detection |
| `cli/repl.py` | the interactive shell |

## Design notes

A few decisions that were not obvious. The full record — including the ones
that turned out wrong, and what this project's layering was informed by — is in
[`docs/decisions.md`](docs/decisions.md).

**A failing tool is never an exception the loop sees.** `Registry.run_all`
catches per tool and returns an `is_error` tool result, so the model gets a
chance to correct itself and the loop stays alive. The except clause catches
`Exception`, never `BaseException` — `asyncio.CancelledError` has to propagate
for Ctrl-C to work, while a bash timeout is contained and reported.

**Error messages are written for the model, not for a debugger.** "old_str
appears 3 times — include more surrounding lines so exactly one location
matches" is a usable instruction. "no match" is not. The same applies to
truncation: every capped result says how much was left and how to ask for the
rest, because otherwise the model assumes it saw everything.

**Tools are async by contract, `SyncTool` absorbs the difference.** File tools
implement a plain synchronous `_run` and never think about the event loop;
`asyncio.to_thread` keeps them off it. `bash` implements `run` directly, so an
`asyncio.timeout` can actually kill a hung command's whole process group
instead of blocking the process and orphaning children.

**Path confinement compares components, not prefixes.** Everything the file
tools touch goes through `Workspace.resolve`, which resolves first — catching a
symlink by its target rather than its name — and then checks
`Path.is_relative_to`. A `startswith` test would accept `/home/me/app-secrets`
when the root is `/home/me/app`.

**The REPL owns the event loop with `asyncio.Runner`, not `asyncio.run`.**
`asyncio.run` installs a SIGINT handler that turns the first Ctrl-C into a task
cancellation, which a task parked in a blocking `input()` can never receive —
measured on a pty, that hangs. Wrapping `input()` in `to_thread` hangs for a
different reason.

**`str_replace` refuses an ambiguous match.** Replacing a string that occurs
twice corrupts the file in a way the model cannot see, so 0 or 2+ matches is an
error rather than a best guess.

## Status

Working: the agent loop, streaming with summarized reasoning, all six tools,
per-turn telemetry, 59 tests that never touch the API.

Not built yet, roughly in the order they are worth doing:

1. **prompt caching** — a cache breakpoint on the system prompt and the tool
   array, with the hit rate in the telemetry line. Six tool schemas now go out
   on every turn; one debugging session measured 41k input tokens by turn 20
2. **session persistence and context management** — what to keep, summarise,
   drop; `@file` mentions
3. **trajectory evaluation** — score a run on whether the task completed, how
   many turns it took, and what it cost. The interesting first experiment is
   hand-written schemas against the Anthropic-defined tool definitions
4. **parallel tool execution** — the model already issues two read-only calls
   per turn routinely; `run_all` still runs them in sequence
5. MCP, sub-agents

`bash` is **not a sandbox**. It runs commands as you, with your permissions, and
unlike the file tools it is not confined to the workspace — the confirmation
prompt is the only thing between the model and the machine. `CICI_YOLO=1`
removes even that, and exists for automated runs.

Failures worth studying are recorded in
[`evals/badcases.md`](evals/badcases.md), including one cici could not solve.

## Development

```bash
uv run pytest -q            # 59 tests, ~1.4s, no API calls
uv run ruff check src tests
uv run ruff format src tests
```

Tests construct tools against a `tmp_path` workspace, so they exercise the real
filesystem behaviour — confinement, snapshots, truncation — without mocking it.
