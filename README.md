# cici

A coding-agent CLI in Python — a minimal Claude Code, built from scratch.

Layering follows [pi](https://github.com/earendil-works/pi) (a TypeScript agent monorepo),
reimplemented in Python.

| pi package | cici module | responsibility |
|---|---|---|
| `pi-ai` | `cici/llm.py` | unified LLM layer (Anthropic first, provider seam kept) |
| `pi-agent-core` | `cici/agent.py`, `cici/session.py` | agent loop, tool calling, state |
| `pi-coding-agent` | `cici/cli/repl.py` | interactive CLI |
| `pi-tui` | `cici/tui.py` | terminal rendering |
| `pi-telemetry` | `cici/telemetry.py` | per-turn telemetry |

## Quickstart

```bash
uv sync                            # creates .venv, installs deps + the project
cp .env.local.example .env.local   # fill in ANTHROPIC_API_KEY
uv run cici
```

Managed with [uv](https://docs.astral.sh/uv/). `uv sync` pins the Python version
(3.12, from `.python-version`) and resolves against `uv.lock`, so the environment
is reproducible — no "works on my machine".

## Status

Early scaffold. See `CLAUDE.md` for architecture, roadmap, and the honest boundaries
on what is and isn't built yet.
