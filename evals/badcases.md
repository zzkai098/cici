# Badcases

Failures worth keeping. A case earns a place here when the task had a knowable
right answer and cici did not reach it — not when the model was merely slow or
wordy.

Two reasons to write them down:

1. **They are the eval set.** Roadmap 4 scores trajectories on three axes —
   did the task complete, how many turns, how many tokens. Cases with a
   verifiable answer are the only ones a code grader can score, and a case that
   already failed is worth more than one invented to pass.
2. **Most fixes are not model fixes.** Nearly every entry below traces back to
   something the tool surface did not expose, or an error message that did not
   say what to do next. That is a property of this repo, not of Claude.

## Format

Keep it short and factual. Guessing at a root cause is worse than leaving it
open — mark it `unresolved` and move on.

```
## NNN — one-line summary
- date / model / turns / input tokens at the end
- task:      what was asked
- expected:  the verifiable right answer
- actual:    what cici did instead
- root cause: why it could not get there
- fix:       what would change, or `unresolved`
```

---

## 001 — cannot diagnose a failing editable install

- 2026-09-10 · claude-opus-5 · 6 turns (15-20) · ~41k input tokens · abandoned by the user
- **task**: why does `uv run cici` fail with `ModuleNotFoundError: No module named 'cici'`
- **expected**: the `.pth` file in site-packages carries the macOS `UF_HIDDEN`
  flag, and Python 3.12's `site.addpackage` silently skips hidden `.pth` files,
  so `src` never reaches `sys.path`. Verifiable with `ls -lO`.
- **actual**: checked the `.pth` contents, the venv layout, the egg-info, and
  `sys.path`. All of them looked correct, because they are. Guessed "stale path
  from before the project moved", then "Desktop vs desktop case mismatch".
  Both wrong. Never ran `ls -lO` or `stat`, and did not reach the flag.
- **root cause**: the evidence is not visible through any tool cici has. `ls`
  does not print file flags; nothing in the tool surface would lead a model to
  suspect one. The agent's ceiling here was the tool surface, not the model.
- **fix**: `unresolved`. Candidates, none obviously right:
  - have `ls` surface unusual file flags — very narrow, probably not worth a
    permanent column for a macOS-only quirk
  - accept it: some root causes need a human who has seen that failure before,
    and the honest lesson is that an agent is bounded by what its tools reveal

### What this case also exposed

Not the failure itself, but things visible in the transcript that are worth
fixing on their own:

- **context growth** — `in=` reached 41k. Every turn resends the full history
  plus all six tool schemas. This is the concrete argument for roadmap 2
  (prompt caching), measured rather than assumed.
- **confirmation fatigue** — six `y` presses in six turns. bash is the right
  tool for read-only investigation, and confirming every `ls` and `cat` makes
  debugging painful. Loosening it is a real tradeoff, not an obvious win: a
  read-only allowlist is exactly the surface an agent would route around.
- **`cd . && uv run pytest`** — the redundant `cd .` says the bash description's
  claim about the working directory is not landing. A description problem, and
  a cheap fix.
- **bash is not confined to the workspace** — it wrote into `/tmp/bt`. Every
  file tool goes through `Workspace.resolve`; bash does not, and human
  confirmation is the only thing in its way. Deliberate, but it should be
  stated as a decision rather than discovered in a transcript.
