"""Trajectory-level evaluation — roadmap 4, the differentiator.

cici_101 had a prompt-eval harness (02_prompt_eval.py): two graders, an
LLM-as-judge forced into structured JSON via assistant prefill + stop sequence,
combined with deterministic AST/JSON/regex validators. That scores a SINGLE
TURN of output. It is not agent evaluation and must not be described as such.

What goes here instead: give the agent a task set of the shape
"change Y to Z in file X", then score three things —

  1. did the task actually complete   (deterministic grader reads the file back)
  2. how many turns did it take
  3. how many tokens did it cost

The distinction that matters: "I built an agent" and "I can tell whether my
agent is any good" are different claims, and only the second one needs numbers
behind it. Completion is the one a code grader can settle on its own; turns and
tokens are what separate an agent that succeeds from one that flails its way
there.

TODO: port the two-grader scaffolding from cici_101/02_prompt_eval.py, then
build the task dataset under evals/datasets/.
"""
