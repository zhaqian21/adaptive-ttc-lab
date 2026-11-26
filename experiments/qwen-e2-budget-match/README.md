# Qwen E2 online comparison with nearby fixed budgets

Run Qwen2.5-7B-Instruct on the same 500 MATH-500 questions as PR #38. Use the
same E2 decoding configuration, question seed rule and 24 concurrent questions
per card. Each generation request has a 2,048 output-token cap.

- GPU 0: Fixed K=5, original highest mean-reward PRM selection.
- GPU 1: Fixed K=4, original highest mean-reward PRM selection.
- GPU 7: Dynamic K=3/5. Return the first answer if the first three extracted
  answers are identical and nonempty after whitespace stripping. Otherwise
  generate two more candidates and apply original PRM selection to all five.

Generation reuses the three equivalent resident vLLM engines. The runner checks
their pinned model revisions, serving options and idle request queues. All arms
use a local Qwen2.5-Math-PRM-7B replica on their own GPU. GPU 7 uses the resident
PRM. GPU 0 and GPU 1 start identical temporary PRM servers with the same request
queue size as GPU 7. The previous GPU 0 PRM is restored on exit.

## Reproduction on server 24

Use the pinned source checkout, model assets and data from PR #38. Link that
source and data into the run directory as `source` and `data`.
Use the vLLM 0.33.0 environment and run the command in a detached tmux session.

~~~bash
bash experiments/qwen-e2-budget-match/run_24.sh "$RUN_DIRECTORY" "$VLLM_PYTHON"
~~~

There is a four-question smoke test for each arm. Formal wall time starts when
the worker schedules the full cohort and ends after the final question
checkpoint. It includes generation, PRM scoring, and saving checkpoints. Model
loading and smoke are timed separately. A resumed cohort records its completed
checkpoint count and omits the uninterrupted wall time comparison.

The result archive contains the engine checks, effective decoding and model
configuration, all 1,500 predictions, candidate traces, and a paired summary
using fixed K=4 as the reference. Generation token totals include input and
output tokens. PRM requests are reported separately.
