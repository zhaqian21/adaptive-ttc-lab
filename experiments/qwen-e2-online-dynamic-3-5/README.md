# Qwen E2 dynamic K=3/5 online comparison

All three arms run on the same 500 MATH-500 questions with Qwen2.5-7B-Instruct, the frozen public-code E2 decoding settings, the original seed rules, and a 2,048 output token cap per generation call. Each arm uses one H20 and 24 concurrent questions. Both E2 arms use the original mean-reward PRM selection and the same PRM weights.

- GPU 0 runs fixed E2 with 10 candidates and PRM selection.
- GPU 1 runs dynamic E2. Return the first answer if the first three extracted answers match after stripping whitespace and the answer is nonempty. Otherwise generate two more candidates and use PRM to select from all five.
- GPU 7 runs Direct with five candidates and majority voting. Empty answers do not receive votes. Use the earliest candidate to break a tie.

The archived E2 pool replay on Qwen gave 403/500 for dynamic K, 407/500 for fixed E2, and estimated savings of 61.4% in generation tokens and 81.1% in PRM requests. The online comparison measures new samples and real elapsed time. Full arm time starts when the question executor starts and ends when all question checkpoints are written. It includes generation and PRM selection, and excludes model loading and the four-question smoke. Report resumed runs separately for wall time.

## Run on server 24

Use the pinned source and data from the 2,048 token E2 job. Run from a detached server session with the vLLM 0.33.0 Python environment. The launcher checks that the resident services are idle and restores them once the experiment exits.

~~~bash
bash experiments/qwen-e2-online-dynamic-3-5/queue_24.sh "$RUN_DIRECTORY" "$VLLM_PYTHON"
~~~

The queue runs the four-question smoke and then all 500 questions per arm. Each question has an atomic checkpoint. The completed export contains a summary, matched results, and compressed candidate traces. The three arms share sampling parameters, but the Direct prompt and voting rule follow the Direct baseline.

The result audit runs after generation completes, even if a resident service needs later restoration. The queue records the restoration status. If GPU 7 has a different active job, leave it running and defer the original service restart.
