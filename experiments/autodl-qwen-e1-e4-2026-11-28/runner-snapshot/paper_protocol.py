"""Table 1 adaptations of the pinned agent and Qwen process reward model."""

import math
import re
from typing import NamedTuple

from src.Evaluator import MATHEvaluator
from src.help_functions.compute_strategies import ComputeStrategy
from src.help_functions.prm_selector import PRMSelector
from src.universal_agent import UniversalAgent


PRM_SYSTEM_PROMPT = r"Please reason step by step, and put your final answer within \boxed{}."


def split_steps(text):
    """Use explicit markers, then paragraph breaks, then nonempty lines."""
    text = text.replace("\r\n", "\n").strip()
    if "<extra_0>" in text:
        parts = text.split("<extra_0>")
    else:
        parts = re.split(r"\n[ \t]*\n+", text)
        if len(parts) == 1:
            parts = text.splitlines()
    return tuple(part.strip() for part in parts if part.strip())


class PaperAgent(UniversalAgent):
    def _execute_direct(self, problem, prompts):
        messages = [
            {"role": "system", "content": prompts["direct_unstructured_final_answer_system_prompt"]},
            {"role": "user", "content": prompts["direct_unstructured_final_answer_user_prompt"].format(problem=problem)},
        ]
        completion = self.client.generate(messages).completion
        answer = MATHEvaluator()._extract_answer_from_model_completion(completion) or ""
        compute_config = {"strategy": "none", "param": 0}
        self.tools_used = ["direct"]
        self.compute_configs_used.append(compute_config)
        self.compute_metadata_history.append({
            "tool": "direct", "compute_strategy": "direct", "compute_config": compute_config,
        })
        self.compute_strategy_outputs.append({"strategy": "none", "output": "Single-pass direct solve"})
        self.reasoner_outputs.append({"reasoner": "direct", "output": completion})
        return completion, list(split_steps(completion)), answer, completion


class PaperPRM(PRMSelector):
    def make_step_rewards(self, logits, token_masks):
        positive_probs = logits.float().softmax(dim=-1)[..., 1]
        return [row[mask].cpu().tolist() for row, mask in zip(positive_probs, token_masks)]

    def combine_fields(self, entry):
        steps = tuple(step for item in entry["reasoning_steps"] for step in split_steps(item))
        completion = entry["answer_unstructured"]
        if completion != entry["reasoning"] or not steps:
            steps += split_steps(completion)
        return list(steps)

    def score_reasoning(self, problem, reasoning_steps, system_prompt):
        return self._score_steps(problem, reasoning_steps, system_prompt)

    def score_search(self, problem, steps):
        return self._score_steps(problem, steps, PRM_SYSTEM_PROMPT)

    def _score_steps(self, problem, reasoning_steps, system_prompt):
        if not reasoning_steps:
            return {"step_rewards": [], "mean_reward": 0.0, "min_reward": 0.0, "final_reward": 0.0, "num_steps": 0}
        scores = super().score_reasoning(problem, reasoning_steps, system_prompt)
        rewards = scores["step_rewards"]
        if len(rewards) != len(reasoning_steps) or not all(
            math.isfinite(reward) and 0 <= reward <= 1 for reward in rewards
        ):
            raise RuntimeError(f"Qwen Math PRM returned invalid rewards for {len(reasoning_steps)} steps")
        return scores


class Action(NamedTuple):
    text: str
    steps: tuple


class PaperSearch(ComputeStrategy):
    """Generate candidate continuations with the upstream reasoners and rank by mean step reward."""

    def __init__(self, strategy, prm, rng=None):
        super().__init__(prm, rng=rng)
        self.strategy = strategy

    def execute(self, reasoner, messages, obs, plan, config, instruction_prompt, prm_scoring_prompt):
        generated = 0
        ranked_rounds = []

        def generate(prefix=()):
            nonlocal generated
            context = messages
            if prefix:
                history = "\n\n".join(step for action in prefix for step in action.steps)
                context = [*messages, {"role": "user", "content": f"Continue from these reasoning steps:\n{history}"}]
            text, reasoning = self._call_reasoner(reasoner, context, obs, plan, instruction_prompt)
            generated += 1
            steps = split_steps(reasoning)
            if text and text not in steps:
                steps += split_steps(text)
            return (*prefix, Action(text, steps))

        def rank(paths, rollout_start=0):
            scored = []
            for path in dict.fromkeys(paths):
                steps = [step for action in path for step in action.steps]
                scores = self.prm_model.score_search(obs, steps)
                prefix_steps = sum(len(action.steps) for action in path[:rollout_start])
                rewards = scores["step_rewards"][prefix_steps:]
                reward = sum(rewards) / len(rewards) if rewards else 0.0
                scored.append((path, {
                    "actions": [action.text for action in path],
                    "steps": steps,
                    "prefix_steps": prefix_steps,
                    "score": reward,
                    "step_rewards": scores["step_rewards"],
                }))
            if self.strategy == "best_of_n":
                self.rng.shuffle(scored)
            scored.sort(key=lambda item: item[1]["score"], reverse=True)
            ranked_rounds.append([trace for _, trace in scored])
            return [path for path, _ in scored]

        param = int(config["param"])
        if self.strategy == "beam_search":
            width = max(2, min(64, param))
            rounds = max(1, min(8, int(config.get("rounds", 2))))
            beam = [()]
            for round_index in range(rounds):
                branches = 2 * width if round_index == 0 else 2
                beam = rank(generate(path) for path in beam for _ in range(branches))[:width]
            chosen = beam[0]
        elif self.strategy == "lookahead":
            depth = max(1, min(4, param))
            seeds = max(2, min(32, int(config.get("seeds", 4))))
            max_steps = max(1, min(8, int(config.get("max_steps", 2))))
            chosen = ()
            for _ in range(max_steps):
                rollouts = []
                for _ in range(seeds):
                    path = chosen
                    for _ in range(depth):
                        path = generate(path)
                    rollouts.append(path)
                best_path = rank(rollouts, rollout_start=len(chosen))[0]
                chosen = best_path[:len(chosen) + 1]
        else:
            if self.strategy != "best_of_n":
                raise ValueError(f"Unsupported paper search strategy: {self.strategy}")
            chosen = rank(generate() for _ in range(max(1, min(1024, param))))[0]

        return chosen[-1].text, {
            "strategy": self.strategy,
            "compute_config": dict(config),
            "num_reasoner_calls": generated,
            "num_scored": sum(map(len, ranked_rounds)),
            "chosen_reasoning": "\n\n".join(step for action in chosen for step in action.steps),
            "ranked_rounds": ranked_rounds,
        }
