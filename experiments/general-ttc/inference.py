"""Domain-neutral strategy selection, branching, and trajectory scoring."""
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, replace
from statistics import fmean
from string import ascii_uppercase


STRATEGIES = ("direct", "best_of_n", "beam", "lookahead")
TOOLS = {
    "decompose": "Break the question into short deductions and combine them.",
    "compare": "Compare plausible options against the question's constraints.",
    "verify": "Check key assumptions and look for contradictions.",
}
FINAL = re.compile(r"(?:FINAL(?: ANSWER)?|ANSWER)\s*:\s*\(?([A-Z])\)?[.]?", re.I)


@dataclass(frozen=True)
class Task:
    id: str
    category: str
    question: str
    options: tuple
    evidence: tuple = ()

    def prompt(self):
        choices = "\n".join(f"({label}) {option}" for label, option in zip(ascii_uppercase, self.options))
        evidence = "\n\nEvidence:\n" + "\n".join(self.evidence) if self.evidence else ""
        return f"Question: {self.question}\nOptions:\n{choices}{evidence}"


@dataclass(frozen=True)
class Trajectory:
    steps: tuple = ()
    answer: str = None
    step_scores: tuple = ()

    @property
    def score(self):
        return self.aggregate()

    def aggregate(self, rule="mean"):
        if not self.step_scores:
            return None
        return min(self.step_scores) if rule == "min" else fmean(self.step_scores)


def parse(text, option_count):
    steps = []
    answer = None
    for line in text.splitlines():
        line = line.strip().strip("*").strip()
        if not line:
            continue
        match = FINAL.fullmatch(line)
        if match and match[1].upper() in ascii_uppercase[:option_count]:
            answer = match[1].upper()
        else:
            steps.append(re.sub(r"^(?:step\s*(?:\d+)?\s*:\s*)", "", line, flags=re.I) or line)
    return tuple(steps), answer


@dataclass(frozen=True)
class Route:
    strategy: str
    tool: str
    plan: str
    parameter: int


class Iteration:
    def __init__(self, task, generate, score_steps, seed, max_tokens, search_steps, fixed_strategy):
        self.task = task
        self.question = task.prompt()
        self.generate = generate
        self.score_steps = score_steps
        self.seed = seed
        self.max_tokens = max_tokens
        self.search_steps = search_steps
        self.fixed_strategy = fixed_strategy
        self.events = []
        self.calls = 0

    def ask(self, prompt, purpose, limit, temperature=0.7):
        seed = (self.seed + self.calls) % (2**31)
        self.calls += 1
        text, usage = self.generate(self.question + "\n\n" + prompt, seed, limit, temperature)
        self.events.append({"kind": "generation", "purpose": purpose, "seed": seed, "usage": usage, "text": text})
        return text

    def route(self):
        text = self.ask(
            'Choose how to solve this question. Use "direct" for straightforward reasoning, "best_of_n" '
            'for independent alternatives, "beam" for competing deductions, or "lookahead" to check '
            'where a deduction leads. Select "decompose", "compare", or "verify" as the reasoning tool. '
            'Reply with only a JSON object: {"strategy":"direct","tool":"decompose",'
            '"plan":"brief solving plan","parameter":2}. Parameter must be 1, 2, or 3.',
            "controller", min(200, self.max_tokens), 0.0,
        )
        start, end = text.find("{"), text.rfind("}") + 1
        try:
            config = json.loads(text[start:end])
            strategy = self.fixed_strategy or config["strategy"]
            tool, plan, parameter = config["tool"], config["plan"], config["parameter"]
            if (strategy in STRATEGIES and tool in TOOLS and isinstance(plan, str)
                    and type(parameter) is int and 1 <= parameter <= 3):
                return Route(strategy, tool, plan, parameter)
        except (KeyError, TypeError, json.JSONDecodeError):
            pass
        self.events.append({"kind": "controller_fallback", "strategy": self.fixed_strategy or "direct"})
        return Route(self.fixed_strategy or "direct", "decompose", "", 2)

    def instruction(self, route, prefix):
        instructions = []
        if route:
            instructions.extend([TOOLS[route.tool], f"Solving plan: {route.plan}"])
        if prefix.steps:
            instructions.append("Continue from these existing steps:\n" + "\n".join(prefix.steps))
        return "\n".join(instructions) + "\n"

    def finish(self, route=None, prefix=Trajectory()):
        text = self.ask(self.instruction(route, prefix) +
                        "Write one reasoning step per line. End with a separate line FINAL: X, "
                        "where X is the letter of one offered option.", "complete", self.max_tokens)
        steps, answer = parse(text, len(self.task.options))
        return Trajectory(prefix.steps + steps, answer)

    def advance(self, route, prefix):
        text = self.ask(self.instruction(route, prefix) +
                        "Write exactly one short new deduction as STEP: ... or, if ready to conclude, "
                        "write only FINAL: X using a letter from the options.",
                        "expand", min(192, self.max_tokens))
        steps, answer = parse(text, len(self.task.options))
        if not steps and answer is None:
            return prefix
        return Trajectory(prefix.steps + steps[:1], answer if not steps else None)

    def score(self, trajectory, purpose):
        if trajectory.step_scores:
            return trajectory
        steps = trajectory.steps
        if trajectory.answer:
            index = ascii_uppercase.index(trajectory.answer)
            steps += (f"The answer is ({trajectory.answer}) {self.task.options[index]}.",)
        if not steps:
            return trajectory
        rewards, tokens = self.score_steps(self.question, steps)
        if len(rewards) != len(steps) or not all(math.isfinite(s) and 0 <= s <= 1 for s in rewards):
            raise ValueError(f"Invalid process scores for {self.task.id}")
        self.events.append({
            "kind": "prm", "purpose": purpose, "steps": steps, "scores": rewards, "input_tokens": tokens,
        })
        return replace(trajectory, step_scores=tuple(rewards))

    @staticmethod
    def rank(trajectories, prefer_answers=False, aggregation="mean"):
        def rank_key(trajectory):
            score = trajectory.aggregate(aggregation)
            return bool(trajectory.answer) if prefer_answers else False, score if score is not None else -1.0

        return sorted(trajectories, reverse=True, key=rank_key)

    def search(self, route):
        if route.strategy == "direct":
            return self.finish(route)
        if route.strategy == "best_of_n":
            candidates = [self.score(self.finish(route), "search") for _ in range(max(2, route.parameter))]
            return self.rank(candidates, prefer_answers=True)[0]

        if route.strategy == "beam":
            beam = [Trajectory()]
            for _ in range(self.search_steps):
                pool = []
                for prefix in beam:
                    if prefix.answer:
                        pool.append(prefix)
                    else:
                        pool.extend(self.advance(route, prefix) for _ in range(2))
                ranked = self.rank(self.score(t, "search") for t in dict.fromkeys(pool))
                beam = ranked[:max(2, route.parameter)]
                if all(t.answer for t in beam):
                    break
            selected = beam[0]
        else:
            selected = Trajectory()
            for _ in range(self.search_steps):
                if selected.answer:
                    break
                options = []
                for _ in range(2):
                    first = rollout = self.advance(route, selected)
                    for _ in range(route.parameter - 1):
                        if rollout.answer:
                            break
                        rollout = self.advance(route, rollout)
                    scored = self.score(rollout, "search")
                    new_scores = scored.step_scores[len(selected.steps):]
                    options.append((fmean(new_scores) if new_scores else -1.0, first))
                _, selected = max(options, key=lambda item: item[0])
        return selected if selected.answer else self.finish(route, selected)


def solve(task, method, k, seed, max_tokens, search_steps, generator, prm, fixed_strategy=None,
          *, prm_aggregation="mean", agreement_stop=None):
    if prm_aggregation not in {"mean", "min"}:
        raise ValueError("Unknown PRM aggregation rule")
    if method != "E2" and (prm_aggregation != "mean" or agreement_stop is not None):
        raise ValueError("Custom PRM aggregation and agreement stopping require E2")
    if agreement_stop is not None and not 2 <= agreement_stop < k:
        raise ValueError("Agreement stopping requires at least two candidates and fewer than k")

    dynamic = method in {"E3", "E4"}
    rerank = method in {"E2", "E4"}
    candidates = []
    routes = []
    events = []
    deferred = []
    early_stopped = False
    for index in range(k if rerank else 1):
        raw_seed = f"{seed}:{task.id}:{'dynamic' if dynamic else 'direct'}:{index}".encode()
        turn_seed = int.from_bytes(hashlib.sha256(raw_seed).digest()[:4], "big") % (2**31)
        iteration = Iteration(task, generator, prm, turn_seed, max_tokens, search_steps, fixed_strategy)
        route = iteration.route() if dynamic else None
        result = iteration.search(route) if route else iteration.finish()
        if rerank and agreement_stop is None:
            result = iteration.score(result, "final")
        elif agreement_stop is not None:
            deferred.append(iteration)
        candidates.append(result)
        routes.append(asdict(route) if route else None)
        events.extend({"iteration": index, **event} for event in iteration.events)
        if len(candidates) == agreement_stop:
            answer = candidates[0].answer
            if answer and all(candidate.answer == answer for candidate in candidates):
                early_stopped = True
                break

    if not early_stopped:
        for index, iteration in enumerate(deferred):
            before = len(iteration.events)
            candidates[index] = iteration.score(candidates[index], "final")
            events.extend({"iteration": index, **event} for event in iteration.events[before:])

    selected = candidates[0]
    if rerank and not early_stopped:
        selected = Iteration.rank(candidates, prefer_answers=True, aggregation=prm_aggregation)[0]
    result = {
        "prediction": selected.answer, "selected": candidates.index(selected),
        "candidates": [
            {**asdict(candidate), "score": candidate.aggregate(prm_aggregation), "route": routes[index]}
            for index, candidate in enumerate(candidates)
        ],
        "events": events,
    }
    if method == "E2":
        result["early_stopped"] = early_stopped
    return result
