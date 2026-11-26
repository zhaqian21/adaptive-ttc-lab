"""Audit a completed MATH-500 smoke run and rebuild metrics from question records."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path


CASES = {"A": (10, 1), "B": (10, 10), "C": (20, 3)}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_problem(folder, problem, index, iterations):
    assert problem["problem_id"] == index, folder
    rows = problem["iterations"]
    observation = problem["observation"]
    assert [row["iteration"] for row in rows] == list(range(1, iterations + 1)), folder
    assert len(observation["traces"]) == iterations, folder
    expected = dict(problem)
    del expected["observation"]
    assert read(folder / "results_final.json")["results"] == [expected], folder

    pathway, = read(folder / "pathway.json")
    assert pathway["problem_id"] == index and len(pathway["iterations"]) == iterations, folder
    with (folder / "results_live.csv").open(newline="", encoding="utf-8") as source:
        csv_rows = list(csv.DictReader(source))
    assert [int(row["problem_id"]) for row in csv_rows] == [index] * iterations, folder
    assert [int(row["iteration"]) for row in csv_rows] == list(range(1, iterations + 1)), folder
    assert [row["selected_by_prm"] for row in csv_rows] == [str(row["selected_by_prm"]) for row in rows], folder

    selected = problem["selected_iteration_idx"]
    assert [row["iteration"] for row in rows if row["selected_by_prm"]] == [selected], folder
    assert all(rows[selected - 1][key] == problem[f"final_{key}"] for key in ("predicted_answer", "is_correct")), folder
    scores = observation["prm_calls"]
    if iterations == 1:
        assert selected == 1 and not scores and not rows[0]["prm_scores"], folder
    else:
        assert len(scores) == iterations and pathway["selected_iteration_idx"] == selected, folder
        means = []
        for row, score in zip(rows, scores):
            rewards = score["step_rewards"]
            assert len(rewards) == score["num_steps"] > 0, folder
            assert all(math.isfinite(r) and 0 <= r <= 1 for r in rewards), folder
            assert math.isclose(score["mean_reward"], sum(rewards) / len(rewards), abs_tol=1e-12), folder
            native = dict(score, iteration=row["iteration"])
            del native["seconds"]
            assert native == row["prm_scores"] == pathway["prm_scores"][row["iteration"] - 1], folder
            means.append(score["mean_reward"])
        winner = means.index(max(means)) + 1 if any(means) else len(means)
        assert selected == winner, folder


def audit(directory, dataset):
    original = read(directory / "summary.json")
    case = original["case"]
    count, iterations = CASES[case]
    folders = sorted(path for path in (directory / "problems").iterdir() if path.is_dir())
    assert [path.name for path in folders] == [f"{index:03d}" for index in range(count)]
    problems = [read(folder / "result.json") for folder in folders]
    assert len({row["dataset_unique_id"] for row in problems}) == count

    strategies = defaultdict(lambda: {"calls": 0, "seconds": 0.0, "llm_calls": 0, "output_tokens": 0})
    for index, (folder, problem) in enumerate(zip(folders, problems)):
        check_problem(folder, problem, index, iterations)
        if dataset:
            assert problem["dataset_unique_id"] == dataset[index]["unique_id"], folder
            assert problem["problem"] == dataset[index]["problem"], folder
            assert problem["gold_answer_raw"] == dataset[index]["answer"], folder
        for call in problem["observation"].get("strategy_calls", []):
            aggregate = strategies[call["strategy"]]
            aggregate["calls"] += 1
            for name in ("seconds", "llm_calls", "output_tokens"):
                aggregate[name] += call[name]

    observations = [problem["observation"] for problem in problems]
    generated = [call for row in observations for call in row["llm_calls"]]
    scored = [call for row in observations for call in row["prm_calls"]]
    local_scores = [call for row in observations for call in row["intra_iteration_scoring"]]
    correct = sum(bool(row["final_is_correct"]) for row in problems)
    seconds = sum(row["seconds"] for row in observations)
    expected = {
        "samples": count, "correct": correct, "accuracy": correct / count,
        "problem_wall_seconds": seconds, "mean_seconds_per_question": seconds / count,
        "gpu_peak_allocated_bytes": max(row["gpu_peak_allocated_bytes"] for row in observations),
        "gpu_peak_reserved_bytes": max(row["gpu_peak_reserved_bytes"] for row in observations),
        "llm_calls": len(generated), "llm_seconds": sum(call["seconds"] for call in generated),
        "input_tokens": sum(call["input_tokens"] for call in generated),
        "output_tokens": sum(call["output_tokens"] for call in generated),
        "prm_calls": len(scored), "prm_seconds": sum(call["seconds"] for call in scored),
        "intra_iteration_scoring_calls": len(local_scores),
        "intra_iteration_scoring_seconds": sum(call["seconds"] for call in local_scores),
    }
    for key, value in expected.items():
        assert math.isclose(original[key], value) if isinstance(value, float) else original[key] == value, key
    return {
        "case": case, "summary_verified": True, "native_files_verified": True,
        "pinned_dataset_checked": bool(dataset), **expected, "strategy_measurements": strategies,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--dataset", type=Path, help="Pinned MATH-500 test.jsonl")
    args = parser.parse_args()
    dataset = None
    if args.dataset is not None:
        with args.dataset.open(encoding="utf-8") as source:
            dataset = [json.loads(line) for line in source]
        assert len(dataset) == 500
    print(json.dumps(audit(args.results, dataset), indent=2))


if __name__ == "__main__":
    main()
