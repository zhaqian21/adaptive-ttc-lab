import ast
import hashlib
import json
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier
from types import SimpleNamespace
import unittest

from shared_question_queue import SharedQuestionQueue


def load_shared_completions():
    path = (Path(__file__).resolve().parents[1] / "aime24-amo-bench-published-code-2026-11-24"
            / "scripts" / "run_published_extensions.py")
    module = ast.parse(path.read_bytes())
    node = next(node for node in module.body if isinstance(node, ast.FunctionDef)
                and node.name == "shared_completions")
    namespace = {"wait": wait, "FIRST_COMPLETED": FIRST_COMPLETED}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["shared_completions"]


class SharedQuestionQueueTest(unittest.TestCase):
    def test_competing_workers_claim_500_questions_once_and_from_the_right_end(self):
        shared_completions = load_shared_completions()
        questions = [{"problem_id": n} for n in range(500)]
        barrier = Barrier(2)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "queue.sqlite3"
            queue = SharedQuestionQueue(path)
            queue.initialize(range(500))

            def run_lane(lane):
                reader = SharedQuestionQueue(path)
                first = True

                def claim():
                    nonlocal first
                    item = reader.claim(lane)
                    if first:
                        first = False
                        barrier.wait(timeout=10)
                    return item

                with ThreadPoolExecutor(max_workers=8) as pool:
                    return [future.result() for future, row in shared_completions(
                        pool, lambda row: row["problem_id"], questions, claim, workers=8)]

            with ThreadPoolExecutor(max_workers=2) as gpu_workers:
                head, tail = gpu_workers.map(run_lane, (0, 1))
            claims = queue.audit_claims()
            self.assertEqual(sorted(head + tail), list(range(500)))
            self.assertEqual(sorted(head), sorted(r["problem_id"] for r in claims if r["lane"] == 0))
            self.assertEqual(sorted(tail), sorted(r["problem_id"] for r in claims if r["lane"] == 1))
            self.assertEqual(next(r["problem_id"] for r in claims if r["lane"] == 0), 0)
            self.assertEqual(next(r["problem_id"] for r in claims if r["lane"] == 1), 499)
            self.assertIsNone(queue.claim(0))
            self.assertIsNone(queue.claim(1))

    def test_shared_audit_requires_500_unique_rows_on_the_claimed_gpus(self):
        path = Path(__file__).with_name("llama_table1.py")
        node = next(node for node in ast.parse(path.read_bytes()).body
                    if isinstance(node, ast.FunctionDef) and node.name == "audit")
        model = ("model", "Llama", "pinned")

        def read(path):
            return json.loads(path.read_bytes())

        def write(path, data):
            path.write_text(json.dumps(data), encoding="utf-8")

        released = SimpleNamespace(read=read, write=write, CASES={"E4": (True, 10)},
                                   CLASSIFICATION="published_code", runner=SimpleNamespace(UPSTREAM_COMMIT="pinned"))
        scope = {"MODEL": model, "released": released, "json": json, "hashlib": hashlib,
                 "check_problem": lambda *args: None}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
        audit = scope["audit"]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / f"gpu-{lane}" for lane in (0, 1)]
            queue = SharedQuestionQueue(root / "queue.sqlite3")
            queue.initialize(range(500))
            for _ in range(250):
                queue.claim(0)
                queue.claim(1)
            claims = queue.audit_claims()
            questions = []
            for row in sorted(claims, key=lambda row: row["problem_id"]):
                index, lane = row["problem_id"], row["lane"]
                problem_dir = paths[lane] / "problems" / f"{index:03d}"
                problem_dir.mkdir(parents=True)
                questions.append({"dataset_unique_id": str(index), "problem": f"Question {index}",
                                  "gold_answer_extracted": "42"})
                write(problem_dir / "result.json", {
                    "dataset_unique_id": str(index), "problem": f"Question {index}", "gold_answer": "42",
                    "final_is_correct": True, "final_predicted_answer": "42", "iterations": [{"is_correct": True}],
                })
            for lane, folder in enumerate(paths):
                predictions = [{"problem_id": row["problem_id"], "dataset_unique_id": str(row["problem_id"]),
                                "is_correct": True, "predicted_answer": "42"} for row in claims if row["lane"] == lane]
                write(folder / "predictions.json", sorted(predictions, key=lambda row: row["problem_id"]))
                write(folder / "summary.json", {"samples": 250, "requested": 250, "correct": 250, "failed": {}})
                write(folder / "config.json", {
                    "assignment": "shared_head_tail", "strict_errors": True,
                    "sample_ids": list(range(500)), "dataset": "math", "seed": 0,
                    "upstream_commit": "pinned", "eval": {"num_iterations": 10},
                    "datasets_and_models": [model], "runtime": {}, "client": {},
                })
            args = SimpleNamespace(output=root)
            self.assertEqual(audit(args, "math", "E4", questions, paths, "0.33.0", 327, claims)["samples"], 500)
            export = read(root / "export" / "math" / "E4" / "audit.json")
            self.assertEqual(export["dispatch"]["claim_order"], claims)
            first = read(paths[0] / "predictions.json")
            first[1] = first[0]
            write(paths[0] / "predictions.json", first)
            with self.assertRaises(AssertionError):
                audit(args, "math", "E4", questions, paths, "0.33.0", 327, claims)

    def test_reopening_keeps_previous_claims_and_incomplete_runs_fail_audit(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "queue.sqlite3"
            queue = SharedQuestionQueue(path)
            queue.initialize(range(500))
            self.assertEqual(queue.claim(0), 0)
            reopened = SharedQuestionQueue(path)
            self.assertEqual(reopened.claim(0), 1)
            self.assertEqual(reopened.claim(1), 499)
            with self.assertRaisesRegex(ValueError, "exactly once"):
                reopened.audit_claims()


if __name__ == "__main__":
    unittest.main()
