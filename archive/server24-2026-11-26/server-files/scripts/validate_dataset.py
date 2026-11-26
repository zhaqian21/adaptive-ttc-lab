"""Verify the pinned test split and exercise the released MATH grader."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

from datasets import load_dataset

from fetch_assets import ASSETS, ROOT

sys.path.insert(0, str(ROOT / "upstream"))
from src import math_core
from src.Evaluator import get_evaluator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare-hub", action="store_true",
                        help="Exercise the official Hub loading route and compare it with the pinned local file.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _, repo, revision = ASSETS[0]
    path = ROOT / "assets" / "dataset" / f"{repo.replace('/', '--')}--{revision}" / "test.jsonl"
    raw = path.read_bytes()
    source = [json.loads(line) for line in raw.splitlines()]
    assert len(source) == len({row["unique_id"] for row in source}) == 500

    def pinned_load(name):
        assert name == repo
        if args.compare_hub:
            splits = load_dataset(name, revision=revision)
            assert list(splits) == ["test"]
            assert list(splits["test"]) == source
            return splits
        return load_dataset("json", data_files={"test": str(path)})

    with patch.object(math_core, "load_dataset", pinned_load):
        normalized = math_core.MATHCore.load_dataset(split="test")
    assert len(normalized) == 500
    for index, (loaded, original) in enumerate(zip(normalized, source, strict=True)):
        assert loaded["problem_id"] == index
        assert loaded["problem"] == original["problem"]
        assert loaded["solution"] == original["solution"]
        assert loaded["gold_answer_raw"] == original["answer"]
        assert loaded["problem_type"] == original["subject"]
        assert loaded["level"] == original["level"]
        assert loaded["gold_answer_extracted"]

    grader = get_evaluator("MATH")
    # Check the pinned grader's behavior and retain known disagreements with mathematical equivalence.
    cases = [
        ("last_boxed", r"We first have \boxed{1}, then \boxed{2}.", "2", "2", True, True),
        ("equivalent_decimal", r"\boxed{0.50}", "0.5", "0.50", True, True),
        ("equivalent_latex_spelling", r"\boxed{\frac{1}{2}}", r"\dfrac{1}{2}", r"\frac{1}{2}", True, True),
        ("incorrect", r"\boxed{3}", "2", "3", False, False),
        ("missing_box", "The answer is 2.", "2", None, None, None),
        ("fraction_vs_decimal", r"\boxed{\frac{1}{2}}", "0.5", r"\frac{1}{2}", False, True),
        ("equivalent_expression", r"\boxed{x+x}", "2x", "x+x", False, True),
        ("scalar_vs_polar_coordinate", r"\boxed{2}", r"\left( 3, \frac{\pi}{2} \right)", "2", True, False),
    ]
    checked = []
    for name, completion, gold, expected_extraction, expected_upstream, mathematical_equivalence in cases:
        extracted = grader._extract_answer_from_model_completion(completion)
        grade = grader._check_answers_equiv(extracted, gold)
        assert extracted == expected_extraction and grade == expected_upstream, (name, extracted, grade)
        checked.append({"case": name, "extracted": extracted, "gold": gold,
                        "upstream_grade": grade, "mathematical_equivalence": mathematical_equivalence})

    result = {
        "repo": repo, "revision": revision, "split": "test", "rows": len(source),
        "unique_ids": len(source), "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "compared_with_pinned_hub": args.compare_hub,
        "subjects": dict(sorted(Counter(row["subject"] for row in source).items())),
        "levels": dict(sorted(Counter(row["level"] for row in source).items())),
        "grader_tests": checked,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        print(json.dumps(result, indent=2), file=output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
