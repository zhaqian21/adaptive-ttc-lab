"""Prepare the frozen MMLU-Pro subset and the two VersaPRM snapshots."""
import argparse
import hashlib
import json
from pathlib import Path
from string import ascii_uppercase
from urllib.request import urlopen


ASSETS = json.loads(Path(__file__).with_name("assets.json").read_text(encoding="utf-8"))
CATEGORIES = ("math", "physics", "computer science", "law", "history")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def prepare_questions(output, per_category, seed, source=None, categories=CATEGORIES):
    if not categories or len(set(categories)) != len(categories):
        raise ValueError('Choose distinct categories from the source dataset')
    spec = ASSETS["mmlu_pro"]
    url = f"https://raw.githubusercontent.com/UW-Madison-Lee-Lab/VersaPRM/{spec['revision']}/{spec['path']}"
    if source:
        raw = source.read_bytes()
    else:
        with urlopen(url, timeout=60) as response:
            raw = response.read()
    rows = json.loads(raw)
    selected = []
    excluded = []
    for category in categories:
        candidates = []
        for row in rows:
            if row["category"] != category:
                continue
            options, answer, index = row["options"], row["answer"], row["answer_index"]
            if not 2 <= len(options) <= 26 or index not in range(len(options)) or answer != ascii_uppercase[index]:
                excluded.append(str(row["id"]))
                continue
            candidates.append(row)

        def order(row):
            return hashlib.sha256(f"{seed}:{row['id']}".encode()).digest()

        if len(candidates) < per_category:
            raise ValueError(f"{category}: only {len(candidates)} valid questions available")
        for row in sorted(candidates, key=order)[:per_category]:
            selected.append({
                "id": str(row["id"]), "category": category, "question": row["question"],
                "options": row["options"], "answer": row["answer"],
            })
    if len({row["id"] for row in selected}) != len(selected):
        raise ValueError("Duplicate question IDs")
    manifest = {
        "source_url": url if source is None else str(source),
        "source_sha256": hashlib.sha256(raw).hexdigest(), "seed": seed,
        "per_category": per_category, "questions": selected,
    }
    if excluded:
        manifest["invalid_source_ids"] = sorted(excluded)
    write_json(output, manifest)


def prepare_models(directory):
    from huggingface_hub import snapshot_download

    for name in ("prm_base", "prm_adapter"):
        spec = ASSETS[name]
        snapshot_download(
            spec["id"], revision=spec["revision"], local_dir=directory / name,
            allow_patterns=["*.safetensors", "*.json"], max_workers=4,
        )
    write_json(directory / "assets.json", {name: ASSETS[name] for name in ("prm_base", "prm_adapter")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="operation", required=True)
    dataset = subcommands.add_parser("mmlu")
    dataset.add_argument("--output", type=Path, required=True)
    dataset.add_argument("--source-json", type=Path)
    dataset.add_argument("--per-category", type=int, default=100)
    dataset.add_argument("--seed", type=int, default=0)
    dataset.add_argument("--category", action="append", help="Select a category; repeat for multiple categories")
    models = subcommands.add_parser("models")
    models.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    if args.operation == "mmlu":
        if args.per_category < 1:
            parser.error("--per-category must be positive")
        prepare_questions(args.output, args.per_category, args.seed, args.source_json,
                          args.category or CATEGORIES)
    else:
        prepare_models(args.directory)


if __name__ == "__main__":
    main()