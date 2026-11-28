"""Archive and check the E2 transfer runs without loading any model."""
from collections import Counter
import csv
import gzip
import hashlib
from itertools import islice
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from audit_transfer import FROZEN_SHA, audit, load

PAIRS = (("llama-dprm", "llama", "dprm"),
         ("qwen-versa", "qwen", "versa"),
         ("llama-versa", "llama", "versa"))
ARMS = (("k10-mean", "mean", None), ("k10-min", "min", None),
        ("stop5-mean", "mean", 5), ("stop5-min", "min", 5))
PAIR_LABELS = {"qwen-dprm": "Qwen + dPRM", "llama-dprm": "Llama + dPRM",
               "qwen-versa": "Qwen + VersaPRM", "llama-versa": "Llama + VersaPRM"}
ARM_LABELS = {"k10-mean": "固定 10 条＋平均步分", "k10-min": "固定 10 条＋最低步分",
              "stop5-mean": "五条一致早停＋平均步分", "stop5-min": "五条一致早停＋最低步分"}


def paths(framework):
    root = framework / "results"
    return root / "mmlu-pro-e2-transfer", root / "mmlu-pro-1400", root / "mmlu-pro-e2-online"


def copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def csv_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def expected_questions(data):
    assert hashlib.sha256(data).hexdigest() == FROZEN_SHA
    expected = {q["id"]: (q["category"], q["answer"]) for q in json.loads(data)["questions"]}
    groups = Counter(category for category, _ in expected.values())
    assert len(expected) == 1400 and len(groups) == 14 and set(groups.values()) == {100}
    return expected


def predictions(source, config, dest):
    order = iter(config["questions"])
    for ordinal in range(1, 5):
        task_ids = list(islice(order, 350))
        with (dest / f"predictions-{ordinal:02d}.jsonl.gz").open("wb") as file:
            with gzip.GzipFile(fileobj=file, filename="", mode="wb", mtime=0, compresslevel=6) as zipped:
                for qid in task_ids:
                    key = hashlib.sha256(qid.encode()).hexdigest()[:20]
                    row = load(source / "questions" / f"{key}.json")
                    zipped.write((json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode())
                    yield row
    assert next(order, None) is None


def archive_pair(framework, runtime, pair, expected):
    result, history, _ = paths(framework)
    target = result / "runs" / pair
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_dir():
        return
    prior = load(history / "runs" / pair / "E2/config.json")
    with TemporaryDirectory(dir=runtime, prefix=f"{pair}-archive-") as temporary:
        out = Path(temporary) / pair
        rows, scores = [], {}
        for arm, _, _ in ARMS:
            source = runtime / "full" / pair / arm
            dest = out / arm
            for name in ("config.json", "summary.json", "runtime-segments.json"):
                copy(source / name, dest / name)
            assert len(list((source / "questions").glob("*.json"))) == 1400
            config = load(dest / "config.json")
            row, scores[arm] = audit(dest, arm, predictions(source, config, dest), expected, prior)
            rows.append(row)
        baseline = scores["k10-mean"]
        paired = [{
            "baseline": "k10-mean", "variant": arm,
            "incorrect_to_correct": sum(not baseline[q] and scores[arm][q] for q in baseline),
            "correct_to_incorrect": sum(baseline[q] and not scores[arm][q] for q in baseline),
        } for arm, _, _ in ARMS[1:]]
        write_json(out / "paired.json", paired)
        csv_rows(out / "summary.csv", rows)
        for file in (runtime / "records" / pair).iterdir():
            if file.is_file():
                copy(file, out / "provenance" / file.name)
        out.replace(target)


def comparison(framework):
    result, _, original = paths(framework)
    data = [("qwen-dprm", original / "summary.csv")]
    data.extend((pair, result / "runs" / pair / "summary.csv") for pair, _, _ in PAIRS)
    return [{"combination": pair, **row} for pair, file in data for row in read_csv(file)]


def write_comparison(framework, runtime):
    result, _, _ = paths(framework)
    data = comparison(framework)
    csv_rows(result / "comparison.csv", data)
    copy(runtime / "queue.log", result / "queue.log")
    lines = [
        "# MMLU-Pro E2 跨模型与评分器在线对照", "",
        "追加的三个组合按 Llama + dPRM、Qwen + VersaPRM、Llama + VersaPRM 顺序在单张 AutoDL RTX 4090 上运行。",
        "四种组合使用同一份 MMLU-Pro 1,400 题，每领域 100 题，种子 0，输出上限 1,024 token，客户端 96 并发。",
        "固定预算为 10 条。早停组在前 5 条均给出同一个有效答案时直接返回，否则补到 10 条并评分。",
        "每组重新启动 vLLM，先以固定 10 条＋平均步分预热相同的 8 题。全量耗时按各组运行记录统计。",
        "总 token 包含生成输入、生成输出和 PRM 输入。Qwen + dPRM 的四组沿用 PR #46 的在线实测。", "",
        "| 组合 | 条件 | 正确率（答对题数） | 总 token | PRM 调用 | 实测耗时（秒） |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in data:
        lines.append(
            f"| {PAIR_LABELS[r['combination']]} | {ARM_LABELS[r['arm']]} | "
            f"{float(r['accuracy']):.2%}（{r['correct']}） | {int(r['total_tokens']):,} | "
            f"{int(r['prm_calls']):,} | {float(r['elapsed_seconds']):.2f} |"
        )
    lines.extend([
        "", "## 归档与复核", "",
        "- [全量指标](comparison.csv) 含 token 分项、早停数和逐题 P95。",
        "- [新增逐题记录](runs/) 共 16,800 条。每组按题单顺序存成四份压缩 JSONL，附配置、计时、预热、选优和配对统计。",
        "- [Qwen + dPRM 原始记录](../mmlu-pro-e2-online/) 及 [冻结题单](../mmlu-pro-1400/mmlu-pro-1400.json.gz) 沿用前序 PR。",
        "- [运行脚本和验收说明](../../e2-transfer/README.md) 可重算全量表格。",
        "",
    ])
    (result / "README.md").write_text("\n".join(lines), encoding="utf-8")
    listing = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(result)}"
               for p in sorted(result.rglob("*")) if p.is_file() and p.name != "SHA256SUMS"]
    (result / "SHA256SUMS").write_text("\n".join(listing) + "\n", encoding="utf-8")


def verify(framework):
    result, history, _ = paths(framework)
    expected = expected_questions(gzip.decompress((history / "mmlu-pro-1400.json.gz").read_bytes()))
    entries = (result / "SHA256SUMS").read_text().splitlines()
    for line in entries:
        checksum, path = line.split("  ", 1)
        assert hashlib.sha256((result / path).read_bytes()).hexdigest() == checksum, path
    for pair, _, _ in PAIRS:
        base = result / "runs" / pair
        prior = load(history / "runs" / pair / "E2/config.json")
        table = {row["arm"]: row for row in read_csv(base / "summary.csv")}
        for arm, _, _ in ARMS:
            folder = base / arm
            parts = sorted(folder.glob("predictions-*.jsonl.gz"))
            assert len(parts) == 4
            def records():
                for part in parts:
                    with gzip.open(part, "rt", encoding="utf-8") as handle:
                        yield from map(json.loads, handle)
            recomputed, _ = audit(folder, arm, records(), expected, prior)
            assert table[arm] == {k: "" if v is None else str(v) for k, v in recomputed.items()}
            print(pair, arm, recomputed["correct"], recomputed["total_tokens"],
                  recomputed["prm_calls"], recomputed["elapsed_seconds"], flush=True)
    assert read_csv(result / "comparison.csv") == comparison(framework)
    print(f"Verified {len(entries)} checksums, 12 runs, 16,800 question records, and 16 comparison rows.")


if __name__ == "__main__":
    verify(Path(__file__).resolve().parent.parent)
