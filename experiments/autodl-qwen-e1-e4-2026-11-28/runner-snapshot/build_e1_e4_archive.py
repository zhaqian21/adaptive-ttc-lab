"""Validate and assemble the AutoDL E1–E4 experiment archive."""
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys

work = Path(__file__).resolve().parent
server = work.parent
repo = server.parent / "github"
h20_ext = repo / "experiments/aime24-amo-bench-published-code-2026-11-24"
e1_math = work / "results-e1-formal-autodl-20261128"
e1_ext = work / "results-e1-other-datasets-resume-20261128"
e3_root = work / "results-e3-all-autodl-20261128"
output = Path(sys.argv[1])
paused = sys.argv[2:] == ["--paused-after-amo"]
if sys.argv[2:] and not paused:
    raise SystemExit("Expected --paused-after-amo")
pause_marker = server / "logs" / "e1-e4-paused-after-amo"
if paused:
    assert pause_marker.is_file(), "AMO pause has not been verified"
pending_runs = {"E4/amc23", "E4/aime26", "E4/gsm8k"} if paused else set()
bundle = output / "bundle"
bundle.mkdir(exist_ok=True)
counts = {"math500":500, "aime24":30, "amo":50, "amc23":40, "aime26":30, "gsm8k":1319}
names = {"math500":"MATH-500","aime24":"AIME24","amo":"AMO-Bench","amc23":"AMC23","aime26":"AIME 2026","gsm8k":"GSM8K"}
methods = ("E1", "E2", "E3", "E4")

def load(path):
    return json.loads(path.read_text())

def save(path, item):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(item, ensure_ascii=False, indent=2) + "\n",encoding="utf-8")
    pending.replace(path)

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def read_rows(path, sha):
    data = path.read_bytes()
    assert digest(data) == sha, path
    return [json.loads(line) for line in data.splitlines()]

def combine(rows, elapsed, retries, prm):
    total=len(rows)
    correct=sum(row["is_correct"] is True for row in rows)
    tokens=sum(row["output_tokens"] for row in rows)
    return {
        "questions":total, "correct":correct, "accuracy":correct/total,
        "elapsed_seconds":elapsed, "output_tokens":tokens,
        "generation_calls":sum(row["llm_calls"] for row in rows),
        "prm_calls":prm, "question_retries":retries,
    }

source_specs=load(server/"datasets/manifest.json")
old_math=load(e1_math/"comparison.json")
old_ext=load(e1_ext/"summary.json")
e3_results=load(e3_root/"summary.json")
paired_e3=load(e3_root/"e3-vs-e1-4090.json")
h20_formal=load(work/"config/formal-runs.json")
h20_extensions=load(h20_ext/"config/formal-runs.json")
assert set(e3_results)==set(paired_e3)==set(counts)

def convert(old, seconds_key="elapsed_seconds"):
    return {
        "questions":old.get("questions",500),"correct":old["correct"],"accuracy":old["accuracy"],
        "elapsed_seconds":old[seconds_key],"output_tokens":old.get("output_tokens",old.get("total_output_tokens")),
        "generation_calls":old["generation_calls"],"prm_calls":0,"question_retries":old["question_retries"],
    }

def common_config(config,past):
    fields=("case","backend","concurrency","package_version","upstream_commit","dataset_sha256","source_sha256",
            "decode","seed","strict_errors","agent","eval","datasets_and_models")
    for key in fields:
        assert config[key]==past[key], (config["case"],config.get("dataset","math500"),key)
    client=config["client"].copy()
    expected=past["client"].copy()
    assert Path(client.pop("model_id")).name==Path(expected.pop("model_id")).name and client==expected

def prm_run(method,name):
    directory=output/method/name
    full=directory/"full"
    cfg=load(full/"config.json")
    current=load(full/"summary.json")
    rows=load(full/"predictions.json")
    assert cfg["case"]==current["case"]==method and cfg["dataset_sha256"]==source_specs[name]["sha256"]
    assert not current["failed"] and current["samples"]==current["requested"]==len(rows)==counts[name]
    assert sorted(cfg["sample_ids"])==list(range(counts[name]))
    assert [row["problem_id"] for row in rows]==list(range(counts[name]))
    total_scores=0
    for row in rows:
        trace=load(full/"problems"/f"{row['problem_id']:03d}"/"result.json")
        obs=trace["observation"]
        assert trace["dataset_unique_id"]==row["dataset_unique_id"]
        assert len(obs["llm_calls"])==row["llm_calls"]
        assert len(obs["prm_calls"])==10, (method,name,row["problem_id"])
        row["prm_calls"]=len(obs["prm_calls"])
        row["question_retries"]=len(obs["failed_attempts"])
        total_scores+=row["prm_calls"]
    assert total_scores==current["prm_calls"]==10*len(rows)
    attempts=[(float(a),float(b),int(c)) for a,b,c in
              (line.split("\t") for line in (directory/"attempts.tsv").read_text().splitlines())]
    assert attempts and attempts[-1][2]==0
    seconds = current["seconds"] if len(attempts)==1 else sum(stop-start for start,stop,_ in attempts)
    totals=combine(rows,seconds,current["question_retries"],total_scores)
    assert totals["correct"]==current["correct"] and totals["output_tokens"]==current["output_tokens"]
    assert totals["generation_calls"]==current["generation_calls"]
    info={"concurrency":cfg["concurrency"],"attempts":attempts,
          "timing_basis":"evaluator elapsed" if len(attempts)==1 else "sum of attempt wall times, including setup and failed work",
          "config":cfg,"summary":current}
    if name=="math500" or name in h20_extensions:
        past=[h20_formal[f"gpu-{i}"][method] if name=="math500" else
              h20_extensions[name][method][f"gpu-{i}"] for i in (0,1)]
        for line in past:
            common_config(cfg,line["config"])
        assert sorted(past[0]["config"]["sample_ids"]+past[1]["config"]["sample_ids"])==list(range(len(rows)))
        old=[p["summary"] for p in past]
        info["h20"]={"questions":sum(x["samples"] for x in old),"correct":sum(x["correct"] for x in old),
                     "elapsed_seconds":sum(x["seconds"] for x in old),
                     "output_tokens":sum(x["output_tokens"] for x in old),
                     "generation_calls":sum(x["generation_calls"] for x in old),
                     "prm_calls":sum(x["prm_calls"] for x in old)}
        info["h20"]["accuracy"]=info["h20"]["correct"]/info["h20"]["questions"]
    return rows,totals,info

records={}
for name,count in counts.items():
    if name=="math500":
        first=read_rows(e1_math/"predictions.jsonl",old_math["predictions_sha256"])
        e1=convert(old_math["aggregate"]["autodl"],"sum_shard_seconds")
        h20=convert(old_math["aggregate"]["h20"],"sum_shard_seconds")
        e1_cfg=[load(e1_math/f"shard-{i}/config.json") for i in (0,1)]
        e1_notes="Two serial 250-question shards on one 4090."
    else:
        prior=old_ext[name]
        first=read_rows(e1_ext/name/"predictions.jsonl",prior["predictions_sha256"])
        e1=convert(prior["autodl"])
        h20=convert(prior["h20"]) if "h20" in prior else None
        e1_notes="; ".join(prior["phase_labels"])
        original=work/"results-e1-other-datasets-20261128"
        if name=="aime24":
            e1_cfg=[load(original/name/f"shard-{i}/config.json") for i in (0,1)]
        elif name=="amo":
            e1_cfg=[load(original/name/"shard-0/config.json"),load(e1_ext/name/"full/config.json")]
        else:
            e1_cfg=[load(e1_ext/name/"full/config.json")]
    identities=[(row["problem_id"],row["dataset_unique_id"]) for row in first]
    assert len(first)==len(set(identities))==count and [i for i,_ in identities]==list(range(count))
    method_rows={"E1":(first,e1,{"h20":h20,"timing_basis":e1_notes,"configurations":e1_cfg})}
    after=e3_results[name]
    e3=read_rows(e3_root/name/"predictions.jsonl",after["predictions_sha256"])
    method_rows["E3"]=(e3,convert(after["autodl"]),{"h20":after.get("h20"),"config":load(e3_root/name/"full/config.json")})
    for method in ("E2","E4"):
        if f"{method}/{name}" not in pending_runs:
            method_rows[method]=prm_run(method,name)
    for method in methods:
        if method not in method_rows:
            continue
        rows, stats, provenance=method_rows[method]
        assert [(row["problem_id"],row["dataset_unique_id"]) for row in rows]==identities, (method,name)
        assert stats["questions"]==count and stats["correct"]==sum(x["is_correct"] is True for x in rows)
        artifact=bundle/"predictions"/method/f"{name}.jsonl.gz"
        artifact.parent.mkdir(parents=True,exist_ok=True)
        raw="".join(json.dumps(row,ensure_ascii=False)+"\n" for row in rows).encode()
        artifact.write_bytes(gzip.compress(raw,mtime=0))
        stats["predictions_sha256"]=digest(raw)
        stats["h20"]=provenance.get("h20")
        stats["concurrency"]=provenance.get("concurrency",(provenance["config"] if "config" in provenance else e1_cfg[0])["concurrency"])
        stats["paired_vs_e1"]=dict(Counter(
            ("e1_right" if prior["is_correct"] is True else "e1_wrong")+
            ("_method_right" if row["is_correct"] is True else "_method_wrong")
            for prior,row in zip(first,rows,strict=True)
        ))
        if name=="gsm8k":
            stats["historical_500_correct"]=sum(row["is_correct"] is True for row in rows[:500])
        records[f"{method}/{name}"]=stats
        save(bundle/"provenance"/method/f"{name}.json",provenance)
        print(method,name,stats["correct"],count,flush=True)

expected_runs = {f"{method}/{name}" for method in methods for name in counts} - pending_runs
assert set(records) == expected_runs
assert sum(item["questions"] for item in records.values()) == (6487 if paused else 7876)
save(bundle/"archive-status.json", {
    "status": "paused_after_amo_bench" if paused else "complete",
    "completed_runs": sorted(records),
    "pending_runs": sorted(pending_runs),
    "completed_questions": sum(item["questions"] for item in records.values()),
    "paused_at": pause_marker.read_text().strip() if paused else None,
})
save(bundle/"summary.json",records)
with (bundle/"summary.csv").open("w",encoding="utf-8",newline="") as handle:
    writer=csv.writer(handle)
    writer.writerow(("method","dataset","questions","correct","accuracy_percent","elapsed_seconds","output_tokens",
                     "generation_calls","prm_calls","concurrency","h20_correct","h20_shard_sum_seconds"))
    for key,item in records.items():
        old=item["h20"] or {}
        writer.writerow((*key.split("/"),item["questions"],item["correct"],f"{100*item['accuracy']:.4f}",
                         f"{item['elapsed_seconds']:.6f}",item["output_tokens"],item["generation_calls"],item["prm_calls"],
                         item["concurrency"],old.get("correct",""),old.get("elapsed_seconds","")))

code=bundle/"runner-snapshot"
code.mkdir(exist_ok=True)
for file in ("run_autodl_e1_formal.py","audit_e1_resume.py","run_autodl_e1_resume.py","run_autodl_e1_datasets.py",
             "run_autodl_e3_math500.py","run_autodl_e3_extensions.py","audit_e3_datasets.py","compare_e3_e1.py",
             "run_autodl_prm_math500.py","run_autodl_prm_extensions.py","serve_autodl_prm.py",
             "engine-actual.requirements.txt","client-actual.requirements.txt","prm-actual.requirements.txt","serve_e1.sh"):
    shutil.copyfile(work/file,code/file)
for file in ("run_published_code.py","run_smoke.py","validate_models.py","audit_results.py","paper_protocol.py"):
    shutil.copyfile(work/"scripts"/file,code/file)
for file in ("run_published_extensions.py","serve_prm.py"):
    shutil.copyfile(h20_ext/"scripts"/file,code/file)
shutil.copyfile(server/"datasets/manifest.json",bundle/"dataset-manifest.json")
shutil.copyfile(server/"datasets/gsm8k-500-questions.json",bundle/"gsm8k-500-questions.json")
shutil.copyfile(e3_root/"e3-vs-e1-4090.json",bundle/"e3-vs-e1-4090.json")
lines=[
    "# Qwen2.5-7B-Instruct 在 AutoDL 4090 上的 E1–E4 复现",
    "",
    "单张 AutoDL 4090 48 GiB 使用原版 Qwen2.5-7B-Instruct BF16、vLLM 0.33.0、固定数据快照和种子。"
    "E2、E4 使用原版 Qwen2.5-Math-PRM-7B BF16 和作者代码的选优规则。"
    + ("阶段归档共 21 组、6,487 条逐题结果。E1–E3 完成了全部六个数据集。"
       "E4 完成 MATH-500、AIME24 和 AMO-Bench，已暂停。AMC23、AIME 2026 和 GSM8K 待续。"
       if paused else "四种方法各评测 1,969 题。")
    + "GSM8K 全量评测使用 1,319 道测试题。",
    "",
    "| 数据集 | 方法 | 4090 答对 | 4090 耗时（秒） | 输出 token | 生成请求 | PRM 请求 | H20 答对 | H20 耗时合计（秒） |",
    "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for name in counts:
    for method in methods:
        if f"{method}/{name}" in pending_runs:
            lines.append(f"| {names[name]} | {method} | 待运行 | — | — | — | — | — | — |")
            continue
        row=records[f"{method}/{name}"]
        old=row["h20"] or {}
        approx="约 " if method=="E1" and name=="amo" else ""
        lines.append(f"| {names[name]} | {method} | {row['correct']}/{row['questions']} ({100*row['accuracy']:.1f}%) | "
                     f"{approx}{row['elapsed_seconds']:.2f} | {row['output_tokens']:,} | {row['generation_calls']:,} | "
                     f"{row['prm_calls']:,} | {str(old['correct'])+'/'+str(old['questions']) if old else '—'} | "
                     f"{format(old['elapsed_seconds'],'.2f') if old else '—'} |")
lines.extend([
    "",
    "H20 对照使用原始两个分片的计时之和。AIME24 的 E1 使用最早完成的两组 15 题。"
    "MATH-500 的 E1 在同一张卡上依次评测两个 250 题分片。"
    "AMO-Bench 的 E1 合并先完成的 6 题和后续 44 题，中断前阶段耗时按日志时间估算。",
    "",
    "GSM8K 历史固定 500 题的答对数为 " +
    "、".join(f"{m} {records[f'{m}/gsm8k']['historical_500_correct']}/500" for m in methods if f"{m}/gsm8k" in records) + "。",
    "",
    "汇总见 [CSV](summary.csv) 和 [JSON](summary.json)。[逐题结果](predictions/) 按方法和数据集保存。"
    "[E3 与 E1 的逐题配对汇总](e3-vs-e1-4090.json) 列出正确率、token、请求数和耗时。"
    "每轮并发、配置和计时口径见 [运行记录](provenance/)。"
    "[完成范围](archive-status.json) 列出完成与待运行的组别。",
    "",
    "[4090 压测与 H20 对照](benchmark/h20-comparison.csv) 包含 vLLM 并发试验。"
    "[压测记录](benchmark/) 保存九组完成的压测、预热记录，以及按要求停止的 E4 24 并发压测（39/48 题）。",
    "",
    "### 实验版本",
    "",
    "上游代码固定于 `d9ca5f71f5c7d155153e8f7decd8b1c524968775`。"
    "模型版本、解码参数和数据校验值保存在运行配置和 [数据清单](dataset-manifest.json)。"
    "[运行器与依赖快照](runner-snapshot/) 是实际运行使用的版本。",
    "",
])
if (output/"e3-retry").is_file():
    lines.append("E3 曾从已保留的逐题结果续跑，保存的评测秒数对应最后一次运行的计时。")
(bundle/"README.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

benchmark = work/"results-autodl"
bench_out = bundle/"benchmark"
bench_out.mkdir(exist_ok=True)
for file in ("comparison.json", "h20-comparison.csv", "h20-comparison.json"):
    shutil.copyfile(benchmark/file, bench_out/file)
for file in ("on_autodl.py", "on_autodl_full.py", "summarize_full.py"):
    shutil.copyfile(work/file, code/file)
shutil.copyfile(archive_path := Path(__file__), code/archive_path.name)
trial_dirs = sorted(path for path in benchmark.iterdir() if path.is_dir() and (path/"summary.json").is_file())
expected_trials = {
    "warmup", "e1-c16", "e1-c32", "e1-c64", "e1-c96",
    "e2-c24", "e2-c48", "e3-c32", "e3-c64", "e3-c96",
}
assert {trial.name for trial in trial_dirs} == expected_trials, trial_dirs
for trial in trial_dirs:
    out = bench_out/trial.name
    out.mkdir(exist_ok=True)
    for file in ("config.json", "summary.json"):
        shutil.copyfile(trial/file, out/file)
    predictions = trial/"predictions.json"
    if predictions.is_file():
        (out/"predictions.json.gz").write_bytes(gzip.compress(predictions.read_bytes(), mtime=0))

interrupted = benchmark/"e4-c24"
interrupted_config = load(interrupted/"config.json")
finished = []
for result in sorted((interrupted/"problems").glob("*/result.json")):
    row = load(result)
    finished.append({
        "problem_id": row["problem_id"],
        "dataset_unique_id": row["dataset_unique_id"],
        "answer": row["final_predicted_answer"],
        "is_correct": row["final_is_correct"],
        "question_seconds": row["total_time_seconds"],
    })
requested = set(interrupted_config["sample_ids"])
finished_ids = {row["problem_id"] for row in finished}
assert len(finished) == len(finished_ids) == 39 and len(requested) == 48 and finished_ids <= requested
stop_record = load(server/"logs"/"h20-full.stopped.json")
save(bench_out/"e4-c24"/"config.json", interrupted_config)
save(bench_out/"e4-c24"/"partial.json", {
    "status": "interrupted_by_request",
    "stopped_at": stop_record["requested_at"],
    "requested": len(requested),
    "completed": len(finished),
    "pending_ids": sorted(requested - finished_ids),
    "predictions": finished,
})

manifest={str(path.relative_to(bundle)):digest(path.read_bytes()) for path in sorted(bundle.rglob("*"))
          if path.is_file() and path.name!="SHA256SUMS"}
(bundle/"SHA256SUMS").write_text("".join(f"{sha}  {name}\n" for name,sha in manifest.items()),encoding="utf-8")
