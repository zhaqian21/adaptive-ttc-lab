from pathlib import Path
import json
import re

from audit_results import check_problem


HERE = Path(__file__).resolve().parent
BATCH = HERE.parent.parent


def read(name):
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def completed_rate(measurement, gpu):
    key = f"{gpu}_committed"
    before = measurement["before"]["leases"].get(key, 0)
    after = measurement["after"]["leases"].get(key, 0)
    return 60 * (after - before) / measurement["elapsed_seconds"]


def main():
    baseline = read("baseline-32-32.json")
    trial = read("trial-64-32.json")
    old, new = baseline["gpu"], trial["gpu"]
    raw = new["0"]["out_tokens_per_second"] / old["0"]["out_tokens_per_second"]
    paired = raw / (new["1"]["out_tokens_per_second"] / old["1"]["out_tokens_per_second"])
    old_rate, new_rate = completed_rate(baseline, 0), completed_rate(trial, 0)
    audited = 0
    for path in (BATCH / "E4" / "problems").glob("*/result.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        observation = row["observation"]
        if observation.get("physical_gpu") == 0 and observation.get("worker_slots") == 64:
            index = int(path.parent.name)
            check_problem(path.parent, row, index, 10, "paper_description")
            audited += 1
    logs = ""
    for file in ("engine.log", "evaluator.log"):
        logs += (HERE / "gpu0-c64-trial" / file).read_text(encoding="utf-8", errors="replace")
    resource_error = bool(re.search(r"CUDA out of memory|OutOfMemoryError|EngineDeadError|CUBLAS_STATUS_ALLOC_FAILED", logs, re.I))
    checks = {"tokens_per_second_at_least_20_percent": raw >= 1.20,
              "paired_gain_at_least_20_percent": paired >= 1.20,
              "completion_rate_at_least_baseline": new_rate >= old_rate,
              "vllm_64_slot_utilization_confirmed": new["0"]["running_median"] >= 48,
              "no_preemptions_or_resource_errors": new["0"]["preemptions_delta"] == 0 and not resource_error,
              "at_least_three_64_slot_checkpoints_audited": audited >= 3}
    result = {"promote_gpu1": all(checks.values()), "checks": checks,
              "raw_throughput_ratio": raw, "paired_throughput_ratio": paired,
              "gpu0_32_tokens_per_second": old["0"]["out_tokens_per_second"],
              "gpu0_64_tokens_per_second": new["0"]["out_tokens_per_second"],
              "gpu1_32_baseline_tokens_per_second": old["1"]["out_tokens_per_second"],
              "gpu1_32_trial_tokens_per_second": new["1"]["out_tokens_per_second"],
              "gpu0_32_questions_per_minute": old_rate, "gpu0_64_questions_per_minute": new_rate,
              "gpu0_c64_verified_questions": audited}
    pending = HERE / "decision.tmp"
    pending.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    pending.replace(HERE / "decision.json")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
