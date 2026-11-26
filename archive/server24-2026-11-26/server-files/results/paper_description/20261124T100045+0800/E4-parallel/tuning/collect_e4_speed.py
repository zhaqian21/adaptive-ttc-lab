"""Measure live E4 token throughput against the unchanged GPU's workload."""
from collections import Counter
import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import statistics
import time
from urllib.request import urlopen


METRICS = {"generation_tokens_total", "num_requests_running", "num_requests_waiting", "kv_cache_usage_perc",
           "num_preemptions_total", "request_decode_time_seconds_sum", "request_decode_time_seconds_count"}


def sample(port):
    with urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5) as reply:
        text = reply.read().decode()
    values = Counter()
    for item in text.splitlines():
        match = re.match(r"^vllm:([a-z_]+)(?:\{[^}]*\})?\s+([0-9.eE+\-]+)$", item)
        if match and match[1] in METRICS:
            values[match[1]] += float(match[2])
    if "generation_tokens_total" not in values:
        raise RuntimeError(f"Missing generation counter on port {port}")
    return dict(values)


def checkpoints(root):
    states = Counter()
    for path in (root / "E4-parallel" / "leases").glob("*.json"):
        finish = json.loads(path.read_text())[-1]
        states[str(finish["gpu"]) + "_" + finish.get("status", "working")] += 1
    return {"total": len(list((root / "E4" / "problems").glob("*/result.json"))), "leases": dict(states)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase")
    parser.add_argument("--seconds", type=int, default=120)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent.parent
    output = root / "E4-parallel" / "tuning" / f"{args.phase}.json"
    tz = timezone(timedelta(hours=8))
    started_at = datetime.now(tz).isoformat(timespec="seconds")
    begin = checkpoints(root)
    samples = []
    start = time.monotonic()
    while True:
        now = time.monotonic()
        samples.append({"elapsed": now - start, "0": sample(18409), "1": sample(18411)})
        if now - start >= args.seconds - 0.1:
            break
        time.sleep(min(15, max(0.1, args.seconds - (time.monotonic() - start))))
    end = checkpoints(root)
    elapsed = samples[-1]["elapsed"] - samples[0]["elapsed"]
    result = {"phase": args.phase, "started_at": started_at, "elapsed_seconds": elapsed,
              "before": begin, "after": end, "completed_per_minute": 60 * (end["total"] - begin["total"]) / elapsed,
              "gpu": {}, "samples": samples}
    for gpu in ("0", "1"):
        first, last = samples[0][gpu], samples[-1][gpu]
        result["gpu"][gpu] = {
            "out_tokens_per_second": (last["generation_tokens_total"] - first["generation_tokens_total"]) / elapsed,
            "running_median": statistics.median(item[gpu].get("num_requests_running", 0) for item in samples),
            "preemptions_delta": last.get("num_preemptions_total", 0) - first.get("num_preemptions_total", 0),
            "peak_kv_usage": max(item[gpu].get("kv_cache_usage_perc", 0) for item in samples),
        }
    pending = output.with_suffix(".tmp")
    pending.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    pending.replace(output)
    print(json.dumps({k:v for k,v in result.items() if k!="samples"},indent=2))


if __name__ == "__main__":
    main()

