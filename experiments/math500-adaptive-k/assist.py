"""Let the free GPU help the original worker using the same immutable run code."""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
from threading import Event, Thread
import time
from types import SimpleNamespace
from unittest.mock import patch

import run_adaptive_k as experiment


baseline = experiment.baseline


def completed(shard):
    return {int(path.stem) for path in (shard / "problems").glob("*.json")}


def run(root):
    shards = [root / f"shard-{lane}" for lane in (0, 1)]
    target, donor = shards
    for _ in range(4_320):
        execution = donor / "execution.json"
        if execution.exists():
            status = baseline.read(execution)
            if status["complete"] == status["requested"] == 250 and not status["failures"]:
                break
        time.sleep(5)
    else:
        raise RuntimeError("GPU 1 did not finish its original shard")

    config = baseline.read(target / "config.json")
    other = baseline.read(donor / "config.json")
    if config["mode"] != "paired" or config["sample"] is not None or any(
        config[key] != other[key] for key in config if key not in ("shard", "question_ids")
    ):
        raise ValueError("The two original shards do not share the formal setup")

    done = completed(target)
    target_ids = config["question_ids"]
    # The original worker submits sorted IDs to a FIFO pool. Stay past its active
    # workers and leave another half-pool of room for writes during startup.
    frontier = len(done) + config["workers"] + config["workers"] // 2
    selected = [index for index in target_ids[frontier:] if index not in done][-32:]
    output = root / "assisted-on-gpu1"
    output.mkdir()
    baseline.write(output / "dispatch.json", {
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "donor_shard": 1,
        "target_shard": 0, "already_complete": len(done), "ids": selected,
    })
    if not selected:
        baseline.write(output / "progress.json", {"linked_ids": [], "already_present": [], "finished": True})
        return 0

    wanted = set(selected)
    linked, already_present = set(), set()

    def publish(finished=False):
        for index in sorted(completed(output) - linked - already_present):
            if index not in wanted:
                raise ValueError("Unexpected question in assistant output")
            name = f"{index:03d}.json"
            try:
                os.link(output / "problems" / name, target / "problems" / name)
            except FileExistsError:
                already_present.add(index)
            else:
                linked.add(index)
        baseline.write(output / "progress.json", {
            "linked_ids": sorted(linked), "already_present": sorted(already_present), "finished": finished,
        })

    stop = Event()

    def sync():
        while not stop.wait(3):
            publish()

    original_dataset = baseline.dataset_for_run

    def subset(args):
        return [row for row in original_dataset(args) if row["problem_id"] in wanted]

    args = SimpleNamespace(output=output, mode="paired", port=18421, prm_port=18422, shard=0,
                           workers=config["workers"], sample=None, model=config["model"], seed=config["seed"],
                           context=config["model_context"], vllm_version=config["vllm_version"])
    reader = Thread(target=sync, daemon=True)
    reader.start()
    try:
        with patch.object(baseline, "dataset_for_run", subset):
            return experiment.run(args)
    finally:
        stop.set()
        reader.join()
        publish(finished=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    raise SystemExit(run(parser.parse_args().root))
