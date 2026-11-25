"""The 10:30 planning window for the optional E2 cohorts."""
from datetime import datetime
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "schedule.json").read_bytes())
TARGET = datetime.fromisoformat(CONFIG["target"])
COHORTS = ("math", "aime24", "amo")
COUNTS = {"math": 500, "aime24": 30, "amo": 50}


def eligible(dataset, now):
    return dataset == "math" or ((TARGET - now).total_seconds() >= CONFIG["optional_min_remaining_seconds"][dataset])
