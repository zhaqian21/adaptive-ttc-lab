"""Policies for paired token limits and small-pool direct voting."""
from collections import Counter


SHORT = 1024
LONG = 2048
DRAWS = 5
DEFAULT_STUDIES = ("continuation", "voting")
STUDIES = (*DEFAULT_STUDIES, "remaining")


def key(answer):
    return answer.strip() if isinstance(answer, str) else ""


def continuation(row):
    first, full = row["short"], row["full"]
    extend = row["output_tokens"] > SHORT and not key(first["answer"])
    return {"fixed1024": first, "fixed2048": full, "on_demand": full if extend else first,
            "extended": extend}


def majority(candidates):
    keys = [key(c["answer"]) for c in candidates]
    counts = Counter(k for k in keys if k)
    if not counts:
        return 0
    return max((i for i, k in enumerate(keys) if k), key=lambda i: counts[keys[i]])
