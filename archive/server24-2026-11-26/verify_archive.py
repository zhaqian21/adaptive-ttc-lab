"""Check archive checksums and the unique coverage of the historical traces."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

def verify(root):
    inventory = {}
    for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or name in inventory:
            raise ValueError(f"Invalid checksum path: {name}")
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Checksum mismatch: {name}")
        inventory[name] = digest
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()} - {"SHA256SUMS"}
    if actual != set(inventory):
        raise ValueError("Archive file list differs from checksum inventory")

    groups = json.loads((root / "trace_manifest.json").read_bytes())
    if len({g["group"] for g in groups}) != len(groups):
        raise ValueError("Repeated experiment group")
    counts = Counter()
    for g in groups:
        ids = set()
        index = set()
        total = 0
        for name, digest in g["files"].items():
            if inventory.get(name) != digest:
                raise ValueError(f"Trace checksum declaration differs: {name}")
            with gzip.open(root / name, "rt", encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    if row["dataset_unique_id"] in ids or row["problem_id"] in index:
                        raise ValueError(f"Repeated question in {g['group']}")
                    ids.add(row["dataset_unique_id"])
                    index.add(row["problem_id"])
                    candidates = row["candidates"]
                    if [c["iteration"] for c in candidates] != list(range(1, 11)):
                        raise ValueError(f"Incomplete candidate order in {g['group']}")
                    total += len(candidates)
        if len(ids) != g["questions"] or total != g["candidates"]:
            raise ValueError(f"Wrong trace counts for {g['group']}")
        counts.update(experiments=1, questions=len(ids), candidates=total)
    prs = json.loads((root / "prior-prs.json").read_bytes())
    if len({p["pr"] for p in prs}) != len(prs):
        raise ValueError("Duplicate source PR reference")
    print(f"{len(inventory) + 1} files verified. {len(prs)} source PRs. "
          f"{counts['experiments']} trace groups, {counts['questions']} question records, "
          f"{counts['candidates']} candidates.")

if __name__ == "__main__":
    verify(Path(__file__).resolve().parent)
