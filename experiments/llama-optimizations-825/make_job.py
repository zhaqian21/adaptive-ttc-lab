"""Create a standalone 825 launcher with a verified copy of the run sources."""

import argparse
import base64
import gzip
import hashlib
import io
from pathlib import Path
import tarfile


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
SOURCES = {
    "math500-published-code-2026-11-24": (
        "environment.lock.txt", "scripts/bootstrap.sh", "scripts/constraints.txt",
        "scripts/audit_results.py", "scripts/fetch_assets.py", "scripts/paper_protocol.py", "scripts/run_smoke.py",
        "scripts/validate_models.py", "scripts/run_published_code.py",
    ),
    "llama-table1-825": ("assets.json", "prepare_assets.py", "llama_table1.py", "answer_fields.py"),
    "aime24-amo-bench-published-code-2026-11-24": ("scripts/serve_prm.py", "scripts/run_published_extensions.py"),
    "amc23-adaptive-revision": ("run_amc23.py",),
    "math500-adaptive-revision": ("run_math500.py", "revision_policy.py", "audit_math500.py"),
    "math500-adaptive-k": ("run_adaptive_k.py", "sampling.py", "audit.py"),
    "math500-prm-answer-aggregation": ("answer_aggregation.py",),
    "math500-online-prm-aggregation": ("compare.py",),
    "math500-prm-routing": ("run_modules.py", "routing.py", "replay_modules.py", "audit.py", "matched.py"),
}


def make(output):
    files = {name: HERE / name for name in ("run_suite.py", "bootstrap_825.sh")}
    for directory, names in SOURCES.items():
        files.update({f"experiments/{directory}/{name}": EXPERIMENTS / directory / name for name in names})
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as stream:
        for name, path in sorted(files.items()):
            data = path.read_bytes().replace(b"\r\n", b"\n")
            member = tarfile.TarInfo(name)
            member.size = len(data)
            stream.addfile(member, io.BytesIO(data))
    compressed = gzip.compress(archive.getvalue(), mtime=0)
    digest = hashlib.sha256(compressed).hexdigest()
    encoded = base64.encodebytes(compressed).decode()
    launcher = f"""#!/usr/bin/env bash
set -euo pipefail
umask 077
mkdir -p -- "$1"
root="$(realpath -- "$1")"
exec 9>"$root/run.lock"
flock -n 9 || {{ echo "Run already has a coordinator" >&2; exit 1; }}
digest={digest}
if [[ -f "$root/bundle.sha256" && "$(cat "$root/bundle.sha256")" != "$digest" ]]; then
    echo "Checkpoint directory belongs to a different upload" >&2
    exit 1
fi
printf '%s\\n' "$digest" > "$root/bundle.sha256"
bundle="$root/bundles/$digest"
mkdir -p -- "$bundle"
base64 -d > "$bundle/job.tar.gz.partial" <<'BUNDLE'
{encoded}BUNDLE
printf '%s  %s\\n' "$digest" "$bundle/job.tar.gz.partial" | sha256sum --check --status
mv -- "$bundle/job.tar.gz.partial" "$bundle/job.tar.gz"
tar -xzf "$bundle/job.tar.gz" -C "$bundle"
exec bash "$bundle/bootstrap_825.sh" "$root"
"""
    data = launcher.encode()
    if len(data) > 1 << 20:
        raise ValueError("The standalone launcher exceeds the upload limit")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(data)
    return digest, len(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    digest, size = make(args.output)
    print(f"{digest} {size} {args.output}")
