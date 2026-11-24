"""Create a checksum-verified task-table launcher from the archived release files."""

import argparse
import base64
import gzip
import hashlib
import io
from pathlib import Path
import tarfile


BASELINE_FILES = (
    "scripts/bootstrap.sh", "scripts/constraints.txt", "scripts/audit_results.py",
    "scripts/fetch_assets.py", "scripts/paper_protocol.py", "scripts/run_smoke.py",
    "scripts/validate_models.py", "environment.lock.txt",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="The archived MATH-500 public-code release directory")
    parser.add_argument("output", type=Path, help="A shell script that can be uploaded to the 825 task table")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    release = here.parent / "aime24-amo-bench-published-code-2026-11-24"
    files = {name: here / name for name in ("assets.json", "bootstrap_825.sh", "prepare_assets.py", "llama_table1.py")}
    files.update({f"baseline/{Path(name).name}": args.baseline / name for name in BASELINE_FILES})
    files.update({f"released/{name}": release / "scripts" / name
                  for name in ("run_published_extensions.py", "serve_prm.py")})
    files["datasets.json"] = release / "config" / "datasets.json"
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as stream:
        for name, path in sorted(files.items()):
            contents = path.read_bytes().replace(b"\r\n", b"\n")
            info = tarfile.TarInfo(name)
            info.size = len(contents)
            stream.addfile(info, io.BytesIO(contents))
    compressed = gzip.compress(archive.getvalue(), mtime=0)
    digest = hashlib.sha256(compressed).hexdigest()
    encoded = base64.encodebytes(compressed).decode()
    launcher = f"""#!/usr/bin/env bash
set -euo pipefail
root=$1
umask 077
bundle="$root/bundles/{digest}"
mkdir -p "$bundle"
base64 -d > "$bundle/job.tar.gz.partial" <<'BUNDLE'
{encoded}BUNDLE
printf '%s  %s\\n' '{digest}' "$bundle/job.tar.gz.partial" | sha256sum --check --status
mv -f "$bundle/job.tar.gz.partial" "$bundle/job.tar.gz"
tar -xzf "$bundle/job.tar.gz" -C "$bundle"
exec bash "$bundle/bootstrap_825.sh" "$root"
"""
    payload = launcher.encode()
    if len(payload) > 1 << 20:
        raise ValueError("The launcher exceeds the 825 task-table upload limit")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    print(f"{digest} {len(payload)} {args.output}")


if __name__ == "__main__":
    main()
