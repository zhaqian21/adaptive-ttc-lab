"""Download and verify the fixed Llama, PRM and benchmark assets."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess


def verified(path, entry):
    if not path.is_file():
        return False
    size = path.stat().st_size
    if size == 0 or ("size" in entry and size != entry["size"]):
        return False
    if "sha256" in entry:
        digest = hashlib.sha256()
        expected = entry["sha256"]
    else:
        digest = hashlib.sha1(f"blob {size}\0".encode())
        expected = entry["sha1"]
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest() == expected


def download(root, entry):
    path = root / entry["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    if verified(path, entry):
        print(f"Verified existing {entry['path']}", flush=True)
        return
    partial = path.with_suffix(path.suffix + ".partial")
    for url in entry["urls"]:
        if verified(partial, entry):
            partial.replace(path)
            return
        if partial.exists() and entry.get("size", float("inf")) <= partial.stat().st_size:
            partial.unlink()
        completed = subprocess.run([
            "curl", "--fail", "--silent", "--show-error", "--location", "--connect-timeout", "15",
            "--retry", "3", "--retry-delay", "3", "--max-time", "1500", "--speed-limit", "1024",
            "--speed-time", "90", "--continue-at", "-", "--output", str(partial), url,
        ], check=False)
        if verified(partial, entry):
            partial.replace(path)
            print(f"Verified {entry['path']}", flush=True)
            return
        # A successful download with a mismatched digest must not seed the next mirror.
        if completed.returncode != 18 and partial.exists():
            partial.unlink()
    raise RuntimeError(f"Could not verify any source for {entry['path']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    entries = json.loads(args.manifest.read_text(encoding="utf-8"))
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(lambda entry: download(args.root, entry), entries))
