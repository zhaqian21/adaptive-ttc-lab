"""Download the pinned checkpoints and the MATH-500 test file."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
ASSETS = (
    ("dataset", "HuggingFaceH4/MATH-500", "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"),
    ("model", "Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28"),
    ("model", "Qwen/Qwen2.5-Math-PRM-7B", "0610740060112df12585d00a1c5f4624d2f59051"),
)


def digest(path, method, git_blob=False):
    result = hashlib.new(method)
    if git_blob:
        result.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def curl(*args):
    return subprocess.run(
        ["curl", "--fail", "--location", "--silent", "--show-error",
         "--connect-timeout", "15", "--retry", "3", "--retry-delay", "2", *args],
        check=True, stdout=subprocess.PIPE,
    ).stdout


def download(endpoint, asset, directory, entry):
    kind, repo, revision = asset
    name = entry["rfilename"]
    target = directory / name
    temporary = target.with_suffix(target.suffix + ".part")
    remote_hash = (entry.get("lfs") or {}).get("sha256")
    method = "sha256" if remote_hash else "sha1"
    expected = remote_hash or entry["blobId"]
    if kind == "dataset":
        prefix = "datasets/"
    else:
        prefix = ""
    if not target.exists():
        print(f"Downloading {repo}/{name}", flush=True)
        if not temporary.exists() or temporary.stat().st_size != entry["size"]:
            curl("--output", str(temporary), "--continue-at", "-",
                 f"{endpoint}/{prefix}{repo}/resolve/{revision}/{name}")
        if temporary.stat().st_size != entry["size"] or digest(temporary, method, not remote_hash) != expected:
            raise ValueError(f"Size or object hash mismatch for {temporary}")
        temporary.rename(target)
    elif target.stat().st_size != entry["size"] or digest(target, method, not remote_hash) != expected:
        raise ValueError(f"Existing file does not match the pinned object: {target}")
    print(f"Verified {repo}/{name} ({entry['size']} bytes)", flush=True)
    return {"name": name, "bytes": entry["size"], "hash_algorithm": method, "object_hash": expected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="https://huggingface.co")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    endpoint = args.endpoint.rstrip("/")
    records = []
    for asset in ASSETS:
        kind, repo, revision = asset
        directory = ROOT / "assets" / kind / f"{repo.replace('/', '--')}--{revision}"
        directory.mkdir(parents=True, exist_ok=True)
        metadata = json.loads(curl("--max-time", "45", f"{endpoint}/api/{kind}s/{repo}/revision/{revision}?blobs=true"))
        if metadata["sha"] != revision:
            raise ValueError(f"Endpoint returned the wrong revision for {repo}")
        files = [
            entry for entry in metadata["siblings"]
            if re.fullmatch(r"[\w.-]+", entry["rfilename"]) and not entry["rfilename"].startswith(".")
        ]
        required = {"test.jsonl"} if kind == "dataset" else {"config.json", "tokenizer.json", "model.safetensors.index.json"}
        names = {entry["rfilename"] for entry in files}
        if not required <= names or (kind == "model" and not any(name.endswith(".safetensors") for name in names)):
            raise ValueError(f"Missing expected checkpoint or data files for {repo}")
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            downloaded = list(pool.map(lambda entry: download(endpoint, asset, directory, entry), files))
        records.append({"kind": kind, "repo": repo, "revision": revision,
                        "path": str(directory.relative_to(ROOT)), "files": downloaded})
        manifest = ROOT / "assets" / "manifest.json"
        pending = manifest.with_suffix(".tmp")
        pending.write_text(json.dumps({"download_endpoint": endpoint, "assets": records}, indent=2) + "\n")
        pending.replace(manifest)


if __name__ == "__main__":
    main()
