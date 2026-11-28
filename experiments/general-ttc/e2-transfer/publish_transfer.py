"""Publish verified E2 transfer records from AutoDL using its GitHub CLI login."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

from archive_transfer import verify

REPO = "zhaqian21/adaptive-ttc-lab"
BASE = "codex/mmlu-pro-e2-online-ablation"
BRANCH = "codex/mmlu-pro-e2-transfer"
EXPECTED_BASE = "275f76459db89fab7ceca84b46d8f0373d224513"
SECRET = re.compile(
    rb"-----BEGIN (?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY-----|"
    rb"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|hf_[A-Za-z0-9]{30,})\b"
)


def api(path, *, method="GET", payload=None, missing=False):
    command = ["gh", "api", "--method", method, f"repos/{REPO}/{path}"]
    body = None
    if payload is not None:
        command.extend(["--input", "-"])
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    for attempt in range(4):
        try:
            reply = subprocess.run(command, input=body, capture_output=True, timeout=100)
        except subprocess.TimeoutExpired as error:
            if attempt == 3:
                raise RuntimeError(f"GitHub timed out for {method} {path}") from error
        else:
            if reply.returncode == 0:
                return json.loads(reply.stdout)
            error = reply.stderr.decode(errors="replace")
            if missing and "HTTP 404" in error:
                return None
            if attempt == 3 or re.search(r"HTTP 40[0134]|HTTP 422", error):
                raise RuntimeError(f"GitHub rejected {method} {path}: {error.strip()}")
        time.sleep(3 * (attempt + 1))


def head(branch):
    ref = api(f"git/ref/heads/{branch}", missing=True)
    return ref["object"]["sha"] if ref else None


def upload_blob(file):
    data = file.read_bytes()
    decoded = gzip.decompress(data) if file.suffix == ".gz" else data
    if SECRET.search(decoded):
        raise ValueError(f"Credential pattern in {file.name}")
    git_id = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    created = api("git/blobs", method="POST", payload={
        "encoding": "base64", "content": base64.b64encode(data).decode("ascii"),
    })
    assert created["sha"] == git_id
    return git_id


def commit_files(repo, files, parent, message):
    base_tree = api(f"git/commits/{parent}")["tree"]["sha"]
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = [
            {"path": file.relative_to(repo).as_posix(), "mode": "100644", "type": "blob", "sha": blob}
            for file, blob in zip(files, pool.map(upload_blob, files))
        ]
    tree = api("git/trees", method="POST", payload={"base_tree": base_tree, "tree": rows})["sha"]
    commit = api("git/commits", method="POST",
                 payload={"message": message, "tree": tree, "parents": [parent]})
    return commit["sha"], rows


def matches(commit, rows):
    tree = api(f"git/commits/{commit}")["tree"]["sha"]
    result = api(f"git/trees/{tree}?recursive=1")
    if result.get("truncated"):
        raise RuntimeError("GitHub tree response was truncated")
    found = {item["path"]: item["sha"] for item in result["tree"]}
    return all(found.get(row["path"]) == row["sha"] for row in rows)


def initialize(repo, root, runtime):
    if head(BASE) != EXPECTED_BASE:
        raise RuntimeError("PR #46 base has changed")
    probe = min((root / "results/mmlu-pro-e2-online/runs").glob("*/predictions.jsonl.gz"),
                key=lambda path: path.stat().st_size)
    uploaded = upload_blob(probe)
    print(f"GitHub verified upload for {probe.stat().st_size:,} bytes ({uploaded[:10]}).", flush=True)
    files = sorted(file for file in (root / "e2-transfer").iterdir() if file.is_file())
    sha, rows = commit_files(repo, files, EXPECTED_BASE, "Queue and verify E2 online transfer experiments")
    current = head(BRANCH)
    if current is None:
        api("git/refs", method="POST", payload={"ref": f"refs/heads/{BRANCH}", "sha": sha})
        current = sha
    if not matches(current, rows):
        raise RuntimeError("Remote experiment branch has unexpected code")
    (runtime / "code-revision").write_text(current + "\n")
    print(f"Published runner {current}", flush=True)


def open_pr(body):
    query = f"pulls?state=all&head=zhaqian21:{BRANCH}&base={BASE}"
    prior = api(query)
    if not prior:
        try:
            pr = api("pulls", method="POST", payload={
                "title": "补齐 MMLU-Pro E2 跨模型与评分器在线对照",
                "body": body, "head": BRANCH, "base": BASE, "draft": False,
            })
        except RuntimeError:
            prior = api(query)
            if not prior:
                raise
            pr = prior[0]
    else:
        pr = prior[0]
    if pr["state"] != "open":
        raise RuntimeError(f"Existing pull request is {pr['state']}")
    return pr


def publish(repo, root, runtime):
    if (runtime / "run.exit").read_text().strip() != "0":
        raise RuntimeError("The run must finish before publication")
    verify(root)
    relative = "experiments/general-ttc/results/mmlu-pro-e2-transfer"
    path = repo / relative
    files = sorted(file for file in path.rglob("*") if file.is_file())
    code = (runtime / "code-revision").read_text().strip()
    if head(BASE) != EXPECTED_BASE:
        raise RuntimeError("PR #46 base has changed")
    current = head(BRANCH)
    if current == code:
        new, rows = commit_files(repo, files, code, "Archive 12 MMLU-Pro E2 online transfer conditions")
        api(f"git/refs/heads/{BRANCH}", method="PATCH", payload={"sha": new, "force": False})
        current = new
    else:
        if api(f"git/commits/{current}")["parents"][0]["sha"] != code:
            raise RuntimeError("The experiment branch moved unexpectedly")
        rows = []
        for file in files:
            data = file.read_bytes()
            object_id = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
            rows.append({"path": file.relative_to(repo).as_posix(), "sha": object_id})
    if not matches(current, rows):
        raise RuntimeError("GitHub tree does not match the verified records")
    readme = (path / "README.md").read_text()
    body = readme.split("\n## 归档与复核", 1)[0]
    body += (
        "\n\n新增 12 轮、16,800 条逐题记录。汇总加入 PR #46 的 Qwen + dPRM 四组，"
        "并按每个组合自己的在线基线做配对。服务器已从逐题记录重算所有指标。\n\n"
        f"[实验汇总、配置和逐题归档](https://github.com/{REPO}/tree/{current}/{relative})"
    )
    pr = open_pr(body)
    if pr["head"]["sha"] != current:
        raise RuntimeError("PR points to an unexpected commit")
    (runtime / "pr-url").write_text(pr["html_url"] + "\n")
    print(pr["html_url"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("initialize", "publish"))
    parser.add_argument("--runtime", required=True, type=Path)
    args = parser.parse_args()
    framework = Path(__file__).resolve().parent.parent
    actions = {"initialize": initialize, "publish": publish}
    actions[args.command](framework.parent.parent, framework, args.runtime.resolve())
