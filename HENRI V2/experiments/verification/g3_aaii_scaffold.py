"""G3 scaffold v2 — source-aware sealed staging for AAII constituents.

Bounded: <=2 data items per constituent, <=600 s wall, diagnostic-only.
Sources (pinned at immutable SHAs, verified 2026-09-05):
  terminal-bench-2.1  -> GitHub harbor-framework/terminal-bench-2-1 (Apache-2.0);
                         HF dataset carries ONLY metadata (LICENSE/README/eval.yaml/
                         registry.json, 25 KB); task content = GitHub tasks/ dir.
  scicode             -> HF SciCode1/SciCode (Apache-2.0): problems_test.jsonl;
                         LICENSE from GitHub scicode-bench/SciCode (pinned).
  hle                 -> cais/hle is gated(auto) -> STAGED_BLOCKED_GATED, no bytes.

Statuses: STAGED_OK / STAGED_BLOCKED_GATED / STAGED_BLOCKED_INFRA.
No score claim. No LLM judge. No placeholder output. No mock loop.
"""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
import urllib.error

GITHUB_COMMITS = "https://api.github.com/repos/{repo}/commits/{branch}"
GITHUB_CONTENTS = "https://api.github.com/repos/{repo}/contents/{path}?ref={sha}"
GITHUB_RAW = "https://raw.githubusercontent.com/{repo}/{sha}/{path}"
HF_META = "https://huggingface.co/api/datasets/{ds}"
HF_RESOLVE = "https://huggingface.co/datasets/{ds}/resolve/{rev}/{path}"
UA = {"User-Agent": "henri-g3-scaffold/0.2"}

CONSTITUENTS = {
    "terminal-bench-2.1": {
        "origin": "github",
        "repo": "harbor-framework/terminal-bench-2-1",
        "branch": "main",
        "license": "Apache-2.0",
        "stage_files": ["LICENSE", "README.md"],
        "task_dir": "tasks",
        "task_files": ["task.toml", "instruction.md"],
        "max_tasks": 2,
        "checker": "published terminal tasks; containerized shell harness absent on Vast -> MODALITY_HARNESS_BLOCKED at exec",
    },
    "scicode": {
        "origin": "hf",
        "hf": "SciCode1/SciCode",
        "license": "Apache-2.0",
        "license_repo": "scicode-bench/SciCode",
        "stage_files": ["problems_test.jsonl"],
        "checker": "published gold solutions + tests; exec via HENRI REPL sandbox (python only)",
    },
    "hle": {
        "origin": "hf",
        "hf": "cais/hle",
        "license": "MIT",
        "gated": True,
        "terms_url": "https://huggingface.co/datasets/cais/hle",
        "checker": "public question subset; private grader components BLOCKED",
    },
}
MAX_TASKS = 2
WALL_BUDGET = 600.0


def sha256b(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def http_json(url: str, timeout: int = 45):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def http_bytes(url: str, timeout: int = 120):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def stage(dst_dir: str, name: str, url: str, data: bytes) -> dict:
    dst = os.path.join(dst_dir, name)
    with open(dst, "wb") as fh:
        fh.write(data)
    return {"name": name, "url": url, "bytes": len(data), "sha256": sha256b(data)}


def github_pin(repo: str, branch: str) -> str:
    return http_json(GITHUB_COMMITS.format(repo=repo, branch=branch))["sha"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/root/g3_data")
    ap.add_argument("--out-dir", default="/tmp/g3_scaffold")
    ap.add_argument("--max-tasks", type=int, default=MAX_TASKS)
    ap.add_argument("--constituent", choices=list(CONSTITUENTS), default=None)
    args = ap.parse_args()
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()
    rows = []

    for name, meta in CONSTITUENTS.items():
        if args.constituent and args.constituent != name:
            continue
        rec = {"constituent": name, "license": meta["license"],
               "checker": meta["checker"], "status": "STAGED_BLOCKED_INFRA"}
        try:
            # HLE gated -> record terms; fetch nothing.
            if meta.get("gated"):
                rec["status"] = "STAGED_BLOCKED_GATED"
                rec["terms_url"] = meta["terms_url"]
                rec["detail"] = "gated(auto); requires terms acceptance; no bytes staged"
                rows.append(rec)
                continue

            staged = []
            if meta["origin"] == "github":
                repo, branch = meta["repo"], meta["branch"]
                sha = github_pin(repo, branch)
                rec["github_revision"] = sha
                for fname in meta["stage_files"]:
                    data = http_bytes(GITHUB_RAW.format(repo=repo, sha=sha, path=fname))
                    staged.append(stage(args.data_dir, f"tb_{fname.replace('/', '__')}",
                                        GITHUB_RAW.format(repo=repo, sha=sha, path=fname), data))
                # Sample up to max_tasks task dirs (deterministic: sorted).
                listing = http_json(GITHUB_CONTENTS.format(repo=repo, path=meta["task_dir"], sha=sha))
                dirs = sorted([e["name"] for e in listing if e.get("type") == "dir"])[: args.max_tasks]
                rec["task_dir_entries"] = len(listing)
                rec["task_dir_sample"] = dirs
                rec["task_files_staged"] = list(meta["task_files"])
                for d in dirs:
                    for fname in meta["task_files"]:
                        fpath = f"{meta['task_dir']}/{d}/{fname}"
                        data = http_bytes(GITHUB_RAW.format(repo=repo, sha=sha, path=fpath))
                        staged.append(stage(args.data_dir, f"tb_{d}_{fname}",
                                            GITHUB_RAW.format(repo=repo, sha=sha, path=fpath), data))
                    rec["task_files_staged"] = list(meta["task_files"])
            else:
                hf = meta["hf"]
                rev = http_json(HF_META.format(ds=hf))["sha"]
                rec["hf_revision"] = rev
                for fname in meta["stage_files"]:
                    url = HF_RESOLVE.format(ds=hf, rev=rev, path=fname)
                    data = http_bytes(url)
                    staged.append(stage(args.data_dir, f"sc_{fname.replace('/', '__')}", url, data))
                if meta.get("license_repo"):
                    lr = meta["license_repo"]
                    lsha = github_pin(lr, "main")
                    rec["license_repo_revision"] = lsha
                    data = http_bytes(GITHUB_RAW.format(repo=lr, sha=lsha, path="LICENSE"))
                    staged.append(stage(args.data_dir, "sc_LICENSE", 
                                        GITHUB_RAW.format(repo=lr, sha=lsha, path="LICENSE"), data))

            rec["status"] = "STAGED_OK"
            rec["staged_files"] = staged
        except (urllib.error.URLError, urllib.error.HTTPError, KeyError, ValueError, json.JSONDecodeError) as exc:
            rec["detail"] = f"{type(exc).__name__}: {exc}"
        rows.append(rec)
        if time.time() - t0 > WALL_BUDGET - 60:
            rows.append({"constituent": "(budget)", "status": "STAGED_BLOCKED_INFRA",
                         "detail": "600s wall budget reached; terminated"})
            break

    manifest = {
        "schema_id": "henri.g3-scaffold.v1",
        "prereg": "experiments/verification/g3_aaii_receipt_prereg.md",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_seconds": round(time.time() - t0, 2),
        "rows": rows,
    }
    out = os.path.join(args.out_dir, "g3_scaffold_receipt.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    print(json.dumps(manifest, indent=2, default=str))
    print(f"RECEIPT={out}")
    print(f"RECEIPT_SHA256={sha256b(open(out, 'rb').read())}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"schema_id": "henri.g3-scaffold.v1", "status": "CRASH",
                          "error": f"{type(exc).__name__}: {exc}"}, indent=2))
        sys.exit(2)
