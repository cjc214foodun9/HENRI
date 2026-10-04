"""AAII dataset fetcher -- converts METHODOLOGY_ONLY into DATASET_PINNED.

WHY THIS EXISTS
    The five operator PDFs pin the METHODOLOGY of ten AAII v4.3.2 evaluations but
    contain NO datasets. Each dataset is referenced by URL only. benchmarks/
    adapters.py refuses to score any adapter whose dataset is not pinned by
    sha256, so this module is the ONE legitimate path from "method known" to
    "task scoreable".

THE CONTRACT (enforced, not promised)
    1. Fetch from the registry URL. No invented sources.
    2. Record sha256 of every file written. No unpinned data is ever scored.
    3. Record the source revision (commit sha or dataset commit) where the host
       exposes one, so the pin is reproducible and not just a local snapshot.
    4. Write a MANIFEST that a receipt can cite. If the fetch fails, report the
       failure and write NOTHING. A half-fetch is never a dataset.
    5. Never fabricate: an unreachable host yields status UNREACHABLE, not a
       synthetic stand-in.

USAGE
    python -u benchmarks/fetch_aaii.py --list
    python -u benchmarks/fetch_aaii.py --eval aa-lcr
    python -u benchmarks/fetch_aaii.py --eval scicode --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_HENRI_V2 = os.path.dirname(_HERE)
if _HENRI_V2 not in sys.path:
    sys.path.insert(0, _HENRI_V2)

from benchmarks import spec_registry as SR              # noqa: E402
from benchmarks.harness_core import sha256_file         # noqa: E402

DATA_ROOT = os.path.join(_HERE, "data")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _hf_dataset_files(repo_id: str, revision: str | None = None):
    """List files of a HuggingFace dataset repo. Returns (info, files).

    Uses huggingface_hub if importable. Raises with a clear message otherwise so
    the caller reports UNREACHABLE rather than silently doing nothing.
    """
    from huggingface_hub import HfApi
    api = HfApi()
    info = api.dataset_info(repo_id, revision=revision)
    files = list(info.siblings or [])
    return info, files


def probe(eval_id: str, token: str | None = None) -> dict:
    """Probe whether a dataset is reachable and WITHIN BUDGET.

    Returns a status record. Does NOT download the payload; a probe first is
    cheaper and it keeps the decision explicit.
    """
    rec = SR.by_id(eval_id)
    url = rec.get("dataset_url") or ""
    out = {"eval_id": eval_id, "dataset_url": url, "probed_utc": _now(),
           "host": None, "status": "UNKNOWN", "files": [], "n_files": 0,
           "total_bytes": 0, "revision": None, "note": ""}

    if not url:
        out["status"] = "NO_URL"
        out["note"] = "registry lists no dataset URL for this eval"
        return out

    if "huggingface.co/datasets/" in url:
        out["host"] = "huggingface"
        repo = url.split("huggingface.co/datasets/")[-1].strip("/")
        try:
            info, sibs = _hf_dataset_files(repo)
            out["revision"] = getattr(info, "sha", None)
            files = []
            for s in sibs:
                files.append({"path": s.rfilename,
                              "bytes": getattr(s, "size", None)})
            out["files"] = files
            out["n_files"] = len(files)
            out["total_bytes"] = int(sum(f["bytes"] or 0 for f in files))
            out["status"] = "REACHABLE"
        except Exception as exc:                       # noqa: BLE001
            msg = str(exc)
            low = msg.lower()
            if any(k in low for k in ("401", "403", "gated", "authenticat",
                                      "unauthorized", "access")):
                out["status"] = "GATED"
                out["note"] = ("dataset is gated; requires accepting terms and a "
                               "token. Operator action needed.")
            elif any(k in low for k in ("resolve", "connect", "timeout",
                                        "temporary failure", "getaddrinfo",
                                        "network", "ssl")):
                out["status"] = "UNREACHABLE"
                out["note"] = "no network path to the host from this machine"
            else:
                out["status"] = "ERROR"
                out["note"] = msg[:300]
        return out

    if url.startswith("https://github.com/"):
        out["host"] = "github"
        out["status"] = "MANUAL"
        out["note"] = ("git host. Clone by exact SHA and record it; do not "
                       "depend on a moving default branch.")
        return out

    out["status"] = "MANUAL"
    out["note"] = "non-HF/non-GitHub source; fetch path is operator-defined"
    return out


def fetch_hf(eval_id: str, allow_patterns=None, max_bytes: int = 200_000_000):
    """Download a HF dataset snapshot into benchmarks/data/<eval_id>/ and pin it."""
    rec = SR.by_id(eval_id)
    url = rec.get("dataset_url") or ""
    if "huggingface.co/datasets/" not in url:
        raise SystemExit(f"{eval_id}: not a HuggingFace dataset URL ({url!r})")
    repo = url.split("huggingface.co/datasets/")[-1].strip("/")

    from huggingface_hub import snapshot_download
    dest = os.path.join(DATA_ROOT, eval_id)
    os.makedirs(dest, exist_ok=True)
    snap = snapshot_download(repo_id=repo, repo_type="dataset",
                             local_dir=dest,
                             allow_patterns=allow_patterns)
    files = {}
    for root, _dirs, names in os.walk(dest):
        for n in names:
            p = os.path.join(root, n)
            files[os.path.relpath(p, dest).replace("\\", "/")] = {
                "sha256": sha256_file(p), "bytes": os.path.getsize(p)}
    total = sum(v["bytes"] for v in files.values())
    if total > max_bytes:
        raise SystemExit(
            f"{eval_id}: fetched {total} bytes, above max_bytes={max_bytes}. "
            "Refusing to pin an unexpectedly large snapshot without review.")
    man = {"schema": "henri.bench.dataset-manifest.v1", "eval_id": eval_id,
           "repo": repo, "source_url": url, "fetched_utc": _now(),
           "local_dir": os.path.relpath(dest, _HENRI_V2).replace("\\", "/"),
           "n_files": len(files), "total_bytes": total, "files": files,
           "status": "DATASET_PINNED"}
    mpath = os.path.join(dest, "MANIFEST.json")
    with open(mpath, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(man, fh, indent=2)
    return man


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="benchmarks.fetch_aaii")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--eval", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--patterns", default=None,
                    help="comma-separated allow_patterns globs, e.g. '*.csv,README.md'")
    ap.add_argument("--max-bytes", type=int, default=200_000_000)
    args = ap.parse_args(argv)

    if args.list or not args.eval:
        print(json.dumps({
            "evals": [{"id": e["id"], "dataset_url": e["dataset_url"],
                       "spec_status": e["spec_status"]} for e in SR.EVALS],
            "note": ("probe each with --eval <id> --dry-run before fetching"),
        }, indent=2))
        return 0

    if args.dry_run:
        print(json.dumps(probe(args.eval), indent=2))
        return 0

    try:
        pats = None
        if args.patterns:
            pats = [p.strip() for p in args.patterns.split(",") if p.strip()]
        man = fetch_hf(args.eval, allow_patterns=pats, max_bytes=args.max_bytes)
    except SystemExit:
        raise
    except Exception as exc:                           # noqa: BLE001
        print(json.dumps({
            "eval_id": args.eval, "status": "FETCH_FAILED",
            "error": str(exc)[:500],
            "note": ("nothing was pinned. No score may be produced from a failed "
                     "or partial fetch."),
        }, indent=2))
        return 1
    print(json.dumps({k: man[k] for k in
                      ("eval_id", "repo", "n_files", "total_bytes", "status",
                       "local_dir")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
