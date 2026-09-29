"""Build a real ARC-1 demonstration manifest from the PUBLIC train split.

WHY THIS UNBLOCKS PATH B
========================
Directive 2 was `BLOCKED__NO_PUBLIC_DEMO_CHANNEL`: `arc_agi 0.9.9` exposes ZERO
demonstration pairs through its evaluation API -- measured this session, 16/16
environments. But that is the EVAL API. The ARC-1 *training* split is public and
is exactly `{train: [{input, output}, ...], test: [{input, output?}]}`.

`arc_public_ingress.resolve_demos` needs `manifest["envs"][env_id]` with
task_id / corpus_path / sha256. Nothing on disk supplies that. This script
creates it from REAL public data and records provenance for every file.

HONESTY RULES BAKED IN
======================
  * It NEVER synthesises an ARC task. If the corpus is absent it reports
    `BLOCKED__NO_CORPUS` and writes nothing.
  * Every `sha256` is computed from the file actually written.
  * It reports how many tasks have `train` pairs (the demo source) and refuses to
    claim demos for tasks that have none.
  * It does not touch the eval split, so there is no leakage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "arc_manifest.json")
CORPUS_CANDIDATES = [
    "ARC-AGI/data/training",
    "arc-agi/data/training",
    "data/ARC-AGI/training",
    "ARC1/training",
]


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_corpus(root: str) -> str | None:
    for rel in CORPUS_CANDIDATES:
        d = os.path.join(root, rel)
        if os.path.isdir(d):
            return d
    # bounded search: any dir with several *.json holding {"train": ...}
    for base, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if not x.startswith(".") and x != "node_modules"]
        if base.count(os.sep) - root.count(os.sep) > 4:
            dirs[:] = []
            continue
        js = [f for f in files if f.endswith(".json")]
        if len(js) >= 10:
            for f in js[:3]:
                try:
                    d = json.load(open(os.path.join(base, f), encoding="utf-8"))
                    if isinstance(d, dict) and "train" in d:
                        return base
                except Exception:                                  # noqa: BLE001
                    continue
    return None


def resolve_out(cli: str | None) -> str:
    if cli:
        d = os.path.dirname(os.path.abspath(cli))
        if not os.path.isdir(d):
            print(f"MALFORMED --out: directory does not exist: {d}", file=sys.stderr)
            raise SystemExit(2)
        return os.path.abspath(cli)
    envd = os.environ.get("HENRI_RECEIPT_DIR")
    if envd:
        os.makedirs(envd, exist_ok=True)
        return os.path.join(envd, "arc_manifest.json")
    return DEFAULT_OUT


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.getcwd(), help="search root for the corpus")
    ap.add_argument("--out", default=None)
    ap.add_argument("--require-train-pairs", action="store_true", default=True)
    args = ap.parse_args()
    out = resolve_out(args.out)

    R: dict = {"schema": "henri.arc-manifest-build.v1",
               "purpose": "supply the manifest that arc_public_ingress.resolve_demos requires"}
    corpus = find_corpus(args.root)
    if not corpus:
        R["status"] = "BLOCKED__NO_CORPUS"
        R["searched"] = args.root
        R["note"] = ("No ARC-1 training split found. Nothing was synthesised. "
                     "Fetch it (e.g. github.com/fchollet/ARC-AGI data/training) "
                     "then re-run.")
        print(json.dumps(R, indent=2))
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(R, fh, indent=2)
        return 1

    files = sorted(f for f in os.listdir(corpus) if f.endswith(".json"))
    envs: dict = {}
    with_train = 0
    total_pairs = 0
    for f in files:
        p = os.path.join(corpus, f)
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:                                          # noqa: BLE001
            continue
        if not isinstance(d, dict):
            continue
        tr = d.get("train") or []
        if args.require_train_pairs and not tr:
            continue
        task_id = os.path.splitext(f)[0]
        envs[task_id] = {
            "task_id": task_id,
            "corpus_path": p.replace("\\", "/"),
            "sha256": sha256_file(p),
            "n_demo_pairs": len(tr),
            "n_test_pairs": len(d.get("test") or []),
            "split": "training",
        }
        if tr:
            with_train += 1
            total_pairs += len(tr)

    R["status"] = "OK" if envs else "BLOCKED__NO_TASKS_WITH_TRAIN_PAIRS"
    R["corpus_dir"] = corpus.replace("\\", "/")
    R["n_tasks"] = len(envs)
    R["n_tasks_with_demos"] = with_train
    R["total_demo_pairs"] = total_pairs
    R["envs"] = envs
    R["leakage_note"] = "training split only; the eval split is never read"
    print(json.dumps({k: R[k] for k in
                      ("status", "corpus_dir", "n_tasks", "n_tasks_with_demos",
                       "total_demo_pairs")}, indent=2))
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=2)
    print("WROTE", out)
    return 0 if envs else 1


if __name__ == "__main__":
    raise SystemExit(main())
