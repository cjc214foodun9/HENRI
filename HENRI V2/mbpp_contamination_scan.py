"""Deterministic MBPP contamination and exposure scan.

The scan searches prior HENRI text artifacts and code manifests for exact task
identifiers and normalized task-text prefixes. It excludes the immutable source
and the current pilot's own provenance artifacts. A clean scan is conditional:
public benchmark contamination cannot be disproved by repository inspection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

TEXT_SUFFIXES = {".json", ".jsonl", ".md", ".py", ".txt", ".yaml", ".yml", ".csv", ".log"}
TOKEN_RE = re.compile(r"[a-z0-9_]+")


def normalize(text: str) -> str:
    return " ".join(TOKEN_RE.findall(text.lower()))


def load_test_items(source_path: Path, minimum: int = 11, maximum: int = 510) -> list[dict[str, Any]]:
    items = []
    for line in source_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            task_id = int(item["task_id"])
            if minimum <= task_id <= maximum:
                items.append(item)
    return sorted(items, key=lambda item: int(item["task_id"]))


def iter_text_files(root: Path, excluded: set[Path]) -> list[Path]:
    if root.is_file():
        return [] if root.resolve() in excluded else [root]
    files = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        resolved = path.resolve()
        if resolved in excluded or any(parent in excluded for parent in resolved.parents):
            continue
        try:
            if path.stat().st_size > 25_000_000:
                continue
        except OSError:
            continue
        files.append(path)
    return sorted(files)


def scan(
    source_path: Path,
    scan_roots: list[Path],
    excluded_paths: list[Path],
) -> dict[str, Any]:
    items = load_test_items(source_path)
    excluded = {path.resolve() for path in excluded_paths}
    files = []
    for root in scan_roots:
        files.extend(iter_text_files(root, excluded))
    files = sorted({path.resolve() for path in files})

    matches = []
    for path in files:
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            matches.append({"path": str(path), "kind": "READ_ERROR", "detail": str(exc)})
            continue
        text = normalize(raw)
        for item in items:
            task_id = str(int(item["task_id"]))
            task_marker = f"mbpp {task_id}"
            prompt = normalize(str(item.get("text", "")))
            prompt_prefix = " ".join(prompt.split()[:12])
            if task_marker in text or (len(prompt_prefix.split()) >= 6 and prompt_prefix in text):
                matches.append({
                    "path": str(path),
                    "task_id": int(item["task_id"]),
                    "kind": "TASK_EXPOSURE_MATCH",
                    "evidence_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                })

    return {
        "schema_id": "henri.mbpp-contamination-scan.v1",
        "source_artifact": str(source_path),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "item_count": len(items),
        "scan_roots": [str(path) for path in scan_roots],
        "scanned_file_count": len(files),
        "excluded_path_count": len(excluded),
        "matches": matches,
        "status": "PASS_NO_REPOSITORY_MATCHES" if not matches else "BLOCKED_TASK_EXPOSURE",
        "limitations": "A clean repository scan does not prove absence from pretraining or external checkpoint lineage.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--scan-root", type=Path, action="append", required=True)
    parser.add_argument("--exclude", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = scan(args.source, args.scan_root, args.exclude)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "matches": len(result["matches"]), "scanned_file_count": result["scanned_file_count"]}, sort_keys=True))
    return 0 if not result["matches"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
