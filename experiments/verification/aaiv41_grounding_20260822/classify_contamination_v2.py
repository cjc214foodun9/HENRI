#!/usr/bin/env python
"""CLASSIFY the recalibrated-gate hit on re.rst (v2 detector semantics).

Reproduces the pilot exactly: register contamination from first-30 HumanEval
prompts+tests via henri_backbone_retrieval.add_contamination_shingles, then
scan. Reports WHICH signal fired (code line vs ident4), WHICH task produced
it, and the exact overlapping text — for a real-leakage vs false-positive
verdict.
"""
import gzip
import hashlib
import io
import json
import pathlib
import re
import sys
import urllib.request

REPO = pathlib.Path(__file__).resolve().parents[3]
CORPUS = REPO / "data" / "backbone_retrieval_corpus"
sys.path.insert(0, str(REPO / "HENRI V2"))

from henri_backbone_retrieval import (  # noqa: E402
    BackboneRetrieval,
    _contamination_lines,
    _ident_shingles,
    add_contamination_shingles,
)

CANONICAL = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
DECOMP_SHA = "1d49078ba3e2b196b9344535bef34a43021f038fad9561d6ee7c53450609a6a2"


def main() -> None:
    raw = gzip.decompress(urllib.request.urlopen(CANONICAL, timeout=90).read())
    assert hashlib.sha256(raw).hexdigest() == DECOMP_SHA
    items = [json.loads(line) for line in raw.decode().splitlines()][:30]

    # reproduce pilot registration, tracking per-task contributions
    task_lines: dict[str, set[str]] = {}
    task_ident4: dict[str, set[str]] = {}
    for it in items:
        text = it["prompt"] + "\n" + "\n".join(it["test"])
        add_contamination_shingles(text)
        task_lines[it["task_id"]] = _contamination_lines(text)
        task_ident4[it["task_id"]] = _ident_shingles(text)

    retr = BackboneRetrieval(CORPUS, enabled=True)
    hits = retr.scan_contamination()
    print("scan hits:", hits)

    if "re.rst" not in hits:
        print("no re.rst hit — gate CLEAN for first-30 HumanEval (expected after v3)")
        return

    re_text = (CORPUS / "re.rst").read_text(encoding="utf-8")
    re_lines = _contamination_lines(re_text)
    re_ident4 = _ident_shingles(re_text)

    print("\n== which signal fired for re.rst ==")
    line_fire = False
    ident_fire = False
    for tid, ls in task_lines.items():
        ov = ls & re_lines
        if ov:
            line_fire = True
            print(f"LINE overlap from {tid}:")
            for o in sorted(ov):
                print(f"   {o!r}")
    for tid, s4 in task_ident4.items():
        ov = s4 & re_ident4
        if ov:
            ident_fire = True
            print(f"IDENT4 overlap from {tid}:")
            for o in sorted(ov):
                print(f"   {o!r}")
    print(f"\nline_fire={line_fire} ident_fire={ident_fire}")


if __name__ == "__main__":
    main()
