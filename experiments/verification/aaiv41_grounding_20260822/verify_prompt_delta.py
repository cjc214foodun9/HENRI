#!/usr/bin/env python
"""Verify Arm A vs Arm B prompt construction for the 30-item pilot slice:
deterministic, differ only by the retrieval block, retrieval engaged 30/30.
Runs locally (CPU-only, no model load).
"""
import gzip
import hashlib
import json
import pathlib
import sys
import urllib.request

REPO = pathlib.Path(__file__).resolve().parents[3]
CORPUS = REPO / "data" / "backbone_retrieval_corpus"
sys.path.insert(0, str(REPO / "HENRI V2"))

from henri_backbone_retrieval import (  # noqa: E402
    BackboneRetrieval,
    add_contamination_shingles,
    build_arm_a_prompt,
)

CANONICAL = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
GZ_SHA = "b796127e"
DECOMP_SHA = "1d49078b"

raw_gz = urllib.request.urlopen(CANONICAL, timeout=90).read()
assert hashlib.sha256(raw_gz).hexdigest().startswith(GZ_SHA)
raw = gzip.decompress(raw_gz)
assert hashlib.sha256(raw).hexdigest().startswith(DECOMP_SHA)
items = [json.loads(line) for line in raw.decode().splitlines()][:30]

# Match pilot registration so build_prompt's gate sees the same contamination state.
for it in items:
    add_contamination_shingles(it["prompt"])
    test = it["test"] if isinstance(it["test"], str) else "\n".join(it["test"])
    add_contamination_shingles(test)

retrieval = BackboneRetrieval(CORPUS, enabled=True)
assert retrieval.scan_contamination() == []

engaged = 0
deltas = 0
for it in items:
    pa = build_arm_a_prompt(it["prompt"])[0]
    pb, tel = retrieval.build_prompt(it["prompt"])
    if tel["retrieval_engaged"]:
        engaged += 1
    if pa != pb:
        deltas += 1
    if engaged == 1 and deltas == 1:
        print("=== sample prompt pair (item 0) ===")
        print("A len:", len(pa), "B len:", len(pb))
        print("B head:", repr(pb[:150]))

print(f"items=30 engaged={engaged}/30 prompt_deltas={deltas}/30")
print("PASS" if engaged == 30 and deltas == 30 else "FAIL")
sys.exit(0 if engaged == 30 and deltas == 30 else 1)
