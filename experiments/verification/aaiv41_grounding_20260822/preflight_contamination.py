#!/usr/bin/env python
"""Contamination-only preflight for CLASS51 P3(a) pilot (no model load).

Verifies that the shipped retrieval module + corpus + benchmark slice pass
the amended v3.1 gate on the remote host. Prints hits and PASS/BLOCKED.
"""
import gzip
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import henri_backbone_retrieval as h  # noqa: E402

GZ = "/root/class51_p3/HumanEval.jsonl.gz"
CORPUS = "/root/class51_p3/backbone_retrieval_corpus"
N = 30

items = [
    json.loads(line)
    for line in gzip.open(GZ, "rt", encoding="utf-8").read().splitlines()
][:N]
for it in items:
    h.add_contamination_shingles(it["prompt"])
    h.add_contamination_shingles(it["test"])
retr = h.BackboneRetrieval(CORPUS, enabled=True)
hits = retr.scan_contamination()
print(f"items={len(items)} detector=v3.1 hits={hits}")
print("PASS" if not hits else "BLOCKED")
sys.exit(0 if not hits else 1)
