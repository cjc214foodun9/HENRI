#!/usr/bin/env python
"""Contamination-only preflight for CLASS51 P3(a) pilot (no model load).

Verifies that the shipped retrieval module + corpus + benchmark slice pass
the amended v3.1 gate on the remote host. Prints hits and PASS/BLOCKED.
"""
import gzip
import hashlib
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import henri_backbone_retrieval as h  # noqa: E402

GZ = "/root/class51_p3/HumanEval.jsonl.gz"
CORPUS = "/root/class51_p3/backbone_retrieval_corpus"
CANONICAL_GZ_SHA = "b796127e"
DECOMPRESSED_SHA = "1d49078b"
N = 30

raw_gz = pathlib.Path(GZ).read_bytes()
assert hashlib.sha256(raw_gz).hexdigest().startswith(CANONICAL_GZ_SHA), "gz digest mismatch"
raw = gzip.decompress(raw_gz)
assert hashlib.sha256(raw).hexdigest().startswith(DECOMPRESSED_SHA), "decompressed digest mismatch"
items = [json.loads(line) for line in raw.decode("utf-8").splitlines()][:N]
for it in items:
    h.add_contamination_shingles(it["prompt"])
    test_field = it["test"]
    # Canonical HumanEval 'test' is a string; join only if a list (matches
    # the pilot loader's fixed semantics; char-splitting a str would register
    # garbage shingles).
    test = test_field if isinstance(test_field, str) else "\n".join(test_field)
    h.add_contamination_shingles(test)
retr = h.BackboneRetrieval(CORPUS, enabled=True)
hits = retr.scan_contamination()
print(f"items={len(items)} detector=v3.1 hits={hits}")
print("PASS" if not hits else "BLOCKED")
sys.exit(0 if not hits else 1)
