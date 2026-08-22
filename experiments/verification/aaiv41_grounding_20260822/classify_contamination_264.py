#!/usr/bin/env python
"""Classify v3.1 contamination hits on the full 264-item surface.

For each hit, show the task-side source (task_id, dataset, kind) and the
corpus-side doc line(s) that match, so a human can judge vacuous-idiom
(e.g. 'import random' doctest lines) vs semantic leakage (MBPP tests that
duplicate stdlib doc examples).
"""
import json
import pathlib
import sys

sys.path.insert(0, "/root/class51_p3")
import henri_backbone_retrieval as h  # noqa: E402

raw_gz = pathlib.Path("/root/class51_p3/HumanEval.jsonl.gz").read_bytes()
raw = __import__("gzip").decompress(raw_gz)
he = [json.loads(l) for l in raw.decode().splitlines()]
mbpp = json.load(open("/root/class51_p3/sanitized-mbpp.json"))

items = []
for it in he:
    test = it["test"] if isinstance(it["test"], str) else "\n".join(it["test"])
    items.append({"task_id": it["task_id"], "dataset": "humaneval",
                  "prompt": it["prompt"], "test": test,
                  "imports": it.get("imports", [])})
for it in mbpp[:100]:
    items.append({"task_id": "mbpp-" + str(it["task_id"]), "dataset": "mbpp",
                  "prompt": it["prompt"], "test_list": it.get("test_list", []),
                  "test_imports": it.get("test_imports", [])})

line_src = {}
shingle_src = {}
for it in items:
    texts = [("prompt", it["prompt"])]
    if it["dataset"] == "humaneval":
        texts += [("test", it["test"]), ("imports", "\n".join(it["imports"]))]
    else:
        texts += [("test_list", "\n".join(it["test_list"])),
                  ("test_imports", "\n".join(it["test_imports"]))]
    for kind, text in texts:
        for ln in h._contamination_lines(text):
            line_src.setdefault(ln, (it["task_id"], it["dataset"], kind))
        for sh in h._ident_shingles(text):
            shingle_src.setdefault(sh, (it["task_id"], it["dataset"], kind))
        h.add_contamination_shingles(text)

corpus = pathlib.Path("/root/henri-matrix264-wt/data/backbone_retrieval_corpus")
manifest = json.loads((corpus / "manifest.json").read_text())
total_l = total_s = 0
for entry in manifest["files"]:
    text = (corpus / entry["file"]).read_text()
    lines = text.splitlines()
    ol = sorted(h._contamination_lines(text) & h._CONTAMINATION_LINES)
    os_ = sorted(h._ident_shingles(text) & h._CONTAMINATION_IDENT4)
    if ol or os_:
        total_l += len(ol)
        total_s += len(os_)
        print("### %s: line_hits=%d shingle_hits=%d" % (entry["file"], len(ol), len(os_)))
        for ln in ol[:10]:
            print("  LINE:", ln[:150], "| src:", line_src.get(ln))
        for sh in os_[:10]:
            print("  SHINGLE:", sh[:150], "| src:", shingle_src.get(sh))
            toks = sh.split()
            for dl in lines:
                if all(t in h._tokenize(dl.lower()) for t in toks):
                    print("    DOC:", dl.strip()[:150])
                    break
print("TOTAL line_hits", total_l, "shingle_hits", total_s)
