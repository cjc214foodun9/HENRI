#!/usr/bin/env python
"""Show the exact verbatim overlapping line(s) between HumanEval first-30 and itertools.rst."""
import gzip
import hashlib
import io
import json
import pathlib
import re
import urllib.request

DECOMP_SHA = "1d49078ba3e2b196b9344535bef34a43021f038fad9561d6ee7c53450609a6a2"
CANONICAL = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"


def norm_lines(text: str) -> set[str]:
    out = set()
    for line in text.splitlines():
        line = line.strip()
        if len(line) >= 8 and not line.startswith("#"):
            out.add(re.sub(r"\s+", " ", line))
    return out


def main() -> None:
    raw = gzip.decompress(urllib.request.urlopen(CANONICAL, timeout=90).read())
    assert hashlib.sha256(raw).hexdigest() == DECOMP_SHA
    items = [json.loads(line) for line in raw.decode().splitlines()][:30]

    item_lines = {}
    for it in items:
        item_lines[it["task_id"]] = norm_lines(it["prompt"]) | norm_lines(
            "\n".join(it["test"])
        )

    corpus_path = pathlib.Path("data/backbone_retrieval_corpus/itertools.rst")
    corpus_text = corpus_path.read_text(encoding="utf-8")
    corpus_lines = norm_lines(corpus_text)

    all_overlap = set()
    print("== exact overlapping lines (task vs corpus) ==")
    for tid, ls in item_lines.items():
        overlap = ls & corpus_lines
        for ol in overlap:
            all_overlap.add(ol)
            print(f"TASK {tid}: {ol!r}")

    if not all_overlap:
        print("(none)")
        return

    print()
    print("== context in itertools.rst around each overlap ==")
    raw_lines = corpus_text.splitlines()
    for ol in sorted(all_overlap):
        for i, ln in enumerate(raw_lines):
            if ln.strip() == ol:
                ctx = raw_lines[max(0, i - 2):i + 3]
                print(f"-- overlap: {ol!r} --")
                for c in ctx:
                    print("   ", c[:140])
                break


if __name__ == "__main__":
    main()
