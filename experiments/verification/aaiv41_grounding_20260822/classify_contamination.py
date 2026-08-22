#!/usr/bin/env python
"""CLASSIFY the contamination-gate hits: genuine leakage vs benign idiom.

Compares three detectors over the first 30 HumanEval items vs the 13-file
CPython docs corpus:
  A) current gate: unfiltered 5-gram shingles (prompt + test)
  B) proposed: identifier-only 4-gram shingles (stopwords stripped)
  C) verbatim normalized-line overlap (strongest signal)

Output: per-file hit counts and sample overlapping shingles.
"""
import gzip
import hashlib
import io
import json
import pathlib
import re
import urllib.request

CANONICAL = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
DECOMP_SHA = "1d49078ba3e2b196b9344535bef34a43021f038fad9561d6ee7c53450609a6a2"
CORPUS_DIR = pathlib.Path("data/backbone_retrieval_corpus")

STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "for", "with", "that", "this",
    "and", "or", "not", "be", "by", "at", "on", "from", "as", "is", "are",
    "it", "its", "their", "his", "her", "was", "were", "will", "would",
    "should", "can", "may", "must", "has", "have", "had", "do", "does",
    "did", "if", "then", "else", "than", "when", "while", "which", "who",
    "whom", "what", "where", "how", "all", "any", "both", "each", "few",
    "more", "most", "other", "some", "such", "no", "nor", "too", "very",
    "only", "into", "over", "under", "out", "up", "down", "off", "about",
    "after", "before", "between", "through", "during", "above", "below",
}


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def id_tokens(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in STOPWORDS and not t.isdigit()]


def shingles(tokens: list[str], n: int) -> set[str]:
    return {" ".join(tokens[i:i + n]) for i in range(max(0, len(tokens) - n + 1))}


def norm_lines(text: str) -> set[str]:
    out = set()
    for line in text.splitlines():
        line = line.strip()
        if len(line) >= 8 and not line.startswith("#"):
            out.add(re.sub(r"\s+", " ", line))
    return out


def main() -> None:
    raw_gz = urllib.request.urlopen(CANONICAL, timeout=90).read()
    raw = gzip.decompress(raw_gz)
    assert hashlib.sha256(raw).hexdigest() == DECOMP_SHA
    items = [json.loads(line) for line in raw.decode("utf-8").splitlines()][:30]

    corpus_files = sorted(CORPUS_DIR.glob("*.rst"))
    assert len(corpus_files) == 13

    # current gate reproduction
    gate_hits = {}
    # proposed identifier gate
    ident_hits = {}
    # verbatim line overlap
    line_hits = {}
    examples_gate = {}
    examples_ident = {}

    corpus_ident = {p.name: shingles(id_tokens(p.read_text(encoding="utf-8")), 4)
                    for p in corpus_files}
    corpus_lines = {p.name: norm_lines(p.read_text(encoding="utf-8"))
                    for p in corpus_files}

    for item in items:
        task_text = item["prompt"] + "\n" + "\n".join(item["test"])
        gate_s = shingles(tokenize(task_text), 5)
        ident_s = shingles(id_tokens(task_text), 4)
        lines = norm_lines(item["prompt"]) | norm_lines("\n".join(item["test"]))
        for p in corpus_files:
            name = p.name
            cs_full = shingles(tokenize(p.read_text(encoding="utf-8")), 5)
            ov_gate = gate_s & cs_full
            if ov_gate:
                gate_hits[name] = gate_hits.get(name, 0) + 1
                examples_gate.setdefault(name, []).append(
                    (item["task_id"], sorted(ov_gate)[:3]))
            ov_ident = ident_s & corpus_ident[name]
            if ov_ident:
                ident_hits[name] = ident_hits.get(name, 0) + 1
                examples_ident.setdefault(name, []).append(
                    (item["task_id"], sorted(ov_ident)[:3]))
            ov_lines = lines & corpus_lines[name]
            if ov_lines:
                line_hits[name] = line_hits.get(name, 0) + 1

    print("== A) CURRENT gate (unfiltered 5-gram): hits per file ==")
    for p in corpus_files:
        print(f"  {p.name}: {gate_hits.get(p.name, 0)}/30")
    print("\n== A) sample overlapping 5-grams (first 2 per file) ==")
    for name, exs in examples_gate.items():
        print(f"  -- {name} --")
        for tid, sh in exs[:2]:
            print(f"    {tid}: {sh}")

    print("\n== B) identifier-only 4-gram: hits per file ==")
    for p in corpus_files:
        print(f"  {p.name}: {ident_hits.get(p.name, 0)}/30")
    print("\n== B) sample overlapping identifier 4-grams ==")
    for name, exs in examples_ident.items():
        print(f"  -- {name} --")
        for tid, sh in exs[:2]:
            print(f"    {tid}: {sh}")

    print("\n== C) verbatim normalized-line overlap: hits per file ==")
    total_line = 0
    for p in corpus_files:
        c = line_hits.get(p.name, 0)
        total_line += c
        print(f"  {p.name}: {c}/30")
    print(f"\nVERDICT: line-overlap total={total_line} "
          f"(>0 => true contamination; ==0 => idiom false positive)")


if __name__ == "__main__":
    main()
