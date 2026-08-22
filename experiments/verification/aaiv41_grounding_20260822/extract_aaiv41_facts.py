#!/usr/bin/env python
"""Deterministic extraction of AAII v4.1 official methodology facts from pinned HTML bytes.

Inputs: methodology.html (386,942 B, sha 577a1de7...), index.html (1,778,846 B, sha 2e18dd8e...),
        blog_intelligence.html (17,067 B, sha 6c29aa0e...)
Output: aaiv41_extracted_facts.json  (benchmark mentions, index formula text, families, weights)
"""
import json, re, hashlib, html, pathlib

D = pathlib.Path(__file__).parent

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]

def strip_html(s):
    s = re.sub(r"<script.*?</script>", " ", s, flags=re.S)
    s = re.sub(r"<style.*?</style>", " ", s, flags=re.S)
    s = re.sub(r"<[^>]+>", " ", s)
    s = html.unescape(s)
    return re.sub(r"\s+", " ", s)

facts = {"inputs": {}, "benchmarks": {}, "index_formula": {}, "notable_excerpts": []}

for name in ("methodology.html", "index.html", "blog_intelligence.html"):
    p = D / name
    facts["inputs"][name] = {"bytes": p.stat().st_size, "sha256_prefix": sha(p)}

text = {}
for name in ("methodology.html", "index.html", "blog_intelligence.html"):
    text[name] = strip_html((D / name).read_text(encoding="utf-8", errors="ignore"))

# Benchmark-name frequency across all pages (official AA v4.1 family candidates)
candidates = [
    "MMLU-Pro", "MMLU", "GPQA", "GPQA Diamond", "HumanEval", "ARC-AGI", "AIME 2025",
    "MATH", "MATH-500", "LiveCodeBench", "Terminal-Bench", "MMMU-Pro", "MMMU",
    "IFBench", "IFEval", "SWE-bench", "Arena", "Omniscience", "LMSYS", "Chatbot Arena",
    "Intelligence Index", "AI-II", "index", "Reasoning", "Coding", "Language",
    "multimodal", "vision", "agent",
]
for name, t in text.items():
    counts = {}
    for c in candidates:
        counts[c] = len(re.findall(re.escape(c), t, flags=re.I))
    facts["benchmarks"][name] = {k: v for k, v in sorted(counts.items(), key=lambda kv: -kv[1]) if v > 0}

# Index formula: find the paragraph(s) around 'Intelligence Index' in methodology
m = re.search(r"(.{0,400}Intelligence Index.{0,900})", text["methodology.html"], flags=re.I)
if m:
    facts["index_formula"]["methodology_context"] = m.group(1)
b = re.search(r"(.{0,300}Intelligence Index.{0,700})", text["blog_intelligence.html"], flags=re.I)
if b:
    facts["index_formula"]["blog_context"] = b.group(1)

# Index calculation / normalization keywords
for kw in ("normaliz", "weighted", "average", "geometric", "percentile", "log"):
    hits = [mm.start() for mm in re.finditer(kw, text["methodology.html"], flags=re.I)][:3]
    if hits:
        facts["notable_excerpts"].append({"kw": kw, "excerpts": [text["methodology.html"][max(0, h-160):h+220] for h in hits]})

D.joinpath("aaiv41_extracted_facts.json").write_text(
    json.dumps(facts, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps({"benchmark_counts": facts["benchmarks"], "formula_keys": list(facts["index_formula"].keys())}, indent=2))
