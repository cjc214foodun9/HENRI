#!/usr/bin/env python3
"""Merge all arxiv_papers*.json into one deduplicated markdown bibliography."""
import json, re, os
from collections import Counter, defaultdict

BASE = r"C:\Users\chan\Desktop\HENRI 7B SWARM"
files = ["arxiv_papers.json", "arxiv_papers_batch2.json", "arxiv_papers_batch3.json"]
seen = set()
all_papers = []

for f in files:
    path = os.path.join(BASE, f)
    if not os.path.exists(path):
        print(f"SKIP: {f} (not found)")
        continue
    with open(path, encoding="utf-8") as fh:
        batch = json.load(fh)
    new = 0
    for p in batch:
        if p["base_id"] in seen:
            continue
        seen.add(p["base_id"])
        all_papers.append(p)
        new += 1
    print(f"  {f}: {new} new (from {len(batch)} raw)")

all_papers.sort(key=lambda p: p["year"], reverse=True)

# Group by domain
by_source = defaultdict(list)
for p in all_papers:
    by_source[p["source"]].append(p)

md = f"# HENRI Research Bibliography — {len(all_papers)} Primary Sources\n\n"
md += f"Generated: 2026-07-21 | ArXiv API | 18+18+8 queries\n\n"
md += "## Summary by Domain\n\n"
for src, papers in sorted(by_source.items(), key=lambda x: -len(x[1])):
    md += f"- **{src}**: {len(papers)} papers\n"
md += f"\n## Complete Paper List\n\n"

for i, p in enumerate(all_papers):
    auth = ", ".join(p["authors"][:3])
    if len(p["authors"]) > 3:
        auth += f" et al. ({len(p['authors'])} authors)"
    cats = ", ".join(p["categories"][:2])
    md += f"{i+1}. **{p['title']}**\n"
    md += f"   - arXiv: [{p['id']}](https://arxiv.org/abs/{p['id']}) | {p['year']} | {cats}\n"
    md += f"   - {auth} — {p['source']}\n\n"

out = os.path.join(BASE, "arxiv_bibliography.md")
with open(out, "w", encoding="utf-8") as f:
    f.write(md)
print(f"\nMerged {len(all_papers)} unique papers → {out}")
