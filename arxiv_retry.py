#!/usr/bin/env python3
"""Retry failed arXiv + S2 queries after rate-limit recovery (~2hr).
Appends to arxiv_papers.json, then regenerates arxiv_bibliography.md.
"""
import urllib.request, urllib.parse, xml.etree.ElementTree as ET, time, json, re, os, sys

BASE_DIR = r"C:\Users\chan\Desktop\HENRI 7B SWARM"
SRC_FILE = os.path.join(BASE_DIR, "arxiv_papers.json")
OUT_FILE = os.path.join(BASE_DIR, "arxiv_bibliography.md")
RETRY_LOG = os.path.join(BASE_DIR, "arxiv_retry.log")

# Load existing papers
existing = set()
if os.path.exists(SRC_FILE):
    with open(SRC_FILE, encoding="utf-8") as f:
        for p in json.load(f):
            existing.add(p["base_id"])

# Queries that failed (all 429 in previous runs)
QUERIES = [
    ("all:google+deepmind+AND+cat:cs.AI", "Google DeepMind"),
    ("all:functor+flow+AND+cat:cs.AI", "Functor Flow"),
    ("all:agentic+OR+all:context+engineering+AND+cat:cs.AI", "Agentic Context Eng"),
    ("all:nested+learning+AND+cat:cs.LG", "Nested/HOPE"),
    ("all:continual+learning+world+models+AND+cat:cs.AI", "Sapient-like"),
    ("all:JEPA+joint+embedding+AND+cat:cs.AI", "JEPA LeCun"),
    ("all:diffusion+generative+AND+cat:cs.LG", "Diffusion"),
    ("all:topological+data+analysis+AND+cat:cs.LG", "Topology+ML"),
    ("all:world+models+AND+cat:cs.AI", "World Models"),
    ("all:reinforcement+learning+AND+cat:stat.ML", "RL Foundations"),
    ("all:optimal+transport+AND+cat:cs.LG", "Optimal Transport"),
    ("all:geometric+deep+learning+AND+cat:cs.LG", "Geometric DL"),
    ("all:stanford+AND+cat:cs.AI", "Stanford CS"),
    ("all:mit+AND+cat:cs.AI", "MIT CS"),
    ("all:university+of+tokyo+AND+cat:cs.LG", "Univ Tokyo"),
    ("all:deepmind+AND+cat:cs.AI", "DeepMind"),
    ("all:openai+AND+cat:cs.AI", "OpenAI"),
    ("all:meta+AI+AND+cat:cs.AI", "Meta AI"),
    ("all:categorical+deep+learning+AND+cat:cs.LG", "Categorical DL"),
    ("all:physics+informed+neural+networks+AND+cat:cs.LG", "PINNs"),
]

NS = {'a': 'http://www.w3.org/2005/Atom'}
BASE = 'https://export.arxiv.org/api/query?max_results=12&sortBy=submittedDate&sortOrder=descending'
new_papers = 0

with open(RETRY_LOG, "a") as log:
    log.write(f"\n--- Retry run: {time.strftime('%Y-%m-%d %H:%M')} ---\n")
    
    for i, (q, label) in enumerate(QUERIES):
        url = BASE + '&search_query=' + urllib.parse.quote(q)
        try:
            d = urllib.request.urlopen(urllib.request.Request(url), timeout=20).read()
            root = ET.fromstring(d)
            entries = root.findall('a:entry', NS)
            n = 0
            for e in entries:
                rid = e.find('a:id', NS).text.strip().split('/abs/')[-1]
                base_id = re.sub(r'v\d+$', '', rid)
                if base_id in existing:
                    continue
                existing.add(base_id)
                new_papers += 1
                n += 1
            msg = f"[{i+1}/{len(QUERIES)}] {label}: {n} OK"
            print(msg)
            log.write(msg + "\n")
        except Exception as exc:
            msg = f"[{i+1}/{len(QUERIES)}] {label}: {exc}"
            print(msg)
            log.write(msg + "\n")
        time.sleep(3.0)

# Load all papers and regenerate bibliography
all_papers = []
if os.path.exists(SRC_FILE):
    with open(SRC_FILE, encoding="utf-8") as f:
        for p in json.load(f):
            all_papers.append(p)

# Also load S2 batch
s2_path = os.path.join(BASE_DIR, "arxiv_s2_batch.json")
if os.path.exists(s2_path):
    with open(s2_path, encoding="utf-8") as f:
        for p in json.load(f):
            if p["base_id"] not in {x["base_id"] for x in all_papers}:
                all_papers.append(p)

all_papers.sort(key=lambda p: p.get("year", "0"), reverse=True)

md = f"# HENRI Research Bibliography — {len(all_papers)} Primary Sources\n\n"
md += f"Generated: {time.strftime('%Y-%m-%d %H:%M')} | arXiv API + Semantic Scholar\n\n"
from collections import Counter
for d, n in Counter(p["source"] for p in all_papers).most_common():
    md += f"- **{d}**: {n} papers\n"
md += f"\n*New this run: {new_papers}*\n\n## Complete Paper List\n\n"
for i, p in enumerate(all_papers):
    auth = ", ".join(p["authors"][:3])
    if len(p["authors"]) > 3: auth += " et al."
    cats = ", ".join(p.get("categories", [])[:2])
    md += f"{i+1}. **{p['title']}**\n"
    md += f"   - [{p['id']}](https://arxiv.org/abs/{p['id']}) | {p.get('year','?')} | {cats}\n"
    md += f"   - {auth}\n\n"

with open(OUT_FILE, "w", encoding="utf-8") as f:
    f.write(md)

msg = f"Done: {new_papers} new, {len(all_papers)} total"
print(msg)
with open(RETRY_LOG, "a") as log:
    log.write(msg + "\n")
