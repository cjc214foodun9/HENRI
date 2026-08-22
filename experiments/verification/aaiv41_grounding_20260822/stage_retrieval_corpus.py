#!/usr/bin/env python
"""CLASS51 P3(a): stage the retrieval corpus from pinned CPython docs.

Reproducible staging: fetch each module .rst from the immutable commit
SHA, save LF-normalized bytes to data/backbone_retrieval_corpus/, and
write manifest.json (henri.corpus-manifest.v1).
"""
import hashlib
import json
import pathlib
import urllib.request

PIN = "f74cdf80a120649e4c353430da8cbd1305c00993"
MODULES = [
    "bisect", "collections", "functools", "heapq", "itertools", "json",
    "math", "os.path", "pathlib", "random", "re", "statistics", "string",
]
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
ROOT = next(p for p in SCRIPT_DIR.parents if (p / ".git").exists())
BASE = ROOT / "data" / "backbone_retrieval_corpus"
BASE.mkdir(parents=True, exist_ok=True)

entries = []
for mod in MODULES:
    fn = mod.replace(".", "_") + ".rst"
    url = f"https://raw.githubusercontent.com/python/cpython/{PIN}/Doc/library/{mod}.rst"
    raw = urllib.request.urlopen(url, timeout=60).read()
    assert len(raw) > 500, f"short/empty fetch for {mod}: {len(raw)} bytes"
    lf = raw.replace(b"\r\n", b"\n")
    (BASE / fn).write_bytes(lf)
    entries.append({
        "module": mod,
        "file": fn,
        "sha256": hashlib.sha256(lf).hexdigest(),
        "bytes_lf": len(lf),
        "source_url": f"https://raw.githubusercontent.com/python/cpython/{PIN}/Doc/library/{mod}.rst",
    })

manifest = {
    "schema_id": "henri.corpus-manifest.v1",
    "source": "CPython Doc/library (PSF-2.0)",
    "pinned_commit": PIN,
    "fetched_utc": "2026-08-22",
    "files": entries,
    "aggregate_sha256": hashlib.sha256(
        "".join(e["sha256"] for e in entries).encode()
    ).hexdigest(),
}
(BASE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({
    "files": len(entries),
    "total_bytes_lf": sum(e["bytes_lf"] for e in entries),
    "aggregate_sha256": manifest["aggregate_sha256"],
}))
