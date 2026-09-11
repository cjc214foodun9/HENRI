"""E5 source harvest: arXiv primary bytes + local corpus + Drive inbox scan.

arXiv: separate native-path file per query (never overwrite), hash the returned
bytes, record as OBSERVED_PRIMARY_BYTES.
Local: scan vault / inbox / known paper files for the digital-twin terms.
Nothing is inferred from stdout; the JSON receipt is the evidence.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

OUT = Path(r"C:\Users\chan\henri-telemetry\e3\e5_sources.json")

QUERIES = {
    "bto_photonic": 'all:%22barium+titanate%22+AND+all:%22photonic%22',
    "bto_pockels": 'all:%22barium+titanate%22+AND+all:%22electro-optic%22',
    "sagnac_nn": 'all:%22Sagnac%22+AND+all:%22interferometer%22+AND+all:%22neural%22',
    "d2nn": 'all:%22diffractive+deep+neural+network%22',
    "micronheater_photonic": 'all:%22microheater%22+AND+all:%22photonic%22',
}

CORPORA = [
    Path(r"C:\Users\chan\Desktop\HENRI Research Vault"),
    Path(r"C:\Users\chan\Documents\HENRI_Research_Vault"),
    Path(r"G:\My Drive\HENRI_Inbox"),
    Path(r"C:\Users\chan\Downloads"),
]
TERMS = ["pockels", "barium", "batio3", "sagnac", "diffractive", "photonic",
         "microheater", "interferom", "ferroelectric", "optoelectronic", "optics"]


def main() -> None:
    rec: dict = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    # ---------------- arXiv primary bytes ---------------------------------
    ax = {}
    for k, q in QUERIES.items():
        out = Path(rf"C:\Users\chan\arx_{k}.xml")
        url = (f"https://export.arxiv.org/api/query?search_query={q}"
               f"&max_results=6")
        try:
            r = subprocess.run(["curl", "-sL", "--max-time", "40", url,
                                "-o", str(out)], timeout=70,
                               capture_output=True, text=True)
            b = out.read_bytes() if out.exists() else b""
            t = b.decode("utf-8", "replace")
            entries = re.findall(
                r"<entry>.*?<id>(http://arxiv\.org/abs/[^<]+)</id>.*?<title>([^<]+)</title>",
                t, re.S)
            ax[k] = {
                "bytes": len(b),
                "sha256": hashlib.sha256(b).hexdigest() if b else None,
                "n_entries": len(entries),
                "entries": [{"id": i.rsplit("/", 1)[-1], "title": ti.strip()[:90]}
                            for i, ti in entries[:5]],
                "curl_rc": r.returncode,
            }
        except Exception as e:
            ax[k] = {"error": str(e)[:120]}
    rec["arxiv_primary_bytes"] = ax

    # ---------------- local corpus ----------------------------------------
    lc = {"roots": [], "hits": []}
    for c in CORPORA:
        rec_root = {"path": str(c), "exists": c.exists(), "kind": None}
        if not c.exists():
            lc["roots"].append(rec_root)
            continue
        rec_root["kind"] = "file" if c.is_file() else "dir"
        lc["roots"].append(rec_root)
        if c.is_file():
            t = c.read_text(encoding="utf-8", errors="replace").lower()
            h = [k for k in TERMS if k in t]
            if h:
                lc["hits"].append({"path": str(c), "terms": h, "how": "content"})
            continue
        n = 0
        for p in c.rglob("*"):
            if not p.is_file():
                continue
            n += 1
            if n > 6000:
                break
            name = p.name.lower()
            h = [k for k in TERMS if k in name]
            if h:
                lc["hits"].append({"path": str(p), "terms": h, "how": "filename"})
    lc["scanned_files_cap"] = 6000
    rec["local_corpus"] = lc

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=2))
    print(json.dumps({"arxiv": {k: {"bytes": v.get("bytes"), "n": v.get("n_entries"),
                                    "sha256": (v.get("sha256") or "")[:16]}
                                for k, v in ax.items()},
                      "local_roots": lc["roots"],
                      "local_hits": len(lc["hits"])}, indent=2))
    print(f"\nWROTE {OUT} sha256={hashlib.sha256(OUT.read_bytes()).hexdigest()[:16]}")


if __name__ == "__main__":
    main()
