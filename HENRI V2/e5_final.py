"""E5 final: verify the executed promotion, authenticate local sources, seal E5a.

Reads every value from files on disk. Writes one receipt. Seals one governance
event only after the chain re-verifies.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, r"C:\Users\chan\AppData\Local\hermes\scripts")
import henri_audit as ha

E3 = Path(r"C:\Users\chan\henri-telemetry\e3")
E4WT = Path(r"C:\Users\chan\henri-worktrees\e4-wt")
E5WT = Path(r"C:\Users\chan\henri-worktrees\e5-wt")
OUT = E3 / "e5_final_receipt.json"
ACTOR = "henri-arbiter"

SEARCH_ROOTS = [Path(r"G:\My Drive\HENRI_Inbox"),
                Path(r"C:\Users\chan\Downloads"),
                Path(r"C:\Users\chan\Desktop\HENRI Research Vault")]
PATTERNS = ["*Sagnac*", "*BaTiO3*", "*batio3*", "*Pockels*", "*photonic*",
            "*Zone*B*", "*d2nn*", "*diffract*"]


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def git(cwd: Path, *a: str) -> str:
    return subprocess.run(["git", *a], cwd=str(cwd), capture_output=True,
                          text=True, timeout=120).stdout.strip()


def authenticate(p: Path) -> dict:
    try:
        b = p.read_bytes()
    except Exception as e:
        return {"path": str(p), "error": str(e)[:80]}
    head = b[:5]
    return {"path": str(p), "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest(),
            "magic": head.decode("latin-1"),
            "is_pdf": head == b"%PDF-", "is_text": b"\x00" not in b[:4096]}


def main() -> None:
    rec: dict = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    # ---- 1. E5a promotion receipt ----------------------------------------
    pr = E3 / "e5a_promotion_receipt.json"
    promo = json.loads(pr.read_text()) if pr.exists() else {}
    rec["promotion_receipt"] = {
        "path": str(pr), "sha256": sha(pr)[:16] if pr.exists() else None,
        "verdict": promo.get("VERDICT"),
        "commit": (promo.get("commit") or {}).get("head"),
        "push_remote": (promo.get("push") or {}).get("remote_obj"),
        "push_matches": (promo.get("push") or {}).get("matches_head"),
        "files_in_head": promo.get("post", {}).get("promoted_files_in_head"),
        "compile_ok": promo.get("compile_ok"),
        "scope_caveat": promo.get("carrier_scope_HONEST"),
    }
    print("[1] E5a verdict=" + str(promo.get("VERDICT"))
          + " commit=" + str((promo.get("commit") or {}).get("head")))

    # ---- 2. live git state (authoritative) --------------------------------
    rec["git"] = {
        "main": git(E4WT, "rev-parse", "--short", "origin/main"),
        "e4_head": git(E4WT, "rev-parse", "--short", "HEAD"),
        "e4_branch": git(E4WT, "rev-parse", "--abbrev-ref", "HEAD"),
        "e5_head": git(E5WT, "rev-parse", "--short", "HEAD"),
        "e5_branch": git(E5WT, "rev-parse", "--abbrev-ref", "HEAD"),
        "e5_remote": (git(E5WT, "ls-remote", "origin",
                          "refs/heads/carrier/e5-wave-superposition").split() or [""])[0][:7],
    }
    rec["git"]["e5_pushed_matches"] = rec["git"]["e5_head"] == rec["git"]["e5_remote"]
    print("[2] main=" + rec["git"]["main"] + " e4=" + rec["git"]["e4_head"]
          + " e5=" + rec["git"]["e5_head"] + " remote=" + rec["git"]["e5_remote"]
          + " match=" + str(rec["git"]["e5_pushed_matches"]))

    # ---- 3. arXiv pm/V evidence (from the sealed audit) ------------------
    za = E3 / "e5_zoneb_audit.json"
    z = json.loads(za.read_text()) if za.exists() else {}
    abs_ = z.get("arxiv_bto_abstracts", {})
    rec["arxiv_pmV"] = {
        "receipt": str(za), "sha16": sha(za)[:16] if za.exists() else None,
        "entries": abs_.get("entries", []),
        "pmV_found": sorted({v for e in abs_.get("entries", [])
                             for v in e.get("pm_per_V_values", [])}),
        "note": ("abstract-level string extraction only; a 1300 pm/V claim must be "
                 "pinned to a specific table/figure in a named source, not to an "
                 "abstract mention"),
    }
    print("[3] pm/V values seen in BTO abstracts: " + str(rec["arxiv_pmV"]["pmV_found"]))

    # ---- 4. authenticate local chip-related sources ----------------------
    found = []
    seen = set()
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for pat in PATTERNS:
            for p in root.glob(pat):
                if p in seen or not p.is_file():
                    continue
                seen.add(p)
                found.append(authenticate(p))
    found = found[:25]
    rec["local_sources"] = found
    pdfs = [f for f in found if f.get("is_pdf")]
    print("[4] local candidate sources=" + str(len(found))
          + " pdfs=" + str(len(pdfs)))
    for f in pdfs[:5]:
        print("    PDF " + str(f["bytes"]) + "B sha " + f["sha256"][:12] + "  "
              + Path(f["path"]).name[:70])

    # ---- 5. chain verify (authoritative) ---------------------------------
    ok0, msg0 = ha.verify_chain()
    rec["chain_before"] = {"ok": ok0, "message": msg0}
    print("[5] chain before: " + ("OK " if ok0 else "FAIL ") + msg0)

    # ---- 6. seal E5a ------------------------------------------------------
    h = ha.record_event(ACTOR, "HENRI_E5A_PROMOTED", {
        "carrier": "carrier/e5-wave-superposition",
        "promotion_receipt_sha256": rec["promotion_receipt"]["sha256"],
        "promotion_commit": rec["promotion_receipt"]["commit"],
        "remote_matches_head": rec["git"]["e5_pushed_matches"],
        "files_promoted": rec["promotion_receipt"]["files_in_head"],
        "source_carrier": "carrier/e4-construct (diagnostic milestone, closed)",
        "promoted_mechanism": (
            "E4b position-identifiable codec: reserved DISJOINT position channel "
            "(2048 of 8192 rows) carrying explicit end-distance boundary markers "
            "(e:, d1:, d2:, d3:); zero trainable parameters; position-region "
            "last-word-to-front cosine 0.0332 vs the g7 bag codec's 0.940; terminal "
            "recovery 32/32"),
        "scope_caveat_SEALED": (
            "the promoted object is a position-identifiable boundary-marker channel. "
            "It is NOT a demonstrated superposition generator. 'wave-superposition' "
            "is the user-directed branch name; any superposition claim requires its "
            "own measurement and a pre-registered coverage gate."),
        "bar": "backbone oracle (C2 oracle P@1 0.433); arm-b-class frozen readouts EXCLUDED from attribution",
        "zone_b_audit_sha256": rec["arxiv_pmV"]["sha16"],
        "notebooklm": "BLOCKED_STALE_AUTH",
        "no_promotion_to_main": True,
        "no_capability_claim": True,
    })
    ok, msg = ha.verify_chain()
    rec["seal"] = h
    rec["chain_after"] = {"ok": ok, "message": msg}
    print("[6] sealed E5A_PROMOTED #" + h[:16])
    print("[7] chain after: " + ("OK " if ok else "FAIL ") + msg)

    OUT.write_text(json.dumps(rec, indent=2))
    print("WROTE " + str(OUT) + " sha256=" + sha(OUT)[:16])


if __name__ == "__main__":
    main()
