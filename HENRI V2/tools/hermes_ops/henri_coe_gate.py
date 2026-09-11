#!/usr/bin/env python3
"""henri_coe_gate.py — ScientistOne Chain-of-Evidence (CoE) enforcement gate.

Converts the §6A CoE doctrine (henri-agent-integration/SKILL.md and
references/scientistone-chain-of-evidence.md) into a deterministic, runnable
check. Doctrine without enforcement is prose.

Rules (from §6A and integrity gates I1-I4):
  R1  required fields present
  R2  status in the allowed label set
  R3  status=observed/derived requires a hashed source artifact, a verifier,
      a verification_command, and a commit identity            (I1)
  R4  claim_type=citation requires a canonical identifier     (I3)
  R5  claim_type=method requires a code location              (I4)
  R6  non-verified claims must state limitations
  R7  local artifacts are re-hashed and compared              (I1 artifact)
  R8  a claim crossing a holon boundary without evidence linkage is rejected

CPR = verified material claims / total material claims.
CPR is a reporting ratio, not a proof. Hard claims are never dropped to
improve CPR.

CLI:
  python henri_coe_gate.py validate <claims.json> [--check-artifacts]
  python henri_coe_gate.py selftest
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ALLOWED = {"observed", "derived", "inferred", "hypothesis",
           "falsified", "blocked", "unverified"}
VERIFIED = {"observed", "derived"}
NOT_VERIFIED = {"inferred", "hypothesis", "falsified", "blocked", "unverified"}
MATERIAL = {"metric", "citation", "method", "specification", "operational"}
CANON_KEYS = ("doi", "arxiv", "semantic_scholar", "semanticscholar", "pubmed",
              "pmid", "s2", "isbn", "canonical_id")
CODE_RE = re.compile(
    r"[\w./\\-]+\.(py|rs|cu|cpp|c|h|hpp|sh|yaml|yml|toml|json|md)(:\d+)?", re.I)


def _empty(v) -> bool:
    return v is None or v == "" or v == [] or v == {}


def check_claim(rec: dict, check_artifacts: bool = False) -> tuple[bool, list[str]]:
    """Return (ok, reasons). Fail-closed: any missing requirement fails."""
    reasons: list[str] = []
    if not isinstance(rec, dict):
        return False, ["R1 record is not an object"]

    missing = [k for k in ("claim_id", "claim_text", "claim_type", "status")
               if _empty(rec.get(k))]
    if missing:
        reasons.append(f"R1 missing required fields: {missing}")

    status = str(rec.get("status", "")).strip().lower()
    if status and status not in ALLOWED:
        reasons.append(f"R2 invalid status '{status}'")

    arts = rec.get("source_artifacts")
    if arts is None:
        arts = []
    if not isinstance(arts, list):
        reasons.append("R1 source_artifacts is not a list")
        arts = []

    if status in VERIFIED:
        if not arts:
            reasons.append("R3 no source_artifacts for a verified claim")
        if not any(isinstance(a, dict) and a.get("sha256") for a in arts):
            reasons.append("R3 no artifact sha256 (digest required)")
        for key in ("verifier", "verification_command"):
            if _empty(rec.get(key)):
                reasons.append(f"R3 missing {key}")
        if _empty(rec.get("commit")):
            reasons.append("R3 missing commit identity")

    ctype = str(rec.get("claim_type", "")).strip().lower()

    if ctype == "citation":
        blob = json.dumps(arts, sort_keys=True).lower()
        if not any(k in blob for k in CANON_KEYS):
            reasons.append("R4 citation without a canonical identifier (I3)")

    if ctype == "method":
        blob = json.dumps([rec.get("verification_command", ""), arts]).lower()
        if not CODE_RE.search(blob):
            reasons.append("R5 method claim without a code location (I4)")

    if status in NOT_VERIFIED and _empty(rec.get("limitations")):
        reasons.append("R6 non-verified claim must state limitations")

    if check_artifacts:
        for art in arts:
            if not isinstance(art, dict):
                continue
            path, want = art.get("uri_or_path"), art.get("sha256")
            if not path or not want:
                continue
            if isinstance(path, str) and path.lower().startswith(("http://", "https://")):
                continue                      # remote artifact: cannot hash locally
            fp = Path(path)
            if not fp.is_file():
                # A declared local artifact that does not exist is the single most
                # common fabrication shape (a reference model asserting a module).
                reasons.append(f"R9 declared artifact does not exist: {path}")
                continue
            got = hashlib.sha256(fp.read_bytes()).hexdigest()
            if got != want:
                reasons.append(
                    f"R7 sha256 mismatch for {path} (want {str(want)[:12]}…, "
                    f"got {got[:12]}…)")

    if rec.get("crosses_holon_boundary") and _empty(rec.get("claim_id")):
        reasons.append("R8 boundary crossing without evidence linkage")

    return (not reasons), reasons


def compute_cpr(records: list[dict], check_artifacts: bool = False) -> dict:
    total = verified = blocked = 0
    failures: list[tuple] = []
    for rec in records:
        ctype = str(rec.get("claim_type", "")).strip().lower()
        if ctype not in MATERIAL:
            continue
        total += 1
        ok, why = check_claim(rec, check_artifacts)
        status = str(rec.get("status", "")).strip().lower()
        if status in VERIFIED and ok:
            verified += 1
        else:
            blocked += 1
        if not ok:
            failures.append((rec.get("claim_id", "<no-id>"), why))
    pct = (100.0 * verified / total) if total else 0.0
    return {"total": total, "verified": verified,
            "blocked_or_failed": blocked, "cpr_pct": pct, "failures": failures}


def notification(res: dict, bundle: str = "coe") -> str:
    top = res["failures"][0][1][0] if res["failures"] else "none"
    return (f"[{bundle}] status | CPR: {res['verified']}/{res['total']} "
            f"({res['cpr_pct']:.1f}%)\n"
            f"Blocked/unverified: {res['blocked_or_failed']} | "
            f"highest-risk reason={top}")


def _cmd_validate(argv: list[str]) -> int:
    if not argv:
        print("usage: validate <claims.json> [--check-artifacts]")
        return 2
    path = Path(argv[0])
    check_artifacts = "--check-artifacts" in argv
    if not path.is_file():
        print(f"BLOCKED: {path} not found")
        return 3
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data.get("claims") if isinstance(data, dict) else data
    if not isinstance(records, list):
        print("BLOCKED: expected a list or {'claims': [...]}")
        return 3

    res = compute_cpr(records, check_artifacts)
    print(f"{'claim_id':28s} {'type':14s} {'status':11s} verdict")
    print("-" * 68)
    for rec in records:
        ok, why = check_claim(rec, check_artifacts)
        print(f"{str(rec.get('claim_id'))[:28]:28s} "
              f"{str(rec.get('claim_type'))[:14]:14s} "
              f"{str(rec.get('status'))[:11]:11s} "
              f"{'ACCEPT' if ok else 'REJECT'}")
        for r in why:
            print(f"    -> {r}")
    print("-" * 68)
    print(notification(res, bundle=path.stem))
    return 0 if res["blocked_or_failed"] == 0 else 1


def _selftest() -> int:
    """Falsifiable proof that the gate rejects. A gate that accepts
    everything is not a gate."""
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    real = tmp / "artifact.txt"
    real.write_bytes(b"observed evidence\n")
    sha = hashlib.sha256(real.read_bytes()).hexdigest()

    cases = [
        ("prose-only claim is REJECTED", {
            "claim_id": "c1", "claim_text": "model reaches 0.9", "claim_type": "metric",
            "status": "observed"}, False),
        ("fully-evidenced claim is ACCEPTED", {
            "claim_id": "c2", "claim_text": "artifact hashes", "claim_type": "metric",
            "status": "observed", "verifier": "sha256sum",
            "verification_command": "sha256sum artifact.txt", "commit": "abc1234",
            "source_artifacts": [{"uri_or_path": str(real), "sha256": sha}]}, False),
        ("citation without canonical id is REJECTED", {
            "claim_id": "c3", "claim_text": "paper says X", "claim_type": "citation",
            "status": "observed", "verifier": "web", "verification_command": "curl",
            "commit": "abc1234",
            "source_artifacts": [{"uri_or_path": "some.pdf", "sha256": sha}]}, False),
        ("method without code location is REJECTED", {
            "claim_id": "c4", "claim_text": "implements CoE", "claim_type": "method",
            "status": "observed", "verifier": "read", "verification_command": "read it",
            "commit": "abc1234",
            "source_artifacts": [{"uri_or_path": "x", "sha256": sha}]}, False),
        ("tampered artifact is REJECTED when hashing enabled", {
            "claim_id": "c5", "claim_text": "artifact hashes", "claim_type": "metric",
            "status": "observed", "verifier": "sha256sum",
            "verification_command": "sha256sum artifact.txt", "commit": "abc1234",
            "source_artifacts": [{"uri_or_path": str(real), "sha256": "0" * 64}]}, True),
        ("blocked claim without limitations is REJECTED", {
            "claim_id": "c6", "claim_text": "unknown", "claim_type": "metric",
            "status": "blocked"}, False),
        ("NONEXISTENT declared artifact is REJECTED", {
            "claim_id": "c7", "claim_text": "module exists", "claim_type": "method",
            "status": "observed", "verifier": "ls",
            "verification_command": "ls missing.py", "commit": "abc1234",
            "source_artifacts": [{"uri_or_path": str(tmp / "missing.py"),
                                  "sha256": sha,
                                  "code_location": "missing.py:1"}]}, True),
    ]

    failed = 0
    for label, rec, ca in cases:
        ok, why = check_claim(rec, ca)
        want_accept = "ACCEPTED" in label
        good = (ok == want_accept)
        failed += (not good)
        print(f"[{'PASS' if good else 'FAIL'}] {label}  "
              f"(gate={'ACCEPT' if ok else 'REJECT'}"
              f"{'' if good else '; ' + '; '.join(why)})")
    print()
    print(f"selftest: {len(cases) - failed}/{len(cases)} passed")
    return 0 if failed == 0 else 1


def main() -> None:
    argv = sys.argv[1:]
    if not argv:
        print(__doc__)
        return
    if argv[0] == "validate":
        sys.exit(_cmd_validate(argv[1:]))
    if argv[0] == "selftest":
        sys.exit(_selftest())
    print(f"unknown command: {argv[0]}")
    sys.exit(2)


if __name__ == "__main__":
    main()
