"""HENRI immutable audit chain — SHA-256 hash-linked governance ledger.

Append-only JSONL ledger recording GOVERNANCE EVENTS from real scripts and
agent actions: paper ingested, plan delivered, human decision, patch pushed,
CI verdict. Each record links to its predecessor's hash; tampering with any
historical line breaks every subsequent hash.

What this is NOT: a thought recorder. Agent reasoning lives in state.db
(sessions/messages), MoA turn traces (moa.save_traces=true, written under
~\\AppData\\Local\\hermes\\moa_traces\\), and `hermes sessions export --redact`.
This chain is the compact, cryptographically sealed *decision* log on top.

CLI:
  python henri_audit.py record <actor> <action> '<json-payload>'
  python henri_audit.py verify
  python henri_audit.py tail [n]

Importable:
  from henri_audit import record_event
  record_event("henri_ingest", "PAPER_INGESTED", {"title": ...})
"""
import hashlib
import json
import sys
import time
from pathlib import Path

LEDGER_DIR = Path(r"C:\Users\chan\AppData\Local\hermes\audit")
LEDGER = LEDGER_DIR / "henri_audit_chain.jsonl"
GENESIS_PREV = "0" * 64


def _hash(idx: int, ts: float, actor: str, action: str, payload: dict, prev: str) -> str:
    body = f"{idx}|{ts}|{actor}|{action}|{json.dumps(payload, sort_keys=True)}|{prev}"
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _read_all() -> list[dict]:
    if not LEDGER.exists():
        return []
    out = []
    with open(LEDGER, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def record_event(actor: str, action: str, payload: dict) -> str:
    """Append one sealed record; returns its hash. Never raises into callers
    that guard with try/except — but integrity errors MUST propagate."""
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    chain = _read_all()
    idx = chain[-1]["idx"] + 1 if chain else 0
    prev = chain[-1]["hash"] if chain else GENESIS_PREV
    ts = time.time()
    h = _hash(idx, ts, actor, action, payload, prev)
    rec = {"idx": idx, "ts": ts, "actor": actor, "action": action,
           "payload": payload, "prev_hash": prev, "hash": h}
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")
        f.flush()
    return h


def verify_chain() -> tuple[bool, str]:
    chain = _read_all()
    if not chain:
        return True, "empty chain (genesis not yet written)"
    prev = GENESIS_PREV
    for i, rec in enumerate(chain):
        if rec["idx"] != i:
            return False, f"index gap at record {i}: idx={rec['idx']}"
        if rec["prev_hash"] != prev:
            return False, f"prev_hash mismatch at idx {i}"
        expect = _hash(rec["idx"], rec["ts"], rec["actor"], rec["action"],
                       rec["payload"], rec["prev_hash"])
        if expect != rec["hash"]:
            return False, f"hash mismatch at idx {i} — record tampered or corrupted"
        prev = rec["hash"]
    return True, f"chain intact: {len(chain)} records, head {prev[:16]}…"


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return
    cmd = args[0]
    if cmd == "record":
        if len(args) < 4:
            print("usage: record <actor> <action> '<json-payload>'")
            sys.exit(2)
        h = record_event(args[1], args[2], json.loads(args[3]))
        print(f"sealed #{h[:16]}…")
    elif cmd == "verify":
        ok, msg = verify_chain()
        print(("✓ " if ok else "✗ ") + msg)
        sys.exit(0 if ok else 1)
    elif cmd == "tail":
        n = int(args[1]) if len(args) > 1 else 5
        for rec in _read_all()[-n:]:
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(rec["ts"]))
            print(f"#{rec['idx']:04d} {ts} {rec['actor']}::{rec['action']} "
                  f"{json.dumps(rec['payload'])[:100]}")
    else:
        print(f"unknown command: {cmd}")
        sys.exit(2)


if __name__ == "__main__":
    main()
