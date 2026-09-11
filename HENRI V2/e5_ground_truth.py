"""Ground truth for the metric-C correction + C2 prereg. All values from disk.

Also probes whether the Drive inbox is WRITABLE by this process, because
Reference 1 (a tool-less advisory) claimed the Drive write failed silently.
That claim must be tested, not believed.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

FUNCTOR = Path(r"G:\My Drive\HENRI_Inbox\Sagnacfunctor.txt")
PROBE = Path(r"G:\My Drive\HENRI_Inbox\.write_probe_henri.tmp")
LED = Path(r"C:\Users\chan\AppData\Local\hermes\audit\henri_audit_chain.jsonl")
E5WT = Path(r"C:\Users\chan\henri-worktrees\e5-wt")
OUT = Path(r"C:\Users\chan\henri-telemetry\e3\e5_ground_truth.json")


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main() -> None:
    rec = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    # ---------- functor true state ----------------------------------------
    b = FUNCTOR.read_bytes()
    t = b.decode("utf-8", errors="replace")
    lines = t.splitlines()
    starts = [i + 1 for i, l in enumerate(lines)
              if l.strip().startswith("def compute_sagnac_delta(")]
    classes = [l.strip() for l in lines if l.strip().startswith("class ")]
    rec["functor"] = {
        "path": str(FUNCTOR), "exists": True, "bytes": len(b),
        "sha256": hashlib.sha256(b).hexdigest(),
        "lines": len(lines),
        "classes": classes,
        "compute_sagnac_delta_def_lines": starts,
        "has_vacuous_divide_by_D": "1.0 - sim" in t and "/ self.D" in t,
        "n_occurrences_of_self_D_div": t.count("/ self.D"),
        "main_guard": 'if __name__ == "__main__":' in t,
        "assert_line": next((l.strip() for l in lines if "assert telemetry" in l), None),
    }
    print("[functor] " + str(rec["functor"]["bytes"]) + "B sha "
          + rec["functor"]["sha256"][:16] + " classes=" + str(classes))
    print("[functor] compute_sagnac_delta at lines " + str(starts)
          + "  vacuous=" + str(rec["functor"]["has_vacuous_divide_by_D"]))
    # print the exact body of the first def
    if starts:
        s = starts[0] - 1
        body = []
        for j in range(s, min(s + 16, len(lines))):
            body.append(str(j + 1) + ": " + lines[j])
            if j > s and lines[j].strip().startswith("def "):
                break
        rec["functor"]["first_def_body"] = body
        for l in body:
            print("      " + l)

    # ---------- Drive writability probe -----------------------------------
    probe = {"attempted": True}
    try:
        payload = b"henri-write-probe-" + str(time.time()).encode()
        PROBE.write_bytes(payload)
        back = PROBE.read_bytes()
        probe.update({"write_ok": True, "readback_matches": back == payload,
                      "bytes": len(back)})
        PROBE.unlink()
        probe["cleanup_ok"] = not PROBE.exists()
    except Exception as e:
        probe.update({"write_ok": False, "error": type(e).__name__ + ": " + str(e)[:200]})
    rec["drive_write_probe"] = probe
    print("[drive] write_ok=" + str(probe.get("write_ok"))
          + " readback=" + str(probe.get("readback_matches"))
          + " cleanup=" + str(probe.get("cleanup_ok"))
          + ("  err=" + probe.get("error", "") if not probe.get("write_ok") else ""))

    # ---------- ledger ------------------------------------------------------
    rows = [json.loads(l) for l in LED.read_text(encoding="utf-8").splitlines() if l.strip()]
    prev, ok = "0" * 64, True
    for i, r in enumerate(rows):
        body = (f"{r['idx']}|{r['ts']}|{r['actor']}|{r['action']}|"
                f"{json.dumps(r['payload'], sort_keys=True)}|{r['prev_hash']}")
        if (r["idx"] != i or r["prev_hash"] != prev
                or hashlib.sha256(body.encode("utf-8")).hexdigest() != r["hash"]):
            ok = False
            break
        prev = r["hash"]
    rec["ledger"] = {"records": len(rows), "intact": ok,
                     "head": rows[-1]["hash"][:16],
                     "last6": [[r["idx"], r["action"]] for r in rows[-6:]]}
    print("[ledger] records=" + str(len(rows)) + " intact=" + str(ok)
          + " head=" + str(rec["ledger"]["head"]))

    # ---------- git ---------------------------------------------------------
    def g(*a):
        return subprocess.run(["git", *a], cwd=str(E5WT), capture_output=True,
                              text=True, timeout=90).stdout.strip()
    rec["git"] = {"head": g("rev-parse", "--short", "HEAD"),
                  "branch": g("rev-parse", "--abbrev-ref", "HEAD"),
                  "remote": g("ls-remote", "origin",
                              "refs/heads/carrier/e5-wave-superposition").split()[:1],
                  "dirty": g("status", "--porcelain=v1", "-uall").splitlines()}
    print("[git] " + rec["git"]["branch"] + " @ " + rec["git"]["head"]
          + " remote=" + str(rec["git"]["remote"]))

    OUT.write_text(json.dumps(rec, indent=2))
    print("WROTE " + str(OUT) + " sha256=" + sha(OUT)[:16])


if __name__ == "__main__":
    main()
