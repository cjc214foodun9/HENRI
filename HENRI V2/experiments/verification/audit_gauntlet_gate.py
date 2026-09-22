#!/usr/bin/env python3
"""Audit the gauntlet telemetry for the Sagnac veto gate's THREE outcomes.

THE QUESTION
    Action 3 requires evidence that `hard_vetoed` takes BOTH values, because a gate
    that only ever reads False is not proven live. The earlier runs could not show
    this: every recorded payload was a bare `{"error": "RuntimeError"}`, which is
    indistinguishable from "the gate is permissive".

    With the width contract in place the payload has three DISTINCT shapes:
        gate UNAVAILABLE  -> has "gate_status", has NO "hard_vetoed" key
        gate PASSED       -> has "hard_vetoed": false and delta_axiom
        gate VETOED       -> has "hard_vetoed": true  and delta_axiom
        gate ERROR        -> has "error" (a real bug, e.g. dtype/device)
    This script counts each shape per arm and reports whether both pass and veto
    occurred. It reads ONLY the on-disk telemetry, never stdout, because the veto
    payload is persisted to JSONL (an earlier check grepped the wrong stream).

NO heredoc, no inline one-liners: argv only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_payloads(path: Path) -> dict:
    """Collect every `sagnac_veto` payload, at any nesting depth."""
    payloads, records, bad_lines = [], 0, 0
    if not path.exists():
        return {"exists": False, "payloads": []}
    with path.open(encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records += 1
            try:
                obj = json.loads(line)
            except Exception:  # noqa: BLE001
                bad_lines += 1
                continue

            stack = [obj]
            while stack:
                cur = stack.pop()
                if isinstance(cur, dict):
                    v = cur.get("sagnac_veto")
                    if isinstance(v, dict):
                        payloads.append(v)
                    stack.extend(cur.values())
                elif isinstance(cur, list):
                    stack.extend(cur)
    return {"exists": True, "records": records, "bad_lines": bad_lines,
            "payloads": payloads}


def classify(payloads: list) -> dict:
    out = {"n": len(payloads), "unavailable": 0, "passed": 0, "vetoed": 0,
           "error": 0, "other": 0, "gate_status_values": {}, "error_msgs": {},
           "hard_vetoed_true": 0, "hard_vetoed_false": 0,
           "delta_axiom_min": None, "delta_axiom_max": None}
    deltas = []
    for p in payloads:
        if "gate_status" in p:
            out["unavailable"] += 1
            gs = str(p.get("gate_status"))
            out["gate_status_values"][gs] = out["gate_status_values"].get(gs, 0) + 1
        elif "error" in p:
            out["error"] += 1
            msg = str(p.get("error"))[:110]
            out["error_msgs"][msg] = out["error_msgs"].get(msg, 0) + 1
        elif "hard_vetoed" in p:
            if bool(p["hard_vetoed"]):
                out["vetoed"] += 1
                out["hard_vetoed_true"] += 1
            else:
                out["passed"] += 1
                out["hard_vetoed_false"] += 1
            d = p.get("delta_axiom")
            if isinstance(d, (int, float)):
                deltas.append(float(d))
        else:
            out["other"] += 1
    if deltas:
        out["delta_axiom_min"] = round(min(deltas), 6)
        out["delta_axiom_max"] = round(max(deltas), 6)
    out["both_values_present"] = (out["hard_vetoed_true"] > 0
                                  and out["hard_vetoed_false"] > 0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--off", required=True, help="telemetry jsonl, bridge OFF arm")
    ap.add_argument("--on", required=True, help="telemetry jsonl, bridge ON arm")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    arms = {"bridge_off": str(Path(args.off)), "bridge_on": str(Path(args.on))}
    result: dict = {"arms": {}, "errors": [], "verdicts": {}}
    for name, path in arms.items():
        data = load_payloads(Path(path))
        if not data.get("exists"):
            result["arms"][name] = {"path": path, "exists": False}
            result["errors"].append(f"{name}: telemetry not found at {path}")
            continue
        cls = classify(data["payloads"])
        result["arms"][name] = {
            "path": path, "exists": True, "records": data.get("records"),
            "bad_lines": data.get("bad_lines"), **cls,
        }

    off = result["arms"].get("bridge_off", {})
    on = result["arms"].get("bridge_on", {})

    # ---- Pre-registered expectations -------------------------------------------
    # OFF: the veto cannot run at this scale, so every payload must be UNAVAILABLE
    #      and NO hard_vetoed key may appear. This proves availability is not being
    #      confused with permission.
    if off.get("exists"):
        result["verdicts"]["off_all_unavailable"] = (
            off.get("n", 0) > 0 and off.get("unavailable", 0) == off.get("n", -1))
        result["verdicts"]["off_no_hard_vetoed_key"] = (
            off.get("hard_vetoed_true", 0) + off.get("hard_vetoed_false", 0) == 0)
        if not result["verdicts"]["off_all_unavailable"]:
            result["errors"].append(
                f"OFF arm: expected every payload UNAVAILABLE, got {off}")
        if not result["verdicts"]["off_no_hard_vetoed_key"]:
            result["errors"].append(
                "OFF arm: a hard_vetoed key appeared while the gate could not run; "
                "availability is being reported as permission")
    # ON: the diagnostic bridge makes it computable, so BOTH values must appear.
    if on.get("exists"):
        result["verdicts"]["on_both_values"] = bool(on.get("both_values_present"))
        if not result["verdicts"]["on_both_values"]:
            result["errors"].append(
                f"ON arm: hard_vetoed did not take both values -> gate not proven "
                f"live. passed={on.get('passed')} vetoed={on.get('vetoed')} "
                f"unavailable={on.get('unavailable')} error={on.get('error')}")

    result["verdict"] = "PASS" if not result["errors"] else "FAIL"
    out = Path(args.out) if args.out else (
        Path(__file__).resolve().parent / "gauntlet_gate_audit.json")
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")

    print("GAUNTLET_GATE_AUDIT=" + result["verdict"]
          + " errors=" + str(len(result["errors"])))
    for name in ("bridge_off", "bridge_on"):
        a = result["arms"].get(name, {})
        if not a.get("exists"):
            print(f"  {name}: MISSING")
            continue
        print(f"  {name}: n={a['n']} unavailable={a['unavailable']} "
              f"passed={a['passed']} vetoed={a['vetoed']} error={a['error']} "
              f"other={a['other']} both={a['both_values_present']} "
              f"delta=[{a['delta_axiom_min']},{a['delta_axiom_max']}]")
        if a["gate_status_values"]:
            print(f"      gate_status={a['gate_status_values']}")
        if a["error_msgs"]:
            print(f"      errors={a['error_msgs']}")
    for vk, vv in result["verdicts"].items():
        print(f"  verdict {vk} = {vv}")
    for e in result["errors"]:
        print("ERR: " + e[:220])
    print("RECEIPT=" + str(out))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
