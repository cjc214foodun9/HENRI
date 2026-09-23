#!/usr/bin/env python3
"""REGRESSION PROBE: why did the veto payload vanish from telemetry after the fix?

THE OBSERVATION
    pre-fix  (bridge OFF) production_run_1790116642.jsonl : 60 sagnac_veto payloads,
             all gate_status=UNAVAILABLE_SHAPE_MISMATCH
    pre-fix  (bridge ON ) production_run_1790116691.jsonl : 60 sagnac_veto payloads
    post-fix (bridge OFF) production_run_1790117826.jsonl : 0  sagnac_veto payloads
                                                            records=73, opine=[opine]x64

    The run itself completed (steps to 63, GAME_OVER, scorecard written), and the
    `[opine]` line printed 64 times, so the OPINE block EXECUTED. Only the payload
    disappeared from the telemetry record.

HYPOTHESES TO SEPARATE (no guessing)
    H1  the macro-option branch is no longer taken, so `opine_info` is built by a
        different arm that omits `sagnac_veto`
    H2  the veto now SUCCEEDS, and the success path writes a payload the scanner
        misses because the key or nesting changed
    H3  telemetry emission of `opine_info` is conditional on something the fix changed
    H4  the scanner is wrong (reporting 0 when payloads exist)

METHOD
    Read all three JSONL files and report, per file:
      * total records
      * how many records contain the substring "opine_info" and/or "sagnac"
      * the SET of keys seen under any `opine_info`-like dict
      * every distinct key that holds a dict containing "delta_axiom" or "gate_status"
      * whether any key named exactly `sagnac_veto` exists at any depth
    That distinguishes H1/H2/H3 from H4 without assuming a schema.
"""
from __future__ import annotations

import json
from pathlib import Path

EXPORTS = Path(r"C:\Users\chan\HENRI_telemetry_exports")
FILES = {
    "pre_fix_bridge_off": "production_run_1790116642.jsonl",
    "pre_fix_bridge_on": "production_run_1790116691.jsonl",
    "post_fix_bridge_off": "production_run_1790117826.jsonl",
}

out: dict = {"files": {}, "notes": []}

for label, name in FILES.items():
    p = EXPORTS / name
    rec: dict = {"path": str(p), "exists": p.exists()}
    if not p.exists():
        out["files"][label] = rec
        continue
    text = p.read_text(encoding="utf-8", errors="ignore")
    lines = [l for l in text.splitlines() if l.strip()]
    rec["records"] = len(lines)
    rec["mentions_opine_info"] = text.count("opine_info")
    rec["mentions_sagnac"] = text.count("sagnac")
    rec["mentions_sagnac_veto_key"] = text.count('"sagnac_veto"')
    rec["mentions_gate_status"] = text.count("gate_status")
    rec["mentions_hard_vetoed"] = text.count("hard_vetoed")
    rec["mentions_delta_axiom"] = text.count("delta_axiom")
    rec["mentions_macro_option"] = text.count("macro_option")

    # key census over any dict whose key is literally sagnac_veto, and over any dict
    # that looks like an opine payload
    sagnac_keys: dict = {}
    opine_like_keys: dict = {}
    veto_like: dict = {}
    for ln in lines:
        try:
            obj = json.loads(ln)
        except Exception:
            continue
        stack = [obj]
        while stack:
            cur = stack.pop()
            if isinstance(cur, dict):
                for k, v in cur.items():
                    if k == "sagnac_veto" and isinstance(v, dict):
                        for kk in v:
                            sagnac_keys[kk] = sagnac_keys.get(kk, 0) + 1
                    if k in ("opine_info", "macro_option_log") and isinstance(v, dict):
                        for kk in v:
                            opine_like_keys[kk] = opine_like_keys.get(kk, 0) + 1
                    if k in ("opine_info", "macro_option_log") and isinstance(v, list):
                        for item in v[:4]:
                            if isinstance(item, dict):
                                for kk in item:
                                    opine_like_keys[kk] = opine_like_keys.get(kk, 0) + 1
                    if isinstance(v, dict) and (
                            "delta_axiom" in v or "gate_status" in v
                            or "hard_vetoed" in v or "error" in v):
                        for kk in v:
                            veto_like[kk] = veto_like.get(kk, 0) + 1
                    stack.append(v)
            elif isinstance(cur, list):
                stack.extend(cur)
    rec["sagnac_veto_payload_keys"] = sagnac_keys
    rec["opine_like_keys"] = opine_like_keys
    rec["veto_like_dict_keys"] = veto_like
    out["files"][label] = rec

# ------------------------------------------------------------------ interpretation
pre = out["files"].get("pre_fix_bridge_off", {})
post = out["files"].get("post_fix_bridge_off", {})
if post.get("exists") and pre.get("exists"):
    post_has_sagnac_key = post.get("mentions_sagnac_veto_key", 0) > 0
    post_has_opine = post.get("mentions_opine_info", 0) > 0
    if post_has_sagnac_key and out["files"]["post_fix_bridge_off"].get(
            "sagnac_veto_payload_keys"):
        out["notes"].append("H2/H4: the key EXISTS post-fix but the earlier scanner "
                            "reported 0 -- the scanner (or its key expectation) was "
                            "the problem, not the run.")
    elif not post_has_opine:
        out["notes"].append(
            "H3: post-fix telemetry carries NO opine_info at all, so emission is "
            "conditional on something the fix changed.")
    elif post_has_opine and not post_has_sagnac_key:
        out["notes"].append(
            "H1: opine_info IS present but WITHOUT a sagnac_veto key -> a different "
            "arm builds it (the fall-back-to-single-action branch). The macro-option "
            "branch is no longer taken.")
    else:
        out["notes"].append("inconclusive; inspect the key censuses")

receipt = Path(r"C:\Users\chan\Desktop\HENRI 7B SWARM\.worktrees\semantic-backbone"
               r"\HENRI V2\experiments\verification\payload_regression.json")
receipt.write_text(json.dumps(out, indent=1), encoding="utf-8")

for label in FILES:
    r = out["files"].get(label, {})
    if not r.get("exists"):
        print(f"{label}: MISSING")
        continue
    print(f"{label}: records={r.get('records')} "
          f"opine_info_mentions={r.get('mentions_opine_info')} "
          f"sagnac_veto_key={r.get('mentions_sagnac_veto_key')} "
          f"gate_status={r.get('mentions_gate_status')} "
          f"delta_axiom={r.get('mentions_delta_axiom')} "
          f"macro_option={r.get('mentions_macro_option')}")
    print(f"    sagnac_payload_keys={json.dumps(r.get('sagnac_veto_payload_keys'))}")
    print(f"    opine_like_keys={json.dumps(r.get('opine_like_keys'))}")
    print(f"    veto_like_dict_keys={json.dumps(r.get('veto_like_dict_keys'))}")
for n in out["notes"]:
    print("NOTE: " + n)
print("RECEIPT=" + str(receipt))
