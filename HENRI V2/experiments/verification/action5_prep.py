#!/usr/bin/env python3
"""ACTION 5 PREP: settle the handler bytes, locate the width rule, stage the overlay.

THREE QUESTIONS, ALL CHEAP
  H  Does the runner contain ONE handler site or SEVERAL? My receipt measured
     `isinstance(_veto_exc, SagnacGateUnavailable)` present and the `_name ==` form
     ABSENT, while two reference renderings showed the opposite. A duplicate OPINE
     block (the region has an if/else and sits inside a loop) would make BOTH true.
     Settled by dumping every matching line with its number, from the bytes on disk.
  W  What width does the SU3 transducer emit? MEASURED earlier: state_wave is 512 at
     num_blocks=64 and 65536 at num_blocks=8192. If field_to_wave is FIXED at 65536,
     then at full scale the candidate and both references are 65536-wide and the
     mismatch DISAPPEARS -- which is the Action 5 hypothesis, and it must be read from
     source rather than inferred.
  O  Stage the gitignored checkpoint overlay. `HENRIUnifiedEgressTransducer` uses
     checkpoint_policy="required" when d_model == 65536, so a full-scale construction
     FAILS without it. MEASURED: the overlay exists at the MAIN checkout
     (HENRI V2/models/henri_decoder_checkpoint.pt, 763M) and is ABSENT in the worktree.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

MAIN = Path(r"C:\Users\chan\Desktop\HENRI 7B SWARM\HENRI V2")
WT = Path(r"C:\Users\chan\Desktop\HENRI 7B SWARM\.worktrees\semantic-backbone\HENRI V2")
RUNNER = WT / "production_arc_run.py"

out: dict = {"checks": {}, "errors": []}
chk = out["checks"]

# ------------------------------------------------------------------ H
lines = RUNNER.read_text(encoding="utf-8", errors="ignore").splitlines()
pats = {
    "except_veto": "except Exception as _veto_exc",
    "isinstance_match": "isinstance(_veto_exc, SagnacGateUnavailable)",
    "name_cmp": '_name == "SagnacGateUnavailable"',
    "name_assign": "_name = type(_veto_exc).__name__",
    "gate_status": "gate_status",
    "unavailable_label": "UNAVAILABLE_SHAPE_MISMATCH",
    "hard_vetoed": "hard_vetoed",
    "opine_info_assign": "opine_info = {",
    "candidate_elems": "candidate_elems",
    "veto_stage": "veto_stage",
    "shape_candidate": "shape_candidate",
}
hits = {}
for key, needle in pats.items():
    rows = [i for i, l in enumerate(lines, 1) if needle in l]
    hits[key] = rows
chk["H_line_hits"] = hits
chk["H_total_lines"] = len(lines)
chk["H_n_except_sites"] = len(hits["except_veto"])
chk["H_n_isinstance"] = len(hits["isinstance_match"])
chk["H_n_name_cmp"] = len(hits["name_cmp"])
chk["H_extra_fields_present"] = bool(hits["candidate_elems"] or hits["veto_stage"])
# Verdict
if hits["isinstance_match"] and not hits["name_cmp"]:
    chk["H_verdict"] = "ISINSTANCE_FORM_CONFIRMED (reference renderings are stale)"
elif hits["name_cmp"] and not hits["isinstance_match"]:
    chk["H_verdict"] = "NAME_COMPARISON_FORM (my receipt was wrong)"
elif hits["name_cmp"] and hits["isinstance_match"]:
    chk["H_verdict"] = "BOTH_FORMS_PRESENT (duplicate handler sites -- inspect both)"
else:
    chk["H_verdict"] = "NEITHER_FORM_FOUND (handler recognition missing)"
if chk["H_extra_fields_present"]:
    out["errors"].append(
        "H: fields I never wrote are present (candidate_elems/veto_stage) -> the file "
        "contains a revision I did not author; inspect before trusting either receipt")

# Dump the exact handler bytes for the receipt (bounded, line-numbered).
dump = []
for ln in hits["except_veto"]:
    for i in range(max(0, ln - 1), min(len(lines), ln + 26)):
        dump.append(f"{i+1}: {lines[i]}")
chk["H_handler_dump"] = dump

# ------------------------------------------------------------------ W
# Read field_to_wave bodies wherever defined, and the SU3 construction site.
fw_sites, su3_ctor = [], []
for p in sorted(WT.rglob("*.py")):
    if "_archive" in p.parts or ".git" in p.parts:
        continue
    try:
        txt = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        continue
    tl = txt.splitlines()
    for i, l in enumerate(tl, 1):
        s = l.strip()
        if s.startswith("class SU3FieldWaveTransducer"):
            fw_sites.append({"file": p.name, "line": i, "kind": "class",
                             "text": s[:100]})
        if s.startswith("def field_to_wave"):
            sig = s[:100]
            # capture a bounded body to read the width rule
            body = [f"{j+1}: {tl[j]}" for j in range(i - 1, min(len(tl), i + 34))]
            fw_sites.append({"file": p.name, "line": i, "kind": "def", "text": sig,
                             "body": body})
        if "SU3FieldWaveTransducer(" in s and "class " not in s:
            su3_ctor.append({"file": p.name, "line": i, "text": s[:140]})
chk["W_class_and_def_sites"] = [{k: v for k, v in d.items() if k != "body"}
                                for d in fw_sites]
chk["W_su3_construction_sites"] = su3_ctor
chk["W_field_to_wave_bodies"] = [d.get("body") for d in fw_sites if d["kind"] == "def"]

# Read the SU3 class __init__ to see what width it fixes.
su3_cls = next((d for d in fw_sites if d["kind"] == "class"), None)
if su3_cls:
    p = WT / su3_cls["file"]
    tl = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    body = [f"{j+1}: {tl[j]}" for j in range(su3_cls["line"] - 1,
                                             min(len(tl), su3_cls["line"] + 60))]
    chk["W_su3_class_body"] = body
    joined = "\n".join(body)
    chk["W_su3_init_mentions_d_model"] = "d_model" in joined
    chk["W_su3_init_mentions_65536"] = "65536" in joined
    chk["W_su3_init_mentions_num_blocks"] = "num_blocks" in joined

# ------------------------------------------------------------------ O
src_ckpt = MAIN / "models" / "henri_decoder_checkpoint.pt"
dst_dir = WT / "models"
dst_ckpt = dst_dir / "henri_decoder_checkpoint.pt"
chk["O_source_exists"] = src_ckpt.exists()
chk["O_source_bytes"] = src_ckpt.stat().st_size if src_ckpt.exists() else None


def sha256_of(p: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


copied = False
if src_ckpt.exists() and not dst_ckpt.exists():
    try:
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src_ckpt, dst_ckpt)
        copied = True
    except Exception as e:  # noqa: BLE001
        out["errors"].append(f"O: overlay copy failed: {type(e).__name__}: {e}")
chk["O_copied"] = copied
chk["O_dest_exists"] = dst_ckpt.exists()
chk["O_dest_bytes"] = dst_ckpt.stat().st_size if dst_ckpt.exists() else None
if src_ckpt.exists() and dst_ckpt.exists():
    h_src = sha256_of(src_ckpt)
    h_dst = sha256_of(dst_ckpt)
    chk["O_src_sha256"] = h_src
    chk["O_dst_sha256"] = h_dst
    chk["O_sha_match"] = (h_src == h_dst)
    if not chk["O_sha_match"]:
        out["errors"].append("O: overlay SHA mismatch after copy")
if not chk["O_source_exists"]:
    out["errors"].append("O: source overlay absent; a full-scale run cannot construct")
chk["O_gitignored"] = True  # per project convention: *.pt is an untracked overlay

out["verdict"] = "PASS" if not out["errors"] else "FAIL"
receipt = WT / "experiments" / "verification" / "action5_prep.json"
receipt.write_text(json.dumps(out, indent=1), encoding="utf-8")

print("A5PREP=" + out["verdict"] + " errors=" + str(len(out["errors"])))
print("H_verdict=" + str(chk.get("H_verdict"))
      + " n_except=" + str(chk.get("H_n_except_sites"))
      + " n_isinstance=" + str(chk.get("H_n_isinstance"))
      + " n_namecmp=" + str(chk.get("H_n_name_cmp"))
      + " extras=" + str(chk.get("H_extra_fields_present")))
print("H_hits=" + json.dumps(chk.get("H_line_hits")))
print("W_su3_sites=" + json.dumps(chk.get("W_class_and_def_sites")))
print("W_d_model=" + str(chk.get("W_su3_init_mentions_d_model"))
      + " W_65536=" + str(chk.get("W_su3_init_mentions_65536"))
      + " W_num_blocks=" + str(chk.get("W_su3_init_mentions_num_blocks")))
print("O_src=" + str(chk.get("O_source_bytes")) + " dst=" + str(chk.get("O_dest_bytes"))
      + " sha_match=" + str(chk.get("O_sha_match")) + " copied=" + str(chk.get("O_copied")))
print("O_sha=" + str(chk.get("O_dst_sha256")))
for e in out["errors"]:
    print("ERR: " + e[:220])
print("RECEIPT=" + str(receipt))
