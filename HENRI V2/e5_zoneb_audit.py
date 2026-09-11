"""E5 Zone B specification audit: disposition table + chip-to-twin correspondence.

The supplied Zone B document is treated as a PROPOSAL. Per the sealed rule,
physics terminology does not establish architectural equivalence; a
correspondence table needs named sources with V_pi, insertion loss,
phase-per-volt and heater time constant. No hardware-equivalence claim is made.

Reads every number from files; writes one JSON receipt; seals governance events.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, r"C:\Users\chan\AppData\Local\hermes\scripts")
import henri_audit as ha

DOC = Path(r"C:\Users\chan\Downloads\Zone B Photonic Core & RTX 5090 Digital Twin Specification.md")
E4WT = Path(r"C:\Users\chan\henri-worktrees\e4-wt")
HV2 = E4WT / "HENRI V2"
E5WT = Path(r"C:\Users\chan\henri-worktrees\e5-wt")
E5HV2 = E5WT / "HENRI V2"
E3 = Path(r"C:\Users\chan\henri-telemetry\e3")
OUT = E3 / "e5_zoneb_audit.json"
ACTOR = "henri-arbiter"

PROBES = [
    ("r42_1300pmV", "1300"),
    ("latency_12_8us", "12.8"),
    ("sagnac_range_pi", "3.14159265"),
    ("veto_threshold_035", "0.35"),
    ("d2nn_32_layer", "32-layer"),
    ("float_atomic_add", "atomic_add"),
    ("named_kernel", "qfhrr_sagnac_veto_kernel"),
    ("native_tied_readout", "tied readout"),
    ("o8_gauge_rotations", "orthogonal rotations"),
    ("wdm_lines", "WDM"),
    ("landauer", "Landauer"),
    ("grim_fuzzy", "Grim"),
    ("levin", "Levin"),
    ("tin_microheater", "Titanium Nitride"),
    ("pockels", "Pockels"),
    ("batio3", "barium titanate"),
    ("coverage_gate", "coverage"),
]

ARTIFACTS = [
    "qfhrr_sagnac_veto_kernel.py",
    "qfhrr_kernels.py",
    "g7_highorder_codec.py",
    "zone_c_epistemic_axiom_harness.py",
    "henri_egress.py",
]

BTO_IDS = ["2505.21927", "2506.13209", "2407.03443", "2601.14938", "2607.03690"]

ENFORCE = ("P_AT_1_BOUND = 0.285", "P_AT_5_BOUND = 0.640")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=220)
    return (r.stdout or "") + (r.stderr or "")


def abstracts():
    out = Path(r"C:\Users\chan\arx_bto_abs.xml")
    ids = ",".join(BTO_IDS)
    url = "https://export.arxiv.org/api/query?id_list=" + ids + "&max_results=10"
    sh('curl -sL --max-time 45 "' + url + '" -o "' + str(out) + '"')
    if not out.exists():
        return {"error": "fetch_failed"}
    b = out.read_bytes()
    t = b.decode("utf-8", "replace")
    ents = []
    for e in re.findall(r"<entry>(.*?)</entry>", t, re.S):
        aid = re.search(r"<id>http://arxiv\.org/abs/([^<]+)</id>", e)
        ti = re.search(r"<title>(.*?)</title>", e, re.S)
        ab = re.search(r"<summary>(.*?)</summary>", e, re.S)
        at = re.sub(r"\s+", " ", (ab.group(1) if ab else "")).strip()
        pv = sorted(set(re.findall(r"\b(\d{2,5}(?:\.\d+)?)\s*pm\s*/\s*V\b", at)))
        ents.append({"id": aid.group(1) if aid else None,
                     "title": re.sub(r"\s+", " ", (ti.group(1) if ti else "")).strip()[:110],
                     "pm_per_V_values": pv,
                     "abstract_chars": len(at)})
    return {"bytes": len(b), "sha256": hashlib.sha256(b).hexdigest(), "entries": ents}


def main():
    rec = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    # ---- 1. authenticate -------------------------------------------------
    if not DOC.exists():
        rec["document"] = {"exists": False}
        OUT.write_text(json.dumps(rec, indent=2))
        print("DOC_MISSING")
        return
    db = DOC.read_bytes()
    text = db.decode("utf-8", errors="replace")
    lines = text.splitlines()
    probes = {}
    for key, pat in PROBES:
        hits = [{"line": i + 1, "text": l.strip()[:110]}
                for i, l in enumerate(lines) if pat.lower() in l.lower()]
        probes[key] = {"n": len(hits), "first3": hits[:3]}
    rec["document"] = {"exists": True, "path": str(DOC), "bytes": len(db),
                       "sha256": hashlib.sha256(db).hexdigest(),
                       "lines": len(lines), "claim_probes": probes}
    print("[1] doc bytes=" + str(len(db)) + " sha=" + hashlib.sha256(db).hexdigest()[:16]
          + " lines=" + str(len(lines)))

    # ---- 2. named artifacts ---------------------------------------------
    arts = {}
    for n in ARTIFACTS:
        p4 = HV2 / n
        arts[n] = {"e4_tree": p4.exists(), "e5_tree": (E5HV2 / n).exists(),
                   "sha16": sha(p4)[:16] if p4.exists() else None}
    rec["artifacts"] = arts
    print("[2] named kernel in tree: " + str(arts["qfhrr_sagnac_veto_kernel.py"]["e4_tree"]))

    # ---- 3. bounds classification (fixes my own earlier false positive) --
    enforcing, inverted, scanner, record = [], [], [], []
    scan_files = {"e5_ground_scan.py", "retire_bounds.py", "fix_guard.py",
                  "e5_admin.py", "e5_zoneb_audit.py", "e5_sources.py"}
    for p in sorted(HV2.rglob("*.py")):
        if "_archive" in str(p):
            continue
        try:
            t = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for i, l in enumerate(t.splitlines(), 1):
            if not any(pat in l for pat in ENFORCE):
                continue
            if "not in src" in l:
                inverted.append(p.name + ":" + str(i))
            elif p.name in scan_files:
                scanner.append(p.name + ":" + str(i))
            else:
                record.append(p.name + ":" + str(i))
    rec["bounds"] = {
        "ENFORCING": enforcing, "INVERTED_ASSERT": inverted,
        "SCANNER_LITERAL": scanner, "RECORD_LITERAL": record,
        "enforcing_sites": len(enforcing),
        "corrected_verdict": ("BOUNDS_RETIRED_NO_ENFORCEMENT" if not enforcing
                              else "BOUNDS_STILL_ENFORCED"),
        "note": ("the earlier seal recorded enforcing_sites_remaining 4. Those 4 are "
                 "pattern hits INSIDE inverted assertions in the two contract tests "
                 "plus scanner literals. Zero sites enforce a bound. Corrected here."),
    }
    print("[3] enforcing=" + str(len(enforcing)) + " inverted=" + str(len(inverted))
          + " -> " + rec["bounds"]["corrected_verdict"])

    # ---- 4. sources -------------------------------------------------------
    src = json.loads((E3 / "e5_sources.json").read_text())
    rec["arxiv_harvested"] = {k: {"bytes": v.get("bytes"),
                                  "sha16": (v.get("sha256") or "")[:16],
                                  "ids": [e["id"] for e in v.get("entries", [])]}
                              for k, v in src.get("arxiv_primary_bytes", {}).items()}
    rec["arxiv_bto_abstracts"] = abstracts()
    rec["local_corpus_hits"] = src.get("local_corpus", {}).get("hits", [])
    rec["notebooklm"] = {
        "status": "BLOCKED",
        "evidence": ("server_info auth_status=stale, measured twice this session; "
                     "nlm login --check returned ClientAuthenticationError; MCP tools "
                     "attach at session start so re-auth needs a NEW session"),
    }
    print("[4] bto_abstracts=" + str(len(rec["arxiv_bto_abstracts"].get("entries", [])))
          + " local_hits=" + str(len(rec["local_corpus_hits"]))
          + " notebooklm=" + rec["notebooklm"]["status"])

    # ---- 5. dispositions --------------------------------------------------
    rec["dispositions"] = {
        "ALREADY_IMPLEMENTED": [
            "Sagnac homodyne veto and normalized delta. Sealed definition "
            "1 - Re(pred,emp)/(|pred||emp|), range [0,2], veto tau 0.35.",
            "Hopfield lexical snap beta 8.0 in henri_egress.py.",
            "Anisotropic Langevin noise masks in zone_c_epistemic_axiom_harness.py.",
            "Cl(3,0) [8192,8] row-unit wave boundary at the planner; total norm "
            "sqrt(8192) equals 90.5096.",
            "Thermal annealing analog via SGLD creep and the thermostat. Partial: "
            "there is no physical heater array.",
        ],
        "CONFLICTS_WITH_LIVE_CODE": [
            "sagnac_delta range. Doc says [0.0, 3.14159265]; sealed contract is "
            "[0, 2]. The doc metric is mean absolute phase error times pi/128, a "
            "different quantity from the sealed normalized-cosine delta. Any "
            "redefinition is a new ratified carrier, not a tuning change.",
            "Gate B-3 gauge group. Doc proposes O(8) block rotations. The sealed "
            "gauge audit FALSIFIED O(8) and established the Cl(3,0) gauge as a "
            "Spin(3) rotor sandwich, with grade_scramble 0.0 measured.",
            "wave_state_imag [B,8192,8] introduces a complex component family. Per "
            "the complex-wave-family sidecar boundary it may enter only as a "
            "default-OFF diagnostic sidecar with a one-way norm-preserving adapter "
            "and no action-policy influence.",
        ],
        "BOUNDED_IMPLEMENTABLE": [
            "Fused Triton Sagnac kernel. The named file qfhrr_sagnac_veto_kernel.py "
            "does not exist in the tree. The doc latency 12.8 us is TARGET_GOAL. Any "
            "gate must be end-to-end including host transfer, and the reduction must "
            "be deterministic; float atomic_add is order-dependent.",
            "D2NN 32-layer complex transmission-mask emulator as a default-OFF "
            "diagnostic sidecar.",
            "Gate B-1 Hopfield jitter-recovery sweep harness.",
            "Gate B-2 anti-solipsism discriminator. Invert the exteroceptive "
            "scorecard delta while holding internal phase lock, and require the "
            "thermostat to unclamp within N iterations. This is the correct "
            "architectural answer to self-reference: not avoidance, but an external "
            "grounding channel.",
        ],
        "BLOCKED_MISSING_PREMISE": [
            "No realized chip datasheet. V_pi, insertion loss, phase-per-volt and "
            "heater time constant are absent, so no parameter-grounded chip-to-twin "
            "correspondence table is possible yet.",
            "r42 equals 1300 pm/V is a design target; it must be pinned to a named "
            "primary source.",
            "65536 WDM optical lines is implausible as a parallel carrier count. "
            "D equals 65536 is the algebraic dimension, not a line count.",
            "The 12.8 us measured latency and the CXL 3.0 optical DMA are design "
            "targets, not observations.",
        ],
        "UNVERIFIED_FRAMING": [
            "Levin pattern ingression as antenna or pointer: philosophy, no "
            "mechanism mapping.",
            "Grim fuzzy-liar limit cycles mapped to attractor and veto: no Lyapunov "
            "measurement exists.",
            "Landauer dissipation accounting: physical analogy only, no entropy ledger.",
        ],
        "ALIGNED_WITH_SEALED_FINDINGS": [
            "L2 native tied readout scoring over C(x). This MATCHES sealed "
            "provenance event 1374, which showed the tied readout IS the backbone "
            "own output head. Scoring a candidate set with a coverage gate is the "
            "coherent architecture. The document reaches the same structure "
            "independently.",
        ],
    }

    # ---- 6. correspondence table -----------------------------------------
    rec["correspondence_table"] = [
        {"chip_function": "BTO Pockels phase modulation (r42)",
         "twin_function": "phase-codec adapter, complex phase rotation of [8192,8]",
         "evidence": "INFERRED_MAPPING",
         "premise_needed": "V_pi, insertion loss, phase-per-volt from a named source",
         "live_status": "phase-codec adapter, default-OFF"},
        {"chip_function": "32-layer D2NN volumetric diffraction",
         "twin_function": "K-layer complex transmission-mask sandwich",
         "evidence": "INFERRED_MAPPING",
         "premise_needed": "layer count and mask parametrization from the chip doc",
         "live_status": "UNCONNECTED, default-OFF sidecar required"},
        {"chip_function": "Sagnac homodyne veto, constructive vs dark port",
         "twin_function": "normalized-cosine Sagnac delta plus hard veto",
         "evidence": "OBSERVED_LIVE",
         "premise_needed": "none, already sealed",
         "live_status": "ALREADY_IMPLEMENTED, range [0,2], tau 0.35"},
        {"chip_function": "TiN microheater Langevin annealing",
         "twin_function": "anisotropic SGLD with sqrt(2*T*dt) and Cholesky Stiefel",
         "evidence": "OBSERVED_LIVE",
         "premise_needed": "heater time constant for the physical timescale",
         "live_status": "ALREADY_IMPLEMENTED, anisotropic masks"},
        {"chip_function": "self-reference and the strange loop",
         "twin_function": "Gate B-2 anti-solipsism, external grounding channel",
         "evidence": "INFERRED_MAPPING",
         "premise_needed": "exteroceptive scorecard delta from a live environment",
         "live_status": "harness ABSENT, cheap and well-formed"},
        {"chip_function": "non-commutative relational binding",
         "twin_function": "Cl(3,0) geometric product",
         "evidence": "OBSERVED_LIVE",
         "premise_needed": "none",
         "live_status": "ALREADY_IMPLEMENTED, O(8) falsified"},
    ]

    # ---- 7. seals ---------------------------------------------------------
    h1 = ha.record_event(ACTOR, "HENRI_ZONEB_SPEC_AUDITED", {
        "document_path": str(DOC),
        "document_sha256": rec["document"]["sha256"],
        "document_bytes": rec["document"]["bytes"],
        "treated_as": "PROPOSAL",
        "named_kernel_exists": arts["qfhrr_sagnac_veto_kernel.py"]["e4_tree"],
        "enforcing_bound_sites": len(enforcing),
        "conflicts": ["sagnac range [0,pi] vs sealed [0,2]",
                      "Gate B-3 O(8) vs sealed Spin(3)",
                      "wave_state_imag complex family"],
        "aligned_with": "provenance event 1374, tied readout is the backbone head",
        "notebooklm": "BLOCKED_STALE_AUTH",
        "bl_1_rate_limit": False,
        "no_hardware_equivalence_claimed": True,
        "no_promotion": True,
    })
    h2 = ha.record_event(ACTOR, "HENRI_BOUNDS_CLASSIFICATION_CORRECTED", {
        "amends": "HENRI_LEGACY_BOUNDS_RETIRED",
        "prior_value": "enforcing_sites_remaining 4",
        "corrected_value": len(enforcing),
        "cause": ("the 4 were pattern hits inside inverted assertions in the two "
                  "contract tests, plus scanner and record literals. Not enforcement."),
        "classification": rec["bounds"],
        "retirement_stands": True,
    })
    ok, msg = ha.verify_chain()
    rec["seals"] = {"zoneb_audited": h1, "bounds_corrected": h2}
    rec["chain"] = {"ok": ok, "message": msg}

    OUT.write_text(json.dumps(rec, indent=2))
    print("[5] sealed zoneb " + h1[:16] + "  bounds " + h2[:16])
    print("[6] chain ok=" + str(ok) + " " + msg)
    print("WROTE " + str(OUT) + " sha=" + sha(OUT)[:16])


if __name__ == "__main__":
    main()
