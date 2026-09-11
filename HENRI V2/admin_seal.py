"""Seal the three user-approved administrative decisions (2026-09-11).

Honest framing per the sealed provenance finding (#1374):
  The section-2 amendment RATIFIES a state that already holds -- the pinned
  "teacher" IS the backbone's own tied output embedding. It is recorded as a
  ratification-with-disclosure, NOT as the introduction of a new dependency.
  The E4c self-reference confound (#1375) is carried in the same seal.

Every value is read from the receipts on disk at seal time. Nothing transcribed.
Writes receipt; prints only after the chain re-verifies.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\chan\AppData\Local\hermes\scripts")
import henri_audit as ha  # noqa: E402

E3 = Path(r"C:\Users\chan\henri-telemetry\e3")
E4WT = Path(r"C:\Users\chan\henri-worktrees\e4-wt")
ACTOR = "henri-arbiter"


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(n: str) -> dict:
    return json.loads((E3 / n).read_text(encoding="utf-8"))


def main() -> None:
    e4a = load("e4a_construct_audit.json")
    e4b = load("e4b_verdict.json")
    e4c = load("e4c_verdict_corrected.json")
    e4cbis = load("e4c_bis.json")
    prov = load("prov.json")
    ret = load("bounds_retirement_receipt.json")
    fix = load("bounds_fix_receipt.json")

    def git(*a):
        return subprocess.run(["git", *a], cwd=str(E4WT), capture_output=True,
                              text=True, timeout=60).stdout.strip()

    head = git("rev-parse", "HEAD")
    main_sha = git("rev-parse", "origin/main")
    remote_e4 = git("ls-remote", "origin", "refs/heads/carrier/e4-construct").split()[0:1]
    assert main_sha == "10f5f23", f"MAIN_MOVED {main_sha}"

    seals = []

    # ---------- 1. SECTION 2 RATIFIED (ratification-with-disclosure) --------
    h1 = ha.record_event(ACTOR, "HENRI_CONTRACT_AMENDMENT_2_RATIFIED", {
        "amendment": "HENRI_UKA_design_20260910.md section 2",
        "nature": "RATIFICATION_WITH_DISCLOSURE",
        "approved_by": "user directive 2026-09-11",
        "what_is_ratified": (
            "token-level scoring is supplied by a frozen, revision-pinned, "
            "contamination-reviewed pretrained backbone: eval-only, zero "
            "trainable parameters, default-OFF (HENRI_BACKBONE=1), fail-closed "
            "on shard SHA mismatch; the wave channel keeps composition / "
            "retrieval / veto roles and is NOT asked to carry the token "
            "distribution"),
        "disclosure_1_already_the_case": (
            "this ratifies a state that ALREADY holds. prov.json shows the "
            "pinned teacher IS the backbone's own tied output embedding "
            "(fp32 canonical SHA identical for both tensors; "
            "tie_word_embeddings=True; no lm_head.weight in the shard). The "
            "E1-E3 'frozen independent teacher' framing was incorrect."),
        "disclosure_2_e4c_confound": (
            "because the teacher is the backbone's own head, E4c arm b (the "
            "'frozen tied readout') was the pretrained classifier, not a HENRI "
            "control. E4c is SELF-REFERENTIAL; its corrected verdict is "
            "E4C_FALSIFIED_TRAINING_DEGRADES__BACKBONE_DOMINANT and the confound "
            "is sealed separately (#1375). No HENRI capability may be claimed "
            "from E4c."),
        "pinned_artifacts": {
            "backbone_repo": "Qwen/Qwen2.5-0.5B",
            "revision": "060db6499f32faf8b98477b0a26969ef7d8b9987",
            "shard_sha256": "88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342",
            "shard_bytes": 988097824,
        },
        "reference": "henri-research/references/world-knowledge-boundary.md (approved re-scope)",
        "provenance_receipt_sha256": sha(E3 / "prov.json"),
        "contamination_status": "BLOCKED - calibrated scan still required; all splits CONDITIONAL",
        "no_promotion": True, "no_main_change": True,
    })
    seals.append(("HENRI_CONTRACT_AMENDMENT_2_RATIFIED", h1))

    # ---------- 2. LEGACY BOUNDS RETIRED (already executed + tests pass) ----
    h2 = ha.record_event(ACTOR, "HENRI_LEGACY_BOUNDS_RETIRED", {
        "retired": {"p_at_1": 0.285, "p_at_5": 0.640},
        "approved_by": "user directive 2026-09-11",
        "reason": (
            "E4a measured, on a fresh never-used split, the context-free "
            "marginal at P@1=0.440 and the strongest trivial baseline at "
            "P@1=0.483. The retired bound 0.285 sits BELOW the trivial baseline, "
            "so that half of the gate is passed by a constant predictor."),
        "evidence": {
            "e4a_receipt_sha256": sha(E3 / "e4a_construct_audit.json"),
            "marginal_p1": e4a["constructs"]["C1_sentence_window"]["marginal_baseline"]["p1"],
            "strongest_trivial_p1": e4a["constructs"]["C1_sentence_window"]["best_trivial_baseline"]["p1"],
        },
        "successor_bounds": e4a["registered_bounds"],
        "execution": {
            "status": "EXECUTED_AND_VERIFIED",
            "mechanism": (
                "hard-coded module constants REPLACED by a lazy per-construct "
                "loader (load_registered_bounds) that reads the sealed E4a "
                "receipt and fails closed if the bound is not strictly above the "
                "trivial baseline; resolution is lazy (PEP 562 __getattr__) so "
                "importing the module never fails closed"),
            "files_changed": [
                "HENRI V2/e3_calibrate.py",
                "HENRI V2/verify_egress_closed_loop.py",
                "HENRI V2/tests/contract/test_e3_egress_reform.py",
                "HENRI V2/tests/contract/test_gate31_p1pk.py",
            ],
            "contract_tests_inverted_and_pass": "16 passed (pytest, local)",
            "resolved_bounds_at_import": {"p_at_1": 0.533, "p_at_5": 0.636},
            "retirement_receipt_sha256": sha(E3 / "bounds_retirement_receipt.json"),
            "fix_receipt_sha256": sha(E3 / "bounds_fix_receipt.json"),
            "verdict": ret.get("verdict"), "fix_verdict": fix.get("VERDICT"),
        },
        "scope": {
            "sealed_prereg_md": "NOT edited - historical sealed records",
            "audit_receipts_json": "NOT edited",
            "ci_workflows": ("repo has .github/workflows/docker-publish.yml only; "
                             "it carries no bounds -> nothing to change there"),
            "remaining_live_hardcode": "none (verified by scan)",
        },
        "disclosed_weakness": (
            "C2 (token-stream) CE bound 11.831 beats uniform 11.931 by only "
            "0.10 nats on this corpus; the C2 P@1 bound (0.167 vs trivial 0.112) "
            "is the meaningful half."),
    })
    seals.append(("HENRI_LEGACY_BOUNDS_RETIRED", h2))

    # ---------- 3. E4 CLOSED / E5 OPENED -----------------------------------
    h3 = ha.record_event(ACTOR, "HENRI_E4_CLOSED_E5_OPENED", {
        "closed_carrier": "carrier/e4-construct",
        "closed_as": "COMPLETED_DIAGNOSTIC_MILESTONE",
        "closed_state": {"head": head, "remote_head": remote_e4,
                         "main_untouched": main_sha},
        "e4_receipts": {
            "e4a_bounds": sha(E3 / "e4a_construct_audit.json"),
            "e4b_position_codec": sha(E3 / "e4b_verdict.json"),
            "e4c_corrected": sha(E3 / "e4c_verdict_corrected.json"),
            "e4c_bis": sha(E3 / "e4c_bis.json"),
            "provenance": sha(E3 / "prov.json"),
        },
        "e4_verdicts": {
            "e4a": e4a["verdict"],
            "e4b": e4b["verdict"],
            "e4c": e4c["CORRECTED_VERDICT"],
            "e4c_bis": "ATTRIB_FUND",
            "e4d_e4e": "BLOCKED (own kill criteria)",
        },
        "opened_carrier": "carrier/e5-wave-superposition",
        "e5_scope_HONEST": {
            "what_e4b_actually_measured": (
                "a RESERVED DISJOINT position channel of 2048 rows out of 8192 "
                "(25% of the carrier) carrying explicit end-distance boundary "
                "markers (e:, d1:, d2:, d3:); zero trainable parameters; "
                "position-region last-word-to-front cosine 0.0332 vs the "
                "g7 bag codec's 0.940; terminal recovery 32/32"),
            "what_it_did_NOT_measure": (
                "superposition of solutions, holographic search, or any "
                "analog/hardware-domain behaviour. The name "
                "'wave-superposition' is the user-directed branch name; the "
                "measured object is a position-identifiable boundary-marker "
                "channel. Any superposition claim requires its own measurement."),
            "naming_caveat_sealed": True,
            "bar": "backbone oracle (C2 oracle P@1 0.433), not the retired bounds",
            "attribution_rule": ("arm-b-class frozen readouts are EXCLUDED from "
                                 "attribution - provenance proved they are the "
                                 "backbone's own head"),
            "zero_trainable_first": True,
        },
        "digital_twin_directive": {
            "user_description": ("Zone B optical core as a physical verifying "
                                 "adversarial network: barium titanate silicon "
                                 "photonic deep diffractive neural network + "
                                 "Sagnac interferometry + thermodynamic Pockels "
                                 "modulation + microvoltage heaters; map the "
                                 "realized chip's function set onto the RTX 5090 "
                                 "digital twin"),
            "status": "RESEARCH_BLOCKED_NO_CHIP_ARCHITECTURE_DOC",
            "reason": ("no realized-chip architecture document, datasheet, or "
                       "capability set has been supplied, and the live tree "
                       "contains NO photonic/Pockels/Sagnac-core artifact "
                       "(grep: 4 incidental hits, all under _archive/). "
                       "Physics terminology does not establish architectural "
                       "equivalence; a mapping table needs named sources with "
                       "V_pi, insertion loss, phase/volt, heater time constant."),
            "action": ("harvest named primary sources first; emit a parameter-"
                       "bounded correspondence table with per-row evidence "
                       "class (OBSERVED source / INFERRED mapping / UNCONNECTED); "
                       "no hardware-equivalence claim from structural analogy"),
        },
        "approved_by": "user directive 2026-09-11",
        "no_promotion": True, "no_main_change": True,
    })
    seals.append(("HENRI_E4_CLOSED_E5_OPENED", h3))

    ok, msg = ha.verify_chain()
    for a, h in seals:
        print(f"sealed {a}  #{h[:16]}")
    print(f"chain {'OK' if ok else 'FAIL'} :: {msg}")


if __name__ == "__main__":
    main()
