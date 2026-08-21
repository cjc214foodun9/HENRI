"""Deterministic spec-claim extraction + live-code reconciliation.

Reads the two inbox spec copies under experiments/verification/specs_inbox_20260821/,
categorizes claims, greps the live tree for the stated mechanisms, and emits a
JSON reconciliation report. Pure stdlib; zero GPU; zero LLM.

Categories: functional_requirements, proposed_mechanisms, benchmark_requirements,
equations_contracts, capability_claims, conflicts_with_live.
"""
import json
import os
import re
import sys

ROOT = r"C:\Users\chan\henri-worktrees\accuracy-main"
SPEC_DIR = os.path.join(ROOT, "experiments", "verification", "specs_inbox_20260821")
SPECS = [
    "HENRI_Epistemic_Engram___Wave-Mechanistic_Entropy_Engine_Specification.md",
    "HENRI_Model_Benchmark_Verification___Holographic_VLA_Architectural_Realization_Specification.md",
]

# (spec symbol, category, live anchor pattern) — reconciliation surface
ANCHORS = [
    ("zone_c_ast_engrams/zone_c_action_engrams", "proposed_mechanisms", r"zone_c_ast_engrams|zone_c_action_engrams"),
    ("zone_c_engrams", "functional_requirements", r"CREATE TABLE IF NOT EXISTS zone_c_engrams"),
    ("Delta_Sagnac>=0.10 veto", "equations_contracts", r"tau_veto: float = 0\.35"),
    ("O-VSA fractional binding", "proposed_mechanisms", r"o_vsa_ingress_tokenizer"),
    ("Dual EDMD / R-EDMD", "proposed_mechanisms", r"LowRankCoupledTransition|RecursiveDualEDMD"),
    ("16-expert Kuramoto syncytium", "proposed_mechanisms", r"class HenriSwarmOrchestrator"),
    ("SGLD anisotropic Langevin", "proposed_mechanisms", r"adapt_in_context_sgld_wave"),
    ("qFHRR Triton LUT kernel", "proposed_mechanisms", r"qfhrr_batch_similarity_triton"),
    ("HumanEval 100% (164/164)", "benchmark_requirements", r"humaneval"),
    ("MBPP 100% (257/257)", "benchmark_requirements", r"mbpp"),
    ("MMLU-Pro 81.6%", "benchmark_requirements", r"mmlu"),
    ("ARC-AGI-3 62.75%", "benchmark_requirements", r"arc_agi"),
    ("I(Psi;Y)>0.85 bits", "equations_contracts", r"mutual_information|mutual info"),
    ("Retrieval <=2.0ms/12.5us", "equations_contracts", r"bridge"),
    ("VRAM ceiling 12.5GB", "equations_contracts", r"12\.5"),
    ("W_task cross-covariance assembly", "proposed_mechanisms", r"W_task|task_functor|HolographicTaskFunctorCompiler"),
]


def categorize(path: str) -> dict:
    cats = {k: [] for k in
            ["functional_requirements", "proposed_mechanisms", "benchmark_requirements",
             "equations_contracts", "capability_claims", "unsupported"]}
    with open(path, encoding="utf-8") as fh:
        lines = fh.readlines()
    for i, ln in enumerate(lines, 1):
        s = ln.strip()
        if not s or s.startswith("#") and len(s) < 4:
            continue
        low = s.lower()
        if "must" in low or "mandatory" in low or "required" in low or "requires" in low:
            cats["functional_requirements"].append({"line": i, "text": s[:160]})
        elif "falsif" in low or "gate" in low or "threshold" in low or "verif" in low or "limit" in low:
            cats["equations_contracts"].append({"line": i, "text": s[:160]})
        elif "benchmark" in low or "scorecard" in low or "dataset" in low or "evaluator" in low:
            cats["benchmark_requirements"].append({"line": i, "text": s[:160]})
        elif "achieves" in low or "100%" in low or "verified" in low or "guarantee" in low:
            cats["capability_claims"].append({"line": i, "text": s[:160]})
        elif any(k in low for k in ["mechanism", "pipeline", "operator", "kernel", "binding", "egress", "ingress"]):
            cats["proposed_mechanisms"].append({"line": i, "text": s[:160]})
        elif "|" in s and "---" not in s:
            cats["equations_contracts"].append({"line": i, "text": s[:160]})
    return cats


def reconcile(spec: str, anchor, pattern: str) -> dict:
    hits = []
    for dirpath, _dirs, files in os.walk(os.path.join(ROOT, "HENRI V2")):
        if "_archive" in dirpath or ".git" in dirpath:
            continue
        for fn in files:
            if not (fn.endswith(".py") or fn.endswith(".sql")):
                continue
            p = os.path.join(dirpath, fn)
            try:
                with open(p, encoding="utf-8", errors="ignore") as fh:
                    for j, ln in enumerate(fh, 1):
                        if re.search(pattern, ln, re.IGNORECASE):
                            hits.append(f"{os.path.relpath(p, ROOT)}:{j}")
                            if len(hits) >= 5:
                                return {"spec_claim": anchor, "in_spec": spec, "live_hits": hits,
                                        "status": "LIVE_SYMBOL_FOUND"}
            except OSError:
                continue
    return {"spec_claim": anchor, "in_spec": spec, "live_hits": hits,
            "status": "NO_LIVE_SYMBOL" if not hits else "LIVE_SYMBOL_FOUND"}


def main() -> int:
    report = {"specs": {}, "reconciliation": []}
    for spec in SPECS:
        p = os.path.join(SPEC_DIR, spec)
        if not os.path.exists(p):
            print(f"MISSING {p}", file=sys.stderr)
            return 1
        cats = categorize(p)
        report["specs"][spec] = {k: len(v) for k, v in cats.items()}
        report["specs"][f"{spec}::sample"] = {k: v[:3] for k, v in cats.items()}
    for anchor, _cat, pattern in ANCHORS:
        for spec in SPECS:
            report["reconciliation"].append(reconcile(spec, anchor, pattern))
    out = os.path.join(SPEC_DIR, "claim_reconciliation_report.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    # Compact summary to stdout
    print("SPECS:")
    for spec in SPECS:
        print(f"  {spec[:60]}: {report['specs'][spec]}")
    print("RECONCILIATION (first live-status per claim):")
    seen = set()
    for r in report["reconciliation"]:
        key = r["spec_claim"]
        if key in seen:
            continue
        seen.add(key)
        print(f"  {key}: {r['status']} ({len(r['live_hits'])} hits) {r['live_hits'][:2]}")
    print(f"REPORT={out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
