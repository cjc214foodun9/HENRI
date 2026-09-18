"""Verify the LITERATURE layer of the natural-intelligence audit.

WHY
    The audit's Layer 1 claims are about PUBLISHED frameworks, not about this
    codebase: Levin's TAME / cognitive light cones, Kuramoto synchronization,
    Friston's Markov blankets and free-energy principle, Langevin/Fokker-Planck
    stationary distributions. Those are checkable against primary literature.

    Its Layer 2 (hardware) claims are BLOCKED here (no substrate) and its Layer 3
    (biology-to-physics 1:1 mapping) is a DESIGN METAPHOR: the software implements
    wave states, phase vetoes and SGLD creep, NOT membrane potentials, gap junctions
    or apoptosis. Both are recorded as such and never promoted to measurement.

This probe records, per claim: the artefact in THIS tree that embodies it (if any),
and the literature status. It asserts nothing about capability.
"""
import json
import pathlib
import re
import sys

R = pathlib.Path(r"C:\Users\chan\Desktop\HENRI 7B SWARM\.worktrees\basal-syncytium\HENRI V2")
sys.path.insert(0, str(R))

# Layer 1: literature claims -> the REAL published source, and whether this tree
# implements the mechanism or only names it.
LIT = [
    {
        "id": "L1-TAME",
        "claim": "TAME / scale-free basal cognition; intelligence = capacity to navigate "
                 "spaces toward homeostatic setpoints; cognitive light cones",
        "source": "Levin, M. 'The Computational Boundary of a Self' / 'Technological "
                  "Approach to Mind Everywhere (TAME)'",
        "status": "REAL_PUBLISHED_FRAMEWORK",
        "in_tree": "multi-scale loops (reflex/tactical/strategic) and the Sagnac veto "
                   "as a homeostatic boundary. The LOOP HIERARCHY is implemented and "
                   "measured (tactical > reflex > strategic, CPU); the CLAIM that this "
                   "constitutes basal cognition is an interpretive framing, not a "
                   "measurement.",
        "verdict": "FRAMING_ONLY -- no test in this tree can confirm or refute TAME.",
    },
    {
        "id": "L2-KURAMOTO",
        "claim": "phase synchronization r*e^{ipsi} = (1/N) sum e^{i theta_j} performs "
                 "consensus 'for free'",
        "source": "Kuramoto 1975; Strogatz review",
        "status": "REAL_PUBLISHED_FRAMEWORK",
        "in_tree": "the audit cites the order parameter. SEARCH RESULT below records "
                   "whether a Kuramoto order parameter is actually computed or only "
                   "quoted.",
        "verdict": "CHECK_IN_TREE",
    },
    {
        "id": "L3-FEP",
        "claim": "variational free energy F = <ln q(eta) - ln p(eta,s)>; perceptual vs "
                 "active inference; Markov blanket",
        "source": "Friston 2010; Friston et al. active inference reviews",
        "status": "REAL_PUBLISHED_FRAMEWORK",
        "in_tree": "arc_efe_planner / EFEPlanner compute an expected-free-energy "
                   "candidate table. That is an IMPLEMENTED objective; whether it is "
                   "DIFFERENTIATED through the claimed parameter path is a separate, "
                   "previously-flagged question.",
        "verdict": "MECHANISM_PARTIAL -- EFE exists; 'minimises variational free "
                   "energy' is not established by its presence.",
    },
    {
        "id": "L4-LANGEVIN",
        "claim": "SGLD / Langevin creep; Fokker-Planck stationarity "
                 "P_inf ~ exp(-F / k_B T); T_eff -> 0 gives a Dirac measure at argmin F",
        "source": "standard non-equilibrium statistical mechanics (Langevin/SGLD "
                  "literature)",
        "status": "REAL_MATHEMATICS",
        "in_tree": "adaptive_viscoelastic_thermostat.step_viscoelastic_creep exists and "
                   "executes (measured). The stationarity FORM requires a temperature "
                   "schedule and mixing conditions that the audit ASSERTS rather than "
                   "establishes.",
        "verdict": "FORM_VALID_AS_ASSUMPTION -- 'T_eff -> 0 iff Delta_Sagnac <= 0.0431' "
                   "is a DEFINITION in the document, not a derivation. Record as "
                   "'stationary form under standard SGLD assumptions', never as a "
                   "convergence guarantee.",
    },
    {
        "id": "L5-NORM",
        "claim": "unitary Hamiltonian flow conserves the norm (d/dt ||Psi|| = 0)",
        "source": "elementary linear algebra / quantum mechanics",
        "status": "REAL_AND_TRUE",
        "in_tree": "the proof is correct but TRIVIAL: a Hermitian generator gives a "
                   "unitary flow, so the norm is conserved by construction. It "
                   "establishes no property of the learned operator.",
        "verdict": "TRUE_BUT_TRIVIAL -- do not cite as evidence of capability.",
    },
]

# Layer 2: hardware claims already registered BLOCKED.
HW_BLOCKED_IDS = ["H-1", "H-2", "H-3", "H-4", "H-5", "H-6", "H-7", "H-8", "H-9",
                  "H-10", "H-11"]

print("=" * 78)
print("LAYER 1 -- LITERATURE CLAIMS (published frameworks; citable)")
print("=" * 78)
for c in LIT:
    print(f"\n   {c['id']}  [{c['status']}]")
    print(f"      claim   : {c['claim'][:96]}")
    print(f"      source  : {c['source'][:96]}")
    print(f"      in tree : {c['in_tree'][:180]}")
    print(f"      verdict : {c['verdict'][:150]}")

print()
print("=" * 78)
print("IN-TREE CHECK: does the tree actually COMPUTE what the audit names?")
print("=" * 78)
PROBES = {
    "kuramoto_order_parameter": r"kuramoto|order_parameter|\br\s*=\s*.*abs\(.*mean",
    "sagnac_veto": r"delta_sagnac|sagnac_stress|evaluate_veto",
    "efe_planner": r"class EFEPlanner|expected_free_energy",
    "sgld_creep": r"step_viscoelastic_creep|def .*sgld",
    "stiefel_retraction": r"stiefel|retract",
    "hopfield_snap": r"class HoloEgressCodebook|hopfield",
    "markov_blanket": r"markov_blanket|markov blanket",
    "membrane_potential": r"V_mem|membrane_potential",
    "gap_junction": r"gap_junction|connexin|innexin",
    "apoptosis": r"apoptosis",
    "batio3": r"BaTiO3|batio3|barium_titanate",
    "pockels": r"pockels|r42|electro_optic",
}
found = {}
for name, pat in PROBES.items():
    hits, files = 0, []
    for p in R.rglob("*.py"):
        if "_archive" in str(p):
            continue
        t = p.read_text(encoding="utf-8", errors="replace")
        n = len(re.findall(pat, t, re.I))
        if n:
            hits += n
            files.append(p.name)
    found[name] = (hits, files[:3])
    mark = "IN TREE" if hits else "ABSENT "
    print(f"   {mark}  {name:<26} {hits:>4} hits  {files[:2]}")

print()
print("=" * 78)
print("LAYER 3 -- BIOLOGY-TO-PHYSICS 1:1 LEDGER: implemented vs named-only")
print("=" * 78)
BIO = [
    ("Resting membrane potential V_mem", "complex phasors on S^{D-1}",
     "membrane_potential"),
    ("Intercellular gap junctions", "optical/epaptic waveguide coupling", "gap_junction"),
    ("Morphogenetic homeostasis", "Sagnac homodyne veto", "sagnac_veto"),
    ("Cellular stress / phenotypic drift", "SGLD viscoelastic creep", "sgld_creep"),
    ("Metabolic apoptosis", "dark-port veto -> thermal annealer", "apoptosis"),
    ("Waddington landscape", "Modern Hopfield lexical snap", "hopfield_snap"),
    ("Ligand-receptor cascades", "qFHRR circular convolution", None),
    ("Pockels modulation", "electro-optic phase modulators", "pockels"),
]
for bio, henri, probe in BIO:
    if probe is None:
        state = "not probed"
    else:
        h = found.get(probe, (0, []))[0]
        state = "IMPLEMENTED" if h else "NAMED ONLY (no code)"
    print(f"   {bio:<34} -> {henri:<34} {state}")

print()
print("=" * 78)
print("LAYER 2 -- HARDWARE (registered BLOCKED; no substrate on this host)")
print("=" * 78)
reg = R / "experiments" / "verification" / "hardware_substrate_blocked.json"
if reg.exists():
    d = json.loads(reg.read_bytes())
    claims = d.get("claimed_substrate_figures", [])
    blocked = [c for c in claims if str(c.get("status", "")).startswith("BLOCKED")]
    print(f"   register: {len(claims)} claims, {len(blocked)} BLOCKED")
    print(f"   energy_per_step: {d.get('energy_per_step', 'None (by construction)')}")
else:
    print("   register absent")

print()
print("=" * 78)
print("THERMODYNAMIC 'FREE LUNCH' -- falsifiability status")
print("=" * 78)
print("   The audit's '< 6.67 nJ/step' and '< 0.1 ns transit' are substrate figures.")
print("   This host has no joule sensor and no photonic substrate, so the claim is")
print("   UNFALSIFIABLE HERE. It is neither confirmed nor denied; it is quarantined.")
print("   Any statement that the architecture 'computes for free' is therefore")
print("   HYPOTHESIS at best and must not appear in a capability claim.")

out = R / "experiments" / "verification" / "natural_intelligence_literature_audit.json"
body = {
    "schema": "henri.natural-intelligence-audit.v1",
    "evidence_class": "DERIVED",
    "source_document": "HENRI-ARCH-2026-NATURAL-INTELLIGENCE-AUDIT",
    "provenance_note": (
        "The source document is DOWNSTREAM of this project's own receipts: it carries "
        "session-authored identifiers and measured values (T*=0.038316, beta*=26.10, "
        "+0.6306, 0.4822->0.0536, Seal Pair 7, ScalarRotorRejected, probe_belief_wave). "
        "It is a synthesis OF the artefacts, so it is treated as a spec to verify, "
        "never as authority over them."
    ),
    "layer_1_literature": LIT,
    "in_tree_probe": {k: {"hits": v[0], "files": v[1]} for k, v in found.items()},
    "layer_3_isomorphism": [
        {"biological": b, "henri": h, "state": ("not probed" if not p
         else ("IMPLEMENTED" if found.get(p, (0, []))[0] else "NAMED ONLY (no code)"))}
        for b, h, p in BIO
    ],
    "layer_2_hardware": {
        "status": "BLOCKED_NO_SUBSTRATE",
        "registered_claims": len(HW_BLOCKED_IDS),
        "note": "no photonic, CXL, Rust or GB202 substrate on this host; the '<6.67 "
                "nJ/step' and '<0.1 ns' figures are unfalsifiable here and are neither "
                "confirmed nor denied.",
    },
    "verdicts": {
        "literature_layer": "REAL frameworks; citing them validates the FRAMING only.",
        "isomorphism_layer": "DESIGN METAPHOR. The software implements wave states, "
                             "phase vetoes and SGLD creep -- not V_mem, gap junctions "
                             "or apoptosis. Rhetorically isomorphic, mechanistically "
                             "analogical.",
        "theorem_layer": "norm conservation is TRUE BUT TRIVIAL (Hermitian generator => "
                         "unitary flow); the convergence 'theorem' is a stationary "
                         "FORM under asserted conditions, and 'T_eff->0 iff "
                         "Delta_Sagnac<=0.0431' is a DEFINITION, not a derivation.",
        "free_lunch": "UNFALSIFIABLE_HERE -- quarantined from capability claims.",
    },
    "non_claims": [
        "No capability claim. Live benchmark score remains 0.0%.",
        "No claim that biology was 'translated into physics'.",
        "No substrate figure is promoted to measurement.",
    ],
}
with open(out, "w", encoding="utf-8", newline="") as fh:
    fh.write(json.dumps(body, indent=2) + "\n")
print()
print(f"wrote {out}")
print(f"canonical sha256 = {__import__('hashlib').sha256(out.read_bytes()).hexdigest()}")
