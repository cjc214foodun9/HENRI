"""Self-check for the Phase 1 novelty->veto coupling. Every check CAN fail.

Constructs a veto whose axiom bank gives a KNOWN delta_sagnac, so every assertion
is exact rather than empirical.
"""
import sys

sys.path.insert(0, ".")
import torch
from henri_core.sagnac_homodyne import SagnacHomodyneVeto

PASS, FAIL = [], []


def ck(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")


D = 64
THR = 0.35


def controlled_veto(phi):
    """Axiom = e0. Candidate = cos(phi) e0 + sin(phi) e1. delta = 1 - cos(phi)."""
    v = SagnacHomodyneVeto(threshold=THR, dim=D)
    ax = torch.zeros(1, D, dtype=torch.complex64)
    ax[0, 0] = 1.0
    v.load_axioms(ax)
    c = torch.zeros(1, D, dtype=torch.complex64)
    c[0, 0] = torch.cos(torch.tensor(phi))
    c[0, 1] = torch.sin(torch.tensor(phi))
    return v, c


phi = float(torch.arccos(torch.tensor(0.70)))          # delta = 0.30 < 0.35
v, c = controlled_veto(phi)

# T0 the fixture itself
d0, _ = v.margin(c)
ck("T0 fixture delta_sagnac == 0.30", abs(float(d0) - 0.30) < 1e-5, f"{float(d0):.6f}")

# T1 lambda=0 is inert against the raw veto
r_raw = v(c)
r_zero = v(c, novelty_score=0.0, novelty_lambda=0.0)
ck("T1 lambda=0 inert (allow equal)", bool(r_raw["allow"][0]) == bool(r_zero["allow"][0]))
ck("T1b lambda=0 delta identical",
   abs(float(r_raw["delta"][0]) - float(r_zero["delta"][0])) < 1e-12)
ck("T1c lambda=0 penalty zero", r_zero["novelty_penalty"] == 0.0)
ck("T1d lambda=0 veto_source", r_zero["veto_source"] == "SAGNAC_HOMODYNE")

# T2 no novelty args at all -> identical to lambda=0
r_none = v(c)
ck("T2 no-args == lambda=0",
   bool(r_none["allow"][0]) == bool(r_zero["allow"][0])
   and abs(float(r_none["delta"][0]) - float(r_zero["delta"][0])) < 1e-12)
ck("T2b no-args has the new keys", "delta_sagnac" in r_none and "veto_source" in r_none)

# T3 penalty arithmetic
r = v(c, novelty_score=0.4, novelty_lambda=0.5)
pen_expected = 0.5 * (1.0 - 0.4)
ck("T3 penalty == lambda*(1-s)",
   abs(float(r["novelty_penalty"]) - pen_expected) < 1e-9,
   f"{float(r['novelty_penalty']):.6f} vs {pen_expected:.6f}")
ck("T3b delta_eff == delta_sagnac + penalty",
   abs(float(r["delta"][0]) - (float(r["delta_sagnac"][0]) + pen_expected)) < 1e-6)

# T4 the FLIP: s=0.0 pushes 0.30 -> 0.80 > 0.35
r_flip = v(c, novelty_score=0.0, novelty_lambda=0.5)
ck("T4 raw veto would PASS", bool(r_raw["allow"][0]) is True)
ck("T4b coupled veto FLIPS to reject", bool(r_flip["allow"][0]) is False)
ck("T4c flipped_by_novelty set", bool(r_flip["flipped_by_novelty"][0]) is True)
ck("T4d veto_source names the coupling",
   r_flip["veto_source"] == "SAGNAC_NOVELTY_COUPLING", r_flip["veto_source"])

# T5 a familiar input (s=1.0) must NOT be flipped
r_fam = v(c, novelty_score=1.0, novelty_lambda=0.5)
ck("T5 s=1.0 no penalty, no flip",
   bool(r_fam["allow"][0]) is True and bool(r_fam["flipped_by_novelty"][0]) is False
   and r_fam["veto_source"] == "SAGNAC_HOMODYNE")

# T6 monotone: higher s -> lower delta_eff
ds = [float(v(c, novelty_score=s, novelty_lambda=0.5)["delta"][0])
      for s in (0.0, 0.25, 0.5, 0.75, 1.0)]
ck("T6 monotone decreasing in s", all(a > b for a, b in zip(ds, ds[1:])), f"{ds}")

# T7 coupling can never RELEASE: delta_eff >= delta_sagnac for all s
addonly = all(float(v(c, novelty_score=s, novelty_lambda=0.5)["delta"][0])
              >= float(r_raw["delta"][0]) - 1e-12 for s in (0.0, 0.5, 1.0))
ck("T7 coupling only ADDS deflection", addonly)

# T8 fail-closed preserved: empty baseplate still rejects, and reports raw source
v_empty = SagnacHomodyneVeto(threshold=THR, dim=D)
r_err = v_empty(c, novelty_score=0.0, novelty_lambda=0.5)
ck("T8 fail-closed on empty baseplate",
   bool(r_err["allow"][0]) is False and r_err["delta"][0] == float("inf"))
ck("T8b error path reports SAGNAC_HOMODYNE (not the coupling)",
   r_err["veto_source"] == "SAGNAC_HOMODYNE")
ck("T8c error dict carries the new keys",
   {"delta_sagnac", "novelty_penalty", "flipped_by_novelty"} <= set(r_err))

# T9 NEGATIVE CONTROL: with lambda=0 the SAME wave is released, so the flip in
# T4 is attributable to the coupling and not to the wave.
ck("T9-nc same wave released at lambda=0",
   bool(v(c, novelty_score=0.0, novelty_lambda=0.0)["allow"][0]) is True)

print(f"\n==== {len(PASS)} passed, {len(FAIL)} failed ====")
if FAIL:
    print("FAILED:", FAIL)
sys.exit(1 if FAIL else 0)
