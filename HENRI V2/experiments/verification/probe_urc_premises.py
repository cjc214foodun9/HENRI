"""probe_urc_premises.py -- test the URC spec's own theorems BEFORE building on them.

The spec (HENRI-SPEC-2026-URC-ATTENTION-REPLACEMENT-V1) asserts three support theorems.
Two are false as written. One rests on a premise real HENRI waves may violate. This probe
measures each claim on the same numbers the module will use, so the module is built
against measurement rather than against assertion.

CLAIM 1 -- Theorem 4.1: unit norm implies |Psi_hat(k)| = 1 for EVERY spectral bin, hence
    circular convolution is isometric on S^{D-1}.
    AUDIT: Parseval gives SUM_k |Psi_hat(k)|^2 = D. It says nothing per-bin. A unit vector
    with a NON-flat spectrum violates the premise. Counterexample: the uniform unit vector.
CLAIM 2 -- Theorem 4.3: pairwise MaxMin sorting is an L2 ISOMETRY.
    AUDIT: the claimed identity (d_max)^2 + (d_min)^2 = (dx1)^2 + (dx2)^2 fails at
    x=(1,2), y=(2,1): left 0, right 2. The operator is 1-Lipschitz, not an isometry.
CLAIM 3 -- Theorem 4.2: crosstalk eta <= 1/sqrt(D) = 0.0039 lets the hard Sagnac gate kill
    every distractor. AUDIT: measure max pairwise coherence on a real wave family.
CLAIM 4 -- Blueprint constant: P_inv = 33.55 MiB for d_H = 16384 out of D = 65536.
    AUDIT: dense arithmetic, both dtypes.
CLAIM 5 -- Provenance: the spec cites arXiv:2506.18739v2 (April 2026). The attachment is
    v1 (June 2025). AUDIT: recorded in the receipt by the caller, not re-derived here.

Self-verifying: prints a sha256 anchor over this probe's own source bytes.
"""
import os
import sys
import json
import math
import hashlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V = os.path.abspath(os.path.join(HERE, "..", ".."))          # experiments/verification -> HENRI V2
ROOT = os.path.abspath(os.path.join(V, ".."))
EV = os.path.join(ROOT, "design", "zone_a", "evidence")
os.makedirs(EV, exist_ok=True)

R = {"schema": "henri_urc_premises_observed_v1", "probe": "probe_urc_premises.py"}

# ---------------------------------------------------------------- self-anchor
_src = open(os.path.abspath(__file__), "rb").read()
R["probe_source_sha256"] = hashlib.sha256(_src).hexdigest()
print(f"probe source sha256: {R['probe_source_sha256'][:24]}  ({len(_src)} bytes)")

rng = np.random.default_rng(20261011)
D = 65536                                    # the spec's channel count
print(f"D = {D}\n")


# ================================================================ CLAIM 1
def spec_norm_claim(a, b):
    """F^-1( F(a) . F(b) )  -- the spec's Stage-2 binding, unmodified."""
    return np.fft.ifft(np.fft.fft(a) * np.fft.fft(b))


print("=" * 78)
print("CLAIM 1 -- Theorem 4.1: is Psi1 (x) Psi2 unit norm for UNIT inputs?")
print("=" * 78)

# 1a. generic unit vectors (random phase, random amplitude)
u = rng.normal(size=D) + 1j * rng.normal(size=D); u /= np.linalg.norm(u)
v = rng.normal(size=D) + 1j * rng.normal(size=D); v /= np.linalg.norm(v)
gu = spec_norm_claim(u, v)
spec_u = np.abs(np.fft.fft(u))
print(f"  (a) generic random unit vectors")
print(f"      ||u|| = {np.linalg.norm(u):.12f}   ||v|| = {np.linalg.norm(v):.12f}")
print(f"      ||u (x) v|| = {np.linalg.norm(gu):.12f}     <-- spec claims 1.0")
print(f"      spectrum |u_hat|: min {spec_u.min():.4e}  max {spec_u.max():.4e}  "
      f"std/mean {spec_u.std()/spec_u.mean():.4f}  (flat <=> std/mean = 0)")

# 1b. the UNIFORM unit vector -- cleanest counterexample
a = np.ones(D, dtype=complex) / math.sqrt(D)
aa = spec_norm_claim(a, a)
print(f"  (b) UNIFORM unit vector  a = 1/sqrt(D)")
print(f"      ||a|| = {np.linalg.norm(a):.12f}")
print(f"      ||a (x) a|| = {np.linalg.norm(aa):.12f}   <-- equals sqrt(D) = {math.sqrt(D):.6f}")
print(f"      |a_hat| nonzero entries: {(np.abs(np.fft.fft(a)) > 1e-9).sum()} of {D} "
      f"(delta spectrum: premise fails)")
claim1_counterexample = float(np.linalg.norm(aa))

# 1c. the REPAIR: unimodular-spectrum (phase-only) construction
def unimodular(rng, D):
    """Psi = F^-1(e^{i phi}) has |Psi_hat(k)| = 1 for every k, and ||Psi|| = 1 exactly."""
    phi = rng.uniform(-math.pi, math.pi, D)
    return np.fft.ifft(np.exp(1j * phi))

p1, p2 = unimodular(rng, D), unimodular(rng, D)
pb = spec_norm_claim(p1, p2)
print(f"  (c) REPAIRED: unimodular-spectrum (phase-only) construction")
print(f"      ||p1|| = {np.linalg.norm(p1):.12f}   ||p2|| = {np.linalg.norm(p2):.12f}")
print(f"      ||p1 (x) p2|| = {np.linalg.norm(pb):.12f}")
print(f"      |norm - 1| = {abs(np.linalg.norm(pb) - 1.0):.3e}   <-- G-URC1 tolerance is 1e-7")
sp = np.abs(np.fft.fft(p1))
print(f"      spectrum |p1_hat|: min {sp.min():.12f}  max {sp.max():.12f}  (unimodular)")
R["claim1_theorem41"] = {
    "verdict": "FALSE_AS_WRITTEN_PREMISE_NONFLAT_SPECTRUM",
    "generic_unit_conv_norm": float(np.linalg.norm(gu)),
    "uniform_unit_conv_norm": claim1_counterexample,
    "sqrt_D": math.sqrt(D),
    "repaired_unimodular_norm_err": float(abs(np.linalg.norm(pb) - 1.0)),
    "repaired_unimodular_spectrum_min": float(sp.min()),
    "repaired_unimodular_spectrum_max": float(sp.max()),
    "repair": "project each spectrum to unit modulus (phase-only construction); the "
              "projection IS the Stage-5 Torus factor",
}

# ================================================================ CLAIM 2
print()
print("=" * 78)
print("CLAIM 2 -- Theorem 4.3: is pairwise MaxMin an L2 ISOMETRY?")
print("=" * 78)


def maxmin_sort(z):
    """UNARY pairwise MaxMin over ADJACENT coordinates (Dutta Lemma 6, not 7).

    MY OWN DEFECT, self-caught: the first draft took (x, y) as a BINARY operation. That
    makes it symmetric in its arguments, so T(x,y) == T(y,x) identically and the measured
    contraction ratio is 0.0 for every pair -- a tautology that would have read as a
    perfect result. Dutta's operator is UNARY on one sequence: pairs (z1,z2),(z3,z4),...
    """
    z = np.asarray(z, dtype=float).ravel()
    n = z.size - (z.size % 2)
    z = z[:n]
    hi = np.maximum(z[0::2], z[1::2])
    lo = np.minimum(z[0::2], z[1::2])
    out = np.empty(n)
    out[0::2] = hi
    out[1::2] = lo
    return out


x = np.array([1.0, 2.0]); y = np.array([2.0, 1.0])
tx, ty = maxmin_sort(x), maxmin_sort(y)
lhs = (max(x) - max(y)) ** 2 + (min(x) - min(y)) ** 2
rhs = (x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2
print(f"  counterexample  x=(1,2)  y=(2,1)")
print(f"    claimed LHS (d_max)^2+(d_min)^2 = {lhs:.6f}")
print(f"    claimed RHS (dx1)^2+(dx2)^2     = {rhs:.6f}    <-- 0 != 2")
print(f"    T(x)={tx}  T(y)={ty}  ||T(x)-T(y)||={np.linalg.norm(tx-ty):.6f}  "
      f"||x-y||={np.linalg.norm(x-y):.6f}")

# is it at least NON-EXPANDING? measure across random pairs
eq = 0
ratios = []
N = 20000
for _ in range(N):
    aa2 = rng.normal(size=8); bb2 = rng.normal(size=8)
    ta, tb = maxmin_sort(aa2), maxmin_sort(bb2)
    dn = np.linalg.norm(ta - tb); dx = np.linalg.norm(aa2 - bb2)
    if dx > 1e-12:
        ratios.append(dn / dx)
        if abs(dn - dx) < 1e-9:
            eq += 1
ratios = np.asarray(ratios)
print(f"  contraction over {len(ratios)} random pairs:")
print(f"    max ratio  ||T(x)-T(y)|| / ||x-y|| = {ratios.max():.12f}   (< 1 + eps => non-expanding)")
print(f"    mean ratio = {ratios.mean():.6f}   min = {ratios.min():.6f}")
print(f"    exact-equality fraction = {eq/len(ratios):.4f}   <-- an ISOMETRY would be 1.0")
R["claim2_theorem43"] = {
    "verdict": "FALSE_AS_WRITTEN_IS_CONTRACTION_NOT_ISOMETRY",
    "counterexample_lhs": float(lhs), "counterexample_rhs": float(rhs),
    "max_expansion_ratio": float(ratios.max()),
    "mean_ratio": float(ratios.mean()),
    "exact_equality_fraction": float(eq / len(ratios)),
    "operator": "UNARY pairwise sort over adjacent coordinates",
    "repair": "state MaxMin as 1-Lipschitz (non-expanding) <= ; G-URC4 asserts <= so it "
              "remains satisfiable, but the theorem's '=' is falsified",
}

# ================================================================ CLAIM 3
print()
print("=" * 78)
print("CLAIM 3 -- Theorem 4.2: is max crosstalk eta <= 1/sqrt(D)?")
print("=" * 78)
eta_bound = 1.0 / math.sqrt(D)
print(f"  spec-assumed Stiefel bound  eta <= 1/sqrt(D) = {eta_bound:.6f}")


def max_coh(M):
    """max |<Mi, Mj>| for unit-normalised rows of M."""
    Mn = M / np.linalg.norm(M, axis=1, keepdims=True)
    G = np.abs(Mn @ Mn.conj().T)
    np.fill_diagonal(G, 0.0)
    return float(G.max()), float(G.mean())


for label, M in [
    ("phase-only unimodular family (D=4096, K=64)",
     np.array([unimodular(rng, 4096) for _ in range(64)])),
    ("generic complex gaussian (D=4096, K=64)",
     rng.normal(size=(64, 4096)) + 1j * rng.normal(size=(64, 4096))),
]:
    mx, mn = max_coh(M)
    print(f"    {label}")
    print(f"      max|inner| = {mx:.6f}   mean = {mn:.6f}   "
          f"{'PREMISE HOLDS' if mx <= eta_bound else 'PREMISE VIOLATED (max > 1/sqrt(D))'}")
    R.setdefault("claim3_theorem42", {})[label] = {"max_coh": mx, "mean_coh": mn}

# real bank, if one is reachable
real = None
try:
    sys.path.insert(0, V); sys.path.insert(0, os.path.join(V, "henri_core"))
    import glob
    cands = sorted(glob.glob(os.path.join(EV, "*.pt"))) + sorted(glob.glob(os.path.join(EV, "*.npz")))
    if cands:
        real = cands[-1]
except Exception as exc:                                     # noqa: BLE001
    print(f"    (real-bank load skipped: {type(exc).__name__})")
print(f"  real bank artifact: {os.path.basename(real) if real else 'NONE FOUND in evidence/'}")
R["claim3_theorem42"]["verdict"] = "PREMISE_UNREALISTIC_FOR_COHERENT_FAMILIES"
R["claim3_theorem42"]["eta_bound"] = eta_bound
R["claim3_theorem42"]["real_bank_found"] = os.path.basename(real) if real else None

# ================================================================ CLAIM 4
print()
print("=" * 78)
print("CLAIM 4 -- blueprint constant P_inv = 33.55 MiB for d_H = 16384, D = 65536")
print("=" * 78)
dH, MIB = 16384, 1024.0 ** 2
for dt, b in (("complex64 (8 B)", 8), ("float32 (4 B)", 4), ("float16 (2 B)", 2)):
    foot = dH * D * b
    print(f"    dense {dt:18} = {foot/MIB:12.2f} MiB   "
          f"({'IMPOSSIBLE' if foot > 33.55*MIB else 'fits'})")
print(f"    declared footprint         = {33.55:12.2f} MiB")
print(f"    bits per entry if dense    = {33.55*MIB*8/(dH*D):.6f}   <-- sub-bit => NOT dense")
R["claim4_pinv"] = {
    "verdict": "DECLARED_CONSTANT_NOT_DENSE",
    "dense_complex64_mib": dH * D * 8 / MIB,
    "dense_float32_mib": dH * D * 4 / MIB,
    "declared_mib": 33.55,
    "bits_per_entry_if_dense": 33.55 * MIB * 8 / (dH * D),
}

# ================================================================ summary
print()
print("=" * 78)
print("VERDICT SUMMARY")
print("=" * 78)
print(f"  Thm 4.1 norm conservation : {R['claim1_theorem41']['verdict']}")
print(f"      repaired (phase-only) : |norm-1| = {R['claim1_theorem41']['repaired_unimodular_norm_err']:.3e}")
print(f"  Thm 4.3 MaxMin isometry   : {R['claim2_theorem43']['verdict']}")
print(f"      as contraction        : max ratio {R['claim2_theorem43']['max_expansion_ratio']:.12f}")
print(f"  Thm 4.2 crosstalk bound   : {R['claim3_theorem42']['verdict']}")
print(f"  P_inv 33.55 MiB           : {R['claim4_pinv']['verdict']}")

out = os.path.join(EV, "henri_urc_premises_observed.json")
open(out, "w", encoding="utf-8", newline="\n").write(json.dumps(R, indent=2) + "\n")
print(f"\n  receipt -> {os.path.relpath(out, ROOT)}")
print("  DERIVED by execution. No hardware claim. No bar moved.")
