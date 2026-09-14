"""Audit the SUPPLIED GradePreservingCliffordTransducer (mandate PDF pp.11-13).

Same A0-A9 discipline as the previous supplied artifact. The decisive question
is NOT whether the pairing "is Hodge" in the abstract -- it is whether the
pairing matches the basis order the LIVE tiling actually uses.
"""
import math, sys
sys.path.insert(0, ".")
import torch

# ------------------------------------------------ verbatim from the mandate
class Transducer(torch.nn.Module):
    def __init__(self, M=64, D=512):
        super().__init__()
        self.M, self.B, self.D = M, 8, D
        assert self.M * 4 * 2 == self.D, f"{M}*8 != {D}"
        angles = torch.linspace(0, 2 * math.pi * (self.M - 1) / float(self.M), self.M)
        self.register_buffer("spin_rotors", torch.exp(1j * angles))

    def clifford_to_complex_wave(self, x):
        b = x.shape[0]
        p0 = torch.complex(x[..., 0], x[..., 7])
        p1 = torch.complex(x[..., 1], x[..., 4])
        p2 = torch.complex(x[..., 2], x[..., 5])
        p3 = torch.complex(x[..., 3], x[..., 6])
        ph = torch.stack([p0, p1, p2, p3], dim=-1)
        ln = torch.linalg.norm(ph, dim=-1, keepdim=True).clamp(min=1e-12)
        pn = ph / ln
        base = pn.reshape(b, -1)
        rotor = self.spin_rotors.unsqueeze(-1).expand(self.M, 4).reshape(1, -1)
        trans = base * rotor
        comp = torch.cat([base, trans], dim=-1)
        return comp / math.sqrt(2.0)

    def complex_wave_to_clifford(self, psi):
        b = psi.shape[0]
        half = psi[..., : self.D // 2] * math.sqrt(2.0)
        ph = half.reshape(b, self.M, 4)
        out = torch.zeros(b, self.M, 8, dtype=torch.float32, device=psi.device)
        out[..., 0] = ph[..., 0].real; out[..., 7] = ph[..., 0].imag
        out[..., 1] = ph[..., 1].real; out[..., 4] = ph[..., 1].imag
        out[..., 2] = ph[..., 2].real; out[..., 5] = ph[..., 2].imag
        out[..., 3] = ph[..., 3].real; out[..., 6] = ph[..., 3].imag
        bn = torch.linalg.norm(out, dim=-1, keepdim=True).clamp(min=1e-12)
        return out / bn

print("=== T0: instantiation at toy scale (M=64, D=512) ===")
tr = Transducer(M=64, D=512)
print(f"  built OK; spin_rotors {tuple(tr.spin_rotors.shape)} complex")
try:
    from pydantic import BaseModel, Field, conint
    class Cfg(BaseModel):
        num_blocks: conint(ge=64, le=16384) = 8192
        block_dim: conint(ge=8, le=8) = 8
        target_D: conint(ge=512, le=131072) = 65536
    print("  pydantic accepted toy M=64/D=512:", Cfg(num_blocks=64, target_D=512) is not None)
    print("  NOTE: unlike the PREVIOUS supplied adapter (ge=1024 blocked toy scale),")
    print("        these bounds DO allow reduced-scale instantiation. FIXED.")
except Exception as e:
    print("  pydantic unavailable:", type(e).__name__)

print()
print("=== T1: round-trip (claim: 'exact left-inverse on the range') ===")
torch.manual_seed(0)
x = torch.randn(4, 64, 8)
psi = tr.clifford_to_complex_wave(x)
back = tr.complex_wave_to_clifford(psi)
unit = x / torch.linalg.norm(x, dim=-1, keepdim=True).clamp(min=1e-12)
err = (back - unit).abs().max().item()
print(f"  max|inv(fwd(x)) - x/|x|_block| = {err:.3e}")
print(f"  OLD adapter round-trip error was 0.9964 -> this is {'FIXED' if err < 1e-5 else 'STILL BROKEN'}")

print()
print("=== T2: is magnitude preserved, or discarded? ===")
s = torch.linalg.norm(psi, dim=-1)
print(f"  |psi| per sample = {s.tolist()}")
print(f"  expected sqrt(M) if per-block unit and no odd scale: {math.sqrt(64):.3f}")
print(f"  claim was 'Unit hypersphere embedding factor' -> |psi| is sqrt(M), NOT 1")
x5 = x * 5.0
print(f"  fwd(5x) == fwd(x) ? {torch.allclose(tr.clifford_to_complex_wave(x5), psi)}  -> magnitude DISCARDED")

print()
print("=== T3: THE DECISIVE TEST -- Hodge pairing vs the LIVE basis order ===")
LIVE = {0: "1", 1: "e1", 2: "e2", 3: "e3", 4: "e12", 5: "e13", 6: "e23", 7: "e123"}
HODGE = {0: 7, 7: 0, 1: 6, 6: 1, 2: 5, 5: 2, 3: 4, 4: 3}
CODE = [(0, 7), (1, 4), (2, 5), (3, 6)]
bad = 0
for a, b in CODE:
    ok = (HODGE[a] == b)
    bad += (not ok)
    print(f"  code pairs idx{a}({LIVE[a]:>4}) with idx{b}({LIVE[b]:>4});  "
          f"live hodge({LIVE[a]:>4}) = {LIVE[HODGE[a]]:>4}   {'OK' if ok else '*** MISMATCH ***'}")
print(f"  -> {len(CODE)-bad}/{len(CODE)} pairs match the LIVE basis order.")

print()
print("  Numeric consequence (block 16, rotor = e^{i*pi/2} exactly):")
x1 = torch.zeros(1, 64, 8); x1[0, 16, 1] = 1.0        # pure e1 in block 16
ang = float(torch.angle(tr.spin_rotors[16]))
b1 = tr.complex_wave_to_clifford(tr.clifford_to_complex_wave(x1))
print(f"    rotor angle at block 16 = {ang:.4f} rad (pi/2 = {math.pi/2:.4f})")
for i in (1, 4, 6):
    print(f"    recovered idx{i} ({LIVE[i]:>3}) = {float(b1[0,16,i]): .6f}")
print("    e1 rotated a quarter turn -> e12 (idx4). Under Hodge it must be e23 (idx6),")
print("    which stays ~0. The module pairs e1 with the WRONG bivector.")

print()
print("  Which basis order WOULD make the code correct?")
PDF_ORDER = {0: "1", 1: "e1", 2: "e2", 3: "e3", 4: "e23", 5: "e13", 6: "e12", 7: "e123"}
for a, b in CODE:
    print(f"    idx{a}({PDF_ORDER[a]:>4}) with idx{b}({PDF_ORDER[b]:>4}) -> "
          f"hodge({PDF_ORDER[a]:>4}) = {PDF_ORDER[HODGE[a]]:>4}  "
          f"{'OK' if HODGE[a]==b else 'MISMATCH'}")
print("  -> the code is Hodge-correct ONLY under an assumed order with e23 at idx4 and")
print("     e12 at idx6. The LIVE order has e12 at idx4 and e23 at idx6: a swap of 4<->6.")

print()
print("=== T4: does the Spin(3) rotor preserve grades? ===")
x0 = torch.zeros(1, 64, 8); x0[0, 16, 0] = 1.0        # pure scalar
b0 = tr.complex_wave_to_clifford(tr.clifford_to_complex_wave(x0))
print(f"  scalar idx0 -> {float(b0[0,16,0]): .6f}, pseudoscalar idx7 -> {float(b0[0,16,7]): .6f}")
# CLAIM RETRACTED 2026-09-14. The first version of this audit printed the two
# numbers above and then asserted "grade 0 converts to grade 3" -- a conclusion
# its own output contradicts (the pseudoscalar slot is 0.0). The rotor is applied
# only to the redundant transverse half, and the inverse reads only the base
# half, so the round-trip CANNOT move grades.
#
# The correct statement, measured below: a complex phasor pairs grade k with its
# Hodge dual grade (3-k), so a PHASE ROTATION applied to that phasor moves
# amplitude between the two grades *of its own pair*. That is why the class is
# named Hodge-dual and not grade-preserving in the corrected module.
print("  -> round-trip is grade-STABLE (the rotor half is not read back), so the")
print("     original 'grade 0 converts to grade 3' conclusion is RETRACTED as")
print("     unsupported by these numbers. Name-overclaim check, corrected:")
print("     a phase rotation on a phasor mixes THAT PAIR's two grades (0<->3):")
half_ = tr.complex_wave_to_clifford(tr.clifford_to_complex_wave(x0)).shape[-1]
psi0 = tr.clifford_to_complex_wave(x0).clone()
slot = 16 * 4 + 0
ph0 = psi0[0, :64 * 4].reshape(64, 4)
ph0[16, 0] = ph0[16, 0] * torch.tensor(1j, dtype=ph0.dtype)
r0 = tr.complex_wave_to_clifford(psi0)[0, 16]
print(f"     after a 90-deg rotation of the scalar's phasor: "
      f"idx0={float(r0[0]): .6f}  idx7={float(r0[7]): .6f}")
print(f"     -> the pair's OTHER grade now carries the amplitude "
      f"({float(abs(r0[7])):.6f} > 0)")

print()
print("=== T5: locality (A8) ===")
xa = torch.zeros(2, 64, 8); xa[0, 0, 1] = 1.0; xa[0, 0, 2] = 0.3
pa = tr.clifford_to_complex_wave(xa)
xb = xa.clone(); xb[0, 0, 1] += 0.5
pb = tr.clifford_to_complex_wave(xb)
changed = int((pa[0] != pb[0]).sum())
print(f"  perturb 1 channel of block 0 -> {changed} of {pa[0].numel()} outputs change "
      f"({100.0*changed/pa[0].numel():.4f}%)")
print(f"  OLD adapter: 65536/65536 (global denominator). FIXED -> block-local normalization.")

print()
print("=== T6: is the second half an independent expansion? ===")
half = tr.D // 2
lhs = psi[..., half:] * math.sqrt(2.0)
rhs = (psi[..., :half] * math.sqrt(2.0)) * tr.spin_rotors[:64].repeat_interleave(4)
print(f"  max|transverse - base*rotor| = {(lhs - rhs).abs().max().item():.3e}")
print("  -> the 'transverse harmonic expansion' is a REDUNDANT copy. Image real-dimension")
print(f"     is M*4 = {64*4} real per sample, emitted as {tr.D} complex. Same A6 class as before.")

print()
print("=== T7: shape contract vs LIVE consumers ===")
print("  transducer IN  : [batch, M, 8] real      <- matches clifford_bivector_tiles [8192, 8]")
print("  transducer OUT : [batch, 65536] complex")
print("  live syncytium : relax takes real phases [n=8192] (one per tile)")
print("  live planner   : real [num_blocks, 8] at the planner boundary")
print("  -> NO live consumer accepts a complex 65536 wave. A phase-extract + reshape")
print("     adapter is still required. C3 is HALF-satisfied, not satisfied.")

print()
print("=== VERDICT ===")
print("  A8 locality                 FIXED (block-local normalization)")
print("  round-trip                  FIXED (~1e-7, was 0.9964) -- but on directions only")
print(f"  A7 grade integrity          PARTIAL: {bad} of {len(CODE)} Hodge pairs wrong vs LIVE order")
print("  A6 redundancy               PERSISTS (transverse half is base*rotor)")
print("  magnitude                   DISCARDED (scale-invariant)")
print("  |psi| = sqrt(M)             'unit hypersphere' claim is per-block, not global")
print("  C3 shape                    HALF-satisfied (consumes [M,8]; emits unconsumed complex)")
print("  DISPOSITION: BOUNDED_IMPLEMENTABLE_WITH_REINDEX -- swap indices 4<->6, or derive the")
print("               pairing from CLIFFORD_GRADES + a Hodge table instead of literals.")
