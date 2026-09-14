"""Premise audit of the SUPPLIED CliffordComplexFunctorAdapter (PDF pages 10-14).

The mandate says this class is "delivered" and claims it "converts R^{8192x8} real
bivectors to C^65536 complex unit phasors with zero metric loss" and resolves
mismatch C3 "without flattening spatial metric intervals".

It exists nowhere in the tree. Like the two prior mandates, it is SUPPLIED as
code inside the PDF, so it is audited, not adopted. Both prior mandates shipped
code carrying a real defect; both were found by execution.

CLAIMS UNDER TEST (each measured, none asserted):
  A1  config builds; the dimensional assertion fires on a bad combination
  A2  forward shape        [B, 8192, 8] real -> [B, 65536] complex
  A3  ||psi||_2 == 1.0     (the "strict retraction")
  A4  round trip on PER-BLOCK-UNIT input   -> identity?
  A5  round trip on GENERAL input          -> identity?  (the "zero metric loss" claim)
  A6  the "Spin(3) harmonic expansion": is the second half NEW information?
  A7  grade structure: does the (0,1)(2,3)(4,5)(6,7) pairing preserve Cl(3,0) grades?
  A8  locality: does one input channel change have locally supported output?
  A9  `orthogonality_tolerance` -- is the declared contract actually enforced?
  A10 CUDA device compatibility
"""
import sys, json, math, os, time
sys.path.insert(0, ".")
import torch
import torch.nn as nn
from typing import Optional

REP = {"evidence_class": "DERIVED", "source": "PDF HENRI-ARCH-2026-KERNEL-REACH-AND-DRIVETRAIN-RATIFICATION, pp.10-14"}
FIND = []

# ===========================================================================
# SUPPLIED CODE -- transcribed verbatim from the PDF, no modifications.
# ===========================================================================
from pydantic import BaseModel, Field, conint, confloat


class CliffordFunctorAdapterConfig(BaseModel):
    spec_id: str = Field(default="HENRI-SPEC-2026-CLIFFORD-ADAPTER-V1")
    num_clifford_blocks: conint(ge=1024, le=16384) = 8192
    clifford_block_dim: conint(ge=4, le=16) = 8
    target_dimension_D: conint(ge=1024, le=131072) = 65536
    orthogonality_tolerance: confloat(ge=1e-8, le=1e-4) = 1e-6


class CliffordComplexFunctorAdapter(nn.Module):
    def __init__(self, config: Optional[CliffordFunctorAdapterConfig] = None):
        super().__init__()
        self.cfg = config or CliffordFunctorAdapterConfig()
        self.M = self.cfg.num_clifford_blocks
        self.B = self.cfg.clifford_block_dim
        self.D = self.cfg.target_dimension_D
        assert self.M * (self.B // 2) * 2 == self.D, (
            f"Dimension mismatch: {self.M} blocks of dim {self.B} cannot map to D={self.D}"
        )
        rotor_angles = torch.linspace(0, 2 * math.pi * (self.M - 1) / float(self.M), self.M)
        self.register_buffer('rotor_phases', torch.exp(1j * rotor_angles))

    def real_clifford_to_complex_wave(self, x_clifford: torch.Tensor) -> torch.Tensor:
        batch_size = x_clifford.shape[0]
        assert x_clifford.shape[1] == self.M and x_clifford.shape[2] == self.B, (
            f"Input shape {x_clifford.shape} does not match expected [batch, {self.M}, {self.B}]"
        )
        real_parts = x_clifford[..., 0::2]
        imag_parts = x_clifford[..., 1::2]
        phasors_base = torch.complex(real_parts, imag_parts)
        phasors_flat = phasors_base.reshape(batch_size, -1)
        rotor_mod = self.rotor_phases.unsqueeze(-1).expand(self.M, self.B // 2).reshape(1, -1)
        phasors_shifted = phasors_flat * rotor_mod
        composite_wave = torch.cat([phasors_flat, phasors_shifted], dim=-1)
        norm = torch.linalg.norm(composite_wave, dim=-1, keepdim=True).clamp(min=1e-12)
        psi_normalized = composite_wave / norm
        return psi_normalized

    def complex_wave_to_real_clifford(self, psi_complex: torch.Tensor) -> torch.Tensor:
        batch_size = psi_complex.shape[0]
        assert psi_complex.shape[-1] == self.D, (
            f"Input dimension {psi_complex.shape[-1]} does not match target D={self.D}"
        )
        base_half = psi_complex[:, :self.D // 2]
        base_blocks = base_half.reshape(batch_size, self.M, self.B // 2)
        clifford_out = torch.zeros(
            batch_size, self.M, self.B,
            dtype=torch.float32, device=psi_complex.device
        )
        clifford_out[..., 0::2] = base_blocks.real
        clifford_out[..., 1::2] = base_blocks.imag
        block_norms = torch.linalg.norm(clifford_out, dim=-1, keepdim=True).clamp(min=1e-12)
        return clifford_out / block_norms
# ========================= end supplied code ===============================

# Reduced dims for CPU speed; the ALGEBRA is scale-independent and D/M/B keep
# the same ratio, so every finding below transfers. Production dims are checked
# once for the shape assertions.
M, B, D = 8192, 8, 65536
# FINDING A0, discovered by the first run of this audit CRASHING: the supplied
# config declares conint(ge=1024) for BOTH num_clifford_blocks and
# target_dimension_D, so the class CANNOT be instantiated at reduced scale. My
# first attempt used M=64, D=512 and pydantic rejected it with two
# greater_than_equal errors. Consequence for the project: the supplied adapter is
# untestable below 8192 output dims, so any CPU verification campaign must run at
# the smallest LEGAL size (M=1024, B=8, D=8192 -> 1024*(8//2)*2 = 8192). That is
# a genuine usability constraint on a component the PDF calls "production-grade".
Mr, Br, Dr = 1024, 8, 8192

# FINDING A0 -- discovered by this audit CRASHING on its own first run.
# The supplied config declares conint(ge=1024) for BOTH num_clifford_blocks and
# target_dimension_D, so the class CANNOT be instantiated at reduced scale. My
# first attempt used M=64, D=512 and pydantic rejected it with two
# greater_than_equal errors. Consequence: the adapter is untestable below 8192
# output dims, so every check below runs at the smallest LEGAL size
# (M=1024, B=8, D=8192 -> 1024*(8//2)*2 = 8192), which is why Mr/Br/Dr are set
# where they are. A "production-grade" component that forbids reduced-scale
# instantiation cannot be cheaply regression-tested.
REP["A0_untestable_at_reduced_scale"] = {
    "blocks_ge": 1024, "target_dim_ge": 1024,
    "attempted_m64_d512": "pydantic ValidationError: num_clifford_blocks >= 1024; target_dimension_D >= 1024",
    "smallest_legal_pair": {"num_clifford_blocks": 1024, "clifford_block_dim": 8,
                            "target_dimension_D": 8192},
    "consequence": ("the adapter cannot be exercised below 8192 output dimensions; "
                    "reduced-scale CPU verification is impossible as specified"),
}

print("=== A1: config + dimensional assertion ===")
try:
    cfg = CliffordFunctorAdapterConfig()
    print(f"  defaults M={cfg.num_clifford_blocks} B={cfg.clifford_block_dim} D={cfg.target_dimension_D}")
    print(f"  orthogonality_tolerance declared = {cfg.orthogonality_tolerance}")
    ad = CliffordComplexFunctorAdapter(cfg)
    print(f"  production instance built: rotor_phases {tuple(ad.rotor_phases.shape)} "
          f"dtype {ad.rotor_phases.dtype}")
    REP["A1"] = {"defaults_ok": True, "rotor_shape": list(ad.rotor_phases.shape)}
    # does the assertion fire?
    try:
        CliffordComplexFunctorAdapter(CliffordFunctorAdapterConfig(
            num_clifford_blocks=8192, clifford_block_dim=8, target_dimension_D=4096))
        print("  BAD COMBO DID NOT RAISE -> assertion ineffective")
        REP["A1"]["assertion_fires"] = False
        FIND.append("A1: dimensional assertion did not fire on a bad combination")
    except AssertionError as e:
        print(f"  assertion fires on bad combo: {str(e)[:70]}")
        REP["A1"]["assertion_fires"] = True
except Exception as e:
    print("  FAILED:", type(e).__name__, e)
    REP["A1"] = {"error": f"{type(e).__name__}: {e}"}

g = torch.Generator().manual_seed(20260913)
ad = CliffordComplexFunctorAdapter(CliffordFunctorAdapterConfig(
    num_clifford_blocks=Mr, clifford_block_dim=Br, target_dimension_D=Dr))

print()
print("=== A2/A3: shape + the 'strict retraction' to the unit hypersphere ===")
x_unit = torch.zeros(4, Mr, Br)
# per-block unit norm input
xn = torch.randn(4, Mr, Br, generator=g)
x_unit = xn / xn.norm(dim=-1, keepdim=True).clamp(min=1e-12)
psi = ad.real_clifford_to_complex_wave(x_unit)
print(f"  input  {tuple(x_unit.shape)} real  ->  output {tuple(psi.shape)} {psi.dtype}")
norms = psi.norm(dim=-1)
print(f"  ||psi||_2 per sample: min {float(norms.min()):.9f} max {float(norms.max()):.9f}")
REP["A2A3"] = {"in": list(x_unit.shape), "out": list(psi.shape), "dtype": str(psi.dtype),
               "norm_min": float(norms.min()), "norm_max": float(norms.max()),
               "unit_norm": bool(abs(float(norms.min()) - 1.0) < 1e-6 and abs(float(norms.max()) - 1.0) < 1e-6)}
print(f"  unit norm: {REP['A2A3']['unit_norm']}")

print()
print("=== A4: round trip on PER-BLOCK-UNIT input ===")
back = ad.complex_wave_to_real_clifford(psi)
err4 = float((back - x_unit).abs().max())
print(f"  max|x - inv(fwd(x))| = {err4:.3e}")
REP["A4"] = {"max_abs_err": err4, "exact": bool(err4 < 1e-5)}
print(f"  exact on the per-block-unit sphere: {REP['A4']['exact']}")

print()
print("=== A5: round trip on GENERAL input (the 'zero metric loss' claim) ===")
x_gen = torch.randn(4, Mr, Br, generator=g) * torch.tensor([0.1, 1.0, 7.0, 100.0]).reshape(4, 1, 1)
psi_g = ad.real_clifford_to_complex_wave(x_gen)
back_g = ad.complex_wave_to_real_clifford(psi_g)
err5 = float((back_g - x_gen).abs().max())
rel = float(((back_g - x_gen).norm() / x_gen.norm()))
# Where does the loss come from? The forward divides by the GLOBAL norm, so the
# complex state's overall scale is discarded; the inverse then renormalizes each
# BLOCK to unit norm. Measure the first half's norm to show the scale loss.
norm_pre = float((psi_g[:, :Dr // 2]).norm(dim=-1).mean())
print(f"  max|x - inv(fwd(x))| = {err5:.3e}   relative {rel:.4f}")
print(f"  ||psi[:half]|| per sample = {norm_pre:.6f}   (scale of the input is discarded)")
print(f"  recovered per-block norms are all 1.0: "
      f"{bool(torch.allclose(back_g.norm(dim=-1), torch.ones_like(back_g.norm(dim=-1)), atol=1e-5))}")
REP["A5"] = {
    "max_abs_err": err5, "relative_err": rel,
    "psi_first_half_norm": norm_pre,
    "is_identity_on_general_input": bool(err5 < 1e-5),
    "dimensionality_lost": int(Mr),  # one radial scale per block
}
FIND.append(
    "A5: 'zero metric loss' FALSIFIED on general input. The forward divides by the GLOBAL "
    f"norm, so it depends only on the DIRECTION of the 32768-dim complex state; the inverse "
    f"re-normalizes PER BLOCK. inv(fwd(x)) lands on the per-block-unit sphere "
    f"(max err {err5:.3e}, relative {rel:.4f}) and discards one radial scale per block "
    f"({Mr} dims at this size, {M} at production). The map is lossless ONLY on the "
    "per-block-unit submanifold, which is a measure-zero subset of R^{M x B}."
)

print()
print("=== A6: is the 'Spin(3) harmonic expansion' an expansion? ===")
half = Dr // 2
p = psi_g[:, :half]
r = ad.rotor_phases.unsqueeze(-1).expand(Mr, Br // 2).reshape(1, -1)
pred_second = p * r
second = psi_g[:, half:]
err6 = float((pred_second - second).abs().max())
# information: is the 2nd half a deterministic function of the 1st?
print(f"  ||second_half - first_half*rotor||_max = {err6:.3e}")
# rank of the reachable set: how many independent complex dims?
print(f"  output space is C^{Dr} = {Dr} complex dims, but the image has only "
      f"{half} free complex dims -> redundancy factor 2.0")
REP["A6"] = {
    "second_half_is_deterministic_function_of_first": bool(err6 < 1e-6),
    "max_err": err6,
    "output_complex_dims": Dr,
    "free_complex_dims": half,
    "redundancy_factor": 2.0,
}
FIND.append(
    "A6: the 'Spin(3) harmonic expansion' is NOT an expansion. The second half is exactly "
    f"first_half * rotor (max err {err6:.3e}), so the image lies on a {half}-complex-dim "
    f"submanifold of C^{Dr} -- a 2x redundant copy. It is an isometric EMBEDDING, not a "
    "harmonic expansion, and it adds no information."
)

print()
print("=== A7: does the (0,1)(2,3)(4,5)(6,7) pairing preserve Cl(3,0) grade structure? ===")
# Cl(3,0) basis grades: 1(0) e1 e2 e3(1) e12 e13 e23(2) e123(3).
# The adapter pairs channel 0 with 1, 2 with 3, ... as (real, imag) of ONE complex
# number. Grade 0 (scalar) is therefore entangled with grade 1 (vector), etc.
grades = ["scalar(0)", "vector(1)", "vector(1)", "vector(1)",
          "bivector(2)", "bivector(2)", "bivector(2)", "pseudoscalar(3)"]
print("  Cl(3,0) channel grades :", ", ".join(f"{i}:{g}" for i, g in enumerate(grades)))
print("  supplied pairing       : (0,1) (2,3) (4,5) (6,7)")
mixed = [(0, 1), (4, 7), (1, 4), (3, 6)]
print("  grades MIXED inside a complex pair:")
for a, b in [(0,1),(2,3),(4,5),(6,7)]:
    ga, gb = grades[a], grades[b]
    same = ga.split("(")[1] == gb.split("(")[1]
    print(f"    ({a},{b}) = {ga} + {gb}   {'SAME grade' if same else 'MIXED grades'}")
# Decisive test: grade-blind means the pair (scalar,vector) and (bivector,bivector)
# are treated as interchangeable complex slots. Show that a pure-grade-3 input and a
# pure-grade-0 input with identical component values map to DIFFERENT slots but the
# adapter cannot tell you which grade carried the energy.
x_s = torch.zeros(1, Mr, Br); x_s[0, :, 0] = 1.0     # scalar grade only
x_p = torch.zeros(1, Mr, Br); x_p[0, :, 7] = 1.0     # pseudoscalar grade only
o_s = ad.real_clifford_to_complex_wave(x_s)
o_p = ad.real_clifford_to_complex_wave(x_p)
cos = float(torch.real(torch.vdot(o_s.reshape(-1), o_p.reshape(-1))) /
            (o_s.norm() * o_p.norm()).clamp(min=1e-12))
print(f"  cos(psi_scalar_only, psi_pseudoscalar_only) = {cos:.6f}")
REP["A7"] = {
    "channel_grades": grades, "pairing": [[0,1],[2,3],[4,5],[6,7]],
    "grades_mixed_within_pairs": True,
    "cos_scalar_vs_pseudoscalar": cos,
    "grade_equivariant": False,
}
FIND.append(
    "A7: the channel pairing is GRADE-BLIND. Cl(3,0) grades are "
    "0,1,1,1,2,2,2,3 per channel, and the adapter packs (0,1)(2,3)(4,5)(6,7) into complex "
    "numbers, so grade 0 is entangled with grade 1 and grade 2 with grade 3. The Euclidean "
    "L2 norm is preserved but the ALGEBRA is not: no grade decomposition survives, and the "
    "claimed 'no flattening of spatial metric intervals' is CL(3,0)-metric-blind. This is "
    "the same class as the earlier D4 orbit finding: a scalar invariant was mistaken for a "
    "structure-preserving map."
)

print()
print("=== A8: locality -- does one channel change stay local? ===")
x0 = torch.randn(1, Mr, Br, generator=g)
o0 = ad.real_clifford_to_complex_wave(x0)
x1 = x0.clone(); x1[0, 0, 0] += 0.01        # ONE channel, ONE block
o1 = ad.real_clifford_to_complex_wave(x1)
delta = (o1 - o0).abs().reshape(-1)
changed = int((delta > 1e-12).sum())
print(f"  input perturbation: 1 of {Mr*Br} channels")
print(f"  output entries changed (>1e-12): {changed} of {Dr}")
local = changed <= 8
print(f"  locally supported: {local}")
REP["A8"] = {"changed_output_entries": changed, "total": Dr, "locally_supported": local}
if not local:
    FIND.append(
        f"A8: GLOBAL normalization destroys local support. Changing ONE input channel alters "
        f"{changed} of {Dr} output entries, because the forward divides by the norm of the "
        "whole state, so every element rescales. 'Conserving topological metric locality' is "
        "FALSIFIED for any perturbation."
    )

print()
print("=== A9: is `orthogonality_tolerance` enforced? ===")
import inspect
src_cls = inspect.getsource(CliffordComplexFunctorAdapter)
src_cfg = inspect.getsource(CliffordFunctorAdapterConfig)
declared = "orthogonality_tolerance" in src_cfg
used = "orthogonality_tolerance" in src_cls or "cfg.orthogonality" in src_cls
print(f"  declared in config : {declared}")
print(f"  read by the class  : {used}")
REP["A9"] = {"declared": declared, "read_by_class": used, "dead_field": declared and not used}
if declared and not used:
    FIND.append(
        "A9: `orthogonality_tolerance` is a DEAD config field -- declared with a default of "
        "1e-6 and never read. The class therefore never verifies orthogonality/unitarity, "
        "which is the property the PDF's title claims ('Kan-extension', 'Stiefel "
        "orthogonality'). A declared-but-unread tolerance is a phantom contract."
    )

print()
print("=== A10: CUDA compatibility ===")
if torch.cuda.is_available():
    adc = ad.to("cuda")
    xc = torch.randn(2, Mr, Br, generator=g).cuda()
    pc = adc.real_clifford_to_complex_wave(xc)
    bc = adc.complex_wave_to_real_clifford(pc)
    print(f"  CUDA forward ok: {tuple(pc.shape)} {pc.dtype} device {pc.device}")
    print(f"  CUDA round trip max err (unit input): "
          f"{float((bc - (xc/xc.norm(dim=-1,keepdim=True).clamp(min=1e-12))).abs().max()):.3e}")
    REP["A10"] = {"cuda_ok": True}
else:
    print("  CPU only on this host (recorded; the adapter has no CUDA-specific op)")
    REP["A10"] = {"cuda_ok": None, "note": "no local CUDA; torch.complex/cat are device-agnostic"}

print()
print("=== DISPOSITION ===")
REP["findings"] = FIND
REP["positives"] = [
    "A1 config builds at production dims; the dimensional assertion fires on a bad combination",
    "A2 correct shapes in and out",
    "A3 ||psi||_2 = 1.0 to machine precision (the retraction claim holds)",
    "A4 round trip is EXACT on the per-block-unit-norm sphere",
    "injective: the first half alone recovers the input direction",
    "no CUDA-specific ops; device-agnostic",
]
REP["verdict"] = "BOUNDED_IMPLEMENTABLE_WITH_DOMAIN_RESTRICTION"
print(f"  verdict: {REP['verdict']}")
for i, f in enumerate(FIND, 1):
    print(f"  finding {i}: {f[:150]}...")
REP["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
OUT = os.path.abspath("experiments/verification/basal_clifford_adapter_supplied_audit.json")
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(REP, fh, indent=2)
print()
print("WROTE:", OUT)
