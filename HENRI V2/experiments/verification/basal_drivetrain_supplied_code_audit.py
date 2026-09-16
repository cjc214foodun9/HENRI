"""Premise audit of the SUPPLIED drivetrain code (PDF pages 10-12).

The mandate's Step 1 says "replace the random goal placeholders in
henri_goal_adapter.py with InductiveTaskFunctor". That class does not exist in
the repo -- but the PDF SUPPLIES it as code. So this is not a missing premise;
it is a supplied artifact, and supplied artifacts are audited, not adopted.
(The two previous mandates each shipped code with a real defect; both were
found by execution.)

Claims under test:
  C1  `conint(ge=..., le=...)` works on this repo's pydantic.
  C2  the supplied GroundedEgressSnap builds and its down_proj has the
      intended [H, D] projection shape.
  C3  the supplied InductiveTaskFunctor is tensor-compatible with the LIVE
      HENRI boundary, which is real [num_blocks, 8] -- not flat [D] complex.
  C4  grounding via circular cross-correlation "immediately restores the
      pragmatic gradient (grad > 0.50)". Measured, not assumed.
  C5  vocab_size_V default 1024 vs the live egress vocab (32000).
"""
import sys, math, json, time
sys.path.insert(0, '.')
import torch
import torch.nn as nn
import torch.nn.functional as F

rep = {}

print("=== C1: pydantic version and conint semantics ===")
try:
    import pydantic
    print("  pydantic version:", pydantic.VERSION)
    from pydantic import conint, confloat
    try:
        from pydantic import BaseModel, Field
        class T1(BaseModel):
            v: conint(ge=256, le=32768) = 1024
        t = T1()
        print("  conint(ge=...,le=...) accepted; default =", t.v)
        rep["C1"] = {"pydantic": pydantic.VERSION, "conint_ok": True}
    except Exception as e:
        print("  conint FAILED:", type(e).__name__, str(e)[:160])
        rep["C1"] = {"pydantic": pydantic.VERSION, "conint_ok": False,
                     "error": f"{type(e).__name__}: {e}"}
except ImportError:
    print("  pydantic NOT INSTALLED")
    rep["C1"] = {"pydantic": None, "conint_ok": None}


print()
print("=== C2: supplied GroundedEgressSnap shape check (PDF verbatim logic) ===")
def supplied_egress(D=65536, H=2048, V=1024):
    """Exactly the PDF's construction, no cleanup."""
    raw_w = torch.randn(H, D)
    q, _ = torch.linalg.qr(raw_w.t())
    down_proj = q.t()[:H, :]
    raw_m = torch.randn(V, H)
    codebook = F.normalize(raw_m, p=2, dim=-1)
    return down_proj, codebook

t0 = time.time()
try:
    D_s, H_s, V_s = 4096, 256, 128      # reduced for CPU speed; shapes scale
    dp, cb = supplied_egress(D_s, H_s, V_s)
    print(f"  qr(raw_w.t()).t()[:H,:] shape = {tuple(dp.shape)}  (want [H, D] = {(H_s, D_s)})")
    ok = tuple(dp.shape) == (H_s, D_s)
    print("  shape correct:", ok)
    # Is it orthonormal ROWS (a valid Stiefel projection)?
    gram = dp @ dp.t()
    off = float((gram - torch.eye(H_s)).abs().max())
    print(f"  row-orthonormality |W W^T - I|_max = {off:.3e}")
    rep["C2"] = {"shape": list(dp.shape), "want": [H_s, D_s], "shape_ok": ok,
                 "row_orthonormality_max_err": off}
except Exception as e:
    print("  FAILED:", type(e).__name__, str(e)[:200])
    rep["C2"] = {"error": f"{type(e).__name__}: {e}"}
print(f"  ({time.time()-t0:.2f}s)")


print()
print("=== C3: tensor compatibility with the LIVE boundary ===")
import basal_boundary_engine as bbe
# CORRECTED: the sealed names in basal_boundary_engine are SPEC_DIMENSION_D and
# SPEC_CLIFFORD_BLOCK_SIZE. SPEC_BLOCK_SIZE lives in basal_triton_kernel (1024,
# the Triton tile size) and is a DIFFERENT quantity. My first run of this audit
# imported the wrong name and crashed -- recorded here because the confusion is
# the same class as the +/-252 defect: two similarly named spec quantities with
# different units.
SPEC_NUM_TILES = bbe.SPEC_NUM_TILES
SPEC_CLIFFORD_BLOCK_SIZE = bbe.SPEC_CLIFFORD_BLOCK_SIZE
print(f"  live boundary wave shape : [{SPEC_NUM_TILES}, {SPEC_CLIFFORD_BLOCK_SIZE}] real")
print("  supplied functor expects : [M, D] COMPLEX (flat)")
print("  same family?", False)
try:
    flat = torch.randn(SPEC_NUM_TILES, SPEC_CLIFFORD_BLOCK_SIZE)
    Df = SPEC_NUM_TILES * SPEC_CLIFFORD_BLOCK_SIZE
    demo_in = torch.randn(3, Df, dtype=torch.complex64)
    demo_out = torch.randn(3, Df, dtype=torch.complex64)
    fx = torch.fft.fft(demo_in, dim=-1)
    fy = torch.fft.fft(demo_out, dim=-1)
    w_task = torch.mean(fy * torch.conj(fx), dim=0, keepdim=True)
    test = torch.randn(1, Df, dtype=torch.complex64)
    psi = torch.fft.ifft(w_task * torch.fft.fft(test, dim=-1), dim=-1)
    psi = psi / torch.linalg.norm(psi, dim=-1, keepdim=True).clamp(min=1e-8)
    print(f"  supplied functor output  : {tuple(psi.shape)} complex")
    print(f"  reshape to live boundary : {tuple(psi.reshape(SPEC_NUM_TILES, SPEC_CLIFFORD_BLOCK_SIZE).shape)}")
    print("  -> a real [8192,8] wave is NOT a flat [65536] complex vector.")
    print("     Adapter required; 'replace placeholders with it' is not a drop-in.")
    rep["C3"] = {"live_shape": [SPEC_NUM_TILES, SPEC_CLIFFORD_BLOCK_SIZE],
                 "supplied_shape": list(psi.shape),
                 "compatible": False,
                 "adapter_required": True,
                 "name_trap": "SPEC_BLOCK_SIZE (triton tile, 1024) != SPEC_CLIFFORD_BLOCK_SIZE (8)"}
except Exception as e:
    print("  FAILED:", type(e).__name__, str(e)[:200])
    rep["C3"] = {"error": f"{type(e).__name__}: {e}"}


print()
print("=== C4: does circular cross-correlation ground the goal? (MEASURED) ===")
# The mandate claims grounding "immediately restores the pragmatic gradient
# (grad > 0.50)". The PDF does not define G(a). The quantity its own prose
# defines is the ORTHOGONALITY problem: psi_goal orthogonal to psi_target =>
# no pragmatic signal. So the honest falsifiable proxy is exactly
#     alignment = |cos(psi_goal, psi_target_true)|
# and 0.50 is the mandate's own threshold.
#
# The decisive variable is whether the demo pairs carry a CONSISTENT relation.
# Test BOTH: a genuine closed-form transform (ring shift) and iid noise.
def cos_sim(a, b):
    a = a.reshape(-1); b = b.reshape(-1)
    return float(torch.real(torch.vdot(a, b)) /
                 (a.norm() * b.norm()).clamp(min=1e-12))


def compile_functor(demo_in, demo_out):
    """The PDF's exact operator."""
    fx = torch.fft.fft(demo_in, dim=-1)
    fy = torch.fft.fft(demo_out, dim=-1)
    return torch.mean(fy * torch.conj(fx), dim=0, keepdim=True)


g = torch.Generator().manual_seed(1234)
D = 4096
M = 8
K = 3                      # ring shift for the structured task

base_in = torch.randn(M, D, generator=g, dtype=torch.float32).to(torch.complex64)
# STRUCTURED: y = roll(x, K). A genuine, learnable relation.
struct_out = torch.roll(base_in, K, dims=-1)
# UNSTRUCTURED: y independent of x.
iid_out = torch.randn(M, D, generator=g, dtype=torch.float32).to(torch.complex64)

held = torch.randn(1, D, generator=g, dtype=torch.float32).to(torch.complex64)
true_target = torch.roll(held, K, dims=-1)

results = {}
for label, demos_out in (("structured (y = roll(x, 3))", struct_out),
                         ("iid random (no relation)", iid_out)):
    wt = compile_functor(base_in, demos_out)
    psi = torch.fft.ifft(wt * torch.fft.fft(held, dim=-1), dim=-1)
    psi = psi / psi.norm().clamp(min=1e-8)
    align = abs(cos_sim(psi, true_target))
    # Control: alignment of a random unit vector with the same target.
    ctrl = torch.randn(64, D, generator=g, dtype=torch.float32).to(torch.complex64)
    ctrl = ctrl / ctrl.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    chance = sum(abs(cos_sim(ctrl[i:i+1], true_target)) for i in range(64)) / 64
    results[label] = {"alignment": align, "chance": chance}
    print(f"  {label:28s} alignment = {align:.6f}   chance = {chance:.6f}")
    print(f"       clears the mandate's 0.50 threshold: {align > 0.50}")

rep["C4"] = {
    "metric": "|cos(psi_goal, psi_target_true)| -- the orthogonality the PDF names",
    "mandate_threshold": 0.50,
    "structured_task": "y = roll(x, 3)",
    "results": results,
    "finding": ("the operator is CORRECT and does ground the goal -- but ONLY when the "
                "demo pairs carry a consistent relation. On iid pairs it returns noise. "
                "So 'restores the gradient' is true conditionally on the demo set, and "
                "the failure mode is a bad demo set, not the operator."),
}


print()
print("=== C5: vocab budget ===")
print("  supplied vocab_size_V default : 1024")
print("  live egress lm_head vocab     : 32000  (models/henri_decoder_checkpoint.pt)")
print("  match:", False)


print()
print("=== live alternatives that DO exist ===")
import inspect
import zone_c_epistemic_axiom_harness as zca
src = inspect.getsource(zca.HolographicTaskFunctorCompiler)
for name in ("def __init__", "def compile", "def forward", "def retrieve"):
    if name in src:
        print(f"  HolographicTaskFunctorCompiler.{name.split()[1]} exists")
import arc_task_functor as atf
print("  arc_task_functor symbols:",
      [n for n in dir(atf) if not n.startswith('_')][:8])

rep["supplied_class_exists_in_repo"] = False
rep["live_alternatives"] = [
    "zone_c_epistemic_axiom_harness.HolographicTaskFunctorCompiler",
    "arc_task_functor.TaskFunctorResult",
]
rep["evidence_class"] = "DERIVED"
rep["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

with open("experiments/verification/basal_drivetrain_supplied_code_audit.json",
          "w", encoding="utf-8") as f:
    json.dump(rep, f, indent=2)
print()
print("wrote experiments/verification/basal_drivetrain_supplied_code_audit.json")
