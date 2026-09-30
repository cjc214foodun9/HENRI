"""TIER-2 STRUCTURAL GATE (CPU-runnable). No task claim, no GPU needed.

WHAT THIS PROVES
  Item 5 of the resume order wires WaveJEPA into the unified VLA "behind its own
  gate". The gate has two halves. This file is the STRUCTURAL half, which is
  falsifiable on CPU. The MEASURED half (T2-a 3-step cosine >= 0.92, T2-c unitary
  preserved) requires the RTX 5090 and is NOT claimed here.

CHECKS
  [1] SUBSTRATE PINNING. WaveJEPA accepted `encoder` and uses the INJECTED
      object (identity), instead of silently constructing its own with defaults.
      This was a real defect: wave_jepa.py built HENRIVisionEncoder(...) with
      defaults (spatial_basis_kind="default", bg_mask=False, fused_superpose=False)
      while the locked Stage-1 config is (incommensurate, bg_mask=True,
      fused_superpose=True, parity_scipy=True). Measuring JEPA on one substrate
      and the contract on another is a silent cross-fixture comparison.

  [2] SUBSTRATE IS OBSERVABLE, not cosmetic. Two encoders differing ONLY in
      spatial_basis_kind must produce different waves. If they did not, the
      "substrate" distinction would be meaningless and [1] would be theatre.

  [3] LEGACY PRESERVATION. encoder=None must reproduce the old construction
      byte-for-byte (same wave as an explicitly-injected default-config encoder).

  [4] FAIL CLOSED. HENRIUnifiedVLAModel.predict_future without a world_model
      must RAISE, not return a fabricated prediction. A stand-in here would let
      a caller mistake a mock for a real transition.

  [5] ROLLOUT SHAPE + FINITENESS. rollout returns len(actions)+1 states, all
      finite. Norms are NOT asserted as evidence of unitarity.

Run:  python experiments/verification/test_tier2_wiring.py
Exit: 0 = all structural checks pass. Non-zero = a check failed.
"""
from __future__ import annotations

import os
import sys

import torch

# repo root = two levels up from experiments/verification/
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from henri_vision_encoder import HENRIVisionEncoder          # noqa: E402
from wave_jepa import WaveJEPA                                # noqa: E402
from henri_unified_vla import HENRIUnifiedVLAModel            # noqa: E402

# d_model = num_blocks * 8 is required by encode_context's .view(num_blocks, 8)
D = 512
NB = 64
R = 8
GRID_A = [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]
GRID_B = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 0, 0]]

FAILS: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILS.append(label)


def mk_enc(**kw):
    base = dict(d_model=D, k_blocks=NB, device="cpu")
    base.update(kw)
    return HENRIVisionEncoder(**base)


print("=" * 78)
print("[1] SUBSTRATE PINNING: injected encoder is the one used")
print("=" * 78)
locked_enc = mk_enc(spatial_basis_kind="incommensurate", bg_mask=True,
                    fused_superpose=True, parity_scipy=True)
jepa = WaveJEPA(d_model=D, num_blocks=NB, r_rank=R, device="cpu",
                encoder=locked_enc)
check("WaveJEPA.encoder IS the injected object (identity)",
      jepa.encoder is locked_enc,
      f"injected={id(locked_enc)} used={id(jepa.encoder)}")
check("locked substrate survives injection",
      jepa.encoder.spatial_basis_kind == "incommensurate"
      and jepa.encoder.bg_mask is True
      and jepa.encoder.fused_superpose is True
      and jepa.encoder.parity_scipy is True,
      f"kind={jepa.encoder.spatial_basis_kind} bg={jepa.encoder.bg_mask} "
      f"fused={jepa.encoder.fused_superpose} scipy={jepa.encoder.parity_scipy}")

print()
print("=" * 78)
print("[2] THE SUBSTRATE IS OBSERVABLE (else [1] is theatre)")
print("=" * 78)
enc_def = mk_enc()                                   # defaults
enc_inc = mk_enc(spatial_basis_kind="incommensurate", bg_mask=True)
w_def = enc_def.encode_grid(GRID_A).detach()
w_inc = enc_inc.encode_grid(GRID_A).detach()
diff = float((w_def - w_inc).abs().max())
check("default vs incommensurate substrates DIFFER on the same grid",
      diff > 1e-6, f"max|diff|={diff:.3e}")
check("both substrates are unit waves",
      abs(float(w_def.norm()) - 1.0) < 1e-5
      and abs(float(w_inc.norm()) - 1.0) < 1e-5,
      f"|def|={float(w_def.norm()):.7f} |inc|={float(w_inc.norm()):.7f}")

print()
print("=" * 78)
print("[3] LEGACY PRESERVATION: encoder=None reproduces the old behaviour")
print("=" * 78)
jepa_legacy = WaveJEPA(d_model=D, num_blocks=NB, r_rank=R, device="cpu")
jepa_inject_default = WaveJEPA(d_model=D, num_blocks=NB, r_rank=R, device="cpu",
                               encoder=mk_enc())
a_t = torch.randn(1, NB, 8)
psi = jepa_legacy.encode_context(GRID_A)
p_legacy = jepa_legacy.predict_future_latent(psi, a_t)
p_inject = jepa_inject_default.predict_future_latent(psi, a_t)
d_legacy = float((p_legacy - p_inject).abs().max())
check("legacy (encoder=None) == explicit default-config injection",
      d_legacy < 1e-6, f"max|diff|={d_legacy:.3e}")
check("legacy encoder is NOT the injected object (own construction)",
      jepa_legacy.encoder is not jepa_inject_default.encoder)

print()
print("=" * 78)
print("[4] FAIL CLOSED: no world_model => raise, never a fabricated prediction")
print("=" * 78)
vla = HENRIUnifiedVLAModel(tokenizer=locked_enc, orchestrator=None,
                          action_gate=None, device="cpu")
check("world_model defaults to None (Stage-1 path unchanged)",
      vla.world_model is None, f"{vla.world_model!r}")
raised, exc_name = False, ""
try:
    vla.predict_future(psi, a_t)
except RuntimeError as exc:
    raised, exc_name = True, str(exc)[:60]
except Exception as exc:                                       # noqa: BLE001
    exc_name = f"WRONG TYPE {type(exc).__name__}"
check("predict_future RAISES RuntimeError without a world model",
      raised and exc_name.startswith("WORLD_MODEL_NOT_WIRED"), exc_name)

print()
print("=" * 78)
print("[5] WIRED PATH: rollout shape + finiteness (no unitarity claim)")
print("=" * 78)
vla_w = HENRIUnifiedVLAModel(tokenizer=locked_enc, orchestrator=None,
                            action_gate=None, device="cpu",
                            world_model=jepa)
one = vla_w.predict_future(psi, a_t)
check("predict_future returns a wave of the state's shape",
      tuple(one.shape) == tuple(psi.shape), f"{tuple(one.shape)} vs {tuple(psi.shape)}")
check("prediction is finite", bool(torch.isfinite(one).all()))

acts = [torch.randn(1, NB, 8) for _ in range(3)]
states = vla_w.rollout(psi, acts)
check("rollout returns len(actions)+1 states", len(states) == 4, f"{len(states)}")
check("all rollout states finite",
      all(bool(torch.isfinite(s).all()) for s in states))
norms = [float(s.norm()) for s in states]
print(f"  recorded norms (NOT asserted as unitarity): "
      f"{', '.join(f'{n:.6f}' for n in norms)}")
check("norms recorded, not used as a unitary claim (see T2-c)",
      len(norms) == 4)

print()
print("=" * 78)
if FAILS:
    print(f"TIER2_STRUCTURAL: FAIL ({len(FAILS)} check(s))")
    for f in FAILS:
        print("   -", f)
    sys.exit(1)
print("TIER2_STRUCTURAL: PASS")
print("  structural wiring only. T2-a (cos >= 0.92) and T2-c (unitary under the")
print("  operator) are NOT measured here and are NOT claimed -- they need the")
print("  RTX 5090 in a single batched session with contract_lock_check.py --live.")
sys.exit(0)
