"""Probe the REAL Zone A construction surface before designing the facade.

The fabricated reference context claimed classes that do not exist
(`DarwinianPhaseSwarm`, `PhaseCodecAdapter`, `HENRIUnifiedVLA` in
`unified_henri_vla.py`). This probe establishes what actually constructs, at
what parameter cost, and which attributes a facade may legitimately reach.

Read-only. No writes, no training.
"""

from __future__ import annotations

import os
import sys
import traceback

# The code directory is the PARENT of tools/. Python puts the SCRIPT's directory
# on sys.path, not the cwd, so add the code dir explicitly (probe defect fixed
# 2026-09-26: the first run failed with ModuleNotFoundError on every live module).
_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

os.environ.setdefault("HENRI_UNIFIED_VLA", "0")

import torch  # noqa: E402

print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("=" * 78)


def probe(label, fn):
    try:
        out = fn()
        print(f"OK    {label}: {out}")
        return out
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL  {label}: {type(exc).__name__}: {exc}")
        traceback.print_exc(limit=2)
        return None


# ---------------------------------------------------------------- syncytium
def _syncytium():
    from darwinian_phase_swarm import GapJunctionSwarmSyncytium

    s = GapJunctionSwarmSyncytium(num_experts=8, d_model=64, r_rank=4)
    n = sum(p.numel() for p in s.parameters())
    trainable = [(tuple(p.shape), p.requires_grad) for p in s.parameters()]
    return f"params={n} tensors={len(trainable)} shapes={trainable[:6]}"


probe("GapJunctionSwarmSyncytium(small)", _syncytium)


# ---------------------------------------------------------------- orchestrator
def _orch():
    from darwinian_phase_swarm import HenriSwarmOrchestrator

    o = HenriSwarmOrchestrator(
        num_experts=8, d_model=64, r_rank=4, num_blocks=8, action_enum_class=None
    )
    attrs = [a for a in ("syncytium", "clifford", "decoder", "planner") if hasattr(o, a)]
    acts = len(getattr(o.decoder, "id_to_action", {}))
    n = sum(p.numel() for p in o.parameters())
    return f"attrs={attrs} actions={acts} params={n}"


probe("HenriSwarmOrchestrator(small)", _orch)


# ---------------------------------------------------------------- decoder only
def _decoder():
    from darwinian_phase_swarm import HolographicActionDecoder

    d = HolographicActionDecoder(d_model=64, action_enum_class=None)
    return f"actions={len(d.id_to_action)} ids={list(d.id_to_action.items())[:4]}"


probe("HolographicActionDecoder(small)", _decoder)


# ---------------------------------------------------------------- EFE planner
def _planner():
    from efe_planner import EFEPlanner

    p = EFEPlanner(num_blocks=8, d_model=64, num_actions=6)
    return f"ok type={type(p).__name__} has_transition={hasattr(p, 'transition')}"


probe("EFEPlanner(small)", _planner)


# ---------------------------------------------------------------- vision encoder
def _encoder():
    from henri_vision_encoder import HENRIVisionEncoder

    e = HENRIVisionEncoder(d_model=64, k_blocks=8, block_dim=8, device="cpu")
    g = torch.randint(0, 10, (4, 4))
    w = e.encode_spatial_grid(g)
    return f"shape={tuple(w.shape)} norm={float(w.norm()):.4f}"


probe("HENRIVisionEncoder.encode_spatial_grid(small)", _encoder)


# ---------------------------------------------------------------- wave jepa
def _jepa():
    from wave_jepa import WaveJEPA

    j = WaveJEPA(d_model=64, num_blocks=8, r_rank=4, device="cpu")
    g0 = torch.randint(0, 10, (4, 4))
    g1 = torch.randint(0, 10, (4, 4))
    c = j.encode_context(g0)
    t = j.encode_target(g1)
    return f"ctx={tuple(c.shape)} tgt={tuple(t.shape)} rollout={hasattr(j, 'rollout')}"


probe("WaveJEPA(small)", _jepa)


# ---------------------------------------------------------------- codec
def _codec_surface():
    import phase_codec_adapter as pca

    names = [n for n in dir(pca) if not n.startswith("_")]
    return f"public={names[:14]}"


probe("phase_codec_adapter public surface", _codec_surface)


# ---------------------------------------------------------------- egress
def _egress():
    from henri_decoder import HENRIUnifiedEgressTransducer

    return f"class present={HENRIUnifiedEgressTransducer is not None}"


probe("HENRIUnifiedEgressTransducer import", _egress)


# ---------------------------------------------------------------- unified vla
def _unified():
    import henri_unified_vla as m

    return f"classes={[n for n in dir(m) if n[0].isupper()]} flag={m.FLAG}"


probe("henri_unified_vla surface", _unified)

print("=" * 78)
print("PROBE COMPLETE")
