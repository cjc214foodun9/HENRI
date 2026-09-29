"""Prove the chunked binding term is EXACTLY the naive one (not an approximation).

WHY THIS TEST EXISTS
====================
`VLMPhaseBridge.binding_term` replaced a form that allocated [B,N,D] twice with a
chunked accumulation. That is a load-bearing numeric claim: if chunking changed
the answer, every psi produced afterwards would be silently different and the
"binding_ratio" diagnostic would be measuring something else. So the equivalence
is asserted against the ORIGINAL expression, at several chunk sizes, on CPU.

The expressions differ only in the ORDER of a floating-point summation, so the
expected agreement is float32 rounding (~1e-6 relative), NOT bitwise identity.
This test states that tolerance explicitly instead of pretending to exactness.
"""

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.abspath(os.path.join(HERE, "..", ".."))
if V2 not in sys.path:
    sys.path.insert(0, V2)

from henri_vla_vlm_bridge import VLMPhaseBridge  # noqa: E402


def naive(bridge, feats):
    """The ORIGINAL expression, kept here only as the reference oracle."""
    pf = bridge.to_phase(bridge.to_pool(bridge.v_proj(feats.to(torch.float32))))
    bound = pf * bridge.pos_codes[None, :, :]
    return bound.sum(dim=1)


def main() -> int:
    torch.manual_seed(0)
    B, N, DVIT, D = 4, 32, 64, 1024
    feats = torch.randn(B, N, DVIT)

    b = VLMPhaseBridge(d_vit=DVIT, n_patches=N, d_model=D, d_pool=32,
                       use_position_binding=True)
    b.eval()

    with torch.no_grad():
        ref = naive(b, feats)
        out = {"schema": "henri.binding_chunk_equiv/1",
               "shape": list(ref.shape), "checks": {}}
        ok = True
        for c in (1, 2, 3, 8, 32, 1024):
            got = b.binding_term(feats, chunk=c)
            d = float((got - ref).abs().max())
            scale = float(ref.abs().max()) + 1e-12
            rel = d / scale
            passed = bool(torch.allclose(got, ref, rtol=1e-5, atol=1e-6))
            out["checks"][f"chunk={c}"] = {
                "max_abs_diff": d, "rel_diff": round(rel, 12),
                "allclose_rtol_1e-5": passed}
            ok = ok and passed

        # and the full forward must agree between chunk sizes on psi
        with_chunk = VLMPhaseBridge(d_vit=DVIT, n_patches=N, d_model=D, d_pool=32,
                                    use_position_binding=True, binding_chunk=1)
        with_chunk.load_state_dict(b.state_dict())
        with_chunk.eval()
        psi_a, psi_b = b(feats), with_chunk(feats)
        out["psi_max_abs_diff"] = float((psi_a - psi_b).abs().max())
        out["psi_allclose"] = bool(torch.allclose(psi_a, psi_b, rtol=1e-5, atol=1e-6))
        out["psi_norm_mean"] = round(float(psi_a.norm(dim=-1).mean()), 6)
        ok = ok and out["psi_allclose"]

        # binding_ratio is reported from the chunked path and must be sane
        out["binding_ratio"] = round(float(b.binding_ratio), 8)
        out["binding_active"] = bool(b.binding_active)
        ok = ok and out["binding_active"] and out["binding_ratio"] > 0

    out["EQUIVALENCE_HOLDS"] = bool(ok)
    import json
    print(json.dumps(out, indent=2))
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
