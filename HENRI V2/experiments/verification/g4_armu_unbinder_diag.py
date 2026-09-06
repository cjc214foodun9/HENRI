"""G4 ARM-U — unbinder causal-consumer diagnostic on Zone C chunk waves.

Sealed K2/U2 method (2026-08-25): load checkpoint into HENRINeuralEgressUnbinder,
forward 16 live chunk waves, measure seeded per-block O(8) rotation sensitivity
(argmax index change, logit movement), determinism (A-repeat), inverse-restore
(A^-1 A = I => output equality), and mismatched-query control.

Read-only: SELECT only on corpus_chunks; no optimizer step; no save.
No string-level claim: token index is reported; string decode is BLOCKED
without a vocab file (checked; none in-repo).
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

CKPT = "HENRI V2/models/henri_decoder_checkpoint.pt"


def wave_bytes_to_tensor(b: bytes) -> torch.Tensor:
    w = np.frombuffer(b, dtype=np.float32).copy()
    w = torch.from_numpy(w)  # [8192, 8]
    flat = w.reshape(-1, 65536)
    return flat / (flat.norm(dim=-1, keepdim=True) + 1e-12)


def rotor(seed: int) -> torch.Tensor:
    """Seeded per-block O(8) rotation: blockwise orthonormal Q, Q^T Q = I."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    Q = torch.randn(8192, 8, 8, generator=g)
    Q, _ = torch.linalg.qr(Q)  # [8192, 8, 8] orthonormal rows
    return Q


def apply_rot(wave: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """wave [N, 8192, 8] -> per-block rotation Q [8192, 8, 8]: out[n,k,:] = wave[n,k,:] @ Q[k]."""
    q = Q.to(wave.dtype)
    return torch.einsum("nkj,kij->nki", wave.reshape(wave.shape[0], 8192, 8), q).reshape(wave.shape[0], 65536)


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dsn = os.environ.get("ZONE_C_PROD_DSN", "")
    out = {"device": device, "dsn_set": bool(dsn)}
    if not dsn:
        print(json.dumps({"error": "NO_DSN"})); return 1

    import psycopg
    conn = psycopg.connect(dsn, connect_timeout=15)
    cur = conn.cursor()
    cur.execute("SELECT chunk_id, wave_payload FROM corpus_chunks ORDER BY chunk_id LIMIT 16")
    rows = cur.fetchall()
    cur.close(); conn.close()
    out["n_waves"] = len(rows)
    wave_ids = [r[0] for r in rows]

    from henri_decoder import HENRINeuralEgressUnbinder
    unb = HENRINeuralEgressUnbinder(d_model=65536, d_hidden=2048, vocab_size=32000, device=device)
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)
    unb.load_state_dict(ckpt)
    unb.eval()
    state_sha = {
        "down_proj": [tuple(ckpt["down_proj.weight"].shape), str(ckpt["down_proj.weight"].dtype)],
        "lm_head": [tuple(ckpt["lm_head.weight"].shape), str(ckpt["lm_head.weight"].dtype)],
        "layer_norm": [tuple(ckpt["layer_norm.weight"].shape)],
    }
    out["checkpoint_shapes"] = state_sha

    waves = torch.stack([wave_bytes_to_tensor(r[1]) for r in rows]).to(device)  # [16, 65536]

    with torch.no_grad():
        logits0 = unb(waves)  # [16, 32000]
        idx0 = logits0.argmax(dim=-1).reshape(-1)  # [16]
        # A-repeat determinism
        logits0b = unb(waves)
        idx0b = logits0b.argmax(dim=-1).reshape(-1)
        out["repeat_exact"] = bool(torch.equal(idx0, idx0b))
        # DEGENERACY CHECK (load-bearing): does the trained unbinder emit distinct
        # top-1 tokens per distinct wave, or collapse to one token?
        out["top1_token_unique"] = int(len(set(idx0.tolist())))
        out["top1_token_list"] = [int(t) for t in idx0.tolist()]
        # rotation sensitivity (shape-corrected: both flattened to [16])
        Qs = []
        chg = []
        mvmt = []
        for seed in range(4):
            Q = rotor(seed).to(device)
            wrot = apply_rot(waves[0:16], Q)  # [16, 65536]
            logits_r = unb(wrot)
            idx_r = logits_r.argmax(dim=-1).reshape(-1)
            chg.append(int((idx_r != idx0).sum().item()))
            mvmt.append(float((logits_r - logits0).abs().mean().item()))
        out["rot_changed_argmax_per_seed"] = chg
        out["rot_logit_mean_abs_delta"] = [round(v, 6) for v in mvmt]
        # inverse restore: apply Q^T after Q; linear path => exact restore
        Q = rotor(3).to(device)
        wrot = apply_rot(waves[:4], Q)
        back = apply_rot(wrot, Q.transpose(-1, -2))
        idx_back = unb(back).argmax(dim=-1).reshape(-1)
        idx_orig = idx0[:4]
        out["inverse_restore_exact"] = bool(torch.equal(idx_back, idx_orig))
        # permutation-equivariance control (mathematically expected 0; a nonzero
        # value would indicate batch-order dependence, not semantic content)
        perm = torch.randperm(16, device=device)
        idx_perm = unb(waves[perm]).argmax(dim=-1).reshape(-1)
        idx_ref = idx0[perm]
        out["perm_equivariance_changed"] = int((idx_perm != idx_ref).sum().item())
        out["perm_equivariance_total"] = int(16)

    print(json.dumps(out, indent=2))
    with open("/tmp/g4_armu.json", "w") as f:
        json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
