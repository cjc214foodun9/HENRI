"""Torus encoder adjoint / wave->grid decode (Phase 10.4 directive 4).

SOURCE OF RECORD
    "Project HENRI: Universal Weight Subspace Integration, Zone A Swarm Geometry, and
    Zone C Causal Substrate Orchestration". Directive 4: "CONSTRUCT THE TORUS ENCODER
    ADJOINT (E+): Derive and implement the Moore-Penrose pseudoinverse in
    o_vsa_torus_encoder.py to permit machine-precision wave-to-grid decoding."
    Spec 4.1 claims the adjoint IS the 2D IDFT over an S x S frequency lattice, and
    that the value channel is an "8-blade Clifford multivector" decoded by
    argmax_c <Psi_val, e_c>.

WHAT THIS MODULE DOES
    It writes the LIVE encoder as an explicit linear operator and MEASURES what is and
    is not invertible. Three facts read from `o_vsa_torus_encoder.py` drive the design:

    1. The live wave is real `[num_blocks, 8]` = [8192, 8], NOT `[S^2, 8]`.
       `S = modulus = 32` is the POSITION MODULUS, not an array length. There is no
       `S x S` frequency lattice in the state.
    2. Each block owns 4 random frequencies (kx, ky) drawn from [1, 31]; slot 0 is
       forced to (0, 0) and DOWN-WEIGHTED by dc_weight = 1/(16*S^2). So the state is
       8192 blocks x 4 frequency samples; it is NOT a full DFT sweep.
    3. `_to_real` L2-normalises EVERY BLOCK. That is a per-block nonlinear rescaling
       which destroys each block's complex amplitude.

THE LIVE FORWARD MAP (mode TORUS_VAL), exactly:

        acc[b, s] = sum_{y,x} phasor( (v(x,y)+1) * w_v[b,s] + x*wx[b,s] + y*wy[b,s] )

    with v = colour id, w_v the per-(block, slot) value frequency. Factoring the phasor:

        acc[b, s] = sum_c phasor(phi[b*SL+s, c]) * ( P[b*SL+s, :] @ onehot_c )

        P[b*SL+s, y*W+x] = dc_w[b,s] * exp(i*(x*wx[b,s] + y*wy[b,s]))
        phi[b*SL+s, c]   = (c+1) * w_v[b,s]                  (a PHASE, not a phasor)

    So the operator is A[(b,s), (c, p)] = P[(b,s), p] * phasor(phi[(b,s), c]), a
    [NB*SL, V*H*W] complex matrix. `forward_flat` computes `A @ n` in FACTORED form
    (no dense A), which is only valid because phi is applied PER COLOUR and summed
    after the position contraction, in that order.

EVIDENCE CLASS: the functions RETURN measurements. `verify()` must pass before any
downstream number is trusted; the evaluator asserts that.
"""
from __future__ import annotations

import torch

BLOCK_SLOTS = 4  # mirrors o_vsa_torus_encoder.BLOCK_SLOTS


def _phasor(angle: torch.Tensor) -> torch.Tensor:
    return torch.complex(torch.cos(angle), torch.sin(angle))


def block_phases(enc, H: int, W: int, V: int):
    """Return (P, ph) for a grid family with V colours, position order p = y*W + x.

        P  [NB*SL, H*W] complex : position phasors, INCLUDING the dc-slot down-weight
        ph [NB*SL, V]   complex : colour phasors phasor(phi[:, :V])
    """
    NB = int(enc.num_blocks)
    SL = int(enc.wx.shape[1])
    dev = enc.device
    xs = torch.arange(W, dtype=torch.float32, device=dev)
    ys = torch.arange(H, dtype=torch.float32, device=dev)
    X, Y = torch.meshgrid(xs, ys, indexing="xy")             # [H, W]; X[i,j]=j, Y[i,j]=i
    pos = (X.reshape(-1)[None, :] * enc.wx.reshape(-1, 1)
           + Y.reshape(-1)[None, :] * enc.wy.reshape(-1, 1))   # [NB*SL, H*W]
    P = _phasor(pos)
    dcf = torch.ones(NB, SL, device=dev)
    if int(getattr(enc, "dc_slots", 0)):
        dcf[:, :int(enc.dc_slots)] = float(enc.dc_weight)
    P = P * dcf.reshape(-1, 1)
    phi = enc.value_phase.permute(1, 2, 0).contiguous().reshape(NB * SL, -1)
    ph = _phasor(phi[:, :V])                                  # [NB*SL, V]
    return P, ph


def indicator_from_grid(grid, V: int, device="cpu") -> torch.Tensor:
    """One-hot colour indicator flattened as c*(H*W) + (y*W + x)."""
    H, W = len(grid), len(grid[0])
    n = torch.zeros(V, H, W, device=device)
    for y in range(H):
        row = grid[y]
        for x in range(W):
            n[min(int(row[x]), V - 1), y, x] = 1.0
    return n.reshape(-1)


def forward_flat(enc, n_flat: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """PRE-normalisation accumulator, flat complex [NB*SL], computed in factored form."""
    V = n_flat.numel() // (H * W)
    P, ph = block_phases(enc, H, W, V)                        # [NB*SL, HW], [NB*SL, V]
    nm = n_flat.reshape(V, H * W).to(torch.complex64)         # [V, HW]
    S = nm @ P.t()                                            # [V, NB*SL]
    return (S * ph.t()).sum(dim=0)                            # [NB*SL]


def forward_wave(enc, n_flat, H: int, W: int) -> torch.Tensor:
    """Stored wave: real [NB, 8], i.e. `_to_real` of the accumulator."""
    acc = forward_flat(enc, n_flat, H, W).reshape(enc.num_blocks, BLOCK_SLOTS)
    return enc._to_real(acc)


def dense_operator(enc, H: int, W: int, V: int) -> torch.Tensor:
    """Dense PRE-normalisation operator A in C^{NB*SL x V*H*W} (small grids only)."""
    P, ph = block_phases(enc, H, W, V)
    A = P.unsqueeze(2) * ph.unsqueeze(1)                      # [NB*SL, HW, V]
    return A.permute(0, 2, 1).reshape(P.shape[0], V * H * W)


# ----------------------------------------------------------------- diagnostics
def frequency_census(enc) -> dict:
    """Count the DISTINCT (kx, ky) pairs actually sampled.

    Decides directive 4.1: an IDFT adjoint presupposes a complete S x S frequency
    lattice. Count what exists instead of assuming it.
    """
    kx = enc.kx.reshape(-1).to(torch.int64)
    ky = enc.ky.reshape(-1).to(torch.int64)
    pairs = set(zip(kx.tolist(), ky.tolist()))
    S = int(enc.modulus)
    nz = [(a, b) for (a, b) in pairs if (a, b) != (0, 0)]
    ks = sorted({a for a, b in nz} | {b for a, b in nz})
    ndc = int(((kx == 0) & (ky == 0)).sum())
    return {
        "modulus_S": S, "num_blocks": int(enc.num_blocks),
        "slots_per_block": int(enc.wx.shape[1]),
        "total_frequency_samples": int(kx.numel()),
        "distinct_pairs_total": len(pairs), "distinct_pairs_nonzero": len(nz),
        "dc_samples": ndc,
        "kx_value_range": [int(ks[0]), int(ks[-1])] if ks else [],
        "nonzero_samples": int(kx.numel()) - ndc,
        "full_lattice_S_x_S": S * S, "full_lattice_1_to_Sminus1": (S - 1) * (S - 1),
        "dc_weight": float(enc.dc_weight), "dc_slots": int(getattr(enc, "dc_slots", 0)),
        "sampling_density_vs_961": (int(kx.numel()) - ndc) / ((S - 1) * (S - 1)),
    }


def rank_test(enc, H: int, W: int, V: int, tol: float = 1e-6) -> dict:
    """Rank of the PRE-normalisation operator on an (H, W, V) grid family."""
    A = dense_operator(enc, H, W, V)
    M, K = A.shape
    s = torch.sort(torch.linalg.svdvals(A), descending=True).values
    nz = int((s > tol * s[0]).sum())
    return {"operator_shape": [M, K], "equations": M, "unknowns": K,
            "rank": nz, "rank_deficient_by": K - nz,
            "full_column_rank": bool(nz == K),
            "largest_sv": float(s[0]), "smallest_sv": float(s[-1]),
            "cond": float(s[0] / s[-1]) if float(s[-1]) > 0 else float("inf")}


def mp_roundtrip(enc, grid, V: int, tol: float = 1e-9) -> dict:
    """LEFT-INVERSE TEST (a): A+ (A n) == n on the PRE-normalisation accumulator.

    Exact here means the operator carries the information; a later failure is then
    attributable to `_to_real`, not to the operator.
    """
    H, W = len(grid), len(grid[0])
    A = dense_operator(enc, H, W, V)
    n = indicator_from_grid(grid, V)
    acc = A @ n.to(torch.complex64)
    n_rec = torch.linalg.lstsq(A, acc).solution
    err = float((n_rec - n.to(torch.complex64)).abs().max())
    grid_rec = n_rec.real.reshape(V, H, W).argmax(0)
    gt = torch.as_tensor(grid, dtype=torch.long)
    return {"exact": bool(err < tol), "max_abs_err": err,
            "grid_exact": bool(torch.equal(grid_rec, gt)),
            "cell_accuracy": float((grid_rec == gt).float().mean())}


def decode_wave_to_grid(enc, wave_real, H: int, W: int, V: int, steps: int = 300,
                        lr: float = 0.1, seed: int = 0) -> dict:
    """LEFT-INVERSE TEST (b): recover the grid from the STORED (normalised) wave.

    `_to_real` fixes only the DIRECTION of each block's complex 4-vector; the magnitude
    is lost. So the unknown per-block scale is PROFILED OUT by maximising a per-block
    cosine -- the exact structure of the normalisation, not an IDFT approximation.
    """
    NB = int(enc.num_blocks)
    torch.manual_seed(seed)
    wc = enc._to_complex(wave_real).reshape(NB, BLOCK_SLOTS)
    wn = wc.abs().pow(2).sum(-1).sqrt() + 1e-12
    P, ph = block_phases(enc, H, W, V)
    logits = torch.zeros(V, H, W, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=lr)
    Pt = P.t()                                                # [HW, NB*SL]
    cosb = torch.zeros(NB)
    for _ in range(steps):
        opt.zero_grad()
        n = torch.softmax(logits.reshape(V, -1), dim=0)       # [V, HW]
        S = n.to(torch.complex64) @ Pt                        # [V, NB*SL]
        acc = (S * ph.t()).sum(dim=0).reshape(NB, BLOCK_SLOTS)
        num = (acc.conj() * wc).sum(-1).real
        den = acc.abs().pow(2).sum(-1).sqrt() * wn + 1e-12
        cosb = num / den
        (1.0 - cosb).mean().backward()
        opt.step()
    with torch.no_grad():
        rec = torch.softmax(logits.reshape(V, -1), dim=0).reshape(V, H, W).argmax(0)
    gt = None
    return {"grid": rec, "mean_block_cos": float(cosb.mean()),
            "min_block_cos": float(cosb.min()), "steps": steps}
