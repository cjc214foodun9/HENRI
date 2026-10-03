"""A-K5 GATE POWER VALIDATION v2 -- corrected after v1's own defect.

WHY THIS FILE EXISTS
    v1 (ak5_gate_power_probe.py) returned AK5_GATE_VACUOUS. That was an artifact
    of v1, not a finding: it had NO positive pairs, so its all-pairs cosine
    DISTANCE saturated near 1.0 in high dimension whether or not content existed.
    See design/zone_a/evidence/ak5_gate_power_probe_defect.json.

    Gate power is a SEPARATE question from the A-K5 verdict. The A-K5 verdict is
    settled by provenance (AK5_BLOCKED_DEGENERATE_TRAINING_TARGET). This probe
    answers a forward-looking question: CAN the A-K5 gate detect content at all?
    If it cannot, the next (approval-gated) experiment would produce another
    uninformative negative.

WHAT CHANGED FROM v1
    V2-a  POSITIVE PAIRS: same content, independent noise.
    V2-b  NEGATIVE PAIRS: different content.
    V2-c  Metric is cos SIMILARITY separation (within minus between), NOT an
          all-pairs cosine distance mean. High-dim random vectors are
          near-orthogonal, which is why v1's metric saturated.
    V2-d  ARRANGEMENT-SENSITIVE readout: dense [H, D] with DISTINCT per-position
          weights, matching the unbinder's down_proj shape. A block-pooled sum is
          permutation-invariant by construction and can never see the control.
    V2-e  SELF-VALIDATION: the probe asserts within-content similarity > 0.5 on
          real waves. If that fails, it reports PROBE_INVALID instead of a gate
          verdict. A probe that cannot pass on a KNOWN-positive input cannot
          certify or condemn the gate it tests.

PRE-REGISTERED (frozen before the run)
    GP1  sep_real = within - between >= 0.20 on real waves
    GP2  sep_ctrl <= 0.05  (block-shuffle destroys the separation)
    GP3  within_real <= 0.999 (dynamic range: not pinned at the ceiling)

VERDICTS
    PROBE_INVALID   -> self-validation failed; this probe proves nothing
    GATE_POWER_OK   -> the gate CAN detect arrangement-borne content
    GATE_POWER_FAIL -> it cannot; the A-K5 gate is unusable as a content test

SCOPE: no trained weights, no optimizer, no checkpoint, no store. Pure function.
Local CPU. 0 USD. Read-only.
"""

from __future__ import annotations

import os
import torch
import torch.nn.functional as F

D = int(os.environ.get("GP2_D", "8192"))
N_BLOCKS = int(os.environ.get("GP2_N_BLOCKS", "1024"))
BLOCK_DIM = int(os.environ.get("GP2_BLOCK_DIM", "8"))
H = int(os.environ.get("GP2_H", "256"))
V = int(os.environ.get("GP2_V", "512"))
N_CONTENT = int(os.environ.get("GP2_N_CONTENT", "12"))
N_PER = int(os.environ.get("GP2_N_PER", "4"))
NOISE = float(os.environ.get("GP2_NOISE", "0.05"))
SEED = int(os.environ.get("GP2_SEED", "20261003"))

GP1_MIN_SEP = 0.20
GP2_MAX_CTRL_SEP = 0.05
GP3_MAX_WITHIN = 0.999
SELF_MIN_WITHIN = 0.50

assert N_BLOCKS * BLOCK_DIM == D, "block geometry mismatch"


def content_wave(c: int, k: int) -> torch.Tensor:
    """Sparse block-structured wave for content id c, replicate k.

    Content is the per-block SLOT PATTERN -- an ARRANGEMENT. Same c gives the
    same pattern; the control (block permutation) destroys the arrangement.
    """
    gs = torch.Generator().manual_seed(SEED * 100000 + c)
    slot = torch.randint(0, BLOCK_DIM, (N_BLOCKS,), generator=gs)
    w = torch.zeros(N_BLOCKS, BLOCK_DIM)
    w[torch.arange(N_BLOCKS), slot] = 1.0
    w = w / w.norm(dim=1, keepdim=True).clamp_min(1e-9)
    flat = w.reshape(D)
    if k >= 0:
        gn = torch.Generator().manual_seed(SEED * 100000 + c * 100 + k + 7)
        flat = F.normalize(flat + torch.randn(D, generator=gn) * NOISE, dim=0)
    return flat


def destroy(w: torch.Tensor, seed: int) -> torch.Tensor:
    """Same-family content-destroyed control: per-sample block permutation.

    Preserves every block vector and the unit norm exactly; destroys arrangement.
    Identical semantics to ak5_armb_egress_gate.destroy_content.
    """
    g = torch.Generator().manual_seed(seed)
    out = w.clone().reshape(-1, N_BLOCKS, BLOCK_DIM)
    for i in range(out.shape[0]):
        p = torch.randperm(N_BLOCKS, generator=g)
        out[i] = out[i][p]
    return out.reshape(w.shape[0], D)


class DenseArrangementReadout:
    """Arrangement-sensitive readout, unbinder-shaped.

    W1 [H, D] has DISTINCT per-position weights, so the block permutation is not
    a symmetry of this map. A block-pooled sum would be invariant (v1's defect).
    """

    def __init__(self, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.W1 = torch.randn(H, D, generator=g) / (D ** 0.5)
        self.W2 = torch.randn(V, H, generator=g) / (H ** 0.5)

    @torch.no_grad()
    def __call__(self, w: torch.Tensor) -> torch.Tensor:
        return torch.relu(w @ self.W1.t()) @ self.W2.t()


def separation(waves: torch.Tensor, labels: torch.Tensor, readout) -> dict:
    """within-content minus between-content mean cosine similarity of logits."""
    z = F.normalize(readout(waves), dim=1)
    S = z @ z.t()
    n = S.shape[0]
    off = ~torch.eye(n, dtype=torch.bool)
    same = labels[:, None] == labels[None, :]
    within = float(S[same & off].mean()) if (same & off).any() else float("nan")
    between = float(S[~same].mean()) if (~same).any() else float("nan")
    return {"within": within, "between": between, "sep": within - between}


def main() -> int:
    print("== A-K5 GATE POWER VALIDATION v2 ==")
    print("   D=%d blocks=%d block_dim=%d H=%d V=%d contents=%d per=%d noise=%.3f"
          % (D, N_BLOCKS, BLOCK_DIM, H, V, N_CONTENT, N_PER, NOISE))
    print("   no trained weights; no optimizer; pure function")

    readout = DenseArrangementReadout(seed=SEED)

    ws, ls = [], []
    for c in range(N_CONTENT):
        for k in range(N_PER):
            ws.append(content_wave(c, k))
            ls.append(c)
    real = torch.stack(ws)
    labels = torch.tensor(ls)
    ctrl = destroy(real, SEED + 1)

    r = separation(real, labels, readout)
    c = separation(ctrl, labels, readout)

    print()
    print("   REAL : within=%.4f between=%.4f sep=%+.4f" % (r["within"], r["between"], r["sep"]))
    print("   CTRL : within=%.4f between=%.4f sep=%+.4f" % (c["within"], c["between"], c["sep"]))
    print()

    # V2-e self-validation BEFORE any gate verdict.
    if not (r["within"] > SELF_MIN_WITHIN):
        print("   SELF-CHECK FAILED: within-content sim %.4f <= %.2f on KNOWN-positive input."
              % (r["within"], SELF_MIN_WITHIN))
        print("   VERDICT = PROBE_INVALID")
        print("   This probe cannot show content even when it exists. Proves nothing.")
        return 3

    gp1 = r["sep"] >= GP1_MIN_SEP
    gp2 = c["sep"] <= GP2_MAX_CTRL_SEP
    gp3 = r["within"] <= GP3_MAX_WITHIN

    print("   GP1 sep_real >= %.2f          : %+.4f -> %s" % (GP1_MIN_SEP, r["sep"], "PASS" if gp1 else "FAIL"))
    print("   GP2 sep_ctrl <= %.2f          : %+.4f -> %s" % (GP2_MAX_CTRL_SEP, c["sep"], "PASS" if gp2 else "FAIL"))
    print("   GP3 within_real <= %.3f      : %.4f -> %s" % (GP3_MAX_WITHIN, r["within"], "PASS" if gp3 else "FAIL"))
    print("   self-check within > %.2f     : %.4f -> PASS" % (SELF_MIN_WITHIN, r["within"]))
    print()

    if gp1 and gp2 and gp3:
        verdict = "GATE_POWER_OK"
        note = ("A gate of this form CAN separate arrangement-borne content from the "
                "same-family content-destroyed control. Gate power is VALIDATED.")
    else:
        verdict = "GATE_POWER_FAIL"
        note = ("This gate form cannot detect content on a KNOWN content-grounded "
                "mapping, so it must not be used as a content verdict for any "
                "checkpoint until redesigned.")
    print("   VERDICT = %s" % verdict)
    print("   %s" % note)
    print("   evidence_class = DERIVED (synthetic readout; no trained artifact)")
    return 0 if verdict == "GATE_POWER_OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
