"""A-K5 GATE POWER VALIDATION -- can the egress gate detect content at all?

WHY THIS EXISTS
    A-K5 returned AK5_EGRESS_FAIL. That verdict is only meaningful if the gate
    COULD have returned PASS on an artifact that does carry content. My gate has
    D0 (artifact changed from init) and Q4 (the wave reaches the head), but
    NEITHER proves the gate can DETECT CONTENT. D0/Q4 are both satisfiable by a
    readout that is content-blind.

    Without this check, "FAIL" is ambiguous: it could mean "the checkpoint carries
    no content" or "the gate is vacuous". This probe separates those two.

METHOD
    Construct a SYNTHETIC content-grounded readout with a KNOWN content->logit
    mapping, using a deterministic function of the wave's OWN block structure.
    Then run the SAME Q1/Q2/Q3/Q4 tests and the SAME same-family content-destroyed
    control used by ak5_armb_egress_gate.py.

    No training. No optimizer. No checkpoint. Pure function. $0. Read-only.

    control_destroy_content (identical semantics to the A-K5 gate): permute the
    8192 blocks of each wave. This preserves every block vector and the unit norm
    exactly, and destroys only the ARRANGEMENT.

PRE-REGISTERED (frozen before the run)
    GP1  Q1 separation on the synthetic readout must exceed the shuffled control
         by >= +0.10.       => the gate CAN detect content
    GP2  Q2 margin on the synthetic readout must exceed the shuffled control
         by >= +0.10.
    GP3  If GP1 or GP2 fail, the A-K5 gate is VACUOUS and the A-K5 verdict is
         UNINTERPRETABLE -- it must be withdrawn, not reinterpreted.

DEFECT DISCLOSURE (2026-10-03) -- READ BEFORE TRUSTING ANY OUTPUT OF THIS FILE
    The first run returned AK5_GATE_VACUOUS. That result is an ARTIFACT OF THIS
    PROBE and must NOT be reported. Two design defects, both local to this file:

      D1  NO POSITIVE PAIRS. Q1 is a mean over ALL sample pairs, but all 24
          samples carry DISTINCT content. Every pair is a different-content pair,
          so the mean tends to ~1.0 BY CONSTRUCTION whether or not content
          exists. A metric with no same-content pair cannot detect content.

      D2  SATURATED READOUT. W scaled by 0.02 against VOCAB=4096 yields
          near-orthogonal logits for any input, so both arms sit at the ceiling.

    Observed: Q1 real=1.0002 ctrl=1.0002; Q2 real=0.1526 ctrl=0.1609. Saturation,
    not absence of content.

    CONSEQUENCE: gate power remains UNVALIDATED. This file establishes NOTHING
    about the A-K5 gate. It is retained as a recorded failed probe.

    IMPORTANT: this does NOT touch the A-K5 verdict. The corrected classification
    AK5_BLOCKED_DEGENERATE_TRAINING_TARGET rests on the provenance audit
    (evidence/ak5_provenance_correction.json, findings 1-5) -- random waves,
    random labels, salted-hash ids, and NO content-grounded head on this host.
    Those findings are independent of gate power.

    CORRECT DESIGN FOR REUSE: include POSITIVE pairs (same content, independent
    noise) and NEGATIVE pairs; require separation between the same-content and
    different-content distributions, not a single all-pairs mean; and verify the
    control actually destroys the positive-pair similarity.

VERDICTS
    GATE_POWER_OK            -> GP1 and GP2 both hold. The A-K5 FAIL is meaningful.
    AK5_GATE_VACUOUS         -> GP1 or GP2 fails. UNRELIABLE ON THIS FILE; see
                                DEFECT DISCLOSURE above before acting on it.
"""

from __future__ import annotations

import os
import torch
import torch.nn.functional as F

D_MODEL = int(os.environ.get("GP_D_MODEL", "8192"))   # reduced D: $0, fast
N_BLOCKS = int(os.environ.get("GP_N_BLOCKS", "1024"))
BLOCK_DIM = int(os.environ.get("GP_BLOCK_DIM", "8"))
VOCAB = int(os.environ.get("GP_VOCAB", "4096"))
N_SAMPLES = int(os.environ.get("GP_N", "24"))
SEEDS = tuple(int(s) for s in os.environ.get("GP_SEEDS", "20261002,20261003,20261004").split(","))

Q1_MIN_DELTA = 0.10
Q2_MIN_DELTA = 0.10

assert N_BLOCKS * BLOCK_DIM == D_MODEL, "block geometry mismatch"


def codec_family_waves(n: int, seed: int) -> torch.Tensor:
    """Sparse block-structured waves in the codec's declared family.

    Mirrors zone_c_world_knowledge_codec._wave_accum semantics at reduced D:
    hash features -> spread to (block, slot) positions -> signed accumulate ->
    per-block L2 normalise. Content = WHICH slot is active within each block.
    """
    g = torch.Generator().manual_seed(seed)
    rows = torch.zeros(n, N_BLOCKS, BLOCK_DIM)
    # per-sample: activate one slot per block, chosen by a content code
    slot = torch.randint(0, BLOCK_DIM, (n, N_BLOCKS), generator=g)
    sign = torch.where(torch.randint(0, 2, (n, N_BLOCKS), generator=g) == 0, 1.0, -1.0)
    bidx = torch.arange(N_BLOCKS).unsqueeze(0).expand(n, N_BLOCKS)
    rows[torch.arange(n).unsqueeze(1), bidx, slot] = sign
    rows = rows / rows.norm(dim=2, keepdim=True).clamp_min(1e-9)
    return rows.reshape(n, D_MODEL)


def destroy_content(waves: torch.Tensor, seed: int) -> torch.Tensor:
    """Same-family content-destroyed control: permute the block axis.

    Preserves every block vector and the unit norm exactly. Destroys only the
    ARRANGEMENT. Identical semantics to ak5_armb_egress_gate.destroy_content.
    """
    g = torch.Generator().manual_seed(seed)
    out = waves.clone().reshape(-1, N_BLOCKS, BLOCK_DIM)
    for i in range(out.shape[0]):
        p = torch.randperm(N_BLOCKS, generator=g)
        out[i] = out[i][p]
    return out.reshape(waves.shape[0], D_MODEL)


class SyntheticContentReadout:
    """Deterministic content-grounded readout. NOT trained; a known mapping.

    Content definition: the SLOT PATTERN (which slot is active per block) is the
    label. This is exactly the information the block-shuffle destroys, so a readout
    that keys on it MUST separate real waves from shuffled ones.
    """

    def __init__(self, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        # projection that keys on slot identity per block, then pools by block
        self.W = torch.randn(VOCAB, N_BLOCKS, BLOCK_DIM, generator=g) * 0.02

    @torch.no_grad()
    def __call__(self, waves: torch.Tensor) -> torch.Tensor:
        b = waves.reshape(waves.shape[0], N_BLOCKS, BLOCK_DIM)
        # (n, V) = sum over blocks of <W_v_block, wave_block>
        return torch.einsum("nbd,vbd->nv", b, self.W)


def pairwise_cos_dist(logits: torch.Tensor) -> float:
    """Mean pairwise cosine distance of the logit vectors (Q1 quantity)."""
    z = F.normalize(logits - logits.mean(dim=1, keepdim=True), dim=1)
    n = z.shape[0]
    sim = z @ z.t()
    off = sim[~torch.eye(n, dtype=torch.bool)]
    return float(1.0 - off.mean())


def top1_margin(logits: torch.Tensor) -> float:
    """Mean (top1 - top2) margin (Q2 quantity)."""
    s, _ = torch.sort(logits, dim=1, descending=True)
    return float((s[:, 0] - s[:, 1]).mean())


def main() -> int:
    print("== A-K5 GATE POWER VALIDATION ==")
    print("   D=%d blocks=%d block_dim=%d vocab=%d n=%d"
          % (D_MODEL, N_BLOCKS, BLOCK_DIM, VOCAB, N_SAMPLES))
    print("   This probe contains NO trained weights and NO optimizer step.")

    readout = SyntheticContentReadout(seed=SEEDS[0])

    q1_real, q1_ctrl = [], []
    q2_real, q2_ctrl = [], []
    for s in SEEDS:
        real = codec_family_waves(N_SAMPLES, s)
        ctrl = destroy_content(real, s + 1)
        lr_ = readout(real)
        lc_ = readout(ctrl)
        q1_real.append(pairwise_cos_dist(lr_))
        q1_ctrl.append(pairwise_cos_dist(lc_))
        q2_real.append(top1_margin(lr_))
        q2_ctrl.append(top1_margin(lc_))
        print("   seed %d: Q1 real=%.4f ctrl=%.4f | Q2 real=%.4f ctrl=%.4f"
              % (s, q1_real[-1], q1_ctrl[-1], q2_real[-1], q2_ctrl[-1]))

    m1r = sum(q1_real) / len(q1_real); m1c = sum(q1_ctrl) / len(q1_ctrl)
    m2r = sum(q2_real) / len(q2_real); m2c = sum(q2_ctrl) / len(q2_ctrl)
    d1, d2 = m1r - m1c, m2r - m2c

    gp1 = d1 >= Q1_MIN_DELTA
    gp2 = d2 >= Q2_MIN_DELTA

    print()
    print("   Q1 separation: real=%.4f control=%.4f delta=%+.4f (floor +%.2f) -> %s"
          % (m1r, m1c, d1, Q1_MIN_DELTA, "PASS" if gp1 else "FAIL"))
    print("   Q2 margin    : real=%.4f control=%.4f delta=%+.4f (floor +%.2f) -> %s"
          % (m2r, m2c, d2, Q2_MIN_DELTA, "PASS" if gp2 else "FAIL"))

    if gp1 and gp2:
        verdict = "GATE_POWER_OK"
        note = ("The A-K5 gate CAN separate a content-grounded readout from the "
                "same-family content-destroyed control. The A-K5 FAIL is therefore "
                "a statement about the checkpoint, not a vacuous instrument.")
    else:
        verdict = "AK5_GATE_VACUOUS"
        note = ("The gate cannot detect content even on a KNOWN content-grounded "
                "mapping. The A-K5 verdict must be WITHDRAWN as uninterpretable.")

    print()
    print("   VERDICT = %s" % verdict)
    print("   %s" % note)
    print("   evidence_class = DERIVED (synthetic readout; no trained artifact)")
    return 0 if gp1 and gp2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
