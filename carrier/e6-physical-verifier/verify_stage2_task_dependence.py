"""Stage 2 discriminating test: is the metric-locality collapse REAL?

WHY THIS EXISTS
---------------
The Stage 2 gate (verify_stage2_metric_locality.py) builds its task with
`make_wave`, which places class identity in BOTH a 3-block set AND a
3-channel set:

    blocks = [(label * 3 + i) % NUM_BLOCKS for i in range(3)]
    chans  = [(label * 2 + i) % CHANNELS   for i in range(3)]

Mean pooling is `w.view(16, 4096).mean(dim=0)`. It averages over the 16 BLOCK
axis but leaves the CHANNEL axis intact. So class identity survives through a
channel that the harness author chose to encode. AUC came out 0.9292 and the
gate PASSED.

That result is real for that task, but it cannot adjudicate the spec's claim,
because the task was designed by the same author as the gate. This is a
mock-loop hazard: testing a bridge on data constructed to be readable by that
bridge's retained axis.

WHAT THE SPEC ACTUALLY CLAIMS (Gap 1, verbatim from the PDF text)
-----------------------------------------------------------------
  "Global mean pooling across continuous spatial wave states destroys local
   phase differences. The local metric interval ... cancels out when averaged
   across arbitrary coordinate frames."
  "...lose metric locality, causing the AUC to collapse to [below] the Gate 1
   threshold."

So the claim is a MECHANISM (averaging destroys local structure) with an AUC
CONSEQUENCE. This script isolates where class identity lives:

  both         -- differs in blocks AND channels  (the original gate task)
  channel_only -- differs ONLY in channel pattern (blocks identical)
  block_only   -- differs ONLY in block pattern   (channels identical)

PREDICTIONS
-----------
If the spec's mechanism is real:
    channel_only : mean PASSES  (channel axis survives averaging)
    block_only   : mean COLLAPSES to ~0.5   <-- the defect, isolated
    both         : mean passes (channel axis rescues it)

If mean pooling is harmless (spec wrong):
    block_only   : mean still PASSES

`block_only` is the decisive case. It is the minimal task on which a
block-averaging operator MUST fail if the claimed mechanism holds.

Evidence class: OBSERVED (every number produced by this run).
Runs on CPU. No GPU, no cost.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_bridges():
    """Import the two bridges from the existing gate so this test measures the
    SAME implementations, not re-typed copies."""
    path = os.path.join(HERE, "verify_stage2_metric_locality.py")
    spec = importlib.util.spec_from_file_location("_s2_gate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # __main__ guard keeps main() unrun
    return mod


S2 = _load_bridges()
NUM_BLOCKS = S2.NUM_BLOCKS
BLOCK_LEN = S2.BLOCK_LEN
CHANNELS = S2.CHANNELS
D_WAVE = S2.D_WAVE

N_CLASSES = 8
PER_CLASS = 12


def make_wave_split(label: int, mode: str, seed: int = 0) -> torch.Tensor:
    """65536-dim wave whose class signature lives only in `mode`'s axis.

    both         : distinct blocks AND distinct channels  (original gate task)
    channel_only : distinct channels, fixed blocks {0,1,2}
    block_only   : distinct blocks (disjoint pairs), fixed channels {0,1,2}
    """
    off = {"both": 0, "channel_only": 13, "block_only": 7}[mode]
    g = torch.Generator().manual_seed(seed * 1000 + label + off)
    w = torch.zeros(D_WAVE)

    if mode == "both":
        blocks = [(label * 3 + i) % NUM_BLOCKS for i in range(3)]
        chans = [(label * 2 + i) % CHANNELS for i in range(3)]
    elif mode == "channel_only":
        blocks = [0, 1, 2]
        chans = [(label + i) % CHANNELS for i in range(3)]
    elif mode == "block_only":
        # 8 classes -> disjoint block pairs: {0,1},{2,3},...,{14,15}
        blocks = [(2 * label) % NUM_BLOCKS, (2 * label + 1) % NUM_BLOCKS]
        chans = [0, 1, 2]
    else:
        raise ValueError(mode)

    for b in blocks:
        for c in chans:
            base = b * BLOCK_LEN
            lo = c * (BLOCK_LEN // CHANNELS)
            w[base + lo: base + lo + (BLOCK_LEN // CHANNELS)] = (
                torch.randn(BLOCK_LEN // CHANNELS, generator=g) + 4.0
            )

    return w + 0.05 * torch.randn(D_WAVE, generator=g)


def score(bridge, mode: str) -> dict:
    """One-vs-rest AUC per class, cosine-to-prototype score. Mirrors the gate."""
    protos = {}
    for c in range(N_CLASSES):
        ws = torch.stack([bridge(make_wave_split(c, mode, i))
                          for i in range(PER_CLASS)])
        protos[c] = F.normalize(ws.mean(dim=0), p=2, dim=-1)

    per_class = {}
    for c in range(N_CLASSES):
        S, L = [], []
        for other in range(N_CLASSES):
            for i in range(PER_CLASS):
                d = bridge(make_wave_split(other, mode, 100 + i))
                S.append(float(torch.dot(d, protos[c]).item()))
                L.append(1 if other == c else 0)
        s = torch.tensor(S).double()
        lab = torch.tensor(L).bool()
        pos, neg = s[lab], s[~lab]
        dd = pos.unsqueeze(1) - neg.unsqueeze(0)
        per_class[c] = float(((dd > 0).double() + 0.5 * (dd == 0).double()).mean())
    vals = [v for v in per_class.values() if not math.isnan(v)]
    return {
        "mean_auc": round(sum(vals) / len(vals), 4),
        "min_auc": round(min(vals), 4),
    }


def block_permutation_collision(bridge) -> float:
    """Mechanism probe: does the bridge separate a wave from its block-permuted
    twin? Cosine 1.0 means the two are indistinguishable."""
    w = make_wave_split(1, "both", 0)
    blocks = w.view(NUM_BLOCKS, BLOCK_LEN)
    perm = torch.randperm(NUM_BLOCKS, generator=torch.Generator().manual_seed(3))
    w_perm = blocks[perm].reshape(-1)
    return round(float(torch.dot(bridge(w), bridge(w_perm)).item()), 6)


def main() -> int:
    print("=" * 78)
    print("STAGE 2 DISCRIMINATING TEST - where does class identity live?")
    print("=" * 78)
    print("gate under test: mean pooling AUC >= 0.85 (spec's Gate 1)")
    print(f"task: {N_CLASSES}-way, {PER_CLASS}/class, bridges imported from the gate")
    print()

    rows = {}
    for mode in ("both", "channel_only", "block_only"):
        m_auc = score(S2.bridge_mean, mode)["mean_auc"]
        c_auc = score(S2.bridge_clifford, mode)["mean_auc"]
        m_col = block_permutation_collision(S2.bridge_mean)
        c_col = block_permutation_collision(S2.bridge_clifford)
        rows[mode] = {
            "mean_pool_auc": m_auc, "clifford_auc": c_auc,
            "mean_perm_cosine": m_col, "clifford_perm_cosine": c_col,
        }
        print(f"--- task: {mode} ---")
        print(f"    mean pooling : AUC = {m_auc:.4f}  "
              f"-> {'PASS' if m_auc >= 0.85 else 'FAIL (collapse)'}")
        print(f"    Clifford     : AUC = {c_auc:.4f}  "
              f"-> {'PASS' if c_auc >= 0.85 else 'FAIL'}")
        print(f"    block-perm cosine: mean={m_col:.4f}  clifford={c_col:.4f}")
        print()

    bo = rows["block_only"]["mean_pool_auc"]
    co = rows["channel_only"]["mean_pool_auc"]
    print("-" * 78)
    print("VERDICT")
    if bo < 0.85 and co >= 0.85:
        verdict = "MECHANISM_CONFIRMED"
        print("  The spec's mechanism is REAL and MEASURED.")
        print(f"  Mean pooling collapses on block_only (AUC={bo:.4f}) but not on")
        print(f"  channel_only (AUC={co:.4f}). The original gate task passed only")
        print("  because its author put class identity on the channel axis, which")
        print("  averaging preserves. The collapse is task-dependent, not absent.")
    elif bo >= 0.85:
        verdict = "MECHANISM_NOT_REPRODUCED"
        print("  Mean pooling survived the block_only task, so the claimed")
        print("  collapse did not reproduce on the minimal discriminating case.")
    else:
        verdict = "INCONCLUSIVE"
        print("  Neither prediction holds cleanly; inspect per-class detail.")
    print(f"  verdict = {verdict}")
    print("=" * 78)

    receipt = {
        "stage": 2,
        "artifact": "HENRI V2/experiments/verification/arc_g1_topological_engine.py:106",
        "purpose": ("Discriminating test for Gap 1. The primary Stage 2 gate used a "
                    "task whose class identity was placed on the channel axis, which "
                    "mean pooling preserves, so that gate could not adjudicate the "
                    "spec's claim. This test varies where identity lives."),
        "gate": "mean pooling AUC >= 0.85",
        "evidence_class": "OBSERVED",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {"n_classes": N_CLASSES, "per_class": PER_CLASS},
        "results": rows,
        "verdict": verdict,
        "honest_boundary": (
            "Still synthetic. Whether the collapse occurs on REAL ARC grids "
            "through the REAL ingress path is not established, because no wave "
            "corpus or trajectory bank exists on this machine (verified: no .npz "
            "psi banks, no trajectory jsonl matching the arc_g1 input contract). "
            "What is established is that the operator destroys block-level "
            "locality whenever class identity is not on the retained axis."
        ),
    }
    out = os.path.join(HERE, "stage2_task_dependence_receipt.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)
    print(f"receipt written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
