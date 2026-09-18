"""ACTION: measure top1_margin on LIVE ENCODER WAVES, not random waves.

WHY
    The sealed action egress ships with HENRI_SEALED_EGRESS_MIN_MARGIN defaulting to
    0.0, and the code REFUSES to invent a floor because the margin distribution on
    live waves is unmeasured. The only prior numbers came from normalized RANDOM
    waves (mean 0.05015, p05 0.00246, normalized entropy 0.97466) which are
    explicitly NOT evidence of policy value.

    A full flag-ON production episode needs live ARC environments and the ARC-AGI-3
    harness. This probe instead measures the SAME readout on waves produced by the
    LIVE VISION ENCODER over the LOCAL ARC CORPUS -- the real ingress path, not
    isotropic noise. That is the cheapest honest source of live-like waves.

WHAT IT MEASURES
    For each real ARC grid:
      - encode through HENRIVisionEncoder (the production ingress used by
        production_arc_run: tokenizer = HENRIVisionEncoder(...))
      - reshape the [d_model] wave to [num_blocks, 8] (the egress boundary shape)
      - decode through the sealed codebook and record top1_margin, normalized
        entropy, chosen action
    Then report the distribution: mean, p01/p05/p10/p25/p50/p90, min/max, plus how a
    candidate floor would behave (abstain rate at each percentile).

HONEST LIMITS RECORDED IN THE RECEIPT
    - These are ENCODER waves, not EFE-planned candidate waves. A planner-selected
      candidate is a transformed wave; its margin distribution may differ.
    - The action alphabet is the env's allowed set; here it is the ARC GameAction
      set, which is what production uses.
    - No task score is produced or implied. Correctness is NOT measured because there
      is no ground-truth action for an arbitrary ARC grid.
"""
import json
import pathlib
import sys
from datetime import datetime, timezone

import numpy as np
import torch

R = pathlib.Path(r"C:\Users\chan\Desktop\HENRI 7B SWARM\.worktrees\basal-syncytium\HENRI V2")
sys.path.insert(0, str(R))
sys.path.insert(0, str(R / "experiments" / "verification"))

from arc_egress_contract import (  # noqa: E402
    ActionEgressVocabulary,
    build_sealed_action_codebook,
    decode_action_egress_sealed,
)
from henri_vision_encoder import HENRIVisionEncoder  # noqa: E402

ARC_CANDIDATES = [
    pathlib.Path(r"C:\Users\chan\henri_data\ARC-AGI\data\training"),
    pathlib.Path(r"C:\Users\chan\henri_data\ARC-AGI\data\evaluation"),
]

D_MODEL = 512
K_BLOCKS = 64
GRID = 8
MAX_TASKS = 120

print("=" * 78)
print("LOCATE THE LOCAL ARC CORPUS")
print("=" * 78)
arc_root = None
for c in ARC_CANDIDATES:
    if c.is_dir():
        n = len(list(c.glob("*.json")))
        print(f"   PRESENT {c}  ({n} tasks)")
        if n and arc_root is None:
            arc_root = c
print(f"   using: {arc_root}")
if arc_root is None:
    print("BLOCKED: no local ARC corpus found; cannot measure live-wave margins.")
    raise SystemExit(2)

print()
print("=" * 78)
print("BUILD the production ingress + sealed codebook")
print("=" * 78)
enc = HENRIVisionEncoder(d_model=D_MODEL, k_blocks=K_BLOCKS)
print(f"   HENRIVisionEncoder(d_model={D_MODEL}, k_blocks={K_BLOCKS})")

try:
    from arcengine import GameAction
    ACTIONS = list(GameAction)
    print(f"   action alphabet: arcengine.GameAction ({[a.name for a in ACTIONS]})")
except Exception as exc:  # noqa: BLE001
    print(f"   arcengine unavailable ({type(exc).__name__}); using a local 8-action enum")
    import enum

    class GameAction(enum.Enum):  # type: ignore
        RESET = 0
        ACTION1 = 1
        ACTION2 = 2
        ACTION3 = 3
        ACTION4 = 4
        ACTION5 = 5
        ACTION6 = 6
        ACTION7 = 7

    ACTIONS = list(GameAction)

vocab = ActionEgressVocabulary(GameAction, ACTIONS)
codebook = build_sealed_action_codebook(vocab, D_MODEL)
print(f"   sealed codebook: {vocab.n_actions} actions, "
      f"M={tuple(codebook.codebook_M.shape)}")

print()
print("=" * 78)
print("ENCODE real ARC grids -> live waves -> decode -> margins")
print("=" * 78)


def grid_to_tensor(g):
    """ARC grid -> [1, H, W] long, resized to GRID x GRID by nearest sampling."""
    a = torch.tensor(g, dtype=torch.float32)
    a = a.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    a = torch.nn.functional.interpolate(a, size=(GRID, GRID), mode="nearest")
    return a.squeeze(0).long()  # [GRID, GRID]


rows, enc_fail = [], 0
tasks = sorted(arc_root.glob("*.json"))[:MAX_TASKS]
for tp in tasks:
    try:
        d = json.loads(tp.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        continue
    for key in ("train", "test"):
        for pair in d.get(key, []):
            for side in ("input", "output"):
                g = pair.get(side)
                if not g:
                    continue
                try:
                    gt = grid_to_tensor(g)
                    with torch.no_grad():
                        w = enc.encode_grid(gt)
                    w = w.reshape(K_BLOCKS, 8).to(torch.float32)
                    res = decode_action_egress_sealed(w, vocab, codebook, D_MODEL)
                    p = res.action_probs
                    t2 = torch.topk(p, 2).values
                    rows.append({
                        "task": tp.stem, "split": key, "side": side,
                        "action": res.action_name,
                        "margin": float(t2[0] - t2[1]),
                        "norm_entropy": res.entropy_bits,
                        "top1": float(p.max()),
                    })
                except Exception:  # noqa: BLE001
                    enc_fail += 1

print(f"   waves decoded : {len(rows)}")
print(f"   encode/decode failures: {enc_fail}")
if not rows:
    print("BLOCKED: no waves decoded.")
    raise SystemExit(2)

margins = np.array([r["margin"] for r in rows])
nents = np.array([r["norm_entropy"] for r in rows])
top1s = np.array([r["top1"] for r in rows])
acts = [r["action"] for r in rows]
distinct = sorted(set(acts))

print()
print("=" * 78)
print("MARGIN DISTRIBUTION ON LIVE ENCODER WAVES")
print("=" * 78)
print(f"   chance top1 (1/|A|)     : {1.0 / vocab.n_actions:.4f}")
print(f"   mean top1 prob          : {top1s.mean():.4f}")
print(f"   mean margin             : {margins.mean():.6f}")
print(f"   margin std              : {margins.std():.6f}")
for q in (1, 5, 10, 25, 50, 75, 90, 95, 99):
    print(f"   margin p{q:<3}              : {np.percentile(margins, q):.6f}")
print(f"   margin min / max        : {margins.min():.6f} / {margins.max():.6f}")
print(f"   mean normalized entropy : {nents.mean():.6f}  (1.0 = uniform)")
print(f"   entropy min / max       : {nents.min():.6f} / {nents.max():.6f}")
print(f"   distinct actions chosen : {len(distinct)} of {vocab.n_actions} -> {distinct}")

print()
print("   action histogram (top10):")
from collections import Counter  # noqa: E402
for a, c in Counter(acts).most_common(10):
    print(f"     {a:<10} {c:>5}  ({100.0 * c / len(acts):5.1f}%)")

print()
print("=" * 78)
print("FLOOR BEHAVIOUR: abstain rate at candidate floors")
print("=" * 78)
print(f"   {'floor':>8}  {'abstain %':>10}  {'surviving':>10}")
floors = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
for f in floors:
    if f == 0.0:
        abst = 0.0
        surv = len(margins)
    else:
        abst = float((margins < f).mean() * 100.0)
        surv = int((margins >= f).sum())
    print(f"   {f:>8.3f}  {abst:>9.2f}%  {surv:>10}")

print()
print("=" * 78)
print("CONTRAST with the earlier RANDOM-wave figures (must not be conflated)")
print("=" * 78)
print(f"   random-wave mean margin   : 0.050150   (n=200, isotropic noise)")
print(f"   LIVE-wave  mean margin    : {margins.mean():.6f}   (n={len(rows)}, real grids)")
print(f"   random-wave mean norm ent : 0.974660")
print(f"   LIVE-wave  mean norm ent  : {nents.mean():.6f}")
delta = margins.mean() - 0.050150
print(f"   difference                : {delta:+.6f}")

print()
print("=" * 78)
print("RECOMMENDED FLOOR (from the LIVE distribution only)")
print("=" * 78)
p05 = float(np.percentile(margins, 5))
p10 = float(np.percentile(margins, 10))
print(f"   p05 = {p05:.6f}  (would abstain on 5% of live waves)")
print(f"   p10 = {p10:.6f}  (would abstain on 10% of live waves)")
print("   -> the floor is a POLICY choice about abstention tolerance; both are")
print("      reported rather than one being asserted as correct.")

out = R / "experiments" / "verification" / "sealed_margin_live_observed.json"
body = {
    "schema": "henri.sealed-margin-live.v1",
    "evidence_class": "OBSERVED",
    "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "source": {"arc_root": str(arc_root), "tasks": len(tasks),
               "encoder": f"HENRIVisionEncoder(d_model={D_MODEL}, k_blocks={K_BLOCKS})",
               "grid": GRID, "action_alphabet": [a.name for a in ACTIONS]},
    "waves_decoded": len(rows),
    "encode_decode_failures": enc_fail,
    "margin": {
        "mean": float(margins.mean()), "std": float(margins.std()),
        "min": float(margins.min()), "max": float(margins.max()),
        **{f"p{q}": float(np.percentile(margins, q))
           for q in (1, 5, 10, 25, 50, 75, 90, 95, 99)},
    },
    "normalized_entropy": {"mean": float(nents.mean()),
                           "min": float(nents.min()),
                           "max": float(nents.max())},
    "top1": {"mean": float(top1s.mean())},
    "distinct_actions": distinct,
    "action_histogram": dict(Counter(acts)),
    "floor_behaviour": {str(f): {"abstain_pct": float((margins < f).mean() * 100.0)
                                 if f > 0 else 0.0,
                                 "surviving": int((margins >= f).sum())}
                        for f in floors},
    "contrast_random_waves": {
        "random_mean_margin": 0.050150, "random_mean_norm_entropy": 0.974660,
        "live_minus_random_margin": delta,
        "note": "the random-wave figures were explicitly NOT evidence of policy value",
    },
    "recommended_floor_from_live": {"p05": p05, "p10": p10},
    "non_claims": [
        "ENCODER waves, NOT EFE-planned candidate waves; a planner-selected wave is "
        "a transformed wave and its margin distribution may differ.",
        "No task was attempted or solved. No ground-truth action exists for an "
        "arbitrary ARC grid, so correctness is NOT measured.",
        "A floor chosen here controls ABSTENTION rate, not accuracy.",
        "The live benchmark score remains 0.0%.",
    ],
}
with open(out, "w", encoding="utf-8", newline="") as fh:
    fh.write(json.dumps(body, indent=2) + "\n")
print(f"\nwrote {out}")
print(f"canonical sha256 = {__import__('hashlib').sha256(out.read_bytes()).hexdigest()}")
