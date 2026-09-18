"""PRIORITY 2 — sealed-egress margin on EFE-PLANNED candidate waves.

WHY THIS EXISTS
    The standing negative (margin 0.023355 vs random 0.050150) was measured on
    ENCODER waves: `HENRIVisionEncoder.encode_spatial_grid` output. But the sealed
    egress does NOT consume encoder waves in production. It consumes the EFE
    planner's chosen prediction:

        production_arc_run.py:2438
            _decode_action_egress_sealed(chosen["predicted_wave"], ...)

    So the honest Priority-2 question is whether the margin on EFE-PLANNED waves
    differs from the encoder-wave negative. This probe answers that.

HONEST SCALE LABEL -- READ THIS BEFORE QUOTING ANY NUMBER
    This runs at D=512, K=64, NOT the production D=65536. The reason is structural,
    not convenience: efe_planner selects checkpoint_policy="required" only when
    d_model == 65536, and no checkpoint exists on this host. So the PRODUCTION
    planner cannot be constructed here at all.

    Verdict class is therefore `OBSERVED_REDUCED_SCALE`. It is NOT a confirmation
    or refutation of the production path. It cannot be.

DESIGN
    * Real ARC grids through the production encoder (no synthetic waves).
    * The transition learner is ENGAGED by training on real encoded ARC
      demonstration pairs, so "planned" predictions are not untrained noise. If the
      learner does not engage, the run reports BLOCKED_NO_ENGAGEMENT instead of a
      margin number.
    * boundary_axioms provenance is printed and labelled. Zone C's real axioms need
      a Postgres DSN that is deliberately absent on this host (fail-closed), so
      seeded unit-norm stand-ins are used and NAMED as stand-ins. They affect the
      planner's action CHOICE; the measured margin is on whatever wave the planner
      then produces.
    * Both arms use the SAME sealed codebook and the SAME action vocabulary, so the
      comparison is like-for-like.
"""
import json
import pathlib
import statistics
import sys
from datetime import datetime, timezone

R = pathlib.Path(r"C:\Users\chan\Desktop\HENRI 7B SWARM\.worktrees\basal-syncytium\HENRI V2")
VERIF = R / "experiments" / "verification"
for p in (str(VERIF), str(R)):
    sys.path.insert(0, p)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from receipt_path import resolve_receipt_path  # noqa: E402

OUT = resolve_receipt_path(
    VERIF / "sealed_margin_efe_planned_observed.json")

D_MODEL = 512
K_BLOCKS = 64
BLOCK_DIM = D_MODEL // K_BLOCKS
N_TASKS = 6
AXIOM_SEED = 20260918
ACTION_SEED = 20260919
N_AXIOMS = 11


def pct(vals, q):
    if not vals:
        return None
    s = sorted(vals)
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return float(s[i])


def main():
    from henri_vision_encoder import HENRIVisionEncoder
    from efe_planner import EFEPlanner
    from arc_egress_contract import (
        ActionEgressVocabulary, build_sealed_action_codebook,
        decode_action_egress_sealed, real_to_phase_wave,
    )
    try:
        from arcengine import GameAction
    except Exception as exc:  # noqa: BLE001
        print(f"BLOCKED_NO_ACTION_ENUM: {type(exc).__name__}: {exc}")
        return 1
    import run_calibration_eval as RCE

    print("=" * 78)
    print("PRIORITY 2 — sealed margin on EFE-PLANNED waves (REDUCED SCALE)")
    print("=" * 78)
    print(f"D_MODEL={D_MODEL}  K_BLOCKS={K_BLOCKS}  BLOCK_DIM={BLOCK_DIM}")
    print("VERDICT CLASS: OBSERVED_REDUCED_SCALE (production is D=65536 and "
          "requires a checkpoint that is absent here)")

    enc = HENRIVisionEncoder(d_model=D_MODEL, k_blocks=K_BLOCKS,
                             block_dim=BLOCK_DIM, device="cpu",
                             spatial_basis_kind="incommensurate", bg_mask=True)

    def flat_wave(grid):
        w = enc.encode_spatial_grid([list(r) for r in grid])
        if w is None:
            return None
        v = w.squeeze(0).reshape(-1).detach().to(torch.float32)
        if v.numel() != D_MODEL or not torch.isfinite(v).all():
            return None
        return F.normalize(v, p=2.0, dim=-1)

    tasks = RCE.load_arc(RCE.ARC_ROOT, N_TASKS)
    if not tasks:
        print("BLOCKED_NO_CORPUS")
        return 1
    tasks = tasks[:N_TASKS]
    print(f"tasks = {len(tasks)}")

    # ---- action vocabulary over the REAL enum
    allowed = [a for a in GameAction if a.name != "RESET"]
    vocab = ActionEgressVocabulary(GameAction, allowed)
    print(f"action vocabulary: n={vocab.n_actions} "
          f"{[a.name for a in vocab.actions]}")
    codebook = build_sealed_action_codebook(vocab, D_MODEL)
    print("sealed codebook:  built (checkpoint-free)")

    # ---- planner at reduced scale
    planner = EFEPlanner(num_blocks=K_BLOCKS, d_model=D_MODEL,
                         num_actions=vocab.n_actions, transition_rank=16)
    print(f"EFEPlanner constructed: num_blocks={K_BLOCKS} d_model={D_MODEL} "
          f"num_actions={vocab.n_actions}")

    # ---- action waves: deterministic seeded unit-norm [num_blocks, 8].
    # HARNESS DEFECT CORRECTED HERE: the first version of this probe passed action
    # INDICES (dtype=long) and bare enum members. The planner consumes action
    # WAVES: efe_planner.py:1258 -- "All tensors are stacked [N, num_blocks, 8]
    # real Clifford waves" -- and score_actions takes (action_id, action_wave)
    # tuples (efe_planner.py:948). The mismatch raised
    # "size of tensor a (4) must match the size of tensor b (508)", which the
    # probe correctly reported as BLOCKED_NO_ENGAGEMENT rather than emitting a
    # margin from untrained noise.
    # These waves are SEEDED STAND-INS, named as such: not a learned embedding.
    ga = torch.Generator().manual_seed(ACTION_SEED)
    action_waves = F.normalize(
        torch.randn(vocab.n_actions, K_BLOCKS, BLOCK_DIM, generator=ga),
        p=2.0, dim=-1)
    candidates = [(i, action_waves[i]) for i in range(vocab.n_actions)]
    print(f"action_waves: {tuple(action_waves.shape)} SEEDED STAND-INS "
          f"(seed={ACTION_SEED}), per-block unit norm, "
          f"{len(candidates)} candidates")

    # ---- boundary axioms: SEEDED STAND-INS, provenance named
    g = torch.Generator().manual_seed(AXIOM_SEED)
    ax = torch.randn(N_AXIOMS, K_BLOCKS, BLOCK_DIM, generator=g)
    ax = F.normalize(ax, p=2.0, dim=-1)
    print(f"boundary_axioms: {tuple(ax.shape)} from SEEDED STAND-INS "
          f"(seed={AXIOM_SEED}) -- NOT Zone C's real axioms (no DSN on this host)")

    # ---- ENGAGE the transition learner on real encoded ARC pairs
    states, actions, nxts = [], [], []
    for ti, (tid, t) in enumerate(tasks):
        for pi, pair in enumerate(t["train"][:3]):
            a = flat_wave(pair["input"])
            b = flat_wave(pair["output"])
            if a is None or b is None:
                continue
            states.append(a.reshape(K_BLOCKS, BLOCK_DIM))
            nxts.append(b.reshape(K_BLOCKS, BLOCK_DIM))
            actions.append(action_waves[(ti * 3 + pi) % vocab.n_actions])
    # NOTE ON SHAPE (harness defect, corrected): train_transition_batch requires
    # BLOCK-SHAPED [N, num_blocks, 8] waves -- efe_planner.py:1258, "All tensors
    # are stacked [N, num_blocks, 8] real Clifford waves". Passing FLAT [N, d]
    # waves makes bind's state_wave[..., :4] slice a 1-D vector to length 4, which
    # raises "size of tensor a (4) must match the size of tensor b (508)". The
    # FIRST failure was the action format (indices instead of waves); this is the
    # state format. Both were caught by the probe's own fail-closed gate.
    eng_loss = None
    if len(states) >= 4:
        try:
            eng_loss = planner.train_transition_batch(
                torch.stack(states), torch.stack(actions),
                torch.stack(nxts), iters=3)
            print(f"transition learner ENGAGED: {len(states)} real pairs, "
                  f"loss={eng_loss}")
        except Exception as exc:  # noqa: BLE001
            print(f"train_transition_batch failed: {type(exc).__name__}: {exc}")
    else:
        print(f"too few pairs to train: {len(states)}")

    if eng_loss is None:
        print("BLOCKED_NO_ENGAGEMENT — the transition learner did not engage, so "
              "'planned' waves would be untrained noise. No margin number emitted.")
        return 2

    # ---- collect ENCODER waves and EFE-PLANNED waves
    enc_rows, plan_rows = [], []
    for tid, t in tasks:
        x = flat_wave(t["test"][0]["input"])
        if x is None:
            continue
        # encoder-wave arm
        for a in vocab.actions:
            r = decode_action_egress_sealed(x.reshape(K_BLOCKS, BLOCK_DIM),
                                           vocab, codebook, D_MODEL)
            enc_rows.append({"task": tid, "margin": r.top1_margin,
                             "entropy": r.entropy_bits,
                             "action": r.action_name})
        # EFE-planned arm
        try:
            res = planner.select_action(
                x.reshape(K_BLOCKS, BLOCK_DIM), candidates, ax)
            # Documented return: (best_action_id, predicted_wave, scores_table,
            # chosen_dict). Try the dict first, then the positional element.
            chosen = res[3] if isinstance(res, tuple) and len(res) > 3 else None
            pw = None
            if isinstance(chosen, dict):
                pw = chosen.get("predicted_wave")
            if pw is None and isinstance(res, tuple) and len(res) > 1 \
                    and torch.is_tensor(res[1]):
                pw = res[1]
            if pw is None:
                print(f"   {tid}: no predicted_wave in chosen_dict keys="
                      f"{list(chosen.keys()) if isinstance(chosen, dict) else type(chosen)}")
                continue
            pw = pw.detach().to(torch.float32)
            if pw.numel() != D_MODEL:
                pw = pw.reshape(-1)
            if pw.numel() != D_MODEL:
                print(f"   {tid}: predicted_wave numel={pw.numel()} != {D_MODEL}")
                continue
            pw = F.normalize(pw, p=2.0, dim=-1)
            r = decode_action_egress_sealed(pw.reshape(K_BLOCKS, BLOCK_DIM),
                                           vocab, codebook, D_MODEL)
            plan_rows.append({"task": tid, "margin": r.top1_margin,
                              "entropy": r.entropy_bits,
                              "action": r.action_name})
        except Exception as exc:  # noqa: BLE001
            print(f"   {tid}: select_action failed: {type(exc).__name__}: {exc}")

    def stats(rows):
        m = [r["margin"] for r in rows if r["margin"] is not None]
        e = [r["entropy"] for r in rows if r["entropy"] is not None]
        return {
            "n": len(rows), "n_margin": len(m),
            "margin_mean": (statistics.fmean(m) if m else None),
            "margin_p05": pct(m, 0.05), "margin_p50": pct(m, 0.50),
            "margin_p95": pct(m, 0.95),
            "entropy_mean": (statistics.fmean(e) if e else None),
            "distinct_actions": len({r["action"] for r in rows}),
        }

    se, sp = stats(enc_rows), stats(plan_rows)
    print()
    print("-" * 78)
    print(f"ENCODER waves  : {se}")
    print(f"EFE-PLANNED    : {sp}")
    print("-" * 78)

    delta = None
    if se["margin_mean"] is not None and sp["margin_mean"] is not None:
        delta = sp["margin_mean"] - se["margin_mean"]
        print(f"margin delta (planned - encoder) = {delta:+.6f}")

    # Random reference from the prior measurement, at the SAME dimension.
    RANDOM_REF = 0.050150
    informative = bool(sp["margin_mean"] is not None
                       and sp["margin_mean"] > RANDOM_REF)

    body = {
        "schema": "henri.sealed-margin-efe-planned.v1",
        "utc": datetime.now(timezone.utc).isoformat(),
        "evidence_class": "OBSERVED",
        "verdict_class": "OBSERVED_REDUCED_SCALE",
        "why_reduced_scale": (
            "efe_planner selects checkpoint_policy='required' only at d_model == "
            "65536, and no checkpoint exists on this host, so the PRODUCTION "
            "planner cannot be constructed. This is a reduced-scale observation, "
            "not a confirmation or refutation of the production path."
        ),
        "question": (
            "The standing negative (margin 0.023355 vs random 0.050150) was "
            "measured on ENCODER waves, but the sealed egress consumes the EFE "
            "planner's chosen prediction (production_arc_run.py:2438). Does the "
            "margin differ on EFE-PLANNED waves?"
        ),
        "design": {
            "d_model": D_MODEL, "k_blocks": K_BLOCKS, "block_dim": BLOCK_DIM,
            "n_tasks": len(tasks), "n_axioms": N_AXIOMS,
            "axiom_provenance": ("SEEDED STAND-INS, not Zone C real axioms "
                                 f"(seed={AXIOM_SEED}); no Postgres DSN on this host"),
            "transition_engaged": True,
            "transition_train_loss": eng_loss,
            "action_vocabulary": [a.name for a in vocab.actions],
            "random_reference_margin": RANDOM_REF,
        },
        "encoder_waves": se,
        "efe_planned_waves": sp,
        "margin_delta_planned_minus_encoder": delta,
        "planned_margin_exceeds_random_reference": informative,
        "encoder_rows": enc_rows,
        "planned_rows": plan_rows,
        "verdict": (
            "PLANNED_MARGIN_EXCEEDS_RANDOM_REDUCED_SCALE — on EFE-planned waves the "
            "sealed margin is ABOVE the random-wave reference, so the encoder-wave "
            "negative does NOT transfer to the planner's predicted waves. This "
            "REFINES the standing negative rather than overturning it: the "
            "production path still requires its own flag-ON measurement at "
            "D=65536, which stays BLOCKED here."
            if informative else
            "PLANNED_MARGIN_ALSO_AT_OR_BELOW_RANDOM_REDUCED_SCALE — the negative "
            "reproduces on EFE-planned waves at reduced scale, so the deficit is "
            "not an artefact of measuring encoder waves. Still not a production "
            "verdict: D=65536 with a loaded checkpoint remains BLOCKED."
        ),
        "limits": [
            "REDUCED SCALE (D=512). Production is D=65536 and cannot be "
            "constructed here without the absent checkpoint.",
            "boundary_axioms are seeded stand-ins, not Zone C's real priors.",
            "The transition learner is engaged on real ARC pairs, but at reduced "
            "rank (16) and few iterations; it is not a trained production model.",
            "No floor is set from this run. Setting HENRI_SEALED_EGRESS_MIN_MARGIN "
            "from a reduced-scale stand-in run would repeat the error the "
            "encoder-wave measurement was meant to prevent.",
            "Not a capability claim; benchmark score remains 0.0%.",
        ],
    }
    OUT.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    print(f"\nVERDICT: {body['verdict'][:100]}...")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
