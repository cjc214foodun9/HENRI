"""
PROJECT HENRI: ARC-AGI-3 Production Benchmark Run.

End-to-end live run of the post-refactor stack against the real ARC-AGI-3
arcade on the RTX 5090:

    O-VSA fractional binding (grid -> wave)
      -> Zone C GRM recall (long-term conditioning)
      -> Swarm relaxation (SGLD creep, IDBD step-sizes)
      -> EFE action selection (top-k through the unitary transition)
      -> Environment step
      -> Sagnac verification + telemetry + Zone C checkpoint

Telemetry: every environment step logs a dense latent record to the
TimescaleDB hypertables (zone_c_resonant_hypersphere for wave statistics,
plus a JSONL mirror in telemetry_logs/ for offline analysis):
    sagnac_delta, sagnac_coherence, free_energy (propagation stress +
    boundary resonance), kuramoto order parameter, EFE table (pragmatic /
    epistemic per candidate), IDBD plasticity stats (mean/max alpha, frozen
    fraction), chosen action, hopfield confidence, recall hits/gates.

Run on the 5090:
    POSTGRES_DSN=postgres://postgres:henri@localhost:10100/henri \
        python3 production_arc_run.py [--envs N] [--steps M]
"""

import argparse
import json
import math
import os
import time
import uuid
from datetime import datetime, timezone

import torch

import arc_agi
from arcengine import GameAction

from darwinian_phase_swarm import HenriSwarmOrchestrator
from o_vsa_ingress_tokenizer import O_VSA_IngressTokenizer
from thermodynamic_telemetry_logger import ThermodynamicTelemetryLogger

DSN = os.environ.get("POSTGRES_DSN", "postgres://postgres:henri@localhost:10100/henri")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Scale: production on GPU, reduced on CPU
if DEVICE == "cuda":
    SCALE = dict(num_experts=1024, d_model=65536, r_rank=16, num_blocks=8192)
else:
    SCALE = dict(num_experts=64, d_model=512, r_rank=8, num_blocks=64)

RELAX_STEPS = 32         # swarm relaxation iterations per environment step
                         # (r collapses to ~0.01 within 8 steps = under-relaxed;
                         # 32 gives the wave the full non-equilibrium budget)
RECALL_EVERY = 5         # recall Zone C conditioning every N steps
CHECKPOINT_EVERY = 10    # persist engram every N steps
EDMD_EVERY = 16          # NL Level 2: mid-frequency EDMD fit every K steps
EDMD_WINDOW = 64         # rolling buffer depth for the mid-frequency fit


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------

class LatentTelemetry:
    """Dense per-step latent-space record: JSONL mirror + hypertable waves."""

    def __init__(self, log_path, db_logger=None):
        self.log_path = log_path
        self.db = db_logger
        self.run_id = str(uuid.uuid4())[:8]
        self._fp = open(log_path, "a", buffering=1)

    def emit(self, record: dict):
        record["run_id"] = self.run_id
        record["ts"] = datetime.now(timezone.utc).isoformat()
        self._fp.write(json.dumps(record) + "\n")

    def close(self):
        self._fp.close()


# ---------------------------------------------------------------------------
# Core per-step pipeline
# ---------------------------------------------------------------------------

def kuramoto_order_parameter(phases: torch.Tensor) -> float:
    """r = |mean(e^{i theta})| over expert phases; 1 = perfect phase-lock."""
    z = torch.exp(1j * phases)
    return torch.abs(z.mean()).item()


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=3)
    ap.add_argument("--steps", type=int, default=30, help="max env steps per environment")
    ap.add_argument("--dsn", type=str, default=DSN)
    args = ap.parse_args()

    print("=" * 70)
    print("  PROJECT HENRI: ARC-AGI-3 PRODUCTION RUN")
    print(f"  device={DEVICE} scale={SCALE} run telemetry=zone_c + jsonl")
    print("=" * 70)

    os.makedirs("telemetry_logs", exist_ok=True)
    log_path = os.path.join(
        "telemetry_logs", f"production_run_{int(time.time())}.jsonl"
    )
    db_logger = None
    try:
        db_logger = ThermodynamicTelemetryLogger(db_conn_str=args.dsn, batch_size=100)
    except Exception as e:
        print(f"[telemetry] hypertable logger offline ({e}); JSONL only")
    tele = LatentTelemetry(log_path, db_logger)

    print(f"[init] orchestrator @ {SCALE}")
    orch = HenriSwarmOrchestrator(action_enum_class=GameAction, **SCALE).to(DEVICE)
    orch.attach_zone_c(dsn=args.dsn if DEVICE == "cuda" else "offline://surrogate")
    tokenizer = O_VSA_IngressTokenizer(
        num_blocks=SCALE["num_blocks"], vocab_size=256, device=DEVICE
    )

    arcade = arc_agi.Arcade()
    env_ids = [e.game_id if hasattr(e, "game_id") else e for e in arcade.available_environments][: args.envs]
    print(f"[init] {len(env_ids)} environments: {env_ids}")

    for env_name in env_ids:
        print(f"\n{'─'*70}\n  ENV: {env_name}\n{'─'*70}")
        try:
            game = arcade.make(env_name)
        except Exception as e:
            print(f"  [skip] make failed: {e}")
            continue
        obs = game.reset()
        if obs is None or not getattr(obs, "frame", None):
            print("  [skip] null initial frame")
            continue

        prev_wave = None
        prev_raw_wave = None
        prev_predicted_prior = None
        train_ctx = None
        action_counts = {}
        edmd_buffer = []  # (state, action_wave, observed_next) triples
        # Wire B valence: outcome signal for the LAST executed action,
        # computed when the next observation arrives and consumed by the
        # deferred T1 update + the current relaxation's thermal schedule.
        valence = 0.0
        last_action_was_reset = False
        # Reset-transition curation (run-10 ablation result): the deferred T1
        # update is HELD after every RESET, permanently. Run 9's retroactive
        # nu-judgment apparatus (k=5 eligibility buffer, replays, preference
        # consolidation) was falsified as the RESET-spam driver — curation
        # alone reproduced the compression (15.8% vs 18.3%, baseline 38.7%).
        # The entire effect reduces to one rule:
        #   don't train on reset transitions.

        for step in range(args.steps):
            t0 = time.perf_counter()
            grid = obs.frame[0].tolist()
            state_wave = tokenizer.encode_spatial_grid(grid).squeeze(0).to(DEVICE)
            raw_wave = state_wave  # pre-blend; recall blending mutates below

            # Valence extraction from observable outcome signals (no
            # teleology: only deltas the environment actually reports).
            # Compare against the PRE-BLEND raw observation wave — recall
            # blending mutates state_wave below and would fake a frame change.
            frame_changed = (
                prev_raw_wave is None
                or not torch.allclose(state_wave, prev_raw_wave, atol=1e-5)
            )
            if last_action_was_reset and not frame_changed:
                valence = -1.0   # null action: RESET with no state change
            elif last_action_was_reset and frame_changed:
                valence = 0.0    # legitimate RESET (new puzzle instance)
            else:
                valence = 0.0    # neutral exploration (WIN handled at terminal)

            # T1: apply the deferred transition-model update using the previous
            # step's (state, executed action) -> this step's observed wave.
            # A RESET's deferred update is HELD (curation): reset transitions
            # never enter the training set, so the transition model stays
            # under-confident on RESET outcomes and the planner routes away.
            transition_loss = None
            if (train_ctx is not None and train_ctx["action_wave"] is not None
                    and not train_ctx.get("pending_reset")):
                # NL Level 1 (fast): surprise-modulated per-step SGLD, now
                # valence-gated (Wire B planner-side: success crystallizes,
                # failure stays plastic but refuses to consolidate).
                transition_loss = orch.planner.train_transition_step(
                    train_ctx["state"], train_ctx["action_wave"], state_wave,
                    lr=0.05, valence=valence,
                )
                # Wire A: consolidate favorable trajectories into the
                # pragmatic preference store.
                if valence > 0.0:
                    orch.planner.register_preference(state_wave)
                # Accumulate the triple for the slower consolidation levels.
                edmd_buffer.append((train_ctx["state"], train_ctx["action_wave"],
                                    state_wave))
                # NL Level 2 (mid-frequency): EDMD fit over the rolling window
                # at strict chunk boundaries (i ≡ 0 mod K, HOPE CMS style).
                if len(edmd_buffer) % EDMD_EVERY == 0:
                    window = edmd_buffer[-EDMD_WINDOW:]
                    edmd_loss = orch.planner.train_transition_batch(
                        torch.stack([t[0] for t in window]),
                        torch.stack([t[1] for t in window]),
                        torch.stack([t[2] for t in window]),
                    )
                    print(f"  [edmd-L2] step {step}: window {len(window)} "
                          f"batch loss {edmd_loss:.4f}")
                    tele.emit({"env": env_name, "step": step, "edmd_L2_loss":
                               round(edmd_loss, 6), "edmd_L2_window": len(window)})
            train_ctx = None

            # Boundary axiom = prediction error: observed state vs the dynamics
            # prior propagated by the EFE transition model (falling back to the
            # raw inter-frame transition before the first prediction exists).
            if prev_predicted_prior is not None:
                boundary = state_wave - prev_predicted_prior
            elif prev_wave is not None:
                boundary = state_wave - prev_wave
            else:
                boundary = state_wave.clone()
            boundary = boundary / (torch.norm(boundary, p=2, dim=-1, keepdim=True) + 1e-9)

            # Zone C recall (conditioning) on schedule; blend the recalled
            # long-term engram into the active wave to bias relaxation.
            recalled = None
            recall_info = {"hits": 0}
            if step % RECALL_EVERY == 0:
                res = orch.segment_cache.retrieve(state_wave.cpu())
                recalled = res["conditioning_wave"]
                recall_info = {"hits": res["hits"],
                               "top_sim": res.get("top_similarity", 0.0),
                               "gates": [round(g, 4) for g in res.get("gates", [])]}
                if recalled is not None:
                    recalled = recalled.to(DEVICE)
                    # Memory-conditioned state: partial blend toward the recalled
                    # attractor (keeps the live observation dominant).
                    state_wave = 0.7 * state_wave + 0.3 * recalled
                    state_wave = state_wave / (torch.norm(state_wave, p=2, dim=-1, keepdim=True) + 1e-9)

            # Swarm relaxation with SGLD creep (valence drives the thermal
            # schedule: failure heats, success cools)
            sagnac_delta = None
            for _ in range(RELAX_STEPS):
                sagnac_delta, _, _ = orch.process_active_reasoning_step(
                    state_wave, boundary,
                    t_shock_max=torch.tensor(0.5, device=DEVICE),
                    valence=valence,
                )

            # Latent metrics
            coherence = orch.sagnac_coherence(state_wave, boundary).item()
            free_energy = orch.compute_free_energy(state_wave, boundary).item()
            order_param = kuramoto_order_parameter(orch.syncytium.expert_phases)
            plasticity = {
                k: round(v, 6)
                for k, v in orch.syncytium.creep_ctrl_A.plasticity_stats().items()
            }

            # EFE action selection (T4: explore when the planner is confused)
            boundary_batch = boundary.unsqueeze(0)
            action, predicted_wave, efe_table, chosen = orch.plan_action(
                state_wave, boundary_batch, top_k=4, return_chosen=True
            )
            explored = bool(chosen.get("explored", False))
            hop_conf = chosen["efe"]  # chosen-candidate EFE as confidence proxy
            loss_ema = orch.planner.loss_ema

            # Epistemic novelty: record the chosen action's predicted outcome so
            # repeating it later is discounted (breaks RESET-spam loops).
            orch.planner.remember_outcome(chosen["predicted_wave"])

            # Environment step
            game_action = action if isinstance(action, GameAction) else GameAction.ACTION1
            obs_next = game.step(game_action)
            step_ms = (time.perf_counter() - t0) * 1000
            last_action_was_reset = (game_action.name == "RESET")

            # T1/T2: train the transition model on the EXECUTED action pair.
            # observed_next is encoded on the NEXT loop iteration; stash the
            # training context now and apply the update once the next frame's
            # wave exists. Loss = Sagnac delta(predicted, observed_next).
            # Subtraction-tautology guard: training is deferred one step and
            # always against the OBSERVED wave, never the planner's own
            # prediction in the same step.
            train_ctx = {
                "state": state_wave.detach(),
                "action_wave": next(
                    (w for a, w in orch.candidate_action_waves(top_k=len(orch.decoder.id_to_action))
                     if a == game_action),
                    None,
                ),
                "pending_reset": game_action.name == "RESET",
            }

            # Track the propagated prior: the EFE planner's predicted wave becomes
            # the dynamics prior that conditions the next step's encoding, so the
            # model's action choices meaningfully differentiate trajectories.
            predicted_prior = predicted_wave.detach()

            # Telemetry emit (dense latent record)
            action_counts[game_action.name] = action_counts.get(game_action.name, 0) + 1
            tele.emit({
                "env": env_name, "step": step,
                "sagnac_delta": round(sagnac_delta, 6),
                "sagnac_coherence": round(coherence, 6),
                "free_energy": round(free_energy, 6),
                "kuramoto_r": round(order_param, 6),
                "plasticity": plasticity,
                "efe_best": round(chosen["efe"], 6),
                "efe_spread": round(efe_table[-1]["efe"] - efe_table[0]["efe"], 6),
                "explored": explored,
                "loss_ema": round(loss_ema, 6),
                "transition_loss": round(transition_loss, 6) if transition_loss is not None else None,
                "valence": valence,
                "preference_store_size": orch.planner.preference_store.num_engrams(),
                "action": str(game_action),
                "recall": recall_info,
                "step_ms": round(step_ms, 1),
            })
            # Wave-level hypertable log (downsampled for DB volume)
            if db_logger is not None and step % 5 == 0:
                db_logger.log_trajectory(
                    domain="ARC_AGI_3", subdomain=env_name,
                    concept_key=f"step_{step}",
                    predicted_wave=predicted_wave.view(-1)[:4096],
                    phase_delta=sagnac_delta, is_valid=sagnac_delta < 0.5,
                )

            # Zone C checkpoint on schedule
            if step % CHECKPOINT_EVERY == 0:
                orch.checkpoint_wave(state_wave.cpu(), domain=f"arc3/{env_name}",
                                      sagnac_stress=sagnac_delta)

            print(f"  step {step:3d} | delta {sagnac_delta:.4f} | F {free_energy:.4f} "
                  f"| r {order_param:.3f} | EFE {efe_table[0]['efe']:+.3f} "
                  f"| act {game_action.name} | recall {recall_info['hits']} | {step_ms:.0f}ms")

            prev_raw_wave = raw_wave
            prev_wave = state_wave
            prev_predicted_prior = predicted_prior
            obs = obs_next
            if obs is None or not getattr(obs, "state", None):
                print("  [end] null observation")
                break
            if obs.state.name in ("WIN", "GAME_OVER"):
                # Terminal valence: WIN is the strongest favorable signal —
                # consolidate the final trajectory into the preference store
                # and mark valence for the deferred T1 update.
                if obs.state.name == "WIN":
                    valence = 1.0
                    orch.planner.register_preference(state_wave)
                    if train_ctx is not None and train_ctx["action_wave"] is not None:
                        orch.planner.train_transition_step(
                            train_ctx["state"], train_ctx["action_wave"],
                            state_wave, lr=0.05, valence=1.0,
                        )
                print(f"  [end] {obs.state.name} at step {step}")
                tele.emit({"env": env_name, "terminal": obs.state.name, "step": step,
                           "valence": valence, "action_counts": action_counts})
                break
        else:
            tele.emit({"env": env_name, "terminal": "BUDGET_EXHAUSTED",
                       "step": args.steps, "action_counts": action_counts})
        # NL Level 3 (slow, "dream cycle"): episode-end deep consolidation.
        # Full-buffer EDMD (the low-pass filter over the entire episode
        # extracts structural invariants the mid-frequency window cannot),
        # then persist the solved transition operator itself to Zone C as a
        # recoverable engram — future sessions inherit the dynamics, not
        # just the states (HOPE systems consolidation analog).
        if len(edmd_buffer) >= 8:
            L3_loss = orch.planner.train_transition_batch(
                torch.stack([t[0] for t in edmd_buffer]),
                torch.stack([t[1] for t in edmd_buffer]),
                torch.stack([t[2] for t in edmd_buffer]),
            )
            # Persist the solved operator itself as a recoverable artifact —
            # the Zone C engram store holds fixed [num_blocks, 8] wave
            # payloads, not 25 MB operators, so the field channel goes to a
            # local .pt (restorable via planner.load_field_channel_wave) and
            # Zone C gets a marker engram recording the consolidation event.
            os.makedirs("field_channel_checkpoints", exist_ok=True)
            fc_path = os.path.join(
                "field_channel_checkpoints",
                f"field_channel_{env_name}_{tele.run_id}.pt")
            torch.save({
                "wave": orch.planner.field_channel_wave(),
                "env": env_name, "l3_loss": L3_loss,
                "buffer_size": len(edmd_buffer),
                "scale": SCALE,
            }, fc_path)
            try:
                orch.checkpoint_wave(edmd_buffer[-1][2].cpu(),
                                    domain=f"arc3/{env_name}/field_channel_consolidated",
                                    sagnac_stress=L3_loss)
            except Exception as e:
                print(f"  [edmd-L3] Zone C marker failed ({e}); artifact kept")
            print(f"  [edmd-L3] episode consolidation: {len(edmd_buffer)} triples, "
                  f"batch loss {L3_loss:.4f}, operator -> {fc_path}")
            tele.emit({"env": env_name, "edmd_L3_loss": round(L3_loss, 6),
                       "edmd_L3_buffer": len(edmd_buffer),
                       "field_channel_path": fc_path})
        # Per-env action entropy (fraction of non-ACTION1 steps)
        total = sum(action_counts.values())
        distinct = len(action_counts)
        print(f"  [env summary] actions: {action_counts} | distinct: {distinct}")
        # Capture the scorecard id for this env's game session
        try:
            scid = getattr(game, "scorecard_id", None)
            if scid:
                tele.emit({"env": env_name, "scorecard_id": str(scid)})
                print(f"  [scorecard] {scid}")
        except Exception as e:
            print(f"  [scorecard] capture failed: {e}")

    tele.close()
    if db_logger is not None:
        db_logger.shutdown()

    # Final scorecard extraction: fetch the full scorecard for each env played
    print(f"\n{'='*70}\n  FINAL SCORECARDS\n{'='*70}")
    scorecard_ids = []
    # Re-read the telemetry log to collect per-env scorecard ids
    try:
        with open(log_path) as fp:
            for line in fp:
                rec = json.loads(line)
                if "scorecard_id" in rec:
                    scorecard_ids.append((rec["env"], rec["scorecard_id"]))
    except Exception as e:
        print(f"  scorecard id collection failed: {e}")

    final_scores = {}
    for env_name, scid in scorecard_ids:
        try:
            sc = arcade.get_scorecard(scid)
            d = sc.__dict__ if hasattr(sc, "__dict__") else {"raw": str(sc)}
            final_scores[env_name] = d
            print(f"  {env_name} [{scid[:8]}]: {d}")
        except Exception as e:
            print(f"  {env_name}: fetch failed: {e}")
    if final_scores:
        with open(log_path.replace(".jsonl", "_scorecards.json"), "w") as fp:
            json.dump(final_scores, fp, indent=1, default=str)
        print(f"  scorecards -> {log_path.replace('.jsonl', '_scorecards.json')}")
    print(f"\n[done] telemetry -> {log_path}")


if __name__ == "__main__":
    run()
