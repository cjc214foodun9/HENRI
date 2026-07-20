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

RELAX_STEPS = 8          # swarm relaxation iterations per environment step
RECALL_EVERY = 5         # recall Zone C conditioning every N steps
CHECKPOINT_EVERY = 10    # persist engram every N steps


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
    orch = HenriSwarmOrchestrator(**SCALE).to(DEVICE)
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
        for step in range(args.steps):
            t0 = time.perf_counter()
            grid = obs.frame[0].tolist()
            state_wave = tokenizer.encode_spatial_grid(grid).squeeze(0).to(DEVICE)

            # Boundary axiom = topological transition from previous state
            if prev_wave is not None:
                boundary = state_wave - prev_wave
                boundary = boundary / (torch.norm(boundary, p=2, dim=-1, keepdim=True) + 1e-9)
            else:
                boundary = state_wave.clone()

            # Zone C recall (conditioning) on schedule
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

            # Swarm relaxation with SGLD creep
            sagnac_delta = None
            for _ in range(RELAX_STEPS):
                sagnac_delta, _, _ = orch.process_active_reasoning_step(
                    state_wave, boundary,
                    t_shock_max=torch.tensor(0.5, device=DEVICE),
                )

            # Latent metrics
            coherence = orch.sagnac_coherence(state_wave, boundary).item()
            free_energy = orch.compute_free_energy(state_wave, boundary).item()
            order_param = kuramoto_order_parameter(orch.syncytium.expert_phases)
            plasticity = {
                k: round(v, 6)
                for k, v in orch.syncytium.creep_ctrl_A.plasticity_stats().items()
            }

            # EFE action selection
            boundary_batch = boundary.unsqueeze(0)
            action, predicted_wave, efe_table = orch.plan_action(
                state_wave, boundary_batch, top_k=4
            )
            hop_conf = efe_table[0]["efe"]  # best EFE as confidence proxy

            # Environment step
            game_action = action if isinstance(action, GameAction) else GameAction.ACTION1
            obs_next = game.step(game_action)
            step_ms = (time.perf_counter() - t0) * 1000

            # Telemetry emit (dense latent record)
            tele.emit({
                "env": env_name, "step": step,
                "sagnac_delta": round(sagnac_delta, 6),
                "sagnac_coherence": round(coherence, 6),
                "free_energy": round(free_energy, 6),
                "kuramoto_r": round(order_param, 6),
                "plasticity": plasticity,
                "efe_best": round(efe_table[0]["efe"], 6),
                "efe_spread": round(efe_table[-1]["efe"] - efe_table[0]["efe"], 6),
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

            prev_wave = state_wave
            obs = obs_next
            if obs is None or not getattr(obs, "state", None):
                print("  [end] null observation")
                break
            if obs.state.name in ("WIN", "GAME_OVER"):
                print(f"  [end] {obs.state.name} at step {step}")
                tele.emit({"env": env_name, "terminal": obs.state.name, "step": step})
                break

    tele.close()
    if db_logger is not None:
        db_logger.shutdown()
    print(f"\n[done] telemetry -> {log_path}")


if __name__ == "__main__":
    run()
