"""Phase 8 compute-envelope probe: batched navigation swarm on RTX 5090.

COMPUTE-ONLY. No environment instance, no env stepping, no scorecard,
no SANS buffer. Reuses the exact production modules read-only:
HENRIVisionEncoder, HenriSwarmOrchestrator / EFEPlanner scoring,
ActionEgressVocabulary.

Gate: HENRI_ARC_BATCHED_NAV_SWARM=1 required; otherwise emits a typed
FEATURE_DISABLED result WITHOUT allocating production tensors.

Labels:
  bandwidth -> LOWER_BOUND_LOGICAL_BANDWIDTH (one logical state read per
  particle per step; NOT an achieved-DRAM-bandwidth measurement).
  ESS / action entropy -> descriptors only, never task-capability evidence.

Schema: henri.phase8-compute-probe.v1
"""
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

_FEATURE_FLAG = "HENRI_ARC_BATCHED_NAV_SWARM"
_SCHEMA_ID = "henri.phase8-compute-probe.v1"
_DEFAULT_BATCHES = [1, 4, 64, 256, 512, 1024]
_LOGICAL_BYTES_PER_PARTICLE = 65536 * 4  # D=65536 x float32
_PEAK_BW_TBPS = 2.1  # RTX 5090 nominal; label is LOWER_BOUND_LOGICAL_BANDWIDTH
_REPO_ROOT = Path(__file__).resolve().parents[2]  # HENRI V2/


def _git_sha() -> str:
    try:
        import subprocess
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=str(_REPO_ROOT), timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _fail_closed(reason: str, **extra) -> dict:
    return {
        "schema_id": _SCHEMA_ID,
        "status": "FAIL_CLOSED",
        "reason": reason,
        **extra,
    }


@torch.no_grad()
def _score_particle(planner, state_wave, candidates, boundary_batch):
    """Production-faithful per-particle scoring (loop over candidates,
    exactly like EFEPlanner.score_actions; returns ranked results)."""
    return planner.score_actions(
        state_wave, candidates, boundary_batch, goal_wave=None, grid_dist=None
    )


def _ess_from_efes(efes):
    """Descriptor: effective sample size from normalized inverse-EFE weights."""
    w = torch.softmax(-torch.tensor(efes, dtype=torch.float32), dim=0)
    return float(1.0 / (w.pow(2).sum().clamp(min=1e-12)))


def _try_vmap_transition(planner, state_batch, action_wave, boundary_batch):
    """Attempt vectorized EFE-core (transition+pragmatic+epistemic+penalty)
    via torch.vmap. Returns (status, reason, max_abs_diff_vs_loop or None).
    Python bookkeeping in score_actions is NOT vmap-able; we vmap only the
    pure-tensor core. Failure -> typed skip, never silent fallback."""
    try:
        from torch import vmap

        @torch.no_grad()
        def core(state):
            predicted = planner.transition(state, action_wave)
            pragmatic = planner.pragmatic_value(predicted, boundary_batch, None)
            epistemic = planner.epistemic_value(
                predicted, state_wave=state, grid_dist=None)
            penalty = planner.constraint_penalty(predicted)
            penalty = 0.0 if penalty is None else penalty
            efe = (planner.pragmatic_weight * pragmatic
                   - planner.epistemic_weight * epistemic
                   + planner._constraint_lambda() * penalty)
            return efe

        with torch.no_grad():
            batched = vmap(core)(state_batch)
            # loop reference on the same inputs
            loop = torch.stack([
                core(state_batch[i]) for i in range(state_batch.shape[0])
            ])
        diff = float((batched - loop).abs().max().item())
        return "OK", "vmap", diff
    except Exception as exc:  # noqa: BLE001 - typed skip reason required
        return "SKIPPED", f"{type(exc).__name__}: {exc}", None


def _try_cuda_graph_transition(planner, state_batch, action_wave):
    """Attempt CUDA-graph capture of the static-batch transition kernel.
    Returns (status, reason, replay_ms_mean, agreement_max_abs_diff)."""
    if not torch.cuda.is_available():
        return "SKIPPED_NO_CUDA", "cuda unavailable", None, None
    try:
        wrapper = _StaticTransition(planner, action_wave)
        wrapper.to("cuda")
        # warmup on static buffers
        static_in = state_batch.contiguous()
        with torch.no_grad():
            for _ in range(3):
                out = wrapper(static_in)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out_static = wrapper(static_in)
            torch.cuda.synchronize()
            # timed replay
            n = 20
            start = time.perf_counter()
            for _ in range(n):
                graph.replay()
            torch.cuda.synchronize()
            replay_ms = (time.perf_counter() - start) / n * 1000.0
            # agreement vs eager on a fresh copy
            eager = wrapper(static_in)
        diff = float((eager - out_static).abs().max().item())
        return "OK", "captured", replay_ms, diff
    except Exception as exc:  # noqa: BLE001
        return "SKIPPED", f"{type(exc).__name__}: {exc}", None, None


class _StaticTransition(torch.nn.Module):
    """Static-shape wrapper for CUDA-graph capture of the transition kernel."""

    def __init__(self, planner, action_wave):
        super().__init__()
        self.planner = planner
        self.register_buffer("action_wave", action_wave.detach().clone())

    def forward(self, state_batch):
        return self.planner.transition(state_batch, self.action_wave)


def run_probe(scale: str = "prod", batches=None, iterations: int = 10,
              warmup: int = 3, seed: int = 20260814) -> dict:
    """Execute the compute-envelope probe. Returns the v1 schema dict.

    scale="prod": num_experts=1024, d_model=65536, num_blocks=8192.
    scale="reduced": num_experts=64, d_model=512, num_blocks=64 (CPU tests).
    """
    torch.manual_seed(seed)
    batches = batches or _DEFAULT_BATCHES
    if os.environ.get(_FEATURE_FLAG, "0") != "1":
        return {
            "schema_id": _SCHEMA_ID,
            "status": "FEATURE_DISABLED",
            "reason": f"{_FEATURE_FLAG} != 1; probe did not allocate",
            "diagnostic_only": True,
            "score_eligible": False,
        }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if scale == "reduced":
        SCALE = dict(num_experts=64, d_model=512, r_rank=8, num_blocks=64)
    else:
        SCALE = dict(num_experts=1024, d_model=65536, r_rank=16, num_blocks=8192)
    d_model = SCALE["d_model"]
    num_blocks = SCALE["num_blocks"]

    from arcengine import GameAction
    from henri_vision_encoder import HENRIVisionEncoder
    from arc_spatial_basis import resolve_spatial_basis
    from darwinian_phase_swarm import (HenriSwarmOrchestrator,
                                       generate_colored_langevin_noise)
    from arc_egress_contract import ActionEgressVocabulary

    spatial_basis, bg_mask = resolve_spatial_basis()
    tokenizer = HENRIVisionEncoder(
        d_model=d_model, k_blocks=num_blocks, device=device,
        spatial_basis_kind=spatial_basis, bg_mask=bg_mask,
    ).to(device)
    orch = HenriSwarmOrchestrator(
        action_enum_class=GameAction,
        constraint_weight_max=5.0,
        constraint_reject_thresh=0.38,
        beta_pragmatic=1.0,
        lambda_goal=0.0,
        learnable_actions=False,
        chimera_mode=False,
        chimera_alpha=1.4,
        chimera_explorer_fraction=0.25,
        happy_tensor_cut=False,
        external_outcome_efe=False,
        external_eig_weight=0.25,
        external_task_weight=1.0,
        task_weighted_eig=False,
        task_eig_gamma=4.0,
        **SCALE,
    ).to(device)
    orch.eval()  # frozen: no dropout, no training mutation

    # Deterministic synthetic grid (compute-only; not an ARC task grid).
    g = torch.Generator().manual_seed(seed)
    grid = torch.randint(0, 10, (10, 10), generator=g).tolist()
    state_wave = tokenizer.encode_spatial_grid(grid).squeeze(0).to(device)
    assert tuple(state_wave.shape) == (num_blocks, 8), (
        f"UWE shape {tuple(state_wave.shape)} != ({num_blocks}, 8)")

    # Legal-action mask surrogate: first 6 GameAction members, deterministic.
    allowed = list(GameAction)[:6]
    vocab = ActionEgressVocabulary(GameAction, allowed)
    candidates = orch.candidate_action_waves(top_k=4, allowed_actions=allowed)
    assert len(candidates) == 4, "expected 4 candidates under allowed mask"
    assert all(w.shape == (num_blocks, 8) for _, w in candidates)

    # Boundary batch: deterministic unit-norm wave [1, num_blocks, 8].
    rng = torch.Generator(device="cpu").manual_seed(seed)
    boundary = torch.randn((num_blocks, 8), generator=rng).to(device)
    boundary = torch.nn.functional.normalize(boundary, p=2, dim=-1)
    boundary_batch = boundary.unsqueeze(0)

    # B=1 identity gate vs production plan_action.
    with torch.no_grad():
        prod_action, prod_pred, prod_table, prod_chosen = orch.plan_action(
            state_wave, boundary_batch, top_k=4, return_chosen=True,
            goal_wave=None, grid_dist=None, allowed_actions=allowed,
        )
        loop_action, _loop_pred, _loop_table, _loop_chosen = (
            orch.planner.select_action(
                state_wave, candidates, boundary_batch, goal_wave=None,
                grid_dist=None)
        )
    identity_match = bool(prod_action == loop_action)
    digest = hashlib.sha256()
    for r in _loop_table:
        digest.update(r["action"].name.encode())
        digest.update(repr(r["efe"]).encode())
    b1_digest = digest.hexdigest()[:32]

    batch_rows = []
    action_wave_0 = candidates[0][1]
    for B in batches:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        try:
            if B == 1:
                # Identity configuration: un-noised production state so the
                # B=1 batch row IS the production-equivalent single particle.
                states = torch.nn.functional.normalize(
                    state_wave.unsqueeze(0), p=2, dim=-1)
            else:
                noise = generate_colored_langevin_noise(
                    (B, num_blocks, 8), alpha=1.0, device=device)
                states = torch.nn.functional.normalize(
                    state_wave.unsqueeze(0) + noise, p=2, dim=-1)
            states = states.contiguous()
            efes = []
            actions = []
            # warmup: score every particle once
            for b in range(min(B, 4)):
                _score_particle(orch.planner, states[b], candidates,
                                boundary_batch)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for i in range(iterations):
                for b in range(B):
                    r = _score_particle(orch.planner, states[b], candidates,
                                        boundary_batch)
                    efes.append(r[0]["efe"])
                    actions.append(r[0]["action"])
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            total_scored = B * iterations
            particles_per_s = total_scored / max(dt, 1e-9)
            lat_ms = dt / max(total_scored, 1) * 1000.0
            eff_b = min(B, iterations)
            ess = _ess_from_efes(efes[:eff_b])
            distinct_actions = len(set(actions[:eff_b]))
            logical_bytes = _LOGICAL_BYTES_PER_PARTICLE
            lb_bw_gbps = (
                logical_bytes * B * iterations / max(dt, 1e-9) / 1e9)
            finite_ok = all(math.isfinite(e) for e in efes)
            vram = {}
            if torch.cuda.is_available():
                vram = {
                    "vram_allocated_mib": round(
                        torch.cuda.memory_allocated() / 2 ** 20, 2),
                    "vram_reserved_mib": round(
                        torch.cuda.memory_reserved() / 2 ** 20, 2),
                    "vram_peak_mib": round(
                        torch.cuda.max_memory_allocated() / 2 ** 20, 2),
                }
            batch_rows.append({
                "B": B,
                "iterations": iterations,
                "total_scored_particles": total_scored,
                "particles_per_s": round(particles_per_s, 2),
                "latency_mean_ms_per_particle": round(lat_ms, 4),
                "logical_bytes_per_particle": logical_bytes,
                "lower_bound_logical_bandwidth_gbps": round(lb_bw_gbps, 3),
                "bandwidth_label": "LOWER_BOUND_LOGICAL_BANDWIDTH",
                "ess_descriptor": round(ess, 4),
                "distinct_actions_descriptor": distinct_actions,
                "finite_ok": finite_ok,
                **vram,
            })
            if not finite_ok:
                return _fail_closed("non-finite EFE observed", batches=batches,
                                    batch=B, rows=batch_rows)
        except torch.cuda.OutOfMemoryError as exc:
            return _fail_closed(f"CUDA OOM at B={B}: {exc}",
                                batches=batches, rows=batch_rows)
        except Exception as exc:  # noqa: BLE001
            return _fail_closed(f"{type(exc).__name__} at B={B}: {exc}",
                                batches=batches, rows=batch_rows)

    # Vectorized EFE-core attempt (descriptor of vectorization ceiling).
    vec_status, vec_reason, vec_diff = _try_vmap_transition(
        orch.planner, states[:4], action_wave_0, boundary_batch)

    # CUDA-graph capture attempt on the static transition kernel.
    graph_status, graph_reason, graph_ms, graph_diff = _try_cuda_graph_transition(
        orch.planner, states[:64], action_wave_0)

    result = {
        "schema_id": _SCHEMA_ID,
        "status": "COMPLETE",
        "feature_gate": _FEATURE_FLAG,
        "scale": scale,
        "device": device,
        "gpu_name": (torch.cuda.get_device_name(0)
                     if torch.cuda.is_available() else None),
        "candidate_sha": _git_sha(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "shapes": {
            "state_wave": [num_blocks, 8],
            "candidates": [4, num_blocks, 8],
            "boundary_batch": [1, num_blocks, 8],
            "dtype": str(state_wave.dtype),
            "device": device,
        },
        "b1_identity": {
            "production_action_match": identity_match,
            "digest_sha256_32": b1_digest,
        },
        "vectorization": {
            "status": vec_status,
            "reason": vec_reason,
            "agreement_max_abs_diff": vec_diff,
        },
        "cuda_graph": {
            "status": graph_status,
            "reason": graph_reason,
            "replay_mean_ms": graph_ms,
            "eager_vs_graph_max_abs_diff": graph_diff,
        },
        "batches": batch_rows,
        "egress": {
            "diagnostic_only": True,
            "score_eligible": False,
            "transducer_loaded": False,
        },
        "checkpoint": {
            "policy": "not_used",
            "load_status": "SKIPPED_POLICY_DISABLED",
        },
        "git": {"candidate_sha": _git_sha()},
    }
    result["raw_log_sha256"] = hashlib.sha256(
        json.dumps(result, sort_keys=True).encode()).hexdigest()
    return result


def main() -> int:
    scale = os.environ.get("HENRI_PROBE_SCALE", "prod")
    batch_env = os.environ.get("HENRI_PROBE_BATCHES", "")
    batches = None
    if batch_env:
        try:
            batches = [int(x) for x in batch_env.split(",") if x.strip()]
        except ValueError:
            print(json.dumps(_fail_closed("invalid HENRI_PROBE_BATCHES")))
            return 2
    iters = int(os.environ.get("HENRI_PROBE_ITERATIONS", "10"))
    result = run_probe(scale=scale, batches=batches, iterations=iters)
    out_path = os.environ.get("HENRI_PROBE_OUT", "")
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if out_path:
        Path(out_path).write_text(payload, encoding="utf-8")
        print(f"[probe] wrote {out_path}")
    return 0 if result.get("status") in ("COMPLETE", "FEATURE_DISABLED") else 1


if __name__ == "__main__":
    sys.exit(main())
