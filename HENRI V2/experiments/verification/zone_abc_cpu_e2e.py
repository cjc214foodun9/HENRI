"""Zone A / B / C CPU END-TO-END driver (diagnostic only, NO score claim).

WHY THIS EXISTS
    No receipt currently spans all three zones. The Zone B<->C sync has unit tests,
    the encoder has contract tests, the planner has contract tests -- but nothing
    exercises the real chain ARC grid -> wave -> plan -> snap -> envelope -> Zone C
    store -> read back, on CPU, against the live DEV database.

SCOPE AND HONESTY BOUNDARY
    - CPU only. This is NOT CUDA verification and NOT a capability result.
    - Reduced scale (num_blocks=256) by necessity: the production scale (8192) is a
      CUDA workload. Reduced-scale runs verify SOFTWARE AND WIRING, never model
      capability (repo contract, henri-architecture).
    - Writes go to the DEV Zone C schema only, through the guarded resolver
      (`zone_c_env.resolve_zone_c_dsn` + `assert_zone_c_env`). The dev DB is an
      explicitly disposable sandbox. `ZONE_C_ENV=prod` is never set here.
    - Every engram is tagged with a run_id so the rows are attributable.

PRE-REGISTERED VERDICTS (decided BEFORE the run, see the docstring of main())
    ENGAGED          every stage produced a real artifact and the DB round-tripped
    FALSIFIED_NO_ENGAGEMENT   a stage ran but produced a degenerate/inert value
    BLOCKED_INFRA    a dependency or the database was unavailable

USAGE
    python experiments/verification/zone_abc_cpu_e2e.py --tasks 3 --steps 3 \
        --out experiments/verification/zone_abc_cpu_e2e_observed.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]          # ...\HENRI V2
sys.path.insert(0, str(REPO))

ARC_ROOT = Path(r"C:\Users\chan\henri_data\ARC-AGI\data")


def sha256_tensor(t) -> str:
    import torch
    b = t.detach().cpu().contiguous().to(torch.float32).numpy().tobytes()
    return hashlib.sha256(b).hexdigest()


def _head_sha() -> str:
    """HEAD commit sha for CLASS49 attribution.

    Returns 'untracked' when git is unavailable. That placeholder is REJECTED by
    the guard on purpose: a string that means 'I do not know' is not provenance,
    and silently accepting it is the defect the guard exists to stop.
    """
    import subprocess
    try:
        out = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=15)
        return (out.stdout or "").strip() or "untracked"
    except Exception:
        return "untracked"


def load_arc_tasks(n: int) -> list:
    import numpy as np
    out = []
    for split in ("training", "evaluation"):
        d = ARC_ROOT / split
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.json"))[:n]:
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not obj.get("train"):
                continue
            out.append({"task_id": p.stem, "split": split,
                        "train": obj["train"], "test": obj.get("test", [])})
            if len(out) >= n:
                return out
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=3)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--num-blocks", type=int, default=256)
    ap.add_argument("--out", default=str(REPO / "experiments" / "verification" /
                                        "zone_abc_cpu_e2e_observed.json"))
    ap.add_argument("--dsn", default=None)
    ap.add_argument("--offline", action="store_true",
                    help="use the explicit in-process surrogate instead of the dev DB")
    args = ap.parse_args(argv)

    import numpy as np
    import torch

    nb = args.num_blocks
    dm = nb * 8
    run_id = f"e2e-cpu-{int(time.time())}"
    # CLASS49 Gate 1: the engram write is fail-closed on missing provenance, so
    # this driver MUST carry real lineage. arm_id is declared by the caller (this
    # is one arm, not an A/B); commit_sha comes from HEAD, never invented.
    arm_id = os.environ.get("HENRI_ARM_ID", "cpu-e2e")
    commit_sha = os.environ.get("HENRI_COMMIT_SHA", "").strip() or _head_sha()
    rep = {
        "schema_id": "henri.zone-abc-cpu-e2e.v1",
        "run_id": run_id,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": "cpu",
        "torch": torch.__version__,
        "python": sys.version.split()[0],
        "scale": {"num_blocks": nb, "d_model": dm, "steps": args.steps},
        "arc_root": str(ARC_ROOT),
        "stages": {},
        "errors": [],
        # Pre-registered: CPU + reduced scale => NOT a capability result.
        "evidence_class": "OBSERVED_WIRING_ONLY",
        "score_claim": None,
    }
    verdicts = {}

    # ---------------- Stage 1: Zone A ingress (grid -> wave) ----------------
    try:
        from henri_vision_encoder import HENRIVisionEncoder
        tasks = load_arc_tasks(args.tasks)
        if not tasks:
            rep["errors"].append(f"no ARC tasks under {ARC_ROOT}")
            verdicts["zone_a_ingress"] = "BLOCKED_INFRA"
            raise RuntimeError("no tasks")
        enc = HENRIVisionEncoder(d_model=dm, k_blocks=nb, block_dim=8, device="cpu")
        g0 = np.asarray(tasks[0]["train"][0]["input"], dtype=np.int64)
        w0_flat = enc.encode_grid(g0)
        # ARCHITECTURE BOUNDARY (measured 2026-09-16): HENRIVisionEncoder.encode_grid
        # returns a FLAT unit-norm real wave [d_model] (cos/sin concat), while the
        # planner/Hopfield/envelope boundaries take [num_blocks, 8]. The flatten is
        # the documented storage convention (complex stores use 2D real width; real
        # stores use D), so the reshape is a boundary conversion, not a repair.
        w0 = w0_flat.view(nb, 8)
        rep["stages"]["zone_a_ingress"] = {
            "tasks_loaded": len(tasks),
            "task_ids": [t["task_id"] for t in tasks],
            "grid_shape": list(g0.shape),
            "wave_flat_shape": list(w0_flat.shape),
            "wave_shape": list(w0.shape),
            "wave_norm": round(float(w0_flat.norm()), 6),
            "per_block_norm_min": round(float(w0.norm(p=2, dim=-1).min()), 6),
            "per_block_norm_max": round(float(w0.norm(p=2, dim=-1).max()), 6),
            "wave_sha256": sha256_tensor(w0_flat),
        }
        # engagement: two DIFFERENT grids must give DIFFERENT waves
        g1 = np.asarray(tasks[0]["train"][min(1, len(tasks[0]["train"]) - 1)]["output"],
                        dtype=np.int64)
        w1_flat = enc.encode_grid(g1)
        s = float(enc.compute_sagnac_similarity(w0_flat, w1_flat))
        rep["stages"]["zone_a_ingress"]["distinct_grid_sagnac_similarity"] = round(s, 6)
        rep["stages"]["zone_a_ingress"]["distinct_waves_differ"] = \
            sha256_tensor(w0_flat) != sha256_tensor(w1_flat)
        verdicts["zone_a_ingress"] = (
            "ENGAGED" if rep["stages"]["zone_a_ingress"]["distinct_waves_differ"]
            else "FALSIFIED_NO_ENGAGEMENT")
    except Exception as e:
        rep["errors"].append(f"zone_a_ingress: {type(e).__name__}: {e}")
        verdicts.setdefault("zone_a_ingress", "BLOCKED_INFRA")
        w0 = None

    # ---------------- Stage 2: Zone A planner (EFE over waves) ----------------
    chosen_actions = []
    try:
        from efe_planner import EFEPlanner
        planner = EFEPlanner(num_blocks=nb, d_model=dm, num_actions=8, transition_rank=8)
        # boundary axioms: identity-ish reference waves, [N, nb, 8], unit per block
        ax = torch.nn.functional.normalize(torch.randn(4, nb, 8,
                                                       generator=torch.Generator().manual_seed(7)),
                                           p=2, dim=-1)
        acts = list(range(8))
        # score_actions/select_action take candidate_actions as a list of
        # (action_id, action_wave[num_blocks, 8]) PAIRS. Passing bare ints raises
        # TypeError (measured 2026-09-16, first E2E revision). The waves below are
        # distinct, seeded, per-block unit-norm action carriers -- a WIRING fixture,
        # not learned action embeddings (no action-outcome store is supplied).
        cand = [(i, torch.nn.functional.normalize(
            torch.randn(nb, 8, generator=torch.Generator().manual_seed(500 + i)), p=2, dim=-1))
            for i in acts]
        for k in range(args.steps):
            state = w0 if k == 0 else torch.nn.functional.normalize(
                torch.randn(nb, 8, generator=torch.Generator().manual_seed(100 + k)), p=2, dim=-1)
            # select_action returns (best_action_id, predicted_wave, scores_table, chosen_dict)
            res = planner.select_action(state, cand, ax)
            best_action = res[0] if isinstance(res, tuple) else res
            scores_table = res[2] if isinstance(res, tuple) and len(res) > 2 else None
            chosen_dict = res[3] if isinstance(res, tuple) and len(res) > 3 else None
            idx = int(best_action)
            chosen_actions.append(idx)
            if k == 0 and isinstance(chosen_dict, dict):
                _efe = chosen_dict.get("efe")
        # discriminates candidates? an EFE table whose entries are all equal means
        # the planner cannot rank (the rescaled-everything failure mode).
        efe_vals = []
        if isinstance(scores_table, list):
            for r in scores_table:
                if isinstance(r, dict) and "efe" in r:
                    efe_vals.append(float(r["efe"]))
        rep["stages"]["zone_a_planner"] = {
            "steps": args.steps,
            "operator_family": planner.__class__.__name__,
            "chosen_actions": chosen_actions,
            "distinct_actions": len(set(chosen_actions)),
            "efe_table_len": (len(scores_table) if scores_table is not None else None),
            "efe_min": (round(min(efe_vals), 8) if efe_vals else None),
            "efe_max": (round(max(efe_vals), 8) if efe_vals else None),
            "efe_discriminates": (bool(max(efe_vals) - min(efe_vals) > 1e-9)
                                  if len(efe_vals) > 1 else None),
            "chosen_efe": (round(float(_efe), 8) if _efe is not None else None),
            "error_fraction": round(float(getattr(planner, "loss_ema", float("nan"))), 6),
        }
        # ENGAGED requires the planner to actually RANK candidates, not merely return.
        rep_ok = (rep["stages"]["zone_a_planner"]["efe_discriminates"] is True
                  and len(set(chosen_actions)) >= 1)
        verdicts["zone_a_planner"] = "ENGAGED" if rep_ok else "FALSIFIED_NO_ENGAGEMENT"
    except Exception as e:
        rep["errors"].append(f"zone_a_planner: {type(e).__name__}: {e}")
        rep["stages"]["zone_a_planner"] = {"traceback": traceback.format_exc()[-600:]}
        verdicts["zone_a_planner"] = "BLOCKED_INFRA"

    # ---------------- Stage 3: Zone B cleanup (Hopfield snap) ----------------
    try:
        from hopfield_cleanup import ContinuousHopfieldCleanup
        hc = ContinuousHopfieldCleanup(dim=dm, beta=8.0)
        bank = torch.nn.functional.normalize(
            torch.randn(32, dm, generator=torch.Generator().manual_seed(3)), p=2, dim=-1)
        n_stored = hc.store_engrams(bank)
        # The query MUST be a noisy copy of a STORED engram, or the cleanup has no
        # correct attractor to snap to and "no improvement" is a FIXTURE ARTIFACT,
        # not a mechanism failure (measured 2026-09-16, first E2E revision queried a
        # wave that was never stored -> spurious FALSIFIED_NO_ENGAGEMENT).
        target = bank[0]
        noise = torch.nn.functional.normalize(
            torch.randn(dm, generator=torch.Generator().manual_seed(9)), p=2, dim=-1)
        noisy = torch.nn.functional.normalize(target + 0.25 * noise, p=2, dim=-1)
        out = hc.retrieve(noisy)
        snapped = out[0] if isinstance(out, tuple) else out
        if isinstance(snapped, torch.Tensor) and snapped.dim() > 1:
            snapped = snapped.reshape(-1)
        tf = target.reshape(-1)
        delta_before = float(1.0 - torch.dot(noisy.reshape(-1), tf).clamp(-1, 1))
        delta_after = float(1.0 - torch.dot(snapped.reshape(-1), tf).clamp(-1, 1))
        rep["stages"]["zone_b_cleanup"] = {
            "engrams_stored": n_stored,
            "query_is_stored_engram_plus_noise": True,
            "noise_fraction": 0.25,
            "sagnac_delta_before": round(delta_before, 6),
            "sagnac_delta_after_snap": round(delta_after, 6),
            "snap_improved": delta_after < delta_before,
        }
        verdicts["zone_b_cleanup"] = ("ENGAGED" if rep["stages"]["zone_b_cleanup"]["snap_improved"]
                                      else "FALSIFIED_NO_ENGAGEMENT")
    except Exception as e:
        rep["errors"].append(f"zone_b_cleanup: {type(e).__name__}: {e}")
        rep["stages"]["zone_b_cleanup"] = {"traceback": traceback.format_exc()[-600:]}
        verdicts["zone_b_cleanup"] = "BLOCKED_INFRA"

    # ---------------- Stage 4: Zone B -> C envelope + store round-trip ----------------
    cache_obj = None
    try:
        from zone_bc_engram_sync import make_envelope, DecoupledEngramSync
        from zone_c_segment_cache import SegmentCache

        if args.offline:
            cache = SegmentCache.connect(dsn="offline://surrogate", num_blocks=nb)
            target = "offline://surrogate"
        else:
            # NEVER hardcode a DSN here. Resolve through the same guarded
            # resolver production uses: it defaults to dev and requires
            # explicit configuration for prod. A masked literal is not a live
            # credential, but it is still a hardcoded connection string, which
            # this project prohibits in new code.
            from zone_c_env import resolve_zone_c_dsn
            os.environ.setdefault("ZONE_C_ENV", "dev")
            dsn = args.dsn or resolve_zone_c_dsn()
            cache = SegmentCache.connect(dsn=dsn, num_blocks=nb)
            target = "dev-timescaledb"
        # KEEP the connected cache for the read-back stage. The in-process surrogate is
        # NOT persistent: a second connect() returns an EMPTY store, so re-connecting in
        # stage 5 would report 0 hits and a spurious FALSIFIED_NO_ENGAGEMENT (measured
        # 2026-09-16, first E2E revision).
        cache_obj = cache

        wave = w0 if w0 is not None else torch.randn(nb, 8,
                                                     generator=torch.Generator().manual_seed(1))
        env = make_envelope(wave, domain=f"{run_id}/zoneA", sagnac_stress=0.123,
                            num_blocks=nb, block_dim=8,
                            run_id=run_id, arm_id=arm_id, commit_sha=commit_sha)
        sync = DecoupledEngramSync(cache.store, capacity=16, enabled=True)
        pub = sync.publish(env)
        drained = sync.drain()
        rt = env.verify()
        rep["stages"]["zone_bc_sync"] = {
            "store_backend": target,
            "envelope_digest": env.digest[:16],
            "publish_status": pub.get("status"),
            "drain": {k: v for k, v in drained.items() if k != "ids"},
            "payload_verifies": bool(rt),
            "committed": sync.committed,
            "bytes_persisted": sync.bytes_persisted,
        }
        verdicts["zone_bc_sync"] = ("ENGAGED" if (pub.get("status") == "QUEUED"
                                                  and sync.committed >= 1 and rt)
                                    else "FALSIFIED_NO_ENGAGEMENT")
    except Exception as e:
        rep["errors"].append(f"zone_bc_sync: {type(e).__name__}: {e}")
        rep["stages"]["zone_bc_sync"] = {"traceback": traceback.format_exc()[-900:]}
        verdicts["zone_bc_sync"] = "BLOCKED_INFRA"

    # ---------------- Stage 5: Zone C read back ----------------
    try:
        from zone_c_segment_cache import SegmentCache
        # REUSE the stage-4 connection. A fresh in-process surrogate is empty by
        # construction, so re-connecting would fabricate a read-back failure.
        if cache_obj is not None:
            cache2 = cache_obj
        elif args.offline:
            cache2 = SegmentCache.connect(dsn="offline://surrogate", num_blocks=nb)
        else:
            from zone_c_env import resolve_zone_c_dsn
            os.environ.setdefault("ZONE_C_ENV", "dev")
            dsn = args.dsn or resolve_zone_c_dsn()
            cache2 = SegmentCache.connect(dsn=dsn, num_blocks=nb)
        hits = cache2.retrieve(w0 if w0 is not None else
                              torch.randn(nb, 8, generator=torch.Generator().manual_seed(2)))
        # retrieve() returns a DICT: {conditioning_wave, gates, hits, top_similarity, ...}
        n_hits = int(hits.get("hits", 0)) if isinstance(hits, dict) else (len(hits) if hits else 0)
        rep["stages"]["zone_c_readback"] = {
            "hits": n_hits,
            "store_count": int(cache2.store.count()) if hasattr(cache2.store, "count") else None,
            "top_similarity": (hits.get("top_similarity") if isinstance(hits, dict) else None),
            "conditioning_wave_present": bool(
                isinstance(hits, dict) and hits.get("conditioning_wave") is not None),
            "oldest_age_hours": (hits.get("oldest_age_hours") if isinstance(hits, dict) else None),
        }
        verdicts["zone_c_readback"] = "ENGAGED" if n_hits >= 1 else "FALSIFIED_NO_ENGAGEMENT"
    except Exception as e:
        rep["errors"].append(f"zone_c_readback: {type(e).__name__}: {e}")
        rep["stages"]["zone_c_readback"] = {"traceback": traceback.format_exc()[-600:]}
        verdicts["zone_c_readback"] = "BLOCKED_INFRA"

    # ---------------- verdict ----------------
    rep["verdicts"] = verdicts
    if "BLOCKED_INFRA" in verdicts.values():
        rep["overall"] = "BLOCKED_INFRA"
    elif "FALSIFIED_NO_ENGAGEMENT" in verdicts.values():
        rep["overall"] = "FALSIFIED_NO_ENGAGEMENT"
    elif verdicts and all(v == "ENGAGED" for v in verdicts.values()):
        rep["overall"] = "ENGAGED_WIRING_ONLY"
    else:
        rep["overall"] = "PARTIAL"

    Path(args.out).write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("overall", "verdicts", "errors")}, indent=1,
                     default=str))
    print(f"receipt: {args.out}")
    return 0 if rep["overall"] in ("ENGAGED_WIRING_ONLY", "PARTIAL") else 1


if __name__ == "__main__":
    raise SystemExit(main())
