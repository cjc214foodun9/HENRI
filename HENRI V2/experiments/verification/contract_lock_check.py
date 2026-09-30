"""Stage-1 contract LOCK gate -- code, not prose.

PURPOSE
  The spec's clause (c) ("step latency <= 15 us") is unreachable for this module
  composition: the measured floor with the per-cell loop entirely removed is
  78.2 us at 4x4 (5.2x over), of which 33.4 us is GPU kernel. It also contains an
  internal contradiction (Tier 1 labelled "20 kHz" -> 50 us vs a 15 us contract,
  3.3x apart) and leaves "step latency" undefined for three different steps.

  Rather than lock a baseplate that fails its own contract, or fabricate a pass,
  the clause is AMENDED from measurement (user decision, Option A; see
  docs/stage1-contract-lock.md) and this file is the enforcement.

WHAT IT CHECKS
  (a) passage  : the REAL smoke entrypoint is executed as a subprocess and its
                 own stdout must contain the marker; checkpoint telemetry and
                 the fail-closed guard are read from the production consumer.
  (b) unitary  : ||Psi||_2 = 1.0 and |norm-1| < 1e-5.
  (c) latency  : c1 perceive <= 600 us, c2 act <= 4300 us, c3 encode <= 325 us,
                 at D=65536. Thresholds carry ~1.25-1.29x headroom over the
                 measured values (465.6 / 3445.5 / 251.0) and each measurement
                 records its run-to-run standard deviation so the gate can be
                 shown non-flapping. Means are of batch means, not of individual
                 iterations.
  (p) provenance: a receipt must carry gpu/D/timestamp/smoke-output-hash, so a
                 stale or foreign receipt cannot silently produce LOCKED.

GATE INTEGRITY RULE (learned the hard way)
  A gate must not be able to manufacture the evidence it checks. A previous
  revision of this file contained
      out["smoke_marker"] = "UNIFIED_VLA_CUDA_SMOKE_PASS"  # implied by reaching here
  which made every LOCKED verdict partially self-issued. The marker is now
  PARSED FROM THE CHILD PROCESS OUTPUT. Do not reintroduce a literal.

MODES
  --receipt PATH  validate a saved receipt JSON (no GPU needed)
  --live          re-measure on the GPU, then validate
  default         --live

Exit codes
  0  LOCKED          (only reachable from a real measurement with provenance)
  1  UNLOCKED        (a check failed / evidence missing -> blocks promotion)
  2  FIXTURE_CHECKED (synthetic fixture: proves gate LOGIC, never the contract)

No score claim. Invariants and latency only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time

# ----------------------------------------------------------------------------
# THE AMENDED CONTRACT. Every number here is a measured value with headroom;
# the measured value it derives from is in the comment. D is explicit.
# ----------------------------------------------------------------------------
CONTRACT = {
    "D": 65536,
    "gpu_required": "RTX 5090 (sm_120)",
    "a_passage": {
        "smoke_script": "experiments/verification/smoke_unified_vla_cuda.py",
        "marker": "UNIFIED_VLA_CUDA_SMOKE_PASS",
        "checkpoint_status": "LOADED",
        "trained_decoder_active": True,
    },
    "b_unitary": {
        "target": 1.0,
        "tol": 1e-5,
    },
    # amended clause (c): measured value -> threshold with headroom
    "c_latency_us": {
        "perceive_1step": {"measured": 465.6, "threshold": 600.0},
        "act_step":       {"measured": 3445.5, "threshold": 4300.0},
        "encode_1step":   {"measured": 251.0, "threshold": 325.0},
    },
    "provenance": {
        "gpu_must_contain": "5090",
        "d_must_equal": 65536,
        "require_fields": ["gpu", "D", "measured_utc", "smoke_exit_code",
                           "smoke_stdout_sha256", "smoke_marker_seen_in_output"],
        # A receipt must not be arbitrarily stale: otherwise one old LOCKED
        # silently stands in for a current measurement forever.
        "max_age_hours": 720,
    },
    "note": (
        "Clause (c) is amended from measurement, not chosen to pass. The "
        "unreachable 15 us clause and the spec's self-contradictory 50 us tier "
        "label are both recorded in docs/stage1-contract-lock.md. Locked cadence "
        "is ~2.1 kHz (465.6 us), a Tier-1 prototype rate, NOT the spec's "
        "photonic target."
    ),
}

FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(f"{label}: {detail}")


# ----------------------------------------------------------------------------
def find_repo() -> str:
    for cand in ("/root/henri/HENRI V2", "/root/henri"):
        if os.path.isdir(os.path.join(cand, "experiments", "verification")):
            return cand
    p = os.path.dirname(os.path.abspath(__file__))
    while p != os.path.dirname(p):
        if os.path.isdir(os.path.join(p, "experiments", "verification")):
            return p
        p = os.path.dirname(p)
    return os.getcwd()


def run_smoke(repo: str, timeout_s: int = 900) -> dict:
    """Run the REAL smoke entrypoint; parse clause-(a) evidence from ITS output.

    The gate must not manufacture the evidence it checks. The marker below is
    searched for in the child's stdout -- it is never assigned as a literal.
    """
    script = os.path.join(repo, CONTRACT["a_passage"]["smoke_script"])
    if not os.path.isfile(script):
        return {"smoke_exit_code": None, "smoke_error": f"missing {script}"}

    env = dict(os.environ)
    env["PYTHONPATH"] = repo + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("HENRI_UNIFIED_VLA", "1")
    t0 = time.perf_counter()
    try:
        proc = subprocess.run([sys.executable, script], cwd=repo, env=env,
                              capture_output=True, text=True, timeout=timeout_s)
        rc, so, se = proc.returncode, proc.stdout or "", proc.stderr or ""
    except subprocess.TimeoutExpired as exc:
        rc = -1
        so = exc.stdout if isinstance(exc.stdout, str) else ""
        se = f"TIMEOUT after {timeout_s}s"
    except Exception as exc:  # noqa: BLE001
        rc, so, se = -1, "", f"LAUNCH_FAILED: {type(exc).__name__}: {exc}"

    text = so + "\n" + se
    marker = CONTRACT["a_passage"]["marker"]
    return {
        "smoke_exit_code": rc,
        "smoke_seconds": round(time.perf_counter() - t0, 2),
        "smoke_stdout_sha256": hashlib.sha256(
            text.encode("utf-8", "replace")).hexdigest(),
        "smoke_output_bytes": len(text),
        "smoke_marker_seen_in_output": marker in text,
        "smoke_marker_count": text.count(marker),
        "smoke_norm_lines": re.findall(r"perceive norm[^\n]*", text)[:3],
        "smoke_tail": text.strip().splitlines()[-6:],
    }


# ----------------------------------------------------------------------------
def measure_live(repo: str) -> dict:
    """Re-measure the locked config on the target. Nothing here is pre-filled."""
    os.environ.setdefault("HENRI_UNIFIED_VLA", "1")
    sys.path.insert(0, repo)
    import torch
    import torch.nn.functional as F

    from henri_vision_encoder import HENRIVisionEncoder

    dev = "cuda"
    assert torch.cuda.is_available(), "CUDA required -- --live must run on the GPU"
    D = CONTRACT["D"]
    cfg = dict(d_model=D, k_blocks=8192, device=dev,
               spatial_basis_kind="incommensurate", bg_mask=True,
               fused_superpose=True, parity_scipy=True)
    G4 = [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]

    def bench(fn, reps, warm=25, batches=7):
        """Mean and std of BATCH MEANS (run-to-run variance, not iteration noise)."""
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        means = []
        for _ in range(batches):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(reps):
                fn()
            torch.cuda.synchronize()
            means.append((time.perf_counter() - t0) / reps * 1e6)
        m = sum(means) / len(means)
        var = sum((x - m) ** 2 for x in means) / max(len(means) - 1, 1)
        return m, var ** 0.5

    out: dict = {
        "measured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "D": D,
        "locked_config": cfg,
    }

    tok = HENRIVisionEncoder(**cfg)
    tok.encode_grid(G4)  # warm the locked path
    out["encode_1step_us"], out["encode_1step_std_us"] = bench(
        lambda: tok.encode_grid(G4), 300)

    from darwinian_phase_swarm import HenriSwarmOrchestrator
    from henri_action_gate import TypedActionGate
    from henri_decoder import HENRIUnifiedEgressTransducer
    from henri_unified_vla import get_unified_vla
    from arcengine import GameAction

    orch = HenriSwarmOrchestrator(action_enum_class=GameAction, d_model=D,
                                  num_blocks=8192, num_experts=1024,
                                  r_rank=16).to(dev)
    gate = TypedActionGate(orch.decoder, seed=0)
    egress = HENRIUnifiedEgressTransducer(d_model=D, hidden_dim=2048,
                                          vocab_size=32000, device=dev,
                                          checkpoint_policy="required")
    boundary = F.normalize(torch.randn(1, 8192, 8, device=dev), p=2, dim=-1)
    vla = get_unified_vla(tokenizer=tok, orchestrator=orch, action_gate=gate,
                          egress_transducer=egress, boundary_axioms=boundary,
                          device=dev)

    wave, digest = vla.perceive(G4)
    out["perceive_norm"] = float(wave.norm(p=2).item())
    out["perceive_norm_err"] = abs(out["perceive_norm"] - 1.0)
    out["perceive_finite"] = bool(torch.isfinite(wave).all().item())
    out["perceive_digest"] = digest[:12]
    out["perceive_shape"] = list(wave.shape)
    out["perceive_1step_us"], out["perceive_1step_std_us"] = bench(
        lambda: vla.perceive(G4), 200, warm=20)

    allowed = list(GameAction)
    # step=0 is passed explicitly on every call: the planner's step counter does
    # not advance across reps, so the mean is over comparable work, not over a
    # monotonically evolving planner state.
    out["act_step_us"], out["act_step_std_us"] = bench(
        lambda: vla.act(wave, G4, allowed, step=0), 40, warm=8)

    # INTEGRITY: the act figure above is a MEAN over 40 live planner calls. If
    # vla.act mutates swarm/planner state, that mean is taken over a drifting
    # process and is not a per-step latency. Two fresh calls with identical
    # inputs must agree, or c2 cannot be asserted. Recorded, not assumed.
    try:
        a1 = vla.act(wave, G4, allowed, step=0)
        a2 = vla.act(wave, G4, allowed, step=0)
        r1, r2 = repr(a1)[:60], repr(a2)[:60]
        out["act_probe_1"], out["act_probe_2"] = r1, r2
        out["act_return_type"] = type(a1).__name__
        out["act_deterministic"] = (r1 == r2)
    except Exception as exc:  # noqa: BLE001
        out["act_deterministic"] = None
        out["act_probe_exc"] = f"{type(exc).__name__}: {exc}"

    tele = egress.checkpoint_telemetry()
    out["checkpoint_status"] = tele.get("checkpoint_load_status")
    out["trained_decoder_active"] = bool(tele.get("trained_decoder_active"))

    # (a): the fail-closed guard must still refuse a generic marker
    from henri_decoder import DecoderEgressFailClosedError
    try:
        vla.egress_decode(wave, "generic marker prompt")
        out["fail_closed_generic"] = False
    except DecoderEgressFailClosedError:
        out["fail_closed_generic"] = True
    except Exception as exc:  # noqa: BLE001
        out["fail_closed_generic"] = False
        out["fail_closed_generic_exc"] = type(exc).__name__

    # (a): run the REAL smoke entrypoint as a child process and read its stdout.
    # NOTE: no marker literal is ever assigned here. See module docstring.
    print("  running the real smoke entrypoint as a subprocess ...")
    out.update(run_smoke(repo))
    return out


# ----------------------------------------------------------------------------
def validate_provenance(rec: dict) -> None:
    p = CONTRACT["provenance"]
    for field in p["require_fields"]:
        if rec.get(field) in (None, ""):
            FAILURES.append(f"provenance: missing field '{field}'")
    gpu = str(rec.get("gpu", ""))
    if p["gpu_must_contain"].lower() not in gpu.lower():
        FAILURES.append(
            f"provenance: gpu '{gpu}' lacks '{p['gpu_must_contain']}'")
    if rec.get("D") != p["d_must_equal"]:
        FAILURES.append(
            f"provenance: D={rec.get('D')} != {p['d_must_equal']} "
            "(D is not interchangeable -- measured 1.02-1.14x)")

    # FRESHNESS. Without this, a single old receipt prints LOCKED forever and the
    # lock stops being a statement about a measurement.
    import calendar
    utc = str(rec.get("measured_utc") or "")
    try:
        age_h = (time.time() - calendar.timegm(
            time.strptime(utc, "%Y-%m-%dT%H:%M:%SZ"))) / 3600.0
        limit = p["max_age_hours"]
        if age_h > limit:
            FAILURES.append(
                f"provenance: receipt is {age_h:.1f} h old (> {limit} h) -- "
                "re-run --live on the target, do not re-use this receipt")
        elif age_h < -1.0:
            FAILURES.append(
                f"provenance: measured_utc is {age_h:.1f} h in the FUTURE")
    except Exception:  # noqa: BLE001
        FAILURES.append(f"provenance: measured_utc unparseable: {utc!r}")


def validate(rec: dict) -> None:
    c = CONTRACT
    print("=" * 80)
    print("CLAUSE (p) -- PROVENANCE (a foreign/stale receipt must not pass)")
    print("=" * 80)
    n_before = len(FAILURES)
    validate_provenance(rec)
    check("receipt carries gpu / D / timestamp / smoke hash",
          len(FAILURES) == n_before,
          f"gpu={rec.get('gpu')} D={rec.get('D')} "
          f"utc={rec.get('measured_utc')}")

    print()
    print("=" * 80)
    print("CLAUSE (a) -- PASSAGE (evidence parsed from the real smoke subprocess)")
    print("=" * 80)
    check("smoke subprocess exited 0",
          rec.get("smoke_exit_code") == 0,
          f"exit={rec.get('smoke_exit_code')} "
          f"({rec.get('smoke_seconds')}s, {rec.get('smoke_output_bytes')} b)")
    check("marker PARSED FROM CHILD STDOUT (not assigned)",
          rec.get("smoke_marker_seen_in_output") is True,
          f"count={rec.get('smoke_marker_count')} "
          f"stdout_sha256={str(rec.get('smoke_stdout_sha256'))[:12]}")
    sob = rec.get("smoke_output_bytes")
    check("smoke output actually captured (>=200 b)",
          isinstance(sob, int) and sob >= 200, f"{sob} b")
    sh = str(rec.get("smoke_stdout_sha256") or "")
    check("smoke stdout digest is a digest (64 hex chars, not a placeholder)",
          len(sh) == 64 and len(set(sh)) > 1, f"{sh[:12]}...")
    check("checkpoint LOADED",
          rec.get("checkpoint_status") == c["a_passage"]["checkpoint_status"],
          str(rec.get("checkpoint_status")))
    check("trained decoder active",
          rec.get("trained_decoder_active") is True,
          str(rec.get("trained_decoder_active")))
    check("fail-closed guard (generic marker refused)",
          rec.get("fail_closed_generic") is True,
          str(rec.get("fail_closed_generic")))

    print()
    print("=" * 80)
    print(f"CLAUSE (b) -- UNITARY INVARIANT (target {c['b_unitary']['target']} "
          f"+- {c['b_unitary']['tol']:g})")
    print("=" * 80)
    ne = rec.get("perceive_norm_err")
    check("|norm-1| within tolerance",
          ne is not None and ne < c["b_unitary"]["tol"],
          f"norm={rec.get('perceive_norm')} err={ne}")
    check("wave finite", rec.get("perceive_finite") is True,
          str(rec.get("perceive_finite")))
    check("wave shape (8192, 8)", rec.get("perceive_shape") == [8192, 8],
          str(rec.get("perceive_shape")))

    print()
    print("=" * 80)
    print(f"CLAUSE (c) -- AMENDED LATENCY (D={c['D']}, explicit)")
    print("=" * 80)
    print(f"  {'step':18s} {'threshold':>10s} {'actual(us)':>11s} "
          f"{'std':>8s} {'headroom':>9s}  verdict")
    for key, spec in c["c_latency_us"].items():
        # `measure_live` writes "<key>_us" (e.g. perceive_1step_us) while the
        # contract table is keyed WITHOUT the suffix. Looking up the bare key
        # here made clause (c) report UNMEASURED for every REAL receipt -- a
        # gate structurally unable to print LOCKED. Caught by
        # contract_gate_selftest.sh case B, which returned FIXTURE_FAILED
        # instead of FIXTURE_CHECKED.
        field = f"{key}_us"
        actual = rec.get(field)
        if actual is None:
            check(f"c {key}", False, f"UNMEASURED (no '{field}' in receipt)")
            continue
        thr = spec["threshold"]
        std = rec.get(f"{key}_std_us")
        ok = actual <= thr
        print(f"  {key:18s} {thr:10.1f} {actual:11.1f} "
              f"{(f'{std:.1f}' if isinstance(std, (int, float)) else 'n/a'):>8s} "
              f"{thr / max(actual, 1e-9):8.2f}x  {'PASS' if ok else 'FAIL'}")
        check(f"c {key} <= {thr:g} us", ok, f"actual {actual:.1f} us")

    # c2 is a MEAN over repeated live calls; it is only meaningful if the
    # process is not drifting. False => the mean is not a per-step latency.
    check("act benchmark is over a stationary process (2 fresh calls agree)",
          rec.get("act_deterministic") is not False,
          f"det={rec.get('act_deterministic')} "
          f"a1={str(rec.get('act_probe_1'))[:20]} "
          f"a2={str(rec.get('act_probe_2'))[:20]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--receipt", default=None,
                    help="validate a saved receipt JSON instead of measuring")
    ap.add_argument("--live", action="store_true",
                    help="re-measure on the GPU (default)")
    ap.add_argument("--out", default="/tmp/contract_lock_receipt.json")
    args = ap.parse_args()

    print("=" * 80)
    print("HENRI STAGE-1 CONTRACT LOCK GATE")
    print(f"  D = {CONTRACT['D']} (explicit; D=2048 is NOT interchangeable --")
    print("  measured 1.02-1.14x, see docs/stage1-contract-lock.md section 5)")
    print("=" * 80)

    if args.receipt:
        print(f"mode: validate saved receipt {args.receipt}")
        with open(args.receipt) as fh:
            rec = json.load(fh)
    else:
        print("mode: LIVE measurement on target")
        repo = find_repo()
        print(f"repo: {repo}")
        rec = measure_live(repo)
        if args.out:
            with open(args.out, "w") as fh:
                json.dump(rec, fh, indent=2)
            print(f"receipt -> {args.out} ({os.path.getsize(args.out)} bytes)")
    print()

    validate(rec)

    print()
    print("=" * 80)
    if rec.get("synthetic_control") is True:
        if FAILURES:
            print(f"LOCK_VERDICT: FIXTURE_FAILED ({len(FAILURES)} failure(s))")
            for f in FAILURES:
                print("   -", f)
            return 1
        print("LOCK_VERDICT: FIXTURE_CHECKED")
        print("  synthetic fixture -- proves the gate's LOGIC only. This is NOT")
        print("  the contract lock. Only `--live` on the target prints LOCKED.")
        return 2
    if FAILURES:
        print(f"LOCK_VERDICT: UNLOCKED ({len(FAILURES)} failure(s))")
        for f in FAILURES:
            print("   -", f)
        return 1
    print("LOCK_VERDICT: LOCKED")
    print("clauses (a) + (b) + amended (c) all PASS on a measured receipt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
