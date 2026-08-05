"""Run 21 structured-codec arm matrix (remote CUDA only).

Runs the production rank probe (mbpp_rank_probe.py --per-item-wtask
--expressible-only) across the five pre-registered arms and emits a
single run21_summary.json with the pre-registered verdict.

Local CPU execution of this runner is a PATH SMOKE ONLY; it must not be
promoted as task evidence.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROBE = ROOT / "mbpp_rank_probe.py"

ARMS = ["legacy", "structured", "structured-nopos", "structured-shuffled", "identity"]
TASKS = [14, 62, 89]


def build_failures_file(out_dir: Path) -> Path:
    """Deterministic harness input: the three expressible CEGIS_MISS tasks.

    The probe reloads canonical items from mbpp.jsonl by task_id; this
    file only selects the expressible subset, mirroring run20's
    expressible-only protocol.
    """
    p = out_dir / "failures_expressible.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for t in TASKS:
            f.write(json.dumps({
                "task_id": t,
                "failure_reason": "CEGIS_MISS",
                "pass": False,
                "source": "run21_matrix_selector",
            }) + "\n")
    return p


def run_arm(device: str, codec: str, failures: Path, out_dir: Path) -> int:
    arm_dir = out_dir / codec
    arm_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(PROBE),
        "--failures-jsonl", str(failures),
        "--sample", "10",
        "--output", str(arm_dir),
        "--device", device,
        "--expressible-only",
        "--per-item-wtask",
        "--codec", codec,
    ]
    print(f"[run21] arm={codec} cmd={' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    (arm_dir / "run21_arm_stdout.log").write_text(proc.stdout or "", encoding="utf-8")
    (arm_dir / "run21_arm_stderr.log").write_text(proc.stderr or "", encoding="utf-8")
    if proc.returncode != 0:
        print(f"[run21] arm={codec} FAILED rc={proc.returncode} "
              f"stderr_tail={proc.stderr[-500:]!r}", flush=True)
    return proc.returncode


def load_summary(codec: str, out_dir: Path) -> dict | None:
    p = out_dir / codec / "rank_probe_summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def verdict(summaries: dict[str, dict], controls: dict[str, dict]) -> dict:
    def control_healthy(arm: str) -> bool:
        c = controls.get(arm, {})
        if c.get("codec_name") != "structured_char_position_qfhrr":
            return True  # legacy arm has no structured gate
        baseline = c.get("random_baseline", 1.0 / 65536.0)
        return (
            c.get("identical_input_sim", 0.0) >= 0.999
            and c.get("nearby_input_sim", 0.0) > 10.0 * baseline
            and c.get("nearby_output_sim", 0.0) > 10.0 * baseline
            and c.get("unrelated_sim", 1.0) <= 10.0 * baseline
            and abs(c.get("position_swap_sim", 1.0) - c.get("identical_input_sim", 0.0)) > 0.05
        )

    def ranks_for(arm: str, variant: str) -> dict[int, int | None]:
        s = summaries.get(arm, {})
        out: dict[int, int | None] = {}
        for v, d in (s.get("by_variant") or {}).items():
            if v == variant:
                for tid, r in (d.get("task_ranks") or {}).items():
                    out[int(tid)] = r
        return out

    # Identity arm has no W_task and no EDMD; its rows carry variant
    # IDENTITY (prompt-wave ranking). It is the no-supervision baseline
    # for both A_EDMD and B_SINGLE_PASS comparisons.
    r_identity = ranks_for("identity", "IDENTITY")

    # Acceptance: structured (full) arm control_healthy AND 62,89 <= 24 in
    # at least one variant AND strictly better than identity AND legacy
    # for that variant.
    accepted_variant = None
    structured_ok = control_healthy("structured")
    for variant in ("A_EDMD", "B_SINGLE_PASS"):
        r_struct = ranks_for("structured", variant)
        if all(v is not None and v <= 24 for k, v in r_struct.items() if k in (62, 89)) and len(r_struct) >= 2:
            r_legacy = ranks_for("legacy", variant)
            beats_id = all(
                r_struct.get(k) is not None
                and (r_identity.get(k) is None or r_struct[k] < r_identity[k])
                for k in (62, 89))
            beats_legacy = all(
                r_struct.get(k) is not None
                and (r_legacy.get(k) is None or r_struct[k] < r_legacy[k])
                for k in (62, 89))
            if beats_id and beats_legacy:
                accepted_variant = variant
                break
    if structured_ok and accepted_variant:
        return {"verdict": "ACCEPTANCE_MET", "accepted_variant": accepted_variant}
    if structured_ok:
        return {"verdict": "FALSIFIED_AT_SCALE", "accepted_variant": None}
    return {"verdict": "INVALID_PLUMBING", "accepted_variant": None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--failures-jsonl", default=None,
                    help="Optional run item_results.jsonl; defaults to the 3-task selector")
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    failures = Path(args.failures_jsonl) if args.failures_jsonl else build_failures_file(out_dir)

    rc_failures = []
    for arm in ARMS:
        rc = run_arm(args.device, arm, failures, out_dir)
        if rc != 0:
            rc_failures.append(arm)
    if rc_failures:
        summary = {
            "verdict": "BLOCKED_INFRASTRUCTURE",
            "failed_arms": rc_failures,
            "note": "one or more arms did not exit 0; verdict is infrastructure-blocked",
        }
    else:
        summaries = {a: load_summary(a, out_dir) for a in ARMS}
        controls = {}
        for a in ARMS:
            cp = out_dir / a / "codec_geometry_control.json"
            if cp.exists():
                controls[a] = json.loads(cp.read_text(encoding="utf-8"))
        summary = verdict(summaries, controls)
        summary["arms"] = {
            a: {
                "control_healthy": (controls.get(a, {}) or {}).get("codec_name") != "structured_char_position_qfhrr" or None,
                "by_variant": (summaries.get(a) or {}).get("by_variant"),
            } for a in ARMS
        }
        summary["control"] = {a: controls.get(a) for a in ARMS}
        summary["exit_codes"] = {a: 0 for a in ARMS}

    out = out_dir / "run21_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if summary.get("verdict") != "BLOCKED_INFRASTRUCTURE":
        (out_dir / "RUN21_DONE").write_text("", encoding="utf-8")
    print("[run21] SUMMARY " + json.dumps(summary, indent=2), flush=True)
    return 0 if summary.get("verdict") not in ("BLOCKED_INFRASTRUCTURE",) else 1


if __name__ == "__main__":
    raise SystemExit(main())
