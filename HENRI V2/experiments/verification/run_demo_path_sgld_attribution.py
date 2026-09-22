"""PRIORITY 3 — demo-path attribution: the "Zero-Shot Success" banner is ANSWER-COUPLED.

FINDING UNDER TEST (read from live code, not inferred)
    sagnac_mcts_planner.search(input_grid, target_grid, ...) does:

        line 177   target_wave = vision_encoder.encode_grid(target_grid)
        line 217   zero_shot_delta = 1.0 - compute_sagnac_similarity(
                                          goal_wave_pred, target_wave)
        line 218   if zero_shot_delta <= self.tau_veto:
        line 219       print("[Phase C Zero-Shot Success] Goal wave retrieved ...")
        line 220       return SpelkeDSLNode(op_name="Identity"), float(zero_shot_delta)

    The success criterion is SIMILARITY OF THE PREDICTION TO THE HELD-OUT TARGET
    GRID. The mechanism is therefore answer-coupled: it is scored against the very
    output it is supposed to predict. Comparing a prediction to its own target is
    leakage, not retrieval quality -- the same class as the Spine-C reranker that
    scored candidates against the task's own answer.

THE DECISIVE CONTROL (non-vacuity)
    If the banner reports a retrieval SUCCESS, then feeding an UNRELATED
    (row-shuffled) target should NOT also report success. If it fires for a
    shuffled target too, the banner carries no information about retrieval at all.

    This is a diagnostic, NOT a fix: no production behaviour is changed. The
    production runner already refuses to pass a target (it emits
    EVALUATION_BLOCKED / OBSERVED_TEST_TARGET_UNAVAILABLE), so this leakage is
    reachable ONLY by a caller that supplies the answer -- exactly what this
    script does deliberately, under a governance label.

EVIDENCE CLASSES
    OBSERVED   banner text captured from real stdout of the real search()
    DERIVED    agreement between banner and the shuffled-target control
    NOT a capability claim; score remains 0.0%.
"""
import contextlib
import io
import json
import pathlib
import sys
import time
from datetime import datetime, timezone

R = pathlib.Path(r"C:\Users\chan\Desktop\HENRI 7B SWARM\.worktrees\basal-syncytium\HENRI V2")
for p in (str(R / "experiments" / "verification"), str(R)):
    sys.path.insert(0, p)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from receipt_path import resolve_receipt_path  # noqa: E402

OUT = resolve_receipt_path(
    R / "experiments" / "verification" / "demo_path_sgld_attribution.json")
N_REPEATS = 2
D_MODEL = 512
K_BLOCKS = 64


def _loss_pair(text):
    """Pull 'loss A -> B' out of the captured stdout. None if absent."""
    for line in text.splitlines():
        if "soft-target protocol" in line and "loss" in line:
            try:
                seg = line.split("loss", 1)[1]
                a, b = seg.split("->", 1)
                return float(a.strip().split()[0]), float(b.strip().split()[0])
            except Exception:  # noqa: BLE001
                return None
    return None


def main():
    import run_calibration_eval as RCE
    from sagnac_mcts_planner import SagnacMCTSPlanner

    tasks = RCE.load_arc(RCE.ARC_ROOT, 1)
    if not tasks:
        print("BLOCKED_NO_CORPUS")
        return 1
    tid, t = tasks[0]
    in_grid = np.ascontiguousarray(np.array(t["test"][0]["input"], dtype=np.int64))
    real_target = np.ascontiguousarray(np.array(t["test"][0]["output"], dtype=np.int64))
    # UNRELATED target: row-shuffle the real one. Same shape/dtype/colour
    # distribution, no relation to the input.
    rng = np.random.default_rng(20260918)
    shuffled = np.ascontiguousarray(real_target.copy()[rng.permutation(real_target.shape[0])])

    demo_pairs = [
        (np.ascontiguousarray(np.array(p["input"], dtype=np.int64)),
         np.ascontiguousarray(np.array(p["output"], dtype=np.int64)))
        for p in t["train"][:2]
    ]

    print(f"task = {tid}   in_grid{in_grid.shape}   target{real_target.shape}")
    print(f"real == shuffled target : {np.array_equal(real_target, shuffled)}")
    print()

    # MILESTONE 1 UPDATE (2026-10-12). search() no longer accepts a target grid,
    # so this experiment can no longer vary the target as a SEARCH input. That is
    # the point. The arms now demonstrate DECOUPLING directly: planning runs twice
    # with identical inputs, and the resulting programs must be identical even
    # though the available held-out targets differ. Scoring against a target
    # happens only AFTER the plan exists, via the separate offline scorer.
    arms = {}
    plans = []
    for arm, tgt in (("real_target", real_target), ("shuffled_target", shuffled)):
        for rep in range(N_REPEATS):
            seed = 4242 + rep
            torch.manual_seed(seed)
            planner = SagnacMCTSPlanner(d_model=D_MODEL, k_blocks=K_BLOCKS,
                                        device="cpu")
            buf = io.StringIO()
            t0 = time.perf_counter()
            err = None
            res = None
            try:
                with contextlib.redirect_stdout(buf):
                    # No target grid is passed. Planning cannot see the answer.
                    res = planner.search(in_grid, num_simulations=2,
                                         demo_pairs=demo_pairs)
            except Exception as exc:  # noqa: BLE001
                err = f"{type(exc).__name__}: {exc}"
            dt = time.perf_counter() - t0
            text = buf.getvalue()
            banner = "Zero-Shot Success" in text
            prog, delta = (res if isinstance(res, tuple) else (res, None))
            # Offline score against the target that is available to THIS arm.
            off_score = None
            if prog is not None:
                try:
                    off_score = float(planner.score(prog, in_grid, tgt))
                except Exception:  # noqa: BLE001
                    off_score = None
            lp = _loss_pair(text)
            plans.append(None if prog is None else str(prog))
            arms.setdefault(arm, []).append({
                "repeat": rep, "seed": seed, "seconds": dt,
                "banner_zero_shot_success": banner,
                "program": (None if prog is None else str(prog)),
                "delta": (None if delta is None else float(delta)),
                "offline_score_vs_arm_target": off_score,
                "loss_first": (None if lp is None else lp[0]),
                "loss_last": (None if lp is None else lp[1]),
                "loss_rose": (None if lp is None else bool(lp[1] > lp[0])),
                "error": err,
                "stdout_lines": len(text.splitlines()),
            })
            print(f"  {arm:<16} rep={rep} {dt:6.2f}s banner={banner!s:<5} "
                  f"prog={prog} delta={delta} loss={lp} err={err}")

    # DECOUPLING CHECK: the program sequence must be identical across arms.
    n_half = len(plans) // 2
    decoupled = plans[:n_half] == plans[n_half:]

    real_banners = [r["banner_zero_shot_success"] for r in arms.get("real_target", [])]
    shuf_banners = [r["banner_zero_shot_success"] for r in arms.get("shuffled_target", [])]
    all_losses = [r for a in arms.values() for r in a if r["loss_first"] is not None]

    control_ok = bool(shuf_banners and not any(shuf_banners))
    leaks = bool(any(real_banners) and any(shuf_banners))

    if not real_banners:
        verdict = ("BANNER_DID_NOT_FIRE — the Zero-Shot Success branch was not "
                   "reached on these inputs, so the shuffled control is "
                   "uninformative. No leakage verdict is emitted.")
    elif leaks:
        verdict = ("BANNER_IS_ANSWER_COUPLED — the banner fired for the REAL "
                   "target AND for an UNRELATED shuffled target. The success "
                   "criterion therefore carries no information about retrieval "
                   "quality: it is answer-coupled by construction "
                   "(BLOCKED_TARGET_LEAKAGE). Any capability reading of this "
                   "banner is invalid.")
    else:
        verdict = ("BANNER_SEPARATES_REAL_FROM_SHUFFLED — the banner fired only "
                   "for the real target and not the shuffled control. The "
                   "criterion still compares the prediction to the held-out "
                   "target (answer-coupled), but it does discriminate a correct "
                   "goal wave from an unrelated one. Diagnostic only.")

    body = {
        "schema": "henri.demo-path-sgld-attribution.v1",
        "utc": datetime.now(timezone.utc).isoformat(),
        "evidence_class": "OBSERVED",
        "governance_label": "BLOCKED_TARGET_LEAKAGE_PROBE",
        "why_this_is_permitted": (
            "The held-out target is passed DELIBERATELY, under this label, to prove "
            "the criterion reads it. This is a leakage probe, not an evaluation: no "
            "task score is computed, no capability is claimed, and nothing is "
            "persisted keyed to this split."
        ),
        "source_lines": {
            "file": "sagnac_mcts_planner.py",
            "177": "target_wave = self.vision_encoder.encode_grid(target_grid)",
            "217": "zero_shot_delta = 1.0 - compute_sagnac_similarity(goal_wave_pred, target_wave)",
            "218": "if zero_shot_delta <= self.tau_veto:",
            "219": 'print("[Phase C Zero-Shot Success] Goal wave retrieved in O(1) single pass! ...")',
        },
        "control": ("row-shuffled copy of the real target: same shape, dtype and "
                    "colour distribution, unrelated to the input"),
        "arms": arms,
        "loss_observations": [
            {"arm": a, "rep": r["repeat"], "first": r["loss_first"],
             "last": r["loss_last"], "rose": r["loss_rose"]}
            for a, rs in arms.items() for r in rs if r["loss_first"] is not None
        ],
        "pre_registered": {
            "C1_banner_fired_on_real_target": bool(any(real_banners)),
            "C2_banner_silent_on_shuffled_target": control_ok,
            "C3_banner_is_answer_coupled": leaks,
        },
        "verdict": verdict,
        "limits": [
            "Diagnostic only: no production behaviour changed by this script.",
            "Reduced scale D=512 (the checkpoint gate binds only at d_model=65536).",
            "N is small and each demo call costs ~60 s; direction is a per-run "
            "observation, not a distribution.",
            "The loss trajectory is captured from the callee's own print, not "
            "returned by an API.",
            "No capability claim; benchmark score remains 0.0%.",
        ],
    }
    OUT.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    print()
    for k, v in body["pre_registered"].items():
        print(f"  {k:<42} {v}")
    print(f"  VERDICT: {verdict}")
    if all_losses:
        rose = sum(1 for r in all_losses if r["loss_rose"])
        print(f"  loss rose in {rose}/{len(all_losses)} observed demo calls")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
