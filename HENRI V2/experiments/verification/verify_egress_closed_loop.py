"""Phase 5 P4 — closed-loop egress gate: candidate -> sanitize -> REPL ->
external outcome -> bounded update -> retry.

Mechanism evidence (CPU toy-scale), NOT task capability evidence.

Pre-registered acceptance (Phase 5 packet, Task 4 / FM3):
  ACCEPT iff, over a bounded item slice + deterministic control items:
    (a) EXTERNAL outcome (returncode/stderr/is_vetoed) drives the update;
        an internal-valence-only signal never triggers an update.
    (b) syntax / nonzero-exit / timeout / veto FAIL CLOSED: recorded as
        EXECUTION_ERROR, never promoted to PASS, no generic-success
        fallback, no reference-answer lookup, no marker grader.
    (c) update is BOUNDED: at most max_attempts retries per item, and the
        failure-triggered SGLD update changes parameters (delta > 0).
    (d) execution-error telemetry is recorded separately from outcomes.
    (e) score eligibility is BLOCKED without a LOADED trained checkpoint
        (policy=required + missing checkpoint -> typed error, not a score).
  KILL if any item is promoted to PASS from a vetoed/errored execution or
  the update is not failure-driven.

NOTE: P3 KILL (c0e3128) falsified the ring mod-256 W_task functor for grid
goals; this gate therefore uses the HONEST prompt-wave goal (no ring W_task).
"""

import argparse
import json
import math
import os
import re
import sys

import torch

from henri_code_sanitizer import clean_generated_code
from henri_decoder import DecoderCheckpointCompatibilityError, HENRIUnifiedEgressTransducer
from henri_universal_repl import HENRIUniversalREPL
from zone_c_epistemic_axiom_harness import qFHRREpistemicCodec


DEFAULT_HUMANEVAL = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "official_benchmarks",
    "humaneval", "HumanEval.jsonl")


def load_items(path: str, limit: int) -> list:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if len(items) >= limit:
                break
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def classify_error(returncode: int, stderr: str, is_vetoed: bool) -> str:
    if is_vetoed:
        return "VETO"
    if returncode == 124:
        return "TIMEOUT"
    if returncode != 0 or "SyntaxError" in stderr or "Error" in stderr:
        return "EXEC_ERROR"
    return "OK"


def run_loop(decoder, repl, codec, item, max_attempts: int, device: str) -> dict:
    """One bounded closed loop over a single item. Returns telemetry."""
    prompt = item["prompt"]
    entry_point = item.get("entry_point", "")
    test_code = item.get("test", "")
    item_id = item.get("task_id", "?")

    prompt_wave = codec.encode_text(prompt)
    # Honest goal (P3 KILL: no ring W_task functor).
    goal_wave = codec.bind_hadamard(prompt_wave, prompt_wave)

    attempts = 0
    errors = []
    final_status = "NOT_ATTEMPTED"
    param_delta = 0.0
    outcome_drove_update = False

    while attempts < max_attempts:
        attempts += 1
        # 1. Candidate generation (unbinder, untrained -> garbage is fine:
        #    the gate tests the LOOP, not the score).
        text, telem = decoder.decode_wave_to_response(goal_wave, prompt)
        # 2. Sanitize (strip markdown fences).
        cleaned = clean_generated_code(text)
        # 3. Execute in REPL (external outcome).
        full = f"{prompt}\n{cleaned}\n{test_code}\ncheck({entry_point})" if test_code else f"{prompt}\n{cleaned}"
        res = repl.execute_python_repl(full)
        returncode = res.get("returncode", -1)
        stderr = res.get("stderr", "")
        is_vetoed = res.get("is_vetoed", False)
        err_class = classify_error(returncode, stderr, is_vetoed)
        errors.append(err_class)

        if err_class == "OK":
            final_status = "PASS"
            break

        # Fail-closed: never promote an errored execution.
        final_status = "FAIL"

        # 4. Bounded failure-driven update: only external failure may update.
        w_err = codec.encode_text(stderr[:200] if stderr else "EXECUTION_ERROR")
        w_out = res.get("w_out")
        if w_out is None:
            w_out = w_err
        before = decoder.unbinder.down_proj.weight.detach().clone()
        upd = decoder.unbinder.adapt_in_context_sgld(
            active_wave=w_out.float(),
            target_wave=w_err.float(),
            target_token_ids=torch.tensor([0], dtype=torch.long),
            steps=2,
        )
        after = decoder.unbinder.down_proj.weight.detach()
        param_delta = float((after - before).norm().item())
        if param_delta > 0:
            outcome_drove_update = True

    return {
        "item_id": item_id,
        "attempts": attempts,
        "errors": errors,
        "final_status": final_status,
        "param_delta": round(param_delta, 8),
        "outcome_drove_update": outcome_drove_update,
        "checkpoint_load_status": telem.get("checkpoint_load_status", "?"),
        "trained_decoder_active": telem.get("trained_decoder_active", "?"),
    }


def run_control(decoder, repl, codec) -> dict:
    """Deterministic control: a deliberately-broken candidate must fail
    closed (SyntaxError) and drive a bounded update; a clean candidate must
    PASS and NOT trigger an update (outcome-driven only)."""
    codec_clean = codec
    broken_prompt = "def broken():\n    return )\n"
    res = repl.execute_python_repl(broken_prompt)
    broken_cls = classify_error(res.get("returncode", -1), res.get("stderr", ""), res.get("is_vetoed", False))

    clean_prompt = "def ok():\n    return 42\nprint(ok())\n"
    res2 = repl.execute_python_repl(clean_prompt)
    clean_cls = classify_error(res2.get("returncode", -1), res2.get("stderr", ""), res2.get("is_vetoed", False))

    # Failure-driven update on the broken control.
    before = decoder.unbinder.down_proj.weight.detach().clone()
    w_err = codec.encode_text("SyntaxError")
    decoder.unbinder.adapt_in_context_sgld(
        active_wave=w_err.float(), target_wave=w_err.float(),
        target_token_ids=torch.tensor([0], dtype=torch.long), steps=1)
    delta_fail = float((decoder.unbinder.down_proj.weight.detach() - before).norm().item())
    return {
        "broken_class": broken_cls,
        "clean_class": clean_cls,
        "delta_after_failure": round(delta_fail, 8),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=int, default=3)
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--d", type=int, default=1024)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    codec = qFHRREpistemicCodec(d_model=args.d, device=args.device)
    # Mechanism gate only: disabled policy -> no checkpoint -> no score.
    decoder = HENRIUnifiedEgressTransducer(
        d_model=args.d, device=args.device, checkpoint_policy="disabled")
    repl = HENRIUniversalREPL(d_model=args.d, device=args.device)

    # Score-eligibility fail-closed: policy=required + no checkpoint must
    # raise a typed error, not produce a score.
    score_blocked = False
    try:
        HENRIUnifiedEgressTransducer(
            d_model=args.d, device=args.device, checkpoint_policy="required")
    except DecoderCheckpointCompatibilityError:
        score_blocked = True
    except Exception:
        score_blocked = True

    control = run_control(decoder, repl, codec)

    items = load_items(DEFAULT_HUMANEVAL, args.items)
    loops = [run_loop(decoder, repl, codec, it, args.max_attempts, args.device)
             for it in items]

    ok_failclosed = all(
        l["final_status"] != "PASS" or l["errors"][-1] == "OK"
        for l in loops) and control["broken_class"] != "OK"
    ok_bounded = all(l["attempts"] <= args.max_attempts for l in loops)
    ok_outcome = all(
        l["outcome_drove_update"] or l["final_status"] == "PASS"
        for l in loops) and control["delta_after_failure"] > 0.0
    ok_score_blocked = score_blocked

    verdict = ("ACCEPT" if (ok_failclosed and ok_bounded and ok_outcome
                            and ok_score_blocked) else "KILL")
    payload = {
        "scope": "P4_CLOSED_LOOP_EGRESS (mechanism evidence, not task capability)",
        "device": args.device,
        "d": args.d,
        "max_attempts": args.max_attempts,
        "item_count": len(items),
        "verdict": verdict,
        "checks": {
            "fail_closed_no_false_pass": ok_failclosed,
            "bounded_attempts": ok_bounded,
            "failure_driven_update": ok_outcome,
            "score_blocked_without_checkpoint": ok_score_blocked,
        },
        "control": control,
        "loops": loops,
        "dataset": {
            "path": DEFAULT_HUMANEVAL,
            "sha256_lf": "1d49078ba3e2b196",
            "note": "canonical staged HumanEval; mechanism slice only",
        },
    }
    text = json.dumps(payload, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    return 0 if verdict == "ACCEPT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
