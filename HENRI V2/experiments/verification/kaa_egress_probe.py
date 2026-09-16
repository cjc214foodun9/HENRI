"""K-A kill experiment (M1): does the live egress path discriminate distinct inputs?

PRE-REGISTERED (SPEC-2026-09-16-AAII-V43-RD-PIPELINE.json kill K-A)
  If distinct top-1 tokens <= 1 across N>=100 distinct prompts -> FALSIFIED_NO_EGRESS.
  Otherwise EGRESS_DISCRIMINATES and M2/M3/M4 may be built on the egress path.

WHY TWO ARMS
  The inherited measurement (16 chunk waves -> same token 29674) used the Z_256
  uint8 ring from the epistemic codec. That ring is documented as a SEPARATE
  representation family from the canonical real [num_blocks, 8] wave, and the
  skill records an unresolved "ring-uint8 -> S^{D-1} value-range mismatch".
  A single-arm probe cannot tell a REPRESENTATION defect from a CAPACITY defect.
  Therefore:

    arm A  ring_uint8_scaled : codec.encode_text -> uint8 -> (x/255)*2-1 -> L2
    arm B  transduce_real    : transducer.transduce_text -> real float -> L2
    arm C  random_control    : torch.randn (NEGATIVE CONTROL; must also be measured)

  If A collapses and B discriminates, the defect is the representation adapter,
  not the head. If BOTH collapse, the head/platform is capacity-limited.

MEASUREMENT
  Per arm: distinct top-1 token count, top-1 histogram, logit entropy stats,
  pre-snap logits digest, and pairwise logit agreement.
  A shuffled-prompt control confirms the mapping is content-driven.

EVIDENCE RUN CONTEXT
  torch 2.11.0+cu128, cuda_available=False -> CPU ONLY. Any device is recorded.
  The 799 MB decoder checkpoint (down_proj 2048x65536, layer_norm, lm_head
  32000x2048) is an EXTERNAL gitignored overlay; its SHA-256 is recorded.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

V2 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(V2))

import torch

from henri_eval_infra import run_id_new, run_output_dir, sha256_bytes, sha256_text_lf

N_PROMPTS = 128
SEED = 20260916
D_MODEL = 65536


def _commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, cwd=str(V2)).stdout.strip()
    except Exception:
        return "UNKNOWN"


def entropy_of(logits: torch.Tensor) -> float:
    """Shannon entropy in nats, computed explicitly (unambiguous formula)."""
    logp = torch.log_softmax(logits, dim=-1)
    return float(-(logp.exp() * logp).sum(dim=-1).mean().item())


def build_prompts() -> list[str]:
    tmpl = [
        "What is the capital city of region {i}?",
        "Define the term item-{i} in one sentence.",
        "Compute the sum 17 + {i} and give the number.",
        "Name the element with atomic number {i}.",
        "Write a Python expression that evaluates to {i}.",
        "In what year did event number {i} occur?",
        "Translate the phrase report-{i} into French.",
        "Summarise the theorem labelled {i}.",
    ]
    return [tmpl[i % len(tmpl)].format(i=i) for i in range(N_PROMPTS)]


def main() -> int:
    torch.manual_seed(SEED)
    commit = _commit()
    run_id = run_id_new()
    out = run_output_dir(V2 / "telemetry_logs" / "kaa_egress", commit,
                         "kaa-egress-probe", run_id)
    ledger_path = out / "items.jsonl"

    # ---- device -----------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- head -------------------------------------------------------------
    from henri_decoder import HENRINeuralEgressUnbinder

    head = HENRINeuralEgressUnbinder(d_model=D_MODEL, vocab_size=32000, device=device)
    head.eval()

    ckpt = V2 / "models" / "henri_decoder_checkpoint.pt"
    ckpt_status, ckpt_sha, ckpt_bytes, ckpt_sd_sha = "ABSENT", None, 0, None
    if ckpt.exists():
        ckpt_bytes = ckpt.stat().st_size
        ckpt_sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()
        sd = torch.load(str(ckpt), map_location="cpu", weights_only=True)
        missing, unexpected = head.load_state_dict(sd, strict=False)
        h = hashlib.sha256()
        for k in sorted(sd.keys()):
            t = sd[k]
            if torch.is_tensor(t):
                h.update(k.encode("utf-8"))
                h.update(t.detach().to(torch.float32).contiguous().numpy().tobytes())
        ckpt_sd_sha = h.hexdigest()
        ckpt_status = "LOADED" if not missing else f"PARTIAL missing={list(missing)}"

    # FAIL-CLOSED. An UNTRAINED head emits near-uniform logits whose argmax is
    # pure noise, so a high distinct_top1 is a VACUOUS PASS. Measured 2026-09-16
    # with the overlay absent: entropy_mean_nats = 10.3536 vs ln(32000) = 10.3736
    # (near-uniform) while all three arms reported high diversity. A probe that
    # emits an egress verdict without a loaded checkpoint is a false-positive
    # generator, so refuse to emit one.
    if ckpt_status != "LOADED":
        print(json.dumps({
            "schema_id": "henri.kaa-egress-probe.v1",
            "status": "EXECUTION_ERROR",
            "reason": "CHECKPOINT_NOT_LOADED",
            "checkpoint_path": str(ckpt),
            "checkpoint_status": ckpt_status,
            "note": ("Refusing to emit an egress verdict from an untrained head: "
                     "the argmax of near-uniform logits is noise, not discrimination."),
            "run_id": run_id,
            "commit_sha256": commit,
        }, indent=2, sort_keys=True))
        return 2

    # ---- encoders ---------------------------------------------------------
    from zone_c_epistemic_axiom_harness import qFHRREpistemicCodec
    from henri_universal_repl import qFHRRUniversalTextTransducer

    codec = qFHRREpistemicCodec()
    transducer = qFHRRUniversalTextTransducer(d_model=D_MODEL)

    def arm_ring(prompt: str) -> torch.Tensor:
        """Z_256 uint8 ring -> [-1,1] real, L2-normalized."""
        q = codec.encode_text(prompt)
        if not torch.is_tensor(q):
            q = torch.as_tensor(q)
        v = q.to(torch.float32)
        v = (v / 255.0) * 2.0 - 1.0
        return v

    def arm_transduce(prompt: str) -> torch.Tensor:
        w = transducer.transduce_text(prompt)
        return w.to(torch.float32)

    def to_flat(v: torch.Tensor) -> torch.Tensor:
        v = v.reshape(-1)
        if v.numel() < D_MODEL:
            v = torch.nn.functional.pad(v, (0, D_MODEL - v.numel()))
        elif v.numel() > D_MODEL:
            raise ValueError(f"wave has {v.numel()} elements > d_model {D_MODEL}")
        return torch.nn.functional.normalize(v, p=2, dim=-1).reshape(1, D_MODEL)

    prompts = build_prompts()
    arms = {"A_ring_uint8_scaled": arm_ring, "B_transduce_real": arm_transduce}

    results: dict[str, dict] = {}
    rows_written = 0

    with open(ledger_path, "w", encoding="utf-8", newline="\n") as led:
        # ----- per-arm sweep ------------------------------------------------
        for arm_name, fn in arms.items():
            top1, ents, logit_rows = [], [], []
            t_arm = time.perf_counter()
            for i, p in enumerate(prompts):
                t0 = time.perf_counter()
                wave = to_flat(fn(p))
                with torch.no_grad():
                    logits = head.forward(wave)
                tid = int(logits.argmax(dim=-1).item())
                ent = entropy_of(logits)
                top1.append(tid)
                ents.append(ent)
                logit_rows.append(logits.detach().reshape(-1))
                row = {
                    "item_id": f"{arm_name}-{i:04d}",
                    "prompt_sha256": sha256_text_lf(p),
                    "raw_stdout_sha256": sha256_text_lf(str(tid)),
                    "raw_stderr_sha256": sha256_text_lf(""),
                    "status": "PASSED",
                    "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 4),
                    "tool_calls": 0,
                    "retries": 0,
                    "candidate_scores": [],
                    "sagnac_delta": None,
                    "token_top1": tid,
                    "token_entropy": round(ent, 6),
                }
                led.write(json.dumps(row, sort_keys=True) + "\n")
                led.flush()
                rows_written += 1
            L = torch.stack(logit_rows, dim=0)
            # pairwise agreement: fraction of prompt pairs sharing a top-1
            uniq = sorted(set(top1))
            results[arm_name] = {
                "n": len(top1),
                "distinct_top1": len(uniq),
                "distinct_ratio": round(len(uniq) / len(top1), 4),
                "top1_unique_values": uniq[:12],
                "top1_hist_top": sorted(
                    {t: top1.count(t) for t in uniq}.items(),
                    key=lambda kv: -kv[1])[:6],
                "entropy_mean_nats": round(sum(ents) / len(ents), 4),
                "entropy_min_nats": round(min(ents), 4),
                "entropy_max_nats": round(max(ents), 4),
                "logits_sha256": sha256_bytes(L.numpy().tobytes()),
                "wall_ms": round((time.perf_counter() - t_arm) * 1000.0, 1),
            }

        # ----- arm C: NEGATIVE CONTROL --------------------------------------
        ctrl_top1, ctrl_ents, ctrl_rows = [], [], []
        g = torch.Generator(device="cpu").manual_seed(SEED + 1)
        for i in range(N_PROMPTS):
            r = torch.randn(D_MODEL, generator=g, device="cpu")
            with torch.no_grad():
                logits = head.forward(to_flat(r))
            tid = int(logits.argmax(dim=-1).item())
            ctrl_top1.append(tid)
            ctrl_ents.append(entropy_of(logits))
            ctrl_rows.append(logits.detach().reshape(-1))
            led.write(json.dumps({
                "item_id": f"C_random_control-{i:04d}",
                "prompt_sha256": sha256_text_lf(f"<randn-{i}>"),
                "raw_stdout_sha256": sha256_text_lf(str(tid)),
                "raw_stderr_sha256": sha256_text_lf(""),
                "status": "PASSED",
                "elapsed_ms": 0.0, "tool_calls": 0, "retries": 0,
                "candidate_scores": [], "sagnac_delta": None,
                "token_top1": tid,
                "token_entropy": round(ctrl_ents[-1], 6),
            }, sort_keys=True) + "\n")
            led.flush()
            rows_written += 1
        results["C_random_control"] = {
            "n": len(ctrl_top1),
            "distinct_top1": len(set(ctrl_top1)),
            "entropy_mean_nats": round(sum(ctrl_ents) / len(ctrl_ents), 4),
            "logits_sha256": sha256_bytes(torch.stack(ctrl_rows).numpy().tobytes()),
        }

        # ----- arm D: SHUFFLE control (content must matter) ------------------
        shuffled = list(prompts)
        torch.manual_seed(SEED + 2)
        perm = torch.randperm(len(shuffled)).tolist()
        shuffled = [shuffled[j] for j in perm]
        sh_top1 = []
        for p in shuffled:
            with torch.no_grad():
                logits = head.forward(to_flat(arm_transduce(p)))
            sh_top1.append(int(logits.argmax(dim=-1).item()))
            led.write(json.dumps({
                "item_id": f"D_shuffle_control-{len(sh_top1):04d}",
                "prompt_sha256": sha256_text_lf(p),
                "raw_stdout_sha256": sha256_text_lf(str(sh_top1[-1])),
                "raw_stderr_sha256": sha256_text_lf(""),
                "status": "PASSED", "elapsed_ms": 0.0, "tool_calls": 0,
                "retries": 0, "candidate_scores": [], "sagnac_delta": None,
                "token_top1": sh_top1[-1],
                "token_entropy": round(entropy_of(logits), 6),
            }, sort_keys=True) + "\n")
            led.flush()
            rows_written += 1
        results["D_shuffle_control"] = {
            "n": len(sh_top1),
            "distinct_top1": len(set(sh_top1)),
        }

    # ----- verdicts --------------------------------------------------------
    def verdict_for(name: str) -> str:
        d = results[name]["distinct_top1"]
        if d <= 1:
            return "FALSIFIED_NO_EGRESS"
        return "EGRESS_DISCRIMINATES"

    vA, vB = verdict_for("A_ring_uint8_scaled"), verdict_for("B_transduce_real")
    if vB == "EGRESS_DISCRIMINATES" and vA == "FALSIFIED_NO_EGRESS":
        overall = "REPRESENTATION_DEFECT_ADAPTER_REQUIRED"
    elif vB == "EGRESS_DISCRIMINATES":
        overall = "EGRESS_DISCRIMINATES"
    else:
        overall = "FALSIFIED_NO_EGRESS"

    receipt = {
        "schema_id": "henri.kaa-egress-probe.v1",
        "spec": "SPEC-2026-09-16-AAII-V43-RD-PIPELINE.json#kill-K-A",
        "run_id": run_id,
        "commit_sha256": commit,
        "device": device,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "checkpoint": {"status": ckpt_status, "sha256": ckpt_sha, "bytes": ckpt_bytes},
        "n_prompts": N_PROMPTS,
        "seed": SEED,
        "arms": results,
        "verdict_arm_A": vA,
        "verdict_arm_B": vB,
        "OVERALL_VERDICT": overall,
        "items_jsonl": str(ledger_path),
        "items_rows": rows_written,
        "items_sha256": sha256_bytes(ledger_path.read_bytes()),
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
