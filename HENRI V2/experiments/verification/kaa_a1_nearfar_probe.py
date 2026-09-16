"""K-A Amendment A1: separate INGRESS loss from EGRESS loss.

Supersedes the vacuous K-A criterion (see KAA_AMENDMENT_A1_nearfar_egress.md).

WHY (measured, commit 7750abb)
  K-A counted distinct top-1 tokens. Content arms scored 39/128 and 19/128, but
  the RANDOM control scored 37/128 -> the gate passed on noise. A criterion the
  negative control also passes detects nothing.

THIS PROBE measures differential response instead:
  * near pair = same template, one content word changed  (should be SIMILAR)
  * far pair  = different template and content topic      (should be DISTANT)
  at two boundaries so the defect localizes:
      W = encoded wave cosine      (ingress)
      L = output logit cosine      (egress)
  plus a MATCHED noise control (identical pair structure on random waves; its
  margin is the floor any content effect must beat) and a determinism check.

VERDICTS (pre-registered, fixed)
  ERROR_NONDETERMINISTIC | INGRESS_LOSES_CONTENT | INGRESS_WEAK
  EGRESS_DESTROYS_CONTENT | EGRESS_CARRIES_CONTENT

This is a CPU mechanism probe. It yields NO external score and NO capability claim.
"""
from __future__ import annotations

import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

V2 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(V2))

import torch
import torch.nn.functional as F

from henri_eval_infra import run_id_new, run_output_dir, sha256_text_lf, sha256_bytes
from henri_decoder import HENRINeuralEgressUnbinder
from zone_c_epistemic_axiom_harness import qFHRREpistemicCodec

D = 65536
SEED = 20260916
N_PAIRS = 32
BOOT = 2000

TEMPLATES = [
    "What is the capital of {t}?",
    "What is the country of {t}?",
    "Write a Python function that returns {t}.",
    "Summarise the theorem labelled {t}.",
    "In what year did event {t} occur?",
    "Name the element with atomic number {t}.",
    "Translate the phrase {t} into French.",
    "Compute the derivative of expression {t}.",
]


def topics(offset: int) -> list[str]:
    return [f"topic-{offset + i:03d}" for i in range(N_PAIRS)]


def build_pairs() -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """near = same topic, template differs by one word; far = different both."""
    near, far = [], []
    ts, tf = topics(0), topics(100)
    for i in range(N_PAIRS):
        near.append((TEMPLATES[0].format(t=ts[i]), TEMPLATES[1].format(t=ts[i])))
        far.append((TEMPLATES[0].format(t=ts[i]), TEMPLATES[2].format(t=tf[i])))
    return near, far


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).item())


def boot_ci(vals: list[float], n: int = BOOT, alpha: float = 0.05) -> tuple[float, float]:
    if not vals:
        return (0.0, 0.0)
    g = torch.Generator().manual_seed(SEED)
    v = torch.tensor(vals, dtype=torch.float64)
    idx = torch.randint(0, v.numel(), (n, v.numel()), generator=g)
    means = v[idx].mean(dim=1)
    return (float(torch.quantile(means, alpha / 2).item()),
            float(torch.quantile(means, 1 - alpha / 2).item()))


def margin(near: list[float], far: list[float]) -> dict:
    """Paired margin = mean(near_cos) - mean(far_cos), with a bootstrap CI."""
    pairs = [n - f for n, f in zip(near, far)]
    lo, hi = boot_ci(pairs)
    return {
        "near_mean": round(statistics.fmean(near), 6),
        "far_mean": round(statistics.fmean(far), 6),
        "margin": round(statistics.fmean(pairs), 6),
        "ci95": [round(lo, 6), round(hi, 6)],
        "ci_excludes_zero": bool(lo > 0.0 or hi < 0.0),
    }


def to_flat(v: torch.Tensor) -> torch.Tensor:
    v = v.reshape(-1).to(torch.float32)
    if v.numel() < D:
        v = F.pad(v, (0, D - v.numel()))
    elif v.numel() > D:
        v = v[:D]
    return F.normalize(v, p=2, dim=-1).reshape(1, D)


def main() -> int:
    torch.manual_seed(SEED)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True, cwd=str(V2)).stdout.strip()
    except Exception:
        commit = "UNKNOWN"

    out = run_output_dir(V2 / "telemetry_logs" / "kaa_a1", commit,
                         "kaa-a1-nearfar", run_id_new())
    rows_path = out / "items.jsonl"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    head = HENRINeuralEgressUnbinder(d_model=D, vocab_size=32000, device=device)
    head.eval()

    ck = V2 / "models" / "henri_decoder_checkpoint.pt"
    ck_status, ck_sha = "ABSENT", None
    if ck.exists():
        ck_sha = hashlib.sha256(ck.read_bytes()).hexdigest()
        sd = torch.load(str(ck), map_location="cpu", weights_only=True)
        missing, _ = head.load_state_dict(sd, strict=False)
        ck_status = "LOADED" if not missing else f"PARTIAL missing={list(missing)}"

    # FAIL-CLOSED: an untrained head's argmax is noise (measured 2026-09-16,
    # entropy 10.3536 vs ln(32000)=10.3735). Refuse to emit a verdict.
    if ck_status != "LOADED":
        print(json.dumps({"status": "EXECUTION_ERROR",
                          "reason": "CHECKPOINT_NOT_LOADED", "detail": ck_status},
                         indent=2))
        return 2

    codec = qFHRREpistemicCodec()

    def encode(txt: str) -> torch.Tensor:
        q = codec.encode_text(txt)
        v = q.to(torch.float32)
        return to_flat((v / 255.0) * 2.0 - 1.0)

    # ---- determinism ------------------------------------------------------
    probe_txt = TEMPLATES[0].format(t="topic-000")
    with torch.no_grad():
        l1 = head.forward(encode(probe_txt)).reshape(-1)
        l2 = head.forward(encode(probe_txt)).reshape(-1)
    deterministic = bool(torch.equal(l1, l2))

    ledger = open(rows_path, "w", encoding="utf-8", newline="\n")

    def record(item_id: str, prompt: str, top1: int, ent: float,
               scores: list[float]) -> None:
        ledger.write(json.dumps({
            "item_id": item_id,
            "prompt_sha256": sha256_text_lf(prompt),
            "raw_stdout_sha256": sha256_text_lf(str(top1)),
            "raw_stderr_sha256": sha256_text_lf(""),
            "status": "PASSED",
            "elapsed_ms": 0.0, "tool_calls": 0, "retries": 0,
            "candidate_scores": scores, "sagnac_delta": None,
            "token_top1": int(top1), "token_entropy": round(float(ent), 6),
        }, sort_keys=True) + "\n")
        ledger.flush()

    near_pairs, far_pairs = build_pairs()

    def measure(pairs, tag):
        w_sims, l_sims = [], []
        for i, (pa, pb) in enumerate(pairs):
            wa, wb = encode(pa), encode(pb)
            with torch.no_grad():
                la = head.forward(wa).reshape(-1)
                lb = head.forward(wb).reshape(-1)
            cw, cl = cos(wa, wb), cos(la, lb)
            w_sims.append(cw)
            l_sims.append(cl)
            ent = float(-(torch.log_softmax(la.unsqueeze(0), -1).exp()
                          * torch.log_softmax(la.unsqueeze(0), -1)).sum().item())
            record(f"{tag}-{i:03d}", pa, int(la.argmax().item()), ent,
                   [round(cw, 6), round(cl, 6)])
        return w_sims, l_sims

    t0 = time.perf_counter()
    nearW, nearL = measure(near_pairs, "near")
    farW, farL = measure(far_pairs, "far")

    # ---- MATCHED noise control: identical pair structure, random waves ----
    # near = small angular perturbation of a random base; far = independent draw.
    # Its margin is the FLOOR any content effect must beat at BOTH boundaries.
    # (An earlier revision computed mean(nL)-mean(nL) = 0.0 -- a tautology.)
    g = torch.Generator().manual_seed(SEED + 7)
    nW_near, nW_far, nL_near, nL_far = [], [], [], []
    for _ in range(N_PAIRS):
        base = F.normalize(torch.randn(D, generator=g), p=2, dim=-1)
        pert = F.normalize(base + 0.10 * F.normalize(
            torch.randn(D, generator=g), p=2, dim=-1), p=2, dim=-1)
        other = F.normalize(torch.randn(D, generator=g), p=2, dim=-1)
        with torch.no_grad():
            lb = head.forward(base.reshape(1, D)).reshape(-1)
            lp = head.forward(pert.reshape(1, D)).reshape(-1)
            lo = head.forward(other.reshape(1, D)).reshape(-1)
        nW_near.append(cos(base, pert))
        nW_far.append(cos(base, other))
        nL_near.append(cos(lb, lp))
        nL_far.append(cos(lb, lo))

    M_W = margin(nearW, farW)
    M_L = margin(nearL, farL)
    NOISE_W = margin(nW_near, nW_far)
    NOISE_L = margin(nL_near, nL_far)

    # ---- pre-registered verdict ------------------------------------------
    if not deterministic:
        verdict = "ERROR_NONDETERMINISTIC"
    elif M_W["margin"] <= 0 or not M_W["ci_excludes_zero"]:
        verdict = "INGRESS_LOSES_CONTENT"
    elif M_W["margin"] <= NOISE_W["margin"]:
        verdict = "INGRESS_WEAK"
    elif M_L["margin"] <= NOISE_L["margin"] or not M_L["ci_excludes_zero"]:
        verdict = "EGRESS_DESTROYS_CONTENT"
    else:
        verdict = "EGRESS_CARRIES_CONTENT"

    ledger.close()
    receipt = {
        "schema_id": "henri.kaa-a1-nearfar.v1",
        "amendment": "KAA_AMENDMENT_A1_nearfar_egress.md",
        "supersedes": "K-A in SPEC-2026-09-16-AAII-V43-RD-PIPELINE.json",
        "reason_for_amendment": (
            "K-A counted distinct top-1 tokens; its random control matched the "
            "content arms (37 vs 39 of 128), so that gate was vacuous."),
        "run_id": out.name,
        "commit_sha256": commit,
        "device": device,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "checkpoint": {"status": ck_status, "sha256": ck_sha,
                       "bytes": ck.stat().st_size if ck.exists() else 0},
        "determinism_same_prompt_twice_bit_identical": deterministic,
        "n_pairs": N_PAIRS,
        "boundary_W_wave_ingress": M_W,
        "boundary_L_logit_egress": M_L,
        "noise_control_W": NOISE_W,
        "noise_control_L": NOISE_L,
        "VERDICT": verdict,
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "items_jsonl": str(rows_path),
        "items_sha256": sha256_bytes(rows_path.read_bytes()),
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True),
                                      encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
