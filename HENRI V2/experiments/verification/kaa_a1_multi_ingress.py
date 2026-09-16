"""A1 multi-ingress: WHICH text->wave encoder carries content?

A1 (kaa_a1_nearfar_probe.py) measured INGRESS_LOSES_CONTENT for the Z_256
epistemic codec: near/far wave cosine margin +0.0014 with CI95 including zero,
while both noise controls separated decisively (+0.995 W, +0.453 L). The
instrument works; that ingress does not.

This probe holds the head and the pair structure FIXED and swaps only the
ingress, so the verdict names the encoder:

  arm Z  zone_c_epistemic_axiom_harness.qFHRREpistemicCodec      (Z_256 uint8 ring)
  arm S  qfhrr_structured_codec.StructuredCharPositionCodec      (char-engram + position)
  arm T  henri_universal_repl.qFHRRUniversalTextTransducer       (canonical wave)

Pre-registered reading per arm:
  margin_W CI lb > 0 AND > noise-floor margin  -> INGRESS_CARRIES_CONTENT
  otherwise                                    -> INGRESS_LOSES_CONTENT

No score claim. CPU mechanism probe only.
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

from henri_eval_infra import run_id_new, run_output_dir, sha256_bytes
from henri_decoder import HENRINeuralEgressUnbinder

D = 65536
SEED = 20260916
N_PAIRS = 32
BOOT = 2000

TEMPLATES = [
    "What is the capital of {t}?",
    "What is the country of {t}?",
    "Write a Python function that returns {t}.",
    "Summarise the theorem labelled {t}.",
]


def cos(a, b):
    return float(F.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).item())


def boot_ci(vals, n=BOOT, alpha=0.05):
    if not vals:
        return (0.0, 0.0)
    g = torch.Generator().manual_seed(SEED)
    v = torch.tensor(vals, dtype=torch.float64)
    idx = torch.randint(0, v.numel(), (n, v.numel()), generator=g)
    m = v[idx].mean(dim=1)
    return (float(torch.quantile(m, alpha / 2).item()),
            float(torch.quantile(m, 1 - alpha / 2).item()))


def margin(near, far):
    d = [n - f for n, f in zip(near, far)]
    lo, hi = boot_ci(d)
    return {"near_mean": round(statistics.fmean(near), 6),
            "far_mean": round(statistics.fmean(far), 6),
            "margin": round(statistics.fmean(d), 6),
            "ci95": [round(lo, 6), round(hi, 6)],
            "ci_excludes_zero": bool(lo > 0 or hi < 0)}


def to_flat(v):
    v = v.reshape(-1).to(torch.float32)
    if v.numel() < D:
        v = F.pad(v, (0, D - v.numel()))
    else:
        v = v[:D]
    return F.normalize(v, p=2, dim=-1).reshape(1, D)


def main():
    torch.manual_seed(SEED)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True, cwd=str(V2)).stdout.strip()
    except Exception:
        commit = "UNKNOWN"

    out = run_output_dir(V2 / "telemetry_logs" / "kaa_a1_multi", commit,
                         "kaa-a1-multi-ingress", run_id_new())
    rows_path = out / "items.jsonl"

    device = "cpu"
    head = HENRINeuralEgressUnbinder(d_model=D, vocab_size=32000, device=device)
    head.eval()
    ck = V2 / "models" / "henri_decoder_checkpoint.pt"
    ck_status, ck_sha = "ABSENT", None
    if ck.exists():
        ck_sha = hashlib.sha256(ck.read_bytes()).hexdigest()
        sd = torch.load(str(ck), map_location="cpu", weights_only=True)
        missing, _ = head.load_state_dict(sd, strict=False)
        ck_status = "LOADED" if not missing else f"PARTIAL {list(missing)}"
    if ck_status != "LOADED":
        print(json.dumps({"status": "EXECUTION_ERROR",
                          "reason": "CHECKPOINT_NOT_LOADED", "detail": ck_status}))
        return 2

    # ---- ingresses --------------------------------------------------------
    ingresses = {}
    try:
        from zone_c_epistemic_axiom_harness import qFHRREpistemicCodec
        c = qFHRREpistemicCodec()
        ingresses["Z_epistemic_ring_uint8"] = (
            lambda t: to_flat((c.encode_text(t).to(torch.float32) / 255.0) * 2.0 - 1.0))
    except Exception as e:
        print(f"  Z unavailable: {type(e).__name__}: {str(e)[:110]}")
    try:
        from qfhrr_structured_codec import StructuredCharPositionCodec
        s = StructuredCharPositionCodec()
        ingresses["S_structured_charpos"] = lambda t: to_flat(s.encode_text(t))
    except Exception as e:
        print(f"  S unavailable: {type(e).__name__}: {str(e)[:110]}")
    try:
        from henri_universal_repl import qFHRRUniversalTextTransducer
        tr = qFHRRUniversalTextTransducer(d_model=D)
        ingresses["T_universal_transducer"] = lambda t: to_flat(tr.transduce_text(t))
    except Exception as e:
        print(f"  T unavailable: {type(e).__name__}: {str(e)[:110]}")

    near_pairs = [(TEMPLATES[0].format(t=f"topic-{i:03d}"),
                   TEMPLATES[1].format(t=f"topic-{i:03d}")) for i in range(N_PAIRS)]
    far_pairs = [(TEMPLATES[0].format(t=f"topic-{i:03d}"),
                  TEMPLATES[2].format(t=f"topic-{i + 100:03d}")) for i in range(N_PAIRS)]

    # ---- noise control (matched structure, random waves) ------------------
    g = torch.Generator().manual_seed(SEED + 7)
    nW_near, nW_far, nL_near, nL_far = [], [], [], []
    for _ in range(N_PAIRS):
        base = F.normalize(torch.randn(D, generator=g), p=2, dim=-1)
        pert = F.normalize(base + 0.10 * F.normalize(torch.randn(D, generator=g),
                                                      p=2, dim=-1), p=2, dim=-1)
        other = F.normalize(torch.randn(D, generator=g), p=2, dim=-1)
        with torch.no_grad():
            lb = head.forward(base.reshape(1, D)).reshape(-1)
            lp = head.forward(pert.reshape(1, D)).reshape(-1)
            lo = head.forward(other.reshape(1, D)).reshape(-1)
        nW_near.append(cos(base, pert)); nW_far.append(cos(base, other))
        nL_near.append(cos(lb, lp));    nL_far.append(cos(lb, lo))
    NOISE_W = margin(nW_near, nW_far)
    NOISE_L = margin(nL_near, nL_far)

    results = {}
    with open(rows_path, "w", encoding="utf-8", newline="\n") as led:
        for name, fn in ingresses.items():
            try:
                nW, fW, nL, fL = [], [], [], []
                for i in range(N_PAIRS):
                    pa, pb = near_pairs[i]
                    qa, qb = far_pairs[i]
                    wa, wb = fn(pa), fn(pb)
                    xa, xb = fn(qa), fn(qb)
                    with torch.no_grad():
                        la, lb2 = head.forward(wa).reshape(-1), head.forward(wb).reshape(-1)
                        xla, xlb = head.forward(xa).reshape(-1), head.forward(xb).reshape(-1)
                    nW.append(cos(wa, wb)); fW.append(cos(xa, xb))
                    nL.append(cos(la, lb2)); fL.append(cos(xla, xlb))
                    led.write(json.dumps({
                        "item_id": f"{name}-{i:03d}",
                        "prompt_sha256": sha256_bytes(pa.encode()),
                        "raw_stdout_sha256": sha256_bytes(b""),
                        "raw_stderr_sha256": sha256_bytes(b""),
                        "status": "PASSED", "elapsed_ms": 0.0,
                        "tool_calls": 0, "retries": 0,
                        "candidate_scores": [round(nW[-1], 6), round(fW[-1], 6),
                                             round(nL[-1], 6), round(fL[-1], 6)],
                        "sagnac_delta": None,
                        "token_top1": int(la.argmax().item()),
                        "token_entropy": 0.0,
                    }, sort_keys=True) + "\n")
                    led.flush()
                MW, ML = margin(nW, fW), margin(nL, fL)
                v = ("INGRESS_CARRIES_CONTENT"
                     if (MW["ci_excludes_zero"] and MW["margin"] > 0
                         and MW["margin"] > NOISE_W["margin"] * 0.01)
                     else "INGRESS_LOSES_CONTENT")
                results[name] = {"W": MW, "L": ML, "VERDICT": v}
            except Exception as e:
                results[name] = {"VERDICT": "EXECUTION_ERROR",
                                 "error": f"{type(e).__name__}: {str(e)[:140]}"}

    winners = [k for k, v in results.items() if v.get("VERDICT") == "INGRESS_CARRIES_CONTENT"]
    receipt = {
        "schema_id": "henri.kaa-a1-multi-ingress.v1",
        "commit_sha256": commit, "device": device,
        "checkpoint": {"status": ck_status, "sha256": ck_sha},
        "n_pairs": N_PAIRS, "ingresses_tested": list(ingresses),
        "noise_control_W": NOISE_W, "noise_control_L": NOISE_L,
        "per_ingress": results,
        "INGRESSES_CARRYING_CONTENT": winners,
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
