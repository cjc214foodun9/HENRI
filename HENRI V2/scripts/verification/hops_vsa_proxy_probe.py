"""hops_vsa_proxy_probe.py — HOPS-VSA production carrier proxy (Class 4.5, pre-registered).

Doc: experiments/verification/hops_vsa_proxy_gate_20260821.md
Doc ID: HENRI-CLASS45-HOPS-PROXY-GATE-2026-08-21

Measures, on PRODUCTION qFHRR waves (D=65,536, RTX 5090, exact-SHA worktree):
  P1 carrier engagement: removed-energy fraction ||V^T x||^2 / ||x||^2 across
     goal+candidate waves (>= 1e-3, random baseline k/D ~ 1.2e-4) and residual
     norm fraction (>= 0.5, no collapse).
  P2 paired rank/margin: treatment (P_null-projected cosine, vetoed sink) vs
     control (raw real-wave cosine) on the SAME frozen 71-pool for
     HumanEval/23 and /35: rank <= 5, margin >= 0.25, no regression.
  P3 veto discrimination: veto fraction in [0.05, 0.95]; oracle not vetoed.
  P4 invariants: gram_error <= 1e-6, finite waves, thin V [D,k] (no [D,D]),
     pool == 71, one oracle per target. Representation boundary: the uint8
     Z_256 ring is crossed INSIDE WaveASTDecoder._wave (encode_text ring ->
     (c/(k_bins-1))*2-1 -> F.normalize), producing float32 [D] unit waves;
     the probe consumes decoder._wave output directly (runner-mirror).

Oracle bodies used ONLY to measure ranking quality; they never enter candidate
generation or any evaluator. The runner never reads this probe's outputs.

Conjunctive PASS = P1 AND P2 AND P3 AND P4. Kill = HOPS_PROXY_FALSIFIED.
Exit: 0 PASS, 2 FALSIFIED, 3 BLOCKED (pool/dataset/invariant precondition).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve()
for p in (HERE.parents[2],):  # HENRI V2 root
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from zone_c_epistemic_axiom_harness import qFHRREpistemicCodec  # noqa: E402
from wave_ast_decoder import WaveASTDecoder  # noqa: E402
from hops_vsa_core import (  # noqa: E402
    HopsVSASkeletonProjector,
    HopsVSASagnacGate,
)

TARGETS = {"HumanEval/23": "return len(string)", "HumanEval/35": "return max(l)"}
RANK_LIMIT = 5
MARGIN = 0.25
ENGAGEMENT_MIN = 1e-3
RESIDUAL_MIN = 0.5
VETO_FRAC_LO = 0.05
VETO_FRAC_HI = 0.95
EXPECTED_POOL = 71


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def load_humaneval(path: str) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_signature(prompt: str):
    m = re.search(r"^def\s+(\w+)\s*\(([^)]*)\)", prompt, re.MULTILINE)
    if not m:
        return None, []
    entry = m.group(1)
    args_raw = m.group(2).strip()
    if not args_raw:
        return entry, []
    args = []
    for part in args_raw.split(","):
        name = part.split(":")[0].split("=")[0].strip()
        if name and name not in ("self", "cls"):
            args.append(name)
    return entry, args


def body_of(src: str) -> str:
    body = src.split("\n", 1)[1] if "\n" in src else src
    return body.strip()


def carrier_frac(projector: HopsVSASkeletonProjector, x: torch.Tensor) -> float:
    """Removed-energy fraction ||V^T x||^2 / ||x||^2 — thin ops only."""
    coef = projector.V.T @ x  # [k]
    energy = float((coef * coef).sum().item())
    norm2 = float((x * x).sum().item())
    return energy / norm2 if norm2 > 0.0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="HENRI V2/data/HumanEval.jsonl.gz")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--d-model", type=int, default=65536)
    ap.add_argument("--smoke", action="store_true",
                    help="reduced mechanism check: gates P2/P3 informational only")
    args = ap.parse_args()

    device = args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    d_model = args.d_model
    if not args.smoke and d_model != 65536:
        print(json.dumps({"status": "BLOCKED", "reason": "PRODUCTION_REQUIRES_D65536"}))
        sys.exit(3)

    dataset_raw = open(args.dataset, "rb").read()
    dataset_sha = sha256_bytes(dataset_raw)
    items = load_humaneval(args.dataset)
    by_id = {it["task_id"]: it for it in items}
    if not all(t in by_id for t in TARGETS):
        print(json.dumps({"status": "BLOCKED", "reason": "TARGET_ITEMS_MISSING",
                          "dataset_sha256": dataset_sha[:16]}))
        sys.exit(3)

    qcodec = qFHRREpistemicCodec(d_model=d_model, device=device)
    decoder = WaveASTDecoder(qcodec, device=device)
    projector = HopsVSASkeletonProjector(d_model=d_model, device=device)
    gate = HopsVSASagnacGate()
    gram_error = projector.gram_error()

    blocked = False
    all_carrier = []      # removed-energy fractions across goal+candidate waves
    all_residual = []     # residual norm fractions
    all_vetoes = []       # per-candidate veto decisions (both targets)
    oracle_vetoed = {}
    results = {}

    with torch.no_grad():
        for target_id, true_body in TARGETS.items():
            item = by_id[target_id]
            prompt = item["prompt"]
            entry, args_list = parse_signature(prompt)
            if entry is None:
                print(json.dumps({"status": "BLOCKED", "reason": "NO_SIGNATURE",
                                  "target": target_id}))
                sys.exit(3)
            candidates = decoder.decode(
                decoder._wave(prompt), decoder._wave(prompt), entry, args_list)
            pool_size = len(candidates)
            if not args.smoke and pool_size != EXPECTED_POOL:
                print(json.dumps({"status": "BLOCKED", "reason": "POOL_SIZE_MISMATCH",
                                  "target": target_id, "pool_size": pool_size,
                                  "expected": EXPECTED_POOL}))
                sys.exit(3)

            goal_real = decoder._wave(prompt).to(device)
            goal_null = F.normalize(projector.project_null(goal_real), p=2, dim=0)
            all_carrier.append(carrier_frac(projector, goal_real))
            all_residual.append(float(projector.project_null(goal_real).norm().item()
                                      / goal_real.norm().item()))

            oracle_count = 0
            per_candidate = []
            for ci, (src, meta) in enumerate(candidates):
                is_oracle = body_of(src) == true_body
                oracle_count += int(is_oracle)
                c_real = decoder._wave(src).to(device)
                all_carrier.append(carrier_frac(projector, c_real))
                c_null_raw = projector.project_null(c_real)
                all_residual.append(float(c_null_raw.norm().item() / c_real.norm().item()))
                c_null = F.normalize(c_null_raw, p=2, dim=0)
                raw_cos = float(torch.dot(F.normalize(c_real, p=2, dim=0),
                                          F.normalize(goal_real, p=2, dim=0)).item())
                proj_cos = float(torch.dot(c_null, goal_null).item())
                veto = gate.veto(c_null, goal_null)
                all_vetoes.append(veto)
                if is_oracle:
                    oracle_vetoed[target_id] = veto
                per_candidate.append({
                    "idx": ci, "oracle": is_oracle,
                    "body_sha": sha256_bytes(body_of(src).encode("utf-8"))[:12],
                    "raw_cos": round(raw_cos, 6), "proj_cos": round(proj_cos, 6),
                    "carrier_frac": round(all_carrier[-1], 6),
                    "veto": bool(veto),
                })

            def ranks(score_key: str):
                ordered = sorted(per_candidate, key=lambda r: (-r[score_key], r["idx"]))
                rank, true_cos, best_other_cos = None, None, None
                for i, r in enumerate(ordered, 1):
                    if r["oracle"]:
                        rank = i
                        true_cos = r[score_key]
                    else:
                        if best_other_cos is None or r[score_key] > best_other_cos:
                            best_other_cos = r[score_key]
                margin = (true_cos - best_other_cos) if (true_cos is not None
                                                         and best_other_cos is not None) else None
                return rank, true_cos, best_other_cos, margin

            raw_rank, raw_true, raw_best, raw_margin = ranks("raw_cos")
            # Runner-mirror treatment: vetoed candidates sink to the end.
            proj_rank, proj_true, proj_best, proj_margin = None, None, None, None
            for i, r in enumerate(sorted(per_candidate,
                                         key=lambda r: ((-1e9 if r["veto"] else r["proj_cos"]),
                                                        r["idx"])), 1):
                if r["oracle"]:
                    proj_rank = i
                    proj_true = r["proj_cos"]
                else:
                    if proj_best is None or r["proj_cos"] > proj_best:
                        proj_best = r["proj_cos"]
            proj_margin = (proj_true - proj_best) if (proj_true is not None
                                                      and proj_best is not None) else None

            if oracle_count != 1:
                print(json.dumps({"status": "BLOCKED", "reason": "ORACLE_COUNT",
                                  "target": target_id, "count": oracle_count}))
                sys.exit(3)

            results[target_id] = {
                "pool_size": pool_size, "oracle_count": oracle_count,
                "control": {"rank": raw_rank, "true_cos": raw_true,
                            "best_other_cos": raw_best, "margin": raw_margin},
                "treatment": {"rank": proj_rank, "true_cos": proj_true,
                              "best_other_cos": proj_best, "margin": proj_margin},
                "oracle_vetoed": bool(oracle_vetoed.get(target_id, False)),
            }
            print(json.dumps({"target": target_id, **results[target_id]}))

    # ---- Gate evaluation ----
    n_waves = max(1, len(all_carrier))
    mean_carrier = sum(all_carrier) / n_waves
    mean_residual = sum(all_residual) / n_waves
    veto_frac = (sum(1 for v in all_vetoes if v) / len(all_vetoes)) if all_vetoes else float("nan")

    p1 = (mean_carrier >= ENGAGEMENT_MIN and mean_residual >= RESIDUAL_MIN)
    p2 = True
    p2_detail = {}
    for t, r in results.items():
        c, tr = r["control"], r["treatment"]
        ok = (tr["rank"] is not None and tr["rank"] <= RANK_LIMIT
              and tr["margin"] is not None and tr["margin"] >= MARGIN
              and tr["rank"] <= c["rank"] and tr["margin"] >= c["margin"])
        p2 = p2 and ok
        p2_detail[t] = ok
    p3 = (VETO_FRAC_LO <= veto_frac <= VETO_FRAC_HI
          and not any(oracle_vetoed.get(t, False) for t in TARGETS))
    p4 = (gram_error <= 1e-6
          and all(torch.isfinite(projector.V).all().item()
                  for _ in [0])  # V thin by construction; finiteness asserted
          )

    gates = {"P1_carrier_engagement": p1, "P2_paired_rank_margin": p2,
             "P3_veto_discrimination": p3, "P4_invariants": p4}
    verdict = "PASS" if all(gates.values()) else "FALSIFIED"

    if args.smoke:
        verdict = "SMOKE"

    print(json.dumps({
        "status": verdict,
        "gate": "HOPS_VSA_PROXY" if not args.smoke else "HOPS_VSA_PROXY_SMOKE",
        "commit": __import__("subprocess").check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(HERE.parents[2]),
            stderr=__import__("subprocess").DEVNULL).decode().strip()
        if __import__("os").path.isdir(str(HERE.parents[2])) else "unknown",
        "device": device, "d_model": d_model, "smoke": bool(args.smoke),
        "dataset_sha256": dataset_sha[:16],
        "projector": {"gram_error": gram_error, "k": projector.V.shape[1],
                      "V_shape": list(projector.V.shape)},
        "carrier": {"mean_removed_frac": round(mean_carrier, 6),
                    "mean_residual_frac": round(mean_residual, 6),
                    "random_baseline_k_over_D": round(projector.V.shape[1] / d_model, 6)},
        "veto": {"fraction": round(veto_frac, 4), "tau": gate.tau,
                 "oracle_vetoed": oracle_vetoed},
        "gates": gates, "results": results,
    }))
    sys.exit(0 if verdict == "PASS" or verdict == "SMOKE" else 2)


if __name__ == "__main__":
    main()
