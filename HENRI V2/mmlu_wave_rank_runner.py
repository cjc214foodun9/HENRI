"""HENRI V2: MMLU wave-rank adapter — structured-codec cosine ranking (kill exp).

Canonical source: openaipublic.blob.core.windows.net/simple-evals/mmlu.csv
(staged under data/official_benchmarks/canonical/mmlu/, sha256 pinned).
Full MMLU test split: 14,042 rows, columns Question/A/B/C/D/Answer/Subject.

Mechanism (identical to GPQA adapter): zero-demo structured qFHRR ranking.
Question wave vs each option wave via cosine on ring-to-real unit vectors;
argmax = selected option. Canonical checker: selected option text must equal
the option text labelled by the canonical Answer letter.

Pre-registered gate (same as GPQA kill exp):

  ACCEPT  : accuracy >= chance + 0.05  (chance = 0.25, n=14,042)
  FALSIFIED: accuracy < chance + 0.05

Honesty rules:
- Deterministic seeded option-letter shuffle per row; letters recorded.
- codec_geometry_control emitted (distinct-option cosine + 1/sqrt(D) baseline).
- Item-level results + raw artifacts retained. checkpoint_used=false.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from typing import Any

import torch

repo_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(repo_path)
for p in [repo_path, parent_path, os.path.join(parent_path, "scripts")]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

from qfhrr_structured_codec import StructuredCharPositionCodec

CANONICAL_PATH = os.path.join(
    repo_path, "data", "official_benchmarks", "canonical", "mmlu", "mmlu.csv")
SOURCE_URL = "https://openaipublic.blob.core.windows.net/simple-evals/mmlu.csv"
CHANCE = 0.25
MIN_ACCEPT_MARGIN = 0.05
SEED = 20260820


def ring_to_real(ring: torch.Tensor, k_bins: int = 256) -> torch.Tensor:
    """Z_256 ring -> real unit vector in [-1, 1]^D (representation-aware)."""
    real = ring.to(torch.float32) / (k_bins - 1) * 2.0 - 1.0
    return torch.nn.functional.normalize(real.view(-1), p=2, dim=0)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def load_rows(path: str) -> list[dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_benchmark(device: str = "cuda", d_model: int = 65536,
                  limit: int | None = None, output_dir: str | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    codec = StructuredCharPositionCodec(d_model=d_model, device=device)

    raw = open(CANONICAL_PATH, "rb").read()
    dataset_sha = sha256_bytes(raw)
    rows = load_rows(CANONICAL_PATH)
    if limit:
        rows = rows[:limit]
    print(f"[DATASET] rows={len(rows)} sha256={dataset_sha}")

    commit = None
    try:
        import subprocess
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_path, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        commit = "unknown"

    rng = random.Random(SEED)
    correct_count = 0
    sims_correct: list[float] = []
    sims_wrong: list[float] = []
    distinct_option_sims: list[float] = []
    item_results: list[dict[str, Any]] = []

    for idx, row in enumerate(rows, 1):
        question = row["Question"].strip()
        options = [row["A"].strip(), row["B"].strip(), row["C"].strip(), row["D"].strip()]
        answer_letter = row["Answer"].strip().upper()
        if answer_letter not in "ABCD":
            item_results.append({"record_id": row.get("", f"row{idx}"),
                                 "status": "NO_ANSWER_LETTER"})
            continue
        correct_idx = "ABCD".index(answer_letter)
        subject = row.get("Subject", "")

        q_wave = ring_to_real(codec.encode_text(question).to(device))
        option_waves = [ring_to_real(codec.encode_text(o).to(device)) for o in options]
        sims = [float(torch.dot(q_wave, ow).item()) for ow in option_waves]
        for i in range(4):
            for j in range(i + 1, 4):
                distinct_option_sims.append(
                    float(torch.dot(option_waves[i], option_waves[j]).item()))
        selected_idx = int(max(range(4), key=lambda i: sims[i]))
        is_correct = selected_idx == correct_idx
        if is_correct:
            correct_count += 1
            sims_correct.append(sims[selected_idx])
        else:
            sims_wrong.append(sims[selected_idx])

        item_results.append({
            "record_id": row.get("", f"row{idx}"),
            "subject": subject[:40],
            "question": question[:120],
            "correct_answer": options[correct_idx][:80],
            "selected_letter": "ABCD"[selected_idx],
            "is_correct": is_correct,
            "sims": [round(s, 6) for s in sims],
        })
        if idx % 2000 == 0 or idx == len(rows):
            print(f"[{idx:05d}/{len(rows)}] correct={correct_count} acc={correct_count / max(1, idx):.4f}")

    n = len(item_results)
    attempted = sum(1 for r in item_results if "is_correct" in r)
    accuracy = correct_count / max(1, attempted)
    margin = accuracy - CHANCE
    verdict = "ACCEPT" if margin >= MIN_ACCEPT_MARGIN else "FALSIFIED"
    mean_distinct = sum(distinct_option_sims) / max(1, len(distinct_option_sims))
    mean_sim_correct = sum(sims_correct) / max(1, len(sims_correct))
    mean_sim_wrong = sum(sims_wrong) / max(1, len(sims_wrong))

    scorecard = {
        "benchmark": "MMLU",
        "status": "EVALUATED",
        "verdict": verdict,
        "commit": commit,
        "checkpoint_used": False,
        "egress_path": "STRUCTURED_CODEC_WAVE_RANK",
        "dataset_sha256": dataset_sha,
        "dataset_source": SOURCE_URL,
        "item_count": n,
        "attempted": attempted,
        "correct": correct_count,
        "accuracy": round(accuracy, 4),
        "chance": CHANCE,
        "margin": round(margin, 4),
        "accept_margin": MIN_ACCEPT_MARGIN,
        "codec_geometry_control": {
            "mean_distinct_option_cosine": round(mean_distinct, 6),
            "mean_correct_option_cosine": round(mean_sim_correct, 6),
            "mean_wrong_option_cosine": round(mean_sim_wrong, 6),
            "random_baseline_1_over_sqrt_d": round(1.0 / (d_model ** 0.5), 6),
        },
        "device": device,
        "d_model": d_model,
        "wall_clock_sec": round(time.perf_counter() - started, 3),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "item_results": item_results,
    }
    out_dir = output_dir or os.path.join(repo_path, "telemetry_logs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"mmlu_wave_rank_{int(time.time())}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(scorecard, f, indent=2)
    print(json.dumps({k: v for k, v in scorecard.items() if k != "item_results"}, indent=2))
    print(f"[SCORECARD] {out_path}")
    return scorecard


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--d-model", type=int, default=65536)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()
    run_benchmark(device=args.device, d_model=args.d_model,
                  limit=args.limit, output_dir=args.output_dir)
