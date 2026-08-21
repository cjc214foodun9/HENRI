"""HENRI V2: HumanEval authentic runner via the proven wave-AST egress path.

Scoring mechanism (the only egress that has produced real external passes on
this codebase, MBPP 11-17/500): bounded AST grammar enumeration under the
item's own signature, transformation-relative wave ranking (weak without
in-context demos, so verification order falls back to grammar order), and
verification against the OFFICIAL HumanEval test code in the sandbox.

Honesty rules:
- The trained decoder checkpoint is NOT loaded: this path scores by grammar +
  sandbox, not by token decode. checkpoint_used=false is recorded.
- Items whose signature the grammar cannot express (0 args, >=5 args with
  unsupported bodies) are recorded NOT_EXPRESSIBLE, never FAIL.
- Every item result, the dataset digest, commit, and raw log are retained.
- A 0-pass run with valid infrastructure is a valid negative result.
"""

from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

repo_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(repo_path)
for p in [repo_path, parent_path, os.path.join(parent_path, "scripts")]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

from mbpp_secure_executor import SecurePythonSandbox
from wave_ast_decoder import WaveASTDecoder
from zone_c_epistemic_axiom_harness import qFHRREpistemicCodec

HUMANEVAL_URL = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
SIG_RE = re.compile(r"^def\s+(\w+)\s*\(([^)]*)\)", re.MULTILINE)
SHIM = "from typing import *\n"

CANDIDATE_ATTEMPTS = int(os.environ.get("HENRI_HE_CANDIDATES", "12"))
MIN_CANDIDATES = int(os.environ.get("HENRI_HE_MIN_CANDIDATES", "4"))


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def parse_signature(prompt: str) -> tuple[str | None, list[str]]:
    m = SIG_RE.search(prompt)
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


def run_benchmark(limit: int = 50, attempts: int = CANDIDATE_ATTEMPTS,
                  device: str = "cuda", smoke_dim: int | None = None,
                  output_dir: str | None = None,
                  reward_rank: bool = False,
                  decoder_rank: bool = False,
                  spec_rank: bool = False,
                  trained_rank: bool = False,
                  trained_head_path: str | None = None,
                  ast_idf_only: bool = False,
                  ast_idf_batched: bool = False,
                  path_b2_codec: bool = False,
                  hops_vsa_rank: bool = False) -> dict[str, Any]:
    from accuracy_profile import (
        FIDELITY_MIGRATION_FLAG,
        FidelityGuardError,
        enabled_sealed_levers,
        fidelity_migration_enabled,
        is_score_promotable,
        runner_execution_profile,
    )
    started = time.perf_counter()
    d_model = smoke_dim or 65536
    device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"

    # Accuracy-first fidelity contract (Class 4 synthesis): the sealed
    # ranking-lever class is CLOSED (2026-08-20 — Gate A' did not transfer;
    # Gate B 2/50 baseline; bottleneck is grammar expressiveness, not
    # candidate order). Under HENRI_ACCURACY_FIRST_CLASS4 any sealed lever
    # fails closed before dataset load. Legacy default (flag OFF) preserves
    # prior behavior byte-for-byte; the profile is recorded as telemetry.
    sealed_active = enabled_sealed_levers({
        "reward_rank": reward_rank,
        "decoder_rank": decoder_rank,
        "spec_rank": spec_rank,
        "trained_rank": trained_rank,
        "ast_idf_only": ast_idf_only,
    })
    if fidelity_migration_enabled() and sealed_active:
        raise FidelityGuardError(
            "Sealed ranking levers active under "
            f"{FIDELITY_MIGRATION_FLAG}: "
            + ", ".join(sealed_active)
            + ". Reopen only with a new semantic representation and a new "
              "pre-registered kill gate.")
    execution_profile = runner_execution_profile(
        sealed_lever_enabled=bool(sealed_active))
    score_promotable = is_score_promotable(execution_profile)

    # 1. Load the OFFICIAL dataset (single source, pinned URL).
    cache_path = os.path.join(repo_path, "data", "HumanEval.jsonl.gz")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if not os.path.exists(cache_path):
        import urllib.request
        print(f"[DATASET] Downloading {HUMANEVAL_URL}")
        urllib.request.urlretrieve(HUMANEVAL_URL, cache_path)
    raw = open(cache_path, "rb").read()
    dataset_sha = sha256_bytes(raw)
    items: list[dict[str, Any]] = []
    with gzip.open(cache_path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
            if len(items) >= limit:
                break
    print(f"[DATASET] items={len(items)} sha256={dataset_sha}")

    # 2. Sandbox: fail-closed subprocess executor with per-candidate timeout.
    codec = qFHRREpistemicCodec(d_model=d_model, device=device)
    decoder = WaveASTDecoder(codec, device=device)
    sandbox = None
    for mode in ("namespace", "container-rlimit"):
        try:
            candidate = SecurePythonSandbox(timeout_sec=8.0, mode=mode)
            probe = candidate.execute("x = 41 + 1\nassert x == 42\n")
            if probe.status == "PASS":
                sandbox = candidate
                print(f"[SANDBOX] mode={mode} preflight PASS")
                break
        except Exception as e:
            print(f"[SANDBOX] mode={mode} unavailable: {str(e)[:120]}")
    if sandbox is None:
        return {"status": "BLOCKED", "reason": "SANDBOX_PREFLIGHT_FAILED",
                "dataset_sha256": dataset_sha}

    # 2b. Trained-decoder ranking head (8.39, default-OFF): the trained
    # 65,536 -> 2048 -> 32,000 unbinder scores candidate waves by
    # token-predictability (softmax entropy). This is a LEARNED prior over
    # code-wave geometry, distinct from token-decode generation (falsified).
    unbinder = None
    decoder_checkpoint_sha = None
    if decoder_rank:
        if d_model != 65536:
            return {"status": "BLOCKED", "reason": "DECODER_RANK_REQUIRES_D65536",
                    "dataset_sha256": dataset_sha}
        checkpoint_path = os.environ.get(
            "HENRI_DECODER_CHECKPOINT") or os.path.join(
            repo_path, "models", "henri_decoder_checkpoint.pt")
        if not os.path.exists(checkpoint_path):
            return {"status": "BLOCKED", "reason": "DECODER_CHECKPOINT_MISSING",
                    "dataset_sha256": dataset_sha}
        from henri_decoder import HENRINeuralEgressUnbinder
        decoder_checkpoint_sha = sha256_bytes(
            open(checkpoint_path, "rb").read())
        unbinder = HENRINeuralEgressUnbinder(
            d_model=65536, d_hidden=2048, vocab_size=32000, device=device)
        raw = torch.load(checkpoint_path, map_location="cpu")
        sd = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
        missing, unexpected = unbinder.load_state_dict(sd, strict=True)
        if missing or unexpected:
            return {"status": "BLOCKED", "reason": "DECODER_CHECKPOINT_MISMATCH",
                    "missing": missing, "unexpected": unexpected,
                    "dataset_sha256": dataset_sha}
        unbinder.eval()
        print(f"[DECODER-RANK] loaded checkpoint sha={decoder_checkpoint_sha[:16]}")

    # 2c. Execution-grounded correctness head (8.39-V3, default-OFF):
    # linear probe over the decoder's relative-direction feature, trained on
    # MBPP (a DIFFERENT benchmark; HumanEval stays unseen). Checkpoint
    # contract: {"w": [D] float32, "provenance": {...}}.
    trained_w = None
    trained_head_sha = None
    if trained_rank:
        hp = trained_head_path or os.environ.get("HENRI_TRAINED_HEAD")
        if not hp or not os.path.exists(hp):
            return {"status": "BLOCKED", "reason": "TRAINED_HEAD_MISSING",
                    "dataset_sha256": dataset_sha}
        head_raw = torch.load(hp, map_location="cpu")
        w = head_raw["w"].to(torch.float32)
        if w.numel() != d_model:
            return {"status": "BLOCKED", "reason": "TRAINED_HEAD_DIM_MISMATCH",
                    "expected": d_model, "got": w.numel(),
                    "dataset_sha256": dataset_sha}
        trained_w = F.normalize(w.view(-1), p=2, dim=0).to(device)
        trained_head_sha = sha256_bytes(open(hp, "rb").read())
        print(f"[TRAINED-RANK] loaded head sha={trained_head_sha[:16]} "
              f"val_acc={head_raw.get('val_acc')} source={head_raw.get('provenance', {}).get('source')}")

    # 2d. Gate A' IDF-only representation (8.39, default-OFF): rank grammar
    # candidates by mean raw phase-cosine vs an IDF-weighted MBPP codebook
    # attractor bank (carrier subtraction DISABLED per spec
    # HENRI-SPEC-GATE-A-PRIME-IDF-2026). MBPP is a DIFFERENT benchmark;
    # HumanEval stays unseen. Proxy Gate A' passed (ranks 3/5 at d=2048).
    ast_idf_encoder = None
    ast_idf_codebook: list[torch.Tensor] = []
    ast_idf_codebook_sha = None
    if ast_idf_only:
        from qfhrr_ast_discriminative_kernel import (  # noqa: E402
            ASTDiscriminativeEncoder, build_idf_frequencies,
            batched_mean_phase_cosine)
        mbpp_path = os.path.join(repo_path, "data", "mbpp.jsonl")
        if not os.path.exists(mbpp_path):
            return {"status": "BLOCKED", "reason": "AST_IDF_MBPP_MISSING",
                    "dataset_sha256": dataset_sha}
        mbpp_codes = []
        with open(mbpp_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                code = item.get("code", item.get("solution", ""))
                if isinstance(code, str) and code.strip():
                    mbpp_codes.append(code)
        freqs, corpus_size = build_idf_frequencies(mbpp_codes)
        ast_idf_encoder = ASTDiscriminativeEncoder(
            d_model=d_model, device=device, idf_weighting=True,
            carrier_subtract=False, node_frequencies=freqs,
            corpus_size=corpus_size)
        cb_vecs = [ast_idf_encoder.encode_code_string(c)
                   for c in mbpp_codes[:100]]
        ast_idf_codebook = [v for v in cb_vecs if v is not None]
        if not ast_idf_codebook:
            return {"status": "BLOCKED", "reason": "AST_IDF_CODEBOOK_EMPTY",
                    "dataset_sha256": dataset_sha}
        ast_idf_codebook_sha = sha256_bytes(open(mbpp_path, "rb").read())
        print(f"[AST-IDF] encoder ready d={d_model} "
              f"codebook={len(ast_idf_codebook)} mbpp_sha={ast_idf_codebook_sha[:16]}")

    # Path B2 hard-negative codec (Class 4.4, default-OFF): trained
    # PathB2DiscriminativeCodec checkpoint must be an EXACT d_model match;
    # missing/mismatched -> fail-closed BLOCKED (never silent fallback).
    path_b2_codec_model = None
    path_b2_codec_sha = None
    if path_b2_codec:
        from path_b2_semantic_codec import PathB2DiscriminativeCodec
        ckpt_path = os.path.join(repo_path, "models", "path_b2_codec.pt")
        if not os.path.exists(ckpt_path):
            return {"status": "BLOCKED", "reason": "PATH_B2_CHECKPOINT_MISSING",
                    "dataset_sha256": dataset_sha}
        ckpt_raw = open(ckpt_path, "rb").read()
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if ckpt.get("d_model") != d_model:
            return {"status": "BLOCKED",
                    "reason": f"PATH_B2_DMODEL_MISMATCH:{ckpt.get('d_model')}",
                    "dataset_sha256": dataset_sha}
        path_b2_codec_model = PathB2DiscriminativeCodec(
            d_model=d_model, d_latent=ckpt.get("d_latent", 512),
            vocab=ckpt.get("vocab"), df=ckpt.get("df"),
            n_docs=ckpt.get("n_docs", 1000), device=device, seed=7)
        path_b2_codec_model.load_state_dict(ckpt["state_dict"], strict=True)
        path_b2_codec_model.eval()
        path_b2_codec_model = path_b2_codec_model.to(device)
        path_b2_codec_sha = sha256_bytes(ckpt_raw)
        print(f"[PATH-B2] codec ready d={d_model} ckpt_sha={path_b2_codec_sha[:16]} "
              f"val_acc={ckpt.get('val_contrastive_acc')}")

    # HOPS-VLA reference core (Class 4.5, default-OFF): invariant-subspace
    # decoupling (P_null = I - V V^dagger over the AST skeleton basis), diagonal
    # Clifford rotor, dual-channel Sagnac veto. Python reference core; the
    # fused CUDA kernel (hops_vla_cuda_core.cu) is a separate gated phase.
    hops_vsa_scorer = None
    if hops_vsa_rank:
        from hops_vsa_core import (
            HopsVSACandidateScorer, HopsVSASkeletonProjector, ring_to_real_wave)
        hops_vsa_scorer = HopsVSACandidateScorer(
            HopsVSASkeletonProjector(d_model=d_model, device=device))
        print(f"[HOPS-VSA] scorer ready d={d_model} "
              f"skel_gram={hops_vsa_scorer.projector.gram_error():.2e}")

    item_results: list[dict[str, Any]] = []
    solved = 0
    not_expressible = 0
    infra_errors = 0
    expressible = 0
    total_candidates = 0
    latencies: list[float] = []
    # Test-time learned positive-exemplar prior (reward-shaped ranking):
    # verified solutions seed (prompt_wave, transformation-relative solution
    # wave) exemplars; subsequent candidate order is re-ranked by similarity
    # to the nearest exemplar direction. Default-OFF; seeded only from
    # favorable (sandbox-verified) outcomes.
    exemplars: list[tuple[torch.Tensor, torch.Tensor]] = []
    items_reordered = 0
    decoder_items_reordered = 0
    docstring_used = 0
    trained_items_reordered = 0
    ast_idf_items_reordered = 0
    path_b2_items_reordered = 0
    hops_vsa_items_reordered = 0
    commit = None
    try:
        import subprocess
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_path, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        commit = "unknown"

    for idx, item in enumerate(items, 1):
        task_id = item["task_id"]
        prompt = item["prompt"]
        test_code = item["test"]
        entry_point = item["entry_point"]
        t0 = time.perf_counter()

        entry, args = parse_signature(prompt)
        if entry is None:
            item_results.append({"task_id": task_id, "status": "NOT_EXPRESSIBLE",
                                 "reason": "NO_SIGNATURE"})
            not_expressible += 1
            continue

        prompt_wave = decoder._wave(prompt)
        # V1 remedy (HENRI-SYNTHESIS-PHASE0-AUDIT-2026, Stage 1): replace the
        # degenerate decode(prompt, prompt) zero-target call with a non-zero
        # target wave derived from the item's own docstring (legitimate input;
        # NOT the test field — test-derived targets would be answer leakage).
        doc_target = None
        if spec_rank:
            import re as _re
            m = _re.search(r'"""(.*?)"""', prompt, _re.DOTALL)
            doc_target = m.group(1).strip() if m else None
            if doc_target:
                target_wave = decoder._wave(doc_target)
            else:
                target_wave = prompt_wave
        else:
            target_wave = prompt_wave
        candidates = decoder.decode(prompt_wave, target_wave, entry, args)
        docstring_used += 1 if (spec_rank and doc_target) else 0
        total_candidates += len(candidates)

        # Gate A' IDF-only ranking (8.39, default-OFF): reorder grammar
        # candidates by mean raw phase-cosine vs the IDF-weighted MBPP
        # codebook bank (same representation as the Gate A' proxy). The
        # candidate SET is unchanged; only attempt order moves.
        prev_first = candidates[0][0] if candidates else None
        if ast_idf_only and ast_idf_encoder is not None:
            if ast_idf_batched and candidates:
                # Phase 3 batched path: one stacked matmul-style scoring
                # pass instead of C x N per-candidate cosine calls.
                valid_idx, valid_vecs = [], []
                for ci, (src, meta) in enumerate(candidates):
                    v = ast_idf_encoder.encode_code_string(src)
                    if v is None:
                        continue
                    valid_idx.append(ci)
                    valid_vecs.append(v)
                if valid_vecs:
                    cand_mat = torch.stack(valid_vecs)           # [C, D] uint8
                    cb_mat = torch.stack(ast_idf_codebook)       # [N, D] uint8
                    scores = batched_mean_phase_cosine(
                        cand_mat, cb_mat).tolist()               # [C]
                    scored = [(-1e9, ci, src, meta)
                              for ci, (src, meta) in enumerate(candidates)]
                    for ci, s in zip(valid_idx, scores):
                        scored[ci] = (s, ci,
                                      candidates[ci][0], candidates[ci][1])
                    scored.sort(key=lambda t: (-t[0], t[1]))
                    candidates = [(src, meta) for _, _, src, meta in scored]
                    if candidates[0][0] != prev_first:
                        ast_idf_items_reordered += 1
            else:
                scored = []
                for ci, (src, meta) in enumerate(candidates):
                    v = ast_idf_encoder.encode_code_string(src)
                    if v is None:
                        scored.append((-1e9, ci, src, meta))
                        continue
                    s = 0.0
                    for cb in ast_idf_codebook:
                        s += ast_idf_encoder.compute_cosine_similarity(v, cb)
                    scored.append((s / max(1, len(ast_idf_codebook)), ci, src, meta))
                scored.sort(key=lambda t: (-t[0], t[1]))
                candidates = [(src, meta) for _, _, src, meta in scored]
                if candidates[0][0] != prev_first:
                    ast_idf_items_reordered += 1

        # Path B2 hard-negative codec ranking (Class 4.4, default-OFF): reorder
        # grammar candidates by PathB2DiscriminativeCodec cosine vs the prompt
        # (goal) wave. New semantic representation with pre-registered Gate A/B
        # (experiments/verification/class4_path_b2_design.md). Candidate SET is
        # unchanged; only attempt order moves.
        prev_first = candidates[0][0] if candidates else None
        if path_b2_codec and path_b2_codec_model is not None:
            goal_wave = path_b2_codec_model.encode_sequence(prompt).to(device)
            scored = []
            for ci, (src, meta) in enumerate(candidates):
                v = path_b2_codec_model.encode_sequence(src).to(device)
                scored.append((float(
                    path_b2_codec_model.cosine_similarity(goal_wave, v).item()),
                    ci, src, meta))
            scored.sort(key=lambda t: (-t[0], t[1]))
            candidates = [(src, meta) for _, _, src, meta in scored]
            if candidates[0][0] != prev_first:
                path_b2_items_reordered += 1

        # HOPS-VLA skeleton-free channel ranking (Class 4.5, default-OFF):
        # reorder grammar candidates by P_null-projected cosine vs the prompt
        # (goal) wave; Sagnac-vetoed candidates sink to the end. Candidate SET
        # is unchanged; only attempt order moves. Ring waves cross the boundary
        # via ring_to_real_wave (explicit, never silent).
        prev_first = candidates[0][0] if candidates else None
        if hops_vsa_rank and hops_vsa_scorer is not None:
            from hops_vsa_core import ring_to_real_wave
            goal_cont = ring_to_real_wave(decoder._wave(prompt))
            scored = []
            for ci, (src, meta) in enumerate(candidates):
                cand_cont = ring_to_real_wave(decoder._wave(src))
                cos, veto = hops_vsa_scorer.score(goal_cont, cand_cont)
                scored.append((-1e9 if veto else cos, ci, src, meta))
            scored.sort(key=lambda t: (-t[0], t[1]))
            candidates = [(src, meta) for _, _, src, meta in scored]
            if candidates[0][0] != prev_first:
                hops_vsa_items_reordered += 1

        # Execution-grounded correctness head (8.39-V3, default-OFF): rank
        # candidates by <w_trained, v_rel(candidate)> — the linear probe
        # trained on execution labels from MBPP (unseen benchmark stays
        # HumanEval). Same relative-direction feature as decode().
        prev_first = candidates[0][0] if candidates else None
        if trained_rank and trained_w is not None:
            scored = []
            for ci, (src, meta) in enumerate(candidates):
                v = decoder._wave(src)
                v_rel = v - prompt_wave * torch.dot(v, prompt_wave).clamp(min=0.0)
                v_rel = torch.nn.functional.normalize(v_rel, p=2, dim=0)
                scored.append((float(torch.dot(v_rel, trained_w).item()), ci, src, meta))
            scored.sort(key=lambda t: (-t[0], t[1]))
            candidates = [(src, meta) for _, _, src, meta in scored]
            if candidates[0][0] != prev_first:
                trained_items_reordered += 1
        if len(candidates) < MIN_CANDIDATES:
            item_results.append({"task_id": task_id, "status": "NOT_EXPRESSIBLE",
                                 "reason": f"GRAMMAR_UNDERFLOW:{len(candidates)}"})
            not_expressible += 1
            continue
        expressible += 1

        # Trained-decoder ranking head (8.39, default-OFF): reorder the
        # grammar candidates by token-predictability under the TRAINED
        # 65,536 -> 2048 -> 32,000 unbinder. Low softmax entropy = the
        # candidate wave maps to a peaked, code-like token distribution
        # under the learned prior. Batched single forward. No tokenizer,
        # no token decode; ranking only.
        prev_first = candidates[0][0]
        if decoder_rank and unbinder is not None:
            with torch.no_grad():
                cand_waves = torch.stack(
                    [decoder._wave(src).to(torch.float32) for src, _ in candidates],
                    dim=0)  # [N, D]
                logits = unbinder(cand_waves)  # [N, 32000]
                probs = torch.softmax(logits, dim=-1)
                ent = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
            order = sorted(range(len(candidates)), key=lambda i: (-ent[i].item(), i))
            candidates = [candidates[i] for i in order]
            if candidates[0][0] != prev_first:
                decoder_items_reordered += 1

        # Reward-shaped re-ranking (8.39, default-OFF): reorder the grammar
        # candidates by transformation-relative similarity to the nearest
        # verified-exemplar direction. The candidate SET is unchanged; only
        # attempt order moves. No pretraining, no dataset leakage: exemplars
        # are seeded exclusively from sandbox-verified favorable outcomes in
        # this same run.
        if reward_rank and exemplars and not (decoder_rank and unbinder is not None):
            scored_ordered = []
            for cand_idx, (src, meta) in enumerate(candidates):
                v = decoder._wave(src)
                v_rel = v - prompt_wave * torch.dot(v, prompt_wave).clamp(min=0.0)
                v_rel = torch.nn.functional.normalize(v_rel, p=2, dim=0)
                best = 0.0
                for _, ex_dir in exemplars:
                    d = float(torch.dot(v_rel, ex_dir).item())
                    if d > best:
                        best = d
                scored_ordered.append((cand_idx, best, src, meta))
            candidates = [
                (src, meta) for _, _, src, meta in
                sorted(scored_ordered, key=lambda t: (-t[1], t[0]))]
            if candidates[0][0] != prev_first:
                items_reordered += 1

        passed = False
        outcome = None
        for src, meta in candidates[:attempts]:
            body = src.split("\n", 1)[1] if "\n" in src else src
            full = SHIM + prompt.rstrip() + "\n" + body + "\n" + test_code + f"\ncheck({entry_point})"
            try:
                res = sandbox.execute(full)
            except Exception as e:  # sandbox launcher-level failure
                infra_errors += 1
                outcome = {"attempted": True, "infra": str(e)[:200]}
                break
            if res.status == "EXECUTION_ERROR":
                infra_errors += 1
                outcome = {"attempted": True, "infra": res.stderr[:200]}
                break
            if res.status == "PASS":
                passed = True
                outcome = {"attempted": True, "pass": True,
                           "body": body.strip()[:120]}
                if reward_rank:
                    # Seed the positive-exemplar prior (bounded; recent
                    # favorable outcomes dominate).
                    exemplars.append((prompt_wave.clone(), decoder._wave(src)))
                    if len(exemplars) > 8:
                        exemplars.pop(0)
                break
            outcome = {"attempted": True, "pass": False,
                       "trace": res.stderr[:200]}
        if passed:
            solved += 1
        latencies.append((time.perf_counter() - t0) * 1000.0)
        item_results.append({"task_id": task_id, "status": "PASS" if passed else "FAIL",
                             "candidates_generated": len(candidates),
                             "candidates_attempted": min(len(candidates), attempts),
                             "outcome": outcome})
        print(f"[{idx:03d}/{len(items)}] {task_id} -> {'PASS' if passed else 'FAIL'} "
              f"(gen={len(candidates)})")

    wall_sec = time.perf_counter() - started
    avg_ms = sum(latencies) / max(1, len(latencies))
    scorecard = {
        "benchmark": "HumanEval",
        "status": "EVALUATED",
        "commit": commit,
        "checkpoint_used": False,
        "egress_path": "WAVE_AST_GRAMMAR_SANDBOX",
        "execution_profile": execution_profile,
        "score_promotable": score_promotable,
        "fidelity_migration_flag": os.environ.get("HENRI_ACCURACY_FIRST_CLASS4", "0"),
        "dataset_sha256": dataset_sha,
        "dataset_source": HUMANEVAL_URL,
        "item_count": len(items),
        "solved": solved,
        "expressible": expressible,
        "not_expressible": not_expressible,
        "infra_errors": infra_errors,
        "total_candidates_generated": total_candidates,
        "reward_rank": reward_rank,
        "items_reordered": items_reordered,
        "decoder_rank": decoder_rank,
        "decoder_items_reordered": decoder_items_reordered,
        "decoder_checkpoint_sha256": decoder_checkpoint_sha,
        "spec_rank": spec_rank,
        "docstring_targets_used": docstring_used,
        "trained_rank": trained_rank,
        "trained_head_sha256": trained_head_sha,
        "trained_items_reordered": trained_items_reordered,
        "ast_idf_only": ast_idf_only,
        "ast_idf_batched": ast_idf_batched,
        "ast_idf_items_reordered": ast_idf_items_reordered,
        "ast_idf_codebook_sha256": ast_idf_codebook_sha,
        "path_b2_codec": path_b2_codec,
        "path_b2_codec_sha256": path_b2_codec_sha,
        "path_b2_items_reordered": path_b2_items_reordered,
        "hops_vsa_rank": hops_vsa_rank,
        "hops_vsa_items_reordered": hops_vsa_items_reordered,
        "accuracy_attempted": solved / max(1, expressible),
        "wall_clock_sec": round(wall_sec, 3),
        "avg_latency_ms_item": round(avg_ms, 3),
        "device": device,
        "item_results": item_results,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    out_dir = output_dir or os.path.join(repo_path, "telemetry_logs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"humaneval_wave_ast_{int(time.time())}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(scorecard, f, indent=2)
    print(json.dumps({k: v for k, v in scorecard.items() if k != "item_results"},
                     indent=2))
    print(f"[SCORECARD] {out_path}")
    return scorecard


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--attempts", type=int, default=CANDIDATE_ATTEMPTS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke-dim", type=int, default=None,
                    help="reduced dimension for local CPU smoke only")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--reward-rank", action="store_true",
                    help="test-time learned positive-exemplar re-ranking (default OFF)")
    ap.add_argument("--decoder-rank", action="store_true",
                    help="rank candidates by trained-decoder token-predictability (default OFF)")
    ap.add_argument("--spec-rank", action="store_true",
                    help="V1: non-zero target wave from docstring spec (default OFF)")
    ap.add_argument("--trained-rank", action="store_true",
                    help="V3: execution-grounded correctness head ranking (default OFF)")
    ap.add_argument("--trained-head", default=None,
                    help="path to correctness-head checkpoint (HENRI_TRAINED_HEAD)")
    ap.add_argument("--ast-idf-only", action="store_true",
                    help="Gate A': IDF-weighted MBPP-codebook candidate ranking (default OFF)")
    ap.add_argument("--ast-idf-batched", action="store_true",
                    help="Phase 3: batched mean-phase-cosine ranking (default OFF; "
                         "requires --ast-idf-only)")
    ap.add_argument("--path-b2-codec", action="store_true",
                    help="Path B2 hard-negative codec candidate ranking "
                         "(Class 4.4, default OFF)")
    ap.add_argument("--hops-vsa-rank", action="store_true",
                    help="HOPS-VLA skeleton-free channel candidate ranking "
                         "(Class 4.5, default OFF)")
    args = ap.parse_args()
    run_benchmark(limit=args.limit, attempts=args.attempts,
                  device=args.device, smoke_dim=args.smoke_dim,
                  output_dir=args.output_dir, reward_rank=args.reward_rank,
                  decoder_rank=args.decoder_rank,
                  spec_rank=args.spec_rank,
                  trained_rank=args.trained_rank,
                  trained_head_path=args.trained_head,
                  ast_idf_only=args.ast_idf_only,
                  ast_idf_batched=args.ast_idf_batched,
                  path_b2_codec=args.path_b2_codec,
                  hops_vsa_rank=args.hops_vsa_rank)
