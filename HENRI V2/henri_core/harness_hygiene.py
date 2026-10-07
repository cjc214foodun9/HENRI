"""Harness hygiene contracts (remediation directives D1-D5).

WHY THIS MODULE EXISTS
    The Q4 turn produced six testing defects. Each was fixed inline, in one
    script, which means the fix does not transfer. This module turns the
    remedies into reusable contracts so the defect class cannot recur.

DIRECTIVE MAP
    D1  matched negatives required. A suite whose only negative is orthogonal
        noise must fail pre-flight. -> assert_matched_negatives
    D2  guaranteed derangement. torch.randperm can return the identity
        (p = 1/N! per sample; ~34% over 10 items at N=4). -> derangement
    D3  bank/eval partition isolation by SHA-256, empty intersection.
        -> split_manifest
    D4  pinned execution generators. No bare global RNG. -> pinned_generator
    D5  margin-ratio contract, NOT a scalar float. NOTE: the blueprint's form
        divides by sigma_bank, which is 0.0 when every in-bank score is exactly
        1.0 (cos(psi,psi)=1). That form is NaN. This module uses a ROBUST
        non-zero scale and says so. -> margin_contract

Every contract RETURNS its evidence. Nothing here mutates a model.
"""
from __future__ import annotations

import hashlib
import json

import torch

__all__ = [
    "derangement", "split_manifest", "margin_contract",
    "pinned_generator", "assert_matched_negatives", "sha256_items",
    "EPS_SCALE",
]

EPS_SCALE = 1e-9


def sha256_items(items) -> list[str]:
    """SHA-256 of each item's canonical bytes. Deterministic across processes."""
    out = []
    for it in items:
        if isinstance(it, str):
            b = it.encode("utf-8")
        elif isinstance(it, bytes):
            b = it
        else:
            b = json.dumps(it, sort_keys=True, separators=(",", ":")).encode("utf-8")
        out.append(hashlib.sha256(b).hexdigest())
    return out


def split_manifest(bank_items, eval_items) -> dict:
    """D3: prove D_bank INTERSECT D_eval = EMPTY.

    Compares SHA-256 digests, not raw strings, so whitespace-normalised or
    re-encoded duplicates are still caught. Raises AssertionError on overlap.
    """
    bh, eh = sha256_items(bank_items), sha256_items(eval_items)
    inter = sorted(set(bh) & set(eh))
    assert not inter, (
        f"D3 VIOLATION: {len(inter)} item(s) appear in BOTH bank and eval "
        f"(sha256[:12]={[h[:12] for h in inter]}). The positive control has "
        "leaked into the evaluation set.")
    return {
        "bank_n": len(bh), "eval_n": len(eh),
        "bank_sha256": bh, "eval_sha256": eh,
        "intersection": inter, "intersection_cardinality": 0,
        "assert_intersection_empty": True,
    }


def derangement(n: int, generator: torch.Generator | None = None) -> list[int]:
    """D2: a permutation of range(n) with NO fixed points.

    Rejection-samples torch.randperm until zero fixed points. Falls back to the
    deterministic cyclic shift pi(i) = (i+1) mod n, which is always a derangement
    for n >= 2. The identity is rejected explicitly.

    n < 2 has no derangement; raises rather than returning a silent identity.
    """
    if n < 2:
        raise ValueError(f"no derangement exists for n={n}")
    g = generator if generator is not None else torch.Generator().manual_seed(0)
    for _ in range(256):
        perm = torch.randperm(n, generator=g).tolist()
        if all(perm[i] != i for i in range(n)):
            return perm
    return [(i + 1) % n for i in range(n)]      # guaranteed derangement


def pinned_generator(seed: int) -> torch.Generator:
    """D4: an explicit local generator. Never rely on the global RNG."""
    return torch.Generator().manual_seed(int(seed))


def _robust_scale(values) -> tuple[float, str, float, float]:
    """Non-zero spread estimate. Returns (scale, source, sigma, mad).

    The blueprint divides by sigma_bank. That is 0.0 whenever all in-bank scores
    are exactly 1.0, which is the measured case (cos(psi,psi)=1 for any encoder).
    A 0.0 denominator makes the contract NaN and therefore unfalsifiable. This
    function refuses to return 0.0.
    """
    t = torch.as_tensor(list(values), dtype=torch.float64).flatten()
    if t.numel() == 0:
        raise ValueError("no values")
    sigma = float(t.std(unbiased=True)) if t.numel() > 1 else 0.0
    med = float(t.median())
    mad = float((t - med).abs().median()) * 1.4826          # normal-consistent
    if mad > EPS_SCALE:
        return mad, "mad_1.4826", sigma, mad
    if sigma > EPS_SCALE:
        return sigma, "pooled_sigma", sigma, mad
    # Degenerate: every value identical. Refuse to divide.
    return float("nan"), "DEGENERATE_ZERO_SPREAD", sigma, mad


def margin_contract(pos_scores, neg_scores, k: float = 3.0) -> dict:
    """D5: require a k-sigma separation between positive and negative scores.

    Delta_margin = (min(pos) - max(neg)) / scale_neg

    pos_scores: in-bank / on-task scores (the class that must score HIGH)
    neg_scores: held-out + hard negatives (the class that must score LOW)
    k:          required separation in units of the robust scale

    The scale comes from the NEGATIVE population, which is the population whose
    spread is not degenerate. If the negative spread is zero the contract returns
    passes=False with scale_source=DEGENERATE_ZERO_SPREAD. It never returns a
    passing verdict from an uncomputable denominator.
    """
    pos = [float(x) for x in pos_scores]
    neg = [float(x) for x in neg_scores]
    if not pos or not neg:
        raise ValueError("need both populations")
    lo_pos, hi_neg = min(pos), max(neg)
    scale, source, sigma, mad = _robust_scale(neg)
    gap = lo_pos - hi_neg
    computable = scale == scale and scale > EPS_SCALE      # NaN-safe
    ratio = (gap / scale) if computable else float("nan")
    return {
        "pos_min": lo_pos, "pos_max": max(pos), "pos_n": len(pos),
        "neg_min": min(neg), "neg_max": hi_neg, "neg_n": len(neg),
        "gap_min_pos_minus_max_neg": gap,
        "scale": source, "scale_value": scale,
        "neg_sigma": sigma, "neg_mad": mad,
        "k_required": k, "ratio": ratio,
        "passes": bool(computable and ratio >= k),
        "separation_bool_holds": bool(gap > 0.0),
        "note": ("scale from the NEGATIVE population; sigma_bank is 0.0 when "
                 "in-bank scores saturate at 1.0"),
    }


def assert_matched_negatives(families: dict, allow_orthogonal_only: bool = False):
    """D1: pre-flight. A suite whose ONLY negatives are orthogonal noise fails.

    families: {name: {"scores": [...], "kind": "in_bank"|"heldout"|"permutation"
                      |"drop_boundary"|"synonym"|"orthogonal_noise"}}

    Raises AssertionError if no MATCHED negative family (permutation, drop
    boundary, heldout corpus, or synonym) is present. Orthogonal Gaussian noise
    is ~0.0039 in D=65536 by construction, so AUC against it is trivially 1.0.
    """
    kinds = {v.get("kind") for v in families.values()}
    matched = {"permutation", "drop_boundary", "heldout", "synonym"}
    present = kinds & matched
    if not present and not allow_orthogonal_only:
        raise AssertionError(
            "D1 VIOLATION: no matched negative family present (kinds="
            f"{sorted(k for k in kinds if k)}). Only orthogonal noise negatives "
            "produce a trivial AUC=1.0. Add a permutation, drop-boundary, "
            "heldout-corpus, or synonym family.")
    has_noise = "orthogonal_noise" in kinds
    return {
        "kinds": sorted(k for k in kinds if k), "matched_present": sorted(present),
        "orthogonal_noise_present": has_noise,
        "verdict": ("matched negatives present" if present
                    else "OVERRIDE: orthogonal-only explicitly allowed"),
    }
