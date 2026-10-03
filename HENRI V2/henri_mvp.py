#!/usr/bin/env python3
"""HENRI MVP entrypoint: one command for the proven loop.

    retrieve from Zone C -> verified text context -> frozen backbone -> answer

WHY THIS FILE EXISTS
    SpecContract A is APPROVED (audit event #10b2d1f45cd8626c, chain 1816->1817)
    with scope "A-K4 egress kill plus MVP wiring". A-K4 is measured and passed
    (commit 7a44d5f). This file wires that exact measured loop for operator use.

THE LOOP THIS WIRES (measured, not asserted)
    ak4_egress_gate.py, 6-fact corpus, 12 items, CPU, frozen Qwen2.5-1.5B:
        hit_A0 = 0.000  (no evidence)
        hit_A1 = 0.833  (correct retrieved chunk)
        hit_A2 = 0.083  (mismatched chunk, target source excluded)
        determinism = 1.000
        verdict AK4_PASS_CONTEXT_IMPROVES_GENERATED_ANSWER

SCOPE BOUNDARY (do not overstate)
    ARM A only: retrieved context reaches the model as VERIFIED TEXT.
    ARM B (Zone C wave conditioning without a text intermediary) is UNMEASURED.
    The recorded top1_token_unique=1 defect lives on the ARM_U unbinder path
    (down_proj [2048,65536] -> lm_head [32000,2048]). This entrypoint does NOT
    exercise that path and does NOT clear it.

DELIBERATELY NOT DONE
    No latency measurement. No training. No GPU spend. Those are outside the
    approved scope.

FAIL-CLOSED
    ABSTAIN when retrieval returns nothing.
    ABSTAIN when verified context is empty.
    ABSTAIN when the checkpoint did not load with zero missing keys.

USAGE
    HENRI_BACKBONE=1 HENRI_BACKBONE_MODEL_DIR="<dir>" \
    python henri_mvp.py ask --domain computing --query "<q>" \
        --files-dir "<corpus_dir>" --dsn-env "<dsn.env>" \
        [--k 3] [--max-new-tokens 32]

OUTPUT
    JSON on stdout: answer, provenance (chunk_id, source_id, char_span,
    chunk_sha256, cosine), model telemetry, context_chars.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# This file lives in HENRI V2/, which is the import root for sibling modules.
_HENRI_V2 = str(Path(__file__).resolve().parent)
if _HENRI_V2 not in sys.path:
    sys.path.insert(0, _HENRI_V2)

# The prompt is byte-identical to zone_c_world_knowledge_harness.generate_answer
# so the MVP reproduces the exact condition A-K4 measured.
PROMPT_TMPL = (
    "Answer the question using ONLY the provided evidence.\n"
    "If the evidence is insufficient, answer: ABSTAIN.\n\n"
    "EVIDENCE:\n{ctx}\n\nQUESTION: {q}\n\nANSWER:"
)

_PROV_FIELDS = ("chunk_id", "source_id", "domain", "chunk_index",
                "char_span", "chunk_sha256", "cosine")


def _emit(obj: dict) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True))


def cmd_ask(args: argparse.Namespace) -> int:
    import zone_c_world_knowledge_harness as wkh
    from henri_backbone_adapter import QwenBackboneAdapter

    # ---- 1. retrieve (fail-closed) ---------------------------------------
    hits = wkh.query_corpus(args.domain, args.query, args.k,
                            dsn_env=args.dsn_env)
    if not hits:
        _emit({"answer": "ABSTAIN", "reason": "no retrieved chunks",
               "domain": args.domain, "query": args.query})
        return 0

    context, used = wkh.build_context(hits, Path(args.files_dir),
                                      max_chars=args.max_context_chars)
    if not context:
        _emit({"answer": "ABSTAIN", "reason": "empty verified context",
               "domain": args.domain, "query": args.query})
        return 0

    # ---- 2. generate with ONE adapter instance (frozen, zero-trainable) ---
    adapter = QwenBackboneAdapter(max_new_tokens=args.max_new_tokens)
    adapter.load()
    tel = adapter.telemetry
    bad = int(getattr(tel, "unexpected_key_count", 0) or 0)
    if bad != 0:
        _emit({"answer": "ABSTAIN",
               "reason": "checkpoint did not load cleanly",
               "unexpected_key_count": bad})
        return 1

    text, _meta = adapter.generate_text(
        PROMPT_TMPL.format(ctx=context, q=args.query))

    _emit({
        "answer": (text or "").strip(),
        "query": args.query,
        "domain": args.domain,
        "provenance": [{k: c.get(k) for k in _PROV_FIELDS} for c in used],
        "model": {
            "architecture": getattr(tel, "architecture", None),
            "total_params": getattr(tel, "total_params", None),
            "loaded_from": getattr(tel, "loaded_from", None),
            "declared_model_id": getattr(tel, "model_id", None),
            "unexpected_key_count": bad,
            "checkpoint_load_status": getattr(tel, "checkpoint_load_status", None),
        },
        "context_chars": len(context),
        "max_new_tokens": args.max_new_tokens,
        "arm": "A (retrieved text context); ARM B wave conditioning UNMEASURED",
        "latency": "not measured (out of approved scope)",
    })
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="henri_mvp")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("ask", help="retrieve -> generate (the A-K4 loop)")
    a.add_argument("--domain", required=True)
    a.add_argument("--query", required=True)
    a.add_argument("--files-dir", required=True)
    a.add_argument("--dsn-env", required=True)
    a.add_argument("--k", type=int, default=3)
    a.add_argument("--max-context-chars", type=int, default=3000)
    a.add_argument("--max-new-tokens", type=int, default=32)
    a.set_defaults(fn=cmd_ask)

    args = p.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
