"""Run17 diagnostic probe: true-solution-in-space and at-what-rank.

Mandate (Egress and ast dev tree.pdf, Part I): for a sample of the
CEGIS_MISS items from a completed run, rebuild the EXACT production
pipeline (W_task -> R-EDMD prediction -> WaveASTDecoder.decode ->
MbppCegisSynthesizer.rank_candidates) and record the absolute integer
rank of the canonical ground-truth solution when it is expressible by
the current grammar.

Per-item classification (pre-registered, commit b475edb):
  SELECTION_HIT   rank <= 12            would pass the primary window
  WINDOW_HIT      rank <= 24            would pass the escalated window
  RANK_MISS       rank >  24            in space, below window ->
                                        recalibrate W_task (transition
                                        operator), do NOT grow grammar
  COVERAGE_MISS   not expressible by    grammar expansion is justified
                  / not generated
  SIG_UNAVAILABLE entry/args not derivable from prompt or tests

Verdict routing: median rank over expressible items <= 24 => selection
fine, misses are coverage => grammar expansion justified; median rank
> 24 => W_task/ranking misalignment => recalibrate the operator first.

The probe is a measurement instrument, not a scored benchmark. It
reuses the production modules; it never duplicates the scoring math.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from mbpp_cegis_synthesizer import (  # noqa: E402
    MbppCegisSynthesizer,
    _RenameArgs,
    parse_entry_from_tests,
    parse_entry_signature,
)
from mbpp_heldout_pilot import (  # noqa: E402
    COMPLEXITY_LAMBDA,
    load_exemplars,
    load_items,
    render_prompt,
)
from mbpp_wave_ast_decoder import WaveASTDecoder  # noqa: E402
from recursive_dual_edmd import RecursiveDualEDMD  # noqa: E402
from zone_c_epistemic_axiom_harness import (  # noqa: E402
    HolographicTaskFunctorCompiler,
    qFHRREpistemicCodec,
)
from qfhrr_structured_codec import ring_to_real  # noqa: E402

PRIMARY_WINDOW = 12
ESCALATED_WINDOW = 24


def canonical_signature(code: str) -> tuple[str, list[str]]:
    """Return (function name, positional args) of the canonical solution."""
    tree = ast.parse(code)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    return fn.name, [a.arg for a in fn.args.args]


def canonical_key(code: str, entry: str) -> Optional[str]:
    """Deterministic structural key of the canonical solution after
    stripping the docstring and renaming the canonical's OWN positional
    args to a0..aN.

    FIX (2026-08-02, post-run17): the old version took `args` from
    parse_entry_signature, which ALREADY renames args to a0/a1/a2; the
    rename map then became identity and descriptive canonical names
    (l, b, h, M, s) were never renamed -> false COVERAGE_MISS on every
    such item. The key now derives the map from the canonical's own
    signature, making the comparison invariant to both the item
    signature parser and the grammar's aN convention.

    Returns None when the canonical does not parse."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    body = [n for n in fn.body if not (
        isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)
        and isinstance(n.value.value, str))]
    fn.body = body
    orig = [a.arg for a in fn.args.args]
    arg_map = {orig[i]: f"a{i}" for i in range(len(orig))}
    tree = _RenameArgs(fn.name, entry, arg_map).visit(tree)
    return ast.dump(tree)


def candidate_key(src: str, entry: str, args: list[str]) -> Optional[str]:
    """Structural key of a generated candidate (already a0..aN named)."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    arg_map = {orig: new for orig, new in zip(
        [a.arg for a in fn.args.args], [f"a{i}" for i in range(len(args))])}
    tree = _RenameArgs(fn.name, entry, arg_map).visit(tree)
    return ast.dump(tree)


def build_pipeline(device: str, r_rank: int = 16, lambda_forget: float = 0.98,
                   regularization: float = 1e-4):
    """Replicate the pilot's HENRI path exactly (lines ~378-490).

    Run19 recalibration knobs (user mandate, 2026-08-02): r_rank is the
    Koopman observable-basis width (capped at min(r_rank, N=20 exemplar
    vectors) per the effective-rank invariant), lambda_forget is the
    online EDMD forgetting factor, regularization is the ridge on the
    low-rank covariance C_t (the 'lambda of the regression')."""
    exemplars = load_exemplars()
    codec = qFHRREpistemicCodec()
    task_compiler = HolographicTaskFunctorCompiler(codec)
    demo_pairs = [
        (codec.encode_text(render_prompt(ex)), codec.encode_text(ex["code"]))
        for ex in exemplars
    ]
    w_task_ring = task_compiler.compile_functor(demo_pairs)
    w_task_real = (
        w_task_ring.to(torch.float32) / (codec.k_bins - 1) * 2.0 - 1.0
    ).view(-1).to(device)

    pred_waves_real = [
        (codec.encode_text(render_prompt(ex)).to(torch.float32) / (codec.k_bins - 1) * 2.0 - 1.0).view(-1).to(device)
        for ex in exemplars
    ]
    sol_waves_real = [
        (codec.encode_text(ex["code"]).to(torch.float32) / (codec.k_bins - 1) * 2.0 - 1.0).view(-1).to(device)
        for ex in exemplars
    ]
    with torch.no_grad():
        _, _, Vt = torch.linalg.svd(
            torch.stack(pred_waves_real + sol_waves_real), full_matrices=False)
        v_basis = Vt.T[:, :r_rank].contiguous().to(device)
    edmd_predictor = RecursiveDualEDMD(
        d_model=65536, r_rank=r_rank, lambda_forget=lambda_forget,
        regularization=regularization, v_basis=v_basis).to(device)
    with torch.no_grad():
        for pw, sw in zip(pred_waves_real, sol_waves_real):
            edmd_predictor.update_online_step(
                pw.view(8192, 8), w_task_real.view(8192, 8), sw.view(8192, 8))
    decoder = WaveASTDecoder(codec, device=device)
    synth = MbppCegisSynthesizer(load_exemplars(), codec, device=device)
    return codec, w_task_real, edmd_predictor, decoder, synth


def per_item_demo_pairs(codec: qFHRREpistemicCodec, item: dict[str, Any],
                        entry: str) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Run20: compile demo pairs from the item's OWN test_list asserts.

    Each assert 'f(a, b, ...) == expected' yields (X_i, Y_i) where
    X_i = the concrete call 'f(a, b, ...)' text and Y_i = repr(expected).
    Transduced via the qFHRR codec exactly as the pilot transduces the
    10-exemplar pairs. This is test-time compilation from the task's own
    demonstration pairs (zero-pretraining invariant), never from the
    canonical solution code.
    """
    import ast
    pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    for t in item.get("test_list") or []:
        try:
            tree = ast.parse(t.strip())
            if len(tree.body) != 1:
                continue
            stmt = tree.body[0]
            if not isinstance(stmt, ast.Assert):
                continue
            cmp = stmt.test
            if not isinstance(cmp, ast.Compare) or len(cmp.ops) != 1:
                continue
            if not isinstance(cmp.ops[0], ast.Eq):
                continue
            left = cmp.left
            if not isinstance(left, ast.Call):
                continue
            args = [ast.literal_eval(a) for a in left.args]
            expected = ast.literal_eval(cmp.comparators[0])
            fname = left.func.id if isinstance(left.func, ast.Name) else entry
            x_str = f"{fname}({', '.join(map(repr, args))})"
            y_str = repr(expected)
            pairs.append((codec.encode_text(x_str), codec.encode_text(y_str)))
        except Exception:
            continue
    return pairs


def codec_geometry_control(codec: qFHRREpistemicCodec) -> dict[str, float]:
    """Deterministic probe of the qFHRR text codec's compositional structure.

    Under a structured codec, semantically close strings produce correlated
    waves. Under the current SHA-256-seeded random-ring encoder, every
    distinct string is an independent random vector: similarity ~ 1/sqrt(D)
    ~ 0.0039. These six numbers decide whether per-item W_task failure is an
    operator problem or a codec-geometry problem.
    """
    import math
    def s(t: str) -> torch.Tensor:
        return ring_to_real(codec, codec.encode_text(t)).view(-1)
    def c(a: str, b: str) -> float:
        return float(torch.nn.functional.cosine_similarity(
            s(a).view(1, -1), s(b).view(1, -1)).item())
    baseline = 1.0 / math.sqrt(codec.d_model)
    control = {
        "identical_input_sim": c("f(3, 3, 3)", "f(3, 3, 3)"),
        "nearby_input_sim": c("f(3, 3, 3)", "f(4, 4, 4)"),
        "commutative_sim": c("return a + b", "return b + a"),
        "position_swap_sim": c("ab", "ba"),
        "identical_output_sim": c("27", "27"),
        "nearby_output_sim": c("27", "28"),
        "unrelated_sim": c("abcd", "1234"),
        "random_baseline": baseline,
    }
    md = getattr(codec, "geometry_metadata", None)
    if callable(md):
        control.update(md())
    else:
        control.update({
            "codec_name": "legacy_sha256_ring",
            "codec_version": "legacy",
            "tokenizer": "sha256_full_string",
            "position_binding": "none",
            "position_index": "none",
            "position_mode": "none",
            "bundling": "none",
        })
    return control


def probe_item(
    item: dict[str, Any],
    codec: qFHRREpistemicCodec,
    pred_wave: torch.Tensor,
    decoder: WaveASTDecoder,
    synth: MbppCegisSynthesizer,
    manifold_proj: Optional[torch.Tensor] = None,
) -> dict[str, Any]:
    task_id = int(item["task_id"])
    prompt = render_prompt(item)
    prompt_wave = codec.encode_text(prompt)
    prompt_wave_real = ring_to_real(codec, prompt_wave).view(-1).to(pred_wave.device)
    sig = parse_entry_signature(prompt) or parse_entry_from_tests(
        item.get("test_list") or [])
    if sig is None:
        return {"task_id": task_id, "classification": "SIG_UNAVAILABLE",
                "rank": None, "n_candidates": 0, "true_score": None,
                "rank1_score": None, "rank1_body": None}
    entry, args = sig

    dec_cands = decoder.decode(
        pred_wave, prompt_wave_real, entry, args,
        manifold_proj=manifold_proj,
        complexity_lambda=COMPLEXITY_LAMBDA)

    cands = synth.build_candidates(prompt, item.get("test_list"))
    anchors = [c for c in cands if c[1].get("morphism") == "identity"]
    union = dec_cands + anchors
    ranked = synth.rank_candidates(union, pred_wave, prompt_wave=prompt_wave_real)

    canon_code = item.get("code", "")
    if not canon_code:
        return {"task_id": task_id, "classification": "SIG_UNAVAILABLE",
                "rank": None, "n_candidates": len(ranked), "true_score": None,
                "rank1_score": None, "rank1_body": None}
    true_key = canonical_key(canon_code, entry)
    if true_key is None:
        return {"task_id": task_id, "classification": "SIG_UNAVAILABLE",
                "rank": None, "n_candidates": len(ranked), "true_score": None,
                "rank1_score": None, "rank1_body": None}

    # Is the canonical structurally generated by the grammar enumerator?
    in_grammar = False
    generated = False
    rank: Optional[int] = None
    for src, _meta in dec_cands:
        if candidate_key(src, entry, args) == true_key:
            in_grammar = True
            break
    for idx, (src, _meta, _sim) in enumerate(ranked):
        if candidate_key(src, entry, args) == true_key:
            generated = True
            rank = idx + 1
            break

    if not generated:
        return {"task_id": task_id,
                "classification": "COVERAGE_MISS" if not in_grammar else "RANK_MISS_ABSENT",
                "rank": rank, "in_grammar": in_grammar,
                "n_candidates": len(ranked), "true_score": None,
                "rank1_score": None, "rank1_body": None}

    assert rank is not None
    if rank <= PRIMARY_WINDOW:
        cls = "SELECTION_HIT"
    elif rank <= ESCALATED_WINDOW:
        cls = "WINDOW_HIT"
    else:
        cls = "RANK_MISS"
    rank1_body = ranked[0][0].splitlines()[1].strip() if len(ranked[0][0].splitlines()) > 1 else ranked[0][0]
    return {"task_id": task_id, "classification": cls, "rank": rank,
            "in_grammar": True, "n_candidates": len(ranked),
            "rank1_body": rank1_body}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--failures-jsonl", required=True,
                    help="item_results.jsonl of a completed run (CEGIS_MISS filter)")
    ap.add_argument("--sample", type=int, default=50)
    ap.add_argument("--output", required=True, help="output dir")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--r-rank", type=int, default=16,
                    help="Koopman observable-basis width (effective rank cap = N exemplars)")
    ap.add_argument("--lambda-forget", type=float, default=0.98,
                    help="online EDMD forgetting factor")
    ap.add_argument("--regularization", type=float, default=1e-4,
                    help="ridge on the low-rank covariance (W_task spectral fit)")
    ap.add_argument("--expressible-only", action="store_true",
                    help="probe only items whose canonical is in the Phase A grammar")
    ap.add_argument("--per-item-wtask", action="store_true",
                    help="Run20: compile W_task from the item's OWN test_list demo "
                         "pairs (variants A=per-item EDMD, B=single-pass projection) "
                         "instead of the global 10-exemplar blend; emit codec "
                         "geometry control")
    ap.add_argument("--codec", default="legacy",
                    choices=["legacy", "structured", "structured-nopos",
                             "structured-shuffled", "identity"],
                    help="Run21 codec arm. identity = no W_task (prompt-wave ranking).")
    args = ap.parse_args()

    rows = []
    with open(args.failures_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    misses = [r for r in rows if str(r.get("failure_reason", "")).startswith("CEGIS")]
    misses = sorted(misses, key=lambda r: int(r.get("task_id", 0)))
    if args.expressible_only:
        # Run19 protocol step 1: isolate the expressible subset (canonical
        # verified in the Phase A grammar via AST-exact key). CPU-only.
        items_map = {int(i["task_id"]): i for i in load_items()}
        dec0 = WaveASTDecoder(None)
        exp: list[dict[str, Any]] = []
        for r in misses:
            it = items_map.get(int(r["task_id"]))
            if it is None or not it.get("code"):
                continue
            sig = parse_entry_signature(render_prompt(it)) or parse_entry_from_tests(
                it.get("test_list") or [])
            if sig is None:
                continue
            entry_name, arg_names = sig
            key = canonical_key(it["code"], entry_name)
            if key is None:
                continue
            for b in dec0._instantiate(entry_name, arg_names):
                src = f"def {entry_name}({', '.join(arg_names)}):\n{b}"
                if canonical_key(src, entry_name) == key:
                    exp.append(r)
                    break
        misses = exp
        print(f"[probe] expressible-only: {len(misses)} items")
    misses = misses[: args.sample]
    print(f"[probe] failures={len(rows)} misses={len([r for r in rows if str(r.get('failure_reason','')).startswith('CEGIS')])} sampled={len(misses)}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.per_item_wtask:
        # Run20: per-item operator compilation. The global 10-exemplar
        # W_task/EDMD blend is NOT built; only the shared codec, grammar
        # decoder, and synthesizer are reused.
        if args.codec == "legacy":
            codec = qFHRREpistemicCodec(device=args.device)
        else:
            from qfhrr_structured_codec import StructuredCharPositionCodec
            mode = {"structured": "full", "structured-nopos": "none",
                    "structured-shuffled": "shuffled"}.get(args.codec, "full")
            codec = StructuredCharPositionCodec(device=args.device, position_mode=mode)
        decoder = WaveASTDecoder(codec, device=args.device)
        synth = MbppCegisSynthesizer(load_exemplars(), codec, device=args.device)
        control = codec_geometry_control(codec)
        control["arm"] = args.codec
        with open(out_dir / "codec_geometry_control.json", "w", encoding="utf-8") as f:
            json.dump(control, f, indent=2)
        print(f"[probe] arm={args.codec} codec_geometry_control {json.dumps(control)}")

        results = []
        items_map = {int(i["task_id"]): i for i in load_items()}
        for i, r in enumerate(misses):
            item = items_map.get(int(r["task_id"]))
            if item is None:
                continue
            sig = parse_entry_signature(render_prompt(item)) or parse_entry_from_tests(
                item.get("test_list") or [])
            if sig is None:
                continue
            entry, _args = sig
            prompt_wave = codec.encode_text(render_prompt(item))
            prompt_wave_real = ring_to_real(codec, prompt_wave).view(-1).to(args.device)
            if args.codec == "identity":
                # Identity arm: no W_task, no EDMD. Prediction = prompt wave.
                try:
                    row = probe_item(item, codec, prompt_wave_real, decoder, synth)
                    row["variant"] = "IDENTITY"
                    row["n_demo_pairs"] = 0
                except Exception as exc:
                    row = {"task_id": int(r["task_id"]), "classification": "PROBE_ERROR",
                           "variant": "IDENTITY", "error": f"{type(exc).__name__}:{exc}",
                           "rank": None}
                row["source_failure"] = str(r.get("failure_reason", ""))
                results.append(row)
                print(f"[probe] {i+1}/{len(misses)} task {row['task_id']} "
                      f"IDENTITY={row['classification']}/{row.get('rank')}")
                continue
            pairs = per_item_demo_pairs(codec, item, entry)
            task_compiler = HolographicTaskFunctorCompiler(codec)
            w_task_ring = task_compiler.compile_functor(pairs)
            w_task_real = ring_to_real(codec, w_task_ring).view(-1).to(args.device)
            try:
                # Variant A: per-item EDMD fit on the item's own pairs.
                x_waves = [ring_to_real(codec, pw).view(-1).to(args.device)
                           for pw, _ in pairs]
                y_waves = [ring_to_real(codec, sw).view(-1).to(args.device)
                           for _, sw in pairs]
                with torch.no_grad():
                    _, _, vt = torch.linalg.svd(
                        torch.stack(x_waves + y_waves).view(len(x_waves) + len(y_waves), -1),
                        full_matrices=False)
                    v_basis = vt.T[:, : min(args.r_rank, len(x_waves) + len(y_waves))].contiguous().to(args.device)
                edmd = RecursiveDualEDMD(
                    d_model=65536, r_rank=min(args.r_rank, len(x_waves) + len(y_waves)),
                    lambda_forget=args.lambda_forget, regularization=args.regularization,
                    v_basis=v_basis).to(args.device)
                with torch.no_grad():
                    for pw, sw in zip(x_waves, y_waves):
                        edmd.update_online_step(
                            pw.view(8192, 8).to(args.device), w_task_real.view(8192, 8),
                            sw.view(8192, 8).to(args.device))
                pred_a = edmd(prompt_wave_real.view(8192, 8),
                              w_task_real.view(8192, 8)).view(-1)
                row_a = probe_item(item, codec, pred_a, decoder, synth,
                                   manifold_proj=v_basis)
                row_a["variant"] = "A_EDMD"
                row_a["n_demo_pairs"] = len(pairs)
            except Exception as exc:
                row_a = {"task_id": int(r["task_id"]), "classification": "PROBE_ERROR",
                         "variant": "A_EDMD", "error": f"{type(exc).__name__}:{exc}",
                         "rank": None}
            try:
                # Variant B: single-pass associative retrieval (no EDMD).
                pred_b_ring = codec.bind_hadamard(w_task_ring, prompt_wave)
                pred_b = ring_to_real(codec, pred_b_ring).view(-1).to(args.device)
                row_b = probe_item(item, codec, pred_b, decoder, synth)
                row_b["variant"] = "B_SINGLE_PASS"
                row_b["n_demo_pairs"] = len(pairs)
            except Exception as exc:
                row_b = {"task_id": int(r["task_id"]), "classification": "PROBE_ERROR",
                         "variant": "B_SINGLE_PASS",
                         "error": f"{type(exc).__name__}:{exc}", "rank": None}
            row_a["source_failure"] = str(r.get("failure_reason", ""))
            row_b["source_failure"] = str(r.get("failure_reason", ""))
            results.extend([row_a, row_b])
            print(f"[probe] {i+1}/{len(misses)} task {row_a['task_id']} "
                  f"A={row_a['classification']}/{row_a.get('rank')} "
                  f"B={row_b['classification']}/{row_b.get('rank')}")
    else:
        codec, w_task_real, edmd_predictor, decoder, synth = build_pipeline(
            args.device, r_rank=args.r_rank, lambda_forget=args.lambda_forget,
            regularization=args.regularization)
        print(f"[probe] recalib r_rank={args.r_rank} lambda_forget={args.lambda_forget} regularization={args.regularization}")

        results = []
        for i, r in enumerate(misses):
            item = next(
                (it for it in load_items() if int(it["task_id"]) == int(r["task_id"])), None)
            if item is None:
                continue
            try:
                row = probe_item(item, codec, w_task_real, edmd_predictor, decoder, synth)
            except Exception as exc:  # fail-closed probe row, never a fake rank
                row = {"task_id": int(r["task_id"]), "classification": "PROBE_ERROR",
                       "error": f"{type(exc).__name__}:{exc}", "rank": None}
            row["source_failure"] = str(r.get("failure_reason", ""))
            results.append(row)
            print(f"[probe] {i+1}/{len(misses)} task {row['task_id']} {row['classification']} rank={row.get('rank')}")

    out_path = out_dir / "rank_probe_results.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")

    classes = {}
    ranks = []
    for row in results:
        classes[row["classification"]] = classes.get(row["classification"], 0) + 1
        if row.get("rank") is not None:
            ranks.append(row["rank"])
    med = sorted(ranks)[len(ranks) // 2] if ranks else None
    summary = {
        "sample": len(results),
        "classes": classes,
        "median_rank_expressible": med,
        "n_expressible": len(ranks),
        "verdict": (
            "RANK_OK_SELECTION_FINE" if (med is not None and med <= ESCALATED_WINDOW)
            else "W_TASK_RECALIBRATE" if (med is not None and med > ESCALATED_WINDOW)
            else "NO_EXPRESSIBLE"),
        "routing": (
            "grammar expansion justified" if (med is not None and med <= ESCALATED_WINDOW)
            else "recalibrate W_task before any grammar growth"),
    }
    variants = {}
    for row in results:
        v = row.get("variant", "GLOBAL")
        d = variants.setdefault(v, {"classes": {}, "ranks": [], "task_ranks": {}})
        d["classes"][row["classification"]] = d["classes"].get(row["classification"], 0) + 1
        if row.get("rank") is not None:
            d["ranks"].append(row["rank"])
            d["task_ranks"][str(row["task_id"])] = row["rank"]
    for v, d in variants.items():
        d["median_rank"] = sorted(d["ranks"])[len(d["ranks"]) // 2] if d["ranks"] else None
        d["accepted"] = d["median_rank"] is not None and d["median_rank"] <= ESCALATED_WINDOW
    summary["by_variant"] = variants
    summary["accepted_variant"] = next(
        (v for v, d in variants.items() if d.get("accepted")), None)
    with open(out_dir / "rank_probe_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("[probe] SUMMARY " + json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
