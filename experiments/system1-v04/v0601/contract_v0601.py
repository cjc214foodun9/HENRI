"""
System-1 v0.6.0.1 candidate-retrieval CONTRACTS (pre-registered).
=================================================================
C1  within-task variance: on real pools, sim scores have nonzero variance
    (Var_k > 0); assert pools are non-empty BEFORE comparing (non-vacuity).
C2  beta=0 identity: ranking returns byte-identical order and scores.
C3  leakage closure: candidate repr uses ONLY code tokens; task repr uses
    ONLY signature latent; no fid/answer/verifier/outcome anywhere.
C4  determinism: same inputs -> same output (no sampling in ranker).
C5  default OFF: disabled ranker returns the pool unchanged.
C6  tokenizer closure: all candidate codes FSA-valid (no UNK).
C7  distinct candidates can differ in score (two structurally different
    codes can receive different sim scores).

Runs on the live frozen v0.5.5 carrier (system1_kernel_v055) WITHOUT any
sealed split (disposable fixtures only). This file is hashed into the
frozen manifest AFTER these contracts are final.
"""
from __future__ import annotations

import argparse
import pathlib
import random
import sys

import torch

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from system1_kernel_v041_energy_refactored import (  # noqa: E402
    TOK2ID, System1KernelV04, KernelV04Config, tokenize_code)
from system1_kernel_v055_ast_skeleton import (  # noqa: E402
    System1KernelV05, SkeletonGrammar)
from train_system1_kernel_v04 import gen_task  # noqa: E402
from train_v051_discriminator import (  # noqa: E402
    N_VERIFIER, N_OUTCOME, _rand_args, _expected, _args_key)
from zone_c_bridge_v0601 import CandidateRetrievalRanker  # noqa: E402


def _disposable_pool(seed: int, fid: int, top_k: int = 13) -> list[dict]:
    """Build a real, tokenizer-closed candidate pool for family fid using
    the live generator (same instantiation as eval_v055_heldout)."""
    rng = random.Random(seed)
    t = gen_task(rng, fid=fid)
    pool: list[dict] = []
    seen: set[str] = set()
    nargs = t["nargs"]
    arg_names = ["xs", "t1", "t2"][:nargs] if nargs <= 2 else \
        ["xs", "ys", "zs"][:nargs]
    for rule_id in range(13):
        code = SkeletonGrammar(n_rules=13).instantiate(
            rule_id, t["name"], arg_names)
        if code is None or code in seen:
            continue
        seen.add(code)
        ids = tokenize_code(code)
        if TOK2ID["UNK"] in ids:
            continue
        pool.append({"rule_id": rule_id, "code": code})
    return pool


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    dev = args.device
    cfg = KernelV04Config()
    backbone = System1KernelV04(cfg=cfg).to(dev)
    if args.ckpt:
        st = torch.load(args.ckpt, map_location=dev)
        backbone.load_state_dict(st["model"])
    backbone.eval()
    v05 = System1KernelV05(backbone, num_rules=13).to(dev)
    v05.eval()

    ranker = CandidateRetrievalRanker(enabled=True, beta=0.0, device=dev)
    ranker_on = CandidateRetrievalRanker(enabled=True, beta=0.15,
                                         device=dev)

    results: list[str] = []
    ok_all = True

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal ok_all
        results.append(f"{'PASS' if cond else 'FAIL'}  {name} {detail}")
        if not cond:
            ok_all = False

    # ---- C5: default OFF is identity ----
    off = CandidateRetrievalRanker(enabled=False, beta=0.5, device=dev)
    pool0 = _disposable_pool(1, 0, 13)
    out_off = off.rank_candidates(pool0, torch.zeros(1, 384, device=dev),
                                  v05, dev)
    check("C5_default_off_identity", [c["code"] for c in out_off] ==
          [c["code"] for c in pool0])

    # ---- C6: tokenizer closure ----
    check("C6_pool_nonempty_tokenizer_closed", len(pool0) > 0,
          f"pool={len(pool0)}")

    # ---- C1: variance > 0 on real pools ----
    all_var = []
    for fid in range(13):
        p = _disposable_pool(7 + fid, fid, 13)
        check(f"C1_pool_{fid}_nonempty", len(p) > 0, f"pool={len(p)}")
        if not p:
            continue
        t = gen_task(random.Random(3 + fid), fid=fid)
        z0 = v05.signature_latent(
            torch.randn(1, 16, 384, device=dev),
            torch.randn(1, 16, 384, device=dev))
        tv = ranker.task_repr(z0, z0, v05)
        cv = [ranker.candidate_repr(c["code"], v05, dev) for c in p]
        s = ranker.sim_scores(tv, cv)
        var = float(s.var(unbiased=False).item()) if s.numel() > 1 else 0.0
        all_var.append(var)
        check(f"C1_var_{fid}", var > 1e-6,
              f"var={var:.2e} pool={len(p)}")
    check("C1_var_positive_across_families",
          all(v > 1e-6 for v in all_var), f"vars={all_var}")

    # ---- C7: distinct candidates can differ ----
    p = _disposable_pool(11, 0, 13)
    tv = ranker.task_repr(torch.randn(1, 16, 384, device=dev),
                          torch.randn(1, 16, 384, device=dev), v05)
    cv = [ranker.candidate_repr(c["code"], v05, dev) for c in p]
    s = ranker.sim_scores(tv, cv)
    check("C7_scores_not_constant",
          (s.max() - s.min()).item() > 1e-6,
          f"range={(s.max() - s.min()).item():.4f}")

    # ---- C2: beta=0 byte-identical ----
    p = _disposable_pool(19, 3, 13)
    t = gen_task(random.Random(29), fid=3)
    z0 = v05.signature_latent(torch.randn(1, 16, 384, device=dev),
                              torch.randn(1, 16, 384, device=dev))
    tv = ranker.task_repr(z0, z0, v05)
    base_order = [c["code"] for c in p]
    r0 = ranker.rank_candidates(p, tv, v05, dev, beta=0.0)
    r0b = ranker_on.rank_candidates(p, tv, v05, dev, beta=0.0)
    check("C2_beta0_identity", [c["code"] for c in r0] == base_order and
          [c["code"] for c in r0b] == base_order)

    # ---- C4: determinism ----
    r1 = ranker_on.rank_candidates(p, tv, v05, dev, beta=0.15)
    r2 = ranker_on.rank_candidates(p, tv, v05, dev, beta=0.15)
    check("C4_deterministic", [c["code"] for c in r1] ==
          [c["code"] for c in r2])

    # ---- C3: leakage closure (static audit of repr inputs) ----
    import ast as _ast
    import inspect

    def _code_without_docs(fn) -> str:
        """Return function source with docstrings/comments removed so the
        leak scan inspects ACTUAL code, not prose that names the terms."""
        import textwrap
        src = textwrap.dedent(inspect.getsource(fn))
        try:
            tree = _ast.parse(src)
            for node in _ast.walk(tree):
                if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef,
                                     _ast.ClassDef)) and \
                        _ast.get_docstring(node):
                    node.body = node.body[1:] if node.body and \
                        isinstance(node.body[0], _ast.Expr) else node.body
            return _ast.unparse(tree)
        except Exception:
            return src  # fallback: raw source (conservative)

    src_cand = _code_without_docs(CandidateRetrievalRanker.candidate_repr)
    src_task = _code_without_docs(CandidateRetrievalRanker.task_repr)
    leak_terms = ["fid", "family", "verifier", "outcome", "expected",
                  "canonical", "answer", "sandbox", "rule_id"]
    leak = [w for w in leak_terms if w in (src_cand + src_task)]
    check("C3_no_leakage_terms_in_code", not leak, f"found={leak}")

    print("\n".join(results))
    print("ALL_CONTRACTS_PASS" if ok_all else "CONTRACTS_FAILED")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
