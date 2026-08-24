"""
System-1 Kernel v0.4 trainer - Token-FSA + name-conditioned curriculum.
=========================================================================
Faithful implementation of the Aletheia v0.4 spec (Drive inbox 2026-08-23
09:43; spec sha b42815d3..., engine sha d582406c...).

Mechanisms (spec-prescribed):
  L1  Token-level FSA (no UNK wildcard; tight paren/colon/IND; depth-aware
      NL/EOS ban) - mask applied at train AND decode.
  L2  UNK logit-mass suppression penalty (softplus on raw UNK logit).
  L3  Prompt-symbol cross-attention name conditioning: signature symbols
      (def NAME ( args ) :) are embedded into z0 AND supplied as the
      cross-attention memory. Names are SUPPLIED, never memorized.
  L4  Masked free-run MLE: free autoregressive decode with the FSA mask,
      supervised per-position (the capability loss; CE gradients see
      grammar-valid generated contexts).
  L5  REINFORCE stage 2 (group-normalized clipped advantages) on shaped
      rewards + Brier energy baseline.
  L6  Extended 1,000-step warm-up + capability abort gates.

Integrity (from system1-curriculum-collapse lessons):
  - NEW digest-sealed splits with NEVER-USED seeds:
      smoke40_v04  (seed 42+99991, disposable - branch smoke only)
      valid20_v04  (seed 42+88891, checkpoint selection)
      heldout40_v04(seed 42+77781, final eval ONCE after training frozen)
    The final heldout is NEVER evaluated in smoke or cap-smoke modes.
  - Capability smoke (--cap-smoke, 500 steps, stage A) evaluates ONLY on the
    disposable smoke split; promotion decision comes from telemetry
    (free_ast > 0 sustained, div >= 2, no NaN, no flat reward).
  - ast.parse is EXTERNAL telemetry/reward only, never a loss.
  - Exact sandbox pass remains the promotion metric.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import pathlib
import random
import re
import subprocess
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/root/henri-system1")
from system1_kernel_v04 import (  # noqa: E402
    VOCAB, TOK2ID, ID2TOK, NEXT_MASK, KernelV04Config, System1KernelV04,
    SwarmEngineV04, TokenFSAGrammarMask, grammar_loss, tokenize_code,
    detokenize)

# ---------------------------------------------------------------------------
# Synthetic curriculum (project-owned templates, benchmark-disjoint)
# ---------------------------------------------------------------------------
STAGE_FIDS = {"A": [0, 1, 2], "B": [0, 1, 2, 3, 4], "C": list(range(7))}


def gen_task(rng: random.Random, fid: int | None = None) -> dict:
    if fid is None:
        fid = rng.randrange(7)   # default stays 7 (v0.4 trainer unchanged)
    if fid == 0:
        name, sig, body = "sum_list", "def sum_list(xs):", "    return sum(xs)"
        n = rng.randint(2, 8); xs = [rng.randint(-10, 10) for _ in range(n)]
        tests = [f"assert {name}({xs}) == {sum(xs)}"]
        prompt = "Write a function that returns the sum of a list of integers."
        nargs = 1
    elif fid == 1:
        name, sig, body = "max_list", "def max_list(xs):", "    return max(xs)"
        n = rng.randint(2, 8); xs = [rng.randint(-10, 10) for _ in range(n)]
        tests = [f"assert {name}({xs}) == {max(xs)}"]
        prompt = "Write a function that returns the largest element of a list of integers."
        nargs = 1
    elif fid == 2:
        name, sig, body = "count_positive", "def count_positive(xs):", \
            "    return sum(1 for x in xs if x > 0)"
        n = rng.randint(2, 8); xs = [rng.randint(-10, 10) for _ in range(n)]
        tests = [f"assert {name}({xs}) == {sum(1 for x in xs if x > 0)}"]
        prompt = "Write a function that counts how many elements of a list are positive."
        nargs = 1
    elif fid == 3:
        name, sig, body = "intersect_tuples", "def intersect_tuples(t1, t2):", \
            "    return tuple(sorted(set(t1) & set(t2)))"
        a = [rng.randint(-5, 5) for _ in range(rng.randint(3, 6))]
        b = [rng.randint(-5, 5) for _ in range(rng.randint(3, 6))]
        inter = sorted(set(a) & set(b))
        tests = [f"assert {name}({tuple(a)}, {tuple(b)}) == {tuple(inter)}"]
        prompt = "Write a function that returns the intersection of two tuples as a sorted tuple."
        nargs = 2
    elif fid == 4:
        name, sig, body = "union_tuples", "def union_tuples(t1, t2):", \
            "    return tuple(sorted(set(t1) | set(t2)))"
        a = [rng.randint(-5, 5) for _ in range(rng.randint(3, 6))]
        b = [rng.randint(-5, 5) for _ in range(rng.randint(3, 6))]
        uni = sorted(set(a) | set(b))
        tests = [f"assert {name}({tuple(a)}, {tuple(b)}) == {tuple(uni)}"]
        prompt = "Write a function that returns the union of two tuples as a sorted tuple."
        nargs = 2
    elif fid == 5:
        name, sig, body = "pair_sums", "def pair_sums(a, b):", \
            "    return [x + y for x, y in zip(a, b)]"
        n = rng.randint(2, 6)
        a = [rng.randint(-10, 10) for _ in range(n)]
        b = [rng.randint(-10, 10) for _ in range(n)]
        tests = [f"assert {name}({a}, {b}) == {[x + y for x, y in zip(a, b)]}"]
        prompt = "Write a function that returns the elementwise sum of two equal-length lists."
        nargs = 2
    elif fid == 6:
        name = "factorial"; nargs = 1
        sig = "def factorial(n):"
        body = "    res = 1\n    for i in range(1, n + 1):\n        res = res * i\n    return res"
        k = rng.randint(0, 8)
        tests = [f"assert {name}({k}) == {math.factorial(k)}"]
        prompt = "Write a function that returns the factorial of a non-negative integer."
    elif fid == 7:
        name, sig, body = "m", "def m(xs):", "    return min(xs)"
        n = rng.randint(2, 8); xs = [rng.randint(-10, 10) for _ in range(n)]
        tests = [f"assert {name}({xs}) == {min(xs)}"]
        prompt = "Write a function that returns the smallest element of a list of integers."
        nargs = 1
    elif fid == 8:
        name, sig, body = "v", "def v(xs):", "    return [abs(x) for x in xs]"
        n = rng.randint(2, 8); xs = [rng.randint(-10, 10) for _ in range(n)]
        tests = [f"assert {name}({xs}) == {[abs(x) for x in xs]}"]
        prompt = "Write a function that returns the absolute values of each element of a list."
        nargs = 1
    elif fid == 9:
        name, sig, body = "n", "def n(xs):", "    return sorted(xs)"
        n = rng.randint(2, 8); xs = [rng.randint(-10, 10) for _ in range(n)]
        tests = [f"assert {name}({xs}) == {sorted(xs)}"]
        prompt = "Write a function that returns a sorted copy of a list of integers."
        nargs = 1
    elif fid == 10:
        name, sig, body = "a", "def a(xs):", "    return sum(range(len(xs)))"
        n = rng.randint(2, 8); xs = [rng.randint(-10, 10) for _ in range(n)]
        tests = [f"assert {name}({xs}) == {sum(range(len(xs)))}"]
        prompt = "Write a function that returns the sum of indices 0..len(xs)-1 of a list."
        nargs = 1
    elif fid == 11:
        name, sig, body = "b", "def b(t1, t2):", \
            "    return [x - y for x, y in zip(t1, t2)]"
        n = rng.randint(2, 6)
        t1 = [rng.randint(-10, 10) for _ in range(n)]
        t2 = [rng.randint(-10, 10) for _ in range(n)]
        tests = [f"assert {name}({t1}, {t2}) == {[x - y for x, y in zip(t1, t2)]}"]
        prompt = "Write a function that returns the elementwise difference of two equal-length lists."
        nargs = 2
    else:
        name, sig, body = "res", "def res(xs):", \
            "    acc = 1\n    for x in xs:\n        acc = acc * x\n    return acc"
        n = rng.randint(2, 7)
        xs = [rng.randint(1, 6) for _ in range(n)]
        prod = 1
        for x in xs:
            prod *= x
        tests = [f"assert {name}({xs}) == {prod}"]
        prompt = "Write a function that returns the product of all elements of a list."
        nargs = 1
    return {"name": name, "code": sig + "\n" + body, "tests": tests,
            "prompt": prompt, "fid": fid, "nargs": nargs}


def sandbox_fraction(code: str, tests: list[str], timeout: int = 4) -> float:
    if not tests:
        return 0.0
    npass = 0
    for tst in tests:
        src = code + "\n" + tst
        try:
            r = subprocess.run([sys.executable, "-c", src],
                               capture_output=True, text=True, timeout=timeout)
            npass += 1 if r.returncode == 0 else 0
        except subprocess.TimeoutExpired:
            pass
    return npass / len(tests)


def sandbox(code: str, tests: list[str], timeout: int = 5) -> int:
    return 1 if sandbox_fraction(code, tests, timeout=timeout) == 1.0 else 0


def shaped_reward(code: str, ids, task: dict) -> tuple[float, dict]:
    comps = {}
    comps["lex"] = 1.0 if TOK2ID["UNK"] not in ids else 0.0
    comps["bal"] = 1.0 if (code.count("(") == code.count(")") and
                           code.count("[") == code.count("]")) else 0.0
    comps["gr"] = 0.0
    ids_body = ids[1:] if ids and ids[0] == TOK2ID["BOS"] else ids
    if ids_body:
        pairs = list(zip([TOK2ID["BOS"]] + ids_body[:-1], ids_body))
        comps["gr"] = sum(int(NEXT_MASK[p, n].item()) for p, n in pairs) / len(pairs)
    comps["parse"] = 0.0
    comps["sig"] = 0.0
    try:
        tree = ast.parse(code)
        comps["parse"] = 1.0
        fn = tree.body[0]
        comps["sig"] = 1.0 if (isinstance(fn, ast.FunctionDef)
                               and fn.name == task["name"]
                               and len(fn.args.args) == task["nargs"]) else 0.0
    except Exception:
        pass
    frac = sandbox_fraction(code, task["tests"])
    comps["exec"] = frac
    r = (0.10 * comps["lex"] + 0.10 * comps["bal"] + 0.15 * comps["gr"]
         + 0.20 * comps["parse"] + 0.15 * comps["sig"] + 0.30 * comps["exec"])
    if frac == 1.0 and task["tests"]:
        r = 1.0
    return r, comps


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4))


def fp_of(t: dict) -> str:
    return f"{t['name']}|{','.join(t['tests'])}"


def sig_ids(task: dict) -> list[int]:
    """Signature symbol ids: [def, NAME, (, args, ..., )] (BOS/NL/EOS removed)."""
    sig = task["code"].splitlines()[0].strip()
    ids = [i for i in tokenize_code(sig) if i not in
           (TOK2ID["BOS"], TOK2ID["NL"], TOK2ID["EOS"])]
    return ids


def pad_tokens(batch: list[list[int]], max_len: int = 48) -> torch.Tensor:
    out = torch.full((len(batch), max_len), TOK2ID["PAD"], dtype=torch.long)
    for i, ids in enumerate(batch):
        for j, t in enumerate(ids[:max_len]):
            out[i, j] = t
    return out


def sig_matrix(model: nn.Module, tasks: list[dict],
               max_k: int = 16, device=None) -> torch.Tensor:
    """[B, K, d_slot] signature symbol embeddings for cross-attention."""
    ids = pad_tokens([sig_ids(t) for t in tasks], max_k)
    if device is not None:
        ids = ids.to(device)
    return model.token_emb(ids)


def load_split(out_dir: str, n: int, seed: int, tag: str) -> list[dict]:
    p = pathlib.Path(out_dir) / f"{tag}.json"
    if p.exists():
        return json.loads(p.read_text())
    rng = random.Random(seed)
    tasks = [gen_task(rng) for _ in range(n)]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(tasks))
    return tasks


def sha256_file(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Evaluation: matched swarm (B=128) / single (B=1) / beam (K=16), exact sandbox
# Deterministic mode (default): byte-identical to v0.4 release behavior.
# Stochastic mode (--stochastic-eval, default-OFF): seeded per-particle
#   decode_sample + energy-weighted vote, MATCHED decode budget across arms,
#   pre-registered mechanism-engagement + statistical promotion gates.
# ---------------------------------------------------------------------------
def _mcnemar_two_sided(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value on discordant pairs (b=swarm_only,
    c=control_only)."""
    import math as _m
    n = b + c
    if n == 0:
        return 1.0
    p = 0.0
    for k in range(min(b, c) + 1):
        p += _m.comb(n, k) / (2.0 ** n)
    return 2.0 * min(p, 0.5)


def _spearman(ranks: list[int], passes: list[int]) -> float:
    """Spearman rank correlation between energy rank (0=best) and pass."""
    import math as _m
    n = len(ranks)
    if n < 3:
        return 0.0
    r = {v: i for i, v in enumerate(sorted(set(ranks)))}
    p = {v: i for i, v in enumerate(sorted(set(passes)))}
    rx = [r[v] for v in ranks]; ry = [p[v] for v in passes]
    mx = sum(rx) / n; my = sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = _m.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = _m.sqrt(sum((b - my) ** 2 for b in ry))
    return 0.0 if dx == 0 or dy == 0 else num / (dx * dy)


def eval_split(eng: SwarmEngineV04, model: System1KernelV04, dev,
               tasks: list[dict], swarm_b: int = 128,
               do_beam: bool = True, do_single: bool = True,
               stochastic: bool = False,
               vote_seed_base: int = 0, beam_width: int = 16,
               assoc_topk: int = 8) -> dict:
    n = len(tasks)
    swarm_pass = single_pass = beam_pass = greedy_pass = 0
    transitions = {"both_pass": 0, "swarm_only": 0, "single_only": 0, "both_fail": 0}
    ast_valid = 0
    recs: list[dict] = []
    seed_replay_identical: bool | None = None
    unique_prog_sum = 0.0
    differs_greedy = 0
    energy_nonuniform = 0
    assoc_ranks: list[int] = []
    assoc_passes: list[int] = []
    with torch.no_grad():
        for t_idx, t in enumerate(tasks):
            sp_sw = sig_matrix(model, [t] * swarm_b, 16, dev)
            z0 = model.encode_tokens(pad_tokens([sig_ids(t)] * swarm_b, 16).to(dev))
            out = eng.forward_swarm(z0, b_target=swarm_b, steps=8)
            e = out["energy"]
            best = int(e.argmax())
            code_s = detokenize(model.decode_greedy(
                out["z"][best:best + 1], sp_sw[best:best + 1])[0].tolist())
            sp = sandbox(code_s, t["tests"])

            z1 = model.encode_tokens(pad_tokens([sig_ids(t)], 16).to(dev))
            sp1 = sig_matrix(model, [t], 16, dev)
            out1 = eng.forward_swarm(z1, b_target=1, steps=8)

            if stochastic:
                # ---- swarm arm: B particles, 1 seeded sample each, energy vote
                s_ids, s_rec = model.decode_vote(
                    out["z"], sp_sw, e, seed_base=vote_seed_base + t_idx * 7 + 1)
                code_s = detokenize(s_ids)
                sp = sandbox(code_s, t["tests"])

                # ---- matched single-sample control: 1 particle, B seeds,
                #      same aggregation rule (uniform: single-particle energy)
                c_ids, c_rec = model.decode_vote(
                    z1.repeat(swarm_b, 1, 1), sp1.repeat(swarm_b, 1, 1), None,
                    seed_base=vote_seed_base + t_idx * 7 + 2)
                code_c = detokenize(c_ids)
                sg = sandbox(code_c, t["tests"])

                # ---- beam anchor (deterministic, width MATCHED to swarm B)
                seq = model.beam_decode(out1["z"][0:1], sp1, width=swarm_b)
                code_b = detokenize(seq)
                bp = sandbox(code_b, t["tests"])

                # ---- greedy capability anchor
                code_g = detokenize(model.decode_greedy(
                    out1["z"][0:1], sp1)[0].tolist())
                greedy_pass += sandbox(code_g, t["tests"])

                # ---- engagement telemetry
                unique_prog_sum += s_rec["unique_programs"]
                g_ids = model.decode_greedy(out["z"][best:best + 1],
                                            sp_sw[best:best + 1])[0].tolist()
                if TOK2ID["EOS"] in g_ids:
                    g_ids = g_ids[:g_ids.index(TOK2ID["EOS"])]
                if s_ids != g_ids:
                    differs_greedy += 1
                if s_rec["weight_var"] > 0:
                    energy_nonuniform += 1
                # top-K energy association
                topk = torch.topk(e, min(assoc_topk, e.shape[0])).indices
                for rank, pi in enumerate(topk.tolist()):
                    pids, _ = model.decode_vote(
                        out["z"][pi:pi + 1], sp_sw[pi:pi + 1], e[pi:pi + 1],
                        seed_base=vote_seed_base + t_idx * 13 + rank + 3)
                    code_p = detokenize(pids)
                    assoc_ranks.append(rank)
                    assoc_passes.append(int(sandbox(code_p, t["tests"])))
                # seed-replay reproducibility (checked once)
                if seed_replay_identical is None:
                    a, _ = model.decode_sample(
                        out["z"][0:1], sp_sw[0:1], seed=12345)
                    b2, _ = model.decode_sample(
                        out["z"][0:1], sp_sw[0:1], seed=12345)
                    seed_replay_identical = (a.tolist() == b2.tolist())
            else:
                code_1 = detokenize(model.decode_greedy(out1["z"][0:1], sp1)[0].tolist())
                sg = sandbox(code_1, t["tests"])
                if do_beam:
                    seq = model.beam_decode(out1["z"][0:1], sp1, width=beam_width)
                    code_b = detokenize(seq)
                    bp = sandbox(code_b, t["tests"])
                else:
                    bp = 0

            swarm_pass += sp; single_pass += sg; beam_pass += bp
            try:
                ast.parse(code_s)
                ast_valid += 1
            except Exception:
                pass
            if sp and sg:
                transitions["both_pass"] += 1
            elif sp and not sg:
                transitions["swarm_only"] += 1
            elif sg and not sp:
                transitions["single_only"] += 1
            else:
                transitions["both_fail"] += 1
            recs.append({"task": t.get("fid", t_idx), "swarm": sp, "single": sg,
                         "beam": bp, "ast": _ast_ok(code_s) if stochastic else 0,
                         "winner_ids": s_rec["winner_ids"] if stochastic else None,
                         "vote": s_rec if stochastic else None})

    strict = swarm_pass > single_pass and swarm_pass > beam_pass
    report = {
        "n": n,
        "swarm_pass": swarm_pass, "single_pass": single_pass, "beam_pass": beam_pass,
        "greedy_pass": greedy_pass if stochastic else single_pass,
        "swarm_pass_rate": round(swarm_pass / n, 4),
        "single_pass_rate": round(single_pass / n, 4),
        "beam_pass_rate": round(beam_pass / n, 4),
        "swarm_wilson_ci95": wilson(swarm_pass, n),
        "single_wilson_ci95": wilson(single_pass, n),
        "beam_wilson_ci95": wilson(beam_pass, n),
        "transitions": transitions,
        "ast_valid_rate": round(ast_valid / n, 4),
        "swarm_superiority_pass": strict,
        "kill_fired": not strict,
        "diagnostic_only": not strict,
    }
    if stochastic:
        mean_unique = unique_prog_sum / max(1, n)
        sw_only, c_only = transitions["swarm_only"], transitions["single_only"]
        mcnemar_p = _mcnemar_two_sided(sw_only, c_only)
        delta = swarm_pass / max(1, n) - single_pass / max(1, n)
        engagement = {
            "seed_replay_identical": bool(seed_replay_identical),
            "mean_unique_programs": round(mean_unique, 4),
            "differs_from_greedy_frac": round(differs_greedy / max(1, n), 4),
            "energy_nonuniform_frac": round(energy_nonuniform / max(1, n), 4),
            "topk_assoc": assoc_topk,
            "topk_sandbox_passes": sum(assoc_passes),
            "energy_assoc_spearman": round(_spearman(assoc_ranks, assoc_passes), 4),
        }
        promo = (strict and
                 engagement["seed_replay_identical"] is True and
                 engagement["mean_unique_programs"] > 1.0 and
                 engagement["differs_from_greedy_frac"] > 0.0 and
                 engagement["energy_nonuniform_frac"] > 0.0 and
                 engagement["topk_sandbox_passes"] > 0 and
                 engagement["energy_assoc_spearman"] > 0.0 and
                 delta >= 0.10 and mcnemar_p < 0.05)
        report["gates"] = engagement
        report["mcnemar_p"] = round(mcnemar_p, 4)
        report["delta_vs_single"] = round(delta, 4)
        report["eval_mode"] = "stochastic_vote"
        report["swarm_superiority_pass"] = promo
        report["kill_fired"] = not promo
        report["diagnostic_only"] = not promo
        report["items"] = recs
    else:
        report["eval_mode"] = "deterministic_greedy"
    return report


def _ast_ok(code: str) -> int:
    try:
        ast.parse(code)
        return 1
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--free-frac", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="/root/henri-system1/ckpt_v04")
    ap.add_argument("--swarm-b", type=int, default=128)
    ap.add_argument("--c1", type=int, default=1000, help="stage A->B step")
    ap.add_argument("--c2", type=int, default=2000, help="stage B->C step")
    ap.add_argument("--warmup-step", type=int, default=1000,
                    help="L6 capability abort check step")
    ap.add_argument("--no-abort", action="store_true",
                    help="telemetry only (branch smoke)")
    ap.add_argument("--cap-smoke", action="store_true",
                    help="capability smoke: 500 steps stage A, eval on the "
                         "DISPOSABLE smoke40_v04 split ONLY (never heldout)")
    ap.add_argument("--stochastic-eval", action="store_true",
                    help="eval-only stochastic vote arms (seeded decode_sample "
                         "per particle + energy-weighted vote, matched compute). "
                         "DEFAULT-OFF: deterministic greedy eval unchanged. "
                         "REFUSES to run against the consumed heldout split.")
    ap.add_argument("--beam-width", type=int, default=16,
                    help="beam width for the deterministic eval anchor "
                         "(stochastic mode matches width to --swarm-b)")
    ap.add_argument("--eval-only", action="store_true",
                    help="skip training; load best checkpoint and run the "
                         "selected eval (heldout/valid in prod mode; smoke40 "
                         "in --cap-smoke mode)")
    args = ap.parse_args()
    if args.stochastic_eval and not args.cap_smoke:
        raise SystemExit("REFUSE: --stochastic-eval against the consumed "
                         "heldout40_v04 split. Use --cap-smoke (smoke40_v04 "
                         "dev split) or a fresh sealed split.")

    torch.manual_seed(args.seed)
    dev = args.device
    cfg = KernelV04Config()
    model = System1KernelV04(cfg=cfg).to(dev)
    eng = SwarmEngineV04(model).to(dev)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    assert params < 30.0, f"FALSIFIED microcore rule: {params:.2f}M >= 30M"
    print(f"PARAMS_M model={params:.2f}M (core="
          f"{sum(p.numel() for p in model.core.parameters()) / 1e6:.3f}M, "
          f"decoder={sum(p.numel() for p in model.decoder.parameters()) / 1e6:.3f}M, "
          f"energy={sum(p.numel() for p in model.energy.parameters()) / 1e6:.3f}M)",
          flush=True)

    # Preflight: every curriculum target tokenizes UNK-free.
    rng_pre = random.Random(99991)
    bad = 0
    for _ in range(200):
        t = gen_task(rng_pre)
        if TOK2ID["UNK"] in tokenize_code(t["code"]):
            bad += 1
    assert bad == 0, f"PREFLIGHT: {bad}/200 targets contain UNK (vocab gap)"
    print(f"PREFLIGHT ok: 0/200 curriculum targets contain UNK", flush=True)

    # Preflight: FSA accepts every curriculum target (no false negatives).
    # NOTE: tokenize_code prepends BOS; skip it (start symbol, not a
    # transition) along with PAD/EOS.
    fb = 0
    for _ in range(200):
        t = gen_task(rng_pre)
        ids = tokenize_code(t["code"])
        prev = TOK2ID["BOS"]
        for tok in ids:
            if tok in (TOK2ID["BOS"], TOK2ID["PAD"], TOK2ID["EOS"]):
                continue
            if not NEXT_MASK[prev, tok].item():
                fb += 1
                break
            prev = tok
    assert fb == 0, f"PREFLIGHT: {fb}/200 targets violate the FSA"
    print(f"PREFLIGHT ok: 0/200 curriculum targets violate the FSA", flush=True)

    # Split selection: heldout/valid ONLY in production mode.
    ckpt_path = args.out + "/checkpoint.pt"
    pathlib.Path(args.out).mkdir(parents=True, exist_ok=True)
    if args.cap_smoke:
        heldout = load_split(args.out, 40, args.seed + 99991, "smoke40_v04")
        valid = []
        eval_tag = "smoke40_v04"
    else:
        heldout = load_split(args.out, 40, args.seed + 77781, "heldout40_v04")
        valid = load_split(args.out, 20, args.seed + 88891, "valid20_v04")
        eval_tag = "heldout40_v04"
    held_fps = {fp_of(t) for t in heldout} | {fp_of(t) for t in valid}
    split_digests = {}
    for tag in ("smoke40_v04", "valid20_v04", "heldout40_v04"):
        p = pathlib.Path(args.out) / f"{tag}.json"
        if p.exists():
            split_digests[tag] = sha256_file(p)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps,
                                                       eta_min=1e-5)
    rng = random.Random(args.seed)
    t0 = time.time()
    best_val = -1.0
    best_step = 0
    win_free_ast = []
    win_pg_norm = []
    win_reward = []
    aborted = None

    if args.eval_only:
        ck = args.out + "/best_val.pt"
        if not pathlib.Path(ck).exists():
            ck = args.out + "/checkpoint.pt"
        st = torch.load(ck, map_location=dev)
        model.load_state_dict(st["model"])
        print(f"EVAL_ONLY loaded {ck} step={st.get('step')}", flush=True)
        report = eval_split(eng, model, dev, heldout, swarm_b=args.swarm_b,
                            stochastic=args.stochastic_eval,
                            vote_seed_base=args.seed, beam_width=args.beam_width)
        print(f"EVAL ({eval_tag}):", json.dumps(report), flush=True)
        out_eval = (args.out + "/eval_stochastic.json" if args.stochastic_eval
                    else args.out + "/eval_only.json")
        pathlib.Path(out_eval).write_text(json.dumps(report, indent=2))
        print(f"EVAL_ONLY wrote {out_eval}", flush=True)
        return

    for step in range(1, args.steps + 1):
        stage = "A" if step <= args.c1 else ("B" if step <= args.c2 else "C")
        tasks = []
        while len(tasks) < args.batch:
            t = gen_task(rng, fid=rng.choice(STAGE_FIDS[stage]))
            if fp_of(t) not in held_fps:
                tasks.append(t)
        p_sched = 0.25 + 0.40 * min(1.0, step / float(args.c2))   # 0.25 -> 0.65
        n_free = max(1, int(args.batch * args.free_frac))

        sig_ids_b = pad_tokens([sig_ids(t) for t in tasks], 16).to(dev)
        z0 = model.encode_tokens(sig_ids_b)
        sp = model.token_emb(sig_ids_b)                            # [B, 16, d]

        out = eng.forward_swarm(z0, b_target=args.batch, steps=6)
        z = out["z"]

        # ---- L1/L2: in-sequence scheduled sampling with FSA mask ----
        tokens = pad_tokens([tokenize_code(t["code"]) for t in tasks]).to(dev)
        logits, masks, ss_frac, raw_ss = model.ss_forward(z, sp, tokens, p_sched)
        # Next-token prediction: logits[t] predicts tokens[t+1]. Position t
        # is ignored when the generated context has FORBIDDEN the teacher
        # token (target logit -1e9): demanding mass there is the 1e9-loss
        # collapse. Realistic divergences teach; impossible ones are skipped.
        with torch.no_grad():
            tgt_ok = masks[:, :-1].gather(
                2, tokens[:, 1:].unsqueeze(-1)).squeeze(-1)          # [B, T-1]
        lg_ss = logits[:, :-1].reshape(-1, VOCAB)
        tg_ss = tokens[:, 1:].reshape(-1)
        ok_ss = tgt_ok.reshape(-1)
        loss_ce = F.cross_entropy(
            lg_ss[ok_ss], tg_ss[ok_ss],
            ignore_index=TOK2ID["PAD"])
        loss_gr = grammar_loss(logits, masks) * cfg.grammar_w
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(-1).mean()

        # ---- L4: masked free-run MLE (the capability loss) ----
        fr_logits, fr_toks, fr_masks, raw_fr = model.free_run_masked(z, sp, tokens)
        with torch.no_grad():
            fr_ok = fr_masks[:, :-1].gather(
                2, tokens[:, 1:].unsqueeze(-1)).squeeze(-1)         # [B, T-1]
        lg_fr = fr_logits[:, :-1].reshape(-1, VOCAB)
        tg_fr = tokens[:, 1:].reshape(-1)
        ok_fr = fr_ok.reshape(-1)
        loss_ce_fr = F.cross_entropy(
            lg_fr[ok_fr], tg_fr[ok_fr],
            ignore_index=TOK2ID["PAD"])
        loss_gr_fr = grammar_loss(fr_logits, fr_masks) * cfg.grammar_w

        # ---- L2: UNK logit-mass suppression (raw logits) ----
        loss_unk = (F.softplus(raw_ss[:, :, TOK2ID["UNK"]]).mean()
                    + F.softplus(raw_fr[:, :, TOK2ID["UNK"]]).mean()) * 0.5 * cfg.unk_w

        # ---- energy regression on shaped rewards of greedy decodes ----
        with torch.no_grad():
            g_ids = model.decode_greedy(z, sp)
        rewards_g = []
        comps_g = []
        for i in range(n_free):
            ids = g_ids[i].tolist()
            code = detokenize(ids)
            r, c = shaped_reward(code, ids, tasks[i])
            rewards_g.append(r); comps_g.append(c)
        rg = torch.tensor(rewards_g, device=dev)
        loss_en = F.binary_cross_entropy(model.energy(z[:n_free]), rg)

        # ---- L5: REINFORCE on sampled decode, stage 2 onward ----
        lam_pg = 0.5 * max(0.0, (step - args.c1) / float(max(1, args.c2 - args.c1)))
        loss_pg = torch.zeros((), device=dev)
        pg_norm = 0.0
        win_pg_norm.append(0)
        if lam_pg > 0:
            toks_s, logps = model.decode_sample(z[:n_free], sp[:n_free])
            rewards_p = []
            for i in range(n_free):
                ids = toks_s[i].tolist()
                code = detokenize(ids)
                r, _ = shaped_reward(code, ids, tasks[i])
                rewards_p.append(r)
            rp = torch.tensor(rewards_p, device=dev)
            adv = ((rp - rp.mean()) / (rp.std() + 1e-8)).clamp(-2.0, 2.0)
            loss_pg = -(logps * adv.unsqueeze(1)).sum(1).mean() * lam_pg
            pg_norm = float(loss_pg.abs().item())

        # ---- greedy free-validity window (L6 gate) ----
        ast_ok = 0
        for i in range(n_free):
            try:
                ast.parse(detokenize(g_ids[i].tolist()))
                ast_ok += 1
            except Exception:
                pass
        win_free_ast.append(ast_ok)
        if len(win_free_ast) > 200:
            win_free_ast.pop(0)
        win_pg_norm.append(pg_norm)
        if len(win_pg_norm) > 100:
            win_pg_norm.pop(0)
        rmean = float(rg.mean().item())
        rmin = float(rg.min().item())
        rmax = float(rg.max().item())
        unique_free = len({detokenize(g_ids[i].tolist()) for i in range(n_free)})
        win_reward.append(rmean)
        if len(win_reward) > 100:
            win_reward.pop(0)

        loss = (0.3 * loss_ce + 0.7 * loss_ce_fr + 0.5 * loss_en
                + loss_gr + loss_gr_fr + loss_unk - 0.01 * entropy + loss_pg)
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        model.core.enforce_stiefel()

        if not torch.isfinite(loss):
            print(f"NAN_AT_STEP {step}: ce={loss_ce.item()} fr={loss_ce_fr.item()} "
                  f"en={loss_en.item()} gr={loss_gr.item()} pg={loss_pg.item()}", flush=True)
            aborted = "NAN_AT_STEP"
            break

        if step % 50 == 0:
            dt = time.time() - t0
            vram = (f" vram={torch.cuda.memory_allocated() / 1e6:.0f}MiB"
                    if str(dev).startswith("cuda") else "")
            print(f"[{step}/{args.steps}] st={stage} ce={loss_ce.item():.3f} "
                  f"fr={loss_ce_fr.item():.3f} "
                  f"en={loss_en.item():.3f} gr={loss_gr.item():.3f} "
                  f"unk={loss_unk.item():.4f} "
                  f"pg={loss_pg.item():.3f} ent={entropy.item():.3f} "
                  f"r={rmean:.3f} [{rmin:.2f},{rmax:.2f}] free_ast={ast_ok}/{n_free} "
                  f"div={unique_free}/{n_free} "
                  f"ss={ss_frac:.2f} gnorm={gn:.2f} B={out['particles']}->{out['b_next']}"
                  f"{vram} t={dt:.0f}s", flush=True)

        if not args.cap_smoke and (step % 500 == 0 or step == args.steps):
            vrep = eval_split(eng, model, dev, valid, swarm_b=16,
                              do_beam=False, do_single=True)
            vrate = vrep["swarm_pass_rate"]
            st = {"model": model.state_dict(), "step": step}
            torch.save(st, ckpt_path)
            print(f"SAVED {ckpt_path} val_swarm_rate={vrate} "
                  f"val_single_rate={vrep['single_pass_rate']}", flush=True)
            if vrate > best_val:
                best_val = vrate
                best_step = step
                torch.save(st, args.out + "/best_val.pt")
                print(f"NEW_BEST_VAL {vrate} at step {step}", flush=True)

        if not args.no_abort:
            if step == args.warmup_step:
                w = sum(win_free_ast)
                if w == 0:
                    print(f"ABORT_NO_VALID_FREE at step {step}: "
                          f"free_ast_window={w}/{len(win_free_ast)}", flush=True)
                    aborted = "ABORT_NO_VALID_FREE"
                    break
            if step % 100 == 0 and step > args.c1:
                rw = win_reward
                if len(rw) == 100 and (max(rw) - min(rw)) < 1e-3 and \
                        sum(rw) / 100 < 0.6:
                    print(f"ABORT_FLAT_REWARD at step {step}: mean={sum(rw)/100:.4f} "
                          f"range={max(rw)-min(rw):.2e}", flush=True)
                    aborted = "ABORT_FLAT_REWARD"
                    break
                if step > args.c1 and sum(win_pg_norm) == 0:
                    print(f"ABORT_PG_DEAD at step {step}: pg_norm_window=0", flush=True)
                    aborted = "ABORT_PG_DEAD"
                    break

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    report = None
    if aborted:
        pathlib.Path(args.out + "/eval.json").write_text(
            json.dumps({"aborted": True, "reason": aborted}, indent=2))
        print(f"ABORTED ({aborted}) - no final eval claim.", flush=True)
    else:
        st = {"model": model.state_dict(), "step": args.steps}
        torch.save(st, ckpt_path)
        if not args.cap_smoke and best_step:
            st = torch.load(args.out + "/best_val.pt", map_location=dev)
            model.load_state_dict(st["model"])
        report = eval_split(eng, model, dev, heldout, swarm_b=args.swarm_b,
                            stochastic=args.stochastic_eval,
                            vote_seed_base=args.seed,
                            beam_width=args.beam_width)
        print(f"EVAL ({eval_tag}):", json.dumps(report), flush=True)
        pathlib.Path(args.out + "/eval.json").write_text(
            json.dumps(report, indent=2))

    receipt = {
        "run": "system1_v04_cap_smoke" if args.cap_smoke else "system1_v04_prod",
        "steps": args.steps, "batch": args.batch, "seed": args.seed,
        "warmup_step": args.warmup_step, "c1": args.c1, "c2": args.c2,
        "swarm_b": args.swarm_b, "free_frac": args.free_frac,
        "reward_weights": {"lex": .10, "bal": .10, "gr": .15, "parse": .20,
                           "sig": .15, "exec": .30},
        "pg_weight_max": 0.5, "ss_schedule": "0.25->0.65 over c2",
        "spec_sha256": {
            "kernel_engine_upload": "d582406c43016466009a144deb5e493fc6b693a88333b184f9d5ad10207460f3",
            "token_fsa_spec_upload": "b42815d356d6b2177309db0723b4726f82d946ab2da29264b01cb1f8fb99cd9e",
        },
        "split_digests": split_digests,
        "best_val_rate": best_val, "best_val_step": best_step,
        "final_telemetry": {
            "reward_mean_last100": round(sum(win_reward) / max(1, len(win_reward)), 4),
            "free_ast_window_last200": sum(win_free_ast),
            "pg_norm_window": round(sum(win_pg_norm) / max(1, len(win_pg_norm)), 4),
        },
        "aborted": aborted, "eval": report,
        "source_sha256": {
            "kernel": hashlib.sha256(
                pathlib.Path(__file__).resolve().parent.joinpath(
                    "system1_kernel_v04.py").read_bytes()).hexdigest(),
            "trainer": hashlib.sha256(
                pathlib.Path(__file__).read_bytes()).hexdigest(),
        },
    }
    pathlib.Path(args.out + "/run_receipt.json").write_text(
        json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
