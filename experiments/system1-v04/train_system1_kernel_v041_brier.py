"""
System-1 v0.4.1 trainer - Binary Brier outcome-energy calibration.
=========================================================================
Attribution experiment (pre-registered 2026-08-23):

  FREEZE the proven v0.4 decoder/FSA/core. Change ONLY the energy-head
  supervision. Ask: does binary sandbox-outcome Brier training restore
  energy -> external-pass rank correlation (raw Spearman rho > 0)?

v0.4 defect (OBSERVED): energy regressed on SHAPED REWARD of greedy
decodes -> raw-energy/pass Spearman = 0.0, winner mass 85-128/128, vote
~= greedy, delta_vs_single = 0.0. The head learned the reward shape, not
the outcome.

v0.4.1 mechanism (per Drive inbox upload 2026-08-23 15:46, engine sha
92208ed0...; audit: BrierOutcomeBaseline ALREADY in live v0.4 kernel -
the change is the TRAINING OBJECTIVE, not the head):
  - per-step: sample n_free candidate programs (seeded decode_sample,
    decoder FROZEN), execute the REAL sandbox -> binary y in {0,1}
  - Brier loss: (E_phi(z) - y)^2 over sampled candidates
  - class-balance monitor: abort if either class vanishes (calibration
    untestable)
  - final eval on a FRESH disposable split (dev41_v04, seed 42+66661)
    NEVER on the consumed heldout40_v04.

Gates (pre-registered):
  PRIMARY   raw Spearman rho > 0 over per-particle (energy, pass) pairs,
            both classes present, n >= 20 pairs.
  PROMOTION rho > 0 AND permutation p < 0.05 AND Brier < constant
            baseline AND AUROC > 0.5 AND energy-vote > matched single
            control (delta >= 0.10, McNemar p < 0.05) AND ast_valid
            preserved (>= 0.9).
  ABORT     NaN/Inf; missing outcome class over 200-step window;
            energy collapse (variance -> 0 sustained).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import random
import sys
import time

import torch
import torch.nn.functional as F

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
# v0.4.1 kernel: byte-identical arch to audited v0.4 kernel (head already
# present); re-issued under the v0.4.1 name per audit disposition.
from system1_kernel_v041_energy_refactored import (  # noqa: E402
    VOCAB, TOK2ID, KernelV04Config, System1KernelV04, SwarmEngineV04,
    grammar_loss, tokenize_code, detokenize)
# Proven helpers: reuse, do not copy.
from train_system1_kernel_v04 import (  # noqa: E402
    gen_task, sandbox, fp_of, sig_ids, pad_tokens, sig_matrix,
    load_split, sha256_file, eval_split, wilson)


# ---------------------------------------------------------------------------
# Calibration evaluation: per-particle (energy, pass) pairs
# ---------------------------------------------------------------------------
def _spearman_raw(xs: list[float], ys: list[int]) -> float:
    """Spearman rho between RAW energy (higher = more confident pass) and
    binary outcome. Positive rho means higher energy predicts pass."""
    n = len(xs)
    if n < 3:
        return 0.0
    rx = {v: i for i, v in enumerate(sorted(set(xs)))}
    ry = {v: i for i, v in enumerate(sorted(set(ys)))}
    a = [rx[v] for v in xs]
    b = [ry[v] for v in ys]
    mx = sum(a) / n
    my = sum(b) / n
    num = sum((u - mx) * (v - my) for u, v in zip(a, b))
    dx = math.sqrt(sum((u - mx) ** 2 for u in a))
    dy = math.sqrt(sum((v - my) ** 2 for v in b))
    return 0.0 if dx == 0 or dy == 0 else num / (dx * dy)


def _auroc(e_pos: list[float], e_neg: list[float]) -> float:
    """P(energy_pos > energy_neg) via Mann-Whitney U; 0.5 = chance."""
    if not e_pos or not e_neg:
        return 0.5
    n, m = len(e_pos), len(e_neg)
    gt = sum(1.0 for a in e_pos for b in e_neg if a > b)
    eq = sum(0.5 for a in e_pos for b in e_neg if a == b)
    return (gt + eq) / (n * m)


def _perm_rho_p(xs: list[float], ys: list[int], seed: int = 42,
                n_perm: int = 2000) -> float:
    """Two-sided permutation p for raw Spearman vs shuffled outcomes."""
    rng = random.Random(seed)
    obs = _spearman_raw(xs, ys)
    if obs == 0.0:
        return 1.0
    cnt = 0
    yl = list(ys)
    for _ in range(n_perm):
        rng.shuffle(yl)
        pr = _spearman_raw(xs, yl)
        if abs(pr) >= abs(obs):
            cnt += 1
    return cnt / n_perm


@torch.no_grad()
def eval_calibration(eng: SwarmEngineV04, model: System1KernelV04, dev,
                     tasks: list[dict], k: int = 16,
                     seed_base: int = 0) -> dict:
    """Per-particle energy/outcome pairs over a split. k particles/task."""
    n = len(tasks)
    e_all: list[float] = []
    y_all: list[int] = []
    ast_ok = 0
    votes = 0
    for t_idx, t in enumerate(tasks):
        sp_sw = sig_matrix(model, [t] * k, 16, dev)
        z0 = model.encode_tokens(pad_tokens([sig_ids(t)] * k, 16).to(dev))
        out = eng.forward_swarm(z0, b_target=k, steps=8)
        e = out["energy"]                              # [k]
        s_ids, s_rec = model.decode_vote(
            out["z"], sp_sw, e, seed_base=seed_base + t_idx * 7 + 1)
        code_v = detokenize(s_ids)
        votes += sandbox(code_v, t["tests"])
        try:
            import ast
            ast.parse(code_v)
            ast_ok += 1
        except Exception:
            pass
        for i in range(k):
            pids, _ = model.decode_sample(
                out["z"][i:i + 1], sp_sw[i:i + 1],
                seed=seed_base + t_idx * 31 + i + 1)
            code_p = detokenize(pids[0].tolist())
            e_all.append(float(e[i].item()))
            y_all.append(int(sandbox(code_p, t["tests"])))
    pos = sum(y_all)
    neg = len(y_all) - pos
    rho = _spearman_raw(e_all, y_all)
    p_perm = _perm_rho_p(e_all, y_all, seed=seed_base)
    e_pos = [e for e, y in zip(e_all, y_all) if y == 1]
    e_neg = [e for e, y in zip(e_all, y_all) if y == 0]
    auroc = _auroc(e_pos, e_neg)
    brier = sum((e - y) ** 2 for e, y in zip(e_all, y_all)) / len(y_all)
    pbar = pos / len(y_all)
    baseline = pbar * (1.0 - pbar)                      # constant predictor
    return {
        "n_pairs": len(y_all), "pos": pos, "neg": neg,
        "pos_rate": round(pos / max(1, len(y_all)), 4),
        "spearman_raw": round(rho, 4),
        "perm_p": round(p_perm, 4),
        "auroc": round(auroc, 4),
        "brier": round(brier, 4),
        "baseline_brier": round(baseline, 4),
        "brier_improve": round(baseline - brier, 4),
        "vote_pass": votes, "vote_rate": round(votes / n, 4),
        "ast_valid_rate": round(ast_ok / n, 4),
        "energy_mean": round(sum(e_all) / len(e_all), 4),
        "energy_std": round((sum((e - sum(e_all) / len(e_all)) ** 2
                                 for e in e_all) / len(e_all)) ** 0.5, 4),
        "both_classes": pos > 0 and neg > 0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--n-free", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--out", default="/root/henri-system1/ckpt_v041")
    ap.add_argument("--ckpt", default="/root/henri-system1/ckpt_v04/best_val.pt",
                    help="pinned v0.4 checkpoint to load (decoder frozen)")
    ap.add_argument("--swarm-b", type=int, default=128)
    ap.add_argument("--dev-n", type=int, default=40)
    ap.add_argument("--dev-seed", type=int, default=42 + 66661)
    ap.add_argument("--no-abort", action="store_true",
                    help="telemetry only (plumbing smoke)")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()
    pathlib.Path(args.out).mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    dev = args.device
    cfg = KernelV04Config()
    model = System1KernelV04(cfg=cfg).to(dev)
    eng = SwarmEngineV04(model).to(dev)

    # Load the pinned v0.4 checkpoint (step 500 best_val, decoder proven).
    st = torch.load(args.ckpt, map_location=dev)
    model.load_state_dict(st["model"])
    print(f"LOADED {args.ckpt} step={st.get('step')}", flush=True)

    # Freeze everything except the energy head (attribution isolation).
    for p in model.parameters():
        p.requires_grad = False
    model.energy.requires_grad_(True)
    n_en = sum(p.numel() for p in model.energy.parameters())
    print(f"FROZEN all except energy head ({n_en / 1e3:.1f}K params)",
          flush=True)

    # Fresh disposable dev split (new seed, NEVER heldout/smoke).
    dev_tasks = load_split(args.out, args.dev_n, args.dev_seed, "dev41_v04")
    held_fps = {fp_of(t) for t in dev_tasks}
    split_p = pathlib.Path(args.out) / "dev41_v04.json"
    split_digest = sha256_file(split_p)
    print(f"DEV_SPLIT dev41_v04 n={args.dev_n} sha={split_digest[:16]}",
          flush=True)

    if args.eval_only:
        report = eval_calibration(eng, model, dev, dev_tasks,
                                  seed_base=args.seed)
        cmp = eval_split(eng, model, dev, dev_tasks, swarm_b=args.swarm_b,
                         stochastic=True, vote_seed_base=args.seed,
                         beam_width=args.swarm_b)
        report["vote_comparison"] = {
            "swarm_pass": cmp["swarm_pass"], "single_pass": cmp["single_pass"],
            "beam_pass": cmp["beam_pass"], "greedy_pass": cmp["greedy_pass"],
            "delta_vs_single": cmp["delta_vs_single"],
            "mcnemar_p": cmp["mcnemar_p"],
            "transitions": cmp["transitions"],
            "engagement": cmp.get("gates"),
        }
        print("CALIBRATION:", json.dumps(report), flush=True)
        pathlib.Path(args.out + "/eval_calibration.json").write_text(
            json.dumps(report, indent=2))
        return

    opt = torch.optim.AdamW(model.energy.parameters(), lr=args.lr,
                            weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.steps, eta_min=1e-5)
    rng = random.Random(args.seed)
    t0 = time.time()
    aborted = None
    win_pos = []
    win_energy_var = []
    best_brier = 1e9

    for step in range(1, args.steps + 1):
        tasks = []
        while len(tasks) < args.batch:
            t = gen_task(rng)
            if fp_of(t) not in held_fps:
                tasks.append(t)
        n_free = min(args.n_free, args.batch)

        sig_ids_b = pad_tokens([sig_ids(t) for t in tasks], 16).to(dev)
        z0 = model.encode_tokens(sig_ids_b)
        sp = model.token_emb(sig_ids_b)
        with torch.no_grad():
            out = eng.forward_swarm(z0, b_target=args.batch, steps=6)
            toks_s, _ = model.decode_sample(
                out["z"][:n_free], sp[:n_free],
                seed=args.seed + step * 1000)

        # Real sandbox binary labels for sampled candidates.
        ys = []
        for i in range(n_free):
            code = detokenize(toks_s[i].tolist())
            ys.append(sandbox(code, tasks[i]["tests"]))
        y = torch.tensor(ys, dtype=torch.float32, device=dev)

        e = model.energy(out["z"][:n_free])              # trainable path
        loss_brier = ((e - y) ** 2).mean()               # Brier score
        opt.zero_grad()
        loss_brier.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.energy.parameters(), 1.0)
        opt.step()
        sched.step()

        pos_rate = sum(ys) / n_free
        win_pos.append(pos_rate)
        if len(win_pos) > 200:
            win_pos.pop(0)
        ev = float(e.var().item())
        win_energy_var.append(ev)
        if len(win_energy_var) > 200:
            win_energy_var.pop(0)

        if not torch.isfinite(loss_brier):
            print(f"NAN_AT_STEP {step}: brier={loss_brier.item()}", flush=True)
            aborted = "NAN_AT_STEP"
            break

        if step % 50 == 0:
            vram = (f" vram={torch.cuda.memory_allocated() / 1e6:.0f}MiB"
                    if str(dev).startswith("cuda") else "")
            print(f"[{step}/{args.steps}] brier={loss_brier.item():.4f} "
                  f"pos={sum(ys)}/{n_free} e_var={ev:.4f} "
                  f"gnorm={gn:.2f}{vram} t={time.time() - t0:.0f}s",
                  flush=True)

        if not args.no_abort:
            if len(win_pos) == 200 and (0.0 in win_pos or 1.0 in win_pos) \
                    and (sum(win_pos) / 200 < 0.02 or
                         sum(win_pos) / 200 > 0.98):
                print(f"ABORT_MISSING_CLASS at step {step}: "
                      f"pos_rate_window={sum(win_pos) / 200:.3f}", flush=True)
                aborted = "ABORT_MISSING_CLASS"
                break
            if len(win_energy_var) == 200 and \
                    max(win_energy_var) < 1e-4:
                print(f"ABORT_ENERGY_COLLAPSE at step {step}: "
                      f"e_var_window_max={max(win_energy_var):.2e}",
                      flush=True)
                aborted = "ABORT_ENERGY_COLLAPSE"
                break

    st = {"model": model.state_dict(), "step": args.steps,
          "energy_only": True, "ckpt_loaded": args.ckpt,
          "loaded_step": st.get("step")}
    torch.save(st, args.out + "/checkpoint.pt")

    # Single final eval on the fresh disposable dev split.
    report = eval_calibration(eng, model, dev, dev_tasks, seed_base=args.seed)
    cmp = eval_split(eng, model, dev, dev_tasks, swarm_b=args.swarm_b,
                     stochastic=True, vote_seed_base=args.seed,
                     beam_width=args.swarm_b)
    report["vote_comparison"] = {
        "swarm_pass": cmp["swarm_pass"], "single_pass": cmp["single_pass"],
        "beam_pass": cmp["beam_pass"], "greedy_pass": cmp["greedy_pass"],
        "delta_vs_single": cmp["delta_vs_single"],
        "mcnemar_p": cmp["mcnemar_p"],
        "transitions": cmp["transitions"],
        "engagement": cmp.get("gates"),
    }
    print("CALIBRATION:", json.dumps(report), flush=True)

    primary = (report["both_classes"] and report["n_pairs"] >= 20
               and report["spearman_raw"] > 0.0)
    promo = (primary and report["perm_p"] < 0.05
             and report["brier_improve"] > 0.0
             and report["auroc"] > 0.5
             and report["vote_comparison"]["delta_vs_single"] >= 0.10
             and report["vote_comparison"]["mcnemar_p"] < 0.05
             and report["ast_valid_rate"] >= 0.9)
    report["primary_gate_pass"] = bool(primary)
    report["promotion_gate_pass"] = bool(promo)
    report["kill_fired"] = not promo
    report["diagnostic_only"] = not promo
    report["aborted"] = aborted

    pathlib.Path(args.out + "/eval_calibration.json").write_text(
        json.dumps(report, indent=2))

    receipt = {
        "run": "system1_v041_brier_energy",
        "steps": args.steps, "batch": args.batch, "n_free": args.n_free,
        "seed": args.seed, "lr": args.lr,
        "ckpt_loaded": args.ckpt, "loaded_step": st.get("step"),
        "energy_only_trainable": True,
        "dev_split": {"tag": "dev41_v04", "n": args.dev_n,
                      "seed": args.dev_seed, "sha256": split_digest},
        "source_sha256": {
            "kernel_v041": sha256_file(
                _HERE / "system1_kernel_v041_energy_refactored.py"),
            "trainer_v041": sha256_file(pathlib.Path(__file__)),
            "kernel_v04_audited": sha256_file(
                _HERE / "system1_kernel_v04.py"),
            "trainer_v04": sha256_file(
                _HERE / "train_system1_kernel_v04.py"),
            "engine_upload_v041":
                "92208ed0c2a64de56c61e112a1f7b13d6c3eec1f2cb010de3ef8824c2cbc10c4",
        },
        "aborted": aborted,
        "eval": report,
    }
    pathlib.Path(args.out + "/run_receipt.json").write_text(
        json.dumps(receipt, indent=2))
    print(f"RECEIPT {args.out}/run_receipt.json", flush=True)


if __name__ == "__main__":
    main()
