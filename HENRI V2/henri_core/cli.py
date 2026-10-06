"""HENRI tri-model CLI.

Commands
    info       model registry, parameter counts, import purity
    gates      run every pre-registered gate with negative controls
    train      train the decoder text head (measured, with baselines)
    ask        run the closed loop on a prompt and show the Sagnac verdict
    smoke      one end-to-end CPU pass at small scale

Every command prints JSON. No off-the-shelf component is loaded anywhere.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

# Allow `python henri_core/cli.py` from the HENRI V2 directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from henri_core import gates as G                      # noqa: E402
from henri_core.model3_decoder import (                # noqa: E402
    DecoderConfig, decoder_param_formula)
from henri_core.system import TriModelSystem           # noqa: E402
from henri_core.tokenizer import ByteBPE               # noqa: E402
from henri_core.train import train_text_head           # noqa: E402

CORPUS = [
    "entity acts on object in context",
    "retrieval transfer result on the ladder",
    "the axiom holds the boundary",
    "measuring the energy saddle",
    "phase interference clears the port",
    "the crystal stores the engram",
    "gradient descent reaches the attractor",
    "the swarm finds the lower energy",
    "memory decays with the relaxation time",
    "the veto rejects the inverted phase",
]


def build(small: bool = True, vocab: int = 512) -> tuple:
    """Build the system and a corpus-fitted tokenizer. Deterministic."""
    tok = ByteBPE().train(CORPUS, vocab_size=vocab)
    system = TriModelSystem(vocab=tok.vocab_size, small=small)
    system.eval()
    return system, tok


def _emit(obj: dict):
    print(json.dumps(obj, indent=2, default=str))


# ------------------------------------------------------------------ commands
def cmd_info(args):
    system, tok = build(small=True, vocab=args.vocab)
    full = DecoderConfig()
    formula = decoder_param_formula(full)
    small_cfg = DecoderConfig(dim=4096, d_model=128, n_layers=2, n_heads=4,
                              n_kv_heads=1, d_ffn=256, n_macro=16,
                              n_invariants=32, vocab=args.vocab)
    smoke = decoder_param_formula(small_cfg)
    _emit({
        "schema": "henri.tri-model.info.v1",
        "models": system.model_registry(),
        "tokenizer": {"vocab_size": tok.vocab_size, "merges": len(tok.merges)},
        "decoder_full_config": {
            "params_formula": formula["total"],
            "parts": formula["parts"],
            "doc_target": formula["doc_target"],
            "delta_vs_doc": formula["total"] - formula["doc_target"],
        },
        "decoder_smoke_config_params": smoke["total"],
        "purity": "torch + stdlib only; see gate G-U7",
    })
    return 0


def cmd_gates(args):
    system, tok = build(small=True, vocab=args.vocab)
    full = decoder_param_formula(DecoderConfig())
    smoke_cfg = DecoderConfig(dim=4096, d_model=128, n_layers=2, n_heads=4,
                              n_kv_heads=1, d_ffn=256, n_macro=16,
                              n_invariants=32, vocab=args.vocab)
    small = decoder_param_formula(smoke_cfg)
    t0 = time.time()
    out = G.run_all(system, tok, full["total"], small["total"])
    out["elapsed_s"] = round(time.time() - t0, 2)
    out["schema"] = "henri.tri-model.gates.v1"
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, default=str)
    _emit(out)
    return 0 if out["overall"] == "ACCEPTED" else 1


def cmd_smoke(args):
    t0 = time.time()
    system, tok = build(small=True, vocab=args.vocab)
    bank = system.build_axioms(CORPUS[:4], tok)
    res = system.solve("retrieval transfer result", tok, patterns=bank)
    res["schema"] = "henri.tri-model.smoke.v1"
    res["elapsed_s"] = round(time.time() - t0, 2)
    res["psi_in"] = f"shape={tuple(res['psi_in'].shape)}"
    res["psi_converged"] = f"shape={tuple(res['psi_converged'].shape)}"
    _emit(res)
    return 0


def cmd_train(args):
    system, tok = build(small=True, vocab=args.vocab)
    rep = train_text_head(system, tok, CORPUS, steps=args.steps, lr=args.lr,
                          batch_pairs=args.batch, seed=args.seed,
                          freeze_body=not args.full)
    out = rep.as_dict()
    out["schema"] = "henri.tri-model.train.v1"
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, default=str)
    _emit({k: v for k, v in out.items() if k != "loss_trace"})
    return 0


def cmd_ask(args):
    system, tok = build(small=True, vocab=args.vocab)
    system.build_axioms(CORPUS[:4], tok)
    res = system.solve(args.query, tok, use_swarm=not args.no_swarm)
    res["schema"] = "henri.tri-model.ask.v1"
    res["psi_in"] = f"shape={tuple(res['psi_in'].shape)}"
    res["psi_converged"] = f"shape={tuple(res['psi_converged'].shape)}"
    _emit(res)
    return 0 if res["sagnac"]["allow"] else 2


def cmd_dust(args):
    """Zeroth-order (node-perturbation) descent diagnostic. DEFAULT OFF.

    Additive and opt-in. The import is deliberately INSIDE this function, so the
    module's default import set is unchanged when the flag is absent: the default
    training path stays byte-identical. With the flag absent this command reports
    the disabled state and exits 0. Pass --enable (or set HENRI_DUST_ZO=1) to run
    the G-DUST-1 alignment measurement against autograd.
    """
    from henri_core.dust_zo import dust_zo_enabled
    if not dust_zo_enabled(cli_flag=args.enable):
        _emit({"schema": "henri.dust.zo.v1", "enabled": False, "default": "OFF",
               "module": "henri_core/dust_zo.py",
               "enable_with": "--enable, or HENRI_DUST_ZO=1",
               "note": "no default-path behavior changes while disabled"})
        return 0
    import subprocess
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = os.path.join(here, "henri_core", "exp_dust_g1.py")
    cmd = [sys.executable, script]
    if args.out:
        cmd += ["--out", args.out]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900,
                       cwd=here)
    _emit({"schema": "henri.dust.zo.v1", "enabled": True,
           "returncode": r.returncode,
           "measurement": "G-DUST-1 cos(ZO pseudo-grad, autograd)",
           "stdout_tail": r.stdout[-600:], "stderr_tail": r.stderr[-300:]})
    return r.returncode


def main(argv=None):
    p = argparse.ArgumentParser(prog="henri", description="HENRI tri-model CLI")
    p.add_argument("--vocab", type=int, default=512)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("info", help="model registry and parameter counts")

    g = sub.add_parser("gates", help="run all gates with negative controls")
    g.add_argument("--out", default=None)

    sub.add_parser("smoke", help="one end-to-end CPU pass")

    t = sub.add_parser("train", help="train the decoder text head")
    t.add_argument("--steps", type=int, default=60)
    t.add_argument("--lr", type=float, default=3e-3)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--full", action="store_true",
                   help="train all parameters, not only the heads")
    t.add_argument("--out", default=None)

    a = sub.add_parser("ask", help="run the closed loop on a prompt")
    a.add_argument("--query", required=True)
    a.add_argument("--no-swarm", action="store_true")

    d = sub.add_parser("dust", help="zeroth-order ZO descent diagnostic (default OFF)")
    d.add_argument("--enable", action="store_true",
                   help="run the ZO measurement; default OFF")
    d.add_argument("--out", default=None,
                   help="write the G-DUST-1 receipt JSON here")

    args = p.parse_args(argv)
    return {"info": cmd_info, "gates": cmd_gates, "smoke": cmd_smoke,
            "train": cmd_train, "ask": cmd_ask, "dust": cmd_dust}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
