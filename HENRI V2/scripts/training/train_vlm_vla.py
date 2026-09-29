"""Train the REAL-VLM VLA and compare it against the random-feature baseline.

THE SCIENTIFIC QUESTION
=======================
The measured baseline is a policy whose "vision" is a FROZEN RANDOM projection:
    SUCCESS 0.4688 vs random 0.1875   (d=65536, 3000 steps, RTX 5090)
If real pretrained visual features are better, that is evidence the VLM bridge
adds capability. If they are not better, the honest report is that they are not.

DESIGN (clean comparison)
=========================
  ARM A: baseline        -- frozen random patch projection (already measured)
  ARM B: VLM features    -- FROZEN Qwen3-VL vision tower + trainable attention
                            pool + projection, then the SAME flow-matching head

Only A's ingress is random; B's visual features come from a 4.4B pretrained model.
The action head, data, seeds, and evaluation seeds are IDENTICAL, so any
difference is attributable to the features.

EFFICIENCY
==========
VLM features are extracted ONCE and cached: N_obs x 196 x d_vit float16. The
frozen tower then never runs again during training, so GPU time goes to the
bridge and the head.

HONEST LIMITS
=============
* Synthetic pixel reach task, not a robot benchmark.
* n=64 eval episodes -> binomial SE ~0.06 on a ~0.5 rate. A difference smaller
  than ~0.12 is NOT resolved at this sample size; the script reports the SE and a
  2-SE flag rather than declaring a winner on noise.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from henri_continuous_action_head import ActionNormalizer, FlowMatchingActionHead
from henri_vla_env import N_DOF, PixelReachEnv
from henri_vla_vlm_bridge import VLMFeatureExtractor, VLMPhaseBridge

DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "vlm_vla_receipt.json")


def resolve_out(cli):
    if cli:
        d = os.path.dirname(os.path.abspath(cli))
        if not os.path.isdir(d):
            print(f"MALFORMED --out: directory does not exist: {d}", file=sys.stderr)
            raise SystemExit(2)
        return os.path.abspath(cli)
    envd = os.environ.get("HENRI_RECEIPT_DIR")
    if envd:
        os.makedirs(envd, exist_ok=True)
        return os.path.join(envd, "vlm_vla_receipt.json")
    return DEFAULT_OUT


def collect(n_episodes: int, seed: int = 0):
    env = PixelReachEnv()
    obs_l, act_l = [], []
    for i in range(n_episodes):
        for o, a in env.expert_episode(seed + i):
            obs_l.append(o.cpu())
            act_l.append(a.cpu())
    return torch.stack(obs_l), torch.stack(act_l)


@torch.no_grad()
def cache_vlm_features(ext, obs, batch=8):
    out = [ext.features(obs[i:i + batch]).to(torch.float16).cpu()
           for i in range(0, obs.shape[0], batch)]
    return torch.cat(out, 0)


@torch.no_grad()
def success_rate(ext, bridge, head, norm, n: int, seed: int, device,
                 steps: int = 8, use_vlm: bool = True) -> float:
    env = PixelReachEnv()
    wins = 0
    for i in range(n):
        env.reset(seed + i)
        done = False
        info = {"success": False}
        while not done:
            img = env.observe().unsqueeze(0).to(device)
            if use_vlm:
                psi = bridge(ext.features(img))
            else:
                raise RuntimeError("baseline arm is measured by train_vla_reach.py")
            a = norm.denormalize(head.sample(psi, steps=steps)[0])
            _, _, done, info = env.step(a)
        wins += int(info["success"])
    return wins / n


def se(p, n):
    return math.sqrt(max(p * (1 - p), 1e-12) / n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--train-episodes", type=int, default=192)
    ap.add_argument("--eval-episodes", type=int, default=64)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--d-pool", type=int, default=512)
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.train_episodes, args.steps = 16, 150
        args.eval_episodes, args.eval_every = 16, 150

    out = resolve_out(args.out)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    R: dict = {"schema": "henri.vlm-vla.v1", "device": dev, "config": vars(args),
               "baseline_for_comparison": {
                   "arm": "random frozen patch projection",
                   "success_rate": 0.4688,
                   "random_policy_success": 0.1875,
                   "source": "scripts/training/vla_reach_receipt.json"}}
    if dev != "cuda":
        R["fatal"] = "VLM arm requires CUDA"
        print(json.dumps(R, indent=2)); return 1
    R["gpu"] = {"name": torch.cuda.get_device_name(0),
                "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)}
    print(f"[vlm-vla] loading {args.model}", flush=True)

    # ------------------------------------------------------------- extractor
    t0 = time.time()
    ext = VLMFeatureExtractor(args.model, device=dev).load()
    R["vlm"] = {"model_id": args.model, "tower_path": ext.tower_path,
                "d_vit": ext.d_vit, "n_patches": ext.n_patches,
                "output_key_used": getattr(ext, "output_key_used", "?"),
                "output_keys_seen": getattr(ext, "output_keys_seen", []),
                "needs_grid_thw": getattr(ext, "needs_grid", None),
                "load_seconds": round(time.time() - t0, 1)}
    print(f"[vlm-vla] tower='{ext.tower_path}' d_vit={ext.d_vit} "
          f"n_patches={ext.n_patches}", flush=True)

    # ---------------------------------------------------------------- data
    obs, act = collect(args.train_episodes, seed=0)
    norm = ActionNormalizer(N_DOF).fit(act)
    act_n = norm.normalize(act)
    R["n_train_samples"] = int(obs.shape[0])
    t1 = time.time()
    feats = cache_vlm_features(ext, obs.to(dev))
    R["feature_cache"] = {"shape": list(feats.shape), "dtype": str(feats.dtype),
                          "gib": round(feats.numel() * feats.element_size() / 2 ** 30, 3),
                          "seconds": round(time.time() - t1, 1)}
    print(f"[vlm-vla] cached features {tuple(feats.shape)} {feats.dtype} "
          f"{R['feature_cache']['gib']} GiB in {R['feature_cache']['seconds']}s", flush=True)

    # ------------------------------------------------------------- trainable
    bridge = VLMPhaseBridge(d_vit=ext.d_vit, n_patches=ext.n_patches,
                            d_model=65536, d_pool=args.d_pool).to(dev)
    head = FlowMatchingActionHead(n_dof=N_DOF, d_cond=512, d_hidden=1024,
                                  n_layers=4, phase_dim=65536).to(dev)
    R["bridge_geometry"] = {
        "ext_n_patches": ext.n_patches, "ext_d_vit": ext.d_vit,
        "tokens_are_flattened": getattr(ext, "tokens_are_flattened", None),
        "probe_shape": getattr(ext, "probe_shape", None),
        "pos_codes_shape": list(bridge.pos_codes.shape),
    }
    R["bridge_trainable"] = bridge.trainable_params()
    R["head_trainable"] = sum(p.numel() for p in head.parameters() if p.requires_grad)
    R["vlm_frozen_params"] = sum(p.numel() for p in ext.model.parameters())

    # BINDING-PATH DIAGNOSTIC: a shape mismatch silently falls back to pooled-only
    # training, so prove which path runs BEFORE any GPU time is spent on the sweep.
    with torch.no_grad():
        _probe_psi = bridge(feats[:2].to(dev).to(torch.float32))
    R["binding_active"] = bool(getattr(bridge, "binding_active", False))
    R["binding_ratio"] = round(float(getattr(bridge, "binding_ratio", 0.0)), 6)
    R["n_feats_at_probe"] = int(getattr(bridge, "n_feats", -1))
    R["psi_probe_shape"] = list(_probe_psi.shape)
    R["psi_probe_unit_norm"] = round(float(_probe_psi.norm(dim=-1).mean()), 6)
    print(f"[vlm-vla] binding_active={R['binding_active']} "
          f"ratio={R['binding_ratio']} n_feats={R['n_feats_at_probe']} "
          f"n_patches={ext.n_patches}", flush=True)
    if not R["binding_active"]:
        R["fatal"] = (
            "BINDING PATH INACTIVE -- psi would be pooled-only, so the "
            "compositional role-filler architecture is NOT what this run "
            "measures. Refusing to train rather than report a mislabelled "
            "result (silent-fallback defect class).")
        print("[vlm-vla] FATAL: " + R["fatal"], flush=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(R, fh, indent=2)
        return 3
    params = [p for p in bridge.parameters() if p.requires_grad] + \
             [p for p in head.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)

    # --------------------------------------------------- pre-normalize psi
    # psi is recomputed each step from cached feats; D=65536 at bs=64 is 16 MB,
    # so this fits comfortably without pre-caching the phase vectors.
    N = feats.shape[0]
    curve, evals = [], []
    t2 = time.time()
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, N, (min(args.batch, N),))
        f_b = feats[idx].to(dev).to(torch.float32)
        a_b = act_n[idx].to(dev)
        psi = bridge(f_b)
        loss = head.loss(a_b, psi)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == 1:
            curve.append({"step": step, "loss": round(float(loss.item()), 6)})
        if step % args.eval_every == 0 or step == args.steps:
            sr = success_rate(ext, bridge, head, norm, args.eval_episodes,
                              90000, dev)
            evals.append({"step": step, "success_rate": sr,
                          "se": round(se(sr, args.eval_episodes), 4)})
            print(f"[vlm-vla] step={step:5d} loss={float(loss.item()):.5f} "
                  f"SUCCESS={sr:.4f}", flush=True)
    R["train_seconds"] = round(time.time() - t2, 1)
    R["loss_first"] = curve[0]["loss"] if curve else None
    R["loss_last"] = curve[-1]["loss"] if curve else None
    R["loss_curve"] = curve
    R["closed_loop_evals"] = evals

    # -------------------------------------------------------------- verdicts
    final = evals[-1]["success_rate"] if evals else 0.0
    base = R["baseline_for_comparison"]["success_rate"]
    nb = args.eval_episodes
    diff = final - base
    combined_se = math.sqrt(se(final, nb) ** 2 + se(base, nb) ** 2)
    R["verdicts"] = {
        "vlm_success_rate": final,
        "random_feature_baseline": base,
        "random_policy_baseline": R["baseline_for_comparison"]["random_policy_success"],
        "delta_vs_random_feature_arm": round(diff, 4),
        "combined_se": round(combined_se, 4),
        "vlm_beats_random_arm_by_2se": bool(diff > 2 * combined_se),
        "vlm_beats_random_policy": bool(final > 0.1875),
        "loss_decreased": bool(curve and curve[-1]["loss"] < curve[0]["loss"]),
        "LOOP_CLOSES": bool(curve and curve[-1]["loss"] < curve[0]["loss"]
                            and final > 0.1875),
    }
    R["peak_vram_mib"] = round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
    R["honest_limits"] = [
        "SYNTHETIC pixel reach task; not robot capability, not a benchmark score",
        f"n={nb} eval episodes -> SE ~0.06; a difference under ~0.12 is not resolved",
        "VLM frozen: gains are attributable to features, not vision-path capacity",
        "3-channel 64x64 env images upsampled to 224 -- the VLM sees a synthetic scene",
    ]
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=2)
    print("\n[vlm-vla] WROTE", out, flush=True)
    print("[vlm-vla] VERDICTS:", json.dumps(R["verdicts"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
