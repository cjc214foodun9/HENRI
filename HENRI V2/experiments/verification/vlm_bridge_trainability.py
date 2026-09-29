"""DECISIVE EXPERIMENT: is psi uninformative at INIT, or only because it never trained?

WHY THIS EXISTS
===============
Measured on the RTX 5090 (commits 3573f4f, 604a8bf):
    VLM arm              SUCCESS 0.078125   loss 1.345031 -> 1.377983 (flat)
    random-feature arm   SUCCESS 0.4688     <- the arm it was meant to beat
    random policy        0.1875
    pooled VLM features  r2 0.7559          <- state IS linearly readable
    psi after the bridge r2 0.0509          <- readability gone

Three explanations were proposed, tested, and FALSIFIED:
    H1 DC/bias in the projection stack      (bias=False: 0.0691 -> 0.0509, no gain)
    H3 L2-normalisation crushes deviation   (matched geometry: psi keeps r2 0.9843)
    H4 fixed-penalty readout artifact       (fixed readout gives 0.7206, not broken)

What SURVIVES is a confound I named and did not resolve: the diagnostic measured a
FRESH UNTRAINED bridge, so r2 0.0509 describes a RANDOM 1024->512->65536
bottleneck. A random init and a broken architecture predict the same number.

THIS SCRIPT SEPARATES THEM -- the cheapest test that can:
  real frozen Qwen3-VL features -> TRAIN bridge + flow head -> re-measure
  readability with the PROVEN scale-aware readout -> closed-loop success.

PRE-REGISTERED VERDICTS (fixed before running; NOT adjusted afterwards)
  binding_active_required : binding must be active or the run refuses (exit 3)
  bridge_trains           : loss_last <= 0.5 * loss_first
  readability_after_train : r2_std(psi_after) >= 0.50
  psi_variation_after     : mean off-diag cos(psi_after) <= 0.50
  beats_random_2se        : succ > 0.1875 + 2*SE
  BRIDGE_IS_THE_BOTTLENECK: readability stays low AND loss flat
  TASK_IS_THE_BOTTLENECK  : loss flat BUT pooled features readable
                            -> the HEAD cannot exploit the condition it is given
  VLM_FEATURES_HELP       : beats the 0.4688 random-feature arm by 2*SE

ENV CONTRACT (read from henri_vla_env.py, not assumed)
  reset(seed) -> obs [3,H,W] float in [0,1]
  step(action[n_dof]) -> (obs, reward, done, info); action[:2] clamped to [-1,1]
    THEN scaled by cfg.step_size, so the expert target is the UNIT direction --
    multiplying it by step_size again (as this script first did) is a 12.5x error.
  expert_action() -> [n_dof] unit direction toward the goal. Use it directly.

HONEST SCOPE: synthetic pixel task; not robot capability; not a benchmark score.
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.abspath(os.path.join(HERE, "..", ".."))
if V2 not in sys.path:
    sys.path.insert(0, V2)

RANDOM_POLICY_BASELINE = 0.1875
RANDOM_FEATURE_BASELINE = 0.4688
D_MODEL = 65536


# ------------------------------------------------------------------- readout
def r2_std(x: torch.Tensor, y: torch.Tensor, frac: float = 0.25,
           lam_rel: float = 1e-3) -> float:
    """Held-out R^2, scale-aware (instrument proven by test_readout_instrument.py).

    Standardises features and regularises RELATIVE to n, so a unit-norm
    near-collinear representation is not spuriously reported as uninformative.
    """
    n = x.shape[0]
    if n < 8:
        return float("nan")
    nte = max(1, int(n * frac))
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    te, tr = perm[:nte], perm[nte:]
    xtr, xte = x[tr].double(), x[te].double()
    ytr, yte = y[tr].double(), y[te].double()
    xm, ym = xtr.mean(0, keepdim=True), ytr.mean(0, keepdim=True)
    s = xtr.std(0, keepdim=True).mean().clamp(min=1e-12)
    xtr, xte = (xtr - xm) / s, (xte - xm) / s
    ytr, yte = ytr - ym, yte - ym
    k = xtr @ xtr.T
    a = torch.linalg.solve(k + lam_rel * n * torch.eye(k.shape[0], dtype=k.dtype), ytr)
    pred = (xte @ xtr.T) @ a
    res = ((yte - pred) ** 2).sum(0)
    tot = (yte ** 2).sum(0).clamp(min=1e-12)
    return float((1.0 - res / tot).mean())


def offdiag_cos(f: torch.Tensor) -> float:
    c = F.normalize(f.reshape(f.shape[0], -1).double(), dim=-1)
    c = c @ c.T
    n = c.shape[0]
    if n < 2:
        return float("nan")
    return float((c.sum() - c.diag().sum()) / (n * (n - 1)))


# ------------------------------------------------------------------- helpers
def _img_from(o: torch.Tensor) -> torch.Tensor:
    """env.reset/step obs -> [3,H,W] float in [0,1] on CPU."""
    img = o.detach().float()
    if img.dim() == 4:
        img = img[0]
    if img.shape[-1] in (1, 3):          # HWC -> CHW
        img = img.permute(2, 0, 1)
    if float(img.max()) > 2.0:
        img = img / 255.0
    return img.clamp(0, 1).contiguous()


def _state_of(env) -> torch.Tensor:
    return torch.cat([env.ee.detach().float().reshape(-1)[:2],
                      env.goal.detach().float().reshape(-1)[:2]]).contiguous()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=128)
    ap.add_argument("--max-steps", type=int, default=24)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--eval-episodes", type=int, default=48)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-pool", type=int, default=512)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--out", default="/root/henri/vlm_bridge_trainability.json")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    R = {"schema": "henri.vlm_bridge_trainability/1",
         "when": time.strftime("%Y-%m-%d %H:%M:%S"), "device": dev,
         "device_name": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
         "config": vars(a),
         "baselines": {"random_policy": RANDOM_POLICY_BASELINE,
                       "random_feature_arm": RANDOM_FEATURE_BASELINE},
         "pre_registered": {
             "bridge_trains": "loss_last <= 0.5 * loss_first",
             "readability_after_train": "r2_std(psi_after) >= 0.50",
             "psi_variation_after": "mean off-diag cos(psi_after) <= 0.50",
             "beats_random_2se": f"succ > {RANDOM_POLICY_BASELINE} + 2*SE",
             "beats_random_feature_2se": f"succ > {RANDOM_FEATURE_BASELINE} + 2*SE",
             "BRIDGE_IS_THE_BOTTLENECK": "readability low AND loss flat",
             "TASK_IS_THE_BOTTLENECK": "loss flat BUT pooled features readable",
             "VLM_FEATURES_HELP": "beats random-feature arm by 2*SE"}}

    # ---------------------------------------------------------------- 1. env
    import henri_vla_env as E
    env = E.PixelReachEnv()
    R["env_class"] = type(env).__name__
    R["env"] = {"n_dof": int(env.n_dof), "horizon": int(env.cfg.horizon),
                "step_size": float(env.cfg.step_size), "tol": float(env.cfg.tol)}

    # ------------------------------- 2. expert demos (stop at env.done) ------
    t0 = time.time()
    obs, st, act = [], [], []
    for e in range(a.episodes):
        o = env.reset(2000 + e)
        k = 0
        while (not env.done) and k < a.max_steps:
            obs.append(_img_from(o))
            st.append(_state_of(env))
            # UNIT direction; env.step applies cfg.step_size itself.
            aa = env.expert_action().detach().float()
            act.append(aa)
            o, _r, _d, _i = env.step(aa)
            k += 1
    O = torch.stack(obs)
    S = torch.stack(st)
    A = torch.stack(act)
    n_dof = int(A.shape[-1])
    R["data"] = {"n_samples": int(O.shape[0]), "obs_shape": list(O.shape[1:]),
                 "n_dof": n_dof, "steps_per_episode_mean":
                 round(O.shape[0] / max(a.episodes, 1), 2),
                 "state_span": [round(float(S.min()), 4), round(float(S.max()), 4)],
                 "collect_seconds": round(time.time() - t0, 1)}
    R["action_norm_mean"] = round(float(A.norm(dim=-1).mean()), 6)
    R["action_norm_max"] = round(float(A.norm(dim=-1).max()), 6)

    # ------------------------------------------------- 3. frozen VLM features
    import henri_vla_vlm_bridge as B
    t0 = time.time()
    ext = B.VLMFeatureExtractor(a.model, device=dev, dtype=torch.bfloat16).load()
    big = F.interpolate(O.to(dev).to(torch.float32), size=(224, 224),
                        mode="bilinear", align_corners=False)
    feats = ext.features(big, batch=8).float()
    R["vlm"] = {"model_id": a.model, "tower_path": ext.tower_path,
                "d_vit": ext.d_vit, "n_patches": ext.n_patches,
                "output_key_used": getattr(ext, "output_key_used", None),
                "frozen_params": sum(p.numel() for p in ext.model.parameters()),
                "feat_shape": list(feats.shape),
                "extract_seconds": round(time.time() - t0, 1)}
    pooled = feats.mean(dim=1)
    R["pooled"] = {"r2_std": round(r2_std(pooled.cpu(), S), 4),
                   "mean_offdiag_cos": round(offdiag_cos(pooled.cpu()), 8)}

    # -------------------------------------- 4. bridge + head + normalizer ----
    import henri_continuous_action_head as H
    bridge = B.VLMPhaseBridge(d_vit=int(feats.shape[-1]), n_patches=int(feats.shape[1]),
                              d_model=D_MODEL, d_pool=a.d_pool).to(dev)
    head = H.FlowMatchingActionHead(n_dof=n_dof, d_cond=512, d_hidden=1024,
                                    n_layers=4, phase_dim=D_MODEL).to(dev)
    norm = H.ActionNormalizer(n_dof=n_dof).fit(A)
    R["bridge_trainable"] = bridge.trainable_params()
    R["head_trainable"] = sum(p.numel() for p in head.parameters() if p.requires_grad)

    # `binding_active` and `binding_ratio` are populated by forward(), so run a
    # probe pass FIRST and read the gate from it. Reading them before any forward
    # call would raise AttributeError (defect in the first revision of this file).
    m = min(256, feats.shape[0])
    ftr = feats[:m].to(dev)
    with torch.no_grad():
        psi0 = bridge(ftr).float().cpu()
    R["binding_active"] = bool(getattr(bridge, "binding_active", False))
    R["binding_ratio"] = round(float(getattr(bridge, "binding_ratio", 0.0)), 6)
    R["n_feats_at_probe"] = int(getattr(bridge, "n_feats", -1))
    R["n_patches"] = int(bridge.n_patches)
    if not R["binding_active"]:
        R["fatal"] = ("BINDING PATH INACTIVE -- psi would be pooled-only, so this "
                      "run would measure a different architecture than it claims. "
                      "Refusing to train on a mislabelled result.")
        json.dump(R, open(a.out, "w"), indent=2)
        print(json.dumps(R, indent=2))
        return 3

    R["psi_at_init"] = {"r2_std": round(r2_std(psi0, S[:m]), 4),
                        "mean_offdiag_cos": round(offdiag_cos(psi0), 8)}

    # ------------------------------------------------------------- 5. train
    An = norm.normalize(A)
    params = [p for p in bridge.parameters() if p.requires_grad] + \
             [p for p in head.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr)
    Ftr, Atr = feats.to(dev), An.to(dev)
    n = Ftr.shape[0]
    g = torch.Generator().manual_seed(0)
    first = last = None
    curve = []
    t0 = time.time()
    bridge.train(); head.train()
    for step in range(1, a.steps + 1):
        idx = torch.randint(0, n, (a.batch,), generator=g)
        psi = bridge(Ftr[idx])
        loss = head.loss(Atr[idx], psi)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        v = float(loss.detach())
        if first is None:
            first = v
        last = v
        if step == 1 or step % max(1, a.steps // 10) == 0:
            curve.append({"step": step, "loss": round(v, 6)})
            print(f"[train] step {step:5d}  loss {v:.6f}", flush=True)
    R["train"] = {"steps": a.steps, "batch": a.batch, "lr": a.lr,
                  "loss_first": round(first, 6), "loss_last": round(last, 6),
                  "ratio_last_over_first": round(last / max(first, 1e-12), 4),
                  "curve": curve, "seconds": round(time.time() - t0, 1),
                  "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1)
                  if dev == "cuda" else None}

    # ------------------------------------ 6. readability AFTER training ------
    bridge.eval()
    with torch.no_grad():
        psi1 = bridge(ftr).float().cpu()
    R["psi_after_training"] = {"r2_std": round(r2_std(psi1, S[:m]), 4),
                               "mean_offdiag_cos": round(offdiag_cos(psi1), 8)}
    R["psi_readability_delta"] = round(
        R["psi_after_training"]["r2_std"] - R["psi_at_init"]["r2_std"], 4)

    # ------------------------------------------ 7. closed-loop evaluation ----
    head.eval()
    wins = 0
    eps = []
    t0 = time.time()
    for e in range(a.eval_episodes):
        o = env.reset(9000 + e)
        k = 0
        while (not env.done) and k < int(env.cfg.horizon):
            img = _img_from(o)[None].to(dev).to(torch.float32)
            x224 = F.interpolate(img, size=(224, 224), mode="bilinear",
                                 align_corners=False)
            with torch.no_grad():
                f = ext.features(x224, batch=1)      # [1,N,d_vit]
                psi = bridge(f)
                a_norm = head.sample(psi, steps=8)   # normalized units
                aa = norm.denormalize(a_norm)[0]
            o, _r, _d, _i = env.step(aa)
            k += 1
        wins += int(bool(env.success))
        eps.append({"seed": 9000 + e, "success": bool(env.success), "steps": k})
    succ = wins / max(a.eval_episodes, 1)
    # Binomial standard error of a success rate. (An earlier revision of this line
    # contained a malformed expression that failed to compile; this is the fix.)
    se = (succ * (1.0 - succ) / max(a.eval_episodes, 1)) ** 0.5
    R["eval"] = {"episodes": a.eval_episodes, "wins": wins,
                 "success_rate": round(succ, 6), "se": round(se, 6),
                 "seconds": round(time.time() - t0, 1),
                 "steps_per_episode_mean": round(
                     sum(x["steps"] for x in eps) / max(len(eps), 1), 2),
                 "trajectory": eps[:16]}
    R["peak_vram_mib"] = (round(torch.cuda.max_memory_allocated() / 2**20, 1)
                          if dev == "cuda" else None)

    # ------------------------------------------------------- 8. verdicts ----
    rd = R["psi_after_training"]["r2_std"]
    lf = R["train"]["ratio_last_over_first"]
    R["verdicts"] = {
        "binding_active": bool(R["binding_active"]),
        "bridge_trains": bool(lf <= 0.5),
        "readability_after_train": bool(rd >= 0.50),
        "psi_variation_after": bool(R["psi_after_training"]["mean_offdiag_cos"] <= 0.50),
        "beats_random_2se": bool(succ > RANDOM_POLICY_BASELINE + 2 * se),
        "beats_random_feature_2se": bool(succ > RANDOM_FEATURE_BASELINE + 2 * se),
        "BRIDGE_IS_THE_BOTTLENECK": bool(rd < 0.50 and lf > 0.5),
        "TASK_IS_THE_BOTTLENECK": bool(lf > 0.5 and R["pooled"]["r2_std"] >= 0.50),
        "VLM_FEATURES_HELP": bool(succ > RANDOM_FEATURE_BASELINE + 2 * se),
    }
    R["honest_limits"] = [
        "SYNTHETIC pixel reach task; not robot capability, not a benchmark score",
        f"n={a.eval_episodes} eval episodes -> SE ~{round(se, 3)}; differences below "
        f"~{round(2 * se, 3)} are unresolved",
        "VLM frozen; any gain is attributable to features, not vision-path capacity",
        "64x64 three-channel env images upsampled to 224 -- OOD for the VLM",
        "comparison to the random-feature arm is cross-run (0.4688 measured earlier)",
    ]
    json.dump(R, open(a.out, "w"), indent=2)
    print(json.dumps({k: R[k] for k in
                      ("binding_active", "binding_ratio", "pooled", "psi_at_init",
                       "psi_after_training", "psi_readability_delta", "train",
                       "eval", "verdicts", "peak_vram_mib")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
