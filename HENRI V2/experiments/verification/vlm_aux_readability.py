"""AUXILIARY-READABILITY RETRAIN -- the fix the stage probe points at.

MEASURED EVIDENCE THIS ADDRESSES
  stage probe      (commit 261ed28): FRESH bridge keeps psi r2 0.3949 (input 0.7249)
  trainability     (commit 6280aee): TRAINED bridge psi r2 -0.5
  pooled input     : r2 0.7249-0.7999 across runs
  dim-matched ctl  : random 512->65536 unit-norm keeps r2 0.6432, so generic
                     high-dimensional geometry does NOT explain psi's loss
=> The forward path is not the destroyer. TRAINING is: the flow-matching objective
   contains no term requiring psi to stay readable, so the optimiser is free to
   collapse it onto whatever minimises action error.

FOUR ARMS -- identical data, identical init seed, identical budget
  CONTROL : L = flow                                     (reproduces the failure)
  PROBE   : L = flow + lam_p * MSE(W psi, s_std)          (privileged state aux)
  GEOM    : L = flow + lam_g * mean_offdiag_cos(psi)^2     (fully unsupervised)
  BOTH    : both auxiliary terms

The PROBE arm uses the 4-D state as auxiliary supervision ONLY during training. At
deployment psi is computed from pixels alone and the probe head is discarded, so no
privileged information reaches the policy. That is standard auxiliary supervision,
and it is legitimate here because the state is already linearly readable from the
pooled features (r2 ~0.75-0.80) -- the loss only asks psi to PRESERVE information
its input already carries.

PRE-REGISTERED (fixed before running; not adjusted afterwards)
  control_collapses       : CONTROL psi r2 < 0.20
  readability_preserved_X : arm X psi r2 >= 0.50
  FIX_CONFIRMED           : some auxiliary arm has r2 >= 0.50 AND
                            success >= CONTROL success + 2*SE
  NO_FIX                  : every auxiliary arm stays below 0.50 (fix is wrong)

STATED LIMIT, fixed in advance: the closed-loop success comparison is UNDERPOWERED
at n=48 eval episodes (2*SE ~ 0.12, far larger than the differences seen so far).
Readability (r2_std) is the PRIMARY endpoint; success is secondary and will be
reported with its interval rather than as a verdict.

HONEST SCOPE: synthetic pixel task; not robot capability; not a benchmark score.
Sample count is matched to the random-feature arm (3072) so that comparison is at
last apples-to-apples -- the earlier 588-vs-3072 mismatch is corrected here.
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.abspath(os.path.join(HERE, "..", ".."))
if V2 not in sys.path:
    sys.path.insert(0, V2)

RANDOM_POLICY_BASELINE = 0.1875
RANDOM_FEATURE_BASELINE = 0.4688
D_MODEL = 65536
INIT_SEED = 77
ARMS = {"CONTROL": (False, False), "PROBE": (True, False),
        "GEOM": (False, True), "BOTH": (True, True)}


def r2_std(x, y, frac=0.25, lam_rel=1e-3):
    """Scale-aware held-out R^2 (instrument proven by test_readout_instrument.py)."""
    n = x.shape[0]
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


def offdiag_cos(f):
    c = F.normalize(f.reshape(f.shape[0], -1).double(), dim=-1)
    c = c @ c.T
    n = c.shape[0]
    return float((c.sum() - c.diag().sum()) / (n * (n - 1)))


def _img_from(o):
    img = o.detach().float()
    if img.dim() == 4:
        img = img[0]
    if img.shape[-1] in (1, 3):
        img = img.permute(2, 0, 1)
    if float(img.max()) > 2.0:
        img = img / 255.0
    return img.clamp(0, 1).contiguous()


def _state_of(env):
    return torch.cat([env.ee.detach().float().reshape(-1)[:2],
                      env.goal.detach().float().reshape(-1)[:2]]).contiguous()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=700)
    ap.add_argument("--max-steps", type=int, default=24)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--eval-episodes", type=int, default=48)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-pool", type=int, default=512)
    ap.add_argument("--lam-probe", type=float, default=1.0)
    ap.add_argument("--lam-geom", type=float, default=1.0)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--out", default="/root/henri/vlm_aux_readability.json")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    R = {"schema": "henri.vlm_aux_readability/1",
         "when": time.strftime("%Y-%m-%d %H:%M:%S"), "device": dev,
         "device_name": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
         "config": vars(a), "init_seed": INIT_SEED,
         "baselines": {"random_policy": RANDOM_POLICY_BASELINE,
                       "random_feature_arm": RANDOM_FEATURE_BASELINE},
         "pre_registered": {
             "control_collapses": "CONTROL psi r2 < 0.20",
             "readability_preserved_X": "arm X psi r2 >= 0.50",
             "FIX_CONFIRMED": "some aux arm r2 >= 0.50 AND succ >= CONTROL + 2SE",
             "NO_FIX": "all aux arms below 0.50",
             "primary_endpoint": "readability (r2_std)",
             "secondary_endpoint": "closed-loop success, UNDERPOWERED at n=48"}}

    # ------------------------------------------------------------ 1. demos
    import henri_vla_env as E
    env = E.PixelReachEnv()
    R["env_class"] = type(env).__name__
    t0 = time.time()
    obs, st, act = [], [], []
    for e in range(a.episodes):
        o = env.reset(4000 + e)
        k = 0
        while (not env.done) and k < a.max_steps:
            obs.append(_img_from(o))
            st.append(_state_of(env))
            aa = env.expert_action().detach().float()   # UNIT direction; env scales it
            act.append(aa)
            o, _r, _d, _i = env.step(aa)
            k += 1
    O, S, A = torch.stack(obs), torch.stack(st), torch.stack(act)
    n_dof = int(A.shape[-1])
    R["data"] = {"n_samples": int(O.shape[0]), "n_dof": n_dof,
                 "steps_per_episode_mean": round(O.shape[0] / max(a.episodes, 1), 2),
                 "collect_seconds": round(time.time() - t0, 1)}
    print(f"[data] n={O.shape[0]} n_dof={n_dof}", flush=True)

    # ------------------------------------------------- 2. frozen VLM features
    import henri_vla_vlm_bridge as B
    import henri_continuous_action_head as H
    t0 = time.time()
    ext = B.VLMFeatureExtractor(a.model, device=dev, dtype=torch.bfloat16).load()
    big = F.interpolate(O.to(dev).to(torch.float32), size=(224, 224),
                        mode="bilinear", align_corners=False)
    feats = ext.features(big, batch=8).float()
    pooled = feats.mean(dim=1)
    R["vlm"] = {"model_id": a.model, "tower_path": ext.tower_path,
                "feat_shape": list(feats.shape),
                "output_key_used": getattr(ext, "output_key_used", None),
                "extract_seconds": round(time.time() - t0, 1)}
    R["pooled_r2"] = round(r2_std(pooled.cpu(), S), 4)
    print(f"[vlm] feats {list(feats.shape)} pooled_r2={R['pooled_r2']}", flush=True)

    dv, npatch = int(feats.shape[-1]), int(feats.shape[1])
    norm = H.ActionNormalizer(n_dof=n_dof).fit(A)
    An = norm.normalize(A)
    Smu, Ssd = S.mean(0, keepdim=True), S.std(0, keepdim=True).clamp(min=1e-6)
    Sstd = (S - Smu) / Ssd                       # unit-variance probe target

    Ftr = feats.to(dev)
    Atr = An.to(dev)
    Sttr = Sstd.to(dev)
    n = Ftr.shape[0]

    m = min(256, n)
    ftr = feats[:m].to(dev)
    R["arms"] = {}
    R["psi_r2_at_init"] = None
    init_r2 = None
    for arm, (use_probe, use_geom) in ARMS.items():
        print(f"\n[arm] {arm} probe={use_probe} geom={use_geom}", flush=True)
        torch.manual_seed(INIT_SEED)
        bridge = B.VLMPhaseBridge(d_vit=dv, n_patches=npatch, d_model=D_MODEL,
                                  d_pool=a.d_pool).to(dev)
        head = H.FlowMatchingActionHead(n_dof=n_dof, d_cond=512, d_hidden=1024,
                                        n_layers=4, phase_dim=D_MODEL).to(dev)
        probe = nn.Linear(D_MODEL, 4).to(dev) if use_probe else None

        with torch.no_grad():
            psi0 = bridge(ftr).float().cpu()
        r2_init = r2_std(psi0, S[:m])
        if init_r2 is None:
            init_r2 = r2_init
            R["psi_r2_at_init"] = round(r2_init, 4)
        R["arms"][arm] = {"binding_active": bool(bridge.binding_active),
                          "binding_ratio": round(float(bridge.binding_ratio), 6),
                          "psi_r2_at_init": round(r2_init, 4)}
        if not bridge.binding_active:
            R["fatal"] = f"{arm}: binding path inactive; refusing to train"
            json.dump(R, open(a.out, "w"), indent=2)
            print(json.dumps(R, indent=2))
            return 3

        params = ([p for p in bridge.parameters() if p.requires_grad] +
                  [p for p in head.parameters() if p.requires_grad])
        if probe is not None:
            params += list(probe.parameters())
        opt = torch.optim.AdamW(params, lr=a.lr)

        g = torch.Generator().manual_seed(0)
        first = last = None
        curve = []
        t0 = time.time()
        bridge.train()
        head.train()
        for step in range(1, a.steps + 1):
            idx = torch.randint(0, n, (a.batch,), generator=g)
            psi = bridge(Ftr[idx])
            loss = head.loss(Atr[idx], psi)
            parts = {"flow": float(loss.detach())}
            if probe is not None:
                lp = F.mse_loss(probe(psi), Sttr[idx])
                loss = loss + a.lam_probe * lp
                parts["probe"] = float(lp.detach())
            if use_geom:
                z = F.normalize(psi, dim=-1)
                gram = z @ z.T
                b_ = gram.shape[0]
                off = (gram.sum() - gram.diag().sum()) / (b_ * (b_ - 1))
                lg = off.pow(2)
                loss = loss + a.lam_geom * lg
                parts["geom"] = float(lg.detach())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            v = float(loss.detach())
            if first is None:
                first = v
            last = v
            if step == 1 or step % max(1, a.steps // 6) == 0:
                curve.append({"step": step, "loss": round(v, 6),
                              "parts": {k: round(x, 6) for k, x in parts.items()}})
                print(f"  step {step:5d} loss {v:.6f} {parts}", flush=True)
        train_s = round(time.time() - t0, 1)

        bridge.eval()
        head.eval()
        with torch.no_grad():
            psi1 = bridge(ftr).float().cpu()
        r2_after = r2_std(psi1, S[:m])
        cos_after = offdiag_cos(psi1)
        R["arms"][arm].update({
            "loss_first": round(first, 6), "loss_last": round(last, 6),
            "ratio": round(last / max(first, 1e-12), 4), "curve": curve,
            "train_seconds": train_s,
            "psi_r2": round(r2_after, 4), "psi_cos": round(cos_after, 8),
            "psi_r2_delta_vs_init": round(r2_after - r2_init, 4)})
        print(f"  -> psi r2 {r2_init:.4f} -> {r2_after:.4f}  cos {cos_after:.6f}",
              flush=True)

        # ------------------------------------------- closed-loop (secondary)
        wins = 0
        t0 = time.time()
        for e in range(a.eval_episodes):
            o = env.reset(9000 + e)
            k = 0
            while (not env.done) and k < int(env.cfg.horizon):
                img = _img_from(o)[None].to(dev).to(torch.float32)
                with torch.no_grad():
                    x224 = F.interpolate(img, size=(224, 224), mode="bilinear",
                                         align_corners=False)
                    f = ext.features(x224, batch=1)
                    psi = bridge(f)
                    aa = norm.denormalize(head.sample(psi, steps=8))[0]
                o, _r, _d, _i = env.step(aa)
                k += 1
            wins += int(bool(env.success))
        succ = wins / max(a.eval_episodes, 1)
        se = (succ * (1.0 - succ) / max(a.eval_episodes, 1)) ** 0.5
        R["arms"][arm].update({"success": round(succ, 6), "wins": wins,
                               "se": round(se, 6),
                               "eval_seconds": round(time.time() - t0, 1)})
        print(f"  -> success {succ:.4f} (+-{se:.4f})", flush=True)

    # ------------------------------------------------------------- verdicts
    ctrl = R["arms"]["CONTROL"]
    v = {"control_collapses": bool(ctrl["psi_r2"] < 0.20)}
    for arm, d in R["arms"].items():
        if arm == "CONTROL":
            continue
        v[f"readability_preserved_{arm}"] = bool(d["psi_r2"] >= 0.50)
        v[f"improves_success_{arm}"] = bool(
            d["success"] >= ctrl["success"] + 2 * max(d["se"], ctrl["se"]))
    fixed = [arm for arm in R["arms"] if arm != "CONTROL"
             and R["arms"][arm]["psi_r2"] >= 0.50
             and R["arms"][arm]["success"] >= ctrl["success"] + 2 * max(
                 R["arms"][arm]["se"], ctrl["se"])]
    v["FIX_CONFIRMED"] = bool(fixed)
    v["fixed_arms"] = fixed
    v["NO_FIX"] = bool(all(R["arms"][x]["psi_r2"] < 0.50
                           for x in R["arms"] if x != "CONTROL"))
    R["verdicts"] = v
    R["peak_vram_mib"] = (round(torch.cuda.max_memory_allocated() / 2**20, 1)
                          if dev == "cuda" else None)
    R["honest_limits"] = [
        "SYNTHETIC pixel task; not robot capability; not a benchmark score",
        f"secondary endpoint underpowered: n={a.eval_episodes} -> 2*SE ~"
        f"{round(2*max(ctrl['se'],1e-9),3)}",
        "PROBE arm uses the 4-D state as training-time auxiliary supervision only",
        "VLM frozen; 64x64 three-channel upsampled to 224 is OOD for it",
        "sample count matched to the random-feature arm (3072) by design",
    ]
    json.dump(R, open(a.out, "w"), indent=2)
    print("\n" + json.dumps({"verdicts": v,
                             "arms": {k: {kk: d[kk] for kk in
                                          ("psi_r2", "psi_cos", "success", "se",
                                           "loss_first", "loss_last", "ratio")}
                                      for k, d in R["arms"].items()},
                             "pooled_r2": R["pooled_r2"],
                             "peak_vram_mib": R["peak_vram_mib"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
