"""DECISIVE MATCHED 3-ARM: is TRAINABILITY of the vision ingress the destroyer?

THE HYPOTHESIS (from measured evidence, not intuition)
=====================================================
    random_ingress receipt : ingress_trainable_params 0      -> SUCCESS 0.4688
                             (frozen random VSA projection; only the head trains)
    vlm_vla receipt        : bridge_trainable 36242432       -> SUCCESS 0.078125
                             (trainable VLM bridge + head)
    stage probe            : FRESH bridge preserves psi r2 (0.7249 -> 0.3949)
                             TRAINED bridge collapses psi   (r2 -0.5, cos 0.99977)

Reading: the arm that WORKED had a FROZEN ingress. The arm that FAILED had a
TRAINABLE one. So the destroyer may be the TRAINABILITY of the ingress, not the
VLM representation -- and the stage probe already showed the forward path is fine.

THIS EXPERIMENT -- 3 arms, IDENTICAL data, seed, budget, eval protocol
    A  RANDOM-FROZEN   : VisionPhaseIngress, frozen  (only the head trains)
                         -> must reproduce ~0.4688, else the protocol is not matched
    B  VLM-TRAINABLE   : VLMPhaseBridge + head both train
                         -> expected to reproduce the failure
    C  VLM-FROZEN      : VLMPhaseBridge frozen (requires_grad=False), head trains
                         -> THE TEST

PRE-REGISTERED (fixed before running; not adjusted afterwards)
    A reproduces            : success_A >= 0.30          (validity gate on the setup)
    B reproduces failure    : success_B <= 0.20
    TRAINABILITY_IS_THE_CAUSE : success_C >= 0.30 AND success_C >= success_B + 2*SE
    FREEZING_FULLY_RECOVERS   : success_C >= success_A - 2*SE
    REPRESENTATION_IS_THE_CAUSE : success_C <= 0.20 (freezing does not help, so the
                                  VLM features themselves are the problem)
All three outcomes are reportable. None is spun.

MATCHED PROTOCOL (taken from the run that reached 0.4688, not invented):
    256 train episodes, 3000 steps, batch 64, lr 2e-3, d_model 65536,
    64 eval episodes at disjoint seeds (90000+), head sample_steps 8, flow head
    4 layers / d_hidden 1024 / d_cond 512.
    The VLM is extracted ONCE and cached (it is frozen in every arm).

HONEST SCOPE: SYNTHETIC pixel reach task; not robot capability; not a benchmark
score. n=64 eval -> 2*SE ~ 0.12, so differences below ~0.12 are UNRESOLVED; the
pre-registered margins are set accordingly and a null result will be reported.
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

D_MODEL = 65536
RANDOM_POLICY = 0.1875
ARMS = ("RANDOM_FROZEN", "VLM_TRAINABLE", "VLM_FROZEN")


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


def _cos(f):
    z = F.normalize(f.reshape(f.shape[0], -1).float(), dim=-1)
    c = z @ z.T
    n = c.shape[0]
    return float((c.sum() - c.diag().sum()) / (n * (n - 1)))


@torch.no_grad()
def eval_closed_loop(psi_fn, head, norm, env, n_eps, seed0, sample_steps):
    head.eval()
    wins = 0
    for e in range(n_eps):
        o = env.reset(seed0 + e)
        k = 0
        while (not env.done) and k < int(env.cfg.horizon):
            psi = psi_fn(_img_from(o))
            a_norm = head.sample(psi, steps=sample_steps)
            aa = norm.denormalize(a_norm)[0]
            o, _r, _d, _i = env.step(aa)
            k += 1
        wins += int(bool(env.success))
    return wins / max(n_eps, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=256)
    ap.add_argument("--max-steps", type=int, default=24)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--eval-episodes", type=int, default=64)
    ap.add_argument("--eval-seed", type=int, default=90000)
    ap.add_argument("--sample-steps", type=int, default=8)
    ap.add_argument("--d-model", type=int, default=D_MODEL)
    ap.add_argument("--d-pool", type=int, default=512)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--out", default="/root/henri/ingress_freeze_arms.json")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.d_model, a.episodes, a.steps, a.eval_episodes = 4096, 24, 150, 16

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    R = {"schema": "henri.ingress_freeze_arms/1",
         "when": time.strftime("%Y-%m-%d %H:%M:%S"),
         "device": dev, "device_name": torch.cuda.get_device_name(0)
         if dev == "cuda" else "cpu",
         "smoke": bool(a.smoke), "config": vars(a),
         "hypothesis": ("the arm that reached 0.4688 had a FROZEN ingress (0 trainable "
                        "params); the arm that failed had a trainable bridge. Test: "
                        "freeze the bridge and hold everything else identical."),
         "pre_registered": {
             "A_reproduces": "success_A >= 0.30",
             "B_reproduces_failure": "success_B <= 0.20",
             "TRAINABILITY_IS_THE_CAUSE":
                 "success_C >= 0.30 AND success_C >= success_B + 2*SE",
             "FREEZING_FULLY_RECOVERS": "success_C >= success_A - 2*SE",
             "REPRESENTATION_IS_THE_CAUSE": "success_C <= 0.20"}}

    import henri_vla_env as E
    import henri_continuous_action_head as H
    env = E.PixelReachEnv()
    R["env"] = {"n_dof": int(env.n_dof), "horizon": int(env.cfg.horizon)}

    # ---------------------------------------------------------- 1. dataset
    t0 = time.time()
    obs, st, act = [], [], []
    for e in range(a.episodes):
        o = env.reset(5000 + e)
        k = 0
        while (not env.done) and k < a.max_steps:
            obs.append(_img_from(o))
            st.append(_state_of(env))
            aa = env.expert_action().detach().float()   # unit dir; env applies step
            act.append(aa)
            o, _r, _d, _i = env.step(aa)
            k += 1
    O, S, A = torch.stack(obs), torch.stack(st), torch.stack(act)
    n_dof = int(A.shape[-1])
    norm = H.ActionNormalizer(n_dof=n_dof).fit(A)
    An = norm.normalize(A)
    R["data"] = {"n_samples": int(O.shape[0]), "n_dof": n_dof,
                 "obs_shape": list(O.shape[1:]),
                 "collect_seconds": round(time.time() - t0, 1)}
    print(f"[data] n={O.shape[0]} n_dof={n_dof}", flush=True)

    # ---------------------------------------------------------- 2. VLM feats
    import henri_vla_vlm_bridge as B
    import henri_vla_vision_ingress as I
    t0 = time.time()
    ext = B.VLMFeatureExtractor(a.model, device=dev, dtype=torch.bfloat16).load()
    big = F.interpolate(O.to(dev).to(torch.float32), size=(224, 224),
                        mode="bilinear", align_corners=False)
    feats = ext.features(big, batch=8).float()
    R["vlm"] = {"model_id": a.model, "tower_path": ext.tower_path,
                "feat_shape": list(feats.shape),
                "output_key_used": getattr(ext, "output_key_used", None),
                "frozen_params": sum(p.numel() for p in ext.model.parameters()),
                "extract_seconds": round(time.time() - t0, 1)}
    print(f"[vlm] feats {list(feats.shape)} in {R['vlm']['extract_seconds']}s",
          flush=True)

    # per-image VLM psi path for closed-loop eval (no cache at eval time)
    def make_vlm_psi_fn(bridge, frozen_pool_only=False):
        def f(img_chw):
            x = img_chw[None].to(dev).to(torch.float32)
            x224 = F.interpolate(x, size=(224, 224), mode="bilinear",
                                 align_corners=False)
            with torch.no_grad():
                ff = ext.features(x224, batch=1)
                return bridge(ff)
        return f

    R["arms"] = {}
    for arm in ARMS:
        print(f"\n[arm] {arm}", flush=True)
        torch.manual_seed(77)
        head = H.FlowMatchingActionHead(n_dof=n_dof, d_cond=512, d_hidden=1024,
                                        n_layers=4, phase_dim=a.d_model).to(dev)
        params = [p for p in head.parameters() if p.requires_grad]
        info = {"head_trainable": sum(p.numel() for p in params)}

        if arm == "RANDOM_FROZEN":
            ing = I.VisionPhaseIngress(d_model=a.d_model, image_size=O.shape[-1],
                                       patch=16, d_patch=256).to(dev).eval()
            for p in ing.parameters():
                p.requires_grad_(False)
            with torch.no_grad():
                psi_all = ing(O.to(dev).to(torch.float32)).float()
            info.update({"ingress": "VisionPhaseIngress FROZEN",
                         "ingress_trainable": 0})
            # DEFECT FIXED 2026-09-29 (caught by the smoke gate, so no full GPU
            # budget was spent). The previous form was
            #     psi_fn = (lambda ing=ing: (lambda img: ing(...)))
            # which binds the OUTER lambda, so psi_fn(img) returned the INNER
            # FUNCTION rather than a tensor. Downstream that surfaced as
            #     AttributeError: 'function' object has no attribute 'shape'
            # at head.sample() -> encode_cond(phase). Single-argument form now.
            psi_fn = (lambda img, _ing=ing: _ing(img[None].to(dev).to(torch.float32)))

        elif arm == "VLM_TRAINABLE":
            br = B.VLMPhaseBridge(d_vit=int(feats.shape[-1]),
                                  n_patches=int(feats.shape[1]),
                                  d_model=a.d_model, d_pool=a.d_pool).to(dev)
            psi_all = br(feats.to(dev)).float()
            params += [p for p in br.parameters() if p.requires_grad]
            info.update({"ingress": "VLMPhaseBridge TRAINABLE",
                         "ingress_trainable": br.trainable_params(),
                         "binding_active": bool(br.binding_active),
                         "binding_ratio": round(float(br.binding_ratio), 6)})
            psi_fn = make_vlm_psi_fn(br)

        else:  # VLM_FROZEN
            br = B.VLMPhaseBridge(d_vit=int(feats.shape[-1]),
                                  n_patches=int(feats.shape[1]),
                                  d_model=a.d_model, d_pool=a.d_pool).to(dev)
            with torch.no_grad():
                psi_all = br(feats.to(dev)).float()
            for p in br.parameters():
                p.requires_grad_(False)
            br.eval()
            info.update({"ingress": "VLMPhaseBridge FROZEN",
                         "ingress_trainable": 0,
                         "binding_active": bool(br.binding_active),
                         "binding_ratio": round(float(br.binding_ratio), 6)})
            psi_fn = make_vlm_psi_fn(br)

        info["psi_dim"] = int(psi_all.shape[-1])
        info["psi_cos_at_init"] = round(_cos(psi_all), 6)
        info["psi_trainable_params"] = sum(p.numel() for p in params
                                           if p.requires_grad)
        print(f"  ingress={info['ingress']} psi {list(psi_all.shape)} "
              f"cos_init={info['psi_cos_at_init']}", flush=True)

        # ---------------------------------------------------- 3. train head
        psi_dev = psi_all.to(dev)
        Atr = An.to(dev)
        n = psi_dev.shape[0]
        opt = torch.optim.AdamW(params, lr=a.lr)
        g = torch.Generator().manual_seed(0)
        first = last = None
        curve = []
        t0 = time.time()
        head.train()
        if arm == "VLM_TRAINABLE":
            br.train()
        for step in range(1, a.steps + 1):
            idx = torch.randint(0, n, (a.batch,), generator=g)
            if arm == "VLM_TRAINABLE":
                # recompute psi so gradients flow into the bridge
                psi = br(feats[idx].to(dev))
            else:
                psi = psi_dev[idx]
            loss = head.loss(Atr[idx], psi)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            v = float(loss.detach())
            if first is None:
                first = v
            last = v
            if step == 1 or step % max(1, a.steps // 6) == 0:
                curve.append({"step": step, "loss": round(v, 6)})
                print(f"  step {step:5d} loss {v:.6f}", flush=True)
        train_s = round(time.time() - t0, 1)

        # psi diagnostics AFTER training
        if arm == "VLM_TRAINABLE":
            br.eval()
            with torch.no_grad():
                psi_after = br(feats[:min(256, n)].to(dev)).float()
            info["psi_cos_after"] = round(_cos(psi_after), 6)
        else:
            info["psi_cos_after"] = info["psi_cos_at_init"]

        # ---------------------------------------------------- 4. eval
        succ = eval_closed_loop(psi_fn, head, norm, env, a.eval_episodes,
                               a.eval_seed, a.sample_steps)
        se = (succ * (1.0 - succ) / max(a.eval_episodes, 1)) ** 0.5
        info.update({"loss_first": round(first, 6), "loss_last": round(last, 6),
                     "ratio": round(last / max(first, 1e-12), 4), "curve": curve,
                     "train_seconds": train_s,
                     "success": round(succ, 6), "se": round(se, 6),
                     "eval_episodes": a.eval_episodes,
                     "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1)
                     if dev == "cuda" else None})
        R["arms"][arm] = info
        print(f"  -> success {succ:.4f} (+-{se:.4f})  cos_after "
              f"{info['psi_cos_after']}", flush=True)

    # --------------------------------------------------------- verdicts
    A_ = R["arms"]["RANDOM_FROZEN"]
    B_ = R["arms"]["VLM_TRAINABLE"]
    C_ = R["arms"]["VLM_FROZEN"]
    se_max_bc = max(B_["se"], C_["se"], 1e-9)
    se_max_ac = max(A_["se"], C_["se"], 1e-9)
    v = {
        "A_reproduces": bool(A_["success"] >= 0.30),
        "B_reproduces_failure": bool(B_["success"] <= 0.20),
        "TRAINABILITY_IS_THE_CAUSE": bool(C_["success"] >= 0.30 and
                                          C_["success"] >= B_["success"] + 2 * se_max_bc),
        "FREEZING_FULLY_RECOVERS": bool(C_["success"] >= A_["success"] - 2 * se_max_ac),
        "REPRESENTATION_IS_THE_CAUSE": bool(C_["success"] <= 0.20),
        "success_A": A_["success"], "success_B": B_["success"],
        "success_C": C_["success"],
        "random_policy_baseline": RANDOM_POLICY,
        "2se_bc": round(2 * se_max_bc, 4), "2se_ac": round(2 * se_max_ac, 4)}
    R["verdicts"] = v
    R["honest_limits"] = [
        "SYNTHETIC pixel reach task; not robot capability; not a benchmark score",
        f"n={a.eval_episodes} eval episodes -> 2*SE ~ {round(2*se_max_bc,3)}; "
        "differences below that are UNRESOLVED",
        "VLM frozen in all arms; only the BRIDGE differs in trainability",
        "arm A uses a different ingress architecture (VSA random projection), so it "
        "is a protocol control, not a like-for-like feature comparison",
        "64x64 three-channel env images upsampled to 224 is OOD for the VLM",
    ]
    json.dump(R, open(a.out, "w"), indent=2)
    print("\n" + json.dumps({"arms": {k: {kk: d[kk] for kk in
                                          ("ingress", "ingress_trainable",
                                           "psi_cos_at_init", "psi_cos_after",
                                           "loss_first", "loss_last", "success", "se")}
                                      for k, d in R["arms"].items()},
                             "verdicts": v}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
