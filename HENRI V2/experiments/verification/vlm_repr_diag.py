"""DECISIVE DIAGNOSTIC: does psi CARRY THE STATE? (VLM arm was FALSIFIED)

MEASURED RESULT THIS EXPLAINS
=============================
vlm_success_rate 0.078125 vs random_feature_baseline 0.4688, loss flat
(1.345031 -> 1.377983), success DECLINING across evals, LOOP_CLOSES false.

A negative result is only useful if we know WHERE the information dies. Two
candidates, and they demand opposite fixes:

  H1 (my bug):      the VLMPhaseBridge destroys state information that the VLM
                    representation DOES contain. Fix = the bridge.
  H2 (representation): Qwen3-VL's pooled/patched features for a 64x64 synthetic
                    dot-field simply do not encode the arm/goal state. Fix = real
                    data or a different visual path, not more bridge tuning.

TEST: linear (ridge) readability of the 4-D state [ee_x, ee_y, goal_x, goal_y]
from three representations, on HELD-OUT observations, plus an effective-rank
measure. If pooled VLM features are unreadable (R^2 ~ 0) while the random-pixel
ingress is readable, H2 holds and the earlier 0.4688 random arm is explained.

This measures. It asserts nothing and it trains nothing.
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


def _find_env():
    """Introspect henri_vla_env for the pixel env class (defensive: this file
    must not fail on an API rename, it must REPORT)."""
    import henri_vla_env as E
    names = [n for n in dir(E) if n[0].isupper()]
    classes = []
    for n in names:
        o = getattr(E, n)
        if isinstance(o, type) and hasattr(o, "reset"):
            classes.append(n)
    cfg = None
    if hasattr(E, "EnvConfig"):
        cfg = E.EnvConfig()
    for n in classes:
        cls = getattr(E, n)
        for attempt in ((cfg,), ()):
            try:
                o = cls(*attempt)
                return E, o, n
            except Exception:
                continue
    return E, None, None


def _obs_of(env, seed: int) -> torch.Tensor:
    """One rendered observation as [3,H,W] float in [0,1]."""
    try:
        r = env.reset(seed=int(seed))
    except TypeError:
        r = env.reset()
    img = None
    if isinstance(r, torch.Tensor) and r.dim() == 3:
        img = r
    if img is None:
        for meth in ("render", "observe", "_render", "image", "obs"):
            if hasattr(env, meth):
                v = getattr(env, meth)
                v = v() if callable(v) else v
                if isinstance(v, torch.Tensor):
                    img = v
                    break
    if img is None:
        raise RuntimeError("no observation found on env")
    img = img.detach().float()
    if img.dim() == 3 and img.shape[-1] in (1, 3):
        img = img.permute(2, 0, 1)
    if img.max() > 2.0:
        img = img / 255.0
    return img.clamp(0, 1)


def _state_of(env) -> list:
    out = []
    for a in ("ee", "goal"):
        v = getattr(env, a, None)
        if v is None:
            return []
        v = v.detach().float().reshape(-1)[:2] if isinstance(v, torch.Tensor) else torch.tensor(v).float()[:2]
        out.append(v)
    return torch.cat(out).tolist()


def _pr(x: torch.Tensor) -> float:
    """Participation ratio = effective number of active directions."""
    g = (x @ x.T).double()
    lam = torch.linalg.eigvalsh(g.cpu()).clamp(min=0.0)
    s = lam.sum()
    return float((s * s) / (lam * lam).sum().clamp(min=1e-30))


def _dc_share(x: torch.Tensor) -> float:
    """Energy fraction along the single common direction of the batch.

    ~1.0 => one DC direction dominates, so the representation carries almost no
    per-sample information (the bridge-collapse signature). ~1/n => healthy.
    This is the direct, interpretable form of `cos_sim ~ 1`.
    """
    if x.shape[0] < 2:
        return float("nan")
    m = x.mean(dim=0)
    n = m.norm()
    if float(n) < 1e-12:
        return 0.0
    mh = m / n
    proj = x @ mh
    denom = (x ** 2).sum(dim=-1).mean().clamp(min=1e-12)
    return float((proj ** 2).mean() / denom)


def _r2(x: torch.Tensor, y: torch.Tensor, frac: float = 0.25, lam: float = 1e-2) -> float:
    n = x.shape[0]
    nte = max(1, int(n * frac))
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    te, tr = perm[:nte], perm[nte:]
    xtr, ytr, xte, yte = x[tr].double(), y[tr].double(), x[te].double(), y[te].double()
    k = xtr @ xtr.T
    a = torch.linalg.solve(k + lam * torch.eye(k.shape[0], dtype=k.dtype), ytr)
    pred = (xte @ xtr.T) @ a
    res = ((yte - pred) ** 2).sum(0)
    tot = ((yte - yte.mean(0, keepdim=True)) ** 2).sum(0).clamp(min=1e-12)
    return float((1.0 - res / tot).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--d-model", type=int, default=65536)
    ap.add_argument("--d-pool", type=int, default=512)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--out", default="/root/henri/vlm_repr_diag.json")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    R = {"schema": "henri.vlm_repr_diag/1", "when": time.strftime("%Y-%m-%d %H:%M:%S"),
         "device": dev, "n_samples": a.n, "psi_dim": a.d_model,
         "device_name": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"}

    E, env, ename = _find_env()
    R["env_class"] = ename
    if env is None:
        R["fatal"] = "could not instantiate an env with .reset()"
        R["env_module_api"] = [n for n in dir(E) if not n.startswith("_")]
        json.dump(R, open(a.out, "w"), indent=2)
        print(json.dumps(R, indent=2))
        return 4

    obs, st = [], []
    for i in range(a.n):
        obs.append(_obs_of(env, 1000 + i))
        st.append(_state_of(env))
    O = torch.stack(obs)                       # [n,3,H,W]
    S = torch.tensor(st, dtype=torch.float32)  # [n,4]
    R["obs_shape"] = list(O.shape)
    R["state_span"] = [round(float(S.min()), 4), round(float(S.max()), 4)]

    # ---- (c) raw pixel flatten: the control that a LINEAR map can read state ---
    flat = O.reshape(a.n, -1)
    R["r2_raw_pixels"] = round(_r2(flat, S), 4)
    R["pr_raw_pixels"] = round(_pr(flat), 2)

    # ---- (b) random-projection ingress psi (the arm that scored 0.4688) -------
    try:
        import henri_vla_vision_ingress as I
        ing = I.VisionPhaseIngress(d_model=a.d_model, image_size=O.shape[-1],
                                   patch=16, d_patch=256).to(dev).eval()
        with torch.no_grad():
            psi_r = ing(O.to(dev).to(torch.float32)).float()
        R["r2_psi_random_ingress"] = round(_r2(psi_r.cpu(), S), 4)
        R["pr_psi_random_ingress"] = round(_pr(psi_r.cpu()), 2)
        R["cos_sim_random_ingress"] = round(
            float((F.normalize(psi_r, dim=-1) @ F.normalize(psi_r, dim=-1).T).mean()), 6)
    except Exception as e:
        R["random_ingress_error"] = f"{type(e).__name__}: {e}"

    # ---- (a) REAL VLM: pooled features, then bridge psi -----------------------
    t0 = time.time()
    import henri_vla_vlm_bridge as B
    ext = B.VLMFeatureExtractor(a.model, device=dev, dtype=torch.bfloat16).load()
    big = F.interpolate(O.to(dev).to(torch.float32), size=(224, 224),
                        mode="bilinear", align_corners=False)
    feats = ext.features(big, batch=8).float()          # [n,N,d_vit]
    R["vlm"] = {"tower_path": ext.tower_path, "d_vit": ext.d_vit,
                "n_patches": ext.n_patches, "load_and_features_s": round(time.time() - t0, 1)}
    pooled = feats.mean(dim=1)                          # [n,d_vit]
    R["r2_vlm_pooled_features"] = round(_r2(pooled.cpu(), S), 4)
    R["pr_vlm_pooled_features"] = round(_pr(pooled.cpu()), 2)
    R["cos_sim_vlm_pooled"] = round(
        float((F.normalize(pooled, dim=-1) @ F.normalize(pooled, dim=-1).T).mean()), 6)
    flatfeat = feats.reshape(a.n, -1)
    R["r2_vlm_patch_features"] = round(_r2(flatfeat.cpu(), S), 4)

    with torch.no_grad():
        bridge = B.VLMPhaseBridge(d_vit=ext.d_vit, n_patches=ext.n_patches,
                                  d_model=a.d_model, d_pool=a.d_pool).to(dev)
        psi_v = bridge(feats.to(dev)).float()
    R["r2_psi_vlm_bridge"] = round(_r2(psi_v.cpu(), S), 4)
    R["pr_psi_vlm_bridge"] = round(_pr(psi_v.cpu()), 2)
    R["cos_sim_vlm_bridge"] = round(
        float((F.normalize(psi_v, dim=-1) @ F.normalize(psi_v, dim=-1).T).mean()), 6)
    R["dc_share_vlm_bridge"] = round(_dc_share(psi_v.cpu()), 4)
    R["dc_share_module"] = round(float(getattr(bridge, "dc_share", float("nan"))), 4)
    R["binding_ratio"] = round(float(getattr(bridge, "binding_ratio", 0.0)), 6)
    R["binding_active"] = bool(getattr(bridge, "binding_active", False))
    if "psi_r" in dir():
        R["dc_share_random_ingress"] = round(_dc_share(psi_r.cpu()), 4)

    # ---- VERDICT --------------------------------------------------------------
    rr = R.get("r2_psi_random_ingress")
    pv = R.get("r2_vlm_pooled_features")
    pb = R.get("r2_psi_vlm_bridge")
    R["verdicts"] = {
        "H2_vlm_representation_uninformative":
            None if pv is None else bool(pv < 0.30),
        "H1_bridge_destroys_information":
            None if (pb is None or pv is None) else bool(pv - pb > 0.15),
        "random_pixels_readable": None if rr is None else bool(rr > 0.50),
        "explains_0.4688_random_arm":
            None if (rr is None or pb is None) else bool(rr > 0.50 and pb < 0.30),
    }
    json.dump(R, open(a.out, "w"), indent=2)
    print(json.dumps(R, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
