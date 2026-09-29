"""WHERE does psi lose the state? -- stage-wise, on REAL VLM features. One GPU run.

MEASURED SO FAR (receipt vlm_bridge_trainability_receipt.json):
    mean-pooled VLM features  r2_std 0.7999    state IS readable in the input
    psi at bridge init        r2_std 0.3553    half lost before any training
    psi after training        r2_std -0.5      overfit; state readability is not
                                               the objective the bridge optimises
Four hypotheses already FALSIFIED: H1 DC/bias, H3 L2-normalisation, H4 fixed-penalty
readout, H5 the VSA binding term. A synthetic probe with the measured GEOMETRY
(cos 0.999998, state in the patch mean) did NOT reproduce the loss -- the bridge kept
r2 0.9999 -- so synthetic inputs cannot localise this. It must be measured on the
REAL features, at each stage.

WHAT THIS DOES
    extracts real Qwen3-VL features for n seeded observations, then measures
    scale-aware held-out readability of the 4-D state at every stage of the real
    bridge, plus a CONTROL: a random 1024 -> 512 projection (does 2:1 compression
    alone damage readability?). The control is what makes the result interpretable:
    without it, any drop at to_pool is indistinguishable from capacity loss.

STAGES
    0 input mean-pooled        [n,1024]     the readable reference
    1 control random 1024->512 [n,512]      capacity check
    2 attention-pooled         [n,1024]     bridge.pool()
    3 to_pool output           [n,512]
    4 to_phase output          [n,65536]    pre-binding, pre-normalisation
    5 accumulator acc          [n,65536]    after binding + phase_offset
    6 psi (normalised)         [n,65536]    what the head actually consumes

VERDICT: the FIRST stage whose r2 falls below half the input's r2 is the destroyer.
Also writes a reusable feature cache to disk so later probes need no GPU.
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
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--d-pool", type=int, default=512)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--out", default="/root/henri/vlm_stage_localise.json")
    ap.add_argument("--cache", default="/root/henri/vlm_feat_cache.pt")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    R = {"schema": "henri.vlm_stage_localise/1",
         "when": time.strftime("%Y-%m-%d %H:%M:%S"), "device": dev,
         "device_name": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
         "n": a.n, "readout": "r2_std (scale-aware), held-out 25%, lam_rel=1e-3"}

    import henri_vla_env as E
    env = E.PixelReachEnv()
    obs, st = [], []
    for i in range(a.n):
        o = env.reset(3000 + i)
        obs.append(_img_from(o))
        st.append(_state_of(env))
    O = torch.stack(obs)
    S = torch.stack(st).float()
    R["obs_shape"] = list(O.shape)
    R["state_span"] = [round(float(S.min()), 4), round(float(S.max()), 4)]

    import henri_vla_vlm_bridge as B
    t0 = time.time()
    ext = B.VLMFeatureExtractor(a.model, device=dev, dtype=torch.bfloat16).load()
    big = F.interpolate(O.to(dev).to(torch.float32), size=(224, 224),
                        mode="bilinear", align_corners=False)
    feats = ext.features(big, batch=8).float()
    R["vlm"] = {"tower_path": ext.tower_path, "d_vit": ext.d_vit,
                "n_patches": ext.n_patches, "feat_shape": list(feats.shape),
                "output_key_used": getattr(ext, "output_key_used", None),
                "seconds": round(time.time() - t0, 1)}

    # reusable cache: real features + states, so later probes need no GPU
    if a.cache:
        try:
            torch.save({"feats": feats.half().cpu(), "states": S.cpu(),
                        "meta": R["vlm"], "env_class": type(env).__name__}, a.cache)
            R["cache_written"] = os.path.getsize(a.cache)
        except Exception as exc:                                     # noqa: BLE001
            R["cache_error"] = f"{type(exc).__name__}: {exc}"

    fc = feats.to(dev)
    stages = {}

    # ---- 0. input, mean-pooled (the readable reference) --------------------
    pooled = feats.mean(dim=1)
    stages["0_input_mean_pooled"] = {"dim": int(pooled.shape[-1]),
                                     "r2_std": round(r2_std(pooled.cpu(), S), 4),
                                     "cos": round(offdiag_cos(pooled), 8)}

    # ---- 1. CONTROL: random 1024 -> 512 linear projection ------------------
    # DEFECT FIXED 2026-09-29 (SECOND fix; the first was itself buggy):
    #   attempt 1: W on CPU while pooled on CUDA
    #     -> "Expected all tensors to be on the same device"
    #   attempt 2: W moved to pooled.device but the generator stayed CPU
    #     -> "Expected a 'cuda' device type for generator but found 'cpu'"
    # The control is tiny (n x 1024) and deterministic, so it is computed on CPU
    # where the CPU generator is valid. No GPU tensor enters this line.
    g = torch.Generator().manual_seed(1234)
    W = torch.randn(int(pooled.shape[-1]), a.d_pool, generator=g) / (a.d_pool ** 0.5)
    stages["1_control_random_proj"] = {
        "dim": a.d_pool, "r2_std": round(r2_std((pooled.cpu() @ W), S), 4),
        "note": "capacity check: 2:1 random compression, no training"}

    # ---- real bridge, fresh init, capturing each stage ---------------------
    bridge = B.VLMPhaseBridge(d_vit=int(feats.shape[-1]), n_patches=int(feats.shape[1]),
                              d_model=D_MODEL, d_pool=a.d_pool).to(dev).eval()
    with torch.no_grad():
        pooled_attn = bridge.pool(fc.to(torch.float32))              # [n, d_vit]
        z = bridge.to_pool(pooled_attn)                              # [n, d_pool]
        p = bridge.to_phase(z) + bridge.phase_offset                 # [n, D]
        psi = bridge(fc)                                             # sets binding_ratio
        bsum = (bridge.binding_term(fc.to(torch.float32))
                if bridge.binding_active else None)
        acc = (bsum + p) if bsum is not None else p

    stages["2_attention_pooled"] = {"dim": int(pooled_attn.shape[-1]),
                                    "r2_std": round(r2_std(pooled_attn.cpu(), S), 4),
                                    "cos": round(offdiag_cos(pooled_attn.cpu()), 8)}
    stages["3_to_pool"] = {"dim": int(z.shape[-1]),
                           "r2_std": round(r2_std(z.cpu(), S), 4),
                           "cos": round(offdiag_cos(z.cpu()), 8)}
    stages["4_to_phase"] = {"dim": int(p.shape[-1]),
                            "r2_std": round(r2_std(p.cpu(), S), 4),
                            "cos": round(offdiag_cos(p.cpu()), 8)}
    stages["5_accumulator"] = {"dim": int(acc.shape[-1]),
                               "r2_std": round(r2_std(acc.cpu(), S), 4),
                               "cos": round(offdiag_cos(acc.cpu()), 8)}
    stages["6_psi_normalised"] = {"dim": int(psi.shape[-1]),
                                  "r2_std": round(r2_std(psi.cpu(), S), 4),
                                  "cos": round(offdiag_cos(psi.cpu()), 8)}

    R["stages"] = stages
    R["binding_active"] = bool(getattr(bridge, "binding_active", False))
    R["binding_ratio"] = round(float(getattr(bridge, "binding_ratio", 0.0)), 6)
    R["dc_share"] = round(float(getattr(bridge, "dc_share", float("nan"))), 6)
    R["vlm_frozen_params"] = sum(p_.numel() for p_ in ext.model.parameters())

    # ---- verdict: first stage that loses half the input readability --------
    base = stages["0_input_mean_pooled"]["r2_std"]
    order = ["0_input_mean_pooled", "2_attention_pooled", "3_to_pool",
             "4_to_phase", "5_accumulator", "6_psi_normalised"]
    first_drop = None
    for k in order:
        if stages[k]["r2_std"] < 0.5 * base:
            first_drop = k
            break
    R["input_r2"] = base
    R["first_stage_below_half_of_input"] = first_drop
    R["control_random_proj_r2"] = stages["1_control_random_proj"]["r2_std"]
    R["capacity_explains_drop"] = bool(
        stages["3_to_pool"]["r2_std"] <= stages["1_control_random_proj"]["r2_std"] + 0.05)
    R["DESTROYER"] = first_drop or "none -- readability survives the whole bridge"

    json.dump(R, open(a.out, "w"), indent=2)
    print(json.dumps(R, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
