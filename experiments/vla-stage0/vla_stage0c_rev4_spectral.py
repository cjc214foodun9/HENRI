"""
Stage-0c-rev4 — r=8 contractive reduced-Koopman spectral evaluation (CPU, deterministic).
=========================================================================================
Reference 3 (gpt-5.6-sol) binding; pre-registration vla_stage0c_rev4_contract.md.
- Calibration: 171 records (lexicographic first-10 episodes, established split).
- Evaluation: rev3 220-record corpus (seeds 2101-3010, manifest f0c9a762...).
  Label: CONDITIONAL_REUSED_EVAL (disjoint from calib; previously scored in rev3).
- r=8 FIXED; separate V8^(a), K8^(a) per action; implicit P8 = V V^T.
- Contractive spectral projection: if rho(K) > 1.0 -> clamp eigenvalues radially, K~ = Re(U diag(w~) U^-1);
  if rho <= 1.0 -> K~ = K; if cond(U) > 1e8 -> BLOCKED_NUMERICAL.
- X0/X1/X5 per action; projected (coefficient-space) metrics; full-space diagnostic.
- Gates: C8 (k8<=10, top8>=0.75, rho(K~)<=1.0000), C9 (SSR_eval<=0.40), C10 (SSR_rollout5<=0.80),
  C11 (2-process determinism), C12 (baselines reported).
- Verdict: CONTRACTIVE_SPECTRAL_VERIFIED or CONTRACT_FAILED at first failing gate.
"""
import hashlib, json, os, pathlib, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
os.environ["HENRI_STAGE0B_REV_ENABLE"] = "1"
sys.path.insert(0, str(ROOT))
import vla_stage0b_rev_encoder as enc_mod

NPZ_SHA = "766e607ad0bc739ea0a139172dd34e16d01a268cca80e990af5aab01006cfcd7"
CALIB_IDS = ["seed_101.jsonl", "seed_1010.jsonl", "seed_1111.jsonl", "seed_1212.jsonl",
             "seed_1313.jsonl", "seed_1414.jsonl", "seed_1515.jsonl", "seed_202.jsonl",
             "seed_303.jsonl", "seed_404.jsonl"]
EVAL_IDS = ["seed_2101.jsonl", "seed_2202.jsonl", "seed_2303.jsonl", "seed_2404.jsonl",
            "seed_2505.jsonl", "seed_2606.jsonl", "seed_2707.jsonl", "seed_2808.jsonl",
            "seed_2909.jsonl", "seed_3010.jsonl"]
CORPUS_DIR = ROOT / "vla_stage0c_corpus"
EVAL_DIR = ROOT / "vla_stage0c_rev3_eval_corpus"
OUT_DIR = ROOT / "vla_stage0c_rev4_results"
OUT_DIR.mkdir(exist_ok=True)
R = 8
HORIZON = 5
TELE = OUT_DIR / "vla_stage0c_rev4_telemetry.json"
OPS = OUT_DIR / "vla_stage0c_rev4_operators.npz"


def load_recs(directory, ids):
    out = []
    for k in ids:
        for line in (directory / k).read_text(encoding="utf-8").splitlines():
            out.append(json.loads(line))
    return out


def encode_flat(enc, obs):
    return enc.encode(np.asarray(obs, dtype=np.float32)).reshape(len(obs), -1).astype(np.float64)


def proj_err(X, Y, V, K):
    pred = (X @ V) @ K
    return float(np.linalg.norm(Y @ V - pred) / np.linalg.norm(Y @ V))


def contractive(K, tol=1e8):
    """Pre-registered contractive spectral projection rule."""
    w, v = np.linalg.eig(K)
    rho = float(np.max(np.abs(w)))
    cond_v = float(np.linalg.cond(v))
    if cond_v > tol:
        return None, {"blocked": "BLOCKED_NUMERICAL", "cond_v": cond_v, "rho": rho}
    if rho > 1.0:
        w2 = w / np.maximum(1.0, np.abs(w))
        K2 = np.real(v @ np.diag(w2) @ np.linalg.inv(v))
        applied = True
    else:
        K2 = K.copy()
        applied = False
    return K2, {"applied": applied, "rho_raw": rho, "rho_proj": float(np.max(np.abs(np.linalg.eigvals(K2)))),
                "cond_v": cond_v, "snorm_raw": float(np.linalg.norm(K, 2)),
                "snorm_proj": float(np.linalg.norm(K2, 2))}


def main():
    t = {}
    enc = enc_mod.Stage0bRevEncoder()

    # ---- inherited sanity diagnostics (NOT in the rev4 gate chain) ----
    probe = np.asarray([[0.01, -0.02, 0.03, -0.04]], dtype=np.float32)
    os.environ["HENRI_STAGE0B_REV_ENABLE"] = "0"
    off = enc_mod.Stage0bRevEncoder()
    t["c1_bypass_diag"] = bool(np.array_equal(off.encode(probe), probe))
    os.environ["HENRI_STAGE0B_REV_ENABLE"] = "1"
    npz_path = ROOT / "vla_stage0b_rev_params.npz"
    t["c3_npz_sha_diag"] = hashlib.sha256(npz_path.read_bytes()).hexdigest() == NPZ_SHA

    # ---- Load data ----
    calib = load_recs(CORPUS_DIR, CALIB_IDS)
    evals = load_recs(EVAL_DIR, EVAL_IDS)
    obs_t = np.asarray([r["obs_t"] for r in calib], dtype=np.float32)
    obs_n = np.asarray([r["obs_next"] for r in calib], dtype=np.float32)
    acts_c = np.asarray([int(r["action"]) for r in calib], dtype=np.int64)
    X_all = encode_flat(enc, obs_t)
    Y_all = encode_flat(enc, obs_n)

    # ---- Per-action calibration spectra + bases (raw SVD via Gram) ----
    per = {}
    for a in (0, 1):
        ma = acts_c == a
        Xa, Ya = X_all[ma], Y_all[ma]
        G = Xa @ Xa.T
        ev = np.clip(np.linalg.eigvalsh(G), 0, None)
        s = np.sqrt(ev)[::-1]
        s2 = s * s
        U = np.linalg.eigh(G)[1][:, ::-1]
        V8 = (Xa.T @ U[:, :R]) / s[:R]
        per[a] = {"X": Xa, "Y": Ya, "s": s, "V8": V8,
                  "k8": float(s[0] / s[R - 1]),
                  "top8_share": float(s2[:R].sum() / s2.sum()),
                  "pr": float((s2.sum()) ** 2 / (s2 ** 2).sum()),
                  "rank_1e6": int((s > 1e-6 * s[0]).sum())}

    # ---- C8: truncation + contraction contract ----
    ops = {}
    c8_meta = {}
    c8_ok = True
    for a in (0, 1):
        Xa, Ya, V = per[a]["X"], per[a]["Y"], per[a]["V8"]
        K, *_ = np.linalg.lstsq(Xa @ V, Ya @ V, rcond=None)
        K2, meta = contractive(K)
        if K2 is None:
            c8_ok = False
            meta["gate_fail"] = "BLOCKED_NUMERICAL"
            K2 = K
        ops[a] = K2
        c8_meta[str(a)] = meta
        per[a]["k8_ok"] = per[a]["k8"] <= 10.0
        per[a]["top8_ok"] = per[a]["top8_share"] >= 0.75
        c8_ok = c8_ok and per[a]["k8_ok"] and per[a]["top8_ok"] and meta.get("rho_proj", 9e9) <= 1.0000
    t["c8_contract"] = {"pass": bool(c8_ok),
                        "k8": {str(a): round(per[a]["k8"], 4) for a in (0, 1)},
                        "top8": {str(a): round(per[a]["top8_share"], 4) for a in (0, 1)},
                        "pr": {str(a): round(per[a]["pr"], 4) for a in (0, 1)},
                        "rho": {str(a): {k2: round(v2, 6) for k2, v2 in c8_meta[str(a)].items() if k2 in ("rho_raw", "rho_proj", "applied", "cond_v", "snorm_raw", "snorm_proj")} for a in (0, 1)}}

    # ---- C9: SSR_eval (projected one-step) on the 220 eval corpus ----
    obs_te = np.asarray([r["obs_t"] for r in evals], dtype=np.float32)
    obs_ne = np.asarray([r["obs_next"] for r in evals], dtype=np.float32)
    acts_e = np.asarray([int(r["action"]) for r in evals], dtype=np.int64)
    Xe = encode_flat(enc, obs_te)
    Ye = encode_flat(enc, obs_ne)
    eps_eval, eps_p1, ssr1 = {}, {}, {}
    for a in (0, 1):
        ma = acts_e == a
        Xa, Ya, V, K = Xe[ma], Ye[ma], per[a]["V8"], ops[a]
        eps_eval[a] = proj_err(Xa, Ya, V, K)
        eps_p1[a] = float(np.linalg.norm((Ya - Xa) @ V) / np.linalg.norm(Ya @ V))
        ssr1[a] = eps_eval[a] / eps_p1[a]
    ssr_agg = float(np.mean(list(ssr1.values())))
    t["c9_ssr_eval"] = {"pass": bool(ssr_agg <= 0.40),
                        "eps_eval_proj": {str(a): round(eps_eval[a], 6) for a in (0, 1)},
                        "persistence1": {str(a): round(eps_p1[a], 6) for a in (0, 1)},
                        "ssr": {str(a): round(ssr1[a], 6) for a in (0, 1)},
                        "ssr_agg": round(ssr_agg, 6)}

    # ---- C10: 5-step open-loop rollout (X0/X5), action-switch full-state re-projection ----
    by_ep = {k: load_recs(EVAL_DIR, [k]) for k in EVAL_IDS}
    roll_err = {0: [], 1: []}
    persist5 = {0: [], 1: []}
    for k, recs in by_ep.items():
        T = len(recs)
        for t0 in range(T - HORIZON):
            acts = [int(recs[t0 + h]["action"]) for h in range(HORIZON)]
            a_last = acts[-1]
            x0 = encode_flat(enc, np.asarray([recs[t0]["obs_t"]], dtype=np.float32))[0]
            x5 = encode_flat(enc, np.asarray([recs[t0 + HORIZON]["obs_t"]], dtype=np.float32))[0]
            xh = x0.copy()
            for h in range(HORIZON):
                a = acts[h]
                V = per[a]["V8"]
                xh = (xh @ V) @ ops[a] @ V.T
            V = per[a_last]["V8"]
            roll_err[a_last].append(float(np.linalg.norm((xh - x5) @ V) / np.linalg.norm(x5 @ V)))
            persist5[a_last].append(float(np.linalg.norm((x5 - x0) @ V) / np.linalg.norm(x5 @ V)))
    eps_r5 = {a: float(np.mean(v)) for a, v in roll_err.items() if v}
    eps_p5 = {a: float(np.mean(v)) for a, v in persist5.items() if v}
    ssr5 = {a: eps_r5[a] / eps_p5[a] for a in eps_r5}
    ssr5_agg = float(np.mean(list(ssr5.values())))
    t["c10_ssr_rollout5"] = {"pass": bool(ssr5_agg <= 0.80),
                             "eps_roll5": {str(a): round(v, 6) for a, v in eps_r5.items()},
                             "persistence5": {str(a): round(v, 6) for a, v in eps_p5.items()},
                             "ssr5": {str(a): round(v, 6) for a, v in ssr5.items()},
                             "ssr5_agg": round(ssr5_agg, 6),
                             "windows": sum(len(v) for v in roll_err.values())}

    # ---- C12: baselines ----
    cal_eps = {}
    for a in (0, 1):
        Xa, Ya, V = per[a]["X"], per[a]["Y"], per[a]["V8"]
        cal_eps[str(a)] = round(proj_err(Xa, Ya, V, ops[a]), 6)
    t["baselines"] = {"calib_proj_eps_diag": cal_eps,
                      "persistence1_eval": {str(a): round(eps_p1[a], 6) for a in (0, 1)},
                      "persistence5_eval": {str(a): round(v, 6) for a, v in eps_p5.items()}}

    # ---- Verdict chain ----
    gates = {"c8": t["c8_contract"]["pass"], "c9": t["c9_ssr_eval"]["pass"], "c10": t["c10_ssr_rollout5"]["pass"]}
    first_fail = next((k for k in ("c8", "c9", "c10") if not gates[k]), None)
    t["verdict"] = "CONTRACTIVE_SPECTRAL_VERIFIED" if first_fail is None else "CONTRACT_FAILED"
    t["first_failing_gate"] = first_fail
    t["split"] = {"calib_ids": CALIB_IDS, "eval_ids": EVAL_IDS,
                  "calib_records": len(calib), "eval_records": len(evals),
                  "eval_label": "CONDITIONAL_REUSED_EVAL",
                  "eval_manifest_sha": "f0c9a7624f26bf70"}

    TELE.write_text(json.dumps(t, indent=1), encoding="utf-8")
    np.savez(OPS, K0=ops[0], K1=ops[1], V0=per[0]["V8"], V1=per[1]["V8"])
    print(json.dumps({k: t.get(k) for k in ("verdict", "first_failing_gate")}))
    print("TELE_SHA", hashlib.sha256(TELE.read_bytes()).hexdigest()[:16])
    print("OPS_SHA", hashlib.sha256(OPS.read_bytes()).hexdigest()[:16])


if __name__ == "__main__":
    main()
