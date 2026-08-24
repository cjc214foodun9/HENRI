"""
Stage-0c-rev3 — r=16 reduced-Koopman spectral evaluation (CPU, deterministic).
=============================================================================
Reference 3 (gpt-5.6-sol) binding; pre-registration vla_stage0c_rev3_contract.md.
- Calibration: 171 records (lexicographic first-10 episodes, established split).
- Evaluation: FRESH 220-record corpus (seeds 2101-3010, manifest f0c9a762...),
  no raw-obs overlap with calibration (0/181/230 measured).
- r=16 FIXED; separate V16^(a), K16^(a) per action; implicit P16 = V V^T.
- Projected (coefficient-space) metrics; full-space diagnostics only.
- Action-switch rollout: maintain predicted FULL lifted state, re-project per action.
- Verdict chain: C7 -> C8 -> C9 -> C10 (C11 determinism, C12 baselines).
"""
import ast, hashlib, inspect, json, os, pathlib, sys, textwrap
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
OUT_DIR = ROOT / "vla_stage0c_rev3_results"
OUT_DIR.mkdir(exist_ok=True)
HORIZON = 5
R = 16
TELE = OUT_DIR / "vla_stage0c_rev3_telemetry.json"
OPS = OUT_DIR / "vla_stage0c_rev3_operators.npz"


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


def class_source(cls):
    raw = textwrap.dedent(inspect.getsource(cls))
    tree = ast.parse(raw)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            doc = ast.get_docstring(node)
            if doc:
                node.body = [n for n in node.body if not (isinstance(n, ast.Expr)
                                                          and isinstance(n.value, ast.Constant)
                                                          and isinstance(n.value.value, str))]
    return ast.unparse(tree)


def main():
    t = {}
    enc = enc_mod.Stage0bRevEncoder()

    # ---- C1: default-OFF bypass byte-identical ----
    probe = np.asarray([[0.01, -0.02, 0.03, -0.04]], dtype=np.float32)
    os.environ["HENRI_STAGE0B_REV_ENABLE"] = "0"
    off = enc_mod.Stage0bRevEncoder()
    t["c1_bypass"] = bool(np.array_equal(off.encode(probe), probe))
    os.environ["HENRI_STAGE0B_REV_ENABLE"] = "1"

    # ---- C2: zero trainable state (class-source AST scan) ----
    src = class_source(enc_mod.Stage0bRevEncoder)
    forbidden = ["torch", "Parameter(", "backward(", "optimizer", "randn", "gym"]
    hits = [f for f in forbidden if f in src]
    t["c2_zero_trainable"] = {"pass": len(hits) == 0, "hits": hits}

    # ---- C3: npz full-hash assert ----
    npz_path = ROOT / "vla_stage0b_rev_params.npz"
    t["c3_artifact"] = {"npz_sha_ok": hashlib.sha256(npz_path.read_bytes()).hexdigest() == NPZ_SHA,
                        "expected": NPZ_SHA}

    # ---- Load data ----
    calib = load_recs(CORPUS_DIR, CALIB_IDS)
    evals = load_recs(EVAL_DIR, EVAL_IDS)
    obs_t = np.asarray([r["obs_t"] for r in calib], dtype=np.float32)
    obs_n = np.asarray([r["obs_next"] for r in calib], dtype=np.float32)
    acts_c = np.asarray([int(r["action"]) for r in calib], dtype=np.int64)
    obs_te = np.asarray([r["obs_t"] for r in evals], dtype=np.float32)
    obs_ne = np.asarray([r["obs_next"] for r in evals], dtype=np.float32)
    acts_e = np.asarray([int(r["action"]) for r in evals], dtype=np.int64)
    X_all = encode_flat(enc, obs_t)
    Y_all = encode_flat(enc, obs_n)
    Xe = encode_flat(enc, obs_te)
    Ye = encode_flat(enc, obs_ne)

    # ---- C4: sphere geometry ----
    z = enc.encode(obs_t[:16])
    err = float(np.max(np.abs(np.linalg.norm(z.reshape(16, -1), axis=1) - 1.0)))
    t["c4_sphere_max_err"] = err

    # ---- C5: sensitivity over DEDUPLICATED calib obs (union obs_t + obs_next) ----
    u = np.unique(np.concatenate([obs_t, obs_n], axis=0), axis=0)
    D = encode_flat(enc, u)
    m = D.shape[0]
    nrm2 = np.einsum("ij,ij->i", D, D)
    d2 = nrm2[:, None] + nrm2[None, :] - 2.0 * (D @ D.T)
    np.fill_diagonal(d2, 0.0)
    iu = np.triu_indices(m, 1)
    l2 = np.sqrt(np.clip(d2[iu], 0, None))
    t["c5_sensitivity"] = {"distinct_n": int(m),
                           "frac_gt_1e3": float((l2 > 1e-3).mean()),
                           "min_l2": float(l2.min()),
                           "collisions": int((l2 <= 1e-3).sum())}

    # ---- Per-action calibration matrices + spectra ----
    per = {}
    for a in (0, 1):
        ma = acts_c == a
        Xa, Ya = X_all[ma], Y_all[ma]
        G = Xa @ Xa.T
        ev = np.clip(np.linalg.eigvalsh(G), 0, None)
        s = np.sqrt(ev)[::-1]
        s2 = s * s
        U = np.linalg.eigh(G)[1][:, ::-1]
        V16 = (Xa.T @ U[:, :R]) / s[:R]
        pr = float((s2.sum()) ** 2 / (s2 ** 2).sum())
        per[a] = {"X": Xa, "Y": Ya, "s": s, "V16": V16,
                  "pr": pr,
                  "top16_share": float(s2[:R].sum() / s2.sum()),
                  "k16": float(s[0] / s[R - 1]),
                  "rank_1e6": int((s > 1e-6 * s[0]).sum())}

    t["c6_rank_support"] = {"a0": {"rank_1e6": per[0]["rank_1e6"], "n": 102, "pr": round(per[0]["pr"], 4)},
                            "a1": {"rank_1e6": per[1]["rank_1e6"], "n": 69, "pr": round(per[1]["pr"], 4)}}

    # ---- C7: conditioning + top-16 share ----
    c7 = all(per[a]["k16"] <= 10.0 and per[a]["top16_share"] >= 0.92 for a in (0, 1))
    t["c7_conditioning"] = {"pass": bool(c7),
                            "k16": {str(a): round(per[a]["k16"], 4) for a in (0, 1)},
                            "top16": {str(a): round(per[a]["top16_share"], 4) for a in (0, 1)}}

    # ---- Full-space floors (diagnostic) ----
    floors = {}
    for a in (0, 1):
        Xa, Ya = per[a]["X"], per[a]["Y"]
        V = per[a]["V16"]
        ynorm2 = float((Ya * Ya).sum())
        YV = Ya @ V
        floors[str(a)] = {"y_target": round(float(np.sqrt(max(0.0, 1.0 - (YV * YV).sum() / ynorm2))), 6),
                          "x_same": round(float(np.sqrt(max(0.0, 1.0 - per[a]["top16_share"]))), 6)}
    t["floors_diag"] = floors

    # ---- Build K16^(a) (calibration only), C8 calibration projected eps ----
    ops = {}
    eps_calib = {}
    for a in (0, 1):
        Xa, Ya, V = per[a]["X"], per[a]["Y"], per[a]["V16"]
        K, *_ = np.linalg.lstsq(Xa @ V, Ya @ V, rcond=None)
        ops[a] = K
        eps_calib[a] = proj_err(Xa, Ya, V, K)
    t["c8_calib_recon"] = {"pass": all(v <= 0.05 for v in eps_calib.values()),
                           "eps_proj": {f"a{a}": round(eps_calib[a], 6) for a in (0, 1)}}

    # ---- C9: fresh-eval projected one-step, persistence, SSR_eval ----
    eps_eval, eps_persist, ssr = {}, {}, {}
    for a in (0, 1):
        ma = acts_e == a
        Xa, Ya, V, K = Xe[ma], Ye[ma], per[a]["V16"], ops[a]
        eps_eval[a] = proj_err(Xa, Ya, V, K)
        eps_persist[a] = float(np.linalg.norm((Ya - Xa) @ V) / np.linalg.norm(Ya @ V))
        ssr[a] = eps_eval[a] / eps_persist[a]
    ssr_agg = float(np.mean(list(ssr.values())))
    t["c9_eval_pred"] = {"pass": bool(ssr_agg <= 0.40),
                         "eps_eval_proj": {str(a): round(eps_eval[a], 6) for a in (0, 1)},
                         "persistence": {str(a): round(eps_persist[a], 6) for a in (0, 1)},
                         "ssr": {str(a): round(ssr[a], 6) for a in (0, 1)},
                         "ssr_agg": round(ssr_agg, 6)}

    # ---- C10: spectral radius + 5-step open-loop rollout (full-state re-projection) ----
    rho = {a: float(np.max(np.abs(np.linalg.eigvals(ops[a])))) for a in (0, 1)}
    win_errors = {0: [], 1: []}
    win_full = {0: [], 1: []}
    by_ep = {}
    for k in EVAL_IDS:
        by_ep[k] = load_recs(EVAL_DIR, [k])
    for k, recs in by_ep.items():
        T = len(recs)
        for t0 in range(T - HORIZON):
            acts = [int(recs[t0 + h]["action"]) for h in range(HORIZON)]
            x_true = encode_flat(enc, np.asarray([recs[t0 + HORIZON]["obs_t"]], dtype=np.float32))[0]
            x_hat = encode_flat(enc, np.asarray([recs[t0]["obs_t"]], dtype=np.float32))[0]
            for h in range(HORIZON):
                a = acts[h]
                x_hat = (x_hat @ per[a]["V16"]) @ ops[a] @ per[a]["V16"].T
            a_last = acts[-1]
            V = per[a_last]["V16"]
            proj = float(np.linalg.norm((x_hat - x_true) @ V) / np.linalg.norm(x_true @ V))
            full = float(np.linalg.norm(x_hat - x_true) / np.linalg.norm(x_true))
            win_errors[a_last].append(proj)
            win_full[a_last].append(full)
    roll_agg = float(np.mean([e for v in win_errors.values() for e in v])) if any(win_errors.values()) else float("nan")
    c10 = all(v <= 1.05 for v in rho.values()) and roll_agg <= 0.35
    t["c10_stability"] = {"pass": bool(c10),
                          "spectral_radius": {str(a): round(rho[a], 6) for a in (0, 1)},
                          "rollout_proj_err": {str(a): (round(float(np.mean(v)), 6) if v else None) for a, v in win_errors.items()},
                          "rollout_full_err_diag": {str(a): (round(float(np.mean(v)), 6) if v else None) for a, v in win_full.items()},
                          "rollout_proj_agg": round(roll_agg, 6)}

    # ---- C12: baselines ----
    t["baselines"] = {"persistence_eval": {str(a): round(eps_persist[a], 6) for a in (0, 1)},
                      "calib_mean_eval": {}}
    for a in (0, 1):
        ma = acts_e == a
        Xa, Ya, V = Xe[ma], Ye[ma], per[a]["V16"]
        t["baselines"]["calib_mean_eval"][str(a)] = round(
            float(np.linalg.norm((Ya - Ya.mean(axis=0)) @ V) / np.linalg.norm(Ya @ V)), 6)

    # ---- Verdict chain ----
    gates = {"c7": t["c7_conditioning"]["pass"], "c8": t["c8_calib_recon"]["pass"],
             "c9": t["c9_eval_pred"]["pass"], "c10": t["c10_stability"]["pass"]}
    first_fail = next((k for k in ("c7", "c8", "c9", "c10") if not gates[k]), None)
    verdict = "REDUCED_KOOPMAN_SPECTRAL_VERIFIED" if first_fail is None else "CONTRACT_FAILED"
    t["verdict"] = verdict
    t["first_failing_gate"] = first_fail
    t["split"] = {"calib_ids": CALIB_IDS, "eval_ids": EVAL_IDS,
                  "calib_records": len(calib), "eval_records": len(evals),
                  "eval_manifest_sha": "f0c9a7624f26bf70"}

    TELE.write_text(json.dumps(t, indent=1), encoding="utf-8")
    np.savez(OPS, K0=ops[0], K1=ops[1], V0=per[0]["V16"], V1=per[1]["V16"])
    print(json.dumps({k: t.get(k) for k in ("verdict", "first_failing_gate")}))
    print("TELE_SHA", hashlib.sha256(TELE.read_bytes()).hexdigest()[:16])
    print("OPS_SHA", hashlib.sha256(OPS.read_bytes()).hexdigest()[:16])


if __name__ == "__main__":
    main()
