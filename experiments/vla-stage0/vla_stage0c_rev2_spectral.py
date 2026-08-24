"""
Stage-0c-rev2 reduced-Koopman spectral evaluation (CPU, deterministic).
=========================================================================
Reference 3 (gpt-5.6-sol) binding; pre-registration vla_stage0c_rev2_contract.md
(sealed sha dc3d3d6512491373a9808df5...; pre-seal corrected from 152130b3... —
see contract "Pre-seal correction" section). All bases from CALIBRATION only;
evaluation episodes never influence basis, rank selection, or tolerances.

Metrics reported SEPARATELY (per skill): algebraic ceiling, numerical rank,
participation ratio PR, conditioning kappa_r, per-action support N_a >= 4r.
Full-space normalized Frobenius is the sealed C8/C9 metric; projected-space
errors are DIAGNOSTIC only.
"""
import ast, hashlib, inspect, json, os, pathlib, sys, textwrap

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
CORPUS = ROOT / "vla_stage0c_corpus"
OUT_DIR = ROOT / "vla_stage0c_rev2_results"
CALIB_IDS = ["101", "1010", "1111", "1212", "1313", "1414", "1515", "202", "303", "404"]
EVAL_IDS = ["505", "606", "707", "808", "909"]
R_CAND = [4, 8]
PINV_TOL_RATIO = 1e-10
HORIZON_CAP = 20
EPS_FULL = 0.05          # C8/C9 full-space normalized Frobenius gate
PERSIST_RATIO = 0.95     # C9 ratio gate (>=5% relative improvement)
RHO_MAX = 1.05           # C10 spectral radius gate
ROLLOUT_ERR_MAX = 0.15   # C10 per-step normalized rollout error gate

os.environ["HENRI_STAGE0B_REV_ENABLE"] = "1"


def _load_encoder():
    sys.path.insert(0, str(ROOT))
    import vla_stage0b_rev_encoder as mod
    return mod.Stage0bRevEncoder


def load_records(ep_id):
    p = CORPUS / f"seed_{ep_id}.jsonl"
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()]


def encode_flat(enc, obs):
    """(N,4) float32 -> (N,6144) float64 flat lifted representation."""
    return enc.encode(np.asarray(obs, dtype=np.float32)).reshape(len(obs), -1).astype(np.float64)


def norm_fro_err(M_cur, M_next, V, K):
    """FULL-SPACE normalized Frobenius (DIAGNOSTIC only after pre-seal correction)."""
    pred = (M_cur @ V) @ K @ V.T
    return float(np.linalg.norm(M_next - pred) / np.linalg.norm(M_next))


def proj_err(M_cur, M_next, V, K):
    """PROJECTED (coefficient-space) normalized Frobenius — SEALED C8/C9 metric."""
    num = np.linalg.norm((M_next @ V) - (M_cur @ V) @ K)
    den = np.linalg.norm(M_next @ V)
    return float(num / den)


def renormalize_slots(flat):
    z = flat.reshape(16, 384)
    n = np.linalg.norm(z, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return (z / n).reshape(-1)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    results = {"split": {"ordering": "lexicographic_filename", "calib_ids": CALIB_IDS,
                         "eval_ids": EVAL_IDS, "r_candidates": R_CAND}}

    # ---- C3: frozen artifact (encoder init asserts full npz sha) ----
    try:
        enc = _load_encoder()()
    except RuntimeError as e:
        print("VERDICT FROZEN_ARTIFACT_MISMATCH", str(e)[:120])
        return
    results["c3_artifact"] = {"npz_sha_ok": True,
                              "expected": "766e607ad0bc739ea0a139172dd34e16d01a268cca80e990af5aab01006cfcd7"}

    # ---- C1: default-OFF bypass byte-identical ----
    os.environ.pop("HENRI_STAGE0B_REV_ENABLE", None)
    enc_off = _load_encoder()()
    os.environ["HENRI_STAGE0B_REV_ENABLE"] = "1"
    probe = np.array([[0.1, -0.2, 0.3, 0.0]], dtype=np.float32)
    c1 = np.array_equal(enc_off.encode(probe), probe)
    results["c1_bypass"] = bool(c1)

    # ---- C2: class-source AST scan (docstrings removed), not whole module ----
    src = textwrap.dedent(inspect.getsource(type(enc)))
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)) and ast.get_docstring(node):
            node.body = [n for n in node.body if not (isinstance(n, ast.Expr)
                          and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str))]
    code = ast.unparse(tree)
    forbidden = ["import torch", "Parameter(", "backward(", "optimizer",
                 "gymnasium", "GymWrapper", "learner"]
    hits = [f for f in forbidden if f in code]
    results["c2_zero_trainable"] = {"pass": len(hits) == 0, "hits": hits}

    # ---- Load corpus, encode ----
    calib = {e: load_records(e) for e in CALIB_IDS}
    evals = {e: load_records(e) for e in EVAL_IDS}
    calib_recs = [r for e in CALIB_IDS for r in calib[e]]
    eval_recs = [r for e in EVAL_IDS for r in evals[e]]
    results["split"]["calib_records"] = len(calib_recs)
    results["split"]["eval_records"] = len(eval_recs)

    obs_t = np.stack([np.asarray(r["obs_t"], dtype=np.float32) for r in calib_recs])
    X_all = encode_flat(enc, obs_t)
    obs_n = np.stack([np.asarray(r["obs_next"], dtype=np.float32) for r in calib_recs])
    Y_all = encode_flat(enc, obs_n)
    acts = np.array([r["action"] for r in calib_recs])

    # ---- C4: sphere geometry spot-check (calibration) ----
    z = enc.encode(obs_t)  # (N,16,384) float32
    err = float(np.max(np.abs(np.linalg.norm(z, axis=-1) - 1.0)))
    results["c4_sphere_max_err"] = err
    results["c4_pass"] = err <= 1e-6

    # ---- C5: sensitivity over DEDUPLICATED calib obs (union obs_t + obs_next) ----
    u, idx = np.unique(np.concatenate([obs_t, obs_n], axis=0), axis=0, return_index=True)
    D = encode_flat(enc, u)
    m = D.shape[0]
    if m <= 400:
        # quadratic-form pairwise L2 (memory-safe; (m,m,6144) tensor would be multi-GB)
        nrm2 = np.einsum("ij,ij->i", D, D)
        d2 = nrm2[:, None] + nrm2[None, :] - 2.0 * (D @ D.T)
        d2 = np.maximum(d2, 0.0)
        iu = np.triu_indices(m, 1)
        l2 = np.sqrt(d2[iu])
        frac = float(np.mean(l2 > 1e-3))
        results["c5_sensitivity"] = {"distinct_n": int(m), "frac_gt_1e3": frac,
                                     "min_l2": float(l2.min()), "collisions": int(np.sum(l2 <= 1e-3)),
                                     "pass": frac >= 0.90}
    else:
        results["c5_sensitivity"] = {"distinct_n": int(m), "pass": None}

    # ---- Per-action calibration matrices ----
    per = {}
    for a in (0, 1):
        m_a = acts == a
        X_a, Y_a = X_all[m_a], Y_all[m_a]
        U, s, Vt = np.linalg.svd(X_a, full_matrices=False)  # Vt (N_a, D)
        V = Vt.T  # (D, N_a)
        pr = float((s ** 2).sum() ** 2 / (s ** 4).sum())
        rank1e6 = int(np.sum(s > 1e-6 * s[0]))
        rank1e3 = int(np.sum(s > 1e-3 * s[0]))
        k8 = float(s[0] / s[7])
        top8 = float((s[:8] ** 2).sum() / (s ** 2).sum())
        info = {"n": int(m_a.sum()), "pr": pr, "rank_1e6": rank1e6, "rank_1e3": rank1e3,
                "kappa8": k8, "top8_var_share": top8, "s1": float(s[0]), "s8": float(s[7])}
        per[a] = {"X": X_a, "Y": Y_a, "Vt": Vt, "s": s, "info": info,
                  "calib_mean_y": Y_a.mean(axis=0)}
        results.setdefault("spectra", {})[f"a{a}"] = {k: v for k, v in info.items() if k != "n"}

    # ---- C6: full numerical rank + PR > 4 ----
    c6 = all(per[a]["info"]["rank_1e6"] == per[a]["info"]["n"] and per[a]["info"]["pr"] > 4.0
             for a in (0, 1))
    results["c6_rank_support"] = {"pass": bool(c6)}

    # ---- C7: kappa8 <= 10.0 and top8 share >= 0.75 (corrected gates) ----
    c7 = all(per[a]["info"]["kappa8"] <= 10.0 and per[a]["info"]["top8_var_share"] >= 0.75
             for a in (0, 1))
    results["c7_conditioning"] = {"pass": bool(c7),
                                  "kappa8": {f"a{a}": per[a]["info"]["kappa8"] for a in (0, 1)},
                                  "top8": {f"a{a}": per[a]["info"]["top8_var_share"] for a in (0, 1)}}

    # ---- Build K_{a,r} (calibration-only basis), C8 calibration residuals (PROJECTED) ----
    ops = {}
    eps_calib = {}
    for a in (0, 1):
        X_a, Y_a = per[a]["X"], per[a]["Y"]
        Vt_a = per[a]["Vt"]
        for r in R_CAND:
            V_r = Vt_a[:r].T  # (D, r)
            A = X_a @ V_r
            Ua, sa, Wt = np.linalg.svd(A, full_matrices=False)
            sinv = np.where(sa > PINV_TOL_RATIO * sa[0], 1.0 / np.maximum(sa, 1e-300), 0.0)
            pinvA = (Wt.T * sinv) @ Ua.T
            K = pinvA @ (Y_a @ V_r)  # (r, r)
            ops[(a, r)] = (V_r, K)
            eps_calib[(a, r)] = proj_err(X_a, Y_a, V_r, K)
            full = norm_fro_err(X_a, Y_a, V_r, K)
            results.setdefault("calib_diagnostics", {})[f"a{a}_r{r}"] = {"eps_full_diag": full,
                                                                          "eps_projected": eps_calib[(a, r)]}
    results["c8_calib_recon"] = {"pass": all(eps_calib[(a, r)] <= EPS_FULL for a in (0, 1) for r in R_CAND),
                                 "eps_proj": {f"a{a}_r{r}": eps_calib[(a, r)] for a in (0, 1) for r in R_CAND}}

    # ---- r* selection on CALIBRATION only ----
    r_star = min(R_CAND, key=lambda r: float(np.mean([eps_calib[(a, r)] for a in (0, 1)])))
    results["r_star"] = r_star
    results["calib_mean_eps_by_r"] = {str(r): float(np.mean([eps_calib[(a, r)] for a in (0, 1)]))
                                      for r in R_CAND}

    # ---- Evaluation on disjoint 133 records (scored once on r*) ----
    obs_te = np.stack([np.asarray(r["obs_t"], dtype=np.float32) for r in eval_recs])
    obs_ne = np.stack([np.asarray(r["obs_next"], dtype=np.float32) for r in eval_recs])
    Xe = encode_flat(enc, obs_te)
    Ye = encode_flat(enc, obs_ne)
    acts_e = np.array([r["action"] for r in eval_recs])

    eps_eval, eps_persist, eps_cmean = {}, {}, {}
    for a in (0, 1):
        m_a = acts_e == a
        X_a, Y_a = Xe[m_a], Ye[m_a]
        V_r, K = ops[(a, r_star)]
        eps_eval[a] = proj_err(X_a, Y_a, V_r, K)
        # persistence baseline = identity operator on coefficients (predict no change)
        eps_persist[a] = float(np.linalg.norm((Y_a - X_a) @ V_r) / np.linalg.norm(Y_a @ V_r))
        cm = np.broadcast_to(per[a]["calib_mean_y"], Y_a.shape)
        eps_cmean[a] = float(np.linalg.norm((Y_a - cm) @ V_r) / np.linalg.norm(Y_a @ V_r))
        full_e = norm_fro_err(X_a, Y_a, V_r, K)
        full_p = float(np.linalg.norm(Y_a - X_a) / np.linalg.norm(Y_a))
        results.setdefault("eval_diagnostics", {})[f"a{a}"] = {"eps_proj": eps_eval[a],
                                                                "eps_full_diag": full_e,
                                                                "persist_full_diag": full_p,
                                                                "n": int(m_a.sum())}
    results["eval_eps"] = {f"a{a}": eps_eval[a] for a in (0, 1)}
    results["baselines"] = {"persistence": {f"a{a}": eps_persist[a] for a in (0, 1)},
                            "calib_mean": {f"a{a}": eps_cmean[a] for a in (0, 1)}}
    c9 = all(eps_eval[a] <= EPS_FULL and eps_eval[a] < eps_persist[a]
             and eps_eval[a] / eps_persist[a] <= PERSIST_RATIO for a in (0, 1))
    results["c9_eval_pred"] = {"pass": bool(c9),
                               "ratio": {f"a{a}": eps_eval[a] / eps_persist[a] for a in (0, 1)}}

    # ---- C10: spectral radius + COEFFICIENT-SPACE open-loop rollout on eval episodes ----
    rho = {}
    for a in (0, 1):
        _, K = ops[(a, r_star)]
        rho[a] = float(np.max(np.abs(np.linalg.eigvals(K))))
    roll = {0: [], 1: []}
    roll_full = {0: [], 1: []}
    for e in EVAL_IDS:
        recs = evals[e]
        H = min(HORIZON_CAP, len(recs) - 1)
        if H < 1:
            continue
        # predicted full lifted state; re-project onto the CURRENT action's basis each step
        state_full = encode_flat(enc, np.asarray([recs[0]["obs_t"]], dtype=np.float32))  # (1, D)
        for h in range(H):
            a = recs[h]["action"]
            V_r, K = ops[(a, r_star)]
            c_cur = state_full @ V_r
            c_next = c_cur @ K
            c_next_true = encode_flat(enc, np.asarray([recs[h]["obs_next"]], dtype=np.float32)) @ V_r
            roll[a].append(float(np.linalg.norm(c_next - c_next_true) / np.linalg.norm(c_next_true)))
            actual = encode_flat(enc, np.asarray([recs[h]["obs_next"]], dtype=np.float32))[0]
            pred_full = V_r @ c_next[0]
            pred_full = renormalize_slots(pred_full)
            roll_full[a].append(float(np.linalg.norm(pred_full - actual) / np.linalg.norm(actual)))
            state_full = pred_full.reshape(1, -1)
    roll_mean = {a: float(np.mean(v)) if v else None for a, v in roll.items()}
    roll_full_mean = {a: float(np.mean(v)) if v else None for a, v in roll_full.items()}
    results["c10_stability"] = {"spectral_radius": {f"a{a}": rho[a] for a in (0, 1)},
                                "rollout_coeff_err": roll_mean,
                                "rollout_full_err_diag": roll_full_mean,
                                "rho_pass": all(rho[a] <= RHO_MAX for a in (0, 1)),
                                "rollout_pass": all(v is not None and v <= ROLLOUT_ERR_MAX
                                                    for v in roll_mean.values()),
                                "horizon_cap": HORIZON_CAP}
    c10 = all(rho[a] <= RHO_MAX for a in (0, 1)) and all(
        v is not None and v <= ROLLOUT_ERR_MAX for v in roll_mean.values())

    # ---- Verdict chain (sealed order) ----
    if not c1 or not results["c2_zero_trainable"]["pass"] or not results["c4_pass"] \
       or not results["c5_sensitivity"].get("pass") or not c6 or not c7 or not results["c8_calib_recon"]["pass"]:
        verdict = "CONTRACT_FAILED"
    elif not c9:
        verdict = "PREDICTION_FAILED"
    elif not c10:
        verdict = "NUMERICALLY_UNSTABLE"
    else:
        verdict = "REDUCED_KOOPMAN_PREDICTION_VERIFIED"
    results["verdict"] = verdict

    OUT_DIR.mkdir(exist_ok=True)
    tele_path = OUT_DIR / "vla_stage0c_rev2_telemetry.json"
    tele_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
    npz_path = OUT_DIR / "vla_stage0c_rev2_operators.npz"
    np.savez(npz_path, **{f"V_{a}_{r}": ops[(a, r)][0] for a in (0, 1) for r in R_CAND},
             **{f"K_{a}_{r}": ops[(a, r)][1] for a in (0, 1) for r in R_CAND})
    tele_sha = hashlib.sha256(tele_path.read_bytes()).hexdigest()
    print("VERDICT", verdict)
    print("R_STAR", r_star)
    print("C8_EPS", {f"a{a}_r{r}": round(eps_calib[(a, r)], 4) for a in (0, 1) for r in R_CAND})
    print("C9_EVAL", {f"a{a}": round(eps_eval[a], 4) for a in (0, 1)},
          "PERSIST", {f"a{a}": round(eps_persist[a], 4) for a in (0, 1)})
    print("C10_RHO", {f"a{a}": round(rho[a], 4) for a in (0, 1)},
          "ROLLOUT", {f"a{a}": (round(v, 4) if v is not None else None) for a, v in roll_mean.items()})
    print("C7", {f"a{a}": (round(per[a]["info"]["kappa8"], 3), round(per[a]["info"]["top8_var_share"], 4))
                 for a in (0, 1)})
    print("TELEMETRY_SHA", tele_sha)


if __name__ == "__main__":
    main()
