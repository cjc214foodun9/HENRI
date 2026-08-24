"""
Stage-0c premise audit v2: lifted-tuple identifiability on REAL CartPole corpus.
=================================================================================
Reference 3 (gpt-5.6-sol) binding. NO learning, NO adapter code.

Loads the fresh development corpus (vla_stage0c_corpus/*.jsonl, built from the
VERIFIED Stage-0a wrapper). Measures singular spectra and effective rank of:
  - Z_t   (state encoding, flat 6144)
  - Y     (successor encoding)
  - X_0, X_1 (action-conditioned state encodings, SEPARATE per action)
  - joint [X_a | Y_a] per action
Fits the reduced basis V_r on a CALIBRATION partition (first 70% of episodes);
evaluates condition and residual on DISJOINT held-out episodes. Requires
N_a >= 4r for EACH action. Freezes r below the measured effective-rank floor.

Design decision (corpus #23): additive action embeddings are REJECTED
(crosstalk). The adapter will use SEPARATE operators K_0, K_1; this audit
therefore reports per-action spectra, NOT a lifted joint matrix.
"""
import hashlib, json, os, pathlib, sys

import numpy as np

import vla_stage0b_encoder as enc

os.environ["HENRI_STAGE0B_ENABLE"] = "1"

D_FLAT = 16 * 384  # 6144
CALIB_FRAC = 0.70
R_CANDIDATES = (4, 8, 16, 32)

CORPUS = pathlib.Path("vla_stage0c_corpus")


def load_corpus():
    episodes = []
    for p in sorted(CORPUS.glob("seed_*.jsonl")):
        recs = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            recs.append({
                "obs_t": np.asarray(r["obs_t"], dtype=np.float32),
                "action": int(r["action"]),
                "obs_next": np.asarray(r["obs_next"], dtype=np.float32),
                "episode_id": int(r["episode_id"]),
                "step_id": int(r["step_id"]),
            })
        episodes.append((p.name, recs))
    return episodes


def encode_flat(obs):
    return enc.encode(np.asarray(obs, dtype=np.float32)).reshape(-1).astype(np.float64)


def spectra(M, name):
    s = np.linalg.svd(M, compute_uv=False)
    s2 = s ** 2
    denom = float((s2 ** 2).sum())
    pr = float((s2.sum() ** 2) / denom) if denom > 0 else 0.0
    k1e3 = int((s > 1e-3 * s[0]).sum()) if s[0] > 0 else 0
    k1e6 = int((s > 1e-6 * s[0]).sum()) if s[0] > 0 else 0
    kappa = {}
    for r in R_CANDIDATES:
        kappa[str(r)] = (float(s[0] / s[r - 1]) if r <= len(s) and s[r - 1] > 0
                         else float("inf"))
    return {"name": name, "n": int(M.shape[0]), "d": int(M.shape[1]),
            "s1": float(s[0]) if len(s) else None, "participation_ratio": pr,
            "rank_gt_1e-3": k1e3, "rank_gt_1e-6": k1e6, "kappa_top_r": kappa,
            "top10_s": [float(x) for x in s[:10]]}


def main():
    os.environ["HENRI_STAGE0B_ENABLE"] = "1"
    episodes = load_corpus()
    n_ep = len(episodes)
    n_calib = max(1, int(round(CALIB_FRAC * n_ep)))
    calib, hold = episodes[:n_calib], episodes[n_calib:]

    def collect(ep_list):
        pairs = []
        for _, recs in ep_list:
            for r in recs:
                pairs.append((r["obs_t"], r["action"], r["obs_next"]))
        # dedupe identical (obs_t bytes, action)
        seen, out = set(), []
        for ot, a, on in pairs:
            key = (ot.tobytes(), a)
            if key not in seen:
                seen.add(key)
                out.append((ot, a, on))
        return out

    cal_pairs = collect(calib)
    hold_pairs = collect(hold)
    n_c = len(cal_pairs)
    n_h = len(hold_pairs)
    print(f"EPISODES {n_ep} calib {n_calib} hold {n_ep - n_calib} "
          f"pairs calib {n_c} hold {n_h}")

    def build(pairs):
        Zt = np.stack([encode_flat(ot) for ot, _, _ in pairs])
        Y = np.stack([encode_flat(on) for _, _, on in pairs])
        X0 = np.stack([encode_flat(ot) for ot, a, _ in pairs if a == 0])
        Y0 = np.stack([encode_flat(on) for _, a, on in pairs if a == 0])
        X1 = np.stack([encode_flat(ot) for ot, a, _ in pairs if a == 1])
        Y1 = np.stack([encode_flat(on) for _, a, on in pairs if a == 1])
        return Zt, Y, X0, Y0, X1, Y1

    Zt_c, Y_c, X0_c, Y0_c, X1_c, Y1_c = build(cal_pairs)

    out = {"encoder_init_hash": enc.init_hash(),
           "corpus_manifest": "vla_stage0c_corpus/manifest.json",
           "n_episodes": n_ep, "n_calib": n_calib, "n_hold": n_ep - n_calib,
           "n_pairs_calib": n_c, "n_pairs_hold": n_h,
           "calib_actions": {"a0": int((np.asarray([p[1] for p in cal_pairs]) == 0).sum()),
                             "a1": int((np.asarray([p[1] for p in cal_pairs]) == 1).sum())},
           "hold_actions": {"a0": int((np.asarray([p[1] for p in hold_pairs]) == 0).sum()),
                            "a1": int((np.asarray([p[1] for p in hold_pairs]) == 1).sum())},
           "lift_design": "separate operators K_0, K_1 (additive embeddings REJECTED, corpus #23)"}

    for M, name in [(Zt_c, "Zt_state"), (Y_c, "Y_successor"),
                    (X0_c, "X0_action0"), (Y0_c, "Y0_action0"),
                    (X1_c, "X1_action1"), (Y1_c, "Y1_action1")]:
        sp = spectra(M, name)
        out[name] = sp
        print("%-12s n=%-4d pr=%-7.1f r1e-3=%-4d r1e-6=%-4d kappa16=%.1e"
              % (name, sp["n"], sp["participation_ratio"], sp["rank_gt_1e-3"],
                 sp["rank_gt_1e-6"], sp["kappa_top_r"]["16"]))

    # effective-rank floors per action (min over Zt, X_a, Y_a)
    floors = {"a0": min(out["Zt_state"]["participation_ratio"],
                        out["X0_action0"]["participation_ratio"],
                        out["Y0_action0"]["participation_ratio"]),
              "a1": min(out["Zt_state"]["participation_ratio"],
                        out["X1_action1"]["participation_ratio"],
                        out["Y1_action1"]["participation_ratio"])}
    n_a0, n_a1 = out["calib_actions"]["a0"], out["calib_actions"]["a1"]
    ok = {}
    for a_name, floor, na in (("a0", floors["a0"], n_a0), ("a1", floors["a1"], n_a1)):
        cands = [r for r in R_CANDIDATES if r < floor and na >= 4 * r]
        ok[a_name] = cands
        print("ACTION %s floor %.1f N %d -> OK_R %s" % (a_name, floor, na, cands))
    out["r_floor_per_action"] = floors
    out["r_candidates_ok_per_action"] = ok

    r = 16
    valid = all(r in ok[a] for a in ("a0", "a1"))
    if not valid:
        r = 8 if all(8 in ok[a] for a in ("a0", "a1")) else 0
    out["basis_rank_chosen"] = r

    if r > 0:
        # Fit basis on CALIBRATION action-0 matrix only (no held-out leakage).
        _, _, Vt = np.linalg.svd(X0_c, full_matrices=False)
        V_r = Vt[:r].astype(np.float32)
        np.save("vla_stage0c_basis_r%d.npy" % r, V_r)
        out["basis_sha256"] = hashlib.sha256(V_r.tobytes()).hexdigest()
        # Held-out condition + residual: project held X0 onto V_r
        X0_h = np.stack([encode_flat(ot) for ot, a, _ in hold_pairs if a == 0])
        Y0_h = np.stack([encode_flat(on) for _, a, on in hold_pairs if a == 0])
        if len(X0_h) >= r:
            proj = X0_h @ V_r.T
            recon = proj @ V_r
            rel_err = float(np.linalg.norm(X0_h - recon) / max(np.linalg.norm(X0_h), 1e-12))
            cond = float(np.linalg.cond(V_r @ V_r.T))
            out["holdout_basis"] = {"n": int(len(X0_h)), "rel_recon_err": rel_err,
                                    "cond_VrVrT": cond}
            print("HOLDOUT basis recon rel_err %.3e cond(VrVrT) %.1e"
                  % (rel_err, cond))
        else:
            out["holdout_basis"] = {"n": int(len(X0_h)), "blocked": "n < r"}
    else:
        out["basis_rank_chosen"] = 0
        print("NO VALID r — IDENTIFIABILITY BLOCKED")

    with open("vla_stage0c_audit.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    audit_sha = hashlib.sha256(
        json.dumps(out, sort_keys=True).encode()).hexdigest()
    print("AUDIT_JSON_SHA", audit_sha[:16])


if __name__ == "__main__":
    main()
