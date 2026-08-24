"""
Stage-0c-rev identifiability audit — frozen RFF encoder, flat 6144-D, per-action.
==================================================================================
Reference 3 (gpt-5.6-sol) binding; contract vla_stage0b_rev_contract.md.

- Calibration: first 10 episodes in LEXICOGRAPHIC filename order (171 records).
- Evaluation: last 5 episodes (133 records) — untouched for rank selection.
- Matrices per action (flat 6144-D): X0 (obs_t,a=0), Y0 (obs_next,a=0),
  X1 (obs_t,a=1), Y1 (obs_next,a=1).
- Metrics per matrix (raw SVD, float64): PR=(sum s^2)^2/sum s^4,
  r(>1e-3 s1), r(>1e-6 s1), kappa16 = s1/s16.
- Gates: G1 PR>=16 on X0,Y0,X1,Y1; G2 kappa16<=100 on X0,X1;
  G3 r in {4,8,16} with N_a >= 4r both actions (32 excluded: 4*32 > 69).
- Freeze largest passing r; else IDENTIFIABILITY_BLOCKED (basis_rank 0).
- G4 (evidence only): eval reconstruction residual with frozen r.
"""
import hashlib, json, os, pathlib, sys

import numpy as np

import vla_stage0b_rev_encoder as enc_mod
from vla_stage0b_rev_encoder import Stage0bRevEncoder

CORPUS_DIR = pathlib.Path(__file__).resolve().parent / "vla_stage0c_corpus"
R_CANDIDATES = [4, 8, 16, 32]


def load_records():
    manifest = json.loads((CORPUS_DIR / "manifest.json").read_text(encoding="utf-8"))
    files = sorted(manifest["files"].items())
    calib, evald = [], []
    for fname, finfo in files:
        recs = [json.loads(l) for l in (CORPUS_DIR / fname).read_text(encoding="utf-8").splitlines()]
        (calib if len(calib) < 171 else evald).extend(recs)
    return calib, evald


def main():
    os.environ["HENRI_STAGE0B_REV_ENABLE"] = "1"
    enc = Stage0bRevEncoder()

    calib, evald = load_records()
    assert len(calib) == 171 and len(evald) == 133, (len(calib), len(evald))

    def feats(recs, key):
        X = np.stack([np.asarray(r[key], dtype=np.float32) for r in recs])
        Z = enc.encode(X).reshape(len(recs), enc_mod.N_SLOTS * enc_mod.D_SLOT)
        return Z.astype(np.float64)

    Zt = feats(calib, "obs_t")
    Zn = feats(calib, "obs_next")
    acts = np.array([int(r["action"]) for r in calib])

    X0, Y0 = Zt[acts == 0], Zn[acts == 0]
    X1, Y1 = Zt[acts == 1], Zn[acts == 1]

    def stats(M):
        s = np.linalg.svd(M, compute_uv=False)
        pr = float((s ** 2).sum() ** 2 / (s ** 4).sum())
        r1e3 = int(np.sum(s > 1e-3 * s[0]))
        r1e6 = int(np.sum(s > 1e-6 * s[0]))
        k16 = float(s[0] / s[15]) if s.shape[0] >= 16 else float("inf")
        return {"n": int(M.shape[0]), "pr": pr, "r1e3": r1e3, "r1e6": r1e6, "k16": k16}

    stats_all = {
        "X0": stats(X0), "Y0": stats(Y0), "X1": stats(X1), "Y1": stats(Y1),
    }

    pr_ok = all(v["pr"] >= 16.0 for v in stats_all.values())
    k_ok = stats_all["X0"]["k16"] <= 100.0 and stats_all["X1"]["k16"] <= 100.0
    n0, n1 = stats_all["X0"]["n"], stats_all["X1"]["n"]

    frozen_r = 0
    for r in R_CANDIDATES:
        if 4 * r <= n0 and 4 * r <= n1 and pr_ok and k_ok:
            frozen_r = r
    verdict = "RANK_SELECTED" if frozen_r else "IDENTIFIABILITY_BLOCKED"

    g4 = None
    if frozen_r:
        # least-squares on calib per action, truncated to frozen_r (evidence only)
        rel_err = {}
        for name, X, Y in (("a0", X0, Y0), ("a1", X1, Y1)):
            U, s, Vt = np.linalg.svd(X, full_matrices=False)
            K = (Vt[:frozen_r].T * (s[:frozen_r] ** -1)) @ (U[:, :frozen_r].T @ Y)
            Xe = feats([r for r in evald if int(r["action"]) == 0] if name == "a0" else [r for r in evald if int(r["action"]) == 1], "obs_t")
            Ye = feats([r for r in evald if int(r["action"]) == 0] if name == "a0" else [r for r in evald if int(r["action"]) == 1], "obs_next")
            pred = Xe @ K
            rel_err[name] = float(np.linalg.norm(pred - Ye) / max(np.linalg.norm(Ye), 1e-12))
        g4 = {"r": frozen_r, "eval_rel_err": rel_err}

    out = {
        "verdict": verdict,
        "frozen_r": frozen_r,
        "gates": {"G1_pr_ge_16": pr_ok, "G2_kappa16_le_100": k_ok,
                  "n_a0": n0, "n_a1": n1, "r_candidates": R_CANDIDATES},
        "stats": stats_all,
        "g4": g4,
        "contract_sha": "c2ca66e76b9c4c1c034b7fe9",
        "corpus_manifest_sha": "54b7350a58c9491bae3ca877",
    }
    path = pathlib.Path(__file__).resolve().parent / "vla_stage0c_rev_audit.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("VERDICT", verdict, "FROZEN_R", frozen_r)
    print("STATS", json.dumps(stats_all))
    print("GATES", out["gates"])
    if g4:
        print("G4", g4)
    print("AUDIT_SHA", hashlib.sha256(path.read_bytes()).hexdigest()[:24])
    return 0


if __name__ == "__main__":
    sys.exit(main())
