#!/usr/bin/env python3
"""Ground the SplitJEPA matched invariant pairs D_inv in REAL protein structure.

DIRECTIVE
    HENRI-SPEC-2026-SPLITJEPA-SYS-V1 section 8 item 1: "Ground the matched invariant
    pairs D_inv in the Fold2Reason protein structural dataset to satisfy Sigma_dL > 0."
    SplitJEPA (Hua et al., arXiv:2610.12349) needs D_inv pairs that SHARE the invariant
    block and VARY in the variant block with sufficient rank. Without that, block
    identification A12 = 0 is unearned.

WHAT THE INVARIANT AND THE VARIANT ARE HERE
    A single NMR entry deposits MANY experimentally determined models of the SAME
    molecule. That is a natural, non-synthetic D_inv:
        z_H  (INVARIANT) = the residue-residue CONTACT MAP, binarised at 8 A. This is the
                           Fold2Reason contact definition and the fold topology. The
                           directive says the invariant is the conserved topological fold
                           class; across models of one molecule the fold is identical.
        z_L  (VARIANT)   = the C-alpha pairwise DISTANCE VECTOR (upper triangle). This is
                           the geometric realisation, which genuinely differs model to
                           model with the thermal/experimental ensemble.

WHY THAT IS NOT A TAUTOLOGY
    A coarse contact map could be constant BY CONSTRUCTION if the threshold were far from
    every distance. So the script includes a negative control that perturbs the
    coordinates and confirms the contact-map operator DOES respond. If z_H could never
    move, its zero variation would prove nothing; the control is what makes it evidence.

    The sufficient-variation condition Sigma_dL > 0 is ALSO falsifiable here: it fails if
    any direction of z_L is frozen across the ensemble. The script reports the singular
    spectrum, the rank, and the smallest positive eigenvalue, and states plainly that
    full positive-definiteness in the 2850-dim space would need more structures than the
    ensemble provides -- the binding constraint is the model count, NOT a property of the
    protein.

WHAT THIS IS NOT
    It is NOT the Fold2Reason FoldingCorpus, which is derived from the OpenFold monomer
    collection and is not in this tree. It is a D_inv construction of the SAME KIND using
    public experimental structures. No protein is claimed to solve any reasoning task.
    Staging structure is not understanding it. No model capability is measured.

FAIL-CLOSED GUARDS
    body must look like PDB text; at least MIN_MODELS models must parse; every model must
    have the same residue count; and the negative control must show the operator responds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time
import urllib.request

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
_VERIF = REPO / "HENRI V2" / "experiments" / "verification"
if str(_VERIF) not in sys.path:
    sys.path.insert(0, str(_VERIF))
from receipt_path import is_redirected, resolve_receipt_path  # noqa: E402

DEFAULT_RECEIPT = REPO / "design" / "zone_a" / "evidence" / "henri_protein_dinv_observed.json"
CONTACT_A = 8.0
MIN_MODELS = 5
UA = "Mozilla/5.0 (compatible; HENRI-protein-dinv/1.0; +https://github.com/cjc214foodun9/HENRI)"
PDB_URL = "https://files.rcsb.org/download/{pid}.pdb"


def _assert_tracked(p: pathlib.Path) -> None:
    if not (p / "design" / "zone_a" / "evidence").is_dir():
        raise RuntimeError(f"resolved repo root {p} has no design/zone_a/evidence/ -- refusing "
                           f"to write into an untracked location")


def fetch_pdb(pid: str, timeout: int = 120) -> str:
    req = urllib.request.Request(PDB_URL.format(pid=pid),
                                 headers={"User-Agent": UA, "Accept": "chemical/x-pdb,text/plain"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read().decode("utf-8", errors="replace")
    # FAIL-CLOSED: RCSB returns an HTML error page for a bad id, which would otherwise
    # parse to zero models and be reported as "no data" rather than "bad request".
    if "ATOM" not in body:
        raise RuntimeError(f"{pid}: response contains no ATOM records "
                           f"(first 60 chars: {body[:60]!r}) -- bad id or an HTML error page")
    return body


def parse_models(text: str) -> list:
    """Return a list of models; each model is a list of (resseq, x, y, z) for CA atoms."""
    models, cur = [], []
    for line in text.splitlines():
        rec = line[:6].strip()
        if rec == "MODEL":
            cur = []
        elif rec == "ENDMDL":
            if cur:
                models.append(cur)
            cur = []
        elif rec == "ATOM" and line[12:16].strip() == "CA":
            try:
                cur.append((int(line[22:26]), float(line[30:38]),
                            float(line[38:46]), float(line[46:54])))
            except ValueError:
                continue
    if cur:                                    # X-ray files have no MODEL record
        models.append(cur)
    out = []
    for m in models:
        m = sorted(m, key=lambda r: r[0])
        out.append(([r[0] for r in m], np.array([r[1:] for r in m], dtype=float)))
    return out


def _upper(D: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(D.shape[0], k=1)
    return D[iu]


def invariants(X: np.ndarray, a: float = CONTACT_A):
    """z_H = binarised contact map (invariant); z_L = distance vector (variant)."""
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    dist = _upper(D)
    return (dist < a).astype(float), dist


def rmsd(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((A - B) ** 2, axis=1))))


def analyse(pid: str, models: list, a: float = CONTACT_A) -> dict:
    n = len(models)
    counts = {len(m[0]) for m in models}
    if len(counts) != 1:
        raise RuntimeError(f"{pid}: models disagree on residue count {sorted(counts)}")
    nres = counts.pop()

    ZH = np.array([invariants(X, a)[0] for _, X in models])       # [n, n_contacts]
    ZL = np.array([invariants(X, a)[1] for _, X in models])       # [n, n_pairs]

    # ---- invariance of z_H across the matched pairs -------------------------------
    zh_var = float(ZH.max(axis=0).max() - ZH.min(axis=0).min())
    differs = int(np.sum(ZH.max(axis=0) != ZH.min(axis=0)))
    # a contact that is unstable across the ensemble isolates WHERE any variation sits
    unstable = int(np.sum((ZH.max(axis=0) - ZH.min(axis=0)) > 0))

    # ---- sufficient variation of z_L: Sigma_dL = 2 * Cov(z_L) ---------------------
    Zc = ZL - ZL.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Zc, compute_uv=False)
    eig = 2.0 * (s ** 2) / max(1, n - 1)
    tol = max(eig) * 1e-10
    rank = int(np.sum(eig > tol))
    pos = eig[eig > tol]

    # ---- negative control: the z_H operator MUST respond to a coordinate change ----
    ctrl = []
    rng = np.random.default_rng(20261011)
    X0 = models[0][1]
    for noise in (0.05, 0.25, 1.0):
        Xp = X0 + rng.normal(0.0, noise, size=X0.shape)
        zh0 = invariants(X0, a)[0]
        zhp = invariants(Xp, a)[0]
        ctrl.append({"noise_angstrom": noise,
                     "contacts_flipped": int(np.sum(zh0 != zhp)),
                     "frac_flipped": round(float(np.mean(zh0 != zhp)), 6)})

    # ---- the ensemble is genuinely different, not a format artifact ----------------
    rms = [rmsd(models[i][1], models[j][1])
           for i in range(min(n, 8)) for j in range(i + 1, min(n, 8))]

    return {
        "pdb_id": pid,
        "n_models": n,
        "n_residues": nres,
        "contact_threshold_angstrom": a,
        "z_H_invariant": {
            "n_contact_features": int(ZH.shape[1]),
            "n_contacts_mean": round(float(ZH.mean()), 3),
            "max_abs_variation": zh_var,
            "n_unstable_contacts": unstable,
            "n_contacts_that_differ": differs,
            "invariance_holds": differs == 0,
        },
        "z_L_variant": {
            "dim": int(ZL.shape[1]),
            "sigma_dL_rank": rank,
            "sigma_dL_min_positive_eig": float(pos.min()) if pos.size else 0.0,
            "sigma_dL_max_eig": float(eig[0]),
            "condition_number": float(eig[0] / pos.min()) if pos.size else float("inf"),
            "positive_definite_on_support": bool(pos.size and pos.min() > 0),
            "full_rank_note": (f"rank {rank} of a {ZL.shape[1]}-dim space. FULL positive "
                              f"definiteness needs more structures than the space has "
                              f"dimensions; the binding constraint is the ensemble size "
                              f"n={n}, not the protein."),
        },
        "ensemble_spread": {
            "pairwise_rmsd_min": round(min(rms), 4),
            "pairwise_rmsd_median": round(float(np.median(rms)), 4),
            "pairwise_rmsd_max": round(max(rms), 4),
        },
        "negative_control_contact_operator_responds": ctrl,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None, help="receipt path override")
    ap.add_argument("--entries", default="1D3Z,2K39",
                    help="comma-separated PDB ids (NMR ensembles preferred)")
    args = ap.parse_args()
    _assert_tracked(REPO)

    rows, failures = [], []
    for pid in [p.strip() for p in args.entries.split(",") if p.strip()]:
        print(f"--- {pid}")
        try:
            text = fetch_pdb(pid)
            models = parse_models(text)
            if len(models) < MIN_MODELS:
                raise RuntimeError(f"only {len(models)} models parsed (< {MIN_MODELS}); an "
                                   f"X-ray single-structure file cannot supply D_inv pairs")
            got = analyse(pid, models)
            got["pdb_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
            rows.append(got)
            print(f"    n_models={got['n_models']} n_res={got['n_residues']} "
                  f"z_H_differs={got['z_H_invariant']['n_contacts_that_differ']} "
                  f"rank(Sigma_dL)={got['z_L_variant']['sigma_dL_rank']}")
        except Exception as exc:                              # noqa: BLE001
            failures.append((pid, f"{type(exc).__name__}: {exc}"))
            print(f"    FAIL {exc}")
        time.sleep(2.0)

    # the negative control must show the operator responds, or z_H invariance is vacuous
    ctl_ok = all(any(c["contacts_flipped"] > 0 for c in r["negative_control_contact_operator_responds"])
                 for r in rows) if rows else False
    inv_ok = all(r["z_H_invariant"]["invariance_holds"] for r in rows) if rows else False
    var_ok = all(r["z_L_variant"]["sigma_dL_rank"] > 0 for r in rows) if rows else False

    receipt = {
        "schema": "henri.protein.dinv.observed.v1",
        "evidence_class": "OBSERVED",
        "what": ("A real-structure construction of SplitJEPA's matched invariant pairs D_inv. "
                 "z_H is the 8 A contact map (fold topology); z_L is the Ca distance vector "
                 "(geometry). Pairs are models WITHIN one NMR entry, which share the molecule "
                 "and therefore the fold."),
        "not_the_fold2reason_corpus": ("This is NOT the Fold2Reason FoldingCorpus (OpenFold "
                                       "monomer collection, not in this tree). It is the same "
                                       "KIND of D_inv from public experimental structures."),
        "entries": sorted(rows, key=lambda r: r["pdb_id"]),
        "failures": [{"pdb_id": p, "reason": r} for p, r in failures],
        "checks": {
            "z_H_invariance_holds_on_all_entries": bool(inv_ok),
            "sigma_dL_positive_rank_on_all_entries": bool(var_ok),
            "negative_control_operator_responds": bool(ctl_ok),
        },
        "verdict": ("DINV_INVARIANCE_AND_VARIATION_OBSERVED" if (inv_ok and var_ok and ctl_ok)
                    else "DINV_UNRESOLVED_SEE_CHECKS"),
        "not_claims": [
            "no model-performance claim",
            "staging structure is not understanding it",
            "no protein is claimed to solve any reasoning task",
            "the invariant block is the contact map by DEFINITION here, not learned",
            "z_H contact-map invariance is verified for THESE ensembles, not universally",
        ],
        "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    default = pathlib.Path(DEFAULT_RECEIPT)
    out = pathlib.Path(args.out).expanduser() if args.out else resolve_receipt_path(default)
    if is_redirected(default, out):
        print(f"   NOTICE: REDIRECTED away from the committed receipt {default}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8",
                   newline="\n")

    print("=" * 74)
    for r in receipt["entries"]:
        print(f"  {r['pdb_id']}: z_H stable={r['z_H_invariant']['invariance_holds']} "
              f"({r['z_H_invariant']['n_contacts_that_differ']} differing contacts), "
              f"rank(Sigma_dL)={r['z_L_variant']['sigma_dL_rank']}, "
              f"min+ eig={r['z_L_variant']['sigma_dL_min_positive_eig']:.4g}, "
              f"rmsd med={r['ensemble_spread']['pairwise_rmsd_median']}")
    print(f"  checks: {receipt['checks']}")
    print(f"  VERDICT: {receipt['verdict']}")
    print(f"  wrote {out}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
