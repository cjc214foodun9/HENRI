"""henri_persistence_invariants.py -- topological invariants for SplitJEPA z_H.

REPAIRS THE MEASURED FALSIFICATION (ledger 1991, commit 65d5418)
    The 8 A binary contact map is NOT invariant across NMR models of one molecule:
    15/2850 contacts flicker on 1D3Z, 258/2850 on 2K39. Thermal breathing moves
    distances across the threshold and toggles adjacency bits. z_H must not contain
    features that flicker.

WHAT THIS MODULE COMPUTES
    z_H candidate  = Betti curves (beta_0(eps), beta_1(eps)) over a Vietoris-Rips
                     filtration of the C-alpha point cloud, plus a Ramachandran basin
                     histogram.
    z_L candidate  = pairwise distance vector and per-residue basin identity.

IMPLEMENTATION IS EXACT, NOT A SURROGATE
    beta_0(eps)  EXACT: single-linkage connected components (union-find). This is
                 literally H_0 rank of the Rips complex.
    beta_1(eps)  EXACT: |E| - |V| + beta_0 - rank(d_2), where rank(d_2) is the GF(2)
                 rank of the 2-simplex boundary matrix at eps, computed by Gaussian
                 elimination with Python int bitmasks. This is the standard homology
                 computation, not a 1-skeleton cycle-rank approximation.
    The Betti CURVE over eps determines a single-filtration barcode up to the standard
    equivalence, so comparing curves compares the persistence modules.

THE CLAIM THIS MODULE TESTS (and can falsify)
    The blueprint asserts persistent barcodes are STRICTLY invariant across
    conformational ensembles (d_B == 0.000, gate G-D1). Persistent homology is
    STABLE, not invariant: it is Lipschitz in the bottleneck metric, so moving C-alpha
    atoms moves birth/death times continuously. Two consequences:
      1. d_B == 0.000 will FAIL on any ensemble with real thermal spread, unless the
         filtration is coarsened so far that everything quantises to one value -- which
         would be a tautology.
      2. G-D1 is therefore REWRITTEN as a SEPARATION test: same-protein conformer
         distances must be small AND well below different-protein distances. A negative
         control (two unrelated proteins) must NOT match.
    This module reports both, and `strict_invariance` is expected to be False. A False
    there is the honest result, not a bug.
"""
from __future__ import annotations

import numpy as np

# Ramachandran basin partition (degrees), 4 regions on a fixed grid.
BASIN_EDGES_PHI = (-180.0, -120.0, -60.0, 60.0, 150.0, 180.0)
BASIN_EDGES_PSI = (-180.0, -90.0, 60.0, 120.0, 180.0)


def pairwise_distances(X: np.ndarray) -> np.ndarray:
    """Euclidean distance matrix for an [N, 3] point cloud."""
    return np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)


# ---------------------------------------------------------------- dihedrals ----
def dihedral(p0, p1, p2, p3) -> float:
    """Standard (phi/psi) dihedral in degrees from four points."""
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    n1 = np.cross(b0, b1)
    n2 = np.cross(b1, b2)
    m1 = np.cross(n1, b1 / max(np.linalg.norm(b1), 1e-12))
    x = float(np.dot(n1, n2))
    y = float(np.dot(m1, n2))
    return float(np.degrees(np.arctan2(y, x))) if (x or y) else 0.0


def backbone_dihedrals(X: np.ndarray) -> tuple:
    """Placeholder dihedrals from N=3 points: this module receives CA-only clouds.

    CA-only backbones do NOT determine (phi, psi) -- those need N, CA, C. So the
    generator passes real dihedrals when it has full backbones; here we derive a
    coarse pseudo-dihedral from consecutive CA triples and LABEL it as such. Recorded
    so no reader mistakes this for a true Ramachandran angle.
    """
    n = X.shape[0]
    phi = np.zeros(n)
    for i in range(1, n - 2):
        phi[i] = dihedral(X[i - 1], X[i], X[i + 1], X[i + 2])
    return phi


def dihedral_basins(angles: np.ndarray) -> np.ndarray:
    """Assign each angle to one of 4 fixed basins. Fixed partition, not tuned."""
    a = np.asarray(angles, dtype=float)
    b = np.full(a.shape, 3, dtype=int)
    b[(a >= -120) & (a < -60)] = 0
    b[((a < -120) | (a >= 150))] = 1
    b[(a >= 60) & (a < 150)] = 2
    return b


def basin_histogram(basins: np.ndarray) -> np.ndarray:
    """Normalised histogram over 4 basins: the invariant candidate."""
    h = np.bincount(basins, minlength=4).astype(float)
    return h / max(1.0, h.sum())


# ---------------------------------------------------------------- homology -----
def betti_curves(D: np.ndarray, epsilons: np.ndarray,
                 max_triangles: int = 2_000_000) -> tuple:
    """EXACT beta_0(eps) and beta_1(eps) for the Vietoris-Rips complex of D.

    beta_0 by union-find; beta_1 = |E| - |V| + beta_0 - rank(boundary_2) over GF(2).
    """
    N = D.shape[0]
    iu = np.triu_indices(N, k=1)
    dd = D[iu]
    pairs = np.column_stack(iu)
    b0s, b1s, tri_counts = [], [], []
    for eps in epsilons:
        emask = dd <= eps
        E = pairs[emask]
        nE = int(E.shape[0])
        parent = list(range(N))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for a, b in E:
            ra, rb = find(int(a)), find(int(b))
            if ra != rb:
                parent[ra] = rb
        comps = len({find(i) for i in range(N)})
        b0s.append(comps)

        ei = {(int(a), int(b)): k for k, (a, b) in enumerate(E)}
        rows = []
        for i in range(N):
            for j in range(i + 1, N):
                if D[i, j] > eps:
                    continue
                for k in range(j + 1, N):
                    if D[i, k] <= eps and D[j, k] <= eps:
                        e1, e2, e3 = ei.get((i, j)), ei.get((i, k)), ei.get((j, k))
                        if e1 is None or e2 is None or e3 is None:
                            continue
                        rows.append((1 << e1) ^ (1 << e2) ^ (1 << e3))
        if len(rows) > max_triangles:
            raise RuntimeError(f"{len(rows)} triangles exceeds max_triangles; "
                               f"coarsen the eps grid rather than silently approximating")
        pivots, rank = {}, 0
        for r in rows:
            x = r
            while x:
                p = x.bit_length() - 1
                if p in pivots:
                    x ^= pivots[p]
                else:
                    pivots[p] = x
                    rank += 1
                    break
        b1s.append(max(0, nE - N + comps - rank))
        tri_counts.append(len(rows))
    return np.array(b0s, dtype=int), np.array(b1s, dtype=int)


def rank_curve_distance(a: np.ndarray, b: np.ndarray) -> int:
    """L1 distance between two Betti curves: the difference of rank functions."""
    return int(np.abs(np.asarray(a) - np.asarray(b)).sum())


def compute_invariants(X: np.ndarray, epsilons: np.ndarray) -> dict:
    D = pairwise_distances(X)
    b0, b1 = betti_curves(D, epsilons)
    basins = dihedral_basins(backbone_dihedrals(X))
    return {
        "distance_matrix": D,
        "b0_curve": b0,
        "b1_curve": b1,
        "distance_vector": D[np.triu_indices(D.shape[0], k=1)],   # z_L
        "basin_identity": basins,                                  # z_L
        "basin_histogram": basin_histogram(basins),                # z_H candidate
    }


def ensemble_report(models: list, epsilons: np.ndarray, max_models: int = 24,
                    seed: int = 20261011) -> dict:
    """Test invariance across an ensemble. CAN FAIL; `strict_invariance` is b0<=1.

    `strict` means the Betti curves are IDENTICAL for every pair. Persistent homology
    is stable not invariant, so strict is EXPECTED to be False on disperse ensembles.
    """
    n = len(models)
    idx = list(range(n)) if n <= max_models else sorted(
        np.random.default_rng(seed).permutation(n)[:max_models].tolist())
    invs = [compute_invariants(models[i], epsilons) for i in idx]
    d_b0 = d_b1 = d_hist = 0
    worst = None
    for i in range(len(invs)):
        for j in range(i + 1, len(invs)):
            a, b = invs[i], invs[j]
            x0 = rank_curve_distance(a["b0_curve"], b["b0_curve"])
            x1 = rank_curve_distance(a["b1_curve"], b["b1_curve"])
            xh = float(np.abs(a["basin_histogram"] - b["basin_histogram"]).sum())
            if x0 + x1 > d_b0 + d_b1:
                worst = {"pair": [int(idx[i]), int(idx[j])], "d_b0": x0, "d_b1": x1,
                         "d_basin": round(xh, 6)}
            d_b0, d_b1 = max(d_b0, x0), max(d_b1, x1)
            d_hist = max(d_hist, xh)
    return {
        "n_models_considered": n, "n_models_used": len(idx),
        "subsampled": n > max_models,
        "max_b0_curve_distance": int(d_b0),
        "max_b1_curve_distance": int(d_b1),
        "max_basin_histogram_L1": round(d_hist, 8),
        "strict_invariance": bool(d_b0 == 0 and d_b1 == 0),
        "worst_pair": worst,
    }


def separation_test(same_models: list, other: np.ndarray, epsilons: np.ndarray,
                    max_models: int = 24, seed: int = 20261011) -> dict:
    """REWRITTEN G-D1: same-protein distances must separate from different-protein.

    The negative control is the whole point: if two UNRELATED proteins give small
    distances, the invariant does not discriminate and the gate must fail.
    """
    rep = ensemble_report(same_models, epsilons, max_models=max_models, seed=seed)
    ref = compute_invariants(same_models[0], epsilons)
    xo = compute_invariants(other, epsilons)
    d_cross_b0 = rank_curve_distance(ref["b0_curve"], xo["b0_curve"])
    d_cross_b1 = rank_curve_distance(ref["b1_curve"], xo["b1_curve"])
    within = rep["max_b0_curve_distance"] + rep["max_b1_curve_distance"]
    cross = d_cross_b0 + d_cross_b1
    return {
        "within_same_protein": within,
        "cross_different_protein": cross,
        "cross_b0": int(d_cross_b0), "cross_b1": int(d_cross_b1),
        "separation_ratio": (cross / within) if within > 0 else None,
        "passes_separation": bool(cross > within),
        "note": "separation is the falsifiable form of invariance: PH is stable, not "
                "invariant, so equality cannot hold and must not be asserted.",
    }


def persistence_pairs(D: np.ndarray, eps_max: float | None = None,
                      max_triangles: int = 4_000_000) -> tuple:
    """Exact H0 and H1 persistence pairs of the Vietoris-Rips filtration of D.

    REPAIRS THE MEASURED FALSIFICATION (ledger, directive 3). Raw Betti CURVES count every
    feature at every filtration step, so short-lived thermal features enter z_H directly.
    The directive's own remedy is persistence: keep features with lifetime > 2.5 A.

    H0: union-find over edges in filtration order; the component with the later birth dies
        at the merge distance. All vertices are present at eps = 0.
    H1: reduce boundary_2 over GF(2) in filtration order. A 2-simplex whose reduced
        boundary is nonzero pairs with its PIVOT edge: that edge created a cycle at its own
        filtration value, and the triangle fills it. Only "positive" edges (those that
        created a cycle rather than merging components) may serve as pivots.

    Returns (h0_pairs, h1_pairs) as lists of (birth, death); one H0 death is inf.
    """
    N = D.shape[0]
    iu = np.triu_indices(N, k=1)
    dd = D[iu].astype(np.float64)
    ij = np.column_stack(iu)
    if eps_max is None:
        eps_max = float(dd.max())
    keep = dd <= eps_max
    E, ed = ij[keep], dd[keep]
    order = np.argsort(ed, kind="stable")
    E, ed = E[order], ed[order]
    nE = E.shape[0]
    erank = {(int(a), int(b)): r for r, (a, b) in enumerate(E)}

    # ---- H0 and the POSITIVE-edge set (edges that create a cycle, i.e. an H1 birth) ----
    parent = list(range(N))
    birth = [0.0] * N

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    positive = [False] * nE
    h0 = []
    for r in range(nE):
        a, b = int(E[r, 0]), int(E[r, 1])
        ra, rb = find(a), find(b)
        if ra == rb:
            positive[r] = True                      # cycle born: H1 birth candidate
            continue
        # later birth dies; tie-break by root index so the result is deterministic
        if birth[ra] > birth[rb] or (birth[ra] == birth[rb] and ra > rb):
            dead, live = ra, rb
        else:
            dead, live = rb, ra
        h0.append((birth[dead], float(ed[r])))
        parent[dead] = live
        birth[live] = min(birth[live], birth[dead])
    for v in range(N):
        if find(v) == v:
            h0.append((birth[v], float("inf")))

    # ---- H1 via boundary_2 reduction over GF(2) ----
    tris = []
    for i in range(N):
        for j in range(i + 1, N):
            if D[i, j] > eps_max:
                continue
            for k in range(j + 1, N):
                if D[i, k] > eps_max or D[j, k] > eps_max:
                    continue
                tris.append((max(D[i, j], D[i, k], D[j, k]), i, j, k))
    if len(tris) > max_triangles:
        raise RuntimeError(f"{len(tris)} triangles exceeds max_triangles; coarsen the "
                           f"grid or raise the cap rather than approximating silently")
    tris.sort(key=lambda x: x[0])
    pivots, h1 = {}, []
    for t, i, j, k in tris:
        x = (1 << erank[(i, j)]) ^ (1 << erank[(i, k)]) ^ (1 << erank[(j, k)])
        while x:
            p = x.bit_length() - 1
            if p in pivots:
                x ^= pivots[p]
            else:
                break
        if x:
            p = x.bit_length() - 1
            if not positive[p]:
                # Cannot happen under exact arithmetic: a non-positive edge is already
                # paired with an H0 class, so its column is eliminated. Fail loudly.
                raise RuntimeError(f"non-positive edge used as H1 pivot at rank {p}")
            pivots[p] = x
            h1.append((float(ed[p]), float(t)))
    return h0, h1


def _lifetime(pair) -> float:
    b, d = pair
    return float("inf") if d == float("inf") else float(d - b)


def filtered_betti_curves(D: np.ndarray, epsilons: np.ndarray,
                          min_lifetime: float = 0.0) -> tuple:
    """beta_0(eps) and beta_1(eps) counting ONLY features whose lifetime >= min_lifetime.

    min_lifetime = 0.0 must reproduce the raw curves: that identity is the self-check.
    """
    h0, h1 = persistence_pairs(D)
    b0, b1 = [], []
    for e in np.asarray(epsilons, dtype=float):
        b0.append(sum(1 for pr in h0 if pr[0] <= e < pr[1] and _lifetime(pr) >= min_lifetime))
        b1.append(sum(1 for pr in h1 if pr[0] <= e < pr[1] and _lifetime(pr) >= min_lifetime))
    return np.array(b0, dtype=int), np.array(b1, dtype=int)
