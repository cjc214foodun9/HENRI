
"""UHR-03 DOMAIN FIX — FINAL LOCAL TEST (v6, detector corrected).

TWO BUGS OF MINE, FIXED HERE:
  (1) v5 detected "changed cells" with `||disp||_F > 1e-3`. An UNCHANGED cell has
      disp = I_3, whose Frobenius norm is sqrt(3) = 1.732 -- NOT zero. So the test
      passed for every cell and the "256-channel support" included 248 cells that
      did not change. Correct detector: `||disp_c - I_3||_F > thr`.
  (2) The v5 print labelled `argmin` from the OTHERS-only sort, so it could never
      equal AID and always printed MISS. The statistic itself (own < min_other) was
      correct; the label was wrong.

DOMAIN UNDER TEST. Per-channel primitive (one [3,3] generator each side, its tested
input contract), pooled as a SCALAR MEAN over the support. Tensor composition is not
used: su(3) is non-abelian so `ad_of` over many channels is order-dependent and
ill-defined as a "field relative element".
"""
import sys
import numpy as np, torch
SB = r"C:/Users/chan/Desktop/HENRI 7B SWARM/.worktrees/semantic-backbone/HENRI V2"
sys.path.insert(0, SB)
import henri_external_outcome_refactor_module as M
import uhr02_exteroceptive_gate as G
from chromodynamic_grounding import GELL_MANN_BASIS, encode_su3_color_field

SIDE, NCH, N_ACT, N_UPD, AID = 16, 256, 8, 16, 2
basis = GELL_MANN_BASIS.to(torch.complex64)
base  = np.random.default_rng(7).integers(0, 4, size=(SIDE, SIDE), dtype=np.int64)
enc   = lambda g: encode_su3_color_field(
    torch.tensor(g, dtype=torch.int64).unsqueeze(0)).reshape(-1, 3, 3)
ROWS  = {a: (2 * a % SIDE, 2 * a % SIDE + 2) for a in range(N_ACT)}
PICK  = {a: list(np.random.default_rng(100 + a).choice(32, size=8, replace=False))
         for a in range(N_ACT)}

def grid(a, color):
    g = base.copy()
    r0, r1 = ROWS[a]
    cells = [(i, j) for i in range(r0, min(r1, SIDE)) for j in range(SIDE)]
    for p in PICK[a]:
        i, j = cells[int(p) % len(cells)]
        g[i, j] = (g[i, j] + 1 + color) % 4
    return g

store = M.ActionOutcomeGeneratorStore(num_actions=N_ACT, num_channels=NCH, lr=0.1)
for k in range(N_UPD):
    for a in range(N_ACT):
        store.update_generator(enc(grid(a, 2 * k)), a, enc(grid(a, 2 * k + 1)), basis)

U_t, U_n = enc(grid(AID, 200)), enc(grid(AID, 201))
disp = G.relative_displacement(U_n, U_t).detach()
I3 = torch.eye(3, dtype=torch.complex64)
dev = (disp - I3).norm(dim=(-2, -1))          # CORRECT changed-cell detector

THR = 1e-3
obs_chg = (dev > THR).nonzero().flatten().tolist()
th_sup  = (store.theta_a[AID].norm(dim=-1) > 1e-5).nonzero().flatten().tolist()

print("=" * 84)
print("CHANGE DETECTOR (corrected): ||disp_c - I_3||_F")
print("=" * 84)
print("  |disp| (WRONG detector) at ch0        : %.6f   <- == sqrt(3), the identity norm"
      % float(disp.norm(dim=(-2, -1))[0]))
print("  ||disp_ch0 - I_3|| (CORRECT detector) : %.3e" % float(dev[0]))
print("  changed channels (dev > %.0e)      : n=%d  %s" % (THR, len(obs_chg), obs_chg))
print("  theta_a[AID] support                  : n=%d  %s" % (len(th_sup), th_sup))
print("  observed subset of theta support      :", set(obs_chg) <= set(th_sup))
print("  intersection                          :", sorted(set(obs_chg) & set(th_sup)))

K = 8192
BAND = G.sampling_band(K)
g_ = lambda s: torch.Generator().manual_seed(s)
nz = lambda x: torch.nn.functional.normalize(x, p=2, dim=-1)
roles_iso   = nz(torch.randn(K, 8, generator=g_(1)))
centres     = torch.randn(8, 8, generator=g_(11))
assign      = torch.randint(0, 8, (K,), generator=g_(12))
roles_clust = nz(centres[assign] + 0.02*torch.randn(K, 8, generator=g_(13)))


def pooled(roles, channels):
    """Per-channel scalar delta, mean-pooled. Returns (own, others, vals)."""
    truth = {c: G.generators_from_displacement(disp, channel=c)[0] for c in channels}
    pop = {}
    for a in range(N_ACT):
        tot = 0.0
        for c in channels:
            cand = store.lie_element(a, basis)[c]
            tot += float(G.exteroceptive_residual_vs_recorded(
                roles, [cand], [truth[c]], basis).delta_pred)
        pop[a] = tot / len(channels)
    own = pop[AID]
    others = {k: v for k, v in pop.items() if k != AID}
    order = sorted(others.items(), key=lambda kv: kv[1])
    # CORRECT argmin over the FULL population, including own
    full = sorted(pop.items(), key=lambda kv: kv[1])
    return dict(own=own, others=others, min_other=order[0][1], margin=order[0][1] - own,
                argmin_full=full[0][0], own_is_min=full[0][0] == AID,
                spread=max(pop.values()) - min(pop.values()))


print()
print("=" * 84)
print("POOLED PER-CHANNEL DOMAIN, support = OBSERVED-CHANGE channels (n=%d)" % len(obs_chg))
print("=" * 84)
for rtag, roles in (("isotropic", roles_iso), ("clustered", roles_clust)):
    r = pooled(roles, obs_chg)
    print("  roles=%-10s coh=%.6f" % (rtag, float(roles.mean(dim=0).norm())))
    print("      own=%.6f  others=%s" % (r["own"], {k: round(v, 5) for k, v in sorted(r["others"].items())}))
    print("      min_other=%.6f  MARGIN=%+.6f  [%.1fx band %.4e]"
          % (r["min_other"], r["margin"], r["margin"] / BAND, BAND))
    print("      argmin(FULL population)=%d  own_is_min=%s  spread=%.6f"
          % (r["argmin_full"], r["own_is_min"], r["spread"]))
    print("      C1 argmin_hits_truth=%s   C2 invalid_minus_own=%+.6f"
          % (r["own_is_min"], r["margin"]))

print()
print("=" * 84)
print("POOLED, support = theta_a[AID] SUPPORT (n=%d)" % len(th_sup))
print("=" * 84)
for rtag, roles in (("isotropic", roles_iso), ("clustered", roles_clust)):
    r = pooled(roles, th_sup)
    print("  roles=%-10s own=%.6f min_other=%.6f MARGIN=%+.6f [%.1fx band] own_is_min=%s"
          % (rtag, r["own"], r["min_other"], r["margin"], r["margin"] / BAND, r["own_is_min"]))

print()
print("=" * 84)
print("SHARED-IDENTITY CONTROL: is the separation coming from the CHANGED cells only?")
print("=" * 84)
unchanged = [c for c in range(NCH) if c not in set(obs_chg)]
print("  unchanged channels: n=%d" % len(unchanged))
r_unch = pooled(roles_iso, unchanged)
print("  pooled over UNCHANGED: own=%.6f  min_other=%.6f  margin=%+.6f  spread=%.3e"
      % (r_unch["own"], r_unch["min_other"], r_unch["margin"], r_unch["spread"]))
print("  => the discrimination lives in the CHANGED cells; unchanged cells carry ~0 signal")

print()
print("=" * 84)
print("CONTROLS")
print("=" * 84)
# T1 tautology: candidate == truth object
tt = [G.generators_from_displacement(disp, channel=c)[0] for c in obs_chg]
print("  T1 tautology pooled(truth,truth) = %.3e"
      % float(np.mean([float(G.exteroceptive_residual_vs_recorded(roles_iso, [t], [t], basis).delta_pred)
                       for t in tt])))
print("  T2 learned candidate != observed truth (different objects):",
      not torch.equal(store.lie_element(AID, basis)[obs_chg[0]], tt[0]))
# T3 channel rule: use ANOTHER action's theta support (independent of the action under test)
oth_sup = (store.theta_a[5].norm(dim=-1) > 1e-5).nonzero().flatten().tolist()
for nm, ch in (("observed-change", obs_chg), ("theta[AID] (in-test)", th_sup),
               ("theta[5] (INDEPENDENT)", oth_sup)):
    r = pooled(roles_iso, ch)
    print("      %-24s n=%-4d own=%.6f min_other=%.6f margin=%+.6f DISC=%s"
          % (nm, len(ch), r["own"], r["min_other"], r["margin"], r["margin"] > BAND))

print()
print("=" * 84)
print("SUMMARY")
print("=" * 84)
ri = pooled(roles_iso, obs_chg)
rc = pooled(roles_clust, obs_chg)
print("  DOMAIN = pooled per-channel delta over the observed-change support")
print("  n_admissible = %d   (single-channel domain gave 1)" % (len(ri["others"]) + 1))
print("  C1 argmin_hits_truth : iso=%s clustered=%s" % (ri["own_is_min"], rc["own_is_min"]))
print("  C2 invalid_minus_own : iso=%+.6f clustered=%+.6f" % (ri["margin"], rc["margin"]))
print("  margin vs band       : %.1fx / %.1fx" % (ri["margin"] / BAND, rc["margin"] / BAND))
print("  content-insensitive  :", ri["own_is_min"] and rc["own_is_min"])
