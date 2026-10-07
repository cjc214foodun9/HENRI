"""Self-check for the D1-D5 hygiene contracts. Each check CAN fail.

Includes the negative controls that demonstrate the defect the contract blocks.
"""
import sys

sys.path.insert(0, ".")
import torch
from henri_core import harness_hygiene as H

PASS, FAIL = [], []


def ck(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")


print("D2 derangement")
perms = [H.derangement(10, torch.Generator().manual_seed(s)) for s in range(200)]
ck("D2 zero fixed points over 200 draws",
   all(all(p[i] != i for i in range(len(p))) for p in perms))
ck("D2 not all identical (real randomness)",
   len({tuple(p) for p in perms}) > 50, f"distinct={len({tuple(p) for p in perms})}")
ck("D2 n=2 -> [1,0]", H.derangement(2, torch.Generator().manual_seed(0)) == [1, 0])
try:
    H.derangement(1)
    ck("D2 n=1 raises", False)
except ValueError:
    ck("D2 n=1 raises", True)
# NEGATIVE CONTROL: the naive call the contract replaces
g = torch.Generator().manual_seed(7)
naive_ids = sum(1 for _ in range(2000)
                if (lambda p: any(p[i] == i for i in range(4)))(
                    torch.randperm(4, generator=g).tolist()))
ck("D2-nc naive randperm DOES produce identity draws", naive_ids > 0,
   f"{naive_ids}/2000 draws had a fixed point")

print("D3 split manifest")
man = H.split_manifest(["a", "b", "c"], ["d", "e"])
ck("D3 disjoint passes", man["intersection_cardinality"] == 0)
ck("D3 hashes are 64-hex", all(len(h) == 64 for h in man["bank_sha256"]))
try:
    H.split_manifest(["a", "b"], ["b", "c"])
    ck("D3 overlap raises", False)
except AssertionError:
    ck("D3 overlap raises", True)

print("D4 pinned generator")
a = [torch.randn(4, generator=H.pinned_generator(5)).tolist() for _ in range(2)]
ck("D4 same seed -> identical", a[0] == a[1])
b = torch.randn(4, generator=H.pinned_generator(6)).tolist()
ck("D4 different seed -> different", b != a[0])

print("D5 margin contract")
neg = [0.20, 0.22, 0.25, 0.28, 0.30]
pos = [1.0, 1.0, 1.0]
m = H.margin_contract(pos, neg)
ck("D5 separable passes", m["passes"], f"ratio={m['ratio']:.3f} scale={m['scale']}")
ck("D5 sigma_bank would be 0", m["neg_sigma"] > 0 and m["scale_value"] > 0)
# NEGATIVE CONTROL: sigma_bank on the POSITIVE population is exactly 0
sigma_bank = torch.tensor(pos, dtype=torch.float64).std(unbiased=True)
ck("D5-nc sigma_bank == 0 -> blueprint formula would be NaN",
   float(sigma_bank) == 0.0, f"sigma_bank={float(sigma_bank)}")
deg = H.margin_contract([0.5, 0.5], [0.5, 0.5])
ck("D5 degenerate -> passes=False, not NaN-pass",
   deg["passes"] is False and deg["scale"] == "DEGENERATE_ZERO_SPREAD")
thin = H.margin_contract([0.80], [0.79, 0.78])
ck("D5 thin gap fails k=3", thin["passes"] is False, f"ratio={thin['ratio']:.2f}")

print("D1 matched negatives")
fam = {"F0": {"kind": "in_bank"}, "F1": {"kind": "drop_boundary"}}
r = H.assert_matched_negatives(fam)
ck("D1 matched present", r["matched_present"] == ["drop_boundary"])
try:
    H.assert_matched_negatives({"F0": {"kind": "in_bank"},
                                "F5": {"kind": "orthogonal_noise"}})
    ck("D1 orthogonal-only raises", False)
except AssertionError:
    ck("D1 orthogonal-only raises", True)

print(f"\n==== {len(PASS)} passed, {len(FAIL)} failed ====")
if FAIL:
    print("FAILED:", FAIL)
sys.exit(1 if FAIL else 0)
