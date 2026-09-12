"""Stage-2 flag wiring verification, done against the ORIGINAL code from git.

The first attempt at this test failed and the failure was ITS OWN bug: it
compared each engine against a hand-retyped `ref_mean` that stopped at
`normalize * 64`. But g1/f22/f23 `_bridge_to_d64_single` end with a SECOND
`F.normalize(...)` after the ingress branch, so they return a unit-norm vector,
while f15's `_bridge_to_d64` has no ingress branch and returns the 64-scaled
vector. The retyped reference matched neither. "FLAG OFF not identical" and
"engines disagree" were both artifacts of that.

This version takes the reference from git (`HEAD:<path>`), i.e. the actual
pre-patch source, so the comparison is a true A/B test.

Checks:
  1. patch present: original has the mean-pool line; patched has the flag branch
  2. FLAG OFF (patched) == ORIGINAL, exactly, including the padding path
  3. FLAG ON  (patched) != ORIGINAL  (the bridge actually changes)
  4. shape stays [4096] in both states
  5. cross-engine comparison is done SCALE-INVARIANTLY (unit-normalize first),
     because f15 vs g1 scale divergence is PRE-EXISTING, not caused by the patch

CPU only.
"""
import importlib.util
import os
import subprocess
import sys
import tempfile

import torch

REPO = r"C:\Users\chan\Desktop\HENRI 7B SWARM"
REL = "HENRI V2/experiments/verification"
VER = os.path.join(REPO, "HENRI V2", "experiments", "verification")
sys.path.insert(0, VER)

ENGINES = [
    ("arc_g1_topological_engine", "_bridge_to_d64_single", "single"),
    ("arc_f15_trajectory_engine", "_bridge_to_d64", "f15"),
    ("arc_f22_resolution_engine", "_bridge_to_d64_single", "single"),
    ("arc_f23_causal_engine", "_bridge_to_d64_single", "single"),
]


def git_original(name):
    """Fetch the pre-patch file from git and write it to a temp dir."""
    r = subprocess.run(
        ["git", "-C", REPO, "show", f"HEAD:{REL}/{name}.py"],
        capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"git show failed for {name}: {r.stderr[:200]}")
    return r.stdout


def load_from_text(text, modname, tmpdir, filename):
    path = os.path.join(tmpdir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


def load_patched(name):
    path = os.path.join(VER, name + ".py")
    spec = importlib.util.spec_from_file_location("_p_" + name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_p_" + name] = mod
    spec.loader.exec_module(mod)
    return mod


def call(mod, fname, kind, w):
    fn = getattr(mod, fname)
    return fn(w) if kind == "single" else fn(w, "cpu")


def main():
    tmp = tempfile.mkdtemp(prefix="stage2_verify_")
    g = torch.Generator().manual_seed(7)
    waves = [
        ("full-65536", torch.randn(65536, generator=g)),
        ("short-1000", torch.randn(1000, generator=g)),   # exercises padding
        ("long-70000", torch.randn(70000, generator=g)),  # exercises truncation
    ]

    print("=" * 78)
    print("STAGE 2 WIRING VERIFICATION vs ORIGINAL CODE FROM GIT")
    print("=" * 78)
    print(f"original source: HEAD:{REL}/<engine>.py")

    # ---- 1. patch presence in source text ----
    print("\n[1] patch presence in source")
    patch_ok = True
    for name, fname, kind in ENGINES:
        orig = git_original(name)
        cur = open(os.path.join(VER, name + ".py"), encoding="utf-8").read()
        has_orig_mean = "w.view(16, 4096).mean(dim=0)" in orig
        has_orig_flag = "HENRI_LOCAL_CLIFFORD_BRIDGE" in orig
        has_cur_flag = "HENRI_LOCAL_CLIFFORD_BRIDGE" in cur
        # original had the mean line and no flag; patched has the flag
        good = has_orig_mean and not has_orig_flag and has_cur_flag
        patch_ok &= good
        print(f"    {name:<32} orig_mean={has_orig_mean} orig_flag={has_orig_flag} "
              f"cur_flag={has_cur_flag}  ok={good}")

    # ---- 2/3. flag OFF == original ; flag ON != original ----
    os.environ.pop("HENRI_LOCAL_CLIFFORD_BRIDGE", None)
    print("\n[2] FLAG OFF: patched must equal ORIGINAL exactly (all wave sizes)")
    off_ok = True
    for name, fname, kind in ENGINES:
        orig = load_from_text(git_original(name), "_o_" + name, tmp, "orig_" + name + ".py")
        cur = load_patched(name)
        for label, w in waves:
            a = call(orig, fname, kind, w)
            b = call(cur, fname, kind, w)
            ident = bool(torch.equal(a, b))
            maxd = float((a - b).abs().max())
            off_ok &= ident
            print(f"    {name:<32} {label:<10} identical={ident} "
                  f"max_abs_diff={maxd:.3e}")

    os.environ["HENRI_LOCAL_CLIFFORD_BRIDGE"] = "1"
    print("\n[3] FLAG ON: patched must DIFFER from original, shape [4096]")
    on_ok = True
    for name, fname, kind in ENGINES:
        for m in list(sys.modules):
            if m == "local_clifford_bridge":
                del sys.modules[m]
        orig = load_from_text(git_original(name), "_o2_" + name, tmp, "orig2_" + name + ".py")
        cur = load_patched(name)
        for label, w in waves:
            a = call(orig, fname, kind, w)
            b = call(cur, fname, kind, w)
            changed = not bool(torch.equal(a, b))
            shape_ok = tuple(b.shape) == (4096,)
            on_ok &= changed and shape_ok
            print(f"    {name:<32} {label:<10} changed={changed} "
                  f"shape_ok={shape_ok} norm={float(b.norm()):.4f}")
    os.environ.pop("HENRI_LOCAL_CLIFFORD_BRIDGE", None)

    # ---- 4. cross-engine, scale-invariant ----
    print("\n[4] cross-engine agreement, FLAG ON, compared after unit-normalize")
    print("     (f15 returns *64 while g1/f22/f23 return unit -- PRE-EXISTING)")
    import torch.nn.functional as F
    os.environ["HENRI_LOCAL_CLIFFORD_BRIDGE"] = "1"
    outs = {}
    for name, fname, kind in ENGINES:
        for m in list(sys.modules):
            if m == "local_clifford_bridge":
                del sys.modules[m]
        cur = load_patched(name)
        v = call(cur, fname, kind, waves[0][1])
        outs[name] = F.normalize(v, p=2, dim=-1)
        print(f"    {name:<32} raw_norm={float(call(cur, fname, kind, waves[0][1]).norm()):.4f}")
    base = outs["arc_g1_topological_engine"]
    agree = True
    for k, v in outs.items():
        d = float((v - base).abs().max())
        agree &= d < 1e-6
        print(f"    {k:<32} max_abs_diff(unit)={d:.3e}")
    os.environ.pop("HENRI_LOCAL_CLIFFORD_BRIDGE", None)

    print("\n" + "-" * 78)
    print(f"PATCH_PRESENT          = {patch_ok}")
    print(f"FLAG_OFF_EQUALS_ORIG   = {off_ok}")
    print(f"FLAG_ON_CHANGES_OUTPUT = {on_ok}")
    print(f"ENGINES_AGREE_UPSCALE  = {agree}")
    ok = patch_ok and off_ok and on_ok and agree
    print(f"VERDICT = {'WIRING_VALIDATED' if ok else 'WIRING_FAILED'}")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
