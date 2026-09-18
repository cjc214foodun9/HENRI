"""Verify the LOAD-BEARING codec claim in the audit register.

CLAIM UNDER TEST (currently INHERITED from the architecture skill, MEASURED 2026-08-02)
    qFHRREpistemicCodec.encode_text is SHA-256-seeded torch.randint: deterministic
    (identical strings -> identical rings, sim 1.0) but NON-COMPOSITIONAL -- every
    distinct string maps to an independent random Z_256 ring, so similarity is about
    1/sqrt(D) for ALL distinct pairs. "cat" vs "dog" scores the same as
    "cat" vs "quantum".

WHY THIS MATTERS
    My audit register names "no word meaning in the wave space" as the single biggest
    gap for AAII SOTA. That conclusion rests entirely on this claim. Citing a skill
    line is not evidence; a probe is. This converts it to OBSERVED (or FALSIFIED).

METHOD
    - Locate the class definition by scanning repo-root .py files (no guessing the
      module name).
    - Encode a mix of SEMANTICALLY RELATED and UNRELATED pairs.
    - Identity pair is the positive control (must be ~1.0).
    - Report cosine similarity, plus the Z_256 circular-agreement fraction as a
      second, geometry-aware view.

READ-ONLY. No repo writes.
"""
import importlib.util
import pathlib
import sys

R = pathlib.Path(
    r"C:\Users\chan\Desktop\HENRI 7B SWARM\.worktrees\basal-syncytium\HENRI V2")
sys.path.insert(0, str(R))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

print("=" * 74)
print("CODEC COMPOSITIONALITY PROBE")
print("=" * 74)

hits = []
for p in sorted(R.glob("*.py")):
    try:
        t = p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        continue
    if "class qFHRREpistemicCodec" in t:
        hits.append(p)
print("class qFHRREpistemicCodec defined in: " + str([h.name for h in hits]))
if not hits:
    print("BLOCKED: class not found by scan")
    raise SystemExit(1)

spec = importlib.util.spec_from_file_location("codecprobe_mod", hits[0])
mod = importlib.util.module_from_spec(spec)
sys.modules["codecprobe_mod"] = mod
spec.loader.exec_module(mod)
Codec = getattr(mod, "qFHRREpistemicCodec")
print("loaded: " + hits[0].name)

D = 65536
try:
    codec = Codec(d_model=D)
except TypeError as exc:
    print("ctor kwargs differ: " + str(exc))
    codec = Codec()
    D = getattr(codec, "d_model", D)
print("d_model = " + str(D))
print("")

PAIRS = [
    ("cat", "cat"),
    ("cat", "dog"),
    ("cat", "quantum"),
    ("a+b", "b+a"),
    ("27", "28"),
]


def flat(x):
    if torch.is_tensor(x):
        return x.detach().reshape(-1)
    return torch.as_tensor(x).reshape(-1)


def cos(a, b):
    a, b = flat(a).to(torch.float32), flat(b).to(torch.float32)
    if a.numel() != b.numel():
        return None
    return float(F.cosine_similarity(a, b, dim=0))


def circ_agree(a, b):
    a, b = flat(a), flat(b)
    if a.numel() != b.numel():
        return None
    d = (a.to(torch.int64) - b.to(torch.int64)).abs() % 256
    d = torch.minimum(d, 256 - d)
    return float((d <= 1).to(torch.float32).mean())


print("pair".ljust(22) + "cos".rjust(12) + "circ<=1".rjust(10))
print("-" * 46)
rows = []
for s, t in PAIRS:
    try:
        wa = codec.encode_text(s)
        wb = codec.encode_text(t)
    except Exception as exc:  # noqa: BLE001
        print(repr(s).ljust(10) + " ERROR " + type(exc).__name__ + ": " + str(exc))
        continue
    c = cos(wa, wb)
    ca = circ_agree(wa, wb)
    rows.append({"a": s, "b": t, "cos": c, "ca": ca})
    label = repr(s) + " vs " + repr(t)
    print(label.ljust(22)
          + ("None" if c is None else format(c, ".6f")).rjust(12)
          + ("None" if ca is None else format(ca, ".4f")).rjust(10))

print("")
by = {(r["a"], r["b"]): r for r in rows}
ident = by.get(("cat", "cat"))
related = by.get(("cat", "dog"))
unrel = by.get(("cat", "quantum"))

if ident is None or ident["cos"] is None:
    print("BLOCKED: identity pair unavailable -> determinism not established")
    raise SystemExit(2)

print("identity cos            = " + format(ident["cos"], ".6f") + "   (determinism)")
if related and unrel and related["cos"] is not None and unrel["cos"] is not None:
    gap = related["cos"] - unrel["cos"]
    print("related  (cat,dog)      = " + format(related["cos"], ".6f"))
    print("unrelated(cat,quantum)  = " + format(unrel["cos"], ".6f"))
    print("semantic gap            = " + format(gap, "+.6f"))
    print("")
    if abs(gap) < 0.01 and related["cos"] > 0.9:
        print("VERDICT: FALSIFIED_DETERMINISM_ONLY -- the codec collapses ALL strings "
              "to the same ring; there is no distinct representation at all.")
    elif abs(gap) < 0.01:
        print("VERDICT: CONFIRMED_NON_COMPOSITIONAL -- 'cat' vs 'dog' is as similar as "
              "'cat' vs 'quantum' (gap below 0.01). The codec carries IDENTITY but not "
              "MEANING. The audit register's central gap claim becomes OBSERVED.")
    else:
        print("VERDICT: PARTIALLY_COMPOSITIONAL -- a measurable semantic gap exists "
              "(" + format(gap, "+.6f") + "). The audit's 'no word meaning' claim "
              "would be TOO STRONG and must be softened.")

print("")
print("PROBE_DONE")
