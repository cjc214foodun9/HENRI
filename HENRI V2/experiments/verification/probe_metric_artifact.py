"""Is my own "0.748 FALSIFIES 1/sqrt(D)" claim a METRIC ARTIFACT?

THE SUSPICION (mine, about my own work)
    My audit register says the catalogue's "~1/sqrt(D) ~ 0.0039" is FALSIFIED
    because I measured cosine ~0.748 for distinct pairs.

    But the codec returns a uint8 ring in [0,255]. Its mean is ~127.5, i.e. a huge
    positive DC component. Cosine between two vectors that are BOTH dominated by the
    same large positive constant is high (~0.75) REGARDLESS of their phase structure.
    So 0.748 may be an artifact of the representation, not evidence about semantics.

    If true, my register's correction is itself a metric conflation, and the
    catalogue's 0.0039 (measured on a centered/real-valued view) is NOT contradicted.

DECISIVE TEST
    Compare, on the SAME encoded pairs:
      raw cosine        (DC-dominated)      -> expect ~0.75
      CENTERED cosine   (DC removed)        -> expect ~0 for distinct pairs if
                                               the codec is non-compositional,
                                               matching the catalogue's ~0.0039
      circular agreement (Z_256 aware)      -> expect ~3/256 = 0.0117 chance rate
    Plus: report the ring mean (demonstrates the DC carrier) and the theoretical
    chance rate for circular agreement within 1 step.

READ-ONLY.
"""
import importlib.util
import pathlib
import sys

R = pathlib.Path(
    r"C:\Users\chan\Desktop\HENRI 7B SWARM\.worktrees\basal-syncytium\HENRI V2")
sys.path.insert(0, str(R))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

hits = [p for p in sorted(R.glob("*.py"))
        if "class qFHRREpistemicCodec" in p.read_text(encoding="utf-8", errors="replace")]
spec = importlib.util.spec_from_file_location("m2", hits[0])
mod = importlib.util.module_from_spec(spec)
sys.modules["m2"] = mod
spec.loader.exec_module(mod)
codec = mod.qFHRREpistemicCodec(d_model=65536)
print("codec: " + hits[0].name)
print("")

PAIRS = [("cat", "cat"), ("cat", "dog"), ("cat", "quantum"),
         ("a+b", "b+a"), ("27", "28")]


def flat(x):
    return (x.detach().reshape(-1) if torch.is_tensor(x)
            else torch.as_tensor(x).reshape(-1))


def raw_cos(a, b):
    a, b = flat(a).to(torch.float32), flat(b).to(torch.float32)
    return float(F.cosine_similarity(a, b, dim=0))


def ctr_cos(a, b):
    """Cosine after removing each vector's own mean (kills the DC carrier)."""
    a, b = flat(a).to(torch.float32), flat(b).to(torch.float32)
    a = a - a.mean()
    b = b - b.mean()
    return float(F.cosine_similarity(a, b, dim=0))


def circ_agree(a, b):
    a, b = flat(a), flat(b)
    d = (a.to(torch.int64) - b.to(torch.int64)).abs() % 256
    d = torch.minimum(d, 256 - d)
    return float((d <= 1).to(torch.float32).mean())


print("ring mean (DC carrier) = "
      + format(float(flat(codec.encode_text('cat')).to(torch.float32).mean()), ".4f")
      + "   [uint8 ring, uniform 0..255 would be 127.5]")
print("circular-agreement chance rate within 1 step = 3/256 = "
      + format(3.0 / 256.0, ".6f"))
print("")
print("pair".ljust(22) + "raw_cos".rjust(10) + "centered".rjust(11) + "circ<=1".rjust(10))
print("-" * 53)
rows = []
for s, t in PAIRS:
    wa, wb = codec.encode_text(s), codec.encode_text(t)
    rc, cc, ca = raw_cos(wa, wb), ctr_cos(wa, wb), circ_agree(wa, wb)
    rows.append({"a": s, "b": t, "raw": rc, "ctr": cc, "ca": ca})
    print((repr(s) + " vs " + repr(t)).ljust(22)
          + format(rc, ".6f").rjust(10)
          + format(cc, ".6f").rjust(11)
          + format(ca, ".4f").rjust(10))

by = {(r["a"], r["b"]): r for r in rows}
ident, rel, unr = by[("cat", "cat")], by[("cat", "dog")], by[("cat", "quantum")]

print("")
print("=" * 70)
print("VERDICT")
print("=" * 70)
print("identity raw / centered : "
      + format(ident["raw"], ".6f") + " / " + format(ident["ctr"], ".6f"))
print("centered gap cat/dog vs cat/quantum: "
      + format(abs(rel["ctr"] - unr["ctr"]), ".6f"))
print("")

raw_high = rel["raw"] > 0.5 and unr["raw"] > 0.5
ctr_flat = abs(rel["ctr"]) < 0.02 and abs(unr["ctr"]) < 0.02
ctr_gap_small = abs(rel["ctr"] - unr["ctr"]) < 0.01
circ_at_chance = (abs(rel["ca"] - 3.0 / 256.0) < 0.004
                  and abs(unr["ca"] - 3.0 / 256.0) < 0.004)
ident_ok = ident["ctr"] > 0.99

if raw_high and ctr_flat and ctr_gap_small and ident_ok:
    print("VERDICT: MY_0.748_IS_A_DC_METRIC_ARTIFACT")
    print("  Raw cosine ~0.75 is produced by the uint8 ring's DC carrier (~127.5),")
    print("  NOT by semantic similarity. After centering, distinct pairs sit at ~0")
    print("  with no semantic gap, and circular agreement sits at the 3/256 chance")
    print("  rate. So the codec IS non-compositional (conclusion holds) but my")
    print("  register's 'the catalogue's 1/sqrt(D) is FALSIFIED' is WRONG: the")
    print("  catalogue number and mine measure DIFFERENT quantities. My correction")
    print("  must itself be retracted as a scale/metric conflation.")
elif ctr_at_chance is False and not ctr_gap_small:
    print("VERDICT: PARTIALLY_COMPOSITIONAL -- a real centered semantic gap exists;")
    print("  the register's 'no meaning' claim is too strong.")
else:
    print("VERDICT: INCONCLUSIVE -- inspect the table before concluding.")
print("")
print("circular at chance = " + str(circ_at_chance)
      + "   identity centered ok = " + str(ident_ok))
print("PROBE_DONE")
