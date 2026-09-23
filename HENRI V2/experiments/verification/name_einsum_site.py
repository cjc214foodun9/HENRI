#!/usr/bin/env python3
"""Name the EXACT file:line of the einsum that fails after the resolution fix.

MEASURED so far: enabling HENRI_MACRO_NUM_CHANNELS=1 makes the Sagnac width mismatch
disappear (the macro field becomes 512-wide and MATCHES the refs) but 64/64 steps then
raise:
    RuntimeError: einsum(): subscript n has size 8192 for operand 1 which does not
                  broadcast with previously seen size 64

CANDIDATE SITES (found by substring search, so NONE is yet confirmed):
    opine_object_mcts.py:108         u_pred = einsum("nij,njk->nik", u_macro, u_t)
    henri_external_outcome_refactor_module.py:61  U_hat = einsum("nij,njk->nik", displacement, U_t)
    henri_external_outcome_refactor_module.py:72  delta_U = einsum("nij,nkj->nik", U_next, U_t.conj())

WHY INSTRUMENT INSTEAD OF READING
    Reading shows three plausible sites; only execution shows which one fires and what
    the two operands' channel counts are. This wraps torch.einsum, runs the gauntlet,
    and prints the caller's file:line plus operand shapes on the FIRST mismatch. It also
    records every OPINEObjectMCTS construction so the resolution at failure is known.
"""
import os
import sys
import traceback

WT = r"C:\Users\chan\Desktop\HENRI 7B SWARM\.worktrees\semantic-backbone\HENRI V2"
os.chdir(WT)
sys.path.insert(0, WT)

_orig_einsum = __import__("torch").einsum
_mismatches = []


def _wrapped(eq, *ops, **kw):
    try:
        return _orig_einsum(eq, *ops, **kw)
    except RuntimeError as e:
        msg = str(e)
        if "does not broadcast" in msg or "subscript" in msg:
            stack = traceback.extract_stack()
            # first frame that lives in the project (not torch internals)
            here = None
            for fr in reversed(stack):
                if WT.replace("\\", "/") in fr.filename.replace("\\", "/"):
                    here = fr
                    break
            shapes = [tuple(o.shape) for o in ops if hasattr(o, "shape")]
            dtypes = [str(getattr(o, "dtype", "?")) for o in ops]
            rec = {"eq": eq, "shapes": shapes, "dtypes": dtypes,
                   "file": (here.filename if here else "?"),
                   "line": (here.lineno if here else -1),
                   "func": (here.name if here else "?")}
            _mismatches.append(rec)
            if len(_mismatches) == 1:
                print("EINSUM_MISMATCH " + repr(rec), flush=True)
        raise


import torch  # noqa: E402
torch.einsum = _wrapped

# record every OPINE construction (resolution at failure)
import opine_object_mcts as _om  # noqa: E402
_orig_init = _om.OPINEObjectMCTS.__init__
_seen = []


def _init(self, num_channels=8192, option_horizon=4):
    _seen.append(num_channels)
    print("OPINE_CTOR num_channels=%s" % num_channels, flush=True)
    return _orig_init(self, num_channels=num_channels, option_horizon=option_horizon)


_om.OPINEObjectMCTS.__init__ = _init

os.environ.setdefault("OPERATION_MODE", "offline")
os.environ.setdefault("ARC_API_KEY", "")
os.environ.setdefault("ENVIRONMENTS_DIR",
                      r"C:\Users\chan\Desktop\HENRI 7B SWARM\environment_files")
os.environ["HENRI_ARC_SAGNAC_VETO"] = "1"
os.environ["HENRI_MACRO_NUM_CHANNELS"] = "1"   # the fix under test

import runpy  # noqa: E402
sys.argv = ["production_arc_run.py", "--mode", "phase823_live_gauntlet",
            "--steps", "40"]
try:
    runpy.run_path(os.path.join(WT, "production_arc_run.py"), run_name="__main__")
except SystemExit:
    pass
except Exception as e:  # noqa: BLE001
    print("OUTER %s: %s" % (type(e).__name__, e), flush=True)

print("=" * 72)
print("OPINE constructions: %s" % _seen[:8])
print("distinct mismatches: %d" % len(_mismatches))
seen_sites = {}
for m in _mismatches:
    k = "%s:%s %s" % (os.path.basename(m["file"]), m["line"], m["func"])
    seen_sites.setdefault(k, m)
for k, m in list(seen_sites.items())[:6]:
    print("  SITE %s" % k)
    print("       eq=%s shapes=%s dtypes=%s" % (m["eq"], m["shapes"], m["dtypes"]))
