"""Governance test: every FALSIFIED repair stays DEFAULT-OFF.

Measured this session in henri_heterosynaptic.py:
    ridge E_p closure       -> |Vbar[0]| 10.475 -> 0.034, |abar^1| -> 0.0000
    norm-preserving closure -> |Vbar[0]| -> 0.250,   |abar^1| -> 0.0000
    mirror Wbar rule        -> |Vbar[0]| -> 6.4e+10, E_p -> 3.7e+05 (exploded)
All three quench or explode the interstream stream. They must never become the default.
"""
import inspect
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "henri_core"))
from henri_heterosynaptic import DualStreamNetwork  # noqa: E402

DIMS = (8, 12, 8)


def test_ep_closure_defaults_off():
    sig = inspect.signature(DualStreamNetwork.__init__)
    assert sig.parameters["ep_closure"].default is False


def test_wbar_rule_defaults_to_none():
    sig = inspect.signature(DualStreamNetwork.__init__)
    assert sig.parameters["wbar_rule"].default == "none"


def test_default_construction_leaves_wbar_untouched_by_closures():
    net = DualStreamNetwork(DIMS, seed=0)
    assert net.ep_closure is False and net.wbar_rule == "none"


def test_falsified_modes_are_still_selectable_for_reproduction():
    """The negatives must stay reproducible, not be deleted."""
    net = DualStreamNetwork(DIMS, seed=0, ep_closure=True, ep_mode="ridge")
    assert net.ep_closure is True and net.ep_mode == "ridge"
    assert hasattr(net, "_wbar_mirror") and hasattr(net, "_ep_closure")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    bad = 0
    for f in fns:
        try:
            f()
            print(f"PASS {f.__name__}")
        except AssertionError as e:
            bad += 1
            print(f"FAIL {f.__name__}: {e}")
    print(f"{len(fns) - bad}/{len(fns)} passed")
    sys.exit(1 if bad else 0)
