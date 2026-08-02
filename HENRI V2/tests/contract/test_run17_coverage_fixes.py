"""Run17 post-verdict contract tests.

Pin the two defects the coverage audit exposed (2026-08-02):
1. Arity dispatch: a 3-arg signature must emit 3-arg bodies. The old
   `if len(args) >= 2:` branch silently collapsed 3/4/5-arg items into
   2-arg candidates (TypeError in the sandbox for every such item).
2. canonical_key rename invariance: descriptive canonical arg names
   (l, b, h, M, words) must be renamed to the aN convention even when
   parse_entry_signature has already renamed the item signature.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from mbpp_rank_probe import canonical_key, canonical_signature  # noqa: E402
from mbpp_wave_ast_decoder import WaveASTDecoder  # noqa: E402

CANON_VOLUME = (
    "def find_Volume(l, b, h):\n"
    "    '''Return the volume of a cuboid.'''\n"
    "    return ((l * b * h) / 2)\n"
)


def test_arity3_dispatch_emits_3arg_bodies():
    dec = WaveASTDecoder(None)
    bodies = dec._instantiate("find_Volume", ["a0", "a1", "a2"])
    assert bodies, "3-arg signature must emit candidates"
    src = f"def find_Volume(a0, a1, a2):\n{bodies[0]}"
    tree = __import__("ast").parse(src)
    fn = tree.body[0]
    assert len(fn.args.args) == 3, "candidate must keep 3-arg arity"
    assert any("a2" in b for b in bodies), "third arg must be used"


def test_arity3_matches_canonical_volume():
    dec = WaveASTDecoder(None)
    key = canonical_key(CANON_VOLUME, "find_Volume")
    assert key is not None
    for b in dec._instantiate("find_Volume", ["a0", "a1", "a2"]):
        src = f"def find_Volume(a0, a1, a2):\n{b}"
        if canonical_key(src, "find_Volume") == key:
            return
    pytest.fail("canonical find_Volume body must be expressible")


def test_canonical_key_renames_descriptive_args():
    """The old key builder took already-renamed args from the signature
    parser; the rename map became identity and descriptive names (l,b,h)
    were never renamed -> false COVERAGE_MISS."""
    key = canonical_key(CANON_VOLUME, "find_Volume")
    assert "l'" not in str(key).replace("Load", ""), "original names must be renamed"
    assert "a0" in key and "a2" in key


def test_arity1_and_2_unchanged():
    dec = WaveASTDecoder(None)
    one = dec._instantiate("f", ["a0"])
    two = dec._instantiate("f", ["a0", "a1"])
    assert one and two
    assert all("a1" not in b for b in one)
    assert any("a0" in b and "a1" in b for b in two)
