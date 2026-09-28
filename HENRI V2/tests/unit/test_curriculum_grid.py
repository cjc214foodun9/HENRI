"""Guards for henri_curriculum_grid.py (Directive 2: 2-D spatial emitter).

The module exists because the 1-D byte tape CANNOT emit a 2-D task: measured, `grid` /
`2d` / `spatial` / `jordan` / `interior` occur 0 times in henri_curriculum_env.py code.
The semantic rungs were therefore BLOCKED__NO_EMITTER, recorded in the committed receipt.

THE CONTROL THAT MATTERS (this project has been burned by its absence): the target must be
computable from the GENERATOR'S OWN BOOKKEEPING, never by calling the operator under test.
A fixture that generates the object its solver searches for scores 1.0 by construction.
"""
import os
import sys


def _root():
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        if os.path.exists(os.path.join(d, "henri_curriculum_grid.py")):
            return d
        d = os.path.dirname(d)
    return os.path.dirname(os.path.abspath(__file__))


C = _root()
if C not in sys.path:
    sys.path.insert(0, C)

import henri_curriculum_grid as CG   # noqa: E402


def test_all_declared_families_build_a_task():
    for fam in CG.FAMILIES:
        tasks = CG.make_batch(fam, 12, seed=20260927)
        assert len(tasks) == 12
        for t in tasks:
            assert t["family"] == fam
            assert len(t["input"]) == len(t["input"][0]), "grid must be square"


def test_no_family_produces_an_identity_task():
    """An identity target would let a trivial no-op pass.

    STRENGTHENED after a measured defect: the reflection family emitted Y == X for a
    square block centred on the rotation axis (measured: h=7, w=9, r0=2, c0=1, rot=2).
    One seed is not a test, so this sweeps many seeds.
    """
    for fam in CG.FAMILIES:
        for seed in range(1, 41):
            for t in CG.make_batch(fam, 3, seed=seed):
                assert t["changed_cells"] > 0, (
                    "%s produced an identity task at seed %d" % (fam, seed))


def test_determinism_same_seed_same_tasks():
    a = CG.make_batch("containment_fill", 6, seed=99)
    b = CG.make_batch("containment_fill", 6, seed=99)
    assert [t["input"] for t in a] == [t["input"] for t in b]
    assert [t["target"] for t in a] == [t["target"] for t in b]


def test_different_seeds_differ():
    a = CG.make_batch("containment_fill", 6, seed=1)
    b = CG.make_batch("containment_fill", 6, seed=2)
    assert [t["input"] for t in a] != [t["input"] for t in b]


def test_containment_fill_changes_EXACTLY_the_enclosed_region():
    """The strongest structural check: the changed set must equal the Jordan interior."""
    from henri_region_selector import enclosed_regions
    checked = 0
    for t in CG.make_batch("containment_fill", 20, seed=314):
        grid, tgt = t["input"], t["target"]
        regions = enclosed_regions(grid, CG.BACKGROUND_VALUES)
        assert len(regions) == 1
        want = set(regions[0]["cells"])
        got = {(r, c) for r in range(len(grid)) for c in range(len(grid))
               if grid[r][c] != tgt[r][c]}
        assert got == want, "changed cells != enclosed region"
        checked += 1
    assert checked == 20


def test_two_rings_changes_exactly_the_cued_region():
    for t in CG.make_batch("two_rings_select", 20, seed=5):
        grid, tgt = t["input"], t["target"]
        cue = t["meta"]["cue_colour"]
        from henri_region_selector import enclosed_regions
        want = set()
        n_match = 0
        for reg in enclosed_regions(grid, CG.BACKGROUND_VALUES):
            if int(reg["bounding_colour"]) == cue:
                want |= set(reg["cells"])
                n_match += 1
        got = {(r, c) for r in range(len(grid)) for c in range(len(grid))
               if grid[r][c] != tgt[r][c]}
        assert got == want
        assert n_match == 1, "the cue must select exactly ONE ring; got %d" % n_match


def test_reflection_target_is_the_rotated_input():
    for t in CG.make_batch("reflection", 10, seed=11):
        grid, tgt = t["input"], t["target"]
        turns = t["meta"]["turns"]
        rot = [list(row) for row in zip(*grid[::-1])]
        for _ in range(turns - 1):
            rot = [list(row) for row in zip(*rot[::-1])]
        assert tgt == rot, "reflection target is not the declared rotation"


def test_reflection_preserves_the_nonzero_cell_count():
    """A rotation is a bijection on occupied cells; a lost/gained cell is a defect."""
    for t in CG.make_batch("reflection", 10, seed=13):
        a = sum(1 for row in t["input"] for v in row if v != 0)
        b = sum(1 for row in t["target"] for v in row if v != 0)
        assert a == b, "rotation changed the occupied-cell count"


def test_curve_and_fill_colour_bands_are_disjoint():
    assert not (set(CG.CURVE_COLOURS) & set(CG.FILL_COLOURS))
    assert not (set(CG.CURVE_COLOURS) & set(CG.BACKGROUND_VALUES))


def test_unknown_family_fails_closed():
    import pytest
    with pytest.raises(CG.GridTaskError):
        CG.make_task("no_such_family", __import__("random").Random(0))


def test_task_optionally_couples_to_the_verified_encoders():
    """The directive's requirement: couple the generator to the verified encoders."""
    import henri_topological_encoder as TE
    import henri_scene_binder as SB
    enc = TE.MultiscaleTopologicalEncoder(d_model=1024, enabled=True)
    bnd = SB.SceneBinder(dim=512)
    t = CG.make_batch("containment_fill", 1, seed=3, size=11)[0]
    info = CG.encode_task(t, encoder=enc, binder=bnd)
    assert info["wave_shape"], "no wave produced"
    assert info["wave_norm"] and info["wave_norm"] > 0, "wave has zero norm"
    assert info["wave_is_finite"] is True, "wave is not finite"
    assert info["n_components"] >= 1, "topological encoder saw no component"
    assert info["n_interior_cells"] >= 1, "no interior cells marked (Jordan defect class)"
    assert info["n_levels"] >= 1
    assert info["scene_shape"] and info["n_objects"] >= 1


def test_encode_task_works_without_the_encoders():
    t = CG.make_batch("reflection", 1, seed=4)[0]
    info = CG.encode_task(t)
    assert info["family"] == "reflection"
    assert "wave_shape" not in info


def test_describe_states_the_generator_is_not_a_solver():
    d = CG.describe()
    assert d["generator_is_not_a_solver"] is True
    assert d["schema"] == "henri.curriculum-grid.v1"
