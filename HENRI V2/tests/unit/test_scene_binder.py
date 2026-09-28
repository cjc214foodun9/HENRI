"""Contract tests for henri_scene_binder.py (Gap 1, hierarchical nesting).

The module exists because the ingress is FLAT: one superposed level, no object
tree. These tests pin the representation properties that make a scene graph
meaningful, and they exist specifically because the repository's obvious binding
primitive (the qFHRR random-ring codec) is MEASURED NON-COMPOSITIONAL, so a
binder built on it would pass a similarity check while carrying no structure.

  PERMUTATION       swapping two objects' roles must change the scene wave.
  ORDER             the same two objects bound in the other order must differ.
  COLOUR/POSITION   changing one object's colour or position must change the wave.
  NESTING           adding a part must change the wave (parts are load-bearing).
  RETRIEVAL         unbinding a role must recover that object's filler better than
                    a random filler (the operation must actually work).
  DEAD-INPUT        content_blind-style control: identical scenes -> identical
                    waves (determinism), and the binder must be a PURE function.
  FAIL-CLOSED       too many objects / bad shape / bad role must raise.
  HONEST            the poisoned-prior warning must be recorded in the module.
"""
import math
import os
import sys

import pytest
import torch

def _project_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    d = here
    for _ in range(6):
        if os.path.exists(os.path.join(d, "henri_scene_binder.py")):
            return d
        d = os.path.dirname(d)
    return here


C = _project_root()
if C not in sys.path:
    sys.path.insert(0, C)

import henri_scene_binder as S  # noqa: E402

DIM = 512


def _objs(n=3, seed=1):
    objs, b = S.make_scene(n, seed, dim=DIM, n_roles=4)
    return objs, b


# ------------------------------------------------------------- determinism
def test_binding_is_deterministic():
    objs, b = _objs()
    a1 = b.bind_scene(objs)
    a2 = b.bind_scene(objs)
    assert float((a1 - a2).abs().max()) == 0.0


def test_binder_is_pure_across_instances():
    """Two independent binders with the same dim must agree exactly."""
    objs, b1 = _objs(3, 5)
    _, b2 = _objs(3, 5)
    assert float((b1.bind_scene(objs) - b2.bind_scene(objs)).abs().max()) == 0.0


def test_binding_preserves_the_hypersphere():
    objs, b = _objs()
    s = b.bind_scene(objs)
    assert abs(float(torch.linalg.vector_norm(s)) - 1.0) < 1e-5


def test_bound_wave_is_finite_and_correct_width():
    objs, b = _objs()
    s = b.bind_scene(objs)
    assert tuple(s.shape) == (DIM,)
    assert torch.isfinite(s).all()


# ------------------------------------------------------------ permutation
def test_swapping_two_objects_ROLES_changes_the_wave():
    objs, b = _objs(3, 7)
    a = b.bind_scene(objs)
    swapped = [objs[1], objs[0], objs[2]]
    c = b.bind_scene(swapped)
    assert float((a - c).abs().max()) > 1e-6, "object order must set the role binding"


def test_changing_one_COLOUR_changes_the_wave():
    objs, b = _objs(3, 9)
    a = b.bind_scene(objs)
    v = [S.SceneObject(o.kind_id, o.colour_id + 1, o.position_id, list(o.parts))
         for o in objs]
    c = b.bind_scene(v)
    assert float((a - c).abs().max()) > 1e-6


def test_changing_one_POSITION_changes_the_wave():
    objs, b = _objs(3, 11)
    a = b.bind_scene(objs)
    v = [S.SceneObject(o.kind_id, o.colour_id, o.position_id + 1, list(o.parts))
         for o in objs]
    c = b.bind_scene(v)
    assert float((a - c).abs().max()) > 1e-6


def test_changing_one_SHAPE_changes_the_wave():
    objs, b = _objs(3, 13)
    a = b.bind_scene(objs)
    v = [S.SceneObject(o.kind_id + 1, o.colour_id, o.position_id, list(o.parts))
         for o in objs]
    c = b.bind_scene(v)
    assert float((a - c).abs().max()) > 1e-6


# ---------------------------------------------------------------- nesting
def test_adding_a_PART_changes_the_wave():
    objs, b = _objs(2, 15)
    a = b.bind_scene(objs)
    v = [S.SceneObject(o.kind_id, o.colour_id, o.position_id, list(o.parts) + [9])
         for o in objs]
    c = b.bind_scene(v)
    assert float((a - c).abs().max()) > 1e-6, "parts must be load-bearing (the 'recursive' term)"


def test_part_ORDER_matters_within_an_object():
    """Nested parts carry their own level keys, so order must not be free."""
    _, b = _objs(1, 17)
    o1 = S.SceneObject(1, 2, 3, parts=[4, 5])
    o2 = S.SceneObject(1, 2, 3, parts=[5, 4])
    assert float((b.bind_object(o1) - b.bind_object(o2)).abs().max()) > 1e-6


def test_object_with_no_parts_still_binds():
    _, b = _objs(1, 19)
    w = b.bind_object(S.SceneObject(1, 2, 3))
    assert abs(float(torch.linalg.vector_norm(w)) - 1.0) < 1e-5


# -------------------------------------------------------------- retrieval
def test_exact_single_filler_recovery_via_role_filler_scene():
    """THE STANDARD HRR READ-OUT: scene = (+)_r bind(role_r, filler_r).

    One filler per role, so unbinding returns that filler with cosine ~1 (up to
    crosstalk of order 1/sqrt(dim)). This is the operation that MUST work for a
    binding scheme to be useful, and it is the test the multi-factor form cannot
    satisfy (see the module docstring on why)."""
    b = S.SceneBinder(dim=4096, n_roles=4)
    fills = {0: b.filler_vector("colour", 3),
             1: b.filler_vector("colour", 7),
             2: b.filler_vector("colour", 5)}
    codebook = {c: b.filler_vector("colour", c) for c in range(1, 12)}
    scene = b.role_filler_scene(list(fills.items()))
    for role in (0, 1, 2):
        got = b.probe_filler(scene, role, list(codebook.items()))
        want = {0: 3, 1: 7, 2: 5}[role]
        assert got == want, f"role {role}: got colour {got}, want {want}"


def test_role_filler_recovery_survives_a_role_swap():
    b = S.SceneBinder(dim=4096, n_roles=4)
    pairs = [(0, b.filler_vector("colour", 2)), (1, b.filler_vector("colour", 9))]
    scene = b.role_filler_scene(pairs)
    assert b.probe_filler(scene, 0, [(c, b.filler_vector("colour", c)) for c in range(1, 12)]) == 2
    assert b.probe_filler(scene, 1, [(c, b.filler_vector("colour", c)) for c in range(1, 12)]) == 9
    swapped = b.role_filler_scene([(0, pairs[1][1]), (1, pairs[0][1])])
    assert b.probe_filler(swapped, 0, [(c, b.filler_vector("colour", c)) for c in range(1, 12)]) == 9
    assert b.probe_filler(swapped, 1, [(c, b.filler_vector("colour", c)) for c in range(1, 12)]) == 2


def test_object_recovery_via_codebook_cleanup():
    """A MULTI-FACTOR object cannot be read factor-by-factor, but the OBJECT is
    recoverable: unbind the role and clean up against object waves."""
    objs, b = _objs(3, 33)
    scene = b.bind_scene(objs)
    codebook = {i: b.bind_object(o) for i, o in enumerate(objs)}
    for role in range(3):
        assert b.probe_role(scene, role, codebook) == role


def test_probe_filler_separates_true_from_wrong():
    b = S.SceneBinder(dim=4096, n_roles=2)
    true_c, other_c = 4, 8
    scene = b.role_filler_scene([(0, b.filler_vector("colour", true_c))])
    x = S.unbind(scene, b.role_vector(0))
    s_true = S.cosine(x, b.filler_vector("colour", true_c))
    s_other = S.cosine(x, b.filler_vector("colour", other_c))
    assert s_true > 0.5 and s_true > 5 * abs(s_other), f"true {s_true:.4f} vs other {s_other:.4f}"


def test_id_spaces_do_not_alias():
    """role_id=k and shape_id=k must NOT produce the same wave (measured defect:
    they shared a phase step and therefore collided)."""
    b = S.SceneBinder(dim=512)
    for k in range(4):
        assert float((b.role_vector(k) - b.filler_vector("shape", k)).abs().max()) > 1e-6
    assert float((b.filler_vector("part", 3) - b.filler_vector("level", 3)).abs().max()) > 1e-6


def test_components_are_near_orthogonal():
    """Binding needs pairwise SEPARATION. A constant phase offset gives |cos| that
    is dimension-INDEPENDENT (measured -0.744), which swamped every retrieval."""
    b = S.SceneBinder(dim=4096)
    import itertools
    cs = [abs(S.cosine(b.role_vector(i), b.role_vector(j)))
          for i, j in itertools.combinations(range(6), 2)]
    # each component has 1/sqrt(ndim/2)... bound loosely: all well under 1/sqrt(dim/8)
    assert max(cs) < 0.15, f"component crosstalk too high: max |cos| {max(cs):.4f}"


# ------------------------------------------------------------- primitives
def test_bind_many_is_associative_in_effect():
    b = S.SceneBinder(dim=DIM)
    a = b.filler_vector("colour", 3)
    c = b.filler_vector("shape", 4)
    e = b.filler_vector("position", 5)
    left = S.bind_many([S.bind_many([a, c]), e])
    right = S.bind_many([a, S.bind_many([c, e])])
    assert float((left - right).abs().max()) < 1e-6


def test_unbind_inverts_binding():
    b = S.SceneBinder(dim=DIM)
    key = b.role_vector(0)
    val = b.filler_vector("colour", 6)
    bound = S.bind_many([val, key])
    back = S.unbind(bound, key)
    assert S.cosine(back, val) > 0.999


def test_bind_component_is_unit_modulus():
    c = S.bind_component(64, S._STEP_ROLE, 3, salt=11)
    # every (re,im) pair has modulus 1 by construction
    re, im = c[0::2], c[1::2]
    mod = torch.sqrt(re * re + im * im)
    assert float((mod - 1.0).abs().max()) < 1e-5
    assert c.numel() == 64, f"declared width 64, got {c.numel()} (the 2*dim defect)"


# ------------------------------------------------------------ fail closed
def test_too_many_objects_raises():
    objs, b = _objs(6, 27)
    b2 = S.SceneBinder(dim=DIM, n_roles=4)
    with pytest.raises(S.SceneBinderError):
        b2.bind_scene(objs[:5])


def test_empty_scene_raises():
    b = S.SceneBinder(dim=DIM)
    with pytest.raises(S.SceneBinderError):
        b.bind_scene([])


def test_bad_dim_raises():
    with pytest.raises(S.SceneBinderError):
        S.SceneBinder(dim=7)
    with pytest.raises(S.SceneBinderError):
        S.SceneBinder(dim=4)


def test_bind_many_shape_mismatch_raises():
    b = S.SceneBinder(dim=DIM)
    with pytest.raises(S.SceneBinderError):
        S.bind_many([b.role_vector(0), torch.zeros(8)])


def test_unbind_shape_mismatch_raises():
    b = S.SceneBinder(dim=DIM)
    with pytest.raises(S.SceneBinderError):
        S.unbind(b.role_vector(0), torch.zeros(8))


def test_bind_many_empty_raises():
    with pytest.raises(S.SceneBinderError):
        S.bind_many([])


def test_unknown_filler_kind_raises():
    b = S.SceneBinder(dim=DIM)
    with pytest.raises(KeyError):
        b.filler_vector("nonsense", 1)


# ------------------------------------------------------- honest boundaries
def test_poisoned_prior_is_recorded_in_the_module():
    """The qFHRR non-compositionality finding must be stated where a future
    session will read it, or the random-ring codec will be reintroduced."""
    src = open(os.path.join(C, "henri_scene_binder.py"), encoding="utf-8").read()
    low = src.lower()
    assert "poisoned prior" in low
    assert "non-compositional" in low
    assert "no spatial-reasoning benchmark score is claimed" in low


def _code_only(path: str) -> str:
    """Strip comments AND docstrings before a SYMBOL audit.

    DEFECT FIXED 2026-09-27: splitting on '#' leaves docstrings intact, so this
    guard fired on the module's own PROSE warning about the random-ring codec --
    a detector firing on its own subject. tokenize removes both.
    """
    import io
    import tokenize
    src = open(path, encoding="utf-8").read()
    out = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            out.append(tok.string)
    except (tokenize.TokenError, IndentationError):
        return src
    return " ".join(out)


def test_module_does_not_use_the_random_ring_codec():
    """Structural guard on CODE only: the binder must not import or call the
    random-ring codec. Its prose names the codec deliberately, to warn future
    sessions, so a docstring must not satisfy or break this check."""
    code = _code_only(os.path.join(C, "henri_scene_binder.py"))
    assert "qFHRREpistemicCodec" not in code
    assert "randint" not in code, "a random binding is non-compositional"
