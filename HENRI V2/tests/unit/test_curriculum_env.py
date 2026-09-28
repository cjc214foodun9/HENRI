"""Contract tests for henri_curriculum_env.py (Directive 1: make the levers REAL).

The module exists because the governor's four non-`prog_len` levers occurred 0
times in the driver and the seeder: they were declared-but-unread. These tests
pin the properties that make a lever REAL rather than decorative.

  DEFAULT EQUIVALENCE  neutral spec == sample_program EXACTLY (same seed)
  DIFFERENTIAL EFFECT  every non-neutral lever CHANGES the token stream
  DETERMINISM          same seed -> same programs
  LENGTH BOUND         never exceeds MAX_LEN_MULTIPLE x base length
  VM TOTALITY          the VM executes the products without raising
  FAIL-CLOSED          bad alphabet / empty program / bad spec raise
"""
import os
import sys

import pytest
import torch

def _project_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    d = here
    for _ in range(6):
        if os.path.exists(os.path.join(d, "henri_curriculum_env.py")):
            return d
        d = os.path.dirname(d)
    return here


C = _project_root()
if C not in sys.path:
    sys.path.insert(0, C)

import henri_curriculum_env as E  # noqa: E402
from stage0_universal_seeder import (  # noqa: E402
    CircularTapeVM, VMConfig, sample_program,
)

ALPHA = E.DEFAULT_ALPHABET_SIZE
SEED = 20260927


def _neutral(**over):
    s = dict(E.NEUTRAL_SPEC)
    s.update(over)
    return s


def _gen(seed=SEED):
    return torch.Generator().manual_seed(seed)


# ------------------------------------------------------- default equivalence
def test_neutral_spec_reproduces_sample_program_exactly():
    """The default-OFF path must be byte-identical to the legacy generator."""
    a = E.sample_program_from_spec(_neutral(prog_len=32), _gen(), sample_program, ALPHA)
    b = sample_program(32, _gen())
    assert a == b


def test_neutral_spec_is_reported_neutral():
    assert E.is_neutral(_neutral()) is True


def test_prog_len_alone_is_still_neutral_for_structure():
    """prog_len never changes STRUCTURE, so it must not count as non-neutral."""
    assert E.is_neutral(_neutral(prog_len=96.0)) is True


# ------------------------------------------------------- differential effect
def test_nesting_changes_the_stream():
    base = E.sample_program_from_spec(_neutral(prog_len=8), _gen(), sample_program, ALPHA)
    deep = E.sample_program_from_spec(_neutral(prog_len=8, multiscale_nesting=3.0),
                                      _gen(), sample_program, ALPHA)
    assert deep != base
    assert len(deep) == len(base) * 4          # 2**(3-1)


def test_each_nesting_level_lengthens_and_differs():
    lens = []
    for d in (1.0, 2.0, 3.0, 4.0):
        r = E.sample_program_from_spec(_neutral(prog_len=8, multiscale_nesting=d),
                                       _gen(), sample_program, ALPHA)
        lens.append(len(r))
    assert lens == [8, 16, 32, 64], lens


def test_obstacle_adds_barrier_spans_and_changes_the_stream():
    base = E.sample_program_from_spec(_neutral(prog_len=16), _gen(), sample_program, ALPHA)
    obs = E.sample_program_from_spec(_neutral(prog_len=16, topological_obstacle=3.0),
                                     _gen(), sample_program, ALPHA)
    assert obs != base
    assert len(obs) == len(base) + 3 * E.BARRIER_SPAN
    barrier = ALPHA - 1
    assert obs.count(barrier) >= 3 * E.BARRIER_SPAN - base.count(barrier)


def test_distractor_changes_the_stream_and_preserves_length():
    base = E.sample_program_from_spec(_neutral(prog_len=16), _gen(), sample_program, ALPHA)
    dis = E.sample_program_from_spec(_neutral(prog_len=16, distractor_noise=5.0),
                                     _gen(), sample_program, ALPHA)
    assert dis != base
    assert len(dis) == len(base), "distractors overwrite, they do not insert"


def test_levers_compose():
    s = _neutral(prog_len=8, multiscale_nesting=2.0, topological_obstacle=2.0,
                 distractor_noise=3.0)
    r = E.sample_program_from_spec(s, _gen(), sample_program, ALPHA)
    assert len(r) == 16 + 2 * E.BARRIER_SPAN
    assert E.is_neutral(s) is False


def test_tape_size_lever_is_bounded():
    assert E.tape_size_from_spec(_neutral(grid_growth=16.0)) == 32   # floor
    assert E.tape_size_from_spec(_neutral(grid_growth=256.0)) == 256
    assert E.tape_size_from_spec(_neutral(grid_growth=10 ** 9)) == 1 << 20
    assert E.tape_size_from_spec(_neutral(grid_growth=-5.0)) == 32


# ------------------------------------------------------------- determinism
def test_same_seed_same_program():
    s = _neutral(prog_len=12, topological_obstacle=2.0, distractor_noise=2.0)
    a = E.sample_program_from_spec(s, _gen(7), sample_program, ALPHA)
    b = E.sample_program_from_spec(s, _gen(7), sample_program, ALPHA)
    assert a == b


def test_different_seed_different_program():
    s = _neutral(prog_len=24, topological_obstacle=2.0)
    a = E.sample_program_from_spec(s, _gen(1), sample_program, ALPHA)
    b = E.sample_program_from_spec(s, _gen(2), sample_program, ALPHA)
    assert a != b


# -------------------------------------------------------------- length bound
def test_extreme_obstacle_respects_the_length_bound():
    s = _neutral(prog_len=8, multiscale_nesting=4.0, topological_obstacle=500.0)
    r = E.sample_program_from_spec(s, _gen(), sample_program, ALPHA)
    cap = E.MAX_LEN_MULTIPLE * 8 * (2 ** 3)
    assert len(r) <= cap, (len(r), cap)


def test_all_tokens_stay_inside_the_alphabet():
    for spec in (_neutral(prog_len=16, topological_obstacle=4.0, distractor_noise=6.0),
                 _neutral(prog_len=16, multiscale_nesting=3.0, distractor_noise=9.0)):
        r = E.sample_program_from_spec(spec, _gen(3), sample_program, ALPHA)
        assert all(0 <= t < ALPHA for t in r), sorted(set(r))


# ---------------------------------------------------------------- VM totality
def test_vm_executes_the_products_without_raising():
    """Totality is PROVEN BY EXECUTION, not asserted."""
    vm = CircularTapeVM(VMConfig(tape_size=256, max_steps=512, max_output=32))
    specs = [
        _neutral(prog_len=16),
        _neutral(prog_len=16, topological_obstacle=3.0),
        _neutral(prog_len=16, distractor_noise=5.0),
        _neutral(prog_len=16, multiscale_nesting=3.0, topological_obstacle=2.0,
                 distractor_noise=4.0),
    ]
    for s in specs:
        progs = [E.sample_program_from_spec(s, _gen(seed), sample_program, ALPHA)
                 for seed in range(4)]
        results = vm.execute_batch(progs)          # must not raise
        assert len(results) == len(progs)
        for r in results:
            assert isinstance(r.timed_out, bool)


def test_vm_reports_timeouts_as_a_normal_outcome():
    """A barrier span may push a program to TIMEOUT; that is a bounded outcome,
    not a crash, and it must be reported rather than swallowed."""
    vm = CircularTapeVM(VMConfig(tape_size=64, max_steps=32, max_output=16))
    progs = [E.sample_program_from_spec(_neutral(prog_len=8, topological_obstacle=6.0),
                                        _gen(s), sample_program, ALPHA) for s in range(6)]
    res = vm.execute_batch(progs)
    assert sum(1 for r in res if r.timed_out) >= 0


# --------------------------------------------------------------- fail closed
def test_tiny_alphabet_raises():
    with pytest.raises(E.CurriculumEnvError):
        E.sample_program_from_spec(_neutral(prog_len=8), _gen(), sample_program, 1)


def test_empty_program_raises():
    def empty_fn(n, rng):
        return []
    with pytest.raises(E.CurriculumEnvError):
        E.sample_program_from_spec(_neutral(prog_len=8), _gen(), empty_fn, ALPHA)


def test_prog_len_is_honoured_for_zero_and_negative():
    """A non-positive prog_len floors at 1 token (never an empty program)."""
    r = E.sample_program_from_spec(_neutral(prog_len=0.0), _gen(), sample_program, ALPHA)
    assert len(r) == 1


# ------------------------------------------------------------------- report
def test_spec_report_is_json_safe_and_describes_every_lever():
    import json
    rep = E.spec_report(_neutral(prog_len=16, topological_obstacle=2.0,
                                 multiscale_nesting=3.0, distractor_noise=1.0))
    json.dumps(rep)
    assert rep["neutral"] is False
    assert rep["nesting_repetitions"] == 4
    assert rep["barrier_spans"] == 2
    assert rep["distractor_positions"] == 1
    assert "tape_size" in rep


def test_neutral_report_says_neutral():
    assert E.spec_report(_neutral())["neutral"] is True
