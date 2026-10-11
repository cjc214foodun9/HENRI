"""Discriminating tests for henri_candidate_bank -- the G-SJ3a/G-SJ3b unblock.

Every test CAN FAIL. The negative controls carry the load: a bank builder that silently
degenerates to one column per trace, or a channel that is constant across candidates,
produces a two-stage selector identical to uniform. Both must be caught HERE, not in a
gate that cannot fail.
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
HC = os.path.abspath(os.path.join(HERE, "..", "..", "henri_core"))
if HC not in sys.path:
    sys.path.insert(0, HC)

import henri_candidate_bank as cb  # noqa: E402


def test_flat_concat_real_and_imag_for_complex():
    z = torch.randn(3, 4, dtype=torch.cfloat)
    out = cb.flat(z)
    assert tuple(out.shape) == (3, 8)


def test_flat_is_identity_for_real():
    r = torch.randn(3, 5)
    assert torch.equal(cb.flat(r), r)


def test_gold_sets_groups_rows_by_trace_and_reports_empty():
    tr = [(1, 2), (3, 4), (1, 2)]
    q = [(1, 2), (9, 9)]
    g = cb.gold_sets(tr, q)
    assert g[0] == {0, 2}
    assert g[1] == set(), "an unseen query must yield an EMPTY gold set, not a default"


def test_dedup_columns_keeps_first_occurrence_in_order():
    tr = [(1,), (2,), (1,), (3,), (2,)]
    assert cb.dedup_columns(tr) == [0, 1, 3]


def test_dedup_reduces_columns_when_multiplicity_exists():
    """NEGATIVE CONTROL for the dedup step: if traces are already distinct, dedup must
    be the identity. A dedup that shrinks a distinct bank is dropping real candidates."""
    tr = [(i,) for i in range(9)]
    assert cb.dedup_columns(tr) == list(range(9))


def test_restrict_remaps_gold_into_the_new_column_space():
    sim = torch.arange(12, dtype=torch.float32).reshape(2, 6)
    cols = [0, 2, 5]
    gold = [{0, 3}, {5}]
    s2, g2 = cb.restrict(sim, cols, gold)
    assert tuple(s2.shape) == (2, 3)
    assert g2[0] == {0}, "index 0 stays at column 0; index 3 is dropped (not retained)"
    assert g2[1] == {2}, "global index 5 maps to local column 2"


def test_max_groups_returns_indices_not_values():
    sim = torch.tensor([[1.0, 3.0, 3.0, 0.0]])
    assert cb.max_groups(sim)[0] == [1, 2]


def test_noop_check_flags_a_constant_channel():
    """KEY NEGATIVE CONTROL. A constant channel makes two_stage == uniform exactly.
    If this test fails, a dead mechanism can ship looking like a working one."""
    const = torch.ones(5, 7)
    r = cb.channel_noop_check(const)
    assert r["frac_constant"] == 1.0
    assert r["is_noop"] is True


def test_noop_check_passes_an_informative_channel():
    ch = torch.arange(7, dtype=torch.float32).unsqueeze(0).expand(5, 7).clone()
    r = cb.channel_noop_check(ch)
    assert r["n_unique_values"] == 7
    assert r["is_noop"] is False


def test_frequency_channel_keeps_multiplicity_and_is_not_constant():
    """THE LOAD-BEARING TEST for the dedup bank. Counts must come from the FULL bank.
    If they came from the restricted list every value would be 1 and the channel would
    be constant -- a strict no-op. Measured requirement: >1 distinct value."""
    full = [(1,), (1,), (2,), (3,), (3,), (3,)]
    cols = [0, 2, 3]
    ch = cb.frequency_channel(full, cols, (4, 3))
    assert tuple(ch.shape) == (4, 3)
    vals = ch[0].tolist()
    assert vals == [2.0, 1.0, 3.0], vals
    assert len(set(vals)) > 1, "constant channel -> two_stage is a no-op"


def test_size_matched_control_grows_with_group_width():
    """A WIDER tie group is easier to hit by chance. If the control did not track group
    width it would be a FIXED-size control, which is confounded by group width and
    cannot test containment."""
    K = 20
    narrow = torch.zeros(1, K)
    narrow[0, 0] = 1.0                      # singleton group
    wide = torch.zeros(1, K)
    wide[0, :12] = 1.0                      # 12-way tie
    gold = [{0}]
    cn = cb.containment_report(narrow, gold)
    cw = cb.containment_report(wide, gold)
    assert cn["mean_group_size"] == 1.0
    assert cw["mean_group_size"] == 12.0
    assert cw["size_matched_control"] > cn["size_matched_control"]


def test_containment_report_bounds_and_chance_denominator():
    sim = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    gold = [{0}, {1}]
    r = cb.containment_report(sim, gold)
    assert r["containment"] == 1.0
    assert 0.0 <= r["size_matched_control"] <= 1.0
    # row-uniform chance uses the ACTUAL candidate count, never a hardcoded K
    assert abs(r["row_uniform_chance"] - (2 / (2 * 3))) < 1e-9


def test_score_selector_guard_rejects_mismatched_trace_list():
    """The bug that crashed my first driver: a 48-trace list against a 28-column matrix.
    The guard must refuse rather than let expand() raise an opaque RuntimeError."""
    import wave_readout as wr  # noqa: F401
    sim = torch.randn(3, 4)
    gold = [{0}, {1}, {2}]
    try:
        cb.score_selector(sim, gold, None, "two_stage", [(0,)] * 9)
    except ValueError as e:
        assert "one trace per column" in str(e)
    else:
        raise AssertionError("guard did not fire on a mismatched trace list")
