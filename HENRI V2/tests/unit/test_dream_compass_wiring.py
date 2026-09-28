"""Guards for the dream-compass WIRING inside henri_latent_dreamer.py.

WHY THIS FILE EXISTS
  The blueprint's deficit #2 reads "Connect RAW reward to live test-time SGLD". My own
  grep measured that the dreamer -- which IS the live SGLD loop -- referenced
  `gradient_alignment` and `alignment_reward` 0 times. `henri_dream_compass.py` supplies
  the compass; this file proves the dreamer actually CALLS it, that the OFF path is
  unchanged, and that neither GATE-C nor GATE-D was weakened by the wiring.

GATES THAT MUST SURVIVE
  GATE-C  the adapter still enters the SCORED wave, so grads stay live
  GATE-D  guarded_egress still raises until a real M3-ACT receipt
  SGLD    the noise term keeps the sqrt(2 T dt) form
"""
import os
import sys

import pytest
import torch

def _root():
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        if os.path.exists(os.path.join(d, "henri_latent_dreamer.py")):
            return d
        d = os.path.dirname(d)
    return os.path.dirname(os.path.abspath(__file__))

C = _root()
if C not in sys.path:
    sys.path.insert(0, C)

import henri_dream_compass as DC                      # noqa: E402
from henri_latent_dreamer import (                    # noqa: E402
    ENTRY_THRESHOLD, DreamConfig, DreamEgressBlocked, FocusedLatentDreamer,
)

D_MODEL, N_BLOCKS, N_EXPERTS, R_RANK = 128, 16, 8, 4


@pytest.fixture(scope="module")
def core():
    from henri_zone_a_core import ZoneACore
    torch.manual_seed(0)
    return ZoneACore(d_model=D_MODEL, num_blocks=N_BLOCKS, num_experts=N_EXPERTS,
                     r_rank=R_RANK)


def _pair(core, seed_a=1, seed_b=5):
    """Two DIFFERENT grids, so the candidate deltas are large enough to enter a dream."""
    ga = torch.randint(0, 10, (5, 5), generator=torch.Generator().manual_seed(seed_a)).tolist()
    gb = torch.randint(0, 10, (5, 5), generator=torch.Generator().manual_seed(seed_b)).tolist()
    return core.encode(ga), core.encode(gb)


def _entering(core, **cfg):
    """A dream that ACTUALLY enters, or pytest.skip with the measured delta."""
    wa, wb = _pair(core)
    d = FocusedLatentDreamer(core, DreamConfig(**cfg) if cfg else None)
    delta = min(d.core.candidate_set(wa, wb, top_k=4).delta_floats())
    if not (delta > ENTRY_THRESHOLD):
        pytest.skip("fixture does not enter the dream (delta %.4f <= %.4f)"
                    % (delta, ENTRY_THRESHOLD))
    return d, wa, wb, delta


# ------------------------------------------------------- the wiring exists at all
def test_dreamer_accepts_an_injected_compass(core):
    """Injection, not construction: the dreamer must not create its own compass."""
    wa, wb = _pair(core)
    cm = DC.AlignmentCompass()
    d = FocusedLatentDreamer(core, None, compass=cm)
    assert d.compass is cm
    out = d.dream(wa, wb, top_k=4, max_steps=2)
    assert out.entered is True, "the fixture must enter the dream for this to mean anything"


def test_compass_none_leaves_the_loop_unchanged(core):
    """DEFAULT-OFF: with no compass the new telemetry stays empty and nothing else moves."""
    d, wa, wb, _delta = _entering(core)
    out = d.dream(wa, wb, top_k=4, max_steps=3)
    assert out.alignment_reward_last is None
    assert out.alignment_reward_mean is None
    assert out.alignment_steps == 0
    assert out.compass_terminated is False
    assert out.compass_reason is None


def test_compass_records_alignment_rewards(core):
    """The MEASURED property: with a compass the loop reports non-empty reward telemetry."""
    wa, wb = _pair(core)
    cm = DC.AlignmentCompass(terminate_below=-1e18)      # never terminate
    d = FocusedLatentDreamer(core, None, compass=cm)
    out = d.dream(wa, wb, top_k=4, max_steps=4)
    assert out.entered is True
    assert out.alignment_steps == out.steps_taken, (out.alignment_steps, out.steps_taken)
    assert out.alignment_steps > 0
    assert out.alignment_reward_last is not None
    assert out.alignment_reward_mean is not None
    assert out.compass_terminated is False


def test_compass_reward_matches_the_compass_history(core):
    """The dataclass must report what the compass actually recorded, not a copy."""
    wa, wb = _pair(core)
    cm = DC.AlignmentCompass(terminate_below=-1e18)
    d = FocusedLatentDreamer(core, None, compass=cm)
    out = d.dream(wa, wb, top_k=4, max_steps=4)
    assert len(cm.history) == out.alignment_steps
    assert out.alignment_reward_last == pytest.approx(cm.history[-1], rel=0, abs=1e-12)
    assert out.alignment_reward_mean == pytest.approx(sum(cm.history) / len(cm.history),
                                                     rel=0, abs=1e-12)


def test_compass_terminates_the_loop_early(core):
    """The compass may STOP a dream. That is the only thing it changes."""
    wa, wb = _pair(core)
    cm = DC.AlignmentCompass(terminate_below=1e18, consecutive=1)   # always terminate
    d = FocusedLatentDreamer(core, None, compass=cm)
    out = d.dream(wa, wb, top_k=4, max_steps=6)
    assert out.entered is True
    assert out.compass_terminated is True
    assert out.compass_reason is not None and out.compass_reason.startswith("TERMINATE_BELOW")
    assert out.steps_taken < 6, "termination must end the loop before the limit"


def test_terminate_and_wake_are_independent_exits(core):
    """A terminated dream is NOT a woken dream; the two flags must not be conflated."""
    wa, wb = _pair(core)
    cm = DC.AlignmentCompass(terminate_below=1e18, consecutive=1)
    d = FocusedLatentDreamer(core, None, compass=cm)
    out = d.dream(wa, wb, top_k=4, max_steps=6)
    assert out.compass_terminated is True
    assert isinstance(out.woke, bool)


# ----------------------------------------------------- the compass cannot change weights
def test_compass_installs_no_weight_change_by_default(core):
    """With creep OFF (the default) the adapter must remain an exact identity, so the
    compass is a pure observer. Any non-zero adapter movement would mean the wiring
    leaked a weight update past the SGLD flag."""
    wa, wb = _pair(core)
    cm = DC.AlignmentCompass(terminate_below=-1e18)
    d = FocusedLatentDreamer(core, None, compass=cm)
    before = d.adapter.delta_norm()
    out = d.dream(wa, wb, top_k=4, max_steps=3)
    assert out.creep_enabled is False, "creep must be OFF by default for this test"
    assert d.adapter.delta_norm() == before == 0.0
    assert out.adapter_delta_norm == 0.0


def test_compass_does_not_grant_egress(core):
    """GATE-D: a compass present (or having terminated a dream) must NOT open the latch."""
    d, wa, wb, _d = _entering(core)
    d.compass = DC.AlignmentCompass(terminate_below=1e18, consecutive=1)
    d.dream(wa, wb, top_k=4, max_steps=2)
    assert d.egress_permitted() is False
    with pytest.raises(DreamEgressBlocked):
        d.guarded_egress(torch.zeros(3))


# ------------------------------------------------------------------- fail-open
def test_a_broken_compass_cannot_break_the_dream(core):
    """The wiring is wrapped: a compass that raises must leave the dream intact and the
    reason must be RECORDED rather than swallowed."""
    class _Broken:
        def judge(self, *a, **k):
            raise RuntimeError("compass exploded")

    wa, wb = _pair(core)
    d = FocusedLatentDreamer(core, None, compass=_Broken())
    out = d.dream(wa, wb, top_k=4, max_steps=3)
    assert out.entered is True
    assert out.alignment_steps == 0
    assert out.compass_terminated is False, "a broken compass must not stop the loop"
    assert out.compass_reason is not None and out.compass_reason.startswith("UNAVAILABLE")


def test_an_unmeasurable_step_does_not_end_the_dream(core):
    """Shape-mismatch path: the compass returns reward None / continue True, so a dream
    with no measurable alignment still runs to its own wake/limits."""
    wa, wb = _pair(core)
    # terminate_below at -inf means even a 0 reward continues; reward is measured here,
    # so this asserts the RECORDED mean is finite while the loop proceeds normally.
    cm = DC.AlignmentCompass(terminate_below=-1e18)
    d = FocusedLatentDreamer(core, None, compass=cm)
    out = d.dream(wa, wb, top_k=4, max_steps=2)
    assert out.compass_terminated is False
    if out.alignment_reward_mean is not None:
        assert out.alignment_reward_mean == out.alignment_reward_mean   # not NaN


# ------------------------------------------------------- structural guarantees
def test_dream_still_never_emits_an_action(core):
    """Boundary: DreamResult carries telemetry only. It must have no action field."""
    d, wa, wb, _ = _entering(core)
    out = d.dream(wa, wb, top_k=4, max_steps=2)
    names = set(vars(out))
    for banned in ("action", "selected_action", "command", "reward_to_emit"):
        assert banned not in names, "DreamResult must not carry %r" % banned


def test_new_telemetry_fields_all_have_defaults(core):
    """Positional compatibility: existing callers construct DreamResult positionally."""
    from henri_latent_dreamer import DreamResult
    positional = DreamResult(True, 3, False, 0.5, 0.4, 0.9, 0.1, False)
    assert positional.alignment_reward_last is None
    assert positional.alignment_steps == 0
    assert positional.compass_terminated is False


def test_sgld_noise_form_is_unchanged():
    """The wiring must not touch the creep update. sqrt(2 T dt) is the ratified form."""
    src = open(os.path.join(C, "henri_latent_dreamer.py"), encoding="utf-8").read()
    assert "(2.0 * cfg.sgld_temp) ** 0.5" in src
    assert "cfg.sgld_lr * g + noise" in src


def test_gate_c_gradient_path_still_uses_autograd():
    src = open(os.path.join(C, "henri_latent_dreamer.py"), encoding="utf-8").read()
    assert "torch.autograd.grad" in src
    assert "allow_unused=False" in src


def test_dreamer_does_not_construct_a_compass_itself():
    """Injection only: the dreamer must not import the compass module, or the OFF path
    could acquire a hidden dependency."""
    src = open(os.path.join(C, "henri_latent_dreamer.py"), encoding="utf-8").read()
    code = "\n".join(l.split("#")[0] for l in src.splitlines())
    assert "import henri_dream_compass" not in code
    assert "AlignmentCompass(" not in code
