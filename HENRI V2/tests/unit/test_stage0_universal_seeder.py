"""MILESTONE 2 contract tests — circular byte-tape VM (CPU reference).

Closes the `.md` M2 clauses that are CPU-testable:

  * 18-token alphabet = 8 Brainfuck + 10 VSA macro primitives;
  * sandboxed circular byte-tape VM with GUARANTEED execution:
    out-of-bounds wraps modulo 256, zero syntax crashes;
  * total execution: a diverging loop returns TIMEOUT, it does not hang.

The GPU kernel is NOT tested here. This is the CPU oracle a CUDA kernel must
match; a CPU pass is a unit result, never CUDA verification.
"""

from __future__ import annotations

import pytest
import torch

from stage0_universal_seeder import (
    ALPHABET,
    BF_TOKENS,
    MACRO_SEMANTICS,
    TIMEOUT_TOKEN,
    VSA_TOKENS,
    VMConfig,
    VMError,
    CircularTapeVM,
    default_input_stream,
    sample_program,
)

TOK = {t: i for i, t in enumerate(ALPHABET)}


def prog(*symbols: str) -> list[int]:
    return [TOK[s] for s in symbols]


# ======================================================================================
# alphabet
# ======================================================================================


def test_alphabet_is_eighteen_tokens():
    assert len(BF_TOKENS) == 8
    assert len(VSA_TOKENS) == 10
    assert len(ALPHABET) == 18
    assert len(set(ALPHABET)) == 18


def test_vsa_macros_are_all_documented():
    for tok in VSA_TOKENS:
        assert tok in MACRO_SEMANTICS, f"VSA token {tok} has no documented semantics"


# ======================================================================================
# fail-closed validation
# ======================================================================================


def test_config_rejects_bad_values():
    for bad in (dict(tape_size=0), dict(max_steps=0), dict(max_output=0), dict(tape_size=-5)):
        with pytest.raises(VMError):
            VMConfig(**bad).validate()


def test_config_rejects_bad_input_byte():
    with pytest.raises(VMError):
        VMConfig(input_stream=(0, 300)).validate()


def test_program_rejects_out_of_range_token():
    with pytest.raises(VMError, match="outside alphabet range"):
        CircularTapeVM().execute([0, 18])


def test_program_rejects_non_int_token():
    with pytest.raises(VMError, match="must be an int"):
        CircularTapeVM().execute([0, 1.5])


def test_program_rejects_bad_tensor_rank():
    with pytest.raises(VMError, match="rank 1"):
        CircularTapeVM().execute(torch.zeros(2, 2, dtype=torch.long))


def test_program_accepts_tensor():
    res = CircularTapeVM().execute(torch.tensor(prog("+", "."), dtype=torch.long))
    assert res.output == (1,)


# ======================================================================================
# guaranteed execution — no crashes, no hangs
# ======================================================================================


@pytest.mark.parametrize("symbol", ALPHABET)
def test_single_token_never_crashes(symbol):
    """Every token in the alphabet executes alone without fault."""
    res = CircularTapeVM().execute(prog(symbol))
    assert isinstance(res.steps, int)


def test_empty_program_executes():
    res = CircularTapeVM().execute([])
    assert res.output == () and res.steps == 0 and not res.timed_out


def test_all_two_token_programs_never_crash():
    """Exhaustive 18 x 18 = 324 pairs: total on all of them."""
    vm = CircularTapeVM(VMConfig(max_steps=256))
    for a in ALPHABET:
        for b in ALPHABET:
            res = vm.execute(prog(a, b))
            assert res.steps <= 257


def test_diverging_loop_returns_timeout_not_hang():
    """`+[]` loops forever; the step budget must convert it to TIMEOUT."""
    vm = CircularTapeVM(VMConfig(max_steps=500))
    res = vm.execute(prog("+", "[", "]"))
    assert res.timed_out is True
    assert TIMEOUT_TOKEN in res.output
    assert res.steps == 500


def test_unmatched_bracket_does_not_crash():
    vm = CircularTapeVM(VMConfig(max_steps=100))
    for p in (prog("[", "+"), prog("]", "+"), prog("[", "[", "]")):
        res = vm.execute(p)
        assert res.steps >= 1


def test_output_budget_is_enforced():
    vm = CircularTapeVM(VMConfig(max_output=8))
    res = vm.execute(prog("+", "+", "+", ".", ".", ".", ".", ".", ".", ".", ".", ".", "."))
    assert res.output_len <= 8


def test_pointer_wraps_modulo_tape_size():
    """Left of cell 0 wraps to the last cell; the value wraps mod 256."""
    vm = CircularTapeVM(VMConfig(tape_size=8))
    # move left once from cell 0 -> cell 7, set to 1, then move right twice -> cell 1
    res = vm.execute(prog("<", "+", ">", ">", "."))
    assert res.tape[7] == 1
    assert res.output == (0,)


def test_value_wraps_modulo_256():
    vm = CircularTapeVM(VMConfig(max_steps=1000))
    res = vm.execute(prog(*(["-"] * 3), "."))
    assert res.output == (253,)  # 0 - 3 mod 256


# ======================================================================================
# semantics of the 8 Brainfuck tokens
# ======================================================================================


def test_brainfuck_increment_and_emit():
    res = CircularTapeVM().execute(prog("+", "+", "+", "."))
    assert res.output == (3,)


def test_brainfuck_input_reads_stream():
    vm = CircularTapeVM(VMConfig(input_stream=(7, 9)))
    res = vm.execute(prog(",", ".", ",", "."))
    assert res.output == (7, 9)


def test_brainfuck_loop_executes_body():
    """`++[-.]` : cell 2 -> countdown emitting 1 then 0."""
    res = CircularTapeVM().execute(prog("+", "+", "[", "-", ".", "]"))
    assert res.output[0] == 1
    assert res.output[-1] == 0


def test_brainfuck_loop_skipped_when_zero():
    res = CircularTapeVM().execute(prog("[", "+", "]", "."))
    assert res.output == (0,)


# ======================================================================================
# semantics of the 10 VSA macros
# ======================================================================================


def test_macro_z_clears_cell():
    res = CircularTapeVM().execute(prog("+", "+", "Z", "."))
    assert res.output == (0,)


def test_macro_x_sets_sixteen():
    res = CircularTapeVM().execute(prog("X", "."))
    assert res.output == (16,)


def test_macro_c_copies_to_next():
    res = CircularTapeVM().execute(prog("+", "+", "+", "C", ">", "."))
    assert res.output == (3,)


def test_macro_n_subtracts_left_neighbour():
    # cell0=5, move right, cell1 = (0 - 5) mod 256 = 251
    res = CircularTapeVM().execute(prog(*(["+"] * 5), ">", "N", "."))
    assert res.output == (251,)


def test_macro_r_rolls_right():
    # p=1, cell1=4 ; R shifts each cell to index+1, so the 4 lands at index 2.
    res = CircularTapeVM().execute(prog(">", *(["+"] * 4), "R", ">", "."))
    assert res.output == (4,)


def test_macro_l_rolls_left():
    # cell2 = 7 ; roll left twice brings it to index 0
    res = CircularTapeVM().execute(prog(">", ">", *(["+"] * 7), "L", "L", "<", "<", "."))
    assert res.output == (7,)


def test_macro_g_and_h_move_pointer_by_cell():
    res_g = CircularTapeVM(VMConfig(tape_size=16)).execute(prog(*(["+"] * 3), "G", "."))
    assert res_g.tape[3] == 0  # pointer moved to a fresh cell
    res_h = CircularTapeVM(VMConfig(tape_size=16)).execute(prog(*(["+"] * 3), "H", "."))
    assert res_h.tape[13] == 0


def test_macro_w_skips_next_instruction_when_zero():
    """W with a ZERO cell skips the following instruction: X never runs."""
    res = CircularTapeVM().execute(prog("W", "X", "."))
    assert res.output == (0,)


def test_macro_w_does_not_skip_when_nonzero():
    """W with a NON-zero cell executes the following instruction: X sets 16."""
    res = CircularTapeVM().execute(prog("+", "W", "X", "."))
    assert res.output == (16,)



def test_macro_v_emits_repeated_byte():
    vm = CircularTapeVM(VMConfig(max_output=64))
    res = vm.execute(prog("+", "+", "+", "V"))
    assert res.output == (3, 3, 3)


# ======================================================================================
# determinism and batching
# ======================================================================================


def test_deterministic_replay():
    p = prog("+", "+", "[", "-", ".", "]", "X", ".", "V")
    vm = CircularTapeVM()
    a = vm.execute(p)
    b = vm.execute(p)
    assert a.output == b.output
    assert a.tape == b.tape
    assert a.steps == b.steps


def test_batch_matches_sequential():
    rng = torch.Generator().manual_seed(3)
    programs = [sample_program(12, rng) for _ in range(8)]
    vm = CircularTapeVM(VMConfig(max_steps=200))
    batch = vm.execute_batch(programs)
    for p, got in zip(programs, batch):
        want = vm.execute(p)
        assert got.output == want.output
        assert got.steps == want.steps


def test_sample_program_is_in_alphabet_and_deterministic():
    rng1 = torch.Generator().manual_seed(5)
    rng2 = torch.Generator().manual_seed(5)
    a = sample_program(30, rng1)
    b = sample_program(30, rng2)
    assert a == b
    assert all(0 <= t < 18 for t in a)
    assert len(a) == 30


def test_sample_program_rejects_bad_length():
    with pytest.raises(VMError):
        sample_program(0)


def test_random_programs_never_crash_or_hang():
    """Fuzz: 200 random programs must all terminate within the step budget."""
    rng = torch.Generator().manual_seed(11)
    vm = CircularTapeVM(VMConfig(max_steps=300, max_output=64))
    for _ in range(200):
        res = vm.execute(sample_program(24, rng))
        assert res.steps <= 301
        assert res.output_len <= 65  # max_output + at most one TIMEOUT symbol


def test_default_input_stream_is_deterministic_and_valid():
    s1 = default_input_stream(8)
    s2 = default_input_stream(8)
    assert s1 == s2
    assert len(s1) == 8
    assert all(0 <= b < 256 for b in s1)
    with pytest.raises(VMError):
        default_input_stream(0)
