"""Stage 0 — universal self-play seeder: circular byte-tape VM (CPU reference).

Protocol authority: `HENRI-ARCH-2026-SELFPLAY-DREAMING-V1` (.md), MILESTONE 2.

The .md specifies:
  * an augmented 18-token alphabet = 8 classical Brainfuck instructions
    { <, >, +, -, [, ], ., , } plus 10 VSA macro primitives
    { Z, R, L, N, C, G, H, W, V, X };
  * a sandboxed circular byte-tape VM with GUARANTEED execution
    (out-of-bounds wraps modulo 256; zero syntax crashes);
  * a self-play loop in which the generator earns the M1 preconditioned
    gradient-alignment reward.

Two honest boundaries:

1. The .md NAMES the 10 VSA macros but does not specify their semantics. This
   module DEFINES them (see MACRO_SEMANTICS) and labels that definition
   `HYPOTHESIS`. A different definition is a different seeder.
2. The .md asks for 10 billion tokens. That is a SCHEDULED REMOTE JOB, not a
   first run. This module provides the mechanism and a bounded seeding run; the
   scale decision follows a MEASURED throughput number from the target card.

This is the CPU reference implementation. It is the oracle a CUDA kernel must
match bit-exactly. It is NOT CUDA verification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import torch

__all__ = [
    "VMError",
    "ALPHABET",
    "BF_TOKENS",
    "VSA_TOKENS",
    "MACRO_SEMANTICS",
    "TIMEOUT_TOKEN",
    "VMConfig",
    "ProgramResult",
    "CircularTapeVM",
    "sample_program",
    "default_input_stream",
]

# --------------------------------------------------------------------------------------
# alphabet
# --------------------------------------------------------------------------------------

BF_TOKENS: tuple[str, ...] = ("<", ">", "+", "-", "[", "]", ".", ",")
VSA_TOKENS: tuple[str, ...] = ("Z", "R", "L", "N", "C", "G", "H", "W", "V", "X")
ALPHABET: tuple[str, ...] = BF_TOKENS + VSA_TOKENS  # 18 tokens

assert len(BF_TOKENS) == 8 and len(VSA_TOKENS) == 10, "alphabet must be 8 + 10 = 18"

# Reserved output symbol for a program that hit the step budget. Its existence is what
# makes execution TOTAL: a diverging `[...]` loop yields TIMEOUT, never a hang.
TIMEOUT_TOKEN: int = 256

MACRO_SEMANTICS: dict[str, str] = {
    "Z": "clear:  cell[p] = 0",
    "R": "roll-R: rotate the whole tape right by one cell",
    "L": "roll-L: rotate the whole tape left by one cell",
    "N": "sub-L:  cell[p] = (cell[p] - cell[p-1]) mod 256",
    "C": "copy-2: cell[p+1] = cell[p]",
    "G": "scan-R: p = (p + cell[p]) mod T",
    "H": "scan-L: p = (p - cell[p]) mod T",
    "W": "gate-branch: if cell[p] == 0, skip the next instruction",
    "V": "emit-str: emit cell[p] copies of the byte cell[p]",
    "X": "set-16: cell[p] = 16",
}

TOKEN_INDEX: dict[str, int] = {t: i for i, t in enumerate(ALPHABET)}


class VMError(ValueError):
    """Raised on a malformed program (wrong token range or shape)."""


# --------------------------------------------------------------------------------------
# configuration and result
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class VMConfig:
    """Static VM settings. All bounded, so execution is total.

    Attributes:
        tape_size: number of cells. Must be >= 1.
        max_steps: instruction budget per program. Must be >= 1. A program that
            exceeds it returns TIMEOUT instead of hanging.
        max_output: output-length budget. Must be >= 1.
        input_stream: bytes fed to the `,` instruction. Cycled when exhausted.
    """

    tape_size: int = 256
    max_steps: int = 4096
    max_output: int = 256
    input_stream: tuple[int, ...] = (0,)

    def validate(self) -> "VMConfig":
        for name in ("tape_size", "max_steps", "max_output"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise VMError(f"{name} must be an int >= 1, got {value!r}")
        for b in self.input_stream:
            if not isinstance(b, int) or isinstance(b, bool) or not (0 <= b < 256):
                raise VMError(f"input byte out of range: {b!r}")
        return self


@dataclass(frozen=True)
class ProgramResult:
    """Outcome of executing one program."""

    output: tuple[int, ...]
    steps: int
    timed_out: bool
    tape: tuple[int, ...] = field(default=())

    @property
    def output_len(self) -> int:
        return len(self.output)


# --------------------------------------------------------------------------------------
# the VM
# --------------------------------------------------------------------------------------


class CircularTapeVM:
    """Sandboxed circular byte-tape VM with guaranteed execution.

    Every pointer move wraps modulo `tape_size`; every cell value wraps modulo 256.
    No instruction can fault, so no program can crash the VM. A program that
    exceeds `max_steps` returns `TIMEOUT_TOKEN` in its output and `timed_out=True`.
    """

    def __init__(self, config: VMConfig | None = None) -> None:
        self.config = (config or VMConfig()).validate()

    # ---------------------------------------------------------------- validation
    def _validate_program(self, program: Sequence[int]) -> list[int]:
        if isinstance(program, torch.Tensor):
            if program.dim() != 1:
                raise VMError(f"program tensor must be rank 1, got rank {program.dim()}")
            program = program.tolist()
        if not isinstance(program, (list, tuple)):
            raise VMError(f"program must be a list/tuple/tensor, got {type(program).__name__}")
        out: list[int] = []
        n_tokens = len(ALPHABET)
        for i, tok in enumerate(program):
            if not isinstance(tok, int) or isinstance(tok, bool):
                raise VMError(f"program[{i}] must be an int, got {type(tok).__name__}")
            if not (0 <= tok < n_tokens):
                raise VMError(f"program[{i}]={tok} outside alphabet range [0, {n_tokens})")
            out.append(tok)
        return out

    # ---------------------------------------------------------------- execution
    def execute(self, program: Sequence[int]) -> ProgramResult:
        """Execute a program. Total: always returns, never raises on program content."""
        prog = self._validate_program(program)
        cfg = self.config
        T = cfg.tape_size

        tape = [0] * T
        p = 0
        pc = 0
        steps = 0
        out: list[int] = []
        timed_out = False
        in_idx = 0
        n = len(prog)

        # Precompute Brainfuck bracket matching. An unmatched bracket cannot crash:
        # it simply targets the program end, which terminates the loop.
        jump: dict[int, int] = {}
        stack: list[int] = []
        for i, tok in enumerate(prog):
            if ALPHABET[tok] == "[":
                stack.append(i)
            elif ALPHABET[tok] == "]":
                if stack:
                    j = stack.pop()
                    jump[j] = i
                    jump[i] = j
        # unmatched brackets fall through to the end of the program

        while pc < n:
            if steps >= cfg.max_steps:
                timed_out = True
                out.append(TIMEOUT_TOKEN)
                break
            steps += 1
            sym = ALPHABET[prog[pc]]

            if sym == "<":
                p = (p - 1) % T
            elif sym == ">":
                p = (p + 1) % T
            elif sym == "+":
                tape[p] = (tape[p] + 1) & 0xFF
            elif sym == "-":
                tape[p] = (tape[p] - 1) & 0xFF
            elif sym == "[":
                if tape[p] == 0:
                    pc = jump.get(pc, n - 1)
            elif sym == "]":
                if tape[p] != 0:
                    pc = jump.get(pc, n - 1)
            elif sym == ".":
                out.append(tape[p])
                if len(out) >= cfg.max_output:
                    break
            elif sym == ",":
                if cfg.input_stream:
                    tape[p] = cfg.input_stream[in_idx % len(cfg.input_stream)]
                    in_idx += 1
            elif sym == "Z":
                tape[p] = 0
            elif sym == "R":
                tape = tape[-1:] + tape[:-1]
            elif sym == "L":
                tape = tape[1:] + tape[:1]
            elif sym == "N":
                tape[p] = (tape[p] - tape[(p - 1) % T]) & 0xFF
            elif sym == "C":
                tape[(p + 1) % T] = tape[p]
            elif sym == "G":
                p = (p + tape[p]) % T
            elif sym == "H":
                p = (p - tape[p]) % T
            elif sym == "W":
                if tape[p] == 0:
                    pc += 1  # skip the next instruction
            elif sym == "V":
                k = tape[p]
                for _ in range(k):
                    if len(out) >= cfg.max_output:
                        break
                    out.append(tape[p])
            elif sym == "X":
                tape[p] = 16
            else:  # pragma: no cover - alphabet is exhaustive
                raise VMError(f"unreachable token {sym!r}")

            pc += 1

        return ProgramResult(
            output=tuple(out),
            steps=steps,
            timed_out=timed_out,
            tape=tuple(tape),
        )

    # ---------------------------------------------------------------- batch
    def execute_batch(self, programs: Sequence[Sequence[int]]) -> list[ProgramResult]:
        """Execute many programs. Each is independent and total."""
        return [self.execute(p) for p in programs]


# --------------------------------------------------------------------------------------
# sampling helpers
# --------------------------------------------------------------------------------------


def sample_program(n_tokens: int, rng: torch.Generator | None = None) -> list[int]:
    """Sample a uniformly random program over the 18-token alphabet."""
    if not isinstance(n_tokens, int) or isinstance(n_tokens, bool) or n_tokens < 1:
        raise VMError(f"n_tokens must be an int >= 1, got {n_tokens!r}")
    idx = torch.randint(0, len(ALPHABET), (n_tokens,), generator=rng)
    return idx.tolist()


def default_input_stream(length: int = 8) -> tuple[int, ...]:
    """Return a deterministic non-trivial input stream for the `,` instruction."""
    if length < 1:
        raise VMError(f"length must be >= 1, got {length}")
    return tuple(((i * 37) + 11) & 0xFF for i in range(length))
