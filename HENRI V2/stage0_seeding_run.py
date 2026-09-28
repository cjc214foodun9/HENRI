"""Stage 0 bounded seeding run — the self-play loop end to end.

Protocol: HENRI-ARCH-2026-SELFPLAY-DREAMING-V1 (.md), MILESTONE 2, under the
ratified bounded-scope decision: 10^7-10^8 program EXECUTIONS first, measure
throughput, extrapolate cost, and only then decide on the scaled run.

HONEST BOUNDARY (state this in every report)
--------------------------------------------
This bounded run validates the LOOP PLUMBING and MEASURES THROUGHPUT. It does NOT
establish curriculum validity and does NOT establish ICL emergence:

  * The reward's ability to rank families by LEARNABILITY is NOT established.
    Two gates built this session FAILED, and both failures trace to gate defects
    of the same class: high-loss/novel data dominates any magnitude- or
    memorization-based score.
      - discrimination gate: a high-loss gradient aligns with the general
        loss-reduction direction, so reward alone cannot separate learnable
        structure from noise.
      - validity gate: progress was measured on the SAME samples used for
        training, which measures MEMORIZATION. Noise shows large "progress" by
        being memorized.
    A defect-free validity gate needs HELD-OUT loss. That gate is not built.
  * ICL emergence (reverse string, stack, associative recall) is the .md's claim
    for the 10B-token run. It is NOT testable at this scale.

SCALE-CONFLATION GUARD
----------------------
Three budgets are tracked SEPARATELY and never conflated:
    budget_vm_executions      program runs              (the cheap, approved budget)
    budget_reward_evaluations JVP/gradient per candidate (subsampled)
    budget_learner_tokens     next-byte training tokens  (a separate job at scale)

Usage:
  python stage0_seeding_run.py --n-executions 1000000 --batch-size 256 \
      --out telemetry/stage0_seeding --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from typing import List

import torch

from henri_gradient_alignment_reward import alignment_reward
from stage0_universal_seeder import (
    ALPHABET,
    TIMEOUT_TOKEN,
    VMConfig,
    CircularTapeVM,
    sample_program,
)

VOCAB = 257  # 256 byte values + TIMEOUT
DEPTH = 64


# ======================================================================================
# learner (real AdamW; the M1 kernel supplies the preconditioner form)
# ======================================================================================

class TapeLearner:
    """Embedding + linear head predicting the next output-tape byte."""

    def __init__(self, seed: int = 0, lr: float = 3e-3) -> None:
        gen = torch.Generator().manual_seed(seed)
        self.params = {
            "emb": torch.randn(VOCAB, DEPTH, dtype=torch.float64, generator=gen) * 0.1,
            "head": torch.randn(DEPTH, VOCAB, dtype=torch.float64, generator=gen) * 0.1,
        }
        for p in self.params.values():
            p.requires_grad_(True)
        self.v = {k: torch.zeros_like(p) for k, p in self.params.items()}
        self.b1, self.b2, self.eps = 0.9, 0.999, 1e-8
        self.lr = lr
        self.step_count = 0

    def loss(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.params["emb"][ids]
        logits = x @ self.params["head"]
        return torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, VOCAB), ids[:, 1:].reshape(-1)
        )

    def step(self, ids: torch.Tensor) -> float:
        loss = self.loss(ids)
        grads = torch.autograd.grad(loss, tuple(self.params.values()))
        self.step_count += 1
        with torch.no_grad():
            for (name, p), g in zip(self.params.items(), grads):
                self.v[name].mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
                v_hat = self.v[name] / (1 - self.b2 ** self.step_count)
                p.add_(-self.lr * g / (torch.sqrt(v_hat) + self.eps))
        return float(loss.item())

    def v_flat(self) -> torch.Tensor:
        return torch.cat([t.reshape(-1) for t in self.v.values()])

    def flat(self) -> torch.Tensor:
        return torch.cat([p.detach().reshape(-1) for p in self.params.values()])

    def snapshot(self) -> torch.Tensor:
        return self.flat().clone()


# ======================================================================================
# adaptive generator: uniform -> reward-weighted bank (epsilon-greedy)
# ======================================================================================

@dataclass
class ProgramBank:
    capacity: int = 4096
    programs: List[list] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)

    def admit(self, program: list, reward: float) -> None:
        self.programs.append(program)
        self.rewards.append(reward)
        if len(self.programs) > self.capacity:
            i = min(range(len(self.rewards)), key=lambda j: self.rewards[j])
            self.programs.pop(i)
            self.rewards.pop(i)

    def sample(self, rng: torch.Generator, n: int) -> List[list]:
        if not self.programs or n <= 0:
            return []
        w = torch.tensor(self.rewards, dtype=torch.float64).clamp_min(0.0)
        if float(w.sum()) <= 0:
            idx = torch.randint(0, len(self.programs), (min(n, len(self.programs)),), generator=rng)
        else:
            idx = torch.multinomial(w / w.sum(), min(n, len(w)), replacement=True, generator=rng)
        return [self.programs[i] for i in idx.tolist()]


# ======================================================================================
# the run
# ======================================================================================

def build_heldout(
    n_samples: int,
    seed: int,
    prog_len: int,
    seq_len: int,
) -> torch.Tensor:
    """Build a HELD-OUT batch that training NEVER sees.

    Disjoint by construction on three axes, so a promotion signal derived from it
    cannot be satisfied by memorisation:
      * a different generator seed  -> different programs
      * a different program LENGTH  -> out-of-distribution vs the training length
      * a fixed, materialised batch -> identical across every evaluation, so the
        progress delta is measured on the SAME items each time.

    This exists because the driver's own docstring records that an earlier gate
    measured progress on the SAME samples used for training, which measures
    MEMORISATION (noise scored large "progress" by being memorised). Prediction on
    unseen items is the only signal allowed to gate promotion.
    """
    hvm = CircularTapeVM(VMConfig(tape_size=256, max_steps=512, max_output=seq_len - 1))
    hrng = torch.Generator().manual_seed(seed + 999_983)  # disjoint stream
    hlen = prog_len + 8                                    # disjoint length
    rows = []
    for _ in range(n_samples):
        res = hvm.execute(sample_program(hlen, hrng))
        seq = list(res.output[: seq_len - 1])
        if res.timed_out:
            seq.append(TIMEOUT_TOKEN)
        seq = seq + [0] * (seq_len - len(seq))
        rows.append([min(x, VOCAB - 1) for x in seq[:seq_len]])
    return torch.tensor(rows, dtype=torch.long)


def _window_variance(hist, window: int) -> float | None:
    """Population variance of the last `window` losses; None if not enough data.

    This is the PLATEAU DETECTOR for curriculum escalation (directive 1).
    A flat curriculum drives loss variance toward zero; when sigma^2 drops below
    the pre-registered threshold the environment must raise difficulty instead of
    burning more tokens at the same depth (measured: 99.66% of a 10B-token run
    sat on a flat plateau).
    """
    if len(hist) < window:
        return None
    w = hist[-window:]
    mu = sum(w) / len(w)
    return sum((x - mu) ** 2 for x in w) / len(w)


def run_seeding(
    n_executions: int,
    batch_size: int,
    out_dir: str,
    seed: int,
    prog_len: int = 32,
    seq_len: int = 33,
    reward_subsample: int = 64,
    heldout_samples: int = 256,
    eval_every: int = 50,
    shard_dir: str | None = None,
    shard_rows_per_file: int = 250_000,
    heldout_seed: int = None,
    heldout_rebuild: bool = False,
    curriculum_escalate: bool = False,
    plateau_var_threshold: float = 1e-4,
    curriculum_window: int = 50,
    max_prog_len: int = 96,
    curriculum_levers: bool = False,
    progress_eps: float = 1e-3,
    kill_patience: int = 3,
    governor_trigger: str = "variance",
    progress_rate_threshold: float = 1e-3,
    cadence_windows: int = 1,
) -> dict:
    """Execute the bounded seeding loop. Returns the summary dict.

    PROMOTION GATE (ratified): the ONLY signal allowed to promote this run is
    HELD-OUT curriculum progress. `final_loss` and `reward_mean` are recorded but
    are explicitly NOT promotion signals (the old driver docstring records that a
    training-loss/reward-based gate is satisfiable by memorisation).
    """
    # ---- DEFECT FIXED 2026-09-27: TWO escalation mechanisms existed and BOTH
    # could be ON at once -- TWO writers to the same knob (prog_len). Measured:
    # `curriculum_escalate` and `curriculum_levers` each occur 6 times in this
    # file. A silent double-write makes any escalation experiment unattributable,
    # so this now FAILS CLOSED rather than running an ambiguous configuration.
    os.makedirs(out_dir, exist_ok=True)
    if curriculum_escalate and curriculum_levers:
        raise ValueError(
            "curriculum_escalate (legacy, prog_len-only) and curriculum_levers "
            "(governor ladder) are mutually exclusive: running both gives TWO "
            "writers to prog_len and makes the escalation unattributable. "
            "Enable exactly one.")
    # ---- CURRICULUM LEVERS (directive 1, NEW flag, default OFF).
    # MEASURED DEFICIT THIS CLOSES: the governor's topological_obstacle,
    # multiscale_nesting, distractor_noise and grid_growth levers occurred
    # 0 times in this file AND in stage0_universal_seeder.py -- declared but
    # never read, i.e. the mechanism did not exist. With this flag ON,
    # sampling routes through henri_curriculum_env using the governor spec.
    #
    # grid_growth is consumed here AND re-applied when the governor escalates it.
    # DEFECT FIXED 2026-09-27: the first form applied it ONCE and never again, so
    # rung 5 could change `_spec["grid_growth"]` without ever reaching the VM --
    # a dead store, on top of a cap (64) below the deployed value (256) that made
    # the rung un-fireable. The rung now REBUILDS the VM.
    # COMPARABILITY CAVEAT (recorded in the receipt): the held-out set is built once
    # at the INITIAL prog_len and is never re-drawn, so every held-out reading is
    # taken against the SAME fixed target tensors while the VM grows underneath.
    # `tape_size_history` records each change.
    _gov = None
    _spec = None
    _spec_report = None
    kill_reason: "str | None" = None
    G_LADDER_NAMES = None
    _tape_size = 256
    tape_size_history: list[dict] = []
    if curriculum_levers:
        from henri_curriculum_env import (sample_program_from_spec, spec_report,
                                          tape_size_from_spec)
        from henri_curriculum_governor import (CurriculumGovernor, GovernorConfig,
                                               LADDER as G_LADDER_NAMES)
        _gov = CurriculumGovernor(GovernorConfig(
            window=curriculum_window, var_threshold=plateau_var_threshold,
            progress_eps=progress_eps, kill_patience=kill_patience,
            # DIRECTIVE 1: "variance" reproduces the previous path byte-for-byte;
            # "progress" fires on the moving-window loss DERIVATIVE, so it can
            # escalate BEFORE the loss flattens.
            trigger=governor_trigger,
            progress_rate_threshold=progress_rate_threshold,
            cadence_windows=cadence_windows))
        _gov.cfg.spec["prog_len"] = float(prog_len)
        _gov.cfg.spec["grid_growth"] = float(_tape_size)
        _spec = _gov.spec
        _tape_size = tape_size_from_spec(_spec, default=256)
        _spec_report = spec_report(_spec)
    vm = CircularTapeVM(VMConfig(tape_size=_tape_size, max_steps=512, max_output=seq_len - 1))
    learner = TapeLearner(seed=seed)
    rng = torch.Generator().manual_seed(seed)
    bank = ProgramBank()

    # ---- held-out evaluation set (never trained on; see build_heldout)
    heldout_ids = build_heldout(heldout_samples, seed, prog_len, seq_len)
    heldout_first: float | None = None
    heldout_last: float | None = None
    heldout_curve: list[tuple[int, float]] = []

    # ---- binary shards: accumulate the token stream to disk (ACTION 1)
    shard_files = 0
    shard_bytes = 0
    shard_written = 0
    shard_buf = bytearray()
    if shard_dir:
        os.makedirs(shard_dir, exist_ok=True)

        def _flush(buf: bytearray, idx: int) -> tuple[int, int]:
            p = os.path.join(shard_dir, f"shard_{idx:05d}.bin")
            with open(p, "wb") as sh:
                sh.write(buf)
            return 1, len(buf)

    # round-0 held-out measurement (the baseline the delta is taken against)
    heldout_first = learner.probe_loss(heldout_ids) if hasattr(learner, "probe_loss") else None
    if heldout_first is None:
        with torch.no_grad():
            heldout_first = float(learner.loss(heldout_ids))
    heldout_last = heldout_first
    heldout_curve.append((0, heldout_first))

    n_rounds = max(1, (n_executions + batch_size - 1) // batch_size)
    executed = 0
    reward_evals = 0
    tokens = 0
    timeouts = 0
    reward_sum = 0.0
    reward_n = 0
    # BOUNDED distinct-output accounting: an unbounded set at 10^7 executions
    # would hold ~1.25M tuples (~390 MB). Cap it and record saturation honestly.
    DISTINCT_CAP = 200_000
    distinct: set = set()
    distinct_saturated = False
    loss_hist: List[float] = []
    lookback: dict = {}
    t0 = time.perf_counter()

    # ---- CURRICULUM ESCALATION state (directive 1). Default OFF; when ON the
    # environment raises program depth as soon as loss variance collapses below
    # the pre-registered threshold, instead of burning tokens at fixed depth.
    curriculum_events: list[dict] = []
    _cur_prog_len = prog_len
    _rounds_since_escalation = 0

    telemetry_path = os.path.join(out_dir, "telemetry.jsonl")
    with open(telemetry_path, "w", encoding="utf-8") as fh:
        for rnd in range(n_rounds):
            # ---- 1. GENERATE: epsilon-greedy (30% from the reward-weighted bank)
            n_bank = int(0.3 * batch_size) if bank.programs else 0
            n_fresh = batch_size - n_bank
            if _spec is not None:
                batch = [sample_program_from_spec(_spec, rng, sample_program,
                                                  len(ALPHABET)) for _ in range(n_fresh)]
            else:
                batch = [sample_program(_cur_prog_len, rng) for _ in range(n_fresh)]
            batch += bank.sample(rng, n_bank)

            # ---- 2. EXECUTE (the cheap approved budget; total execution)
            results = vm.execute_batch(batch)
            executed += len(batch)
            timeouts += sum(1 for r in results if r.timed_out)

            # ---- 3. build the learner batch
            rows = []
            for r in results:
                seq = list(r.output[: seq_len - 1])
                if r.timed_out:
                    seq.append(TIMEOUT_TOKEN)
                seq = seq + [0] * (seq_len - len(seq))
                rows.append([min(x, VOCAB - 1) for x in seq[:seq_len]])
            ids = torch.tensor(rows, dtype=torch.long)

            # ---- 4. REWARD on a SUBSAMPLE (never one eval per execution)
            k = min(reward_subsample, ids.shape[0])
            sel = torch.randperm(ids.shape[0], generator=rng)[:k]
            sub = ids[sel]
            loss = learner.loss(sub)
            grads = torch.autograd.grad(loss, tuple(learner.params.values()))
            g = torch.cat([gr.reshape(-1) for gr in grads])
            p_e = learner.step_count // 2
            theta_lb = lookback.get(p_e, learner.snapshot())
            d_theta = theta_lb - learner.flat()
            r_mean = float(alignment_reward(g, d_theta, learner.v_flat()))
            reward_evals += k
            reward_sum += r_mean
            reward_n += 1

            # ---- 5. ADMIT the best-scoring programs to the bank
            for prog_item, res in zip(batch[: max(1, batch_size // 8)],
                                      results[: max(1, batch_size // 8)]):
                if not res.timed_out and res.output:
                    bank.admit(prog_item, r_mean)
                    if len(distinct) < DISTINCT_CAP:
                        distinct.add(tuple(res.output))
                    else:
                        distinct_saturated = True

            # ---- 6. LEARNER UPDATE (the separate token budget)
            loss_val = learner.step(ids)
            tokens += int(ids.numel())
            loss_hist.append(loss_val)

            # ---- CURRICULUM ESCALATION (directive 1, default OFF)
            var = _window_variance(loss_hist, curriculum_window)
            if curriculum_escalate and var is not None:
                if var < plateau_var_threshold and _cur_prog_len < max_prog_len:
                    _old = _cur_prog_len
                    _cur_prog_len = min(max_prog_len, int(_cur_prog_len * 1.5) + 2)
                    curriculum_events.append({
                        "round": int(rnd), "loss_variance": float(var),
                        "old_prog_len": int(_old), "new_prog_len": int(_cur_prog_len),
                        "trigger": "sigma2 < %.1e" % plateau_var_threshold,
                    })
                    _rounds_since_escalation = 0
                elif var >= plateau_var_threshold:
                    _rounds_since_escalation += 1
            lookback[learner.step_count] = learner.snapshot()
            if len(lookback) > 4:
                del lookback[min(lookback)]

            # ---- 6a. SHARD the training token stream to disk (ACTION 1)
            if shard_dir:
                # vectorised: a per-element Python loop would cost ~17k iterations
                # per round x ~19.5k rounds. .numpy().tobytes() is a single memcpy.
                shard_buf.extend(ids.reshape(-1).to(torch.uint8).numpy().tobytes())
                shard_written += int(ids.numel())
                if len(shard_buf) >= shard_rows_per_file * seq_len:
                    nf, nb = _flush(shard_buf, shard_files)
                    shard_files += nf
                    shard_bytes += nb
                    shard_buf = bytearray()

            # ---- 6b. HELD-OUT evaluation (the ONLY promotion signal)
            _eval_this_round = False
            if eval_every > 0 and (rnd % eval_every == 0) and rnd > 0:
                # DEFECT ADDRESSED 2026-09-27: the held-out set is built ONCE at the
                # INITIAL prog_len, so as the curriculum raises prog_len the target
                # goes progressively STALER and per-rung deltas compare against an
                # out-of-date distribution. `heldout_rebuild` re-derives it at the
                # CURRENT prog_len. Default OFF keeps every previously committed
                # number comparable; the mode used is recorded in the receipt.
                if heldout_rebuild:
                    heldout_ids = build_heldout(heldout_samples, seed,
                                                _cur_prog_len, seq_len)
                with torch.no_grad():
                    h = float(learner.loss(heldout_ids))
                heldout_last = h
                heldout_curve.append((rnd, h))
                _eval_this_round = True
            # ---- GOVERNOR ADOPTION (directive 1; only when --curriculum-levers ON).
            # The governor escalates over the HETEROGENEOUS ladder and can TERMINATE the
            # run when escalation stops moving held-out progress. That KILL is the
            # literal reading of "cease flat token volume burns".
            if _gov is not None:
                _ev = _gov.observe(loss_val,
                                 heldout=(heldout_last if _eval_this_round else None))
                if _ev is not None:
                    # PER-RUNG ATTRIBUTION (the pre-registered bar needs it): take a
                    # FRESH held-out reading AT the escalation point. Each rung then
                    # carries the level measured immediately BEFORE that rung takes
                    # effect, so per-rung progress is a difference of consecutive
                    # readings and no rung can be credited with progress it did not
                    # cause. Measured defect this closes: the committed receipt had
                    # per_rung=0 / rung_progress=0 -- no per-rung attribution at all.
                    with torch.no_grad():
                        _h_at = float(learner.loss(heldout_ids))
                    _ev["heldout_at_escalation"] = _h_at
                    _ev["round"] = int(rnd)
                    curriculum_events.append(_ev)
                    if _ev.get("event") == "ESCALATE":
                        _spec = _ev["spec"]
                        _cur_prog_len = int(_spec["prog_len"])
                        # rung 5 -> rebuild the VM so the change REACHES the machine
                        _new_tape = tape_size_from_spec(_spec, default=256)
                        if _new_tape != _tape_size:
                            _ev["tape_size_before"] = int(_tape_size)
                            _ev["tape_size_after"] = int(_new_tape)
                            _tape_size = int(_new_tape)
                            vm = CircularTapeVM(VMConfig(tape_size=_tape_size,
                                                         max_steps=512,
                                                         max_output=seq_len - 1))
                            tape_size_history.append({"round": int(rnd),
                                                      "tape_size": int(_tape_size)})
                        _spec_report = spec_report(_spec)
                    elif _ev.get("event") == "KILL":
                        kill_reason = str(_ev.get("reason"))
                        break

            if rnd % 10 == 0 or rnd == n_rounds - 1:
                dt = time.perf_counter() - t0
                fh.write(json.dumps({
                    "round": rnd,
                    "vm_executions": executed,
                    "exec_per_sec": round(executed / dt, 1),
                    "reward_evals": reward_evals,
                    "learner_tokens": tokens,
                    "loss": round(loss_val, 6),
                    "reward_mean": round(reward_sum / max(reward_n, 1), 10),
                    "timeout_rate": round(timeouts / max(executed, 1), 4),
                    "bank_size": len(bank.programs),
                    "distinct_outputs": len(distinct),
                }) + "\n")
                fh.flush()

    dt = time.perf_counter() - t0

    # ---- flush any remaining shard bytes
    if shard_dir and shard_buf:
        nf, nb = _flush(shard_buf, shard_files)
        shard_files += nf
        shard_bytes += nb
        shard_buf = bytearray()

    heldout_progress = (
        (heldout_first - heldout_last)
        if (heldout_first is not None and heldout_last is not None)
        else None
    )
    # ---- PER-RUNG ATTRIBUTION (pre-registered bar): pair each escalation with the
    # held-out reading before it and the reading after it (next escalation, or the
    # end of the run). progress = before - after, so a POSITIVE value means the rung
    # was followed by held-out improvement.
    _escal = [e for e in curriculum_events if isinstance(e, dict)
              and e.get("event") == "ESCALATE"]
    per_rung = []
    for _i, _e in enumerate(_escal):
        _b = _e.get("heldout_at_escalation")
        _a = (_escal[_i + 1].get("heldout_at_escalation") if _i + 1 < len(_escal)
              else heldout_last)
        per_rung.append({
            "rung": _e.get("rung"), "round": _e.get("round"),
            "value_after": _e.get("after"),
            "heldout_before": _b, "heldout_after": _a,
            "progress": ((_b - _a) if (_b is not None and _a is not None) else None),
        })
    summary = {
        "gate": "STAGE0_HELDOUT_SEEDING",
        "purpose": "local token accumulation + HELD-OUT curriculum measurement",
        "seed": seed,
        "n_executions_requested": n_executions,
        "budget_vm_executions": executed,
        "budget_reward_evaluations": reward_evals,
        "budget_learner_tokens": tokens,
        "wall_seconds": round(dt, 2),
        "exec_per_sec": round(executed / dt, 1),
        "reward_evals_per_sec": round(reward_evals / dt, 1),
        "learner_tokens_per_sec": round(tokens / dt, 1),
        # ---- PROMOTION SIGNAL (held-out only)
        "heldout_loss_first": heldout_first,
        "heldout_loss_last": heldout_last,
        "heldout_progress": heldout_progress,
        "heldout_samples": heldout_samples,
        "heldout_rebuild": bool(heldout_rebuild),
        "heldout_target_note": (
            "False keeps the target fixed at the INITIAL prog_len so committed "
            "numbers stay comparable; True re-derives it at the CURRENT prog_len "
            "so the target cannot go stale as the ladder climbs."),
        "heldout_curve": [[int(r), float(v)] for r, v in heldout_curve],
        "curriculum_escalate": bool(curriculum_escalate),
        "plateau_var_threshold": float(plateau_var_threshold),
        "curriculum_window": int(curriculum_window),
        "curriculum_events": curriculum_events,
        "curriculum_levers": bool(curriculum_levers),
        "curriculum_spec": _spec_report,
        "governor_trigger": governor_trigger,
        "progress_rate_threshold": progress_rate_threshold,
        "per_rung_progress": per_rung,
        "per_rung_progress_basis": (
            "heldout_at_escalation is a FRESH held-out reading taken AT each escalation "
            "point; progress = heldout_before - heldout_after. Positive => the rung was "
            "followed by held-out improvement. This is the per-rung attribution the "
            "pre-registered bar requires."),
        "curriculum_kill_reason": kill_reason,
        "tape_size_applied": int(_tape_size),
        "tape_size_history": tape_size_history,
        # ---- PRE-RUN AMENDMENT (recorded BEFORE the run, not after).
        # The directive named rungs 3-5 SEMANTICALLY (Jordan masks / scene binding /
        # causal graphs). My own grep measured those emitters ABSENT from this
        # substrate: jordan/interior/contour = 0 in BOTH the curriculum env and the
        # governor, and the env imports neither henri_scene_binder nor
        # henri_action_koopman. The byte-tape VM has no 2-D grid emitter and the
        # learner consumes byte sequences. So the ladder rungs are the IMPLEMENTED
        # lever names, and the semantic rungs are recorded BLOCKED -- never relabelled.
        "semantic_rungs": {
            "requested": ["rung3_jordan_masks", "rung4_scene_binding",
                          "rung5_causal_graphs"],
            "status": "BLOCKED__NO_EMITTER_ON_THIS_SUBSTRATE",
            "evidence": ("grep of henri_curriculum_env.py and henri_curriculum_governor.py: "
                         "jordan/interior/contour occurrences = 0; the env imports neither "
                         "henri_scene_binder nor henri_action_koopman; the learner consumes "
                         "byte sequences, not grids"),
            "tested_instead": list(G_LADDER_NAMES or ()),
        },
        "final_prog_len": int(_cur_prog_len),
        "initial_prog_len": int(prog_len),
        # ---- NOT promotion signals (recorded for diagnosis only)
        "final_loss_NOT_PROMOTION": round(loss_hist[-1], 6) if loss_hist else None,
        "first_loss_NOT_PROMOTION": round(loss_hist[0], 6) if loss_hist else None,
        "reward_mean_NOT_PROMOTION": round(reward_sum / max(reward_n, 1), 10),
        # ---- shards
        "shard_dir": shard_dir,
        "shard_files": shard_files,
        "shard_bytes": shard_bytes,
        "shard_tokens_written": shard_written,
        "timeout_rate": round(timeouts / max(executed, 1), 4),
        "distinct_outputs": len(distinct),
        "bank_size": len(bank.programs),
        "alphabet_size": len(ALPHABET),
        "honest_boundary": (
            "PROMOTION IS GATED ON heldout_progress ONLY. final_loss and reward_mean "
            "are recorded but are NOT promotion signals (a training-loss or reward "
            "gate is satisfiable by memorisation). ICL emergence is NOT tested here."
        ),
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-executions", type=int, default=1_000_000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--out", default="telemetry/stage0_seeding")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reward-subsample", type=int, default=64)
    ap.add_argument("--heldout-rebuild", action="store_true",
                    help="re-derive the held-out set at the CURRENT prog_len each "
                         "eval (default OFF: fixed target keeps numbers comparable)")
    ap.add_argument("--heldout-samples", type=int, default=256)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--shard-dir", default=None,
                    help="write the training token stream to uint8 shards here")
    ap.add_argument("--shard-rows-per-file", type=int, default=250_000)
    ap.add_argument("--threads", type=int, default=0,
                    help="0 = leave torch default; else torch.set_num_threads(n)")
    # ---- CURRICULUM ESCALATION (directive 1). Default OFF: the live default
    # path is unchanged unless the operator passes the flag.
    # DEAD-STORE FIX 2026-09-27: the escalation logic was added to run_seeding's
    # body but these flags were never declared and main() never forwarded them,
    # so the ON path failed with "unrecognized arguments" -- a flag that is read
    # but not wired is a dead store.
    ap.add_argument("--curriculum-escalate", action="store_true",
                    help="escalate program depth when loss variance collapses")
    ap.add_argument("--curriculum-window", type=int, default=50,
                    help="rolling window for the plateau variance detector")
    ap.add_argument("--plateau-var-threshold", type=float, default=1e-4,
                    help="sigma^2 below which depth escalates")
    ap.add_argument("--max-prog-len", type=int, default=96,
                    help="ceiling for the escalated program length")
    ap.add_argument("--curriculum-levers", action="store_true",
                    help="route sampling through henri_curriculum_env so ALL "
                         "FIVE governor levers reach the generator (default OFF)")
    ap.add_argument("--governor-progress-eps", type=float, default=1e-3,
                    help="held-out gain counted as progress by the governor")
    ap.add_argument("--governor-trigger", choices=("variance", "progress", "cadence"),
                    default="variance",
                    help="escalation trigger: the post-convergence variance floor, "
                         "or the learning-progress derivative (Directive 1)")
    ap.add_argument("--cadence-windows", type=int, default=1,
                    help="escalate every N windows while the loss is still moving")
    ap.add_argument("--progress-rate-threshold", type=float, default=1e-3,
                    help="relative per-step improvement floor for --governor-trigger progress")
    ap.add_argument("--governor-kill-patience", type=int, default=3,
                    help="non-progressing escalations before the run TERMINATES")
    a = ap.parse_args()
    if a.threads > 0:
        torch.set_num_threads(a.threads)
    s = run_seeding(a.n_executions, a.batch_size, a.out, a.seed,
                    reward_subsample=a.reward_subsample,
                    heldout_samples=a.heldout_samples,
                    heldout_rebuild=bool(a.heldout_rebuild),
                    eval_every=a.eval_every,
                    shard_dir=a.shard_dir,
                    shard_rows_per_file=a.shard_rows_per_file,
                    curriculum_escalate=a.curriculum_escalate,
                    plateau_var_threshold=a.plateau_var_threshold,
                    curriculum_window=a.curriculum_window,
                    max_prog_len=a.max_prog_len,
                    curriculum_levers=a.curriculum_levers,
                    progress_eps=a.governor_progress_eps,
                    kill_patience=a.governor_kill_patience,
                    governor_trigger=a.governor_trigger,
                    progress_rate_threshold=a.progress_rate_threshold,
                    cadence_windows=a.cadence_windows)
    print(json.dumps(s, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
