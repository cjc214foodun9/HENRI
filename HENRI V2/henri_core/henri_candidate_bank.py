"""henri_candidate_bank.py -- MINIMUM VIABLE candidate bank (G-SJ3a/G-SJ3b unblock).

WHY THIS MODULE EXISTS
    The tree already carried a selector (henri_core/wave_readout.py) and a corpus
    generator (henri_core/m4_generative.py build_corpus), but NO component that turns
    waves into the candidate bank the selector consumes. That missing link is exactly
    why G-SJ3a reported BLOCKED_BANK_ABSENT and every downstream selector claim stayed
    unmeasurable. This module closes that link end to end.

THE TWO BANK IDENTITIES (bank identity DECIDES the verdict -- measured earlier)
    full   one row per (program, input) training pair. Row-wise chance is inflated by
           trace multiplicity, so a top-1 read here is NOT a representation claim.
    dedup  one row per DISTINCT trace. Containment 0.1367 caps any within-group rule
           below 0.1500 BY CONSTRUCTION.
    Both are produced from the same waves. The receipt records which one it used.

CONTRACT
    Pure functions over real waves. No training. No global RNG use beyond the caller's
    seeded generator. Tie-robust statistics only: top-k is tie arbitration when the
    argmax always ties, so the primary statistic is P(gold in the max group) with a
    per-query SIZE-MATCHED control.

SELF-CAUGHT DEFECT (fixed, recorded)
    The first `size_matched_control` drew its pick from INSIDE the argmax tie group. That
    computes E[|gold n group| / |group|] -- the within-group precision -- not the
    containment control. Measured: 0.1523 / 0.1275 versus within_group_gold_precision
    0.14815 / 0.12654, i.e. the same quantity reported twice. The unit test
    `test_size_matched_control_grows_with_group_width` failed (0.075 > 1.0) and exposed it.
    The correct statistic draws a SAME-SIZE subset uniformly from the K columns.

EVIDENCE CLASS: every number this module returns is OBSERVED at the call site. Nothing
here is a model-performance claim.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

BANK_SCHEMA_ID = "henri.candidate-bank.v1"
CONTROL_SEED = 20261010


def flat(x: torch.Tensor) -> torch.Tensor:
    """[N, ...] -> [N, D]. Concatenate real/imag for complex waves."""
    x = x.reshape(x.shape[0], -1)
    return torch.cat([x.real, x.imag], -1) if torch.is_complex(x) else x


def build_waves(gen, specs: list[str]) -> torch.Tensor:
    """Frozen wave encoder -> L2-normalised [N, D]. NaN is zeroed before normalising."""
    with torch.no_grad():
        Z = flat(gen.wave(list(specs)))
        Z = torch.nan_to_num(Z)
        Z = F.normalize(Z, dim=-1)
    return Z


def gold_sets(train_traces, query_traces) -> list[set]:
    """For each query, the SET of train row indices whose trace equals the query trace."""
    tmap: dict[tuple, set] = {}
    for j, t in enumerate(train_traces):
        tmap.setdefault(tuple(t), set()).add(j)
    return [tmap.get(tuple(q), set()) for q in query_traces]


def dedup_columns(train_traces) -> list[int]:
    """First-occurrence column index per DISTINCT trace, in first-seen order."""
    seen: dict[tuple, int] = {}
    keep: list[int] = []
    for j, t in enumerate(train_traces):
        k = tuple(t)
        if k not in seen:
            seen[k] = j
            keep.append(j)
    return keep


def restrict(sim: torch.Tensor, cols: list[int], gold: list[set]) -> tuple:
    """Restrict a [Q, K] similarity to `cols`, remapping gold to the new column space."""
    pos = {c: i for i, c in enumerate(cols)}
    sim_d = sim[:, cols]
    gold_d = [set(pos[g] for g in gs if g in pos) for gs in gold]
    return sim_d, gold_d


def max_groups(sim: torch.Tensor) -> list[list[int]]:
    """[Q, K] -> per-row argmax tie group (index list). Tie-robust."""
    mx = sim.max(dim=1).values
    mask = sim >= mx.unsqueeze(1)
    return [torch.nonzero(mask[q]).flatten().tolist() for q in range(sim.shape[0])]


def containment_report(sim: torch.Tensor, gold: list[set],
                       control_seed: int = CONTROL_SEED, n_draw: int = 40) -> dict:
    """P(gold in max group), group statistics, and a SIZE-MATCHED random control.

    The size-matched control is required: a fixed-size control is confounded by group
    width, because a wider tie group is easier to hit by chance.
    """
    Q, K = sim.shape
    groups = max_groups(sim)
    hit = 0
    prec = 0.0
    for q, members in enumerate(groups):
        inter = gold[q] & set(members)
        hit += int(len(inter) > 0)
        prec += len(inter) / max(1, len(members))
    sizes = torch.tensor([len(g) for g in groups], dtype=torch.float32)
    # SIZE-MATCHED CONTAINMENT CONTROL (self-caught defect fixed here).
    # MY BUG: the first version drew a pick from INSIDE the group and tested membership,
    # which computes E[|gold n group| / |group|] -- the WITHIN-GROUP PRECISION, not the
    # containment control. Measured proof: MVP printed size_matched_control 0.1523 /
    # 0.1275 against within_group_gold_precision 0.14815 / 0.12654 -- the same quantity,
    # twice. My own docstring requires the control to GROW with group width; the buggy
    # version did not, and the unit test `..._grows_with_group_width` failed 0.075 > 1.0.
    # CORRECT STATISTIC: draw a same-size subset uniformly from the K columns and ask
    # whether gold lands in it. Exact expectation = 1 - C(K-g, k)/C(K, k), deterministic.
    from math import comb

    def _size_matched(K: int, gsz: int, k: int) -> float:
        if k <= 0 or gsz <= 0:
            return 0.0
        if K - gsz < k:
            return 1.0
        return 1.0 - comb(K - gsz, k) / comb(K, k)

    sm = sum(_size_matched(K, len(gold[q]), len(groups[q])) for q in range(Q)) / Q
    return {
        "bank_columns": K,
        "n_queries": Q,
        "containment": hit / Q,
        "mean_group_size": float(sizes.mean()),
        "frac_singleton_group": float((sizes <= 1).float().mean()),
        "within_group_gold_precision": prec / Q,
        "row_uniform_chance": sum(len(x) for x in gold) / (Q * K),
        "size_matched_control": sm,
        "within_group_control": prec / Q,
        "control_note": ("size_matched_control = same-size random subset containment "
                         "(grows with group width); within_group_control == "
                         "within_group_gold_precision (pick inside the group)"),
    }


def score_selector(sim: torch.Tensor, gold: list[set], channel: torch.Tensor | None,
                   mode: str, train_traces, seed: int = CONTROL_SEED,
                   n_draw: int = 40) -> dict:
    """Top-1 for a selector mode, plus index drift under a column permutation.

    Imported lazily so this module stays usable without the selector present.
    """
    import sys, os
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    import wave_readout as wr

    Q, K = sim.shape

    def picks_for(s: torch.Tensor, chan, m: str):
        gen = torch.Generator().manual_seed(seed)
        if m == "argmax":
            return wr.select(s, mode="argmax")
        if m == "two_stage":
            return wr.select(s, channel=chan, mode="two_stage", generator=gen)
        out = []
        for _ in range(n_draw):
            out.append(wr.select(s, mode="uniform", generator=gen))
        return out

    def hit_rate(s, picks):
        if isinstance(picks[0], list) and len(picks) == n_draw and not isinstance(picks[0], int):
            return sum(sum(1 for q in range(Q) if int(p[q]) in gold[q]) / Q
                       for p in picks) / len(picks)
        return sum(1 for q in range(Q) if int(picks[q]) in gold[q]) / Q

    ch = channel
    if mode == "two_stage" and ch is None:
        # GUARD: the trace list MUST match the column count, or the expand() fails.
        if len(train_traces) != sim.shape[1]:
            raise ValueError(
                f"two_stage needs one trace per column: len(train_traces)="
                f"{len(train_traces)} vs sim.shape[1]={sim.shape[1]}. Build the channel "
                f"with frequency_channel(full_traces, cols, sim.shape) instead.")
        ch = wr.trace_frequency_channel(train_traces, sim.shape)
    top1 = hit_rate(sim, picks_for(sim, ch, mode))

    # permutation drift: permute columns, translate picks back with order[pick]
    order = torch.randperm(K, generator=torch.Generator().manual_seed(seed + 7))
    perm_sim = sim[:, order]
    perm_ch = ch[:, order] if ch is not None else None
    p_perm = picks_for(perm_sim, perm_ch, mode)
    if mode == "uniform":
        perm_top1 = sum(sum(1 for q in range(Q) if int(order[p[q]]) in gold[q]) / Q
                        for p in p_perm) / len(p_perm)
    else:
        perm_top1 = sum(1 for q in range(Q) if int(order[p_perm[q]]) in gold[q]) / Q
    return {"mode": mode, "top1": top1, "top1_under_permutation": perm_top1,
            "drift": abs(top1 - perm_top1)}


def frequency_channel(full_traces, cols: list[int], sim_shape) -> torch.Tensor:
    """[B, K] channel = FULL-bank multiplicity of each RETAINED column's trace.

    MY BUG, FIXED. The first driver called wave_readout.trace_frequency_channel with the
    FULL 48-row trace list against a 28-column restricted matrix -> RuntimeError: expanded
    size (28) must match existing size (48). The channel must be built over the columns
    that actually survive restriction.

    WHY THE COUNTS STILL COME FROM THE FULL BANK: if counts were taken over the restricted
    (deduplicated) list, every value would be 1 and the channel would be CONSTANT. A
    constant-across-candidates channel makes the two-stage objective a strict no-op --
    argmax(channel) is undefined, ties break by position, and the stage adds nothing. This
    is the same constant-penalty defect measured earlier. Reading full-bank multiplicity
    onto the dedup columns is what keeps the channel informative.
    """
    counts: dict[tuple, int] = {}
    for t in full_traces:
        k = tuple(t)
        counts[k] = counts.get(k, 0) + 1
    row = torch.tensor([float(counts[tuple(full_traces[c])]) for c in cols])
    return row.unsqueeze(0).expand(sim_shape[0], sim_shape[1]).clone()


def channel_noop_check(channel: torch.Tensor) -> dict:
    """PRE-REGISTERED no-op check. A constant channel cannot discriminate.

    If the channel takes one value on every candidate, then argmax(channel) is the whole
    group and the two-stage objective equals uniform-in-group exactly. `frac_constant`
    1.0 means the stage is a no-op and the mechanism must be redesigned, NOT tuned.
    """
    if channel is None:
        return {"constant": None, "frac_constant": None}
    flat = channel.reshape(-1)
    n_unique = int(torch.unique(flat).numel())
    return {
        "n_unique_values": n_unique,
        "min": float(flat.min()), "max": float(flat.max()),
        "frac_constant": 1.0 if n_unique <= 1 else 0.0,
        "is_noop": bool(n_unique <= 1),
        "term_balance_mean_channel_bits": float(flat.mean()),
    }


def build(corpus, dedup: bool) -> dict:
    """Convenience: build the bank description from a corpus. Waves supplied by caller."""
    raise NotImplementedError("use build_waves + restrict at the call site")
