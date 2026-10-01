# Historical instruction snapshot

This is the previous entry file. It is not current policy. Use current SKILL.md for routing. Revalidate every dated runtime claim.

---
name: henri-co-scientist-rigor
description: "Use for E_log claim clipping, scaffold audits, UCB ranking."
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [henri, co-scientist, execution-logs, hallucination-clipping, scaffold, trueskill, ucb, rigor]
---

# Co-Scientist Scientific-Rigor Protocol

Purpose: map the Google co-scientist real-world reliability mechanisms onto HENRI's
fail-closed governance, add the missing deterministic machinery, and keep every
adoption falsifiable. Where this protocol conflicts with a HENRI gate, the HENRI
gate wins.

## Source and provenance

- User-supplied digest (2026-09-02) of "Accelerating Scientific Research with
  Gemini in the Real-World" (Google co-scientist extension).
- Primary abstract verified 2026-09-02: alphaXiv/arXiv 2608.26701
  (https://www.alphaxiv.org/abs/2608.26701). Claims that exist only in the full
  PDF are UNVERIFIED until that PDF is audited.
- This skill is HYPOTHESIS until a live HENRI run exercises at least one
  mechanism below. Do not claim co-scientist parity.

## Mechanism map

| Research mechanism | HENRI state | Addition in this skill |
|---|---|---|
| Hallucination clipping vs E_log | Evidence labels + deterministic receipts | Mandatory claim-clipping pass before any report delivery |
| Joint objective with penalties | Governance, not a trainable loss | Hard rule: claim without log line is demoted or removed |
| Scaffold → transition → full-scale | Staged remote ladder exists | Static transition audit list between scaffold and full-scale |
| TrueSkill + UCB tournament | Absent | Conditional bounded tournament for candidate selection (default OFF) |
| Length calibration | Envelopes exist (Photon ≤1800 chars, leaf ≤500 tokens) | Pre-send envelope check in the completion standard |
| Tiered triage | Exists (leaf routing, Judge-C) | Pointer only; no new machinery |

## 1. Log-grounded claim clipping (LogAuditor + Clipping)

Rule for every deliverable that carries empirical claims:

1. The run writes a machine-readable artifact E_log: stdout/stderr redirected to
   file, plus run_id, exit code, and one line per metric.
2. Before delivery, map each claim to an E_log line or an artifact hash. An
   unmapped claim is removed, relabeled INFERRED/HYPOTHESIS, or re-run.
3. Empty or missing E_log ⇒ BLOCKED. Do not generate downstream claims from an
   empty log.
4. Send only compact derived summaries upward. Keep raw logs on disk and name
   the path in the report.
5. This enforces the paper's penalty term hard: set the report rule to
   "no claim without a log line" instead of tuning a soft lambda weight.

## 2. Staged scaffold protocol

Phase 1 Scaffold: run on a minimal subsample (default ≤16 rows or equivalent)
under a strict timeout (default ≤600 s). Validate schema, imports, device
placement, and the data path at scaffold cost.

Phase 2 Transition audit: static scan for (a) mock stubs, (b) subsample
variables that silently shrink the data, (c) hardcoded answers or paths,
(d) placeholder callbacks, (e) SELF-CONFIRMING TEST DATA (see below). Strip
each finding or fail the transition. Apply the mock-loop filter from the
operating standards before Phase 3.

### (e) Self-confirming test data

A gate whose fixtures are authored by the same person who authored the gate can
encode the expected answer on the axis the operator under test PRESERVES. The
gate then passes and proves nothing.

Worked example (measured 2026-09-12, HENRI carrier Stage 2). The claim was that
mean pooling destroys metric locality so AUC collapses. The fixture placed class
identity on two axes at once:

```python
blocks = [(label * 3 + i) % 16 for i in range(3)]   # averaged away by .mean(0)
chans  = [(label * 2 + i) % 8  for i in range(3)]   # RETAINED by .mean(0)
```

The operator was `w.view(16, 4096).mean(dim=0)`. It averages the block axis and
keeps the channel axis, so identity survived and AUC was 0.9292 -> PASS. On a
variant where identity lived only in blocks, the same operator scored 0.5014
(chance). The mechanism was real; the fixture hid it.

Audit rule. For every metric gate, ask: which axis/subspace does the operator
preserve, and where does the fixture put the signal? If those are the same axis,
the gate cannot fail and is not evidence. Then run the MINIMAL DISCRIMINATING
CASE: a fixture where the signal lives only on the axis the operator destroys.
If the gate does not fail there, the claimed mechanism is not reproduced.

General form: vary where the signal lives while holding the operator fixed.
A single fixture that passes is not a test. A pair that separates PASS from FAIL
by signal placement is.

Phase 3 Full scale: only after the scaffold passes and the transition audit is
clean, release the full data loop and compute budget on the remote CUDA target.

Kill rule: fix a syntax, import, or API mismatch at scaffold cost. Never let it
reach full-scale cost.

## 3. Bayesian tournament ranking (CONDITIONAL, default OFF)

Use when candidate ranking decides the next experiment and pairwise deterministic
probes exist.

1. Each candidate i carries N(mu_i, sigma_i^2).
2. Run pairwise tournaments; the outcome comes from a deterministic probe only.
   LLM consensus is advisory, never the tournament score.
3. Update TrueSkill-style mu and sigma. Select the next candidate by
   UCB_i = mu_i + kappa * sigma_i.
4. Pre-register kappa, the pair budget, the number of generations, and the kill
   criteria before the tournament starts.
5. Keep the candidate set small and pre-registered. HENRI measured ranking
   dilution near 110 candidates (MBPP run12); large pools defeat selection.
6. A tournament result is selection evidence, not an external task outcome.
7. A load-bearing experiment using this mechanism requires a sealed human
   decision (APPROVE_REMOTE_RUN or higher).

## 4. Length calibration

Fixed envelopes already exist. Re-state the checks:

- Photon reports ≤1800 characters using the mobile contract format.
- Leaf results are bounded JSON ≤500 tokens.
- No raw logs, full diffs, or unbounded tool output moves upward.
- Add the pre-send length check to the completion standard when the destination
  is a mobile channel.

## 5. Falsification hooks

- This skill is validated only when a live HENRI run uses at least one mechanism
  and the claim-clipping pass catches or blocks a claim. Until then the mapping
  is HYPOTHESIS.
- A negative result of any mechanism is a governance win. Record the verdict
  (FALSIFIED or BLOCKED) with the artifact path.
- Do not cite this paper to weaken a HENRI fail-closed gate, an approval level,
  or an evidence class.

## References

- Primary: https://www.alphaxiv.org/abs/2608.26701 (abstract verified 2026-09-02).
- HENRI: henri-research section "Evidence discipline";
  henri-agent-integration reference index (benchmark outcome gates, ARC ladder,
  cache playbook); henri-mobile-governance (Photon envelope contract).
