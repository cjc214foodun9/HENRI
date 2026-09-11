# M-2 — Prompt-Cache Hit-Rate Protocol Pre-registration

**Date:** 2026-09-11 · **Status:** `REGISTERED` (not executed) · **Owner:** henri-agent-integration
**Target:** raise aggregate MoA reference prompt-cache hit rate from the measured
**21.7%** to **≥90%**, or report honestly that the target is unreachable.

All numbers below are `OBSERVED` unless labelled otherwise.

## 1. Measured baseline (`scripts/henri_cache_diag.py --newest 3`)

3 newest traces, 27 turns carrying usage.

| slot | n | cache_read | fresh_in | cache_write | hitA% | $ | $/Mtok |
|---|---:|---:|---:|---:|---:|---:|---:|
| z-ai/glm-5.3-flash | 27 | 1,691,904 | 5,410,700 | **0** | 23.8 | 0.97 | 0.137 |
| meta/muse-spark-1.3 | 27 | 1,606,197 | 5,305,993 | **0** | 23.2 | 7.20 | 1.041 |
| minimax/minimax-m3 | 27 | 1,244,596 | 5,657,374 | **0** | 18.0 | 1.85 | 0.267 |
| **TOTAL** | 27 | **4,542,697** | **16,374,067** | **0** | **21.7** | 10.02 | 0.479 |

`hitA = cache_read / (cache_read + fresh_input)`.

## 2. Three findings that constrain the design

**F1 — `cache_write_tokens` is always 0.** The field is present in 12/12 usage
records but never populated. Therefore `cache_write` is **not a usable
diagnostic**, and any protocol step justified by "reduce cache writes" is
unfounded. Reported writes are not evidence that no cache was written.

**F2 — The wall-clock TTL hypothesis is FALSIFIED.** `henri_cache_ttl_probe.py`
computed `Spearman(inter-turn gap, hit%) = -0.118` (n=24). Mean hit% for short
gaps 18.7% vs long gaps — no decay with time. Counter-examples inside one trace:
gap 127 s → 31.0% hit, while gap 1,943 s → **65.3%** hit. Elapsed time does not
govern the hit rate.

**F3 — Hits quantise at 0% / ≈33% / ≈66%.** Per-turn hit rates cluster at
fractions of one third. That is the signature of *some slots hitting while
others miss in the same turn*, i.e. per-slot prompt divergence — not a global
cache expiry event.

## 3. Mechanism hypothesis

> **H-PREFIX.** The hit-killer is prefix *content* churn, not cache lifetime.
> The MoA prompt is rebuilt each turn with volatile content (tool results, file
> listings, session-specific text) interleaved ahead of otherwise stable
> material, so the provider's prefix-cache boundary moves and the reusable
> prefix shrinks.

Consistent with F1–F3. Competes with:

> **H-CADENCE.** Long idle gaps expire the cache (config `prompt_caching.cache_ttl: 5m`).

`FALSIFIED` by F2 as the dominant factor.

## 4. Arithmetic ceiling (`DERIVED`)

`ceiling = 1 - floor_new / median_prompt`, where `floor_new` is the minimum
`fresh + write` tokens any single call needed.

| slot | floor_new | median prompt | ceiling |
|---|---:|---:|---:|
| glm-5.3-flash | 105 | 223,452 | 100.0% |
| muse-spark-1.3 | 126 | 215,842 | 99.9% |
| minimax-m3 | 266 | 202,868 | 99.9% |

**The ≥90% target is arithmetically reachable on this token mix.** It is an
upper bound, not a promise.

## 5. Protocol under test

Both arms are *discipline*, not prompt shrinking. Reducing prompt size to buy
hit percentage is explicitly rejected: it trades capability for a metric.

- **P1 Freeze the prefix.** No skill, tool, or system-prompt edit during a live
  session. Note: this session edited 6 skills, which by construction voids the
  prefix they sit in.
- **P2 Pin the roster.** No slot rotation mid-session.
- **P3 One continuous session.** Prefer a single long session over many short
  ones so the prefix is reused rather than rebuilt.
- **P4 Append volatile content last.** Tool results and file listings go after
  the stable block; never interleave.
- **P5 Probe the TTL ceiling.** Read the provider's supported maximum for
  `prompt_caching.cache_ttl` (currently `5m`) and raise it if supported.
  `BLOCKED` until the provider capability is actually checked.

## 6. Pre-registered acceptance

| outcome | criterion |
|---|---|
| **ACCEPT** | Arm B aggregate `hitA` ≥ 90.0% over ≥5 traces |
| **ESCALATE** | 70.0% ≤ hitA < 90.0% — H-PREFIX supported but incomplete; find the residual zone |
| **REJECT** | hitA < 70.0% after one frozen-prefix session — H-PREFIX falsified |

Cheapest kill experiment: a single session under P1–P4 with no mid-session edits,
measured by `henri_cache_diag.py --newest 3`. If the rate does not move
materially, H-PREFIX is falsified and the cause is provider-side.

Measure with `henri_cache_diag.py` (per-slot + ceiling) and
`henri_cache_ttl_probe.py` (TTL vs prefix). Numerics are authoritative.

## 7. What NOT to do

- Do **not** claim >90% before it is measured. Registering the protocol is not
  achieving it.
- Do **not** add hard per-slot output caps. GLM's mean output is 8,267 tokens
  with reasoning included; a small cap would truncate real work. The cost lever
  is slot *selection* (muse-spark-1.3 bills at 1.041 USD/Mtok, **7.6×** glm's
  0.137 USD/Mtok, for the same token volume).
- Do **not** quote an aggregator cache or spend figure: the aggregator slot
  carries no usage block at all (`UNVERIFIED`, 0/27 turns).
- Do **not** change the roster mid-session; that voids the very prefix the
  protocol depends on.

## 8. Findings added after the kill experiment (`OBSERVED` 2026-09-11)

**F4 — The miss is NOT global; it is per-slot.** `scripts/henri_cache_slot_probe.py`
on the current session trace:

| slot | cache_read | fresh_in | hit% | turns with a hit |
|---|---:|---:|---:|---:|
| z-ai/glm-5.3-flash | 162,368 | 146,978 | 52.5 | 1/4 |
| meta/muse-spark-1.3 | 6,257 | 297,749 | 2.1 | 1/4 |
| minimax/minimax-m3 | 384 | 307,778 | 0.1 | 3/4 |

In **4 of 4** multi-slot turns the slots disagreed (0 turns where all slots
agreed). A single global prefix protocol therefore cannot reach 90%; each slot's
own prefix must be stabilised. `LIMITATION`: n=4 turns in one trace — a strong
lead, not a settled law. Re-run with `--newest 5` before acting.

**F5 — Per-slot `max_tokens` is NOT a supported key.** The live `config.yaml`
exposes exactly one token knob, `moa.max_tokens: 4096`, at MoA level. No
per-slot field exists. Adding `max_tokens` inside a `reference_models[]` entry
would be a **phantom flag** — written but never read. Per-slot output bounding is
`BLOCKED` until the runtime exposes the field. Slot *selection* remains the lever.

**F6 — `prompt_caching.cache_ttl` is a closed enum, not a free duration.**
`hermes-agent/agent/agent_init.py` reads `prompt_caching.cache_ttl` and **keeps
`"5m"` for unknown values**; `agent_runtime_helpers.py` refers to "the two
cache_ttl tiers accepted by config". P5 in §5 must read the accepted pair before
proposing any value. A made-up TTL would silently fall back to 5m and appear to
work.

## 9. Revised acceptance (supersedes §6)

Because F4 shows per-slot behaviour, the ≥90% target is evaluated **per slot**,
not only in aggregate:

| outcome | criterion |
|---|---|
| **ACCEPT** | every paid slot reaches hitA ≥ 90.0% over ≥5 traces under one frozen-prefix session |
| **PARTIAL** | ≥1 slot reaches ≥90% but others do not — report per slot; do not average |
| **REJECT** | no slot moves above 70.0% — H-PREFIX falsified for this provider |

Aggregate-only reporting is explicitly insufficient: an aggregate can look
healthy while a single expensive slot runs at 0%.

