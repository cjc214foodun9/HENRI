# CLASS49 Packet Audit Receipt (2026-08-21)

Document: `CLASS49_Pre-Registered_Packet.md` (HENRI-PACKET-CLASS49-ATTRIBUTION-SAGNAC-2026)
SHA-256 (raw == LF): `8fa75fd2d70378b41c29c8261ad877fb7812c26cca3d9d309127965f937b57c0` (7,843 B)
Baseline: `13941d9` verified (CLASS48 sealed). Gates were pre-registered before any mutation.

**Verdict: REFACTOR_REQUIRED**

## Blocking findings (against live code)

1. **F1 — Migration targets the wrong table.** Recall runs on
   `phylogenetic_engrams_65536` (`zone_c_segment_cache.py:204`), which has NO
   domain_tag and NO run_id/arm_id/commit_sha columns. Intervention 1 ALTERs only
   `zone_c_engrams` (the stress rollup). Gate 1 and Gate 4 are unenforceable on the
   table that actually feeds recall.
2. **F2 — View predicates match zero live tags.** Live vocabulary is
   `arc3/{env}`, `{env}:ACTION{n}`, `arc3/{env}/field_channel_consolidated`.
   The packet's views filter `IN ('ast','code','text','math')` /
   `IN ('action','grid','ode','control')` → empty. Gate 4 would pass vacuously.
3. **F3 — Intervention 2 cites the wrong site.** The logger writes
   `zone_c_resonant_hypersphere` with no attribution parameters; the real
   un-attributed writes are in `TimescaleZoneCStore.write_engram` (both INSERTs).

## Advisory findings

4. **F4 — Gate 3 vs corpus.** NotebookLM corpus (INFERRED, cited) holds that a
   relative differential over a saturated ~1.0 channel measures uncorrelated
   crosstalk and masks phase-stability collapse; labeled remediation = online EDMD
   with Stiefel retractions. CLASS48 dry-run: δ = −0.0017 PASS (DERIVED, not a
   verdict). Fix: pre-register a sanity floor — if BOTH arms' mean Sagnac ≥ 0.95,
   Gate 3 is INCONCLUSIVE, not PASS.
5. **F5 — Arm config freeze is sound** (matched envs/seeds/freeze, single
   treatment difference, default-OFF).

## Corpus consultations (ca4bb787, auth refreshed)

- View-based namespace isolation: **SUPPORTS** the packet (INFERRED; citations
  fae3966c, fe2e7026, 82577c97).
- Relative Sagnac gate: **CONFLICTS** with the packet (INFERRED; citations
  56cfbe73, fe2e7026, b6f856bc, e0e7d020).

## CLASS48 preserved

RT-MCTS stays default-OFF; Δscore=+2; engagement 897 re-ranks; absolute Gate S
FAIL (0.986 > 0.35) — sealed `13941d9`, not reinterpreted.

## Approval gate

No schema mutation, no arm launch, no production default change until explicit
user approval of the bounded refactor scope (dev Docker first, then prod,
default-OFF, all four gates still pre-registered).
