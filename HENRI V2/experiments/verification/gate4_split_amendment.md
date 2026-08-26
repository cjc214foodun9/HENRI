# Gate 4 — Heldout Split Amendment (parent prereg `821aeb8c`)

**Status:** PREREGISTERED AMENDMENT (hypothesis — execution receipts upgrade to observed)
**Date:** 2026-08-26
**Parent seal:** `821aeb8c-c7f6-4604-93d7-47476158dfa0` (Gate 4 prereg, hypothesis)
**Candidate commit:** `804a793c9957806628ab3da23527b285a528787b` (branch `feat/temporal-navigation-t0`, remote-reconciled)
**Seed:** `20260826` (pinned in prereg; used for split selection)

## 1. Why an amendment

The sealed prereg pins: environment universe source, exclusion evidence, seed, timeouts, verdict classes, and kill criteria. It does NOT pin the heldout selection algorithm or count. Sol (slot-3 advisory) requires any unpinned choice to be amendment-sealed before generation. This document pins the algorithm; it changes no other prereg field.

## 2. Split algorithm (deterministic, single-use)

1. **Universe** = official catalog metadata from `arc_agi.Arcade.get_environments()` (metadata only; no env instantiated, no `make()`, no step). Ordered by `game_id` ascending; canonical SHA-256 over the JSON-serialized id list (recorded at generation).
2. **Exclusion set (exposed, agent-touched evidence)** = the 12 envs with observed run/probe/ledger/sans evidence: `ar25 bp35 cn04 dc22 ft09 g50t ka59 lp85 ls20 m0r0 re86 sb26`.
   - Sources: `gate1-run/out/*.json` + `*_sans` dirs (12), `temporal_ledger.jsonl` env ids (11), `arc_action_probe.py` probes (ft09/ka59/m0r0), p79d probe logs (ka59/ls20).
   - Note: `environment_files/` local caches (25 dirs on the persistent Vast checkout) are evaluator machinery, not demonstrations; presence of cache files does NOT by itself constitute model exposure. No corpus/model store contains ARC task bytes (zero-pretraining invariant). If any selected id later appears in a model-side artifact, the contamination gate fires and the split is void.
3. **Unseen set** = universe − exclusion (13 ids expected).
4. **Selection** = `random.Random(20260826)` shuffling the sorted unseen list; take the first 8. The remaining 5 stay reserved for a future single-use split.
5. **Selection code** = `gate4_heldout_generate.py` (committed beside this amendment); its SHA-256 is recorded in the heldout manifest.
6. **Heldout artifact** = `gate4_heldout_manifest.json` with fields: `catalog_sha256`, `exclusion_sha256` (sorted JSON id list), `unseen_ids`, `selected_ids`, `seed`, `selection_code_sha256`, `generated_utc`, `consumed=false`. The file's own SHA-256 is recorded in the commit + seal event.
7. **Single-use rule:** selected envs must NOT be instantiated, probed, or stepped while `score_eligible=false`. Any step on a selected env before eligibility burns the split and voids the heldout.

## 3. Kill criteria (unchanged from prereg)

- Any selected id in any prior model-side artifact → `BLOCKED_CONTAMINATION` (split void).
- `score_eligible=false` at launch → `BLOCKED_ACTION_HEAD_NOT_CALIBRATED` (preregistered verdict class); no `game.step` on selected envs.
- Catalog or exclusion digest mismatch at generation → `BLOCKED_SPLIT_DIGEST_MISMATCH`.
- `consumed=true` before eligibility → `BLOCKED_SPLIT_CONSUMED`.
