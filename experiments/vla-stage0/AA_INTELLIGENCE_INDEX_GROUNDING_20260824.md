# Artificial Analysis Intelligence Index — Official-Source Grounding (2026-08-24)

**Evidence class:** OBSERVED_PRIMARY_BYTES (direct HTTPS, pinned, hashed)

## Result

The user-goal label "Artificial Analysis Intelligence Index v4.1" is **VERIFIED
as the family prefix**; the precise current official version is **v4.1.1**.

## Pinned evidence

| Item | Value |
|---|---|
| URL | `https://artificialanalysis.ai/` (official homepage) |
| Retrieval | 2026-08-24, direct curl, UA spoof only |
| HTTP | 200 |
| Bytes | 1,790,335 |
| SHA-256 | `52b130a12bbc31bec16ee183…` (file `C:/Users/chan/aa_homepage.html`) |
| Exact string | `Artificial Analysis Intelligence Index v4.1.1 incorporates 9 evaluations: GDPval-AA v2, 𝜏³-Banking, Terminal-Bench v2.1, SciC…` |
| Nav confirms | Intelligence Index methodology lives under `/methodology/intelligence-benchmarking` |
| 404 evidence | `/methodology/intelligence-index` and `/methodology/intelligence-index-v4` → HTTP 404 (17,067 B, sha `2e520885…`) |

## Boundary

- This grounds the LABEL only. The 9-evaluation composition table, category
  weights, item counts, and scoring rules must be extracted from pinned bytes
  of `/methodology/intelligence-benchmarking` before any benchmark claim.
- HENRI currently has **no capability path** into these evaluations (VLA
  gate 0/12; Stage-0a = `DYNAMICAL_SUBSTRATE_VERIFIED` only). No AA capability
  claim is derived from this note.

## Next falsification

Any claim that HENRI targets specific AA v4.1.1 components must cite the
pinned composition-table bytes and the live capability path for that component.
