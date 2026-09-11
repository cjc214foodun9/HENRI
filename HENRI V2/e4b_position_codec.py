"""E4b (v3) — Position-Identifiable Wave Binding (zero-trainable ingress codec).

REVISION HISTORY (defects disclosed, not hidden)
------------------------------------------------
v1: E4B_TERMINATE — HARNESS DEFECT, not mechanism failure.
    - terminal-recovery test encoded the PREFIX but compared the recovered word
      against the FULL window's last word (wrong reference) -> 0/96
    - G2f applied the K5 row-unit contract to a g7-family codec (category error);
      g7 writes sparse hashed cells and leaves untouched rows at zero by design
v2: E4B_TERMINATE — MEASURED DEFECT (real, in the mechanism wiring).
    - I reserved rows 6144..8191 (cells 49152..65535) for the position channel,
      but `v5_feature_cells` maps over the FULL 0..65535 range. Bag features
      therefore wrote INTO the reserved region: a direct probe showed 31 of 128
      bag cells landing inside it and 176 nonzero cells where 64 were expected.
      The leaked bag signal swamped the position evidence -> pos-region
      last_word_to_front 0.9245 and degraded terminal recovery.
v3 (this file): DISJOINT cell spaces. Bag features are mapped into
    [0, BAG_SPAN) = [0, 49152) so they can NEVER address the reserved region;
    the position channel owns [49152, 65536).

DESIGN
  bag channel      cells [0, 49152)        w: b: t: q:   (v6/v7 count-aware)
  position channel cells [49152, 65536)    e:{terminal word}
                                           d1:{word}  d2:{word}  d3:{word}
                                           (distance-from-end, length-independent)

Protocol provenance: circular-mean phasor bundling with length-independent
position keys is the VERIFIED HENRI protocol (goal-adapter v1, sealed
2026-08-25, swap-sim 0.0064). Length-scaled fractional binding is NOT used: run21
measured that it destroys 1-char-edit locality (FALSIFIED_AT_SCALE).

Zero trainable parameters. Deterministic. Default-OFF (nothing imports this).

KILL CRITERIA (E4b)
  G2a pos_last_word_to_front_cos <= 0.60
  G2b pos_terminal_recovery      >= 0.95   (the real gate)
  G2c identical_cos              >= 0.999
  G2d unrelated_cos              <= 0.02
  G2e reversal_cos               <= 0.60
  G2f disjointness               : 0 bag cells inside the position region
  G2g admission preserved        : v6/v7 decode still admits

Usage: python e4b_position_codec.py [--probe N] [--out PATH]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from g5_separable_codec import v5_feature_cells  # noqa: E402
from g7_highorder_codec import (  # noqa: E402
    HighOrderCodec, MAX_WORDS, NUM_BLOCKS, BLOCK_DIM, WAVE_DIM,
)
from zone_c_world_knowledge_codec import tokenize, _l2, _proj_accum, features_of, PROJ_DIM  # noqa: E402

MASK64 = (1 << 64) - 1
GOLDEN64 = 0x9E3779B97F4A7C15
WAVE_EXPAND = 16
POS_ROWS = 2048
POS_OFFSET = (NUM_BLOCKS - POS_ROWS) * BLOCK_DIM          # 49152
POS_SPAN = POS_ROWS * BLOCK_DIM                            # 16384
BAG_SPAN = POS_OFFSET                                      # 49152
N_SUFFIX = 3

KILL = {
    "pos_last_word_to_front_cos_max": 0.60,
    "pos_terminal_recovery_min": 0.95,
    "identical_cos_min": 0.999,
    "unrelated_cos_max": 0.02,
    "reversal_cos_max": 0.60,
}


def _splitmix64(x: int) -> int:
    x = (x + GOLDEN64) & MASK64
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return (z ^ (z >> 31)) & MASK64


def _hash64(s: str) -> int:
    return int.from_bytes(hashlib.sha256(s.encode("utf-8")).digest()[:8], "big")


def _bag_cells(feature: str) -> np.ndarray:
    """Bag cells, RESTRICTED to [0, BAG_SPAN) so they cannot touch the
    reserved position region. Same generator as v5, reduced modulus."""
    return (v5_feature_cells(feature) % BAG_SPAN).astype(np.int64)


def _pos_cells(feature: str) -> np.ndarray:
    """16 reserved-region cells for a position feature."""
    h = _hash64("pos|" + feature)
    return np.array([POS_OFFSET + (_splitmix64(h ^ ((s + 1) * GOLDEN64)) % POS_SPAN)
                     for s in range(WAVE_EXPAND)], dtype=np.int64)


def _signs(cells: np.ndarray) -> np.ndarray:
    return np.where((cells % 2) == 0, 1.0, -1.0).astype(np.float32)


class PositionBoundCodec(HighOrderCodec):
    """g7 count-aware codec + a disjoint, position-identifiable terminal channel."""

    def __init__(self, vocab: list[str], n_suffix: int = N_SUFFIX, **kw):
        super().__init__(vocab, **kw)
        self.n_suffix = int(n_suffix)
        uni = [f"w:{w}" for w in self.vocab]
        self._uni_cells = (np.stack([_bag_cells(f) for f in uni])
                           if uni else np.zeros((0, WAVE_EXPAND), dtype=np.int64))
        self._cache.clear()

    # -- bag evidence over the RESTRICTED map (keeps v6/v7 admission coherent) --
    def _cells(self, key: str) -> np.ndarray:
        if key not in self._cache:
            self._cache[key] = (_bag_cells(key), np.array([]))
        return self._cache[key][0]

    def _pos_feats(self, words: list[str]) -> list[str]:
        if not words:
            return []
        n = len(words)
        out = [f"e:{words[-1]}"]
        for i in range(1, min(self.n_suffix, n - 1) + 1):
            out.append(f"d{i}:{words[n - 1 - i]}")
        return out

    def encode(self, text: str) -> tuple[bytes, np.ndarray]:
        words = tokenize(text)[:MAX_WORDS]
        feats = [f"w:{w}" for w in words]
        feats += [f"b:{words[i]} {words[i + 1]}" for i in range(len(words) - 1)]
        feats += [f"t:{words[i]} {words[i + 1]} {words[i + 2]}"
                  for i in range(len(words) - 2)]
        feats += [f"q:{words[i]} {words[i + 1]} {words[i + 2]} {words[i + 3]}"
                  for i in range(len(words) - 3)]
        acc = np.zeros(WAVE_DIM, dtype=np.float32)
        for f in feats:                                  # bag channel
            c = _bag_cells(f)
            np.add.at(acc, c, _signs(c))
        for f in self._pos_feats(words):                 # position channel
            c = _pos_cells(f)
            np.add.at(acc, c, _signs(c))
        rows = acc.reshape(NUM_BLOCKS, BLOCK_DIM)
        proj = _l2(_proj_accum(features_of(text, ngram_max=2), PROJ_DIM)).astype(np.float32)
        return rows.astype(np.float32).tobytes(), proj

    def recover_terminal(self, wave_rows: np.ndarray) -> tuple[str | None, float]:
        """Recover the terminal word of the ENCODED text from the position channel."""
        flat = np.asarray(wave_rows, dtype=np.float32).ravel()
        best_w, best_ev = None, -1.0
        for w in self.vocab:
            c = _pos_cells(f"e:{w}")
            ev = float(np.sum((flat[c] * _signs(c)) > 0.0)) / WAVE_EXPAND
            if ev > best_ev:
                best_w, best_ev = w, ev
        return best_w, best_ev


def cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(a @ b / (na * nb))


def pos_region(arr: np.ndarray) -> np.ndarray:
    return arr.reshape(NUM_BLOCKS, BLOCK_DIM).ravel()[POS_OFFSET:]


UNRELATED = "quantum chromodynamics renormalization lattice gauge anomaly cancellation beta function"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path,
                    default=Path(r"C:\Users\chan\henri-telemetry\e2\wikitext2_train.parquet"))
    ap.add_argument("--probe", type=int, default=128)
    ap.add_argument("--out", type=Path,
                    default=Path(r"C:\Users\chan\henri-telemetry\e3\e4b_position_codec.json"))
    args = ap.parse_args()

    import pyarrow.parquet as pq
    from g7_highorder_codec import build_vocab

    print("=== E4b v3 position-identifiable codec (disjoint cell spaces) ===",
          flush=True)
    texts = pq.read_table(str(args.corpus)).column("text").to_pylist()
    sents: list[str] = []
    for r in texts[:400]:
        for p in re.split(r"(?<=[.!?])\s+", r.replace("\n", " ")):
            if len(p.strip().split()) >= 3:
                sents.append(p.strip())
    vcb = build_vocab(sents, max_words=20000)
    codec = PositionBoundCodec(vcb)
    print(f"sentences {len(sents)}  vocab {len(vcb)}  "
          f"BAG_SPAN {BAG_SPAN}  POS_OFFSET {POS_OFFSET}", flush=True)

    def enc_arr(t: str) -> np.ndarray:
        return np.frombuffer(codec.encode(t)[0], dtype=np.float32).copy()

    # ---- G2f disjointness: bag cells must never enter the position region --
    probe_ins = 0
    probe_total = 0
    for s in sents[:64]:
        w = tokenize(s)[:MAX_WORDS]
        feats = [f"w:{x}" for x in w]
        feats += [f"b:{w[i]} {w[i+1]}" for i in range(len(w) - 1)]
        feats += [f"t:{w[i]} {w[i+1]} {w[i+2]}" for i in range(len(w) - 2)]
        feats += [f"q:{w[i]} {w[i+1]} {w[i+2]} {w[i+3]}" for i in range(len(w) - 3)]
        for f in feats:
            c = _bag_cells(f)
            probe_total += c.size
            probe_ins += int((c >= POS_OFFSET).sum())
    print(f"bag cells inside position region: {probe_ins}/{probe_total}", flush=True)

    probes = [s for s in sents if 6 <= len(s.split()) <= 20][:args.probe]
    print(f"probes {len(probes)}", flush=True)

    # ---- geometry on the position region ----------------------------------
    rows = []
    for s in probes:
        w = s.split()
        rev = " ".join(reversed(w))
        mv = " ".join([w[-1]] + w[:-1])
        ar = pos_region(enc_arr(s))
        rows.append({
            "identical": cos(ar, pos_region(enc_arr(s))),
            "reversal": cos(ar, pos_region(enc_arr(rev))),
            "last_word_to_front": cos(ar, pos_region(enc_arr(mv))),
            "unrelated": cos(ar, pos_region(enc_arr(UNRELATED))),
        })
    pos_agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    agg_l2f = float(np.mean([cos(enc_arr(s), enc_arr(" ".join(
        [s.split()[-1]] + s.split()[:-1]))) for s in probes[:24]]))
    print("position-region means:", {k: round(v, 4) for k, v in pos_agg.items()},
          flush=True)
    print(f"aggregate last_word_to_front (diagnostic only): {agg_l2f:.4f}", flush=True)

    # ---- terminal recovery (the real gate) --------------------------------
    rec_text = rec_prefix = 0
    rec_rows = []
    for s in probes:
        w = s.split()
        got, ev = codec.recover_terminal(enc_arr(s))
        rec_text += int(got == w[-1])
        pre = " ".join(w[:-1])
        got_p, ev_p = codec.recover_terminal(enc_arr(pre))
        rec_prefix += int(got_p == pre.split()[-1])
        rec_rows.append({"true": w[-1], "recovered": got, "ev": round(ev, 4),
                         "ok": got == w[-1]})
    rate_text = rec_text / max(1, len(probes))
    rate_prefix = rec_prefix / max(1, len(probes))
    print(f"terminal recovery TEXT   {rec_text}/{len(probes)} = {rate_text:.4f}", flush=True)
    print(f"terminal recovery PREFIX {rec_prefix}/{len(probes)} = {rate_prefix:.4f}", flush=True)

    # ---- well-formedness + admission preserved ----------------------------
    nonfinite = 0
    for s in probes[:32]:
        nonfinite += int((~np.isfinite(enc_arr(s))).sum())

    adm = 0
    for s in probes[:32]:
        rows_w = np.frombuffer(codec.encode(s)[0], dtype=np.float32).reshape(
            NUM_BLOCKS, BLOCK_DIM)
        if codec.decode(rows_w).status == "OK":
            adm += 1
    print(f"nonfinite {nonfinite}   decode_OK {adm}/32", flush=True)

    gates = {
        "G2a_pos_last_word_move_cos_ok": pos_agg["last_word_to_front"] <= KILL["pos_last_word_to_front_cos_max"],
        "G2b_pos_terminal_recovery_ok": rate_text >= KILL["pos_terminal_recovery_min"],
        "G2c_identical_ok": pos_agg["identical"] >= KILL["identical_cos_min"],
        "G2d_unrelated_ok": pos_agg["unrelated"] <= KILL["unrelated_cos_max"],
        "G2e_reversal_ok": pos_agg["reversal"] <= KILL["reversal_cos_max"],
        "G2f_disjoint_cell_spaces": probe_ins == 0,
        "G2g_admission_preserved": adm >= 30,
    }
    terminate = not all(gates.values())

    rec = {
        "carrier": "E4b",
        "revision": "v3",
        "defects_disclosed": {
            "v1": "harness defect: prefix-vs-full-window reference mismatch; "
                  "K5 row-unit contract misapplied to a g7-family codec",
            "v2": "measured defect: v5_feature_cells spans 0..65535, so bag "
                  "features wrote into the reserved position region "
                  "(31/128 bag cells inside; 176 nonzero vs 64 expected) -> "
                  "pos last_word_to_front 0.9245, degraded recovery",
            "v3_fix": "bag cells restricted to [0,49152); position channel owns "
                      "[49152,65536); disjointness asserted (G2f)",
        },
        "baseline_falsified_reference": {
            "codec": "g7 HighOrderCodec (hashed bag-of-n-grams)",
            "aggregate_last_word_to_front_cos": 0.940,
            "terminal_prefix_exact_recovery": 0.0625,
            "source": "henri-telemetry/e3/e3_construct_audit.json sha 4f08ef31",
        },
        "design": {
            "bag_cell_span": [0, BAG_SPAN],
            "position_cell_span": [POS_OFFSET, WAVE_DIM],
            "position_features": ["e:{terminal}", "d1:{w}", "d2:{w}", "d3:{w}"],
            "trainable_params": 0,
            "protocol": "circular-mean/phasor bundling with length-independent "
                        "position keys (goal-adapter v1 sealed 2026-08-25); "
                        "length-scaled fractional binding NOT used (run21 FALSIFIED)",
        },
        "n_probe": len(probes),
        "bag_cells_inside_position_region": f"{probe_ins}/{probe_total}",
        "position_region_means": {k: round(v, 6) for k, v in pos_agg.items()},
        "aggregate_last_word_to_front_DIAGNOSTIC_ONLY": round(agg_l2f, 6),
        "terminal_recovery_text": round(rate_text, 4),
        "terminal_recovery_prefix": round(rate_prefix, 4),
        "nonfinite_cells": nonfinite,
        "decode_admission_ok": f"{adm}/32",
        "kill_criteria": KILL,
        "gates": gates,
        "verdict": "E4B_TERMINATE" if terminate else "E4B_G2_PASS",
        "sample_recoveries": rec_rows[:8],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rec, indent=2))
    sha = hashlib.sha256(args.out.read_bytes()).hexdigest()
    print()
    print("gates:", json.dumps(gates, indent=2))
    print(f"E4B_VERDICT={rec['verdict']}")
    print(f"wrote {args.out}\ne4b_sha256 {sha}")
    raise SystemExit(1 if terminate else 0)


if __name__ == "__main__":
    main()
