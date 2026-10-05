"""Proprietary byte-level BPE tokenizer. Reproducible, no third-party assets.

Why not a borrowed tokenizer: gate G-U7 forbids off-the-shelf components, and the
prior `hash(text) % V` labels were not reproducible across processes (measured:
25859 / 13472 / 15667 in three runs). BPE merges are frozen to the corpus, so the
same corpus yields the same ids in every process.

Vocabulary is DATA-DETERMINED: size = 256 bytes + learned merges. The decoder's
text head matches the tokenizer it was trained with.
"""
from __future__ import annotations

import json
from collections import Counter


class ByteBPE:
    """Byte-level byte-pair encoding. Deterministic tie-break: lowest pair tuple."""

    def __init__(self, merges=None):
        # merges: list of (int, int) -> new_id, applied in order
        self.merges: list[tuple[int, int]] = list(merges or [])
        self._rank = {p: i for i, p in enumerate(self.merges)}

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges)

    # ---------------------------------------------------------------- training
    def train(self, texts, vocab_size: int = 512, max_rounds: int | None = None):
        """Learn merges until vocab_size or no pair repeats twice."""
        target = max(257, int(vocab_size))
        seqs = [list(t.encode("utf-8")) for t in texts if t]
        if not seqs:
            return self
        rounds = 0
        while self.vocab_size < target:
            if max_rounds is not None and rounds >= max_rounds:
                break
            pairs = Counter()
            for s in seqs:
                for i in range(len(s) - 1):
                    pairs[(s[i], s[i + 1])] += 1
            if not pairs:
                break
            # deterministic: highest count, then lowest pair tuple
            best = max(pairs.items(), key=lambda kv: (kv[1], tuple(-x for x in kv[0])))
            if best[1] < 2:
                break
            pair = best[0]
            new_id = self.vocab_size
            self.merges.append(pair)
            self._rank[pair] = len(self.merges) - 1
            seqs = [self._merge_seq(s, pair, new_id) for s in seqs]
            rounds += 1
        return self

    @staticmethod
    def _merge_seq(seq, pair, new_id):
        out, i = [], 0
        while i < len(seq):
            if i < len(seq) - 1 and seq[i] == pair[0] and seq[i + 1] == pair[1]:
                out.append(new_id)
                i += 2
            else:
                out.append(seq[i])
                i += 1
        return out

    # ------------------------------------------------------------- application
    def encode(self, text: str) -> list[int]:
        seq = list(text.encode("utf-8"))
        if not self.merges:
            return seq
        while len(seq) > 1:
            best, best_rank = None, None
            for i in range(len(seq) - 1):
                r = self._rank.get((seq[i], seq[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = i, r
            if best is None:
                break
            pair = self.merges[best_rank]
            seq = self._merge_seq(seq, pair, 256 + best_rank)
        return seq

    def decode(self, ids) -> str:
        # invert merges recursively down to bytes
        def expand(i):
            if i < 256:
                return bytes([i])
            a, b = self.merges[i - 256]
            return expand(a) + expand(b)

        return b"".join(expand(int(i)) for i in ids).decode("utf-8", errors="replace")

    # ------------------------------------------------------------------- io
    def to_json(self) -> str:
        return json.dumps({"merges": self.merges})

    @classmethod
    def from_json(cls, blob: str):
        return cls(merges=[tuple(p) for p in json.loads(blob)["merges"]])


def train_byte_bpe(texts, vocab_size: int = 512) -> ByteBPE:
    return ByteBPE().train(texts, vocab_size=vocab_size)
