"""DOES THE CODEC WAVE FAMILY CARRY RECOVERABLE CONTENT? -- v1, VERDICT REFUSED.

DEFECT DISCLOSURE (2026-10-03) -- READ BEFORE TRUSTING ANYTHING BELOW.
    This file printed CODEC_WAVES_CARRY_NO_RECOVERABLE_TOKEN. That verdict is
    REFUSED and is NOT a finding about the codec:
      real waves  train loss 4.4473 -> 4.4151  (barely moved from ln(128)=4.85)
      shuffled    train loss 0.0573 -> 0.0010  (memorized PERFECTLY)
    Both arms have identical shapes, parameter counts, and block multisets.
    The shuffled control fitting to near-zero while real waves stall is an
    asymmetry this file cannot explain. So "no recoverable content" cannot be
    separated from "optimization failed at this input scale". The verdict is
    withdrawn and this instrument is superseded by codec_content_recovery_v2.py
    (centroid, no optimizer, compulsory positive-control ladder).
    Retained because the rejected family must be retained.

WHY THIS EXISTS
    A-K5 measured a DEGENERATE checkpoint (trained on random waves and random
    labels; see evidence/ak5_provenance_correction.json). That verdict says
    nothing about the codec. This probe answers the prior question:

        Can ANY head learn to recover token content from the wave family that
        CompositionalTextCodec actually emits?

    Both outcomes are decisive:
      YES -> the codec carries content; A-K5's negative isolates to that artifact.
      NO  -> the codec itself discards recoverable content; the memory design is
             implicated, not a single checkpoint.

WHY A TRAINED HEAD IS THE RIGHT INSTRUMENT HERE
    The codec is a deterministic sha256 -> positioned-accumulate map. Whether its
    waves are INVERTIBLE cannot be settled by reading the source alone: the
    question is whether a gradient-trained readout finds the mapping. That is
    exactly what this measures, and it is the "small real-data scaffold" step
    ScientistTwo prescribes before full scale.

CONTENT DEFINITION (order-free, matching the codec)
    The codec is a bag of word/byte/trigram features; it carries NO word order.
    So the target must be order-free: given the wave of a sentence containing
    word w, predict w. Order carries no information, so this is the strongest
    target the representation can support.

PRE-REGISTERED (frozen before the run)
    Chance = 1/V.
    G1 LEARNABLE   train-template accuracy >= 0.50
    G2 GENERALIZES held-out-template accuracy >= 0.25
                   (content, not template memorisation)
    G3 RANDOM-LABEL control held-out accuracy <= 0.05  (must fail)
    G4 SHUFFLED-WAVE control held-out accuracy <= 0.05 (must fail)

VERDICTS
    CODEC_WAVES_CARRY_RECOVERABLE_CONTENT   G1 and G2 pass with both controls dead
    CODEC_LEARNABLE_NOT_GENERAL             G1 passes, G2 fails (memorisation only)
    CODEC_WAVES_CARRY_NO_RECOVERABLE_TOKEN   G1 fails
    PROBE_INVALID                            controls fail to stay dead

SCOPE / SAFETY
    Local CPU. 0 USD. No GPU. No optimizer step against any existing checkpoint.
    Writes NO file into the repository. Trains a fresh throwaway head in memory.
    Never touches models/henri_decoder_checkpoint.pt.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

_V2 = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, _V2)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import zone_c_world_knowledge_codec as C  # noqa: E402

SEED = int(os.environ.get("CC_SEED", "20261003"))
N_HEAD_WORDS = int(os.environ.get("CC_V", "128"))
HIDDEN = int(os.environ.get("CC_HIDDEN", "256"))
STEPS = int(os.environ.get("CC_STEPS", "300"))
BATCH = int(os.environ.get("CC_BATCH", "64"))
LR = float(os.environ.get("CC_LR", "1e-3"))

TRAIN_TEMPLATES = (
    "the {w} report",
    "{w} is the word",
    "describe {w} now",
    "note the {w} here",
    "{w} appears again",
    "we discuss {w} today",
    "another {w} example",
    "the {w} section follows",
)
EVAL_TEMPLATES = (
    "a {w} was mentioned",
    "consider {w} carefully",
    "the {w} value holds",
)

D = C.WAVE_DIM
assert D == 65536, "codec geometry changed; re-read the codec before proceeding"

WORDS = [
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
    "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa",
    "quebec", "romeo", "sierra", "tango", "uniform", "victor", "whiskey", "xray",
    "yankee", "zulu", "anchor", "beacon", "cipher", "dagger", "ember", "falcon",
    "granite", "harbor", "ivory", "jasper", "kernel", "lantern", "marble", "nectar",
    "onyx", "pilot", "quartz", "raven", "sable", "timber", "umbra", "violet",
    "willow", "xenon", "yarrow", "zephyr", "amber", "bronze", "cobalt", "denim",
    "elm", "flint", "garnet", "hazel", "indigo", "jade", "kelp", "lilac",
    "maple", "nickel", "opal", "pearl", "quill", "ruby", "slate", "topaz",
    "umber", "vermil", "wheat", "xylem", "yolk", "zinc", "aspen", "birch",
    "cedar", "dogwood", "elder", "fir", "ginkgo", "holly", "iris", "juniper",
    "kudzu", "larch", "mango", "nutmeg", "oak", "pine", "quince", "rowan",
    "spruce", "teak", "ulmus", "vine", "walnut", "yew", "zelkova", "alder",
    "basil", "clove", "dill", "fennel", "ginger", "hyssop", "mint", "oregano",
    "parsley", "rosemary", "sage", "thyme", "anise", "borage", "cumin", "dill2",
    "endive", "fennel2", "garlic", "horserad", "lovage", "marjoram", "nasturt",
    "sorrel",
]


def wave_of(codec, text: str) -> np.ndarray:
    b, _proj = codec.encode(text)
    a = np.frombuffer(b, dtype=np.float32)
    if a.size != D:
        raise RuntimeError("payload %d != WAVE_DIM %d" % (a.size, D))
    return a.copy()


def build(codec, templates):
    X, y = [], []
    for wi, w in enumerate(WORDS):
        for t in templates:
            X.append(wave_of(codec, t.format(w=w)))
            y.append(wi)
    return torch.from_numpy(np.stack(X)), torch.tensor(y, dtype=torch.long)


def shuffle_blocks(X: torch.Tensor, seed: int) -> torch.Tensor:
    """Same-family content-destroyed control: permute the block axis.

    Preserves every block vector and the per-block norm exactly; destroys the
    arrangement. Identical semantics to the A-K5 gate control.
    """
    g = torch.Generator().manual_seed(seed)
    out = X.clone().reshape(-1, C.NUM_BLOCKS, C.BLOCK_DIM)
    for i in range(out.shape[0]):
        p = torch.randperm(C.NUM_BLOCKS, generator=g)
        out[i] = out[i][p]
    return out.reshape(X.shape[0], D)


class Head(nn.Module):
    def __init__(self, d_in: int, hidden: int, n_out: int):
        super().__init__()
        self.fc1 = nn.Linear(d_in, hidden)
        self.fc2 = nn.Linear(hidden, n_out)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


@torch.no_grad()
def accuracy(model, X, y, bs=256) -> float:
    model.eval()
    ok = 0
    for i in range(0, X.shape[0], bs):
        p = model(X[i:i + bs]).argmax(dim=1)
        ok += int((p == y[i:i + bs]).sum())
    return ok / float(X.shape[0])


def train_head(Xtr, ytr):
    torch.manual_seed(SEED)
    model = Head(D, HIDDEN, len(WORDS))
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    g = torch.Generator().manual_seed(SEED)
    n = Xtr.shape[0]
    model.train()
    for step in range(STEPS):
        idx = torch.randint(0, n, (BATCH,), generator=g)
        loss = F.cross_entropy(model(Xtr[idx]), ytr[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (step + 1) % 100 == 0:
            print("   step %4d  loss %.4f" % (step + 1, float(loss)), flush=True)
    return model


def main() -> int:
    t0 = time.time()
    print("== CODEC CONTENT-RECOVERY SCAFFOLD ==")
    print("   D=%d blocks=%d slots=%d V=%d hidden=%d steps=%d lr=%g"
          % (D, C.NUM_BLOCKS, C.BLOCK_DIM, len(WORDS), HIDDEN, STEPS, LR))
    print("   chance = %.4f" % (1.0 / len(WORDS)))

    codec = C.get_codec()
    Xtr, ytr = build(codec, TRAIN_TEMPLATES)
    Xev, yev = build(codec, EVAL_TEMPLATES)
    print("   train=%d  eval=%d  (eval uses DIFFERENT templates)"
          % (Xtr.shape[0], Xev.shape[0]))

    # sanity: waves are not degenerate
    nz = float((Xtr[0] != 0).float().mean())
    print("   wave[0] nnz fraction=%.6f  norm=%.4f" % (nz, float(Xtr[0].norm())))

    print("   training head ...", flush=True)
    model = train_head(Xtr, ytr)

    acc_tr = accuracy(model, Xtr, ytr)
    acc_ev = accuracy(model, Xev, yev)
    print("   TRAIN-template accuracy = %.4f" % acc_tr)
    print("   EVAL -template accuracy = %.4f" % acc_ev)

    # G3: random-label control. Same waves, shuffled labels.
    gl = torch.Generator().manual_seed(SEED + 1)
    ytr_rand = ytr[torch.randperm(ytr.shape[0], generator=gl)]
    m_rand = train_head(Xtr, ytr_rand)
    acc_rand = accuracy(m_rand, Xev, yev)
    print("   CONTROL random-label eval accuracy = %.4f" % acc_rand)

    # G4: shuffled-wave control. Same labels, block-shuffled waves.
    Xtr_s = shuffle_blocks(Xtr, SEED + 2)
    Xev_s = shuffle_blocks(Xev, SEED + 3)
    m_shuf = train_head(Xtr_s, ytr)
    acc_shuf = accuracy(m_shuf, Xev_s, yev)
    print("   CONTROL shuffled-wave eval accuracy = %.4f" % acc_shuf)

    chance = 1.0 / len(WORDS)
    g1 = acc_tr >= 0.50
    g2 = acc_ev >= 0.25
    g3 = acc_rand <= 0.05
    g4 = acc_shuf <= 0.05

    print()
    print("   G1 learnable   train >= 0.50 : %.4f -> %s" % (acc_tr, "PASS" if g1 else "FAIL"))
    print("   G2 generalizes eval  >= 0.25 : %.4f -> %s" % (acc_ev, "PASS" if g2 else "FAIL"))
    print("   G3 random-label  <= 0.05    : %.4f -> %s" % (acc_rand, "PASS" if g3 else "FAIL"))
    print("   G4 shuffled-wave <= 0.05    : %.4f -> %s" % (acc_shuf, "PASS" if g4 else "FAIL"))

    if not (g3 and g4):
        verdict = "PROBE_INVALID"
        note = "A control did not stay dead; this scaffold cannot adjudicate."
    elif g1 and g2:
        verdict = "CODEC_WAVES_CARRY_RECOVERABLE_CONTENT"
        note = ("A fresh head recovers token content from codec waves and GENERALISES "
                "to unseen templates, while both controls stay at chance. The codec "
                "carries recoverable content; A-K5's negative isolates to that "
                "degenerate checkpoint, not to the wave family.")
    elif g1 and not g2:
        verdict = "CODEC_LEARNABLE_NOT_GENERAL"
        note = "Recovers seen templates only; no content generalisation."
    else:
        verdict = "CODEC_WAVES_CARRY_NO_RECOVERABLE_TOKEN"
        note = ("No head learns the mapping even on training templates. The codec "
                "itself does not expose recoverable token content at this capacity.")

    print()
    print("   VERDICT = %s" % verdict)
    print("   %s" % note)
    print("   elapsed=%.1fs  evidence_class=DERIVED  cost=0 USD" % (time.time() - t0))
    return 0 if verdict == "CODEC_WAVES_CARRY_RECOVERABLE_CONTENT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
