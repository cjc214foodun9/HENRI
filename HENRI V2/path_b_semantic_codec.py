"""PathBSemanticCodec — supervised semantic phase codec (Class 4.3, Path B).

Learned contrastive codec mapping code text (AST node types + lexical tokens +
scope symbols) onto unit-norm REAL phase waves on S^{D-1} with a frozen
block-phasor lift. Binding = per-pair block rotation (real-valued FHRR
binding), norm-preserving by construction. No dense [D, D] operator: the only
trainable parameters are the embedding [|V|, d_latent]; the lift is a frozen
carrier + per-phasor block index + scale (O(D) storage).

Representation contract (representation-core-audit): this codec emits the
CONTINUOUS float32 wave family (interleaved cos/sin pairs), NOT the Z_256
uint8 ring. Feeding a uint8 ring vector raises RepresentationBoundaryError.

Default-OFF experimental component. Not imported by any production path unless
the runner flag --path-b-semantic-codec is explicitly set.
"""
from __future__ import annotations

import ast
import math
import re
from typing import Optional, Sequence

import torch
import torch.nn as nn

TOKEN_RE = re.compile(
    r"[A-Za-z_]\w*|\d+\.\d+|\d+|<=|>=|==|!=|->|\*\*|//|[^\sA-Za-z0-9]"
)
UNK = "<UNK>"


class RepresentationBoundaryError(TypeError):
    """Raised when a Z_256 uint8 ring vector crosses into the continuous codec."""


def lex_code(code: str) -> list[str]:
    return TOKEN_RE.findall(code)


def ast_node_types(code: str) -> list[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    return [type(n).__name__ for n in ast.walk(tree)]


class PathBSemanticCodec(nn.Module):
    def __init__(
        self,
        d_model: int = 65536,
        d_latent: int = 512,
        vocab: Optional[Sequence[str]] = None,
        device: str = "cpu",
        seed: int = 7,
    ) -> None:
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("d_model must be even (interleaved cos/sin pairs)")
        self.d_model = int(d_model)
        self.d_latent = int(d_latent)
        self.num_phasors = self.d_model // 2
        self.device = device

        vocab_list = list(vocab) if vocab else []
        self.vocab: dict[str, int] = {v: i for i, v in enumerate(vocab_list)}
        if UNK not in self.vocab:
            self.vocab[UNK] = len(self.vocab)
        self.emb = nn.Embedding(len(self.vocab), self.d_latent)
        g = torch.Generator()
        g.manual_seed(seed)
        nn.init.normal_(self.emb.weight, std=0.02)

        # Frozen functional lift: each phasor i belongs to latent block
        # b = i % d_latent; angle phi_i = carrier_i + scale_i * e[b].
        # Storage O(D), never [D, d_latent] (~128 MB dense matrix banned).
        carrier = torch.rand(self.num_phasors, generator=g) * (2.0 * math.pi)
        scale = (torch.rand(self.num_phasors, generator=g) * 2.0 - 1.0) * 0.5
        block = torch.arange(self.num_phasors, dtype=torch.long) % self.d_latent
        self.register_buffer("_carrier", carrier)
        self.register_buffer("_scale", scale)
        self.register_buffer("_block", block)
        self.register_buffer("_norm", torch.tensor(float(math.sqrt(self.num_phasors))))
        self.lift = "frozen-block-phasor"

    # ---- tokenization -------------------------------------------------
    def _token_ids(self, text: str) -> torch.Tensor:
        toks = lex_code(text) + ast_node_types(text)
        ids = [self.vocab.get(t, self.vocab[UNK]) for t in toks]
        if not ids:
            ids = [self.vocab[UNK]]
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    # ---- encoding -----------------------------------------------------
    def _embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self.emb(ids).mean(dim=0)  # [d_latent]

    def _phasor_angles(self, e: torch.Tensor) -> torch.Tensor:
        return self._carrier + self._scale * e[self._block]  # [num_phasors]

    def encode_sequence(self, text) -> torch.Tensor:
        if not isinstance(text, str):
            raise TypeError(f"encode_sequence requires str, got {type(text).__name__}")
        ids = self._token_ids(text)
        e = self._embed(ids)
        phi = self._phasor_angles(e)
        wave = torch.empty(self.d_model, dtype=torch.float32, device=self.device)
        wave[0::2] = torch.cos(phi)
        wave[1::2] = torch.sin(phi)
        wave = wave / self._norm
        return wave  # unit norm on S^{D-1} (1e-6)

    # ---- binding (real-valued FHRR, norm-preserving) ------------------
    def bind(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        self._validate(a)
        self._validate(b)
        out = torch.empty_like(a)
        out[0::2] = a[0::2] * b[0::2] - a[1::2] * b[1::2]
        out[1::2] = a[0::2] * b[1::2] + a[1::2] * b[0::2]
        # Each input pair is a unit phasor scaled by 1/sqrt(num_phasors);
        # the product pair has magnitude 1/num_phasors, so the raw norm is
        # 1/sqrt(num_phasors). Multiply by _norm = sqrt(num_phasors) to
        # restore unit norm exactly (norm-preserving by construction).
        return out * self._norm

    # ---- similarity ---------------------------------------------------
    @staticmethod
    def _validate(x: torch.Tensor) -> None:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"expected torch.Tensor, got {type(x).__name__}")
        if x.dtype == torch.uint8:
            raise RepresentationBoundaryError(
                "uint8 Z_256 ring vector crossed into the continuous Path B codec; "
                "map through ring_to_real ((c/255)-1)*2 then normalize, or use "
                "PathBSemanticCodec.encode_sequence"
            )
        if x.dtype != torch.float32:
            raise RepresentationBoundaryError(
                f"Path B codec requires float32 waves, got {x.dtype}"
            )

    def cosine_similarity(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        self._validate(a)
        self._validate(b)
        if a.shape != (self.d_model,) or b.shape != (self.d_model,):
            raise ValueError(
                f"expected [{self.d_model}] waves, got {tuple(a.shape)} / {tuple(b.shape)}"
            )
        return torch.dot(a, b)  # both unit norm

    # ---- contrastive training ------------------------------------------
    def train_contrastive(
        self,
        dataset: Sequence[tuple[str, str]],
        val_dataset: Sequence[tuple[str, str]],
        steps: int = 1000,
        batch_size: int = 32,
        lr: float = 3e-3,
        tau: float = 0.07,
        seed: int = 7,
    ) -> dict:
        """InfoNCE over (anchor, positive-variant) pairs; negatives = batch.

        dataset: list of (item_id, code). Positives are generated by
        deterministic identifier-rename + statement-reorder variants.
        """
        opt = torch.optim.AdamW(self.parameters(), lr=lr)
        g = torch.Generator()
        g.manual_seed(seed)
        idx = torch.randperm(len(dataset), generator=g).tolist()
        n = len(dataset)
        loss_accum: list[float] = []
        self.train()
        for step in range(steps):
            batch_ids = idx[(step * batch_size) % n : (step * batch_size) % n + batch_size]
            if len(batch_ids) < 2:
                batch_ids = idx[:batch_size]
            anchors, positives = [], []
            for i in batch_ids:
                code = dataset[i][1]
                anchors.append(self.encode_sequence(code))
                positives.append(self.encode_sequence(_variant(code, i, seed)))
            A = torch.stack(anchors)
            P = torch.stack(positives)
            A = A / A.norm(dim=1, keepdim=True)
            P = P / P.norm(dim=1, keepdim=True)
            logits = (A @ P.T) / tau
            labels = torch.arange(len(batch_ids), device=self.device)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_accum.append(loss.item())
        self.eval()
        val_acc = self.contrastive_accuracy(val_dataset, batch_size=batch_size, seed=seed)
        return {"train_loss_mean": float(sum(loss_accum) / max(1, len(loss_accum))),
                "val_contrastive_acc": val_acc}

    def contrastive_accuracy(
        self, dataset: Sequence[tuple[str, str]], batch_size: int = 32, seed: int = 7
    ) -> float:
        g = torch.Generator()
        g.manual_seed(seed)
        idx = torch.randperm(len(dataset), generator=g).tolist()
        correct = total = 0
        self.eval()
        with torch.no_grad():
            for start in range(0, len(idx), batch_size):
                b = idx[start : start + batch_size]
                if len(b) < 2:
                    continue
                anchors, positives = [], []
                for i in b:
                    anchors.append(self.encode_sequence(dataset[i][1]))
                    positives.append(self.encode_sequence(_variant(dataset[i][1], i, seed)))
                A = torch.stack(anchors)
                P = torch.stack(positives)
                A = A / A.norm(dim=1, keepdim=True)
                P = P / P.norm(dim=1, keepdim=True)
                sims = A @ P.T
                preds = sims.argmax(dim=1)
                correct += int((preds == torch.arange(len(b), device=self.device)).sum())
                total += len(b)
        return correct / max(1, total)


def _variant(code: str, i: int, seed: int) -> str:
    """Deterministic semantic-preserving variant: rename identifiers using a
    fixed pool of in-vocabulary names (stable across training)."""
    RENAME_POOL = ["x", "y", "z", "a", "b", "c", "n", "m", "i", "j", "k",
                   "arr", "lst", "s", "t", "res", "val", "item"]
    names = sorted(set(re.findall(r"\b[a-zA-Z_]\w*\b", code)) - {
        "def", "return", "if", "else", "elif", "for", "while", "in", "import",
        "from", "lambda", "not", "and", "or", "True", "False", "None",
        "pass", "break", "continue", "raise", "try", "except", "with", "as",
        "assert", "del", "global", "nonlocal", "yield", "class", "print", "len",
    })
    g = torch.Generator()
    g.manual_seed(seed + i)
    mapping = {}
    pool = RENAME_POOL
    for idx, nm in enumerate(names):
        mapping[nm] = pool[(idx + int(torch.randint(0, len(pool), (1,), generator=g).item())) % len(pool)]
    out = code
    for old, new in mapping.items():
        out = re.sub(rf"\b{old}\b", new, out)
    return out
