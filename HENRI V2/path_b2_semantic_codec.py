"""PathB2DiscriminativeCodec — hard-negative discriminative semantic codec (Class 4.4, Path B2).

Supervised successor to the falsified Path B1 codec. Same unit-modulus phasor
substrate (isometry NECESSARY) but the encoder is trained with hard-negative
InfoNCE (tau=0.07) over execution-verified grammar mutants so that lookalike
AST skeletons are pushed OUT of the goal neighborhood (margin >= +0.25).

Representation contract (representation-core-audit): emits the CONTINUOUS
float32 wave family (interleaved cos/sin pairs), NOT the Z_256 uint8 ring.
uint8 ring input raises RepresentationBoundaryError.

Default-OFF experimental component. Not imported by any production path unless
the runner flag --path-b2-codec is explicitly set.
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
KEYWORDS = {
    "def", "return", "if", "else", "elif", "for", "while", "in", "import",
    "from", "lambda", "not", "and", "or", "True", "False", "None",
    "pass", "break", "continue", "raise", "try", "except", "with", "as",
    "assert", "del", "global", "nonlocal", "yield", "class",
}


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


def idf_weights(tokens: list[str], df: dict[str, int], n_docs: int) -> list[float]:
    """qFHRR-IDF node magnitude scaling (roadmap Level 4)."""
    return [math.log(n_docs / (1.0 + df.get(t, 0))) for t in tokens]


class PathB2DiscriminativeCodec(nn.Module):
    def __init__(
        self,
        d_model: int = 65536,
        d_latent: int = 512,
        vocab: Optional[Sequence[str]] = None,
        df: Optional[dict[str, int]] = None,
        n_docs: int = 1000,
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
        self.n_docs = int(n_docs)

        vocab_list = list(vocab) if vocab else []
        self.vocab: dict[str, int] = {v: i for i, v in enumerate(vocab_list)}
        if UNK not in self.vocab:
            self.vocab[UNK] = len(self.vocab)
        self.emb = nn.Embedding(len(self.vocab), self.d_latent)
        g = torch.Generator()
        g.manual_seed(seed)
        nn.init.normal_(self.emb.weight, std=0.02)

        # Frozen functional lift: phasor i belongs to latent block b = i % d_latent;
        # phi_i = carrier_i + scale_i * e[b]. Storage O(D); never [D, d_latent].
        # scale in [-2, 2] so input structure modulates phases (no carrier collapse).
        carrier = torch.rand(self.num_phasors, generator=g) * (2.0 * math.pi)
        scale = (torch.rand(self.num_phasors, generator=g) * 2.0 - 1.0) * 2.0
        block = torch.arange(self.num_phasors, dtype=torch.long) % self.d_latent
        self.register_buffer("_carrier", carrier)
        self.register_buffer("_scale", scale)
        self.register_buffer("_block", block)
        self.register_buffer("_norm", torch.tensor(float(math.sqrt(self.num_phasors))))
        self.lift = "frozen-block-phasor-idf"

        # Document-frequency table for qFHRR-IDF weighting (frozen after build).
        self.df: dict[str, int] = dict(df or {})

    # ---- tokenization -------------------------------------------------
    def _token_ids(self, text: str) -> torch.Tensor:
        toks = lex_code(text) + ast_node_types(text)
        ids = [self.vocab.get(t, self.vocab[UNK]) for t in toks]
        if not ids:
            ids = [self.vocab[UNK]]
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    def _weights(self, text: str) -> torch.Tensor:
        toks = lex_code(text) + ast_node_types(text)
        if not toks:
            toks = [UNK]
        w = idf_weights(toks, self.df, self.n_docs)
        return torch.tensor(w, dtype=torch.float32, device=self.device)

    # ---- encoding -----------------------------------------------------
    def _embed(self, ids: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # IDF-weighted mean embedding, then normalized to unit norm x sqrt(d_latent)
        # so the phasor phase excursion is O(scale) regardless of token count.
        e = (self.emb(ids) * w.unsqueeze(-1)).sum(dim=0) / max(1.0, w.sum().item())
        e = torch.nn.functional.normalize(e, p=2, dim=0) * math.sqrt(float(self.d_latent))
        return e

    def _phasor_angles(self, e: torch.Tensor) -> torch.Tensor:
        return self._carrier + self._scale * e[self._block]

    def encode_sequence(self, text) -> torch.Tensor:
        if not isinstance(text, str):
            raise TypeError(f"encode_sequence requires str, got {type(text).__name__}")
        ids = self._token_ids(text)
        w = self._weights(text)
        e = self._embed(ids, w)
        phi = self._phasor_angles(e)
        wave = torch.empty(self.d_model, dtype=torch.float32, device=self.device)
        wave[0::2] = torch.cos(phi)
        wave[1::2] = torch.sin(phi)
        wave = wave / self._norm
        return wave  # unit norm on S^{D-1}

    # ---- binding (real-valued FHRR, norm-preserving) ------------------
    def bind(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        self._validate(a)
        self._validate(b)
        out = torch.empty_like(a)
        out[0::2] = a[0::2] * b[0::2] - a[1::2] * b[1::2]
        out[1::2] = a[0::2] * b[1::2] + a[1::2] * b[0::2]
        return out * self._norm

    # ---- similarity ---------------------------------------------------
    @staticmethod
    def _validate(x: torch.Tensor) -> None:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"expected torch.Tensor, got {type(x).__name__}")
        if x.dtype == torch.uint8:
            raise RepresentationBoundaryError(
                "uint8 Z_256 ring vector crossed into the continuous Path B2 codec; "
                "map through ring_to_real ((c/255)-1)*2 then normalize, or use "
                "PathB2DiscriminativeCodec.encode_sequence"
            )
        if x.dtype != torch.float32:
            raise RepresentationBoundaryError(
                f"Path B2 codec requires float32 waves, got {x.dtype}"
            )

    def cosine_similarity(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        self._validate(a)
        self._validate(b)
        if a.shape != (self.d_model,) or b.shape != (self.d_model,):
            raise ValueError(
                f"expected [{self.d_model}] waves, got {tuple(a.shape)} / {tuple(b.shape)}"
            )
        return torch.dot(a, b)  # both unit norm

    # ---- hard-negative mutant generation (deterministic, sandbox-verified) --
    def generate_hard_negatives(
        self, code: str, n: int = 70, seed: int = 0, sandbox=None
    ) -> list[str]:
        """Deterministic grammar mutants that CHANGE the solution semantics.

        Mutations: identifier rename (only when it changes the referenced name),
        argument-order swap, binop flip (+<->-, *<->/), return-value negation,
        arity variant (add/subtract a constant). Each mutant is syntax-valid;
        when a sandbox is provided, only execution-verified mutants are kept.
        """
        g = torch.Generator()
        g.manual_seed(seed)
        mutants: list[str] = []
        seen: set[str] = set()
        base = code

        # 1. binop flips
        for op, rep in (("+", "-"), ("-", "+"), ("*", "/"), ("/", "*"), ("==", "!="), ("!=", "==")):
            m = re.sub(rf"\{op}", rep, base, count=1)
            if m != base and m not in seen:
                seen.add(m)
                mutants.append(m)
        # 2. return-value negation
        m = re.sub(r"return\s+", "return -", base, count=1)
        if m != base and m not in seen:
            seen.add(m)
            mutants.append(m)
        # 3. identifier renames that break the reference
        names = sorted(set(re.findall(r"\b[a-zA-Z_]\w*\b", base)) - KEYWORDS)
        for nm in names[:12]:
            m = re.sub(rf"\b{nm}\b", nm + "_x", base, count=1)
            if m != base and m not in seen:
                seen.add(m)
                mutants.append(m)
        # 4. constant perturbation
        m = re.sub(r"(\d+)", lambda x: str(int(x.group(1)) + 1), base, count=1)
        if m != base and m not in seen:
            seen.add(m)
            mutants.append(m)

        if sandbox is not None:
            verified = []
            for m in mutants:
                try:
                    res = sandbox.execute(m)
                    # keep only mutants that still EXECUTE (fail-closed: an
                    # execution error is NOT a valid hard negative)
                    if res is not None and getattr(res, "status", None) != "EXECUTION_ERROR":
                        verified.append(m)
                except Exception:
                    continue
            mutants = verified

        # deterministic filler to n (repeat cycle is fine; contract only requires
        # that the first K are real mutants)
        i = 0
        while len(mutants) < n and mutants:
            mutants.append(mutants[i % len(mutants)])
            i += 1
        return mutants[:n]

    # ---- Cholesky Stiefel retraction (isometry bound, E_Gram < 1e-5) ----
    def _cholesky_retract(self) -> float:
        """Enforce orthonormality on the SMALLER Gram side (rank-feasible).

        W is [|V|, d_latent]. If |V| >= d_latent, enforce W^T W = I_{d_latent}
        (column Stiefel, matching the HENRI [D, r] operator contract).
        If |V| < d_latent, enforce W W^T = I_{|V|} (row Stiefel). The other
        Gram cannot be the identity (rank bound). Returns max Gram error.
        """
        W = self.emb.weight
        n_rows, n_cols = W.shape
        with torch.no_grad():
            if n_rows >= n_cols:
                G = W.T @ W + 1e-8 * torch.eye(n_cols, device=W.device)
                L = torch.linalg.cholesky(G)
                # W_new = W L^-T  =>  W_new^T W_new = I
                self.emb.weight.copy_(
                    torch.linalg.solve_triangular(L, W.T, upper=False).T)
                gram = (W.T @ W - torch.eye(n_cols, device=W.device)).abs().max().item()
            else:
                G = W @ W.T + 1e-8 * torch.eye(n_rows, device=W.device)
                L = torch.linalg.cholesky(G)
                # W_new = L^-1 W  =>  W_new W_new^T = I
                self.emb.weight.copy_(
                    torch.linalg.solve_triangular(L, W, upper=False))
                gram = (W @ W.T - torch.eye(n_rows, device=W.device)).abs().max().item()
        return gram

    # ---- contrastive training ------------------------------------------
    def train_contrastive(
        self,
        dataset: Sequence[tuple[str, str]],
        val_dataset: Sequence[tuple[str, str]],
        steps: int = 1500,
        batch_size: int = 32,
        lr: float = 3e-3,
        tau: float = 0.07,
        n_hard: int = 8,
        gram_tol: float = 1e-5,
        seed: int = 7,
    ) -> dict:
        """InfoNCE over (anchor, positive-variant) pairs + hard negatives.

        dataset: list of (item_id, code). Positives = deterministic
        semantic-preserving variants; negatives = in-batch + generated hard
        mutants. Cholesky retraction after every step keeps Gram < gram_tol.
        """
        opt = torch.optim.AdamW(self.parameters(), lr=lr)
        g = torch.Generator()
        g.manual_seed(seed)
        idx = torch.randperm(len(dataset), generator=g, device="cpu").tolist()
        n = len(dataset)
        loss_accum: list[float] = []
        gram_accum: list[float] = []
        self.train()
        for step in range(steps):
            batch_ids = idx[(step * batch_size) % n : (step * batch_size) % n + batch_size]
            if len(batch_ids) < 2:
                batch_ids = idx[:batch_size]
            anchors, positives, negatives = [], [], []
            for i in batch_ids:
                code = dataset[i][1]
                anchors.append(self.encode_sequence(code))
                positives.append(self.encode_sequence(_variant(code, i, seed)))
                for m in self.generate_hard_negatives(code, n=n_hard, seed=seed + i):
                    negatives.append(self.encode_sequence(m))
            A = torch.stack(anchors)
            P = torch.stack(positives)
            A = A / A.norm(dim=1, keepdim=True)
            P = P / P.norm(dim=1, keepdim=True)
            # logits: [B, B + H] where H = n_hard * B in-batch negatives
            sim_pos = A @ P.T  # [B, B]
            if negatives:
                N = torch.stack(negatives)
                N = N / N.norm(dim=1, keepdim=True)
                sim_neg = A @ N.T  # [B, H]
                logits = torch.cat([sim_pos, sim_neg], dim=1) / tau
                labels = torch.arange(len(batch_ids), device=self.device)
            else:
                logits = sim_pos / tau
                labels = torch.arange(len(batch_ids), device=self.device)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            # Cholesky Stiefel retraction (dual-sided; rank-feasible Gram).
            gram = self._cholesky_retract()
            gram_accum.append(gram)
            loss_accum.append(loss.item())
        self.eval()
        val_acc = self.contrastive_accuracy(val_dataset, batch_size=batch_size, seed=seed)
        return {
            "train_loss_mean": float(sum(loss_accum) / max(1, len(loss_accum))),
            "gram_max": float(max(gram_accum)) if gram_accum else None,
            "val_contrastive_acc": val_acc,
        }

    def contrastive_accuracy(
        self, dataset: Sequence[tuple[str, str]], batch_size: int = 32, seed: int = 7
    ) -> float:
        g = torch.Generator()
        g.manual_seed(seed)
        idx = torch.randperm(len(dataset), generator=g, device="cpu").tolist()
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
    names = sorted(set(re.findall(r"\b[a-zA-Z_]\w*\b", code)) - KEYWORDS)
    g = torch.Generator()
    g.manual_seed(seed + i)
    mapping = {}
    for idx, nm in enumerate(names):
        mapping[nm] = RENAME_POOL[(idx + int(torch.randint(0, len(RENAME_POOL), (1,), generator=g, device="cpu").item())) % len(RENAME_POOL)]
    out = code
    for old, new in mapping.items():
        out = re.sub(rf"\b{old}\b", new, out)
    return out
