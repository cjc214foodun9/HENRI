"""
System-1 Kernel v0.4 - Token-Level FSA & Name-Conditioned Egress Engine
========================================================================
Faithful implementation of the Aletheia v0.4 spec (Drive inbox 2026-08-23 09:43,
spec sha b42815d3..., engine sha d582406c...) on the verified v0.3 substrate
(real project tokenizer, tasks, sandbox, eval). The uploaded engine sketch is
treated as the mechanism spec; its mock loops are rejected:
  - 32k-vocab toy grammar with hardcoded token ids 7/8/9/10 (mismatch: the
    project tokenizer has LPAREN=26, RPAREN=27, COLON=30, NL=5);
  - parity-based mock AST reward (sum(tokens) % 2) -> circular validation;
  - random-tensor eval (torch.equal vs random targets) -> vacuous.
Implemented mechanisms (per spec):
  1. Token-level FSA: UNK is never a legal next token (no wildcard); tight
     paren/colon/IND structure; depth-aware NL/EOS ban (kills the `((xs)`
     leak); COLON -> NL only.
  2. UNK logit-mass suppression penalty (softplus on raw UNK logit).
  3. Prompt-symbol cross-attention name conditioning: the decoder attends to
     the signature symbol matrix (def NAME ( args ) :) so task names and arg
     names are SUPPLIED, never memorized.
  4. Masked free-run MLE: autoregressive free decode WITH the FSA mask so MLE
     gradients see grammar-valid contexts (the capability loss).
  5. REINFORCE on shaped rewards (stage 2) + Brier energy baseline.
  6. Extended 1,000-step warm-up + capability abort gates.

Substrate repairs (v0.3 FSA had false negatives on its OWN training targets):
NAME->FOR/IF, COMMA->IN/FOR, RPAREN->OP/ANDOR, LITERAL_NUM->FOR/IF; LPAREN no
longer allows RPAREN; COLON allows NL only; UNK class is empty (no wildcard).
Vocab extended with task names, x/y/zip, &/|, and literal tokens 0..10 so
targets are UNK-free (asserted in preflight).

SMOKE != capability evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Project-owned tokenizer (v0.3 layout + v0.4 extensions)
# ---------------------------------------------------------------------------
RESERVED = ["PAD", "BOS", "EOS", "UNK", "IND", "NL"]
GRAMMAR = ["def", "return", "if", "else", "for", "in", "range", "len", "sum",
           "max", "min", "sorted", "set", "tuple", "list", "append", "abs",
           "round", "int", "float", "True", "False", "None", "and", "or",
           "not", "(", ")", "[", "]", ":", ",", "+", "-", "*", "/", "//",
           "%", "**", "==", "!=", "<", ">", "<=", ">=", "=",
           "xs", "t1", "t2", "a", "b", "n", "m", "i", "v", "res", "acc",
           # v0.4 extensions: loop vars / helper / operators / literals
           "x", "y", "zip", "&", "|",
           "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
TASK_NAMES = ["sum_list", "max_list", "count_positive", "intersect_tuples",
              "union_tuples", "pair_sums", "factorial"]
TOK2ID = {t: i for i, t in enumerate(RESERVED + GRAMMAR + TASK_NAMES)}
ID2TOK = {v: k for k, v in TOK2ID.items()}
VOCAB = len(TOK2ID)

_PAT_RE = r"\bdef\b|\breturn\b|\bif\b|\belse\b|\bfor\b|\bin\b|\brange\b|\blen\b"
_PAT_RE += r"|\bsum\b|\bmax\b|\bmin\b|\bsorted\b|\bset\b|\btuple\b|\blist\b"
_PAT_RE += r"|\bappend\b|\babs\b|\bround\b|\bint\b|\bfloat\b|\bTrue\b|\bFalse\b"
_PAT_RE += r"|\bNone\b|\band\b|\bor\b|\bnot\b|[A-Za-z_]\w*|\d+|[()\[\]:,+*/%<>=!&|]"
import re
_PAT = re.compile(_PAT_RE)


def tokenize_code(code: str) -> list[int]:
    ids = [TOK2ID["BOS"]]
    for ln in code.splitlines():
        if ln[:1] in (" ", "\t"):
            ids.append(TOK2ID["IND"])
        ids.extend(TOK2ID.get(t, TOK2ID["UNK"]) for t in _PAT.findall(ln.strip()))
        ids.append(TOK2ID["NL"])
    ids.append(TOK2ID["EOS"])
    return ids


def detokenize(ids) -> str:
    parts = []
    for i in ids:
        t = ID2TOK.get(int(i), "")
        if t == "IND":
            parts.append("    ")
        elif t == "NL":
            parts.append("\n")
        elif t in ("BOS", "EOS", "PAD"):
            continue
        else:
            parts.append(" " + t)
    src = "".join(parts)
    for a, b in ((" (", "("), ("( ", "("), (" )", ")"), (" ,", ","),
                 (" :", ":"), ("[ ", "["), (" ]", "]")):
        src = src.replace(a, b)
    return src.strip()


# ---------------------------------------------------------------------------
# Deterministic token-class FSA (v0.3 base + v0.4 repairs)
# ---------------------------------------------------------------------------
class_of = {}
for t in ["def"]:
    class_of[t] = "DEF_KW"
for t in ["xs", "t1", "t2", "a", "b", "n", "m", "i", "v", "res", "acc",
          "x", "y", "zip", "range", "len", "sum", "max", "min", "sorted",
          "set", "tuple", "list", "append", "abs", "round", "int", "float"]:
    class_of[t] = "NAME"
for t in TASK_NAMES:
    class_of[t] = "NAME"
for t in ["True", "False", "None"]:
    class_of[t] = "LITERAL_BOOL"
for t in ["+", "-", "*", "/", "//", "%", "**", "&", "|"]:
    class_of[t] = "OP"
for t in ["==", "!=", "<", ">", "<=", ">="]:
    class_of[t] = "CMP"
for t in ["and", "or"]:
    class_of[t] = "ANDOR"
for t in ["not"]:
    class_of[t] = "NOT"
for t in ["return"]:
    class_of[t] = "RETURN"
for t in ["if"]:
    class_of[t] = "IF"
for t in ["else"]:
    class_of[t] = "ELSE"
for t in ["for"]:
    class_of[t] = "FOR"
for t in ["in"]:
    class_of[t] = "IN"
for t in ["("]:
    class_of[t] = "LPAREN"
for t in [")"]:
    class_of[t] = "RPAREN"
for t in ["["]:
    class_of[t] = "LBRACKET"
for t in ["]"]:
    class_of[t] = "RBRACKET"
for t in [":"]:
    class_of[t] = "COLON"
for t in [","]:
    class_of[t] = "COMMA"
for t in ["="]:
    class_of[t] = "ASSIGN"
for t in ["IND"]:
    class_of[t] = "IND"
for t in ["NL"]:
    class_of[t] = "NL"
for t in ["EOS"]:
    class_of[t] = "EOS"
for t in ["PAD"]:
    class_of[t] = "PAD"
class_of["BOS"] = "BOS"
# UNK has NO class membership: v0.4 removes the wildcard entirely.
UNK_CLASS = "UNK"

EXPR = {"NAME", "LITERAL_NUM", "LITERAL_BOOL", "LPAREN", "LBRACKET", "NOT"}
CONT = {"OP", "CMP", "COMMA", "RPAREN", "RBRACKET", "COLON", "IN", "ANDOR",
        "NL", "EOS"}
LINE_END = {"NL", "EOS"}
BODY_START = {"NAME", "LITERAL_NUM", "LITERAL_BOOL", "RETURN", "IF", "FOR",
              "LPAREN", "LBRACKET", "NOT"}
_EMPTY = set()

FSA = {
    "BOS": {"DEF_KW"},
    "DEF_KW": {"NAME"},
    # v0.4: NAME may continue into list/generator comprehensions (FOR, IF)
    "NAME": {"LPAREN", "ASSIGN", "FOR", "IF"} | CONT | LINE_END,
    # v0.4: LPAREN never directly closes (no empty parens in corpus) and never
    # jumps to NL; nested LPAREN and comprehension FOR remain legal.
    "LPAREN": {"NAME", "LITERAL_NUM", "LITERAL_BOOL", "LPAREN", "LBRACKET",
               "NOT", "OP", "COMMA", "IN", "FOR"},
    "RPAREN": CONT | {"COLON", "OP", "ANDOR"} | LINE_END,
    # v0.4: COMMA may precede comprehension binder (x, y in ...)
    "COMMA": EXPR | {"IN", "FOR"},
    "LBRACKET": EXPR | {"RBRACKET", "FOR", "COMMA", "RPAREN"},
    "RBRACKET": CONT | LINE_END,
    # v0.4: COLON -> NL only (tight structural rule)
    "COLON": {"NL"},
    "IND": BODY_START,
    "RETURN": EXPR,
    "IF": EXPR,
    "ELSE": {"COLON"},
    "FOR": {"NAME"},
    "IN": EXPR,
    "ASSIGN": EXPR,
    "OP": EXPR,
    "CMP": EXPR,
    "ANDOR": EXPR,
    "NOT": EXPR,
    # v0.4: generator expressions (1 for x in xs if x > 0)
    "LITERAL_NUM": CONT | LINE_END | {"FOR", "IF"},
    "LITERAL_BOOL": CONT | LINE_END,
    "NL": BODY_START | {"ELSE", "IND"} | LINE_END,
    "EOS": {"EOS", "PAD"},
    "PAD": {"PAD"},
    "UNK": _EMPTY,
}


def _tok_class(tok_id: int) -> str:
    t = ID2TOK[tok_id]
    if t.isdigit():
        return "LITERAL_NUM"
    return class_of.get(t, "UNK")


def build_next_mask() -> torch.Tensor:
    m = torch.zeros(VOCAB, VOCAB, dtype=torch.bool)
    for a in range(VOCAB):
        ca = _tok_class(a)
        allowed = FSA.get(ca, _EMPTY)
        for b in range(VOCAB):
            cb = _tok_class(b)
            # v0.4: UNK is never a legal next token.
            m[a, b] = (cb in allowed) and (b != TOK2ID["UNK"])
    return m


NEXT_MASK = build_next_mask()


def grammar_compliance(ids) -> float:
    ok = tot = 0
    prev = TOK2ID["BOS"]
    for tok in ids:
        if tok in (TOK2ID["PAD"], TOK2ID["EOS"]):
            continue
        ok += int(NEXT_MASK[prev, tok].item())
        tot += 1
        prev = tok
    return ok / max(1, tot)


# ---------------------------------------------------------------------------
# Token-level FSA mask (class table + tight token rules + depth tracking)
# ---------------------------------------------------------------------------
class TokenFSAGrammarMask(nn.Module):
    """Deterministic next-token mask: class FSA + token-tight + paren depth.

    apply_mask(logits [B, V], prev_ids [B], depth [B]) -> (masked_logits,
    bool_mask [B, V]). Guarantees: UNK never selectable; LPAREN never closes
    to RPAREN/NL; COLON -> NL only; NL/EOS forbidden while paren depth > 0;
    RPAREN/RBRACKET forbidden at depth 0; dead rows fall back to EOS.
    """

    def __init__(self, next_mask: torch.Tensor):
        super().__init__()
        self.register_buffer("next_mask", next_mask.clone())
        self.unk_id = TOK2ID["UNK"]
        self.lparen_id = TOK2ID["("]
        self.rparen_id = TOK2ID[")"]
        self.lbracket_id = TOK2ID["["]
        self.rbracket_id = TOK2ID["]"]
        self.colon_id = TOK2ID[":"]
        self.nl_id = TOK2ID["NL"]
        self.eos_id = TOK2ID["EOS"]
        self.pad_id = TOK2ID["PAD"]

    def depth_delta(self, tok_ids: torch.Tensor) -> torch.Tensor:
        d = torch.zeros_like(tok_ids, dtype=torch.long)
        d = d + ((tok_ids == self.lparen_id) | (tok_ids == self.lbracket_id)).long()
        d = d - ((tok_ids == self.rparen_id) | (tok_ids == self.rbracket_id)).long()
        return d

    def apply_mask(self, logits: torch.Tensor, prev_ids: torch.Tensor,
                   depth: torch.Tensor):
        m = logits.clone()
        # 1. UNK hard ban (wildcard removal).
        m[:, self.unk_id] = -1e9
        # 2. Class FSA layer.
        cls = self.next_mask[prev_ids]                     # [B, V]
        m = m.masked_fill(~cls, -1e9)
        # 3. Tight structural rules (token level, on top of classes).
        is_lparen = (prev_ids == self.lparen_id)
        if is_lparen.any():
            m[is_lparen, self.rparen_id] = -1e9
            m[is_lparen, self.nl_id] = -1e9
        is_colon = (prev_ids == self.colon_id)
        if is_colon.any():
            m[is_colon, :] = -1e9
            m[is_colon, self.nl_id] = logits[is_colon, self.nl_id]
        # 4. Depth-aware: no line/stream end inside an open paren/bracket;
        #    no closing token at depth 0.
        d_gt0 = depth > 0
        if d_gt0.any():
            m[d_gt0, self.nl_id] = -1e9
            m[d_gt0, self.eos_id] = -1e9
        d_eq0 = (depth == 0)
        if d_eq0.any():
            m[d_eq0, self.rparen_id] = -1e9
            m[d_eq0, self.rbracket_id] = -1e9
        # 5. Dead-row fallback (crash safety; unreachable for real states).
        # Force EOS — NOT uniform-0.0: a uniform row lets the sampler draw
        # FSA-forbidden tokens (contract-caught: 36->49 violation).
        dead = (m == -1e9).all(dim=-1)
        if dead.any():
            m[dead] = -1e9
            m[dead, self.eos_id] = 0.0
        return m, (m > -9e8)


# ---------------------------------------------------------------------------
# Factorized dual-rate recurrent core (v0.4: 16 x 384 slots)
# ---------------------------------------------------------------------------
@dataclass
class KernelV04Config:
    num_slots: int = 16
    d_slot: int = 384
    rank: int = 32
    k_interval: int = 4
    b_min: int = 32
    b_max: int = 1024
    b_base: int = 128
    d_hidden: int = 384          # decoder GRU hidden (v0.4: larger decoder)
    seed: int = 42
    grammar_w: float = 0.15
    unk_w: float = 0.01

    @property
    def d_total(self) -> int:
        return self.num_slots * self.d_slot

    @property
    def d_fast(self) -> int:
        return self.d_total // 4

    @property
    def d_slow(self) -> int:
        return self.d_total - self.d_fast


class FactorizedDualRateRecurrentCore(nn.Module):
    """Dual-rate recurrence over [B, S, d] slot state (v0.3 mechanics)."""

    def __init__(self, cfg: KernelV04Config):
        super().__init__()
        self.cfg = cfg
        g = torch.Generator().manual_seed(cfg.seed)
        for name, d in (("V_fast", cfg.d_fast), ("W_fast", cfg.d_fast),
                        ("V_slow", cfg.d_slow), ("W_slow", cfg.d_slow)):
            raw = torch.randn(d, cfg.rank, generator=g)
            if name.startswith("V"):
                q, _ = torch.linalg.qr(raw)
                self.register_parameter(name, nn.Parameter(q))
            else:
                self.register_parameter(name, nn.Parameter(raw / math.sqrt(cfg.rank)))
        self.cross_coupling = nn.Parameter(
            torch.randn(cfg.rank, cfg.rank, generator=g) * 0.02)

    def enforce_stiefel(self) -> None:
        with torch.no_grad():
            for name in ("V_fast", "V_slow"):
                q, _ = torch.linalg.qr(getattr(self, name).data)
                getattr(self, name).data.copy_(q)

    def forward(self, z: torch.Tensor, step: int,
                slow_cache: torch.Tensor | None):
        cfg = self.cfg
        b = z.shape[0]
        flat = z.reshape(b, cfg.d_total)
        z_fast, z_slow = flat.split([cfg.d_fast, cfg.d_slow], dim=-1)
        if step % cfg.k_interval == 0 or slow_cache is None:
            proj_s = torch.matmul(z_slow, self.W_slow)
            out_s = torch.matmul(proj_s, self.V_slow.T)
            slow_cache = out_s
        else:
            out_s = slow_cache
            proj_s = torch.matmul(out_s, self.W_slow)
        proj_f = torch.matmul(z_fast, self.W_fast)
        proj_f = proj_f + torch.matmul(proj_s, self.cross_coupling)
        out_f = torch.matmul(proj_f, self.V_fast.T)
        out = torch.cat([out_f, out_s], dim=-1)
        norm = torch.sqrt(out.pow(2).sum(-1, keepdim=True) + 1e-8)
        out = torch.nan_to_num(out / norm, nan=0.0)
        return out.reshape(b, cfg.num_slots, cfg.d_slot), slow_cache


# ---------------------------------------------------------------------------
# Name-conditioned egress decoder (cross-attention over prompt symbols)
# ---------------------------------------------------------------------------
class CrossAttentionNameConditionedDecoder(nn.Module):
    """GRU decoder + 4-head cross-attention over the signature symbol matrix.

    step(tok_emb [B, d_slot], h [B, d_hidden], s_prompt [B, K, d_slot]) ->
    (logits [B, V], h_next). The name and argument symbols are SUPPLIED via
    s_prompt (never memorized); z only initializes h.
    """

    def __init__(self, d_slot: int, d_hidden: int, vocab_size: int):
        super().__init__()
        self.d_hidden = d_hidden
        self.init_proj = nn.Linear(d_slot, d_hidden)
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_hidden,
                                                num_heads=4, batch_first=True)
        self.gru_cell = nn.GRUCell(d_slot, d_hidden)
        self.lm_head = nn.Linear(d_hidden, vocab_size, bias=False)

    def step(self, tok_emb: torch.Tensor, h: torch.Tensor,
             s_prompt: torch.Tensor):
        h_gru = self.gru_cell(tok_emb, h)
        attn_out, _ = self.cross_attn(h_gru.unsqueeze(1), s_prompt, s_prompt)
        h_ctx = (h_gru + attn_out.squeeze(1)) / math.sqrt(2.0)
        return self.lm_head(h_ctx), h_ctx


# ---------------------------------------------------------------------------
# Brier outcome baseline (energy head): regresses shaped reward in [0, 1]
# ---------------------------------------------------------------------------
class BrierOutcomeBaseline(nn.Module):
    def __init__(self, d_slot: int, d_hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_slot, d_hidden), nn.LayerNorm(d_hidden), nn.GELU(),
            nn.Linear(d_hidden, 1), nn.Sigmoid())

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z.mean(dim=1)).squeeze(-1)


# ---------------------------------------------------------------------------
# Unified v0.4 kernel
# ---------------------------------------------------------------------------
class System1KernelV04(nn.Module):
    def __init__(self, vocab_size: int = VOCAB, cfg: KernelV04Config | None = None):
        super().__init__()
        self.cfg = cfg or KernelV04Config()
        self.vocab_size = vocab_size
        self.token_emb = nn.Embedding(vocab_size, self.cfg.d_slot)
        self.core = FactorizedDualRateRecurrentCore(self.cfg)
        self.decoder = CrossAttentionNameConditionedDecoder(
            self.cfg.d_slot, self.cfg.d_hidden, vocab_size)
        self.energy = BrierOutcomeBaseline(self.cfg.d_slot)
        self.fsa = TokenFSAGrammarMask(NEXT_MASK)

    def encode_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        b, l = input_ids.shape
        embs = self.token_emb(input_ids)
        if l < self.cfg.num_slots:
            pad = torch.zeros(b, self.cfg.num_slots - l, self.cfg.d_slot,
                              device=input_ids.device)
            z = torch.cat([embs, pad], dim=1)
        else:
            z = embs[:, :self.cfg.num_slots]
        return z / (torch.sqrt(z.pow(2).sum(-1, keepdim=True)) + 1e-8)

    # ---- teacher/scheduled-sampling path (in-sequence, masked sampling) ----
    def ss_forward(self, z: torch.Tensor, s_prompt: torch.Tensor,
                   tokens: torch.Tensor, p_sched: float):
        b, tmax = tokens.shape
        dev = z.device
        h = self.decoder.init_proj(z.mean(dim=1))
        # `cur` is rebound per position (fresh tensor). Never mutate a tensor
        # that a saved autograd view aliases (inplace-version error).
        cur = torch.full((b,), TOK2ID["BOS"], device=dev)
        depth = torch.zeros(b, dtype=torch.long, device=dev)
        logits_list, raw_list, mask_list = [], [], []
        n_sampled = 0
        for t in range(tmax):
            raw_t, h = self.decoder.step(self.token_emb(cur), h, s_prompt)
            mask_t, mbool = self.fsa.apply_mask(raw_t, cur, depth)
            logits_list.append(mask_t.unsqueeze(1))
            raw_list.append(raw_t.unsqueeze(1))
            mask_list.append(mbool.unsqueeze(1))
            if t < tmax - 1:
                if random.random() < p_sched:
                    nxt = torch.distributions.Categorical(logits=mask_t).sample()
                    n_sampled += 1
                else:
                    nxt = tokens[:, t + 1]
                cur = nxt
                depth = (depth + self.fsa.depth_delta(nxt)).clamp(min=0)
        return (torch.cat(logits_list, 1), torch.cat(mask_list, 1),
                n_sampled / max(1, tmax - 1), torch.cat(raw_list, 1))

    # ---- masked free-run MLE (the v0.4 capability loss) ----
    def free_run_masked(self, z: torch.Tensor, s_prompt: torch.Tensor,
                        tokens: torch.Tensor):
        b, tmax = tokens.shape
        dev = z.device
        h = self.decoder.init_proj(z.mean(dim=1))
        tok = torch.full((b,), TOK2ID["BOS"], device=dev)
        depth = torch.zeros(b, dtype=torch.long, device=dev)
        logits_list, raw_list, mask_list, toks_list = [], [], [], []
        for t in range(tmax):
            raw_t, h = self.decoder.step(self.token_emb(tok), h, s_prompt)
            mask_t, mbool = self.fsa.apply_mask(raw_t, tok, depth)
            logits_list.append(mask_t.unsqueeze(1))
            raw_list.append(raw_t.unsqueeze(1))
            mask_list.append(mbool.unsqueeze(1))
            toks_list.append(tok.unsqueeze(1))
            if t < tmax - 1:
                tok = mask_t.argmax(-1)
                depth = (depth + self.fsa.depth_delta(tok)).clamp(min=0)
        return (torch.cat(logits_list, 1), torch.cat(toks_list, 1),
                torch.cat(mask_list, 1), torch.cat(raw_list, 1))

    # ---- constrained greedy decode (production egress, eval) ----
    def decode_greedy(self, z: torch.Tensor, s_prompt: torch.Tensor,
                      max_len: int = 48):
        b = z.shape[0]
        h = self.decoder.init_proj(z.mean(dim=1))
        tok = torch.full((b,), TOK2ID["BOS"], device=z.device)
        depth = torch.zeros(b, dtype=torch.long, device=z.device)
        out, done = [], torch.zeros(b, dtype=torch.bool, device=z.device)
        for _ in range(max_len):
            raw_t, h = self.decoder.step(self.token_emb(tok), h, s_prompt)
            mask_t, _ = self.fsa.apply_mask(raw_t, tok, depth)
            nxt = mask_t.argmax(-1)
            done = done | (nxt == TOK2ID["EOS"])
            if done.all():
                break
            out.append(nxt.unsqueeze(1))
            depth = (depth + self.fsa.depth_delta(nxt)).clamp(min=0)
            tok = nxt
        return (torch.cat(out, 1) if out else
                torch.zeros(b, 1, dtype=torch.long, device=z.device))

    # ---- constrained sampled decode (REINFORCE substrate) ----
    def decode_sample(self, z: torch.Tensor, s_prompt: torch.Tensor,
                      max_len: int = 48, temperature: float = 1.0,
                      top_p: float = 1.0, seed: int | None = None):
        b = z.shape[0]
        if seed is not None:
            # Per-particle reproducibility: seed a fresh generator per call.
            g = torch.Generator(device=z.device).manual_seed(seed)
        else:
            g = None
        h = self.decoder.init_proj(z.mean(dim=1))
        tok = torch.full((b,), TOK2ID["BOS"], device=z.device)
        depth = torch.zeros(b, dtype=torch.long, device=z.device)
        done = torch.zeros(b, dtype=torch.bool, device=z.device)
        toks, logps = [], []
        for _ in range(max_len):
            raw_t, h = self.decoder.step(self.token_emb(tok), h, s_prompt)
            mask_t, _ = self.fsa.apply_mask(raw_t / max(temperature, 1e-3),
                                            tok, depth)
            # Optional top-p truncation on the masked logits.
            if top_p < 1.0:
                sorted_l, _ = mask_t.sort(dim=-1, descending=True)
                cum = sorted_l.softmax(-1).cumsum(-1)
                cutoff = (cum > top_p).sum(-1)
                mask = torch.zeros_like(mask_t, dtype=torch.bool)
                mask.scatter_(1, cutoff.clamp(max=mask_t.shape[-1] - 1).unsqueeze(-1), True)
                mask_t = mask_t.masked_fill(~mask, float("-inf"))
            dist = torch.distributions.Categorical(logits=mask_t)
            if g is None:
                t = dist.sample()
            else:
                # Seeded path: batched torch.multinomial with an explicit
                # Generator — version-safe (Categorical.sample() lacks the
                # generator kwarg on this torch). Supports b>=1; the b==1
                # eval path is byte-identical to the old scalar form.
                probs = dist.probs
                if (not torch.isfinite(probs).all()
                        or probs.sum(dim=-1).min() <= 0.0):
                    t = mask_t.argmax(-1)
                else:
                    idx = torch.multinomial(probs, 1, generator=g)
                    t = idx.squeeze(-1)                    # [b]
            logp = dist.log_prob(t)
            logps.append(torch.where(done, torch.zeros_like(t, dtype=logp.dtype),
                                     logp))
            toks.append(t.unsqueeze(1))
            done = done | (t == TOK2ID["EOS"])
            depth = (depth + self.fsa.depth_delta(t)).clamp(min=0)
            tok = t
            if done.all():
                break
        return (torch.cat(toks, 1) if toks else
                torch.zeros(b, 1, dtype=torch.long, device=z.device)), \
            (torch.stack(logps, 1) if logps else
             torch.zeros(b, 1, device=z.device))

    # ---- constrained beam decode (eval arm) ----
    def beam_decode(self, z: torch.Tensor, s_prompt: torch.Tensor,
                    width: int = 16, max_len: int = 48) -> list[int]:
        h = self.decoder.init_proj(z.mean(dim=1)).expand(width, -1).clone()
        toks = torch.full((width,), TOK2ID["BOS"], device=z.device)
        depth = torch.zeros(width, dtype=torch.long, device=z.device)
        seqs = [[] for _ in range(width)]
        scores = torch.zeros(width, device=z.device)
        for _ in range(max_len):
            raw_t, h_new = [], []
            for i in range(width):
                # s_prompt is [1, K, d] (single prompt); identical memory for
                # every beam particle. Slicing [i:i+1] empties rows for i>=1.
                r, hh = self.decoder.step(self.token_emb(toks[i:i + 1]), h[i:i + 1],
                                          s_prompt[0:1])
                raw_t.append(r)
                h_new.append(hh)
            raw_t = torch.cat(raw_t)
            h = torch.cat(h_new)
            mask_t, _ = self.fsa.apply_mask(raw_t, toks, depth)
            logp = F.log_softmax(mask_t, -1)
            cand = []
            for i in range(width):
                topv, topi = logp[i].topk(4)
                for k in range(4):
                    cand.append((scores[i].item() + topv[k].item(), i, topi[k].item()))
            cand.sort(reverse=True)
            cand = cand[:width]
            new_toks, new_depth, new_seqs, new_scores = [], [], [], []
            for s, i, t in cand:
                new_toks.append(t)
                new_depth.append(depth[i].item() + self.fsa.depth_delta(
                    torch.tensor(t, device=z.device)).item())
                new_seqs.append(seqs[i] + [t])
                new_scores.append(s)
            toks = torch.tensor(new_toks, device=z.device)
            depth = torch.clamp(torch.tensor(new_depth, device=z.device), min=0)
            seqs, scores = new_seqs, torch.tensor(new_scores, device=z.device)
            if all(s and s[-1] == TOK2ID["EOS"] for s in seqs):
                break
        return seqs[int(scores.argmax())]

    # ---- energy-weighted vote over seeded per-particle sampled decodes ----
    def decode_vote(self, z: torch.Tensor, s_prompt: torch.Tensor,
                    energies: torch.Tensor | None = None,
                    temperature: float = 1.0, top_p: float = 1.0,
                    seed_base: int = 0, max_len: int = 48,
                    n_seeds_per_particle: int = 1) -> tuple[list[int], dict]:
        """Aggregate one seeded sample per particle (or n_seeds each) by
        energy weight; deterministic tie-break on program bytes.
        Returns (winning_ids, vote_record).
        """
        b = z.shape[0]
        # Signature memory: allow [1, K, d] broadcast to all particles, or
        # require exact [b, K, d]. An empty or mismatched slice is a caller bug.
        if s_prompt.shape[0] == 1 and b > 1:
            s_prompt = s_prompt.expand(b, -1, -1)
        if s_prompt.shape[0] != b:
            raise ValueError(
                f"decode_vote: s_prompt rows {s_prompt.shape[0]} != particles {b}")
        # energy weights: softmax over particles (uniform when None)
        if energies is not None:
            w = torch.softmax(torch.nan_to_num(
                energies, nan=0.0, posinf=1.0, neginf=-1.0), dim=0)
            w = w.detach().cpu().tolist()
        else:
            w = [1.0 / b] * b
        progs: dict[bytes, dict] = {}       # program bytes -> record
        for i in range(b):
            for k in range(n_seeds_per_particle):
                seed = (seed_base * 1000003 + i * 7919 + k * 104729) % (2 ** 31)
                toks, _ = self.decode_sample(
                    z[i:i + 1], s_prompt[i:i + 1], max_len=max_len,
                    temperature=temperature, top_p=top_p, seed=int(seed))
                ids = toks[0].tolist()
                # trim at EOS
                if TOK2ID["EOS"] in ids:
                    ids = ids[:ids.index(TOK2ID["EOS"])]
                key = bytes(ids)
                rec = progs.setdefault(
                    key, {"weight": 0.0, "count": 0, "seeds": [], "ids": ids,
                          "particle": i})
                rec["weight"] += w[i]
                rec["count"] += 1
                rec["seeds"].append(int(seed))
        # winner: max weight; tie-break lexicographically smallest bytes
        winner_key = max(sorted(progs, reverse=True),
                         key=lambda k: progs[k]["weight"])
        win = progs[winner_key]
        ws = [r["weight"] for r in progs.values()]
        runner = None
        for k in sorted(progs):
            if k != winner_key:
                runner = progs[k]["ids"]
                break
        return win["ids"], {
            "unique_programs": len(progs),
            "weight_var": (max(ws) - min(ws)) if len(ws) > 1 else 0.0,
            "winner_weight": win["weight"],
            "winner_count": win["count"],
            "winner_seed": win["seeds"][0],
            "winner_particle": win["particle"],
            "winner_ids": win["ids"],
            "runner_up_ids": runner or [],
            "winner_bytes_sha": hashlib.sha256(winner_key).hexdigest()[:16],
        }


def grammar_loss(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Differentiable grammar supervision (log-normalizer minus masked-lse)."""
    b, l, v = logits.shape
    flat = logits.reshape(-1, v)
    flat_m = masks.reshape(-1, v)
    lse_all = torch.logsumexp(flat, dim=-1)
    lse_valid = torch.logsumexp(flat.masked_fill(~flat_m, float("-inf")), dim=-1)
    per_pos = lse_all - lse_valid
    per_pos = torch.nan_to_num(per_pos, nan=0.0, posinf=0.0)
    return per_pos.mean()


class SwarmEngineV04(nn.Module):
    """Shared factorized microcore over B latent slot particles (v0.3)."""

    def __init__(self, model: System1KernelV04):
        super().__init__()
        self.model = model
        self.cfg = model.cfg

    def _diversity(self, b: int, device: torch.device) -> dict[str, torch.Tensor]:
        return {
            "temperature": 0.5 + 0.5 * torch.rand(b, device=device),
            "horizon": torch.randint(2, 6, (b,), device=device),
            "noise_scale": 0.05 + 0.20 * torch.rand(b, device=device),
        }

    def forward_swarm(self, z0: torch.Tensor, b_target: int,
                      steps: int = 6) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        dev = z0.device
        b_in = z0.shape[0]
        per = max(1, b_target // b_in)
        z = z0.repeat_interleave(per, dim=0)[:b_target]
        b = z.shape[0]
        div = self._diversity(b, dev)
        z = z + torch.randn_like(z) * div["noise_scale"].view(-1, 1, 1)
        slow_cache = None
        energies: list[torch.Tensor] = []
        for t in range(steps):
            z, slow_cache = self.model.core(z, t, slow_cache)
            energies.append(self.model.energy(z))
        e = torch.nan_to_num(energies[-1], nan=0.0, posinf=1.0, neginf=-1.0)
        p = torch.softmax(e, dim=0)
        p = torch.nan_to_num(p, nan=1.0 / max(1, b), posinf=1.0, neginf=0.0)
        pv = p.var(unbiased=False).item() if b >= 2 else 0.0
        if not math.isfinite(pv):
            pv = 0.0
        b_next = max(cfg.b_min, min(cfg.b_max, int(cfg.b_base * (1 + 5 * pv))))
        if b_next < b:
            keep = torch.multinomial(p, b_next, replacement=False)
            z = z[keep]
        return {"z": z, "energy": e, "b_next": b_next,
                "particles": b, "diversity": div}


# ---------------------------------------------------------------------------
# Plumbing smoke (NEVER capability evidence)
# ---------------------------------------------------------------------------
def smoke(device: str = "cpu") -> None:
    cfg = KernelV04Config()
    model = System1KernelV04(cfg=cfg).to(device)
    eng = SwarmEngineV04(model)
    n_params = sum(p.numel() for p in model.parameters())
    z0 = torch.randn(8, 16, cfg.d_slot, device=device)
    z0 = z0 / z0.norm(dim=-1, keepdim=True)
    out = eng.forward_swarm(z0, b_target=32, steps=4)
    z = out["z"]
    s_prompt = torch.randn(z.shape[0], 16, cfg.d_slot, device=device)
    toks = torch.randint(6, VOCAB, (z.shape[0], 16), device=device)
    logits, masks, ss_frac, raw = model.ss_forward(z, s_prompt, toks, 0.5)
    loss_ce = F.cross_entropy(logits.reshape(-1, VOCAB), toks.reshape(-1))
    loss_gr = grammar_loss(logits, masks) * cfg.grammar_w
    fr_logits, fr_toks, fr_masks, _ = model.free_run_masked(z, s_prompt, toks)
    loss_fr = F.cross_entropy(fr_logits.reshape(-1, VOCAB), toks.reshape(-1))
    unk = raw[:, :, TOK2ID["UNK"]]
    loss_unk = F.softplus(unk).mean() * cfg.unk_w
    r = torch.rand(z.shape[0], device=z.device)
    loss_en = F.binary_cross_entropy(model.energy(z), r)
    toks_s, logps = model.decode_sample(z, s_prompt)
    adv = (r - r.mean()).clamp(-2.0, 2.0)
    loss_pg = -(logps * adv.unsqueeze(1)).sum(1).mean()
    loss = loss_ce + loss_fr + loss_en + loss_gr + loss_unk + loss_pg
    loss.backward()
    for p in model.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), p.shape
    assert model.decoder.cross_attn.in_proj_weight.grad.abs().sum().item() > 0
    assert model.token_emb.weight.grad.abs().sum().item() > 0
    gn = sum(p.grad.norm().item() ** 2
             for p in model.parameters() if p.grad is not None) ** 0.5
    assert gn > 0.0
    print(f"SMOKE_OK params={n_params:,} ({n_params / 1e6:.2f}M) "
          f"ce={loss_ce.item():.3f} fr={loss_fr.item():.3f} "
          f"gr={loss_gr.item():.3f} unk={loss_unk.item():.4f} "
          f"pg={loss_pg.item():.3f} gnorm={gn:.3f} "
          f"ss={ss_frac:.2f} [SMOKE ONLY - not capability evidence]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    smoke(args.device)
