"""ARC Egress Transducer Contract — Phase 6 (bounded, default-off).

Bridges the continuous UWE wave action path ([num_blocks, 8]) to the
HENRIUnifiedEgressTransducer neural decoder head, producing a COMPLETE
(GameAction, data) action tuple with fail-closed semantics.

Contracts (per Phase 6 task packet and henri-agent-integration):
1. An ARC action is (GameAction, data), not a bare enum. Coordinate-bearing
   actions (ACTION6) receive screen-space payloads via arc_action_payloads.
2. The transducer consumes FLAT [D] waves; ARC candidates are [num_blocks, 8]
   Clifford UWE. A deterministic row-major reshape is the ONLY boundary
   mapping (documented; never a lossy projection).
3. The 32k code-token vocabulary is NOT an action vocabulary. The
   action-legal vocabulary occupies the FIRST N logit positions in a
   deterministic order over the environment's allowed actions. Positions
   >= N are code tokens and are never interpreted as actions.
4. Missing checkpoint / decode failure / illegal shape raise typed
   EgressFailClosedError. There is NO silent bare-enum fallback.
5. SGLD adaptation requires in-context demonstration pairs (X_i, Y_i).
   Absent demos raise NoDemonstrationsError (emitted as
   BLOCKED_NO_DEMONSTRATIONS by the caller). Labels are never bootstrapped
   and pseudo-demonstrations are never fabricated.
6. Adaptation uses the corrected protocol (adapt_in_context_sgld_wave):
   frozen soft targets snapshot before adaptation, scheduled thermal noise
   T(t)=T0(1+0.05t)^-0.55, unit-normalized Langevin increments, Cholesky
   Stiefel retraction, Sagnac term L = CE + 0.25 * (1 - cos(p, p_target)).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch


class EgressFailClosedError(RuntimeError):
    """Raised when the egress transducer cannot legally produce an action."""


class NoDemonstrationsError(RuntimeError):
    """Raised when SGLD adaptation is requested without demo pairs."""


@dataclass(frozen=True)
class EgressDecodeResult:
    action: object
    action_index: int
    action_name: str
    action_logits: torch.Tensor  # [N] action-legal logits
    action_probs: torch.Tensor   # [N] softmax over action-legal logits
    top3: List[Tuple[str, float]]
    entropy_bits: float          # action-legal logit entropy
    token_entropy_bits: float    # full-vocab logit entropy (diagnostic)


class ActionEgressVocabulary:
    """Deterministic id <-> GameAction mapping over the allowed action set.

    The action-legal logits are the first N positions of the decoder's logit
    vector. Positions >= N are code tokens and are never interpreted as
    actions (fail-closed by construction).
    """

    def __init__(self, action_enum_class: object, allowed_actions: Sequence):
        self.action_enum_class = action_enum_class
        seen = set()
        for a in allowed_actions:
            if not hasattr(a, "name"):
                raise EgressFailClosedError(f"action {a!r} has no .name")
            if a in seen:
                raise EgressFailClosedError(f"duplicate action in allowed set: {a!r}")
            seen.add(a)
        self.actions: List = sorted(set(allowed_actions), key=lambda a: a.name)
        if len(self.actions) != len(set(allowed_actions)):
            raise EgressFailClosedError("allowed set collapsed under sorting")
        self.id_to_action: Dict[int, object] = {
            i: a for i, a in enumerate(self.actions)
        }
        self.action_to_id: Dict[object, int] = {
            a: i for i, a in enumerate(self.actions)
        }

    @property
    def n_actions(self) -> int:
        return len(self.actions)


def flatten_uwe(wave: torch.Tensor, d_model: int) -> torch.Tensor:
    """Deterministic [num_blocks, 8] -> [1, D] row-major flatten.

    D must equal num_blocks * 8 (65536 at CUDA scale, 512 at reduced CPU
    scale). Any other shape raises EgressFailClosedError.
    """
    if wave.dim() != 2 or wave.shape[-1] != 8:
        raise EgressFailClosedError(
            f"expected [num_blocks, 8] UWE, got {tuple(wave.shape)}"
        )
    flat = wave.reshape(-1)
    if flat.numel() != d_model:
        raise EgressFailClosedError(
            f"flatten numel {flat.numel()} != d_model {d_model}"
        )
    return flat.unsqueeze(0).to(torch.float32)


def _full_vocab_entropy(logits: torch.Tensor) -> float:
    p = torch.softmax(logits, dim=-1)
    return float(-(p * torch.log2(p + 1e-12)).sum().item())


def decode_action_egress(
    transducer: object,
    predicted_wave: torch.Tensor,
    vocab: ActionEgressVocabulary,
    device: str = "cpu",
    require_loaded: bool = True,
) -> EgressDecodeResult:
    """Fail-closed action decode through the transducer head.

    Args:
        transducer: HENRIUnifiedEgressTransducer (or contract-compatible stub).
        predicted_wave: [num_blocks, 8] chosen candidate wave.
        vocab: deterministic action-legal vocabulary for this environment.
        device: target device for the decode.
        require_loaded: if True, raise unless checkpoint_load_status == "LOADED".

    Returns:
        EgressDecodeResult with the legal action, logits, probs, top3,
        action entropy and full-vocab token entropy.
    """
    status = getattr(transducer, "checkpoint_load_status", "SKIPPED_POLICY_DISABLED")
    if require_loaded and status != "LOADED":
        raise EgressFailClosedError(
            f"transducer not LOADED (status={status})"
        )
    d_model = getattr(transducer, "d_model", None)
    if d_model is None:
        raise EgressFailClosedError("transducer has no d_model")
    flat = flatten_uwe(predicted_wave, d_model)
    unbinder = getattr(transducer, "unbinder", None)
    if unbinder is None:
        raise EgressFailClosedError("transducer has no unbinder head")
    with torch.no_grad():
        logits = unbinder.forward(flat.to(device)).to(torch.float32)  # [1, V]
    if logits.shape[-1] < vocab.n_actions:
        raise EgressFailClosedError(
            f"logit vocab {logits.shape[-1]} smaller than action vocab {vocab.n_actions}"
        )
    action_logits = logits[0, : vocab.n_actions]
    probs = torch.softmax(action_logits, dim=-1)
    idx = int(torch.argmax(action_logits).item())
    action = vocab.id_to_action[idx]
    entropy_bits = entropy_bits_of(probs)  # normalized in [0, 1]
    k = min(3, vocab.n_actions)
    top_idx = torch.topk(action_logits, k=k).indices.tolist()
    top3 = [(vocab.id_to_action[i].name, float(probs[i])) for i in top_idx]
    return EgressDecodeResult(
        action=action,
        action_index=idx,
        action_name=action.name,
        action_logits=action_logits.detach().cpu(),
        action_probs=probs.detach().cpu(),
        top3=top3,
        entropy_bits=entropy_bits,
        token_entropy_bits=_full_vocab_entropy(logits[0]),
    )


def adapt_sgld_from_demos(
    transducer: object,
    demo_pairs: Sequence[Tuple],
    tokenizer: object,
    device: str = "cpu",
    steps: int = 500,
    seed: int = 0,
    d_model: Optional[int] = None,
) -> Optional[Dict[str, float]]:
    """Online test-time SGLD adaptation on in-context demo pairs.

    Encodes each (input, output) grid through the production tokenizer path
    (encode_spatial_grid), flattens [num_blocks, 8] -> [D], and runs
    adapt_in_context_sgld_wave (corrected protocol).

    Returns None when steps <= 0. Raises NoDemonstrationsError when
    demo_pairs is empty.
    """
    if steps <= 0:
        return None
    if not demo_pairs:
        raise NoDemonstrationsError(
            "no in-context demonstration pairs available for SGLD adaptation"
        )
    unbinder = getattr(transducer, "unbinder", None)
    if unbinder is None or not hasattr(unbinder, "adapt_in_context_sgld_wave"):
        raise EgressFailClosedError("transducer unbinder lacks adapt_in_context_sgld_wave")
    encode = getattr(tokenizer, "encode_spatial_grid", None)
    if encode is None:
        raise EgressFailClosedError("tokenizer lacks encode_spatial_grid")
    active: List[torch.Tensor] = []
    target: List[torch.Tensor] = []
    for x, y in demo_pairs:
        _x = x.tolist() if hasattr(x, "tolist") else x
        _y = y.tolist() if hasattr(y, "tolist") else y
        wx = encode(_x).squeeze(0)
        wy = encode(_y).squeeze(0)
        if wx.dim() == 2 and wx.shape[-1] == 8:
            wx = wx.reshape(-1)
        if wy.dim() == 2 and wy.shape[-1] == 8:
            wy = wy.reshape(-1)
        active.append(wx)
        target.append(wy)
    A = torch.stack(active).to(device).to(torch.float32)
    T_ = torch.stack(target).to(device).to(torch.float32)
    metrics = unbinder.adapt_in_context_sgld_wave(A, T_, steps=steps, seed=seed)
    if isinstance(metrics, dict):
        metrics["demo_pair_count"] = len(demo_pairs)
        metrics["steps"] = steps
    return metrics


def reset_decoder_optimizer(transducer: object) -> bool:
    """Per-episode decoder reset: fresh AdamW optimizer state.

    Prevents SGLD adaptation from leaking across episodes. Returns True when
    the reset happened.
    """
    unbinder = getattr(transducer, "unbinder", None)
    if unbinder is None or not hasattr(unbinder, "parameters"):
        return False
    unbinder.optimizer = torch.optim.AdamW(
        unbinder.parameters(), lr=1e-3, weight_decay=1e-4
    )
    return True


def entropy_bits_of(probs: torch.Tensor) -> float:
    """Normalized entropy in bits over the action-legal distribution."""
    if probs.numel() <= 1:
        return 0.0
    h = float(-(probs * torch.log2(probs + 1e-12)).sum().item())
    return h / math.log2(probs.numel()) if probs.numel() > 1 else h


# ===========================================================================
# Typed Probe Contract — Zone A typed epistemic probe boundary
#
# DEFAULT-OFF. When HENRI_TYPED_PROBE_CONTRACT is unset this section is
# unreachable from the shipped decode path: it ADDS symbols and changes no
# existing function, so decode_action_egress / adapt_sgld_from_demos keep their
# byte-identical behaviour (contracts A1/R4, asserted by test).
#
# WHY THIS EXISTS (measured 2026-09-17, references/zone_a_typed_probe_contract.md)
#   A typed answer cannot enter the wave core as a SCALAR rotor e^{i*theta}.
#   The precise statement differs per representation, and both were measured:
#     - COMPLEX unit-modulus qFHRR family: a global phase is UNOBSERVABLE. The
#       phase-invariant overlap |<psi',psi>|/(|psi'|*|psi|) = 0.999999940 while
#       the raw vector moves by ||psi'-psi|| ~ 64. Naive instrumentation reads
#       that as a large update when the state carried no information.
#     - REAL [cos,sin] representation: a global phase IS observable, as a real
#       overlap of exactly cos(theta), but a SINGLE scalar still cannot select
#       one of K answers.
#   Either way the conclusion is identical: an answer must be bound
#   per-dimension into all D dimensions (key (x) value phase sum) before it can
#   change a decision.
#
#   Consequence encoded here: the scalar-rotor path is rejected at runtime, so
#   a mock feedback loop cannot be mistaken for learning.
#
# TERMINOLOGY: HENRI names are `choice` / `graded` / `truth`. These are the
#   semantics of TypeSafe's Choice / Score / Noul; the vendor labels are not
#   imported into the architecture.
# ===========================================================================

PROBE_CONTRACT_FLAG = "HENRI_TYPED_PROBE_CONTRACT"
K_PHASE = 256            # Z_256 phase ring, matches qfhrr_kernels.K_PHASE

QW_CHOICE = "choice"
QW_GRADED = "graded"
QW_TRUTH = "truth"
QUESTION_WORDS = (QW_CHOICE, QW_GRADED, QW_TRUTH)

ST_OK = "OK"
ST_ABSTAIN_LOW_CONF = "ABSTAIN_LOW_CONF"
ST_ABSTAIN_NO_ORDER = "ABSTAIN_NO_ORDER"
ST_ABSTAIN_INVALID = "ABSTAIN_INVALID"
PROBE_STATUSES = (ST_OK, ST_ABSTAIN_LOW_CONF, ST_ABSTAIN_NO_ORDER, ST_ABSTAIN_INVALID)

PROB_SUM_TOL = 1e-6


class ProbeContractViolation(EgressFailClosedError):
    """Raised when a ProbeEnvelope or its feedback path violates the contract."""


class ScalarRotorRejected(ProbeContractViolation):
    """Raised when feedback applies a scalar (gauge) rotor to a D-dim wave."""


def probe_contract_enabled(env: Optional[Dict[str, str]] = None) -> bool:
    """True only when the typed-probe contract flag is explicitly enabled."""
    src = os.environ if env is None else env
    return str(src.get(PROBE_CONTRACT_FLAG, "0")).strip().lower() in {"1", "true", "yes", "on"}


def confidence_from_probabilities(probs: Sequence[float]) -> float:
    """THE single confidence definition site. Recalibrate here, not at call sites.

    Definition: confidence = 1 - H(p)/log(K), clamped to [0, 1].
    A point mass -> 1.0. A uniform distribution -> 0.0.

    This is monotone in concentration (R1): moving probability mass onto the
    modal option can only raise it. It is a DIFFERENT statistic from any model's
    internal score, and it is not a guarantee of correctness. Whether a
    confidence of c corresponds to an accuracy of c is a question for
    `henri_probe_calibration.evaluate_calibration` and is NOT answered here.
    """
    vals = [float(p) for p in probs]
    if not vals:
        raise ProbeContractViolation("confidence_from_probabilities: empty distribution")
    if len(vals) == 1:
        return 1.0
    total = sum(vals)
    if total <= 0.0:
        raise ProbeContractViolation("confidence_from_probabilities: non-positive mass")
    norm = [v / total for v in vals]
    h = -sum(p * math.log(p + 1e-12) for p in norm if p > 0.0)
    h_max = math.log(len(norm))
    if h_max <= 0.0:
        return 1.0
    return max(0.0, min(1.0, 1.0 - h / h_max))


def _phase_codes(ids: Sequence[int], latent_dim: int, seed: int) -> torch.Tensor:
    """Deterministic Z_256 phase codes per id. Same id -> same code, always."""
    out = torch.empty(len(ids), latent_dim, dtype=torch.int64)
    for row, i in enumerate(ids):
        gen = torch.Generator().manual_seed(int(seed) + 1000 * int(i))
        out[row] = torch.randint(
            0, K_PHASE, (latent_dim,), generator=gen, dtype=torch.int64
        )
    return out


def bind_key_value_wave(
    key_id: int,
    value_id: int,
    num_blocks: int = 8192,
    block_dim: int = 8,
    key_seed: int = 101,
    value_seed: int = 202,
) -> torch.Tensor:
    """Bind (key (x) value) into a REAL [num_blocks, block_dim] wave.

    Binding is modular phase addition on the Z_256 ring -- the same algebra as
    `qfhrr_kernels.fhrr_bind` -- so the result is a valid member of the
    production UWE family and can be consumed by the existing egress path.

    This is the ONLY supported feedback channel: the rotation is per-dimension,
    never scalar. A scalar rotor raises ScalarRotorRejected (see
    `apply_wave_binding`).
    """
    if num_blocks < 2 or block_dim != 8:
        raise ProbeContractViolation(
            f"bind_key_value_wave: unsupported geometry [{num_blocks}, {block_dim}]"
        )
    d_model = num_blocks * block_dim
    latent = d_model // 2
    kc = _phase_codes([key_id], latent, key_seed)[0]
    vc = _phase_codes([value_id], latent, value_seed)[0]
    codes = (kc + vc) % K_PHASE
    theta = codes.to(torch.float32) * (2.0 * math.pi / K_PHASE)
    real = torch.cat([torch.cos(theta), torch.sin(theta)], dim=-1)
    return real.view(num_blocks, block_dim)


def retrieve_value_from_binding(
    wave: torch.Tensor,
    key_id: int,
    value_ids: Sequence[int],
    key_seed: int = 101,
    value_seed: int = 202,
) -> Tuple[int, float]:
    """Decode which value is bound to `key_id` in `wave`.

    Returns (index_into_value_ids, best_ring_cosine). Fails closed if the wave
    geometry is illegal. This is the readout used to prove that per-dimension
    binding carries answer identity. An identity codebook recovers its own
    binding partly BY CONSTRUCTION; that is not calibration evidence and not a
    generalisation claim.
    """
    if not value_ids:
        raise ProbeContractViolation("retrieve_value_from_binding: no candidate values")
    flat = wave.reshape(-1).to(torch.float32)
    d_model = flat.numel()
    if d_model % 2 != 0 or d_model < 16:
        raise ProbeContractViolation(
            f"retrieve_value_from_binding: illegal wave length {d_model}"
        )
    latent = d_model // 2
    cos_part, sin_part = flat[:latent], flat[latent:]
    theta = torch.atan2(sin_part, cos_part)
    codes = torch.round(theta * (K_PHASE / (2.0 * math.pi))).to(torch.int64) % K_PHASE
    kc = _phase_codes([key_id], latent, key_seed)[0]
    residual = (codes - kc) % K_PHASE
    candidates = _phase_codes(list(value_ids), latent, value_seed)
    delta = (residual.unsqueeze(0) - candidates) % K_PHASE
    ring_cos = torch.cos(delta.to(torch.float32) * (2.0 * math.pi / K_PHASE)).mean(dim=1)
    best = int(torch.argmax(ring_cos).item())
    return best, float(ring_cos[best].item())


def reject_scalar_rotor(rotor: torch.Tensor, d_model: int) -> None:
    """Fail closed when feedback would broadcast a scalar rotor over D dims.

    A numel-1 rotor is a global phase: a gauge transformation that cannot change
    the state. Rejecting it here is what stops a mock 'incessant learning' loop
    from being mistaken for a real update (contract A3).
    """
    n = int(rotor.numel())
    if n == 1:
        raise ScalarRotorRejected(
            "scalar rotor e^{i*theta} is a U(1) gauge transformation: it cannot "
            "change the wave state (measured normalized overlap 0.999999940). "
            "Bind the answer per-dimension with bind_key_value_wave instead."
        )
    if n != d_model:
        raise ScalarRotorRejected(
            f"rotor numel {n} != wave dim {d_model}; feedback must be elementwise"
        )


def apply_wave_binding(
    wave: torch.Tensor,
    rotor: torch.Tensor,
) -> torch.Tensor:
    """Apply elementwise phase feedback to a real wave. Scalar rotors rejected.

    The wave is split into its (cos, sin) halves; the rotor is a PHASE-SPACE
    vector of length latent = d_model // 2, one phase offset per latent
    dimension. A scalar (numel 1) rotor is rejected as a gauge no-op, and a
    d_model-sized vector is rejected as a misuse of the phase geometry.
    The unit-normalized real wave is returned in the SAME geometry.
    """
    flat = wave.reshape(-1).to(torch.float32)
    d_model = flat.numel()
    latent = d_model // 2
    reject_scalar_rotor(rotor, latent)
    cos_part, sin_part = flat[:latent], flat[latent:]
    theta = torch.atan2(sin_part, cos_part)
    r = rotor.reshape(-1).to(torch.float32)
    theta_new = theta + r[:latent]
    out = torch.cat([torch.cos(theta_new), torch.sin(theta_new)], dim=-1)
    return torch.nn.functional.normalize(out, p=2.0, dim=-1).view(wave.shape)


@dataclass(frozen=True)
class ProbeEnvelope:
    """One typed Zone A crossing. No `str` reaches a machine consumer.

    `status != OK` means the answer is NOT consumable: an abstention is a valid
    terminal result, never a fallback to a guessed answer (contract R3).
    Metadata is free-form and carried for provenance, not for judgement.
    """

    probe_id: int
    question_word: str
    state_snapshot_id: str
    probabilities: Tuple[float, ...]
    answer: object
    status: str = ST_OK
    option_ids: Tuple[int, ...] = ()
    wave_binding: Optional[torch.Tensor] = None
    confidence: float = 0.0
    metadata: Dict[str, object] = field(default_factory=dict)
    question_type: str = ""

    def __post_init__(self) -> None:
        if self.question_type and not self.question_word:
            object.__setattr__(self, "question_word", self.question_type)
        if self.question_word not in QUESTION_WORDS:
            raise ProbeContractViolation(
                f"question_word must be one of {QUESTION_WORDS}, got {self.question_word!r}"
            )
        if self.status not in PROBE_STATUSES:
            raise ProbeContractViolation(
                f"status must be one of {PROBE_STATUSES}, got {self.status!r}"
            )
        if not self.state_snapshot_id:
            raise ProbeContractViolation(
                "state_snapshot_id is mandatory: parallel probes on a rotating "
                "state are mutually incoherent without a pinned snapshot"
            )
        if self.status != ST_OK:
            return
        probs = tuple(float(p) for p in self.probabilities)
        if not probs:
            raise ProbeContractViolation("status OK requires a non-empty distribution")
        total = sum(probs)
        if abs(total - 1.0) > PROB_SUM_TOL:
            raise ProbeContractViolation(
                f"probabilities sum to {total!r}, expected 1 +/- {PROB_SUM_TOL}"
            )
        if any(p < 0.0 for p in probs):
            raise ProbeContractViolation("probabilities must be non-negative")
        expected = max(range(len(probs)), key=lambda i: probs[i])
        if isinstance(self.answer, int) and self.answer != expected:
            raise ProbeContractViolation(
                f"answer {self.answer} is not argmax {expected} of the distribution"
            )
        if self.option_ids and len(self.option_ids) != len(probs):
            raise ProbeContractViolation(
                f"option_ids length {len(self.option_ids)} != probabilities {len(probs)}"
            )
        if self.wave_binding is None:
            raise ProbeContractViolation(
                "status OK requires wave_binding: it is the sole feedback channel"
            )
        if self.question_word == QW_CHOICE and not self.option_ids:
            raise ProbeContractViolation("a choice probe requires a bounded option set")

    @property
    def is_ok(self) -> bool:
        return self.status == ST_OK

    @property
    def is_consumable(self) -> bool:
        """Only an OK probe may influence a decision (contract R3)."""
        return self.status == ST_OK

    def to_dict(self) -> dict:
        return {
            "probe_id": self.probe_id,
            "question_word": self.question_word,
            "state_snapshot_id": self.state_snapshot_id,
            "option_ids": list(self.option_ids),
            "probabilities": list(self.probabilities),
            "answer": self.answer,
            "status": self.status,
            "confidence": self.confidence,
            "has_wave_binding": self.wave_binding is not None,
            "metadata": dict(self.metadata),
        }


def consume_probe(envelope: ProbeEnvelope) -> ProbeEnvelope:
    """Gate a probe before a consumer acts on it. Raises when not consumable."""
    if not envelope.is_consumable:
        raise ProbeContractViolation(
            f"probe {envelope.probe_id} has status {envelope.status!r}: not consumable"
        )
    return envelope


def probe_from_logits(
    action_logits: torch.Tensor,
    option_ids: Sequence[int],
    state_snapshot_id: str,
    probe_id: int = 0,
    wave_binding: Optional[torch.Tensor] = None,
    metadata: Optional[Dict[str, object]] = None,
    low_confidence_floor: Optional[float] = None,
) -> ProbeEnvelope:
    """Adapt the EXISTING action-legal logits into a typed probe envelope.

    This is the wiring bridge: it reuses the decode path already in this module
    rather than adding a parallel head. When `low_confidence_floor` is set and
    the derived confidence is below it, the result abstains instead of guessing.
    """
    logits = action_logits.reshape(-1).to(torch.float32)
    if logits.numel() != len(option_ids):
        raise ProbeContractViolation(
            f"logits {logits.numel()} != option_ids {len(option_ids)}"
        )
    probs = torch.softmax(logits, dim=-1)
    plist = tuple(float(p) for p in probs.tolist())
    conf = confidence_from_probabilities(plist)
    status = ST_OK
    if low_confidence_floor is not None and conf < low_confidence_floor:
        status = ST_ABSTAIN_LOW_CONF
    if status != ST_OK or wave_binding is None:
        return ProbeEnvelope(
            probe_id=probe_id,
            question_word=QW_CHOICE,
            state_snapshot_id=state_snapshot_id,
            probabilities=plist,
            answer=int(torch.argmax(logits).item()),
            status=status,
            option_ids=tuple(int(o) for o in option_ids),
            wave_binding=None,
            confidence=conf,
            metadata=dict(metadata or {}),
        )
    return ProbeEnvelope(
        probe_id=probe_id,
        question_word=QW_CHOICE,
        state_snapshot_id=state_snapshot_id,
        probabilities=plist,
        answer=int(torch.argmax(logits).item()),
        status=ST_OK,
        option_ids=tuple(int(o) for o in option_ids),
        wave_binding=wave_binding,
        confidence=conf,
        metadata=dict(metadata or {}),
    )


def state_snapshot_id_of(wave: torch.Tensor) -> str:
    """Content hash of a pinned wave state. Cheap, deterministic, collision-shy."""
    import hashlib

    flat = wave.detach().to(torch.float32).reshape(-1).cpu().numpy()
    return hashlib.sha256(flat.tobytes()).hexdigest()[:16]
