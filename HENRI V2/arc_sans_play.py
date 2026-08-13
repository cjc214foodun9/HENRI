"""ARC SANS Epistemic Play + Action-Head Calibration (Phase 7.2 Step 2).

Success-and-Near-Success (SANS) self-generated trajectory buffer:

1. Epistemic play: before the formal rollout, execute bounded random/heuristic
   interactions in the LIVE arc_agi MDP (with screen-space payloads when the
   action requires them). Record (state_hidden, action_index, delta_nu) where
   delta_nu = externally observed frame-change cell count.
2. Commit an interaction to the SANS buffer ONLY when delta_nu >= 2 cells
   (strict floor: bare-enum single-cell noise is excluded).
3. Calibrate a fresh Linear(hidden -> |A|) action head by cross-entropy
   exclusively on the SANS buffer, with a seeded hold-out split.
4. Persist the checkpoint with full provenance (file sha256, state-dict
   sha256, dims, ordered action labels, calibration dataset digest, split
   identity, held-out metrics) under models/henri_sans_action_head.pt.
5. trained_action_head_active is set ONLY when held-out accuracy beats the
   majority-class baseline by a margin and a minimum sample floor is met.

Fail-closed statuses (pre-registered):
    SANS_PLAY_COLLECTED            - buffer collected, calibration pending
    BLOCKED_SANS_BUFFER_INSUFFICIENT - fewer than MIN_SAMPLES interactions
    BLOCKED_SANS_DEGENERATE_LABELS - buffer has < 2 distinct action labels
    SANS_CALIBRATION_FAILED        - held-out metrics below the bar
    SANS_HEAD_CALIBRATED           - provenance-complete calibrated head
    BLOCKED_IMPORT_FAILED          - missing module/API on the path

This is self-generated data from live environment feedback (external state
deltas), NOT benchmark leakage: no grid labels, game logic, hidden state, or
score deltas are reconstructed.
"""

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

STATUS_PLAY_COLLECTED = "SANS_PLAY_COLLECTED"
STATUS_BUFFER_INSUFFICIENT = "BLOCKED_SANS_BUFFER_INSUFFICIENT"
STATUS_DEGENERATE_LABELS = "BLOCKED_SANS_DEGENERATE_LABELS"
STATUS_CALIBRATED = "SANS_HEAD_CALIBRATED"
STATUS_CALIBRATION_FAILED = "SANS_CALIBRATION_FAILED"
STATUS_IMPORT_FAILED = "BLOCKED_IMPORT_FAILED"

MIN_SAMPLES = 6
MIN_DISTINCT_LABELS = 2
HOLDOUT_FRACTION = 0.2
CALIBRATION_STEPS = 120
CALIBRATION_LR = 1e-3
ACCURACY_FLOOR = 0.6
ACCURACY_MARGIN = 0.15
DELTA_NU_FLOOR = 2  # changed cells required to commit to the buffer

# Phase 7.9c: SANS calibration optimizer (default "adamw" = control).
OPTIMIZER_ADAMW = "adamw"
OPTIMIZER_SGLD = "sgld"
SGLD_T0 = 0.5        # Langevin temperature at t=0
SGLD_DECAY = 0.05    # T(t) = T0 * (1 + DECAY*t)^-EXP
SGLD_EXP = 0.55

DEFAULT_HEAD_PATH = "models/henri_sans_action_head.pt"


@dataclass
class SANSResult:
    status: str = ""
    reason: str = ""
    buffer_size: int = 0
    positive_size: int = 0
    distinct_labels: int = 0
    held_out_accuracy: Optional[float] = None
    majority_baseline: Optional[float] = None
    head_path: str = ""
    action_head_sha256: str = ""
    action_head_state_dict_sha256: str = ""
    calibration_dataset_digest: str = ""
    split_identity: str = ""
    action_labels: List[str] = field(default_factory=list)
    hidden_dim: int = 0
    action_dim: int = 0
    calibration_optimizer: str = ""
    init_param_digest: str = ""
    final_param_digest: str = ""
    train_loss: Optional[float] = None
    held_out_loss: Optional[float] = None
    provenance: dict = field(default_factory=dict)


def _state_dict_sha256(sd) -> str:
    h = hashlib.sha256()
    for k in sorted(sd.keys()):
        t = sd[k].detach().cpu().contiguous()
        h.update(k.encode("utf-8"))
        h.update(b"\x00")
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def _digest_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _dataset_digest(rows: Sequence[Tuple[torch.Tensor, int, int]]) -> str:
    h = hashlib.sha256()
    for hidden, action_idx, delta_nu in rows:
        h.update(hidden.detach().cpu().contiguous().numpy().tobytes())
        h.update(str(action_idx).encode("utf-8"))
        h.update(str(delta_nu).encode("utf-8"))
    return h.hexdigest()


def _time_tail_split(n: int, holdout_fraction: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Contiguous time-tail holdout over one trajectory.

    Rows are appended in play order; the most recent holdout_fraction of
    the buffer forms the held-out split. Adjacent frames never cross the
    boundary (no interleaved train/holdout leakage).
    """
    n_hold = max(1, int(n * holdout_fraction))
    n_train = n - n_hold
    return torch.arange(n_train), torch.arange(n_train, n)


def _param_digest(model: nn.Module) -> str:
    """Deterministic digest of a module's state dict (sorted keys)."""
    return _state_dict_sha256(model.state_dict())


def _sgld_noise_scale(temp: float, dt: float) -> float:
    """Langevin noise scale: sqrt(2 * T * dt) (HENRI SGLD invariant)."""
    return math.sqrt(2.0 * temp * dt)


def calibrate_action_head(
    X: torch.Tensor,
    y: torch.Tensor,
    n_actions: int,
    *,
    seed: int,
    optimizer: str,
    device: str,
) -> Dict[str, Any]:
    """Calibrate a fresh Linear(hidden -> n_actions) head.

    Deterministic init derived from `seed` (identical across optimizers).
    optimizer="adamw" -> AdamW CE (control); "sgld" -> SGD base + Langevin
    noise, T(t)=T0*(1+DECAY*t)^-EXP, noise scale sqrt(2*T*dt), unit-norm
    per-parameter noise on a derived RNG. Returns metrics + digests;
    never persists.
    """
    if optimizer not in (OPTIMIZER_ADAMW, OPTIMIZER_SGLD):
        raise ValueError(f"unknown calibration optimizer: {optimizer!r}")

    n = X.shape[0]
    hidden_dim = int(X.shape[1])
    train_idx, hold_idx = _time_tail_split(n, HOLDOUT_FRACTION)
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_ho, y_ho = X[hold_idx], y[hold_idx]

    # Deterministic init: derive from seed, identical for both arms.
    rng_state = torch.get_rng_state()
    torch.manual_seed(seed ^ 0x9E3779B9)
    try:
        model = nn.Linear(hidden_dim, n_actions)
    finally:
        torch.set_rng_state(rng_state)
    model.to(device).to(torch.float32)

    crit = nn.CrossEntropyLoss()
    init_digest = _param_digest(model)
    model.train()

    if optimizer == OPTIMIZER_ADAMW:
        opt = torch.optim.AdamW(model.parameters(), lr=CALIBRATION_LR)
        for _ in range(CALIBRATION_STEPS):
            opt.zero_grad()
            loss = crit(model(X_tr), y_tr)
            loss.backward()
            opt.step()
    else:
        opt = torch.optim.SGD(model.parameters(), lr=CALIBRATION_LR)
        noise_rng = torch.Generator(device="cpu").manual_seed(seed ^ 0x5F3759DF)
        for step in range(CALIBRATION_STEPS):
            opt.zero_grad()
            loss = crit(model(X_tr), y_tr)
            loss.backward()
            opt.step()
            temp = SGLD_T0 * (1.0 + SGLD_DECAY * step) ** (-SGLD_EXP)
            scale = _sgld_noise_scale(temp, CALIBRATION_LR)
            with torch.no_grad():
                for p in model.parameters():
                    noise = torch.randn(
                        p.shape, generator=noise_rng, dtype=p.dtype)
                    noise = F.normalize(noise, p=2.0, dim=-1) * scale
                    p.add_(noise.to(p.device))

    if not all(torch.isfinite(p).all().item() for p in model.parameters()):
        return {"non_finite": True}

    model.eval()
    with torch.no_grad():
        pred_ho = torch.argmax(model(X_ho), dim=-1)
        held_out_accuracy = float((pred_ho == y_ho).float().mean().item())
        train_loss = float(crit(model(X_tr), y_tr).item())
        held_out_loss = float(crit(model(X_ho), y_ho).item())
    counts = torch.bincount(y_tr, minlength=n_actions).float()
    majority_baseline = float(counts.max().item() / max(1, counts.sum().item()))

    return {
        "non_finite": False,
        "model": model,
        "init_param_digest": init_digest,
        "final_param_digest": _param_digest(model),
        "held_out_accuracy": held_out_accuracy,
        "majority_baseline": majority_baseline,
        "train_loss": train_loss,
        "held_out_loss": held_out_loss,
        "train_size": int(X_tr.shape[0]),
        "held_out_size": int(X_ho.shape[0]),
    }


def run_sans_play(
    game: Any,
    tokenizer: Any,
    egress_transducer: Any,
    action_head: nn.Module,
    vocab: Any,
    n_steps: int,
    device: str = "cpu",
    seed: int = 0,
    env_name: str = "",
    tele: Any = None,
    camera: Any = None,
    head_path: str = DEFAULT_HEAD_PATH,
    optimizer: str = OPTIMIZER_ADAMW,
) -> SANSResult:
    """Execute bounded epistemic play and calibrate the action head.

    vocab: object exposing id_to_action (list of action objects with .name).
    camera: optional CameraParams for screen-space ACTION6 payloads.
    """
    res = SANSResult()
    if n_steps <= 0:
        res.status = STATUS_BUFFER_INSUFFICIENT
        res.reason = "n_steps <= 0"
        return res

    from arc_action_head import unbinder_hidden
    from arc_egress_contract import flatten_uwe

    unbinder = getattr(egress_transducer, "unbinder", None)
    encode = getattr(tokenizer, "encode_spatial_grid", None)
    if unbinder is None or encode is None:
        res.status = STATUS_IMPORT_FAILED
        res.reason = "transducer unbinder or tokenizer encoder missing"
        return res
    d_model = getattr(egress_transducer, "d_model", None)
    if d_model is None:
        res.status = STATUS_IMPORT_FAILED
        res.reason = "transducer lacks d_model"
        return res

    rng = torch.Generator(device="cpu").manual_seed(seed)
    allowed = list(vocab.id_to_action)
    res.action_labels = [getattr(a, "name", str(a)) for a in allowed]

    # ---- Epistemic play ----
    rows: List[Tuple[torch.Tensor, int, int]] = []
    obs = game.reset() if hasattr(game, "reset") else None
    if obs is None or not getattr(obs, "frame", None):
        res.status = STATUS_BUFFER_INSUFFICIENT
        res.reason = "game reset produced no frame"
        return res

    for _step in range(n_steps):
        import numpy as np
        grid = np.asarray(obs.frame[0]).tolist()
        wave_blocks = encode(grid).squeeze(0).to(device).to(torch.float32)
        flat = flatten_uwe(wave_blocks, d_model)
        hidden = unbinder_hidden(
            egress_transducer, flat, device=device
        ).detach().cpu().flatten()

        # Seeded random legal action.
        if not allowed:
            break
        idx = int(torch.randint(0, len(allowed), (1,), generator=rng).item())
        act = allowed[idx]

        # Step with a screen-space payload when the env/action needs one.
        obs_next = None
        payload_info = None
        try:
            if camera is not None:
                from arc_action_payloads import step_with_payload
                obs_next, payload_info = step_with_payload(
                    game, act, grid, enabled=True, seed=seed, camera=camera)
            else:
                obs_next = game.step(act)
        except Exception:
            obs_next = None

        delta_nu = 0
        if obs_next is not None and getattr(obs_next, "frame", None):
            try:
                post = np.asarray(obs_next.frame[0]).tolist()
                if len(post) == len(grid) and all(
                    len(p) == len(g) for p, g in zip(post, grid)
                ):
                    delta_nu = sum(
                        1 for pr, gr in zip(post, grid) for a, b in zip(pr, gr) if a != b
                    )
            except Exception:
                delta_nu = 0

        if delta_nu >= DELTA_NU_FLOOR:
            rows.append((hidden, idx, delta_nu))
        obs = obs_next if obs_next is not None else obs
        if getattr(obs, "state", None) and obs.state.name == "GAME_OVER":
            break

    res.buffer_size = len(rows)
    res.positive_size = res.buffer_size
    if tele is not None:
        tele.emit({"env": env_name, "event_type": "SANS_PLAY",
                   "interactions": n_steps, "buffer_size": res.buffer_size,
                   "delta_nu_floor": DELTA_NU_FLOOR})

    if res.buffer_size < MIN_SAMPLES:
        res.status = STATUS_BUFFER_INSUFFICIENT
        res.reason = f"SANS buffer {res.buffer_size} < min {MIN_SAMPLES}"
        return res

    labels = [r[1] for r in rows]
    res.distinct_labels = len(set(labels))
    if res.distinct_labels < MIN_DISTINCT_LABELS:
        res.status = STATUS_DEGENERATE_LABELS
        res.reason = f"only {res.distinct_labels} distinct action labels"
        return res

    res.calibration_dataset_digest = _dataset_digest(rows)
    res.status = STATUS_PLAY_COLLECTED

    # ---- Calibration (deterministic init; time-tail hold-out; optimizer) ----
    res.split_identity = f"sans-play:{env_name or 'env'}:seed-{seed}"
    res.calibration_optimizer = optimizer
    X = torch.stack([r[0] for r in rows]).to(device).to(torch.float32)
    y = torch.tensor([r[1] for r in rows], dtype=torch.long, device=device)
    if not torch.isfinite(X).all():
        res.status = STATUS_CALIBRATION_FAILED
        res.reason = "non-finite hidden features in SANS buffer"
        return res
    n = X.shape[0]
    hidden_dim = int(X.shape[1])
    n_actions = len(allowed)
    res.hidden_dim = hidden_dim
    res.action_dim = n_actions

    cal = calibrate_action_head(
        X, y, n_actions, seed=seed, optimizer=optimizer, device=device)
    if cal.get("non_finite"):
        res.status = STATUS_CALIBRATION_FAILED
        res.reason = "non-finite parameters after calibration"
        return res

    res.init_param_digest = cal["init_param_digest"]
    res.final_param_digest = cal["final_param_digest"]
    res.train_loss = cal["train_loss"]
    res.held_out_loss = cal["held_out_loss"]
    held_out_accuracy = cal["held_out_accuracy"]
    majority_baseline = cal["majority_baseline"]
    model = cal["model"]

    res.held_out_accuracy = held_out_accuracy
    res.majority_baseline = majority_baseline

    passed = (
        held_out_accuracy >= ACCURACY_FLOOR
        and held_out_accuracy >= majority_baseline + ACCURACY_MARGIN
    )
    res.provenance = {
        "schema_id": "henri.sans-action-head.v1",
        "env": env_name,
        "buffer_size": res.buffer_size,
        "distinct_labels": res.distinct_labels,
        "split_identity": res.split_identity,
        "calibration_dataset_digest": res.calibration_dataset_digest,
        "held_out_accuracy": held_out_accuracy,
        "majority_baseline": majority_baseline,
        "accuracy_floor": ACCURACY_FLOOR,
        "accuracy_margin": ACCURACY_MARGIN,
        "calibration_steps": CALIBRATION_STEPS,
        "delta_nu_floor": DELTA_NU_FLOOR,
        "hidden_dim": hidden_dim,
        "action_dim": n_actions,
    }

    if not passed:
        res.status = STATUS_CALIBRATION_FAILED
        res.reason = (
            f"held-out acc {held_out_accuracy:.3f} vs floor "
            f"{ACCURACY_FLOOR} / majority+margin "
            f"{majority_baseline + ACCURACY_MARGIN:.3f}"
        )
        return res

    # ---- Persist with provenance ----
    # Copy into the live head only after provenance-complete persistence.
    sd = {"head.weight": model.weight.detach().cpu().clone(),
          "head.bias": model.bias.detach().cpu().clone()}
    ckpt = {
        "schema_id": "henri.sans-action-head.v1",
        "d_model": d_model,
        "hidden_dim": hidden_dim,
        "action_dim": n_actions,
        "action_labels": res.action_labels,
        "calibration_dataset_digest": res.calibration_dataset_digest,
        "split_identity": res.split_identity,
        "held_out_accuracy": held_out_accuracy,
        "majority_baseline": majority_baseline,
        "state_dict": sd,
    }
    path = Path(head_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path, _use_new_zipfile_serialization=True)
    raw = path.read_bytes()
    res.action_head_sha256 = _digest_bytes(raw)
    res.action_head_state_dict_sha256 = _state_dict_sha256(sd)
    res.head_path = str(path)
    res.status = STATUS_CALIBRATED
    res.reason = (
        f"held-out acc {held_out_accuracy:.3f} > majority {majority_baseline:.3f} + "
        f"{ACCURACY_MARGIN}"
    )
    return res


def apply_calibrated_head(
    action_head: nn.Module,
    action_head_state: Any,
    sans: SANSResult,
) -> Any:
    """Copy a calibrated head into the live ActionHead and mark it active.

    Only called with STATUS_CALIBRATED (provenance-complete). Returns a new
    ActionHeadState with trained_action_head_active=True.
    """
    from arc_action_head import ActionHeadState

    if sans.status != STATUS_CALIBRATED:
        return action_head_state
    if action_head.d_hidden != sans.hidden_dim or action_head.n_actions != sans.action_dim:
        raise ValueError(
            f"calibrated head dims ({sans.hidden_dim}, {sans.action_dim}) "
            f"!= live head ({action_head.d_hidden}, {action_head.n_actions})"
        )
    # Reload through the strict loader path so provenance checks apply.
    from arc_action_head import load_action_head
    state = load_action_head(
        action_head, sans.head_path, policy="required",
        expected_hidden=sans.hidden_dim, expected_actions=sans.action_dim,
        calibration_dataset_digest=sans.calibration_dataset_digest,
    )
    # Strict loader already verified sha + dims + digest; attach SANS metrics.
    state.split_identity = sans.split_identity
    state.held_out_accuracy = sans.held_out_accuracy
    state.majority_baseline = sans.majority_baseline
    return state
