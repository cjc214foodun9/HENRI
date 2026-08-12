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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

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

# Phase 7.6 typed fail-closed status: sagnac steering requested without
# valid axiom waves (no silent random fallback).
STATUS_BLOCKED_AXIOMS = "BLOCKED_SAGNAC_AXIOMS_UNAVAILABLE"

DEFAULT_HEAD_PATH = "models/henri_sans_action_head.pt"

# Phase 7.6 hard-axiom steering (default OFF). Canonical real-wave metric:
# delta = 1 - 0.5*(1 + cos) on unit-norm flattened waves; identical -> 0,
# orthogonal -> 0.5. Matches the production advisory-veto epsilon semantics.
SAGNAC_VETO_EPSILON = 0.35


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
    provenance: dict = field(default_factory=dict)
    selection_mode: str = "random"
    veto_steps: int = 0
    steered_steps: int = 0
    veto_rate: Optional[float] = None


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


def select_action_sagnac(
    wave_blocks: torch.Tensor,
    axiom_waves: torch.Tensor,
    rng: torch.Generator,
    n_actions: int,
    epsilon: float = SAGNAC_VETO_EPSILON,
) -> Optional[int]:
    """Phase 7.6 hard-axiom steering (default OFF).

    Canonical real-wave metric: delta = 1 - 0.5 * (1 + cos) on unit-norm
    flattened waves (identical -> 0, orthogonal -> 0.5). If the BEST-matching
    axiom still leaves delta > epsilon, the state violates the invariant
    subspace and ALL actions are vetoed (return None -> caller skips the
    step). Otherwise sample uniformly among legal actions, confining the
    explorer to valid topological states.
    """
    if axiom_waves is None or axiom_waves.ndim != 3:
        return None
    wa = wave_blocks.reshape(-1).to(torch.float32)
    wa = wa / (wa.norm() + 1e-12)
    ax = axiom_waves.reshape(axiom_waves.shape[0], -1).to(
        wave_blocks.device
    ).to(torch.float32)
    ax = ax / (ax.norm(dim=-1, keepdim=True) + 1e-12)
    cos = float((ax @ wa).max().item())
    delta = 1.0 - 0.5 * (1.0 + cos)
    if delta > epsilon:
        return None
    return int(
        torch.randint(0, n_actions, (1,), generator=rng).item()
    )


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
    selection_mode: str = "random",
    axiom_waves: Optional[torch.Tensor] = None,
) -> SANSResult:
    """Execute bounded epistemic play and calibrate the action head.

    vocab: object exposing id_to_action (list of action objects with .name).
    camera: optional CameraParams for screen-space ACTION6 payloads.
    """
    res = SANSResult()
    if selection_mode not in ("random", "sagnac"):
        selection_mode = "random"
    res.selection_mode = selection_mode
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
    if selection_mode == "sagnac":
        # Fail closed: sagnac steering requires [M, num_blocks, 8] axioms.
        # Never silently fall back to random selection.
        if (
            axiom_waves is None
            or axiom_waves.ndim != 3
            or axiom_waves.shape[-1] != 8
            or axiom_waves.shape[1] * axiom_waves.shape[2] != d_model
        ):
            res.status = STATUS_BLOCKED_AXIOMS
            res.reason = (
                "sagnac steering requires [M, num_blocks, 8] axiom_waves "
                f"with num_blocks*8 == d_model {d_model}"
            )
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

        # Action selection: seeded random (default) OR Phase 7.6 Sagnac
        # hard-axiom steering (selection_mode="sagnac", default OFF).
        if not allowed:
            break
        vetoed_step = False
        if (
            selection_mode == "sagnac"
            and axiom_waves is not None
            and axiom_waves.ndim == 3
            and axiom_waves.shape[1:] == tuple(wave_blocks.shape)
        ):
            idx = select_action_sagnac(
                wave_blocks, axiom_waves, rng, len(allowed),
                epsilon=SAGNAC_VETO_EPSILON,
            )
            if idx is None:
                vetoed_step = True
                res.veto_steps += 1
            else:
                res.steered_steps += 1
                act = allowed[idx]
        else:
            idx = int(
                torch.randint(0, len(allowed), (1,), generator=rng).item()
            )
            act = allowed[idx]

        if vetoed_step:
            continue  # hard-axiom veto: no environment step this iteration

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
    res.veto_rate = (res.veto_steps / n_steps) if n_steps > 0 else None
    if tele is not None:
        tele.emit({"env": env_name, "event_type": "SANS_PLAY",
                   "interactions": n_steps, "buffer_size": res.buffer_size,
                   "delta_nu_floor": DELTA_NU_FLOOR,
                   "selection_mode": res.selection_mode,
                   "veto_steps": res.veto_steps,
                   "steered_steps": res.steered_steps,
                   "veto_rate": res.veto_rate})

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

    # ---- Calibration with seeded hold-out split ----
    res.split_identity = f"sans-play:{env_name or 'env'}:seed-{seed}"
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

    perm = torch.randperm(n, generator=rng)
    n_hold = max(1, int(n * HOLDOUT_FRACTION))
    hold_idx = perm[:n_hold]
    train_idx = perm[n_hold:]
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_ho, y_ho = X[hold_idx], y[hold_idx]

    model = nn.Linear(hidden_dim, n_actions).to(device).to(torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=CALIBRATION_LR)
    crit = nn.CrossEntropyLoss()
    model.train()
    for _ in range(CALIBRATION_STEPS):
        opt.zero_grad()
        loss = crit(model(X_tr), y_tr)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        pred_ho = torch.argmax(model(X_ho), dim=-1)
        held_out_accuracy = float((pred_ho == y_ho).float().mean().item())
    counts = torch.bincount(y_tr, minlength=n_actions).float()
    majority_baseline = float(counts.max().item() / max(1, counts.sum().item()))

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
        "selection_mode": res.selection_mode,
        "veto_steps": res.veto_steps,
        "steered_steps": res.steered_steps,
        "veto_rate": res.veto_rate,
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
