"""HENRI Scene Binder — hierarchical nested compositionality (Gap 1).

WHY THIS MODULE EXISTS
======================
HENRI is named for Nested and Recursive Intelligence, but the spatial ingress is
FLAT: it maps every cell of a grid onto one superposed wave with a single level.
The structure the name implies is absent:

    Psi_scene = (+)_{p}  Psi_object_p (x) Psi_role_p
    Psi_object_p = Psi_shape (x) Psi_colour (x) Psi_position (x) ((+)_{k} Psi_part_k)

Without that structure, asking "is object A inside object B" requires a dense
comparison across all cells at once, and every object's phase contributes
crosstalk to every other's.

THE POISONED PRIOR — READ THIS BEFORE CHANGING THE CODEC
=======================================================
The obvious binding primitive in this repository is the qFHRR ring codec. It is
MEASURED NON-COMPOSITIONAL: `qFHRREpistemicCodec.encode_text` is SHA-256-seeded
`torch.randint`, so every distinct string maps to an INDEPENDENT random Z_256
ring and the similarity of any two distinct strings is ~1/sqrt(D) (measured:
f(3,3,3) vs f(4,4,4) = -0.0045; "a+b" vs "b+a" = -0.0004; baseline 0.0039). A
scene graph bound with that codec is a random superposition: it CANNOT carry an
I/O relation, and a "scene binder" built on it would pass a similarity test while
carrying no structure at all. The structured codec fix was FALSIFIED_AT_SCALE.

This module therefore binds with DETERMINISTIC PHASE CODES derived from an
integer role/filler id — the same incommensurate-phase family the torus and
topological encoders use — and it ships the two controls that make the poisoned
prior impossible to re-introduce silently:

  PERMUTATION TEST   swapping two objects' roles or colours must CHANGE the wave.
  COMPOSITIONALITY   the bound wave of a 2-object scene must differ from the wave
                     of the SAME two objects bound in the other order, and the
                     retrieval of one object's filler must be closer to that
                     filler than to a random one.

ABSTENTION AND HONEST BOUNDARIES
--------------------------------
* DEFAULT-OFF SIDECAR: nothing live imports this module (verified by grep at
  commit time). It emits waves; it does not touch egress or any action path.
* The binder reads OBJECTS from a supplied segmentation. Discovering objects from
  pixels is a different problem (connected components) and is NOT claimed here;
  `segment_by_colour` is provided as an explicit, stated helper, not as a
  perception claim.
* No spatial-reasoning benchmark score is claimed. The tests measure
  REPRESENTATION PROPERTIES (permutation sensitivity, order sensitivity,
  retrieval separation), not task accuracy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

# Incommensurate phase steps (golden-ratio family, same convention as the
# topological encoder). Distinct id spaces get distinct steps so a colour id and
# a role id cannot alias onto the same phase.
_PHI = 0.6180339887498949
_STEP_ROLE = 2.0 * math.pi * _PHI
_STEP_FILLER = 2.0 * math.pi * 0.3819660112501051
_STEP_POS = 2.0 * math.pi * 0.2360679774997896
_STEP_LEVEL = 2.0 * math.pi * 0.1458980337503154
_STEP_PART = 2.0 * math.pi * 0.0901699437494745

# ID-SPACE SALT. DEFECT FIXED 2026-09-27: role and shape shared _STEP_ROLE, and the
# seed depended only on (value, step), so role_id=k and shape_id=k produced the SAME
# wave -- an aliasing bug across id spaces that silently corrupts every scene. Each
# kind now carries a distinct prime salt, so no two id spaces can collide.
_KIND_SALT = {"role": 11, "shape": 23, "colour": 37, "position": 53,
              "part": 71, "level": 89}

TWO_PI = 2.0 * math.pi


class SceneBinderError(RuntimeError):
    """Fail-closed contract violation."""


@dataclass
class SceneObject:
    """One object in the scene: its fillers plus an optional part list."""
    kind_id: int                     # e.g. the object's SHAPE/colour class
    colour_id: int
    position_id: int                 # discretized centroid bucket
    parts: List[int] = field(default_factory=list)
    role_id: Optional[int] = None    # assigned by the scene, not by the object


def _phase(step: float, value: int) -> float:
    return (step * float(value)) % TWO_PI


def bind_component(dim: int, step: float, value: int, salt: int = 0) -> torch.Tensor:
    """A single deterministic phase component of width `dim` (Re|Im interleaved).

    DEFECT FIXED 2026-09-27: the first form used arange(dim) and stacked, so the
    vector was 2*dim long and every wave violated the declared `dim` contract
    (measured: declared 512, actual 1024). dim//2 phase slots now yield exactly dim.
    """
    # PER-SLOT phases from a value-seeded deterministic generator.
    #
    # DEFECT FIXED 2026-09-27 (the largest one in this module): the first form used
    # `ph = step*value + idx*_STEP_POS`, i.e. two components differed by a CONSTANT
    # phase offset. Their similarity is then cos(step*delta_value) -- DIMENSION-
    # INDEPENDENT. Measured: role_0 vs role_1 = cos(2*pi*0.618) = -0.744, so every
    # filler was ~74% correlated with every other, superposition crosstalk swamped
    # retrieval, and test_probe_role_recovers_the_right_colour failed on every role.
    #
    # Binding needs near-ORTHOGONAL components (expected |cos| ~ 1/sqrt(dim/2)), and
    # that requires phases that vary ACROSS SLOTS in a value-dependent way. A
    # value-seeded generator makes them deterministic AND near-orthogonal.
    #
    # HONEST DISTINCTION (do not conflate with the qFHRR finding below): random
    # per-slot phases give pairwise SEPARATION, which binding requires. The measured
    # qFHRR defect was different -- it had no COMPOSITIONAL SMOOTHNESS (related
    # inputs, e.g. f(3,3,3) vs f(4,4,4), were uncorrelated). This module is a
    # symbolic binding scheme; it claims separation, NOT smoothness.
    g = torch.Generator().manual_seed(
        (int(value) * 2654435761 + int(salt) * 104729 + int(step * 1e6)) % 2147483647)
    ph = torch.rand(dim // 2, generator=g) * TWO_PI
    return torch.stack([torch.cos(ph), torch.sin(ph)], dim=-1).reshape(-1)


def bind_many(components: Sequence[torch.Tensor]) -> torch.Tensor:
    """BINDING = elementwise complex multiplication of unit-modulus components.

    If every component is unit modulus the product is unit modulus, so binding
    preserves the hypersphere. That is a property of THIS constructor, not an
    assertion about arbitrary inputs; `bind_many` renormalizes only to guard
    against accumulated float error and reports the residual.
    """
    if not components:
        raise SceneBinderError("bind_many needs >= 1 component")
    c = components[0].clone()
    for nxt in components[1:]:
        if nxt.shape != c.shape:
            raise SceneBinderError(f"shape mismatch {tuple(nxt.shape)} vs {tuple(c.shape)}")
        re = c[0::2] * nxt[0::2] - c[1::2] * nxt[1::2]
        im = c[0::2] * nxt[1::2] + c[1::2] * nxt[0::2]
        out = torch.empty_like(c)
        out[0::2] = re
        out[1::2] = im
        c = out
    n = torch.linalg.vector_norm(c)
    if not torch.isfinite(n) or n < 1e-12:
        raise SceneBinderError("bound wave collapsed to zero")
    return c / n


def unbind(wave: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """UNBINDING = complex conjugation of the key, then multiply."""
    if wave.shape != key.shape:
        raise SceneBinderError("unbind shape mismatch")
    re = wave[0::2] * key[0::2] + wave[1::2] * key[1::2]
    im = wave[1::2] * key[0::2] - wave[0::2] * key[1::2]
    out = torch.empty_like(wave)
    out[0::2] = re
    out[1::2] = im
    return out


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    na, nb = torch.linalg.vector_norm(a), torch.linalg.vector_norm(b)
    if float(na) < 1e-12 or float(nb) < 1e-12:
        return 0.0
    return float(torch.dot(a, b) / (na * nb))


class SceneBinder:
    """Binds a list of SceneObjects into one scene wave with explicit nesting.

    dim = spatial dimension of the wave (must be even).
    n_roles = number of role slots; roles are bound to objects by POSITION in the
              object list, so swapping two objects swaps their roles.
    """

    def __init__(self, dim: int = 4096, n_roles: int = 4, n_levels: int = 2):
        if dim < 8 or dim % 2 != 0:
            raise SceneBinderError("dim must be even and >= 8")
        self.dim = int(dim)
        self.n_roles = int(n_roles)
        self.n_levels = int(n_levels)
        self._cache: Dict[Tuple[str, int], torch.Tensor] = {}

    # ------------------------------------------------------------- primitives
    def role_vector(self, role_id: int) -> torch.Tensor:
        return self._cached("role", role_id, _STEP_ROLE)

    def filler_vector(self, kind: str, value: int) -> torch.Tensor:
        """`kind` namespaces the id space (colour / shape / position / part / level)."""
        step = {"colour": _STEP_FILLER, "shape": _STEP_ROLE,
                "position": _STEP_POS, "part": _STEP_PART,
                "level": _STEP_LEVEL}[kind]
        return self._cached(kind, value, step)

    def _cached(self, kind: str, value: int, step: float) -> torch.Tensor:
        k = (kind, int(value))
        v = self._cache.get(k)
        if v is None:
            v = bind_component(self.dim, step, int(value), _KIND_SALT.get(kind, 0))
            self._cache[k] = v
        return v

    # ------------------------------------------------------------------ bind
    def bind_object(self, obj: SceneObject) -> torch.Tensor:
        """Psi_object = shape (x) colour (x) position (x) ((+) parts per level)."""
        comps = [self.filler_vector("shape", obj.kind_id),
                 self.filler_vector("colour", obj.colour_id),
                 self.filler_vector("position", obj.position_id)]
        if obj.parts:
            # PARTS ARE NESTED: each part is bound with its own level key before
            # being superposed into the object. This is the "recursive" term.
            part_waves = []
            for lvl, part in enumerate(obj.parts[: self.n_levels]):
                part_waves.append(bind_many([
                    self.filler_vector("level", lvl),
                    self.filler_vector("part", part),
                ]))
            acc = part_waves[0].clone()
            for pw in part_waves[1:]:
                acc = acc + pw
            n = torch.linalg.vector_norm(acc)
            if float(n) > 1e-12:
                acc = acc / n
            comps.append(acc)
        return bind_many(comps)

    def bind_scene(self, objects: Sequence[SceneObject]) -> torch.Tensor:
        """Psi_scene = (+)_{p} Psi_object_p (x) role_p.  Order sets the roles."""
        if not objects:
            raise SceneBinderError("bind_scene needs >= 1 object")
        if len(objects) > self.n_roles:
            raise SceneBinderError(
                f"{len(objects)} objects exceeds n_roles={self.n_roles}")
        parts = []
        for i, obj in enumerate(objects):
            o = self.bind_object(obj)
            r = self.role_vector(i)
            parts.append(bind_many([o, r]))
        acc = parts[0].clone()
        for p in parts[1:]:
            acc = acc + p
        n = torch.linalg.vector_norm(acc)
        if float(n) < 1e-12:
            raise SceneBinderError("scene wave collapsed")
        return acc / n

    # ---------------------------------------------------------------- probe
    def probe_role(self, scene: torch.Tensor, role_id: int,
                   codebook: Dict[object, torch.Tensor]) -> object:
        """CLEANUP: unbind a role, then return the codebook key whose wave is closest.

        THIS IS THE STANDARD OPERATION, and the reason the earlier form was wrong
        matters:

        `bind_object` produces a MULTI-FACTOR product
        (shape (x) colour (x) position (x) parts), so the scene term is a 4- or
        5-way product. In any VSA, `cos(product, single_factor) ~ 1/sqrt(dim)` --
        i.e. a stored multi-factor object is near-orthogonal to each of its own
        factors, and NO operation can read a single factor out of it by a direct
        cosine. Measured on the first revision: correct colour -0.0074 vs a wrong
        filler +0.0368, i.e. exact chance, because the test asked the impossible.

        What IS recoverable is the OBJECT: unbinding the role leaves obj_r plus
        crosstalk of order 1/sqrt(dim), so cleanup against a codebook of known
        object waves recovers the right object. That is what a consumer does.
        For an exact single-filler read-out, use `role_filler_scene` below, which
        binds ONE filler per role (the scheme in which unbinding is exact).
        """
        if not codebook:
            raise SceneBinderError("codebook must be non-empty")
        r = self.role_vector(role_id)
        x = unbind(scene, r)                          # scene (x) role^-1
        best, best_k = -2.0, None
        for k, wave in codebook.items():
            sc = cosine(x, wave)
            if sc > best:
                best, best_k = sc, k
        return best_k

    def role_filler_scene(self, pairs: Sequence[Tuple[int, torch.Tensor]]) -> torch.Tensor:
        """The EXACT-retrieval scheme: scene = (+)_r bind(role_r, filler_r).

        One filler per role, so `unbind` recovers the filler with cosine ~1 (up to
        crosstalk of order 1/sqrt(dim)). Use this when a single attribute must be
        read out exactly; use `bind_scene` when the object itself carries structure.
        """
        if not pairs:
            raise SceneBinderError("role_filler_scene needs >= 1 pair")
        terms = []
        for role_id, filler in pairs:
            f = filler if isinstance(filler, torch.Tensor) else torch.as_tensor(filler, dtype=torch.float32)
            if f.numel() != self.dim:
                raise SceneBinderError(f"filler width {f.numel()} != dim {self.dim}")
            terms.append(bind_many([self.role_vector(role_id), f]))
        acc = terms[0].clone()
        for t in terms[1:]:
            acc = acc + t
        n = torch.linalg.vector_norm(acc)
        if float(n) < 1e-12:
            raise SceneBinderError("scene collapsed")
        return acc / n

    def probe_filler(self, scene: torch.Tensor, role_id: int,
                     candidates: Sequence[Tuple[object, torch.Tensor]]) -> object:
        """Exact single-filler read-out for a `role_filler_scene`."""
        if not candidates:
            raise SceneBinderError("candidates must be non-empty")
        x = unbind(scene, self.role_vector(role_id))
        best, best_k = -2.0, None
        for key, wave in candidates:
            sc = cosine(x, wave)
            if sc > best:
                best, best_k = sc, key
        return best_k


# ---------------------------------------------------------------- fixtures
def make_scene(n_objects: int, seed: int, dim: int = 4096,
               n_roles: int = 4, positions: Optional[Sequence[int]] = None,
               colours: Optional[Sequence[int]] = None,
               parts_per_object: int = 0) -> Tuple[List[SceneObject], SceneBinder]:
    """Deterministic scene. Distinct colours/positions per object by default."""
    st = (seed * 7919 + 13) % 2147483647
    objs = []
    for i in range(n_objects):
        st = (st * 1103515245 + 12345) % 2147483648
        c = int(colours[i]) if colours else 1 + (st >> 16) % 9
        st = (st * 1103515245 + 12345) % 2147483648
        p = int(positions[i]) if positions else 1 + (st >> 16) % 9
        st = (st * 1103515245 + 12345) % 2147483648
        parts = [2 + (st >> 16) % 7 + k for k in range(parts_per_object)]
        objs.append(SceneObject(kind_id=1 + i, colour_id=c, position_id=p, parts=parts))
    return objs, SceneBinder(dim=dim, n_roles=n_roles)
