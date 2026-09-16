"""Contract suite for ``henri_wave_kb`` (the HENRI V2 universal knowledge backbone).

Reference: HENRI-ARCH-2026-VLA-TOKENIZER-KNOWLEDGE-BACKBONE (Tier-1 Sagnac
invariant sieve, Tier-2 universal subspace backbone, wave-space engram store,
provenance-bound answer path).

Every assertion below is a CONTRACT on the honest implementation, i.e. on the
repairs recorded in ``henri_wave_kb``'s module docstring:

  R-1  Tier-1 epsilon must be DERIVED from a measured self-consistency
       distribution, and the reference's hardcoded 0.0431 must be shown to veto
       its own same-origin data.
  R-2  Tier-2 must FAIL CLOSED without a pinned external basis artifact, and the
       reference's ``qr(randn(...))`` construction exists only as a named control.
  R-3  There must exist an arm that provably carries nothing (NULL, built from a
       diagonal of ones) whose failure can be observed.
  R-4  The answer path must ABSTAIN below a floor, never generate text, and must
       carry its retrieval scores.

Evidence class per test is stated in its docstring. Measurements are taken in
this process (CPU only, deterministic seeds); nothing is mocked except where a
value is quoted from the reference document, and those are labelled OBSERVED-DOC.

Run:
  cd 'C:/Users/chan/henri-worktrees/aaii-v43/HENRI V2' && \
  env -u VIRTUAL_ENV -u PYTHONPATH -u PYTHONHOME \
      PYTHONPATH='C:/Users/chan/henri-worktrees/aaii-v43/HENRI V2' \
      PYTHONDONTWRITEBYTECODE=1 C:/Python314/python.exe \
      -m pytest tests/contract/test_wave_kb.py -q --tb=short
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import henri_wave_kb as kb                                        # noqa: E402
from henri_vla_tokenizer import HoloVLAConfig, HoloVLATokenizer   # noqa: E402

DIM = 2048          # the reference document's own local-CPU ambient dimension
FEAT = 256
K = 16
DOMAINS = ("General", "Scientific", "Coding", "Action Planning")

# Independent corpus (NOT the module's internal one) so the contract does not
# depend on module-private fixtures.
CORPUS = {
    "General": (
        "The harbour closes to ferries when the swell exceeds three metres.",
        "A kettle left on the stove will boil dry within the hour.",
        "The post office moved to the corner of Oak and Fifth last spring.",
        "Moths gathered around the porch light until the bulb failed.",
        "Her bicycle needed a new chain before the trip to the coast.",
        "The orchard yields more apples after a cold winter.",
        "A brass bell hangs above the door of the old shop.",
        "The ferry timetable changes again on the first of March.",
    ),
    "Scientific": (
        "Photons lose energy when they climb out of a gravitational well.",
        "A catalyst lowers the barrier without shifting the equilibrium.",
        "The refractive index of water falls as wavelength increases.",
        "Neurons fire when the summed input crosses a threshold.",
        "Half life measures how quickly a sample decays.",
        "Surface tension lets an insect rest on still water.",
        "Magnetic domains align when the material is cooled slowly.",
        "Turbulence dissipates energy across a cascade of eddies.",
    ),
    "Coding": (
        "A ring buffer overwrites its oldest element when full.",
        "Floating point equality should use an explicit tolerance.",
        "A deadlock needs two locks acquired in opposite orders.",
        "Memoisation trades memory for repeated subproblem work.",
        "A stable sort preserves the order of equal keys.",
        "Connection pools must recycle sockets after idle timeouts.",
        "Undefined behaviour lets the optimiser assume unreachable code.",
        "Logging inside a hot loop dominates the runtime.",
    ),
    "Action Planning": (
        "Close the supply valve before loosening the union nut.",
        "Mark the belt direction before removing the tensioner.",
        "Lower the mast with two people on the halyard.",
        "Check the spare fuse rating against the panel label.",
        "Drain the radiator before removing the lower hose.",
        "Support the frame on stands before any wheel is lifted.",
        "Tag the breaker with your name before starting work.",
        "Wipe the mating face clean before fitting the gasket.",
    ),
}
UNRELATED = (
    "banana quokka zephyr trombone lattice ossuary",
    "quiet orange velvet lantern parade sundial",
    "walnut apricot cinder mule ferry lantern",
    "xylophone marzipan vulture kelp obelisk",
    "3f9a1c77b0e2d4a8c6f1 90ab12cd34ef5600",
    "0xDEADBEEF 0xCAFEBABE 0xFEEDFACE",
    "a1b2c3d4 e5f6a7b8 c9d0e1f2 a3b4c5d6",
    "drizzle of nutmeg on roasted persimmon carpaccio",
    # OBSERVED (measured 2026-09-16 at D=2048, this corpus): fluent English that
    # shares no content with the corpus still scores high on this tokenizer's
    # wave geometry (0.9187 / 0.9182), i.e. ABOVE the registered 0.90 floor.
    # Keeping them in the negative set is what makes the miscalibration claim a
    # measurement instead of an assumption.
    "the travellers waited for the carriage in the cold rain",
    "the market stalls were closing as the light began to fade",
)


# ================================================================ fixtures
@pytest.fixture(scope="module")
def tokenizer():
    cfg = HoloVLAConfig(ambient_dim_D=DIM, num_blocks=256, grid_size_S=16,
                        feat_dim=FEAT)
    return HoloVLATokenizer(cfg)


@pytest.fixture(scope="module")
def texts():
    return [t for d in DOMAINS for t in CORPUS[d]]


@pytest.fixture(scope="module")
def labels():
    return [d for d in DOMAINS for _ in CORPUS[d]]


@pytest.fixture(scope="module")
def waves(tokenizer, texts):
    return tokenizer.encode_text(texts)                      # [32, D] complex64


# ============================================================ helper metrics
def _stress(a, b):
    return float(kb.ZoneCInvariantSieve.sagnac_stress(a, b).item())


def _float_twin(waves):
    t = kb.wave_real_twin(waves)
    return t / t.norm(dim=-1, keepdim=True)


def _multi_page_pdf(path: Path, pages):
    """Build a REAL multi-page PDF artifact (no mocking) and return its digest."""
    import pymupdf
    doc = pymupdf.open()
    for body in pages:
        page = doc.new_page()
        page.insert_text((72, 96), body[:900], fontsize=10)
    doc.save(str(path))
    doc.close()
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ============================================================ TIER 1 (sieve)
class TestTier1SagnacSieve:
    """ZoneCInvariantSieve: zero-parameter homodyne gate + measured calibration."""

    def test_identical_waves_accepted_and_random_waves_rejected(self, waves):
        """OBSERVED. 1 - Re(<a,a>)/||a||^2 = 0 for identical waves; unrelated
        corpus waves sit far above the gate."""
        sieve = kb.ZoneCInvariantSieve()
        identical = _stress(waves[0], waves[0])
        assert abs(identical) < 1e-5, f"identical pair stress {identical} != 0"
        assert bool(sieve.valid(waves[0], waves[0]).all().item())
        unrelated = _stress(waves[0], waves[9])       # different domain
        assert unrelated > kb.HARDCODED_EPSILON, (
            f"unrelated pair stress {unrelated} should exceed the hardcoded gate")
        assert not bool(sieve.valid(waves[0], waves[9]).any().item())

    def test_sieve_accepts_batch_identical_and_rejects_all_unrelated(self, waves):
        """OBSERVED. The gate is a per-row filter; the reject set must be non-empty
        (a gate that accepts everything is vacuous)."""
        sieve = kb.ZoneCInvariantSieve()
        rep = sieve.filter(waves, waves)
        assert rep["accept_rate"] == 1.0
        shifted = torch.roll(waves, shifts=1, dims=0)
        rep2 = sieve.filter(waves, shifted)
        assert len(rep2["rejected_idx"]) >= len(waves) - 1
        assert rep2["accept_rate"] <= 1.0 / len(waves)

    def test_hardcoded_epsilon_self_vetoes_measured_self_pairs(
            self, tokenizer, texts, waves):
        """OBSERVED. The reference's 0.0431 rejects genuine same-origin pairs.

        Same-origin = two ingress views of the same source (re-spaced text). This
        is the R-1 defect measured on real data, not quoted from the document.
        """
        respaced = tokenizer.encode_text([kb.respace_view(t) for t in texts])
        stress = kb.ZoneCInvariantSieve._stresses(
            [(waves[i], respaced[i]) for i in range(len(texts))])
        assert float(stress.min().item()) > 0.0, "views must not be wave-identical"
        accepted = int((stress <= kb.HARDCODED_EPSILON).sum().item())
        assert accepted < len(texts), (
            "the hardcoded epsilon accepted every same-origin pair; the self-veto "
            "this module documents would not be reproducible")

    def test_calibrated_epsilon_differs_and_accepts_the_measured_pairs(
            self, tokenizer, texts, waves):
        """DERIVED. epsilon := quantile(0.999) of the measured self-consistency
        distribution; the calibrated gate accepts that distribution, the hardcoded
        one does not."""
        respaced = tokenizer.encode_text([kb.respace_view(t) for t in texts])
        pairs = [(waves[i], respaced[i]) for i in range(len(texts))]
        negatives = [(waves[i], waves[(i + 5) % len(texts)])
                     for i in range(len(texts))]
        cal = kb.ZoneCInvariantSieve.calibrate_epsilon(pairs,
                                                       negative_pairs=negatives)
        assert abs(cal.calibrated_epsilon - kb.HARDCODED_EPSILON) > 1e-6
        assert cal.calibrated_epsilon > kb.HARDCODED_EPSILON, (
            "calibrating to the measured distribution must widen, not tighten, the "
            "reference's gate on this data")
        assert cal.n_pairs == len(texts)
        # quantile(0.999) of n samples may exclude at most the interpolation
        # boundary sample; anything worse means the calibration is broken.
        assert cal.calibrated_accept_rate_on_self >= 1.0 - 1.0 / cal.n_pairs
        assert cal.calibrated_accept_rate_on_self > cal.hardcoded_accept_rate_on_self
        calibrated = kb.ZoneCInvariantSieve(epsilon=cal.calibrated_epsilon)
        assert all(bool(calibrated.valid(p, c).all().item()) for p, c in pairs
                   if _stress(p, c) <= cal.calibrated_epsilon)

    def test_verdict_is_a_function_of_the_negative_control(self, tokenizer, texts,
                                                           waves):
        """DERIVED. Same self-distribution, two negative controls -> the gate is
        NON_VACUOUS against different-origin pairs and VACUOUS against the
        document's own reported self-stress."""
        respaced = tokenizer.encode_text([kb.respace_view(t) for t in texts])
        pairs = [(waves[i], respaced[i]) for i in range(len(texts))]
        negatives = [(waves[i], waves[(i + 5) % len(texts)])
                     for i in range(len(texts))]
        good = kb.ZoneCInvariantSieve.calibrate_epsilon(pairs,
                                                        negative_pairs=negatives)
        assert good.verdict == "NON_VACUOUS"
        assert good.negative_accept_rate_at_calibrated == 0.0
        assert good.separation_margin is not None and good.separation_margin > 0.0

        # The document's own reported same-origin stress, inserted as a self-pair:
        theta = math.acos(1.0 - kb.DOC_REPORTED_SELF_STRESS)
        doc_pair = (waves[0], waves[0] * torch.exp(
            torch.tensor(1j * theta, dtype=torch.complex64)))
        assert abs(_stress(*doc_pair) - kb.DOC_REPORTED_SELF_STRESS) < 1e-4
        naive = kb.ZoneCInvariantSieve.calibrate_epsilon(
            pairs + [doc_pair], negative_pairs=negatives)
        assert naive.verdict == "VACUOUS", (
            "calibrating to the reference's own self-stress must be reported "
            "VACUOUS, because it destroys the gate's discrimination")
        assert naive.negative_accept_rate_at_calibrated == 1.0

    def test_calibration_without_negatives_is_inconclusive_not_a_pass(self, waves):
        """CONTRACT. Omitting the negative control yields UNMEASURED, never
        NON_VACUOUS: 'nothing executed' is not a pass."""
        cal = kb.ZoneCInvariantSieve.calibrate_epsilon([(waves[0], waves[1])])
        assert cal.non_vacuous is None
        assert cal.verdict == "UNMEASURED_NO_NEGATIVE_CONTROL"

    def test_doc_reported_self_stress_is_rejected_by_the_hardcoded_gate(self, waves):
        """OBSERVED-DOC / DERIVED. 0.993424 > 0.0431: the reference's hardcoded
        epsilon vetoes the reference's own measurement (defect D-8)."""
        assert kb.DOC_REPORTED_SELF_STRESS > kb.HARDCODED_EPSILON
        sieve = kb.ZoneCInvariantSieve()
        theta = math.acos(1.0 - kb.DOC_REPORTED_SELF_STRESS)
        pair = (waves[0], waves[0] * torch.exp(
            torch.tensor(1j * theta, dtype=torch.complex64)))
        assert not sieve.accepts_one(*pair)
        assert kb.ZoneCInvariantSieve.calibrate_epsilon([pair]).calibrated_epsilon \
            > kb.HARDCODED_EPSILON

    def test_sieve_carries_no_tunable_parameters(self):
        """CONTRACT. 'Zero-parameter physical-invariant filter': the only numeric
        state is epsilon; there are no torch Parameters and no gradients."""
        sieve = kb.ZoneCInvariantSieve()
        assert not isinstance(sieve, torch.nn.Module)
        params = [v for v in vars(sieve).values() if isinstance(v, torch.nn.Parameter)]
        assert params == []
        assert isinstance(sieve.epsilon, float)

    def test_zero_norm_input_fails_closed(self):
        """CONTRACT. A degenerate wave yields +inf stress (reject), not a division
        by zero and not an accept."""
        z = torch.zeros(8, dtype=torch.complex64)
        v = torch.ones(8, dtype=torch.complex64)
        stress = kb.ZoneCInvariantSieve.sagnac_stress(z, v)
        assert math.isinf(float(stress.item()))
        assert not bool(kb.ZoneCInvariantSieve().valid(z, v).any().item())


# ========================================================== TIER 2 (subspace)
class TestTier2SubspaceAdapter:
    """UniversalSubspaceAdapter: W = U_k diag(c) U_k^T with pinned/random/null arms."""

    def test_fails_closed_without_a_pinned_basis(self):
        """CONTRACT (R-2). Requesting the production arm with no pinned artifact
        raises a typed error -- the reference silently substituted
        ``qr(randn(2048, 16))`` here."""
        with pytest.raises(kb.MissingPinnedBasisError):
            kb.UniversalSubspaceAdapter(mode="pinned", k=K, ambient_real_dim=2 * DIM)
        with pytest.raises(kb.PinnedBasisError):     # typed family
            kb.UniversalSubspaceAdapter(mode="pinned", k=K, ambient_real_dim=2 * DIM)
        with pytest.raises(kb.MissingPinnedBasisError):
            kb.load_pinned_basis(os.path.join("does", "not", "exist.pt"))

    def test_pinned_artifact_round_trip_and_pin_mismatch(self, tokenizer, waves,
                                                         tmp_path):
        """OBSERVED. A fitted basis survives save/load only with matching digests;
        a wrong pin, a wrong format and a corrupt file all fail closed."""
        fitted = kb.fit_pinned_basis_from_waves(waves, K)
        art = tmp_path / "pinned.pt"
        manifest = kb.save_pinned_basis(fitted, str(art))
        assert manifest["artifact_sha256"] == kb.sha256_file(str(art))

        arm = kb.UniversalSubspaceAdapter(mode="pinned", k=K,
                                          ambient_real_dim=fitted.ambient_real_dim,
                                          artifact_path=str(art),
                                          expected_artifact_sha256=manifest["artifact_sha256"])
        assert arm.orthonormality_error < 1e-5
        assert arm.carries_world_knowledge is True
        assert arm.is_information_free is False
        assert torch.allclose(arm.basis, fitted.basis, atol=1e-6)

        with pytest.raises(kb.BasisPinMismatchError):
            kb.load_pinned_basis(str(art), expected_artifact_sha256="0" * 64)
        with pytest.raises(kb.BasisPinMismatchError):
            kb.load_pinned_basis(str(art), expected_k=K + 1)

        wrong = tmp_path / "wrong_format.pt"
        torch.save({"format": "not-ours", "basis": fitted.basis}, str(wrong))
        with pytest.raises(kb.BasisPinMismatchError):
            kb.load_pinned_basis(str(wrong))

        corrupt = tmp_path / "corrupt.pt"
        corrupt.write_bytes(art.read_bytes()[: 64])
        with pytest.raises(kb.BasisPinMismatchError):
            kb.load_pinned_basis(str(corrupt))

    def test_null_basis_is_provably_information_free(self):
        """CONTRACT (R-3). The NULL arm's operator is exactly diagonal, its
        per-domain outputs are bit-identical, and it refuses to be tuned."""
        n = 128
        null = kb.UniversalSubspaceAdapter.null_arm(n, K)
        assert null.carries_world_knowledge is False
        assert null.is_information_free is True
        assert null.orthonormality_error < 1e-6
        # built from a diagonal of ones, NOT random
        expected = torch.zeros(n, K)
        expected[:K, :K] = torch.eye(K)
        assert torch.equal(null.basis, expected)
        # operator is identity-like (diagonal, ones on the first K diagonal entries)
        W = null.operator(DOMAINS[0])
        assert torch.allclose(W, torch.diag(torch.diagonal(W)), atol=1e-12)
        assert null.offdiagonal_frobenius(DOMAINS[0]) == 0.0
        assert null.is_diagonal(DOMAINS[0])
        diag = torch.diagonal(W)
        assert torch.allclose(diag[:K], torch.ones(K), atol=1e-12)
        assert torch.allclose(diag[K:], torch.zeros(n - K), atol=1e-12)
        # per-domain outputs are IDENTICAL across domains
        x = torch.randn(3, n, generator=torch.Generator().manual_seed(7))
        outs = [null.transform(x, d) for d in DOMAINS]
        for o in outs[1:]:
            assert torch.equal(outs[0], o)
        # and it cannot be turned into a fitted arm
        with pytest.raises(kb.NullBasisDomainError):
            null.set_coordinates(DOMAINS[0], torch.full((K,), 0.5))
        assert torch.all(null.coordinates(DOMAINS[0]) == 1.0)

    def test_random_basis_differs_from_null_and_is_not_diagonal(self):
        """CONTRACT. The reference's qr(randn) arm is NOT silently the control: it
        mixes coordinates (off-diagonal operator mass > 0) and its per-domain
        outputs differ once domain coordinates are fitted."""
        n = 128
        null = kb.UniversalSubspaceAdapter.null_arm(n, K)
        rnd = kb.UniversalSubspaceAdapter.random_arm(n, K, seed=1234)
        assert rnd.carries_world_knowledge is False
        assert rnd.is_information_free is False
        assert not torch.allclose(rnd.basis, null.basis)
        assert rnd.offdiagonal_frobenius(DOMAINS[0]) > 0.0
        assert not rnd.is_diagonal(DOMAINS[0])
        assert rnd.orthonormality_error < 1e-5

        x = torch.randn(4, n, generator=torch.Generator().manual_seed(11))
        assert not torch.allclose(rnd.transform(x, DOMAINS[0]),
                                  null.transform(x, DOMAINS[0]))
        # domain conditioning is possible on the random arm, impossible on NULL
        rnd.set_coordinates(DOMAINS[0], torch.full((K,), 0.3))
        rnd.set_coordinates(DOMAINS[1], torch.full((K,), 0.9))
        assert not torch.equal(rnd.transform(x, DOMAINS[0]),
                               rnd.transform(x, DOMAINS[1]))
        assert torch.equal(null.transform(x, DOMAINS[0]),
                           null.transform(x, DOMAINS[1]))

    def test_operator_matches_the_low_rank_form_and_has_rank_at_most_k(self):
        """DERIVED. W = U diag(c) U^T agrees with the explicit reconstruction, and
        rank(W) <= k (the design's whole claim)."""
        n = 64
        rnd = kb.UniversalSubspaceAdapter.random_arm(n, K, seed=99)
        c = torch.arange(1, K + 1, dtype=torch.float32) / K
        rnd.set_coordinates(DOMAINS[0], c)
        W = rnd.operator(DOMAINS[0])
        explicit = (rnd.basis * c.unsqueeze(0)) @ rnd.basis.t()
        assert torch.allclose(W, explicit, atol=1e-6)
        assert torch.allclose(W, W.t(), atol=1e-6)          # symmetric
        assert int(torch.linalg.matrix_rank(W, tol=1e-6).item()) <= K
        # c is the only tunable surface: length k, one vector per canonical domain
        assert set(rnd.domains) == set(DOMAINS)
        assert rnd.coordinates_matrix().shape == (len(DOMAINS), K)
        with pytest.raises(kb.WaveKBError):
            rnd.set_coordinates(DOMAINS[0], torch.ones(K + 1))
        before = rnd.transform(torch.ones(1, n), DOMAINS[0])
        rnd.set_coordinates(DOMAINS[0], torch.full((K,), -1.0))
        assert not torch.allclose(before, rnd.transform(torch.ones(1, n), DOMAINS[0]))
        params = [v for v in vars(rnd).values() if isinstance(v, torch.nn.Parameter)]
        assert params == [], "the adapter must expose no trainable parameters"

    def test_null_control_does_not_beat_random_or_pinned_on_held_out_probe(
            self, tokenizer, texts, labels, waves):
        """OBSERVED (the required control result). On held-out waves the NULL arm
        must not beat the random arm nor the pinned arm, and its information-free
        structure must be exact."""
        fit_rows, ho_rows = [], []
        for d in DOMAINS:
            w = tokenizer.encode_text(list(CORPUS[d]))
            fit_rows.append(w[:5])
            ho_rows.append(w[5:])
        fit_waves = torch.cat(fit_rows, 0)
        ho_real = _float_twin(torch.cat(ho_rows, 0))
        fit_labels = [d for d in DOMAINS for _ in range(5)]

        pinned = kb.UniversalSubspaceAdapter(
            mode="pinned", k=K, ambient_real_dim=fit_waves.shape[1] * 2,
            basis=kb.fit_pinned_basis_from_waves(fit_waves, K).basis)
        null = kb.UniversalSubspaceAdapter.null_arm(ho_real.shape[1], K)

        s_real = kb.probe_basis(pinned, ho_real, arm="pinned")
        s_null = kb.probe_basis(null, ho_real, arm="null")
        randoms = [kb.probe_basis(
            kb.UniversalSubspaceAdapter.random_arm(ho_real.shape[1], K, seed=500 + s),
            ho_real, arm=f"random[{s}]") for s in range(kb.RANDOM_ARM_SEEDS)]
        s_rand = kb.BasisProbeScore(
            arm="random_mean", k=K, ambient_real_dim=ho_real.shape[1],
            n_probes=ho_real.shape[0],
            energy_capture=sum(r.energy_capture for r in randoms) / len(randoms),
            principal_direction_cos=sum(r.principal_direction_cos for r in randoms) / len(randoms),
            mean_domain_delta=0.0, domains_bit_identical=False,
            offdiagonal_frobenius=randoms[0].offdiagonal_frobenius,
            carries_world_knowledge=False, is_information_free=False)

        rule = kb.evaluate_control_rule(null_score=s_null, random_score=s_rand,
                                        real_score=s_real)
        assert rule["C2_null_below_real_by_margin"], (
            f"NULL energy {s_null.energy_capture} must fall short of the pinned arm "
            f"{s_real.energy_capture} by {kb.CONTROL_MARGIN}")
        assert rule["C1_null_not_above_random"], (
            f"NULL energy {s_null.energy_capture} exceeded the random arm "
            f"{s_rand.energy_capture} beyond the k/n tie tolerance")
        assert rule["C3_null_domain_delta_zero"] and rule["C4_null_operator_diagonal"]
        assert rule["null_control_fails_as_required"] is True
        # the pinned arm must actually be better, i.e. the probe discriminates
        assert s_real.energy_capture > 0.5
        assert s_real.energy_capture > s_rand.energy_capture * 10.0
        assert s_real.principal_direction_cos > s_null.principal_direction_cos

        # domain conditioning: the null arm refuses, the fitted arms accept
        cond_null = kb.probe_domain_conditioning(
            null, fit_waves=fit_waves, fit_labels=fit_labels,
            held_out_waves=ho_real, held_out_labels=[d for d in DOMAINS for _ in range(3)])
        assert cond_null["coordinates_fitted"] is False
        assert cond_null["refusal_error"] == "NullBasisDomainError"
        assert cond_null["mean_domain_delta"] == 0.0


# ======================================================= TIER 3 (engram store)
class TestWaveEngramStore:
    """Append-only wave engram store with hash/offset-only provenance."""

    @pytest.fixture(scope="class")
    def store_bundle(self, tmp_path_factory, tokenizer):
        """Build a store from a REAL multi-page PDF artifact (no mocking)."""
        tmp = tmp_path_factory.mktemp("wavekb_artifact")
        pages = [
            "Photons lose energy when they climb out of a gravitational well. "
            "A catalyst lowers the barrier without shifting the equilibrium.",
            "A ring buffer overwrites its oldest element when full. Floating point "
            "equality should use an explicit tolerance.",
            "Close the supply valve before loosening the union nut. Mark the belt "
            "direction before removing the tensioner.",
        ]
        pdf = tmp / "artifact.pdf"
        pdf_sha = _multi_page_pdf(pdf, pages)

        import pymupdf
        doc = pymupdf.open(str(pdf))
        assert doc.page_count == 3, "the artifact must really be multi-page"
        page_texts = []
        offset = 0
        for pno in range(doc.page_count):
            body = doc[pno].get_text().strip()
            page_texts.append(body)
            offset += len(body)
        doc.close()

        cfg = HoloVLAConfig(ambient_dim_D=DIM, num_blocks=256, grid_size_S=16,
                            feat_dim=FEAT)
        waves = tokenizer.encode_text(page_texts)
        store = kb.WaveEngramStore(ambient_dim=DIM)
        provs = []
        for i, body in enumerate(page_texts):
            prov = kb.Provenance(source_sha256=pdf_sha, char_start=0,
                                 char_end=len(body), page_or_line=f"page:{i + 1}",
                                 label=f"page-{i + 1}")
            provs.append(prov)
            store.append(waves[i], prov, domain=DOMAINS[i % len(DOMAINS)])
        return {"store": store, "waves": waves, "provs": provs, "pdf_sha": pdf_sha,
                "pdf": pdf, "page_texts": page_texts, "dir": tmp}

    def test_engram_round_trip_preserves_provenance_and_sha256(self, store_bundle):
        """OBSERVED. Row metadata is byte-stable through append -> retrieve; the
        stored wave is byte-identical to the appended one, so its digest
        identifies exactly what was stored."""
        store = store_bundle["store"]
        waves = store_bundle["waves"]
        assert len(store) == 3
        for i, row in enumerate(store.rows):
            assert row.provenance.page_or_line == f"page:{i + 1}"
            assert row.provenance.source_sha256 == store_bundle["pdf_sha"]
            assert row.wave_sha256 == kb.wave_sha256(row.wave)
            assert row.wave_sha256 == kb.wave_sha256(waves[i])
            assert torch.equal(row.wave, waves[i]), (
                "an already-unit-norm wave must be stored byte-identically")
            assert abs(float(row.wave.norm().item()) - 1.0) < 1e-5
            assert row.created_utc.endswith("+00:00")
        hit = store.retrieve(waves[1], k=1)[0]
        row = store.get(hit.engram_id)
        assert hit.provenance == row.provenance
        assert hit.wave_sha256 == row.wave_sha256
        assert hit.provenance.as_dict() == row.provenance.as_dict()
        # an off-scale wave IS normalised, and the digest reports the stored form
        # (on a SEPARATE store: the shared fixture must stay append-only-stable)
        other = kb.WaveEngramStore(ambient_dim=DIM)
        scaled = waves[0] * 7.5
        row2 = other.append(scaled, store_bundle["provs"][0], domain=DOMAINS[0])
        assert abs(float(row2.wave.norm().item()) - 1.0) < 1e-6
        assert row2.wave_sha256 == kb.wave_sha256(row2.wave)
        assert row2.wave_sha256 != kb.wave_sha256(scaled)
        assert len(store) == 3, "the shared fixture store must not have been mutated"

    def test_append_only_and_schema_validation(self, store_bundle):
        """CONTRACT. Re-using an id, a bad provenance digest, a prose label and a
        wrong-width wave are all typed failures."""
        store = store_bundle["store"]
        wave = store_bundle["waves"][0]
        prov = store_bundle["provs"][0]
        with pytest.raises(kb.EngramSchemaError):
            store.append(wave, prov, domain=DOMAINS[0], engram_id="eng-000000")
        with pytest.raises(kb.EngramSchemaError):
            store.append(wave, kb.Provenance(source_sha256="nothex", char_start=0,
                                             char_end=1, page_or_line="page:1",
                                             label="x"), domain=DOMAINS[0])
        with pytest.raises(kb.EngramSchemaError):
            kb.Provenance(source_sha256="a" * 64, char_start=0, char_end=1,
                          page_or_line="page:1", label="page one\n" + "body text")
        with pytest.raises(kb.EngramSchemaError):
            store.append(torch.zeros(7, dtype=torch.complex64), prov, domain=DOMAINS[0])
        with pytest.raises(kb.EngramSchemaError):
            store.append(wave, prov, domain="NotADomain")
        with pytest.raises(kb.EngramSchemaError):
            store.retrieve(torch.zeros(7, dtype=torch.complex64), k=1)
        assert len(store) == 3                      # nothing was appended by the failures

    def test_cosine_retrieval_recovers_the_correct_engram_above_chance(
            self, store_bundle):
        """OBSERVED. Wave-space cosine retrieval on the page engrams: exact-string
        queries recover their own page (top-1 = 3/3) at chance 1/3, and every hit
        carries its score and provenance."""
        store = store_bundle["store"]
        waves = store_bundle["waves"]
        per_page = store_bundle["page_texts"]
        assert len(per_page) == 3
        hits = [store.retrieve(waves[i], k=1)[0] for i in range(len(waves))]
        expected = [store.rows[i].engram_id for i in range(len(waves))]
        correct = sum(1 for h, e in zip(hits, expected) if h.engram_id == e)
        assert correct == len(waves), (
            f"top-1 exact retrieval {correct}/{len(waves)}; chance is "
            f"{1.0 / len(waves):.3f}")
        assert correct / len(waves) > 1.0 / len(waves)          # strictly above chance
        assert all(h.score > 0.999 for h in hits)
        # the margin over the best WRONG engram is a real, measured gap (0.074-0.078
        # on this artifact), not a tie broken by ordering
        margins = []
        for i in range(len(waves)):
            row = (store.wave_matrix() * waves[i].conj().unsqueeze(0)).sum(-1).real
            row = row / store.wave_matrix().norm(dim=-1).clamp(min=1e-30)
            row = row / waves[i].norm().clamp(min=1e-30)
            others = torch.cat([row[:i], row[i + 1:]])
            margins.append(float(row[i].item() - others.max().item()))
        assert min(margins) > 0.05, f"retrieval margins too thin: {margins}"
        # scores are returned alongside provenance for every hit
        multi = store.retrieve(waves[0], k=3)
        assert len(multi) == 3
        assert all(isinstance(h.score, float) for h in multi)
        assert all(h.provenance.page_or_line for h in multi)
        assert [h.score for h in multi] == sorted([h.score for h in multi],
                                                  reverse=True)
        # retrieval is wave-driven, not index-driven: a scaled copy of the same
        # page must still win, and a page's own text must beat a different page
        assert store.retrieve(waves[0] * 3.0, k=1)[0].engram_id == expected[0]

    def test_contamination_guard_fires_negative_control_and_is_silent_when_clean(
            self, store_bundle):
        """CONTRACT. The guard MUST be able to fire, otherwise the cleanliness
        claim it supports is vacuous."""
        store = store_bundle["store"]
        blocked = store.rows[0].provenance.source_sha256
        report = store.contamination_report([blocked])
        assert len(report) == 3 and report[0]["engram_id"] == store.rows[0].engram_id
        with pytest.raises(kb.ContaminationError):
            store.assert_clean([blocked])
        with pytest.raises(kb.ContaminationError):
            store.assert_clean([blocked.upper()])       # digests are case-folded
        store.assert_clean(["f" * 64])                  # clean input: no raise
        store.assert_clean([])                          # empty blocklist: no raise

    def test_serialized_form_contains_only_hashes_ids_offsets(self, store_bundle):
        """CONTRACT (no text on disk). The serialised store must contain no
        document text, no wave bytes, and only schema keys."""
        store = store_bundle["store"]
        meta = store.serialize()
        blob = store.to_json(indent=None)
        assert meta["document_text_serialized"] is False
        assert meta["waves_serialized"] is False
        assert set(meta.keys()) == {"format", "ambient_dim", "domains", "row_count",
                                    "waves_serialized", "document_text_serialized",
                                    "rows"}
        for row in meta["rows"]:
            assert set(row.keys()) == set(kb.WaveEngramStore.SERIALIZED_KEYS)
            assert set(row["provenance"].keys()) == set(
                kb.WaveEngramStore.PROVENANCE_KEYS)
            assert len(row["wave_sha256"]) == 64
            assert len(row["provenance"]["source_sha256"]) == 64
            assert isinstance(row["provenance"]["char_start"], int)
            assert isinstance(row["provenance"]["char_end"], int)
        # no document token from the source pages may appear in the JSON
        for body in store_bundle["page_texts"]:
            for token in body.split():
                if len(token) >= 8:
                    assert token not in blob, (
                        f"document token {token!r} leaked into the serialised store")
        assert "wave" not in json.loads(blob)["rows"][0]

    def test_module_writes_no_document_text_to_disk(self, tokenizer, tmp_path):
        """CONTRACT. Running ingest -> retrieve -> answer -> serialise creates no
        files at all, so no raw document text can have been persisted."""
        workdir = tmp_path / "empty_workdir"
        workdir.mkdir()
        before = set(os.listdir(workdir))
        cfg = HoloVLAConfig(ambient_dim_D=DIM, num_blocks=256, grid_size_S=16,
                            feat_dim=FEAT)
        store = kb.WaveEngramStore(ambient_dim=DIM)
        body = ("The post office moved to the corner of Oak and Fifth last spring. "
                "A kettle left on the stove will boil dry within the hour.")
        wave = tokenizer.encode_text([body])[0]
        store.append(wave, kb.Provenance(source_sha256=hashlib.sha256(
            body.encode("utf-8")).hexdigest(), char_start=0, char_end=len(body),
            page_or_line="line:1", label="probe"), domain="General")
        store.retrieve(wave, k=1)
        kb.grounded_answer(store, wave, k=1)
        store.assert_clean(["0" * 64])
        blob = store.to_json()
        after = set(os.listdir(workdir))
        assert after == before == set()
        assert body[:40] not in blob
        # the ONLY file the module ever writes is an explicit pinned-basis artifact
        art = tmp_path / "pinned.pt"
        kb.save_pinned_basis(kb.fit_pinned_basis_from_waves(
            tokenizer.encode_text([body, body + " X"]), 1), str(art))
        assert art.is_file()
        assert body[:40].encode("utf-8") not in art.read_bytes()


# ===================================================== TIER 4 (answer path)
class TestGroundedAnswerPath:
    """grounded_answer: assemble from provenance, or abstain below the floor."""

    @pytest.fixture(scope="class")
    def store(self, tokenizer, texts, labels, waves):
        s = kb.WaveEngramStore(ambient_dim=DIM)
        for i, t in enumerate(texts):
            s.append(waves[i], kb.Provenance(
                source_sha256=hashlib.sha256(t.encode("utf-8")).hexdigest(),
                char_start=0, char_end=len(t), page_or_line=f"line:{i + 1}",
                label=labels[i]), domain=labels[i])
        return s

    def test_abstains_below_the_floor(self, tokenizer, store):
        """CONTRACT + OBSERVED. A query whose best wave-space score is below the
        floor must ABSTAIN and must still return the scores that justify the
        abstention. Measured: 8 of 10 unrelated queries abstain at the registered
        floor; the 2 that leak are reported in the miscalibration test below."""
        neg_waves = tokenizer.encode_text(list(UNRELATED))
        scored = [(store.retrieve(neg_waves[j], k=1)[0].score, j)
                  for j in range(neg_waves.shape[0])]
        n_abstain = sum(1 for s, _ in scored if s < kb.REGISTERED_SCORE_FLOOR)
        assert n_abstain == 8, f"expected 8 of 10 to abstain, got {n_abstain}"
        assert n_abstain > len(scored) // 2, "the floor must refuse most negatives"

        worst = min(scored)[1]
        q = neg_waves[worst]
        best = store.retrieve(q, k=1)[0].score
        assert best < kb.REGISTERED_SCORE_FLOOR, (
            f"probe query scored {best}; it is not below the registered floor")
        ans = kb.grounded_answer(store, q, k=3)
        assert ans.abstained is True
        assert ans.provenance == ()
        assert ans.engram_ids == ()
        assert ans.answer_text.startswith(kb.ABSTAIN_PREFIX)
        assert len(ans.retrieval_scores) == 3
        assert ans.best_score == pytest.approx(max(ans.retrieval_scores))
        d = ans.as_dict()
        assert d["abstained"] is True and d["provenance"] == []
        assert set(d.keys()) == {"answer_text", "provenance", "retrieval_scores",
                                 "abstained", "floor", "best_score", "engram_ids",
                                 "query_wave_sha256"}

    def test_answers_above_the_floor_and_assembles_from_provenance(
            self, tokenizer, texts, store):
        """OBSERVED. A same-origin query answers; the answer is assembled from
        provenance records only (ids/hashes/offsets) with no document text."""
        probe = tokenizer.encode_text([texts[3] + " Indeed."])[0]
        ans = kb.grounded_answer(store, probe, k=3)
        assert ans.abstained is False
        assert ans.best_score is not None and ans.best_score >= kb.REGISTERED_SCORE_FLOOR
        assert len(ans.retrieval_scores) == 3
        assert len(ans.provenance) >= 1
        assert len(ans.engram_ids) == len(ans.provenance)
        for prov in ans.provenance:
            assert set(prov.keys()) == set(kb.WaveEngramStore.PROVENANCE_KEYS)
            assert prov["source_sha256"] == store.get(ans.engram_ids[0]).provenance.source_sha256
        # no generated text: the answer contains no corpus prose
        for token in texts[3].split():
            if len(token) >= 8:
                assert token not in ans.answer_text, (
                    "the answer must be assembled from provenance, never from text")
        assert ans.answer_text.startswith("GROUNDED[")

    def test_floor_above_every_score_forces_abstention_above_floor_allows_it(
            self, tokenizer, texts, store):
        """CONTRACT. The verdict is a function of the comparison, not of the
        query: the SAME query flips between answered and abstained as the floor
        moves past its score."""
        q = tokenizer.encode_text([texts[0]])[0]
        score = store.retrieve(q, k=1)[0].score
        assert score > 0.99
        answered = kb.grounded_answer(store, q, k=1, floor=score - 1e-3)
        abstained = kb.grounded_answer(store, q, k=1, floor=min(1.0, score + 1e-3))
        assert answered.abstained is False
        assert abstained.abstained is True
        assert abstained.retrieval_scores[0] == pytest.approx(score, abs=1e-5)

    def test_registered_floor_does_not_separate_but_calibration_does(
            self, tokenizer, texts, store):
        """OBSERVED (honest finding). The pre-registered 0.90 does NOT separate
        the measured positive/negative distributions; the calibrated floor does.
        A miscalibrated constant must not pass silently."""
        probes = tokenizer.encode_text([t + " Indeed." for t in texts])
        pos = [store.retrieve(probes[i], k=1)[0].score for i in range(len(texts))]
        neg_waves = tokenizer.encode_text(list(UNRELATED))
        neg = [store.retrieve(neg_waves[j], k=1)[0].score
               for j in range(neg_waves.shape[0])]
        cal = kb.calibrate_floor(pos, neg)
        assert cal.separable is True, "measured distributions should have a gap here"
        assert cal.floor is not None
        assert cal.max_negative < cal.floor < cal.min_positive
        assert cal.max_negative > kb.REGISTERED_SCORE_FLOOR, (
            "measured: fluent but unrelated queries reach 0.9187, above the "
            "registered 0.90 floor")
        assert cal.registered_floor_separates is False, (
            "if the registered floor DID separate the measured distributions this "
            "finding would be wrong and the test must be revisited")
        assert cal.verdict == "REGISTERED_FLOOR_MISCALIBRATED"
        assert cal.gap > 0.03
        # grounded_answer honours the CALIBRATED floor on the query that the
        # registered floor would wrongly answer
        leak = max(range(len(neg)), key=lambda j: neg[j])
        assert neg[leak] > kb.REGISTERED_SCORE_FLOOR
        assert kb.grounded_answer(store, neg_waves[leak], k=1).abstained is False
        assert kb.grounded_answer(store, neg_waves[leak], k=1,
                                  floor=cal.floor).abstained is True
        # and the verdict is a function of the input: pass a cleaner negative set
        clean = kb.calibrate_floor(pos, [s - 0.2 for s in neg])
        assert clean.registered_floor_separates is True
        assert clean.verdict == "REGISTERED_FLOOR_SEPARATES"
        # overlapping distributions are reported, and demanding a floor raises
        overlap = kb.calibrate_floor(pos, [0.99] * len(neg))
        assert overlap.separable is False and overlap.floor is None
        assert overlap.verdict == "OVERLAPPING_NO_SEPARATING_FLOOR"
        with pytest.raises(kb.FloorCalibrationError):
            kb.calibrate_floor(pos, [0.99] * len(neg), require_separable=True)

    def test_grounded_answer_on_empty_store_abstains(self, tokenizer):
        """CONTRACT. No rows -> abstain with an empty score list, never a crash
        and never an invented answer."""
        empty = kb.WaveEngramStore(ambient_dim=DIM)
        q = tokenizer.encode_text(["anything at all"])[0]
        ans = kb.grounded_answer(empty, q, k=3)
        assert ans.abstained is True
        assert ans.retrieval_scores == ()
        assert ans.provenance == ()
        assert ans.best_score is None


# ================================================== module-level self-check
class TestSelfcheckHarness:
    """The runnable ``--selfcheck`` receipt must be honest about its own state."""

    def test_selfcheck_receipt_reports_a_pass_with_all_checks_true(self):
        """OBSERVED. The receipt is machine-readable, carries both epsilon values
        and both floors, and treats an exception as INCONCLUSIVE rather than a
        pass."""
        r = kb.selfcheck_receipt()
        assert r["verdict"]["status"] == "PASS"
        assert r["verdict"]["n_failed"] == 0
        assert r["verdict"]["n_checks"] == len(r["checks"]) > 0
        assert r["inconclusive"] == []
        assert r["constants"]["hardcoded_epsilon"] == kb.HARDCODED_EPSILON
        t1 = r["tier1_sieve"]
        assert t1["hardcoded_epsilon"] != t1["calibrated_epsilon"]
        assert t1["hardcoded_accepts_doc_self_stress"] is False
        assert t1["verdict"] == "NON_VACUOUS"
        assert t1["calibration_with_doc_pair"]["verdict"] == "VACUOUS"
        assert r["tier2_subspace"]["control_rule"]["null_control_fails_as_required"] is True
        assert r["tier3_store"]["serialized_contains_document_text"] is False
        assert r["tier4_answer_path"]["floor_calibration"]["separable"] is True
        assert r["tier4_answer_path"]["floor_calibration"]["verdict"] == \
            "REGISTERED_FLOOR_MISCALIBRATED"
        assert r["tier4_answer_path"]["abstained"]["abstained"] is True
        assert r["tier4_answer_path"]["answered"]["abstained"] is False
        assert r["tier1_sieve"]["degraded_view_calibration"]["verdict"] == "VACUOUS"
        assert isinstance(r["honest_negatives"], list) and r["honest_negatives"]
        json.dumps(r)                                  # must be serialisable

    def test_cli_refuses_to_run_silently(self):
        """CONTRACT. Without --selfcheck the CLI exits non-zero (fail closed)
        instead of pretending to have done something."""
        assert kb.main([]) == 2
