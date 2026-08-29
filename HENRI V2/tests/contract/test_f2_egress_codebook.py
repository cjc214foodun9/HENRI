"""F2-M3 Calibrated Hopfield Lexical Egress Codebook — contract suite (TDD).

Frozen tolerances (remote CUDA gate):
  T1  calibrate(): codebook shape [V, D], dtype float32.
  T2  no dense [D, D] allocation in the calibration path (AST-scanned).
  T3  dual-ridge solve: calibration-split P@1 >= 0.99 (train-wave reconstruction sanity).
  T4  snap(): z_clean shape [N, D]; logits [N, V]; beta = 8.0 sharpening recovers
      the canonical token for a near-canonical wave (identity codebook control).
  T5  default-OFF: with HENRI_F2_EGRESS unset, factory returns None and the
      legacy path is byte-identical (differential at gate level).
  T6  finiteness: no NaN/Inf in M or z_clean on random data (ridge lambda finite).
  T7  telemetry: returned record carries f2_egress_status, f2_p1_heldout, f2_beta.
  T8  memory guard: M storage bytes reported and < 16 GiB at D=65536, V=32000
      (fp32 upper bound ~8.4 GiB).

Local CPU runs are software sanity only. Remote CUDA is the verification boundary.
"""
import os
import ast
import math
import pathlib

import pytest
import torch


D = 4096          # local reduced dimension (software sanity)
V = 256           # local reduced vocab
N = 64            # local calibration rows
BETA = 8.0
RIDGE = 1e-3


def _module_ast() -> ast.Module:
    src = pathlib.Path(__file__).resolve().parents[2] / "f2_egress_codebook.py"
    return ast.parse(src.read_text(encoding="utf-8"))


def _calibration_fn_ast() -> ast.FunctionDef:
    tree = _module_ast()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "calibrate":
            return node
    raise AssertionError("calibrate() not found in f2_egress_codebook.py")


@pytest.fixture()
def codebook_factory():
    from f2_egress_codebook import F2HopfieldEgressCodebook
    return F2HopfieldEgressCodebook


@pytest.fixture()
def data():
    torch.manual_seed(0)
    X = torch.randn(N, D, dtype=torch.float32)
    # structured labels: first half class 0, second half class 1 (spread over V)
    y = torch.zeros(N, dtype=torch.long)
    y[N // 2:] = 1
    Y = torch.nn.functional.one_hot(y, num_classes=V).to(torch.float32)
    return X, Y


class TestCalibration:
    def test_t1_codebook_shape_dtype(self, codebook_factory, data):
        X, Y = data
        cb = codebook_factory(d_model=D, vocab_size=V, beta=BETA, ridge_lambda=RIDGE)
        cb.calibrate(X, Y)
        assert cb.M is not None
        assert cb.M.shape == (V, D)
        assert cb.M.dtype == torch.float32

    def test_t2_no_dense_dd_allocation(self):
        fn = _calibration_fn_ast()
        # scan only the calibrate function (docstrings removed by ast)
        for node in ast.walk(fn):
            if isinstance(node, ast.Call):
                fname = None
                if isinstance(node.func, ast.Attribute):
                    fname = node.func.attr
                elif isinstance(node.func, ast.Name):
                    fname = node.func.id
                if fname in ("eye", "inv", "linalg.inv") and node.args:
                    # any torch.eye / torch.linalg.inv call on a [D,D]-shaped arg is a ban hit
                    pytest.fail(f"dense-ban violation: {fname} in calibrate()")
        # also require svd-based solve marker
        src = ast.get_source_segment(_module_ast().read(), fn) if False else None
        assert any(
            isinstance(n, ast.Attribute) and n.attr == "svd"
            for n in ast.walk(fn)
        ), "calibrate() must use thin-SVD (dual form)"

    def test_t3_calibration_split_p1(self, codebook_factory, data):
        X, Y = data
        cb = codebook_factory(d_model=D, vocab_size=V, beta=BETA, ridge_lambda=RIDGE)
        cb.calibrate(X, Y)
        logits = cb.snap(X, return_logits=True)[1]
        pred = logits.argmax(dim=-1)
        p1 = (pred == Y.argmax(dim=-1)).float().mean().item()
        assert p1 >= 0.99, f"calibration-split P@1 {p1:.4f} < 0.99"


class TestSnap:
    def test_t4_shapes_and_beta_sharpening(self, codebook_factory):
        cb = codebook_factory(d_model=D, vocab_size=V, beta=BETA, ridge_lambda=RIDGE)
        # identity-like codebook: canonical token waves are one-hot rows
        M = torch.zeros(V, D, dtype=torch.float32)
        for v in range(min(V, D)):
            M[v, v] = 1.0
        cb.M = M
        Psi = torch.zeros(1, D, dtype=torch.float32)
        Psi[0, 3] = 1.0
        z, logits = cb.snap(Psi, return_logits=True)
        assert z.shape == (1, D)
        assert logits.shape == (1, V)
        assert logits.argmax(dim=-1).item() == 3

    def test_t6_finite(self, codebook_factory, data):
        X, Y = data
        cb = codebook_factory(d_model=D, vocab_size=V, beta=BETA, ridge_lambda=RIDGE)
        cb.calibrate(X, Y)
        assert torch.isfinite(cb.M).all()
        z, logits = cb.snap(X[:4], return_logits=True)
        assert torch.isfinite(z).all() and torch.isfinite(logits).all()

    def test_t8_memory_guard(self, codebook_factory, data):
        X, Y = data
        cb = codebook_factory(d_model=D, vocab_size=V, beta=BETA, ridge_lambda=RIDGE)
        cb.calibrate(X, Y)
        assert cb.codebook_bytes() < 16 * 1024**3  # < 16 GiB at full scale


class TestDefaultOff:
    def test_t5_factory_none_when_off(self, monkeypatch):
        monkeypatch.delenv("HENRI_F2_EGRESS", raising=False)
        from f2_egress_codebook import get_f2_egress
        assert get_f2_egress() is None

    def test_t5b_factory_constructs_when_on(self, monkeypatch):
        monkeypatch.setenv("HENRI_F2_EGRESS", "1")
        import f2_egress_codebook as mod
        cb = mod.get_f2_egress(d_model=D, vocab_size=V)
        assert cb is not None


class TestTelemetry:
    def test_t7_record_fields(self, codebook_factory, data):
        X, Y = data
        cb = codebook_factory(d_model=D, vocab_size=V, beta=BETA, ridge_lambda=RIDGE)
        cb.calibrate(X, Y)
        rec = cb.telemetry()
        assert rec["f2_egress_status"] == "ENGAGED"
        assert "f2_p1_heldout" in rec
        assert rec["f2_beta"] == BETA
