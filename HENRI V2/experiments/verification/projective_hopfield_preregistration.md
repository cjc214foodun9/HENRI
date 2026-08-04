# Projective Hopfield Reference Experiment

Status: **pre-registered, default-off, not production evidence**

## Scope

This experiment compares a reference projective Hopfield update on local
\(\mathbb{C}P^{d-1}\) states with the existing flattened `ContinuousHopfieldCleanup`
control. It does not modify production egress, planner code, qFHRR codecs, Zone C,
checkpoints, or service configuration.

The projective component uses complex tensors:

- state: `[B, N, d]`
- memory bank: `[P, N, d]`
- local state: normalized complex qudit modulo independent U(1) phase

For neuron `i` it computes:

\[
O_\mu^{(i)} = \frac{2}{N}\sum_{j\ne i}
\left(|\langle s_j,\xi_j^\mu\rangle|^2-\frac{1}{d}\right),
\qquad
K_i=\sum_\mu O_\mu^{(i)}|\xi_i^\mu\rangle\langle\xi_i^\mu|.
\]

The update selects the largest eigenvector of the Hermitian kernel. The reference
implementation uses asynchronous sweeps. The paper's LLG energy theorem is not
transferred to this finite-precision eigensolver.

## Execution

Run only on the configured Vast CUDA target or canonical HENRI CI:

```text
python experiments/verification/projective_hopfield_cuda_matrix.py \
  --loads 4,8,16 \
  --corruptions 0.10,0.30,0.50 \
  --repeats 8 \
  --neurons 16 \
  --qudit-dim 4 \
  --sweeps 2 \
  --seed 20260804 \
  --output projective_hopfield_cuda_matrix.json
```

The script must fail closed when CUDA is unavailable. It must record the exact
interpreter, PyTorch/CUDA versions, GPU name, seed, dimensions, command, return
code, and per-cell results.

## Primary metrics

For each memory load and corruption level, record:

- projective top-1 retrieval rate;
- flattened-control top-1 retrieval rate;
- mean and minimum spectral gap;
- maximum Hermitian residual;
- maximum local norm error;
- projective and control latency;
- peak CUDA allocation;
- execution errors.

The control is a flattened cosine/codebook control. It is not claimed to be the
paper's spherical vector Hopfield baseline.

## Acceptance criteria

The reference component passes its implementation gate only if all are true:

1. CUDA execution completes with return code 0 and zero execution errors.
2. All contract tests pass.
3. Projective fidelity is invariant under independent local phase rotations.
4. The kernel Hermitian residual is at most `3e-5` in the contract tests.
5. The implementation's top eigenvalue agrees with `torch.linalg.eigvalsh` within
   `4e-5` relative/absolute tolerance.
6. Local norm error is at most `3e-5`.
7. No NaN or Inf values occur.
8. The declared kernel safety limit is enforced.

These criteria establish implementation validity only.

## Kill criteria

Mark the experiment **FALSIFIED** if any of the following occurs:

- phase rotation changes projective retrieval or ranking;
- the independent eigensolver check fails;
- non-finite values occur;
- the safety limit is bypassed or memory allocation exceeds the declared bound;
- the projective result fails to outperform the flattened control on the
  pre-registered matched matrix by more than the measured bootstrap uncertainty;
- the spectral gap collapses across the tested load before a meaningful
  comparison can be made;
- a benchmark or production claim is emitted from this component.

A failure to outperform the flattened control is a valid negative result. It does
not justify tuning thresholds or adding coherence terms.

## Interpretation boundary

A passing contract suite proves software and numerical component behavior. It does
not prove SU(d) capacity scaling, lossless continuous-to-discrete transduction,
HENRI benchmark improvement, or physical LLG realization. Production wiring needs
a separate approved design after this experiment.
