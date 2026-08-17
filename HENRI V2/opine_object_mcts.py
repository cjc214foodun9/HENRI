"""Phase 8.22 C1: OPINE Object-Centric Option Discovery Engine.

Spec: HENRI-SPEC-2026-08-PHASE8.21-8.22-WIRING (§2.1)
Gate: G1-8.22 — macro-option unitarity error < 1e-6.

Ontology-Error-Prioritized Interactive Exploration (OPINE) Object MCTS.
Groups primitive GameAction steps into temporal macro-options over su(3)^8192
to enable k-step temporal jumps in Zone B without single-step search decay.
"""
import torch


class OPINEObjectMCTS(torch.nn.Module):
    """Groups primitive action steps into temporal macro-options over
    su(3)^num_channels. Default-OFF additive component; never changes the
    production EFE control arm unless explicitly wired behind a flag."""

    def __init__(self, num_channels: int = 8192, option_horizon: int = 4):
        super().__init__()
        self.num_channels = num_channels
        self.option_horizon = option_horizon

    def construct_macro_option(
        self, generator_sequence: list,
        device: str | None = None,
    ) -> torch.Tensor:
        """Compose a sequence of [3,3] complex Lie generators into one
        [num_channels, 3, 3] unitary macro-operator.

        generator_sequence: list of [3,3] complex matrices (su(3) Lie
            elements, i.e. i * sum_k theta_k lambda_k). The same generator is
            applied to every channel (channel-homogeneous option); the
            composite is SU(3) in each channel when each exp(gen) is SU(3).
        Returns: [num_channels, 3, 3] composite unitary (det 1 per channel).
        """
        dev = device or (
            generator_sequence[0].device if generator_sequence else "cpu")
        composite_u = torch.eye(
            3, dtype=torch.complex64, device=dev
        ).repeat(self.num_channels, 1, 1)
        for gen in generator_sequence:
            u_step = torch.matrix_exp(gen.to(dev)).unsqueeze(0)
            composite_u = torch.einsum(
                "nij,njk->nik", composite_u, u_step.expand(
                    self.num_channels, 3, 3))
        return composite_u

    def unitarity_error(self, macro_option: torch.Tensor) -> float:
        """Mean ||U U^dag - I||_F over channels (G1-8.22 metric)."""
        diff = (
            torch.einsum("nij,nkj->nik", macro_option, macro_option.conj())
            - torch.eye(3, dtype=macro_option.dtype,
                        device=macro_option.device).unsqueeze(0)
        )
        return float(diff.abs().pow(2).sum(dim=(-2, -1)).sqrt().mean())


def _verify_opine_option() -> int:
    """G1-8.22 self-test: composed macro-option stays unitary (< 1e-6)."""
    torch.manual_seed(0)
    opine = OPINEObjectMCTS(num_channels=64, option_horizon=4)
    gens = []
    for _ in range(4):
        h = torch.randn(3, 3, dtype=torch.complex64)
        h = 0.5 * (h + h.conj().T)          # Hermitian
        h = h - torch.eye(3, dtype=torch.complex64) * (h.trace() / 3.0)  # traceless
        gens.append(1j * 0.1 * h)           # su(3) Lie element, small angle
    macro = opine.construct_macro_option(gens)
    err = opine.unitarity_error(macro)
    dets = torch.linalg.det(macro)
    det_err = float((dets - 1.0).abs().max())
    assert err < 1e-6, f"G1-8.22 FAIL: unitarity error {err:.3e}"
    assert det_err < 1e-5, f"G1-8.22 FAIL: det error {det_err:.3e}"
    print(f"[verify_opine_option] G1-8.22 PASS: unitarity error {err:.3e}, "
          f"det error {det_err:.3e}")
    return 0


if __name__ == "__main__":
    import argparse

    _ap = argparse.ArgumentParser()
    _ap.add_argument("--mode", default=None)
    _args = _ap.parse_args()
    if _args.mode == "verify_opine_option":
        raise SystemExit(_verify_opine_option())
    raise SystemExit(f"unknown --mode {_args.mode!r} "
                     f"(expected verify_opine_option)")
