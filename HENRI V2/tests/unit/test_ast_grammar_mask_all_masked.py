"""Phase 4 contract tests: grammar-mask all-masked fail-closed behavior."""
import torch

from henri_ast_grammar_mask import (
    GrammarMaskAllMaskedError,
    HENRIASTGrammarMask,
)


def test_step0_all_masked_raises_typed_error():
    """A vocab without the mandatory 'def ' token must fail closed."""
    masker = HENRIASTGrammarMask(vocab_map={5: "False\n"})
    logits = torch.zeros(6)
    try:
        masker.mask_logits_for_step(logits, [], 0)
    except GrammarMaskAllMaskedError:
        return
    raise AssertionError("expected GrammarMaskAllMaskedError on all-masked step 0")


def test_step0_normal_vocab_returns_single_unmasked():
    masker = HENRIASTGrammarMask()
    logits = torch.zeros(10)
    out = masker.mask_logits_for_step(logits, [], 0)
    assert int(torch.argmax(out).item()) == 0


def test_1d_and_2d_preserved():
    masker = HENRIASTGrammarMask()
    logits_1d = torch.zeros(10)
    logits_2d = torch.zeros(1, 10)
    out1 = masker.mask_logits_for_step(logits_1d, [], 0)
    out2 = masker.mask_logits_for_step(logits_2d, [], 0)
    assert out1.dim() == 1
    assert out2.dim() == 2
