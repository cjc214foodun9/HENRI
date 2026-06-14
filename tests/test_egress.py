import pytest
import torch
from henri_core.egress import QuantizedEgressAssembler

def test_adc_quantization_and_ste():
    wave_dim = 128
    decoder_hidden_dim = 64
    vocab_size = 1000
    
    assembler = QuantizedEgressAssembler(
        wave_dim=wave_dim, 
        decoder_hidden_dim=decoder_hidden_dim, 
        vocab_size=vocab_size
    )
    
    # Generate continuous wave
    x = torch.randn(2, wave_dim, requires_grad=True)
    
    # 1. Forward pass of ADC
    quantized_scaled = assembler._simulate_4bit_adc(x)
    
    # Values should correspond to quantized wave in target scale
    assert quantized_scaled.shape == (2, wave_dim)
    assert not torch.isnan(quantized_scaled).any()
    
    # 2. Test STE Gradient propagation
    # If STE works, backpropagating through quantized_scaled should yield non-zero gradients on x
    loss = quantized_scaled.sum()
    loss.backward()
    
    assert x.grad is not None
    assert (x.grad != 0.0).any()
    # The gradient of quantized_wave (quantized - (wave*scale) + wave*scale) with respect to wave should be scale * 1 / scale = 1.
    # So gradient of sum(quantized_wave) with respect to wave should be approximately 1.
    assert torch.allclose(x.grad, torch.ones_like(x), atol=1e-3)

def test_autoregressive_egress_generation():
    wave_dim = 256
    decoder_hidden_dim = 128
    vocab_size = 5000
    
    assembler = QuantizedEgressAssembler(
        wave_dim=wave_dim, 
        decoder_hidden_dim=decoder_hidden_dim, 
        vocab_size=vocab_size
    )
    
    # Ingest wave: Batch=3, Dim=256
    final_hrr_wave = torch.randn(3, wave_dim)
    
    # Generate sequence of length 10
    tokens = assembler(final_hrr_wave, target_sequence_length=10)
    
    assert tokens.shape == (3, 10)
    assert ((tokens >= 0) & (tokens < vocab_size)).all()
