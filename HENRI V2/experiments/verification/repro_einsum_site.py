#!/usr/bin/env python3
"""Reproduce the einsum width error at the CALL boundary, and name the site.

Hypothesis measured here (from source reading):
    displacement  <- store.theta_a[action] with store.num_channels = 64   (my fix)
    U_t           <- su3_field, produced by _pad_su3_field(nb=8192)       (unfixed)
    -> einsum("nij,njk->nik", [64,3,3], [8192,3,3]) must raise
       "subscript n has size 8192 for operand 1 ... previously seen size 64"

A mismatch of this exact form, at this exact line, is the evidence that names the site.
"""
import sys, traceback, torch

torch.manual_seed(0)

# Minimal stand-ins for the two producers, at the two known widths.
NB_PAD = 8192        # _pad_su3_field hardcoded default
NB_STORE = 64        # SCALE["num_blocks"] bound into the store by the flag fix

def displacement(n):
    return torch.eye(3, dtype=torch.complex64).unsqueeze(0).repeat(n, 1, 1)

U_t = torch.eye(3, dtype=torch.complex64).unsqueeze(0).repeat(NB_PAD, 1, 1)
disp = displacement(NB_STORE)
print(f"operand0 displacement = {tuple(disp.shape)}  (store num_channels={NB_STORE})")
print(f"operand1 U_t          = {tuple(U_t.shape)}  (padded field nb={NB_PAD})")
try:
    torch.einsum("nij,njk->nik", disp, U_t)
    print("REPRO=NO_ERROR  <- hypothesis FALSIFIED: this pairing does not raise")
except RuntimeError as e:
    msg = str(e)
    print(f"REPRO=RAISED     {type(e).__name__}: {msg}")
    match = ("8192" in msg and "operand 1" in msg and "64" in msg)
    print(f"REPRO_MATCHES_OBSERVED_ERROR={match}")

# And prove the fix direction: matching widths succeed.
disp2 = displacement(NB_PAD)
try:
    out = torch.einsum("nij,njk->nik", disp2, U_t)
    print(f"MATCHED_WIDTH_OK out={tuple(out.shape)}  <- equal widths compose cleanly")
except RuntimeError as e:
    print(f"MATCHED_WIDTH_FAILED {e}")
