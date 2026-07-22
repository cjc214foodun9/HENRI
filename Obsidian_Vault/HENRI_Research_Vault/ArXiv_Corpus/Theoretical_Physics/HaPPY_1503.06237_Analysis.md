# HaPPY Paper: Holographic Quantum Error-Correcting Codes
**arXiv:1503.06237** | Pastawski, Yoshida, Harlow, Preskill (2015) | JHEP 06 (2015) 149

## Relevance to HENRI: LOW — structure mismatch

### What the paper ACTUALLY proves
- Perfect tensors (2n-index, any n→n bipartition = maximal entanglement) exist for qubits (v=2, n=3) and qutrits (v=3, n=2)
- Tensor networks of perfect tensors on hyperbolic tilings form QECCs
- Ryu-Takayanagi formula holds exactly for connected boundary regions
- Bulk operators can be "pushed" to the boundary via greedy algorithm on tensor indices
- Entanglement wedge = bulk region reconstructible from boundary subregion

### What DOES NOT transfer to HENRI
1. **Discrete qubits ≠ continuous waves**: HaPPY operates on finite-dim tensor products (v^n states). HENRI's waves are continuous complex vectors on S^(D-1). An index ranging over v=2 values has NO analog for a 65536-dim continuous phase.
2. **Hyperbolic tiling ≠ flat block grid**: HENRI's 8192 blocks form a flat 2D grid (90×90). HaPPY requires negatively curved hyperbolic tessellation for the RT formula.
3. **Perfect tensor = maximal bipartite entanglement**: This property depends on discrete index values. For continuous tensors, "maximal entanglement across any bipartition" doesn't translate.
4. **Operator pushing**: Relies on discrete isometry property T^†T = I — does not hold for continuous binding operators.

### What MIGHT transfer (speculative)
- The cross-covariance spectral entropy idea (not in HaPPY, but inspired by RT area law) could provide a non-collapsing epistemic signal
- The "bulk reconstruction from boundary" metaphor aligns with HENRI's EDMD constraint channel (field_V projects bulk→boundary)

### Vault location
- arXiv PDF: `ingest_input/arxiv_1503.06237_happy.pdf`
- Full text cached at: `C:\Users\chan\AppData\Local\hermes\cache\web\arxiv.org-3488f4a877.md`
