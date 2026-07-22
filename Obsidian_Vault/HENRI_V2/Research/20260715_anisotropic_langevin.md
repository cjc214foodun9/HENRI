---
id: "anisotropic_langevin_dynamics"
created_at: "2026-07-15T10:00:00Z"
updated_at: "2026-07-15T10:00:00Z"
module: "Anisotropic Langevin"
tags:
  - henri-v2/theory
status: "verified"
---

# Anisotropic Langevin Noise Injection

The SGLD noise covariance Gamma(theta) is anisotropic in phase space. Noise
must scale as sqrt(2*T*dt), not raw T — raw-T noise on near-orthonormal
matrices blows the Newton-Schulz retraction basin on step one. Use Cholesky
orthogonalization (A <- L^{-1}A) which is unconditionally stable.
