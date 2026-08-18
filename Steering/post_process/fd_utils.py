"""Fréchet Distance utilities for activation-space evaluation.

True multivariate FD (the FID formula):

    FD(P, Q) = ‖μ_P - μ_Q‖² + Tr(Σ_P + Σ_Q − 2·sqrt(Σ_P · Σ_Q))

where sqrt is the matrix square root (via scipy.linalg.sqrtm).

**Why the old formula was wrong**
The previous implementation used:
    mean_d( (μ_d − μ_d')² + (σ_d − σ_d')² )
This is a per-dimension diagonal average — not the true FD.
The correct diagonal special case would still be a *sum*, not a mean:
    Σ_d (μ_d − μ_d')² + Σ_d (σ_d − σ_d')²
And the truly correct formula uses full covariance matrices.

Usage
-----
    from Steering.post_process.fd_utils import compute_fd, compute_fd_diagonal

    fd, fd_ratio = compute_fd(ref_tensor, gen_tensor, seed=42)
    fd_diag, fd_ratio_diag = compute_fd_diagonal(ref_tensor, gen_tensor, seed=42)
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_numpy_2d(t: torch.Tensor) -> np.ndarray:
    """Accept [N, D] or [N, 1, D] tensor → float64 numpy [N, D]."""
    if t.ndim == 3:
        t = t[:, 0, :]
    elif t.ndim != 2:
        raise ValueError(f"Expected 2-D or 3-D tensor, got shape {tuple(t.shape)}")
    return t.detach().cpu().to(torch.float64).numpy()


def _matrix_sqrt(M: np.ndarray) -> np.ndarray:
    """Numerically stable matrix square root via scipy sqrtm.

    Imaginary parts arise from floating-point noise; they are discarded
    after an assertion that they are small relative to the real part.
    """
    from scipy.linalg import sqrtm  # lazy import — scipy may not always be available

    sqrt_M, _ = sqrtm(M, disp=False)   # disp=False returns (result, est_err)
    if np.iscomplexobj(sqrt_M):
        imag_norm = np.linalg.norm(sqrt_M.imag)
        real_norm = np.linalg.norm(sqrt_M.real) + 1e-12
        if imag_norm / real_norm > 1e-3:
            warnings.warn(
                f"sqrtm imaginary part is large (|imag|/|real| = {imag_norm/real_norm:.4f}). "
                "FD may be inaccurate — ensure N >> D.",
                RuntimeWarning,
                stacklevel=3,
            )
        sqrt_M = sqrt_M.real
    return sqrt_M


def _noise_baseline(
    ref: np.ndarray,
    seed: int = 42,
) -> np.ndarray:
    """Standard-normal noise with shape [N, D] at fixed seed."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(ref.shape).astype(np.float64)


# ---------------------------------------------------------------------------
# Full multivariate Fréchet Distance
# ---------------------------------------------------------------------------

def compute_fd(
    ref_latents: torch.Tensor,
    gen_latents: torch.Tensor,
    seed: int = 42,
    eps: float = 1e-6,
) -> tuple[float, float]:
    """True multivariate Fréchet Distance between two activation sets.

    FD(ref, gen) = ‖μ_ref − μ_gen‖² + Tr(Σ_ref + Σ_gen − 2·sqrtm(Σ_ref @ Σ_gen))

    Parameters
    ----------
    ref_latents : Tensor [N, D] or [N, 1, D]   Reference / target activations.
    gen_latents : Tensor [N, D] or [N, 1, D]   Generated / denoised activations.
    seed        : int                            Seed for the noise baseline.
    eps         : float                          Ridge added to covariance diagonals
                                                 for numerical stability.

    Returns
    -------
    fd       : float   Fréchet Distance between ref and gen.
    fd_ratio : float   fd / fd_noise  (< 1 means gen is closer to ref than noise).
    """
    ref = _to_numpy_2d(ref_latents)
    gen = _to_numpy_2d(gen_latents)
    n = min(ref.shape[0], gen.shape[0])
    if n < 2:
        raise ValueError("Need at least 2 samples per set to compute FD.")

    ref, gen = ref[:n], gen[:n]
    noise = _noise_baseline(ref, seed=seed)

    def _fd_pair(a: np.ndarray, b: np.ndarray) -> float:
        mu_a = a.mean(0)
        mu_b = b.mean(0)
        sigma_a = np.cov(a, rowvar=False) + eps * np.eye(a.shape[1])
        sigma_b = np.cov(b, rowvar=False) + eps * np.eye(b.shape[1])

        mean_sq = float(np.sum((mu_a - mu_b) ** 2))
        sqrt_ab = _matrix_sqrt(sigma_a @ sigma_b)
        trace_term = float(np.trace(sigma_a + sigma_b - 2.0 * sqrt_ab))
        return mean_sq + trace_term

    fd = _fd_pair(ref, gen)
    fd_noise = _fd_pair(ref, noise)

    if fd_noise < 1e-8:
        return float("inf"), float("inf")
    return float(fd), float(fd / fd_noise)



# ---------------------------------------------------------------------------
# Per-dimension FD (for analysis DataFrames)
# ---------------------------------------------------------------------------

def compute_fd_1d(
    ref_latents: torch.Tensor,
    gen_latents: torch.Tensor,
) -> dict:
    """Per-dimension FD using the diagonal formula.

    Returns a dict with numpy arrays of length D:
        mu_ref, mu_gen, sigma_ref, sigma_gen, fd_1d

    fd_1d[d] = (μ_ref_d − μ_gen_d)² + (σ_ref_d − σ_gen_d)²
    (This is the correct per-dim contribution to FD_diagonal.)
    """
    ref = _to_numpy_2d(ref_latents).astype(np.float64)
    gen = _to_numpy_2d(gen_latents).astype(np.float64)
    n = min(ref.shape[0], gen.shape[0])
    ref, gen = ref[:n], gen[:n]

    mu_ref   = ref.mean(0);   sigma_ref   = ref.std(0, ddof=0)
    mu_gen   = gen.mean(0);   sigma_gen   = gen.std(0, ddof=0)

    fd_1d = (mu_ref - mu_gen) ** 2 + (sigma_ref - sigma_gen) ** 2

    return {
        "mu_ref":    mu_ref,
        "mu_gen":    mu_gen,
        "sigma_ref": sigma_ref,
        "sigma_gen": sigma_gen,
        "fd_1d":     fd_1d,
    }
