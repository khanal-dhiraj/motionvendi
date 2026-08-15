"""Similarity kernels over behavior vectors.

The kernel IS the definition of "same behavior": k(i, j) ~ 1 means episodes i
and j differ only by nuisance (already quotiented in normalize.py), k ~ 0
means genuinely different motion. The Vendi score is only as honest as this
matrix, so we (a) validate PSD explicitly and (b) expose the bandwidth sweep —
too wide collapses everything (metric goes blind), too narrow calls everything
unique (VS -> n via distance concentration).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform


def pairwise_sq_dists(X: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance matrix for rows of X (n, d)."""
    return squareform(pdist(np.asarray(X, dtype=np.float64), metric="sqeuclidean"))


def median_bandwidth(X: np.ndarray) -> float:
    """Median heuristic: sigma^2 = median of pairwise squared distances / 2."""
    d2 = pdist(np.asarray(X, dtype=np.float64), metric="sqeuclidean")
    med = float(np.median(d2[d2 > 0])) if np.any(d2 > 0) else 1.0
    return med / 2.0


def rbf_kernel(X: np.ndarray, sigma2: float | None = None) -> np.ndarray:
    """RBF (Gaussian) kernel with unit diagonal; PSD by construction."""
    if sigma2 is None:
        sigma2 = median_bandwidth(X)
    K = np.exp(-pairwise_sq_dists(X) / (2.0 * sigma2))
    np.fill_diagonal(K, 1.0)
    return K


def validate_psd(K: np.ndarray, tol: float = 1e-8) -> tuple[bool, float]:
    """Check symmetry + eigenvalue floor. Returns (ok, min_eigenvalue)."""
    if not np.allclose(K, K.T, atol=1e-10):
        return False, float("nan")
    eigs = np.linalg.eigvalsh(K)
    return bool(eigs.min() > -tol), float(eigs.min())


def combine_kernels(kernels: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """Convex combination of PSD kernels — PSD only because weights are
    CONSTANT across the matrix. Never renormalize weights per-pair: entry-wise
    reweighting voids the PSD guarantee. If a modality is missing for some
    episodes, restrict the analysis run to a modality-complete subset instead.
    """
    w = np.asarray(weights, dtype=np.float64)
    if np.any(w < 0) or not np.isclose(w.sum(), 1.0):
        raise ValueError("weights must be non-negative and sum to 1")
    out = np.zeros_like(kernels[0])
    for K, wi in zip(kernels, w):
        if K.shape != out.shape:
            raise ValueError("kernel shape mismatch")
        out += wi * K
    return out


def kernel_histogram_stats(K: np.ndarray) -> dict:
    """Off-diagonal similarity stats — the cheap sanity check for a degenerate
    kernel (near-diagonal => VS saturates at n; near-ones => VS collapses to 1).
    """
    off = K[~np.eye(len(K), dtype=bool)]
    return {
        "mean": float(off.mean()),
        "p05": float(np.percentile(off, 5)),
        "p50": float(np.percentile(off, 50)),
        "p95": float(np.percentile(off, 95)),
    }
