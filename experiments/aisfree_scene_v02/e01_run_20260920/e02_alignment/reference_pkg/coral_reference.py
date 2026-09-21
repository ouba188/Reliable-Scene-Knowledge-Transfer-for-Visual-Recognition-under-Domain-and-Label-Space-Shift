"""Whitening/re-colouring reference, with samples stored as ROWS.

This is a numerical reference, not an adaptation training pipeline.
Fit only on the source-reference / explicitly allowed unlabelled adaptation pools.
Call .transform on evaluation data without refitting.

For origin O -> destination D:
  A = (Cov(O)+ridge*I)^(-1/2) @ (Cov(D)+ridge*I)^(1/2)
  X_new = (X - mean(O)) @ A + mean(D)

With ridge=0 and full-rank sample covariances, A.T @ Cov(O) @ A == Cov(D).
With ridge>0 the identity applies to the REGULARIZED covariances, not necessarily
exactly to the raw empirical covariance of the mapped samples.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def _samples(x: np.ndarray, *, min_rows: int = 1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < min_rows or x.shape[1] < 1:
        raise ValueError(f"Expected [N,D] with N>={min_rows} and D>=1, got {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("Samples contain NaN or infinity")
    return x


def symmetric_power(cov: np.ndarray, exponent: float) -> np.ndarray:
    cov = np.asarray(cov, dtype=np.float64)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or not np.isfinite(cov).all():
        raise ValueError("Expected a finite square covariance matrix")
    if not np.allclose(cov, cov.T, rtol=1e-10, atol=1e-12):
        raise ValueError("Covariance is not symmetric")
    w, v = np.linalg.eigh((cov + cov.T) * 0.5)
    scale = max(float(np.max(np.abs(w))), 1.0)
    if np.min(w) <= np.finfo(float).eps * scale:
        raise ValueError("Covariance must be positive definite; use positive ridge if needed")
    return (v * np.power(w, exponent)) @ v.T


@dataclass(frozen=True)
class AffineTransport:
    origin_mean: np.ndarray
    destination_mean: np.ndarray
    matrix: np.ndarray
    origin_covariance: np.ndarray
    destination_covariance: np.ndarray
    ridge: float

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = _samples(x)
        if x.shape[1] != self.matrix.shape[0]:
            raise ValueError("Evaluation feature dimension does not match fitted transport")
        return (x - self.origin_mean) @ self.matrix + self.destination_mean

    def regularized_covariance_residual(self) -> float:
        d = self.matrix.shape[0]
        co = self.origin_covariance + self.ridge * np.eye(d)
        cd = self.destination_covariance + self.ridge * np.eye(d)
        return float(np.linalg.norm(self.matrix.T @ co @ self.matrix - cd, ord='fro') /
                     max(np.linalg.norm(cd, ord='fro'), np.finfo(float).eps))


def fit_origin_to_destination(origin: np.ndarray, destination: np.ndarray,
                              ridge: float = 1e-6) -> AffineTransport:
    origin = _samples(origin, min_rows=2)
    destination = _samples(destination, min_rows=2)
    if origin.shape[1] != destination.shape[1]:
        raise ValueError("Origin and destination feature dimensions differ")
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be finite and nonnegative")
    co = np.atleast_2d(np.cov(origin, rowvar=False, ddof=1))
    cd = np.atleast_2d(np.cov(destination, rowvar=False, ddof=1))
    eye = np.eye(origin.shape[1])
    matrix = (symmetric_power(co + ridge * eye, -0.5) @
              symmetric_power(cd + ridge * eye, +0.5))
    return AffineTransport(origin.mean(0), destination.mean(0), matrix, co, cd, float(ridge))


def fit_target_to_source(source_reference: np.ndarray, target_adapt: np.ndarray,
                         ridge: float = 1e-6) -> AffineTransport:
    """Keep the source classifier fixed; map target-adapt/eval to source coordinates."""
    return fit_origin_to_destination(target_adapt, source_reference, ridge)


def fit_source_to_target(source_reference: np.ndarray, target_adapt: np.ndarray,
                         ridge: float = 1e-6) -> AffineTransport:
    """Map SOURCE before fitting a fresh classifier; evaluate on unmapped target features.
    This is the direction used by the original linear CORAL procedure, with explicit means.
    It is a separate baseline, not a drop-in prediction-only replacement.
    """
    return fit_origin_to_destination(source_reference, target_adapt, ridge)
