"""Forecast loss functions for Phase 4 EWMA outputs."""

from __future__ import annotations

import numpy as np

from src.utils import project_psd


def calculate_daily_mspe(forecasts: np.ndarray, ground_truths: np.ndarray) -> np.ndarray:
    """Return daily Frobenius norm squared forecast errors."""
    _validate_matching_arrays(forecasts, ground_truths)
    errors = forecasts - ground_truths
    return np.sum(np.square(errors), axis=(1, 2))


def calculate_daily_qlike(
    forecasts: np.ndarray,
    ground_truths: np.ndarray,
    floor: float = 1e-10,
) -> np.ndarray:
    """Return daily QLIKE losses using solve instead of explicit inversion."""
    _validate_matching_arrays(forecasts, ground_truths)
    losses: list[float] = []

    for forecast, ground_truth in zip(forecasts, ground_truths):
        forecast_psd = project_psd(forecast, floor=floor)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            sign, log_det = np.linalg.slogdet(forecast_psd)
        if sign <= 0:
            raise np.linalg.LinAlgError("Forecast matrix is not positive definite")
        if not np.isfinite(log_det):
            raise FloatingPointError("Forecast log determinant is not finite")

        solved = np.linalg.solve(forecast_psd, ground_truth)
        trace_term = float(np.trace(solved))
        loss = float(log_det + trace_term)
        if not np.isfinite(loss):
            raise FloatingPointError("QLIKE produced a non-finite value")
        losses.append(loss)

    return np.asarray(losses, dtype=np.float64)


def _validate_matching_arrays(forecasts: np.ndarray, ground_truths: np.ndarray) -> None:
    if forecasts.shape != ground_truths.shape:
        raise ValueError(f"Shape mismatch: forecasts={forecasts.shape}, ground_truths={ground_truths.shape}")
    if forecasts.ndim != 3 or forecasts.shape[1] != forecasts.shape[2]:
        raise ValueError(f"Expected arrays shaped (days, assets, assets), got {forecasts.shape}")
    if not (np.isfinite(forecasts).all() and np.isfinite(ground_truths).all()):
        raise ValueError("Forecast and ground-truth arrays must be finite")
