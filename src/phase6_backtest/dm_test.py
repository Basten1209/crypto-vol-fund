"""Diebold-Mariano style loss-differential test helpers."""

from __future__ import annotations

import math

import numpy as np


def diebold_mariano_test(loss_a: np.ndarray, loss_b: np.ndarray, lag: int = 0) -> dict[str, float | int]:
    """Return a two-sided normal-approximation DM test for loss_a - loss_b.

    Negative statistics indicate model/strategy A has lower average loss than B.
    The HAC variance uses Bartlett weights up to ``lag``.
    """
    loss_a = np.asarray(loss_a, dtype=np.float64)
    loss_b = np.asarray(loss_b, dtype=np.float64)
    if loss_a.shape != loss_b.shape:
        raise ValueError("DM test losses must have matching shapes")
    diff = loss_a - loss_b
    diff = diff[np.isfinite(diff)]
    n_obs = len(diff)
    if n_obs < 2:
        return {"n_obs": n_obs, "lag": int(lag), "mean_loss_diff": float("nan"), "dm_stat": float("nan"), "p_value": float("nan")}

    lag = int(max(0, min(lag, n_obs - 1)))
    centered = diff - np.mean(diff)
    gamma0 = float(np.dot(centered, centered) / n_obs)
    long_run_var = gamma0
    for k in range(1, lag + 1):
        gamma = float(np.dot(centered[k:], centered[:-k]) / n_obs)
        weight = 1.0 - k / (lag + 1)
        long_run_var += 2.0 * weight * gamma

    if long_run_var <= 0 or not np.isfinite(long_run_var):
        dm_stat = float("nan")
        p_value = float("nan")
    else:
        dm_stat = float(np.mean(diff) / math.sqrt(long_run_var / n_obs))
        p_value = float(math.erfc(abs(dm_stat) / math.sqrt(2.0)))

    return {
        "n_obs": n_obs,
        "lag": lag,
        "mean_loss_diff": float(np.mean(diff)),
        "dm_stat": dm_stat,
        "p_value": p_value,
    }
