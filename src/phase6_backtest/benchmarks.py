"""Benchmark portfolio construction for Phase 6."""

from __future__ import annotations

import numpy as np


def equal_weight_targets(rebalance_dates: np.ndarray, n_assets: int) -> np.ndarray:
    weights = np.full((len(rebalance_dates), n_assets), 1.0 / n_assets, dtype=np.float64)
    return weights


def single_asset_targets(rebalance_dates: np.ndarray, tickers: np.ndarray, ticker: str) -> np.ndarray:
    tickers = np.asarray(tickers, dtype="U")
    matches = np.where(tickers == ticker)[0]
    if len(matches) != 1:
        raise ValueError(f"Ticker {ticker!r} must appear exactly once")
    weights = np.zeros((len(rebalance_dates), len(tickers)), dtype=np.float64)
    weights[:, int(matches[0])] = 1.0
    return weights
