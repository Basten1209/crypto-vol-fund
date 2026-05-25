"""Performance metrics for Phase 6 backtests."""

from __future__ import annotations

import numpy as np


def annualized_return(daily_returns: np.ndarray, annualization: int = 365) -> float:
    daily_returns = np.asarray(daily_returns, dtype=np.float64)
    if daily_returns.size == 0:
        return float("nan")
    total_return = float(np.prod(1.0 + daily_returns) - 1.0)
    return (1.0 + total_return) ** (annualization / len(daily_returns)) - 1.0


def annualized_volatility(daily_returns: np.ndarray, annualization: int = 365) -> float:
    daily_returns = np.asarray(daily_returns, dtype=np.float64)
    if daily_returns.size < 2:
        return float("nan")
    return float(np.std(daily_returns, ddof=1) * np.sqrt(annualization))


def sharpe_ratio(daily_returns: np.ndarray, risk_free_rate: float = 0.0, annualization: int = 365) -> float:
    daily_returns = np.asarray(daily_returns, dtype=np.float64)
    daily_rf = risk_free_rate / annualization
    excess = daily_returns - daily_rf
    vol = np.std(excess, ddof=1)
    if len(excess) < 2 or vol <= 0:
        return float("nan")
    return float(np.mean(excess) / vol * np.sqrt(annualization))


def max_drawdown(equity: np.ndarray) -> float:
    equity = np.asarray(equity, dtype=np.float64)
    if equity.size == 0:
        return float("nan")
    peaks = np.maximum.accumulate(equity)
    drawdowns = equity / peaks - 1.0
    return float(np.min(drawdowns))


def calmar_ratio(daily_returns: np.ndarray, equity: np.ndarray, annualization: int = 365) -> float:
    ann_return = annualized_return(daily_returns, annualization=annualization)
    mdd = max_drawdown(equity)
    if not np.isfinite(mdd) or mdd >= 0:
        return float("nan")
    return float(ann_return / abs(mdd))


def information_ratio(
    strategy_daily_returns: np.ndarray,
    benchmark_daily_returns: np.ndarray,
    annualization: int = 365,
) -> float:
    strategy_daily_returns = np.asarray(strategy_daily_returns, dtype=np.float64)
    benchmark_daily_returns = np.asarray(benchmark_daily_returns, dtype=np.float64)
    if strategy_daily_returns.shape != benchmark_daily_returns.shape or len(strategy_daily_returns) < 2:
        return float("nan")
    active = strategy_daily_returns - benchmark_daily_returns
    tracking_error = np.std(active, ddof=1)
    if tracking_error <= 0:
        return float("nan")
    return float(np.mean(active) / tracking_error * np.sqrt(annualization))


def performance_row(
    strategy: str,
    cycle_days: int | None,
    daily_returns: np.ndarray,
    equity: np.ndarray,
    btc_daily_returns: np.ndarray | None = None,
    turnover_mean: float | None = None,
    realized_risk_mean: float | None = None,
    annualization: int = 365,
    risk_free_rate: float = 0.0,
) -> dict[str, float | int | str | None]:
    total_return = float(equity[-1] - 1.0) if len(equity) else float("nan")
    return {
        "strategy": strategy,
        "cycle_days": cycle_days,
        "n_days": int(len(daily_returns)),
        "total_return": total_return,
        "annualized_return": annualized_return(daily_returns, annualization=annualization),
        "annualized_volatility": annualized_volatility(daily_returns, annualization=annualization),
        "sharpe_ratio": sharpe_ratio(
            daily_returns,
            risk_free_rate=risk_free_rate,
            annualization=annualization,
        ),
        "max_drawdown": max_drawdown(equity),
        "calmar_ratio": calmar_ratio(daily_returns, equity, annualization=annualization),
        "information_ratio_vs_btc": information_ratio(
            daily_returns,
            btc_daily_returns,
            annualization=annualization,
        )
        if btc_daily_returns is not None
        else float("nan"),
        "turnover_mean": turnover_mean,
        "realized_risk_annualized_mean": realized_risk_mean,
    }
