"""Long-only minimum variance portfolio optimizer for Phase 5.

This module consumes Phase 4 EWMA forecasts and Phase 3 jump-volatility
matrices, then writes cycle-specific rebalance weights for downstream
backtesting.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.utils import ensure_dir, project_psd, repo_relative_path  # noqa: E402


@dataclass(frozen=True)
class MinimumVarianceParams:
    cycles: tuple[int, ...] = tuple(config.CYCLES)
    rebalance_frequency: str = "cycle"
    gross_exposure: float = config.MIN_VAR_C0
    single_asset_cap: float | None = config.SINGLE_ASSET_CAP
    min_asset_weight: float = config.MIN_ASSET_WEIGHT
    psd_floor: float = config.PSD_FLOOR
    active_tol: float = 1e-6
    kkt_tol: float = 1e-8
    max_active_iter: int = 500
    max_projected_gradient_iter: int = 20_000

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycles": list(self.cycles),
            "rebalance_frequency": self.rebalance_frequency,
            "gross_exposure": self.gross_exposure,
            "single_asset_cap": self.single_asset_cap,
            "min_asset_weight": self.min_asset_weight,
            "psd_floor": self.psd_floor,
            "active_tol": self.active_tol,
            "kkt_tol": self.kkt_tol,
            "max_active_iter": self.max_active_iter,
            "max_projected_gradient_iter": self.max_projected_gradient_iter,
            "method": "active_set_long_only_minimum_variance_with_min_weight_pruning"
            if self.min_asset_weight > 0
            else "active_set_long_only_minimum_variance",
        }


def solve_long_only_minimum_variance(
    covariance: np.ndarray,
    params: MinimumVarianceParams | None = None,
) -> dict[str, Any]:
    """Solve min w'Qw subject to sum(w)=1, w>=0, and optional w<=cap.

    With c0=1 and sum(w)=1, the plan's ||w||_1 <= c0 constraint is equivalent
    to long-only weights. A single-asset cap adds an upper-bound constraint.
    A projected-gradient fallback is included for numerical edge cases, but the
    active-set KKT solver should handle the normal 50-asset daily matrices.
    """
    params = params or MinimumVarianceParams()
    if abs(params.gross_exposure - 1.0) > 1e-12:
        raise NotImplementedError("Only gross_exposure=1 is supported for long-only minimum variance")

    q = _validate_covariance(covariance, params.psd_floor)
    cap = _resolve_single_asset_cap(params.single_asset_cap, q.shape[0])
    _resolve_min_asset_weight(params.min_asset_weight, cap)
    if cap is not None:
        return _solve_capped_minimum_variance(q, cap, params)

    active = list(range(q.shape[0]))
    last_state: tuple[int, ...] | None = None

    for iteration in range(1, params.max_active_iter + 1):
        weights, nu, used_lstsq = _restricted_minimum_variance(q, active)
        if not np.isfinite(weights).all():
            break

        min_pos = int(np.argmin(weights))
        if weights[min_pos] < -params.active_tol and len(active) > 1:
            active.pop(min_pos)
            state = tuple(active)
            if state == last_state:
                break
            last_state = state
            continue

        full_weights = np.zeros(q.shape[0], dtype=np.float64)
        full_weights[active] = np.maximum(weights, 0.0)
        full_weights = _normalize_simplex(full_weights)

        grad = np.dot(q, full_weights)
        active_mask = full_weights > params.active_tol
        if active_mask.any():
            nu = float(np.mean(grad[active_mask]))
        inactive_idx = np.where(~active_mask)[0]
        if len(inactive_idx):
            violations = nu - grad[inactive_idx]
            worst_pos = int(np.argmax(violations))
            if violations[worst_pos] > params.kkt_tol:
                candidate = int(inactive_idx[worst_pos])
                if candidate not in active:
                    active.append(candidate)
                    active.sort()
                    state = tuple(active)
                    if state == last_state:
                        break
                    last_state = state
                    continue

        return _solution_result(
            q=q,
            weights=full_weights,
            status="optimal_active_set_lstsq" if used_lstsq else "optimal_active_set",
            iterations=iteration,
            params=params,
        )

    fallback = _projected_gradient_minimum_variance(q, params)
    fallback["status"] = f"fallback_projected_gradient_after_active_set:{fallback['status']}"
    return fallback


def compute_phase5_portfolios(
    forecast_path: Path | str,
    prvm_path: Path | str,
    output_dir: Path | str,
    cycles: tuple[int, ...] | list[int] | None = None,
    limit_rebalances: int | None = None,
    single_asset_cap: float | None = None,
    min_asset_weight: float | None = None,
    rebalance_frequency: str = "cycle",
) -> dict[str, Any]:
    """Run Phase 5 optimization and write cycle-specific artifacts."""
    forecast_path = Path(forecast_path)
    prvm_path = Path(prvm_path)
    output_dir = Path(output_dir)
    params = MinimumVarianceParams(
        cycles=tuple(config.CYCLES if cycles is None else cycles),
        rebalance_frequency=str(rebalance_frequency),
        single_asset_cap=config.SINGLE_ASSET_CAP if single_asset_cap is None else float(single_asset_cap),
        min_asset_weight=config.MIN_ASSET_WEIGHT if min_asset_weight is None else float(min_asset_weight),
    )
    if params.rebalance_frequency not in {"cycle", "monthly"}:
        raise ValueError("rebalance_frequency must be one of: cycle, monthly")
    if limit_rebalances is not None and limit_rebalances < 1:
        raise ValueError("limit_rebalances must be >= 1")

    loaded = _load_inputs(forecast_path, prvm_path)
    objective_matrices, lagged_jv_dates = _build_objective_matrices(
        forecasts=loaded["forecasts"],
        target_dates=loaded["target_dates"],
        origin_dates=loaded["origin_dates"],
        prvm_dates=loaded["prvm_dates"],
        jv=loaded["jv"],
        params=params,
    )

    cycle_results: dict[int, dict[str, Any]] = {}
    for cycle in params.cycles:
        cycle_results[int(cycle)] = _optimize_cycle(
            cycle_days=int(cycle),
            target_dates=loaded["target_dates"],
            origin_dates=loaded["origin_dates"],
            lagged_jv_dates=lagged_jv_dates,
            tickers=loaded["tickers"],
            objective_matrices=objective_matrices,
            params=params,
            limit_rebalances=limit_rebalances,
        )

    report = _write_outputs(
        output_dir=output_dir,
        forecast_path=forecast_path,
        prvm_path=prvm_path,
        params=params,
        tickers=loaded["tickers"],
        n_forecast_days=len(loaded["target_dates"]),
        cycle_results=cycle_results,
        limit_rebalances=limit_rebalances,
    )

    print("=== Phase 5 minimum variance portfolio ===")
    print(f"forecast_input: {forecast_path}")
    print(f"prvm_input: {prvm_path}")
    print(f"output_dir: {output_dir}")
    for cycle in params.cycles:
        summary = report["cycle_summary"][str(int(cycle))]
        print(
            f"cycle={int(cycle)}d rebalances={summary['n_rebalances']} "
            f"active_mean={summary['active_count_mean']:.3g} "
            f"top_weight_mean={summary['top_weight_mean']:.3g} "
            f"max_kkt_violation={summary['kkt_violation_max']:.3g}"
        )
    print(f"saved npz: {report['outputs']['npz']}")
    print(f"saved report: {report['outputs']['report_json']}")
    return report


def _validate_covariance(covariance: np.ndarray, floor: float) -> np.ndarray:
    q = np.asarray(covariance, dtype=np.float64)
    if q.ndim != 2 or q.shape[0] != q.shape[1]:
        raise ValueError(f"Expected square covariance matrix, got {q.shape}")
    if not np.isfinite(q).all():
        raise ValueError("Covariance matrix contains NaN or Inf")
    return project_psd(q, floor=floor)


def _restricted_minimum_variance(q: np.ndarray, active: list[int]) -> tuple[np.ndarray, float, bool]:
    sub = q[np.ix_(active, active)]
    ones = np.ones(len(active), dtype=np.float64)
    used_lstsq = False
    try:
        raw = np.linalg.solve(sub, ones)
    except np.linalg.LinAlgError:
        raw = np.linalg.lstsq(sub, ones, rcond=None)[0]
        used_lstsq = True
    denom = float(ones @ raw)
    if not np.isfinite(denom) or denom <= 0:
        return np.full(len(active), np.nan), float("nan"), used_lstsq
    weights = raw / denom
    nu = 1.0 / denom
    return weights, nu, used_lstsq


def _resolve_single_asset_cap(cap: float | None, n_assets: int) -> float | None:
    if cap is None:
        return None
    cap = float(cap)
    if cap <= 0:
        raise ValueError("single_asset_cap must be positive")
    if cap * n_assets < 1.0 - 1e-12:
        raise ValueError(f"single_asset_cap={cap} is infeasible for {n_assets} assets")
    if cap >= 1.0:
        return None
    return cap


def _resolve_min_asset_weight(min_weight: float | None, cap: float | None) -> float:
    if min_weight is None:
        return 0.0
    min_weight = float(min_weight)
    if min_weight < 0:
        raise ValueError("min_asset_weight must be non-negative")
    if min_weight >= 1.0:
        raise ValueError("min_asset_weight must be below 1")
    if cap is not None and min_weight > cap + 1e-12:
        raise ValueError("min_asset_weight cannot exceed single_asset_cap")
    return min_weight


def _solve_capped_minimum_variance(
    q: np.ndarray,
    cap: float,
    params: MinimumVarianceParams,
) -> dict[str, Any]:
    n_assets = q.shape[0]
    lower = np.zeros(n_assets, dtype=bool)
    upper = np.zeros(n_assets, dtype=bool)
    previous_state: tuple[tuple[int, ...], tuple[int, ...]] | None = None

    for iteration in range(1, params.max_active_iter + 1):
        weights, lambda_, used_lstsq = _restricted_capped_minimum_variance(q, lower, upper, cap)
        if not np.isfinite(weights).all():
            break

        free = ~(lower | upper)
        if np.any(free):
            free_idx = np.where(free)[0]
            below_idx = free_idx[weights[free_idx] < -params.active_tol]
            above_idx = free_idx[weights[free_idx] > cap + params.active_tol]
            if len(below_idx) or len(above_idx):
                below_violation = -np.min(weights[below_idx]) if len(below_idx) else -np.inf
                above_violation = np.max(weights[above_idx] - cap) if len(above_idx) else -np.inf
                if below_violation >= above_violation:
                    lower[int(below_idx[np.argmin(weights[below_idx])])] = True
                else:
                    upper[int(above_idx[np.argmax(weights[above_idx] - cap)])] = True
                state = _bound_state(lower, upper)
                if state == previous_state:
                    break
                previous_state = state
                continue

        weights = _project_capped_simplex(weights, cap)
        grad = np.dot(q, weights)
        free = (weights > params.active_tol) & (weights < cap - params.active_tol)
        if np.any(free):
            lambda_ = float(np.mean(grad[free]))

        lower_bound = weights <= params.active_tol
        upper_bound = weights >= cap - params.active_tol
        lower_violations = lambda_ - grad[lower_bound]
        upper_violations = grad[upper_bound] - lambda_
        lower_violation = float(np.max(lower_violations)) if len(lower_violations) else 0.0
        upper_violation = float(np.max(upper_violations)) if len(upper_violations) else 0.0
        if max(lower_violation, upper_violation) > params.kkt_tol:
            if lower_violation >= upper_violation:
                lower_candidates = np.where(lower_bound)[0]
                release_idx = int(lower_candidates[np.argmax(lambda_ - grad[lower_candidates])])
                lower[release_idx] = False
            else:
                upper_candidates = np.where(upper_bound)[0]
                release_idx = int(upper_candidates[np.argmax(grad[upper_candidates] - lambda_)])
                upper[release_idx] = False
            state = _bound_state(lower, upper)
            if state == previous_state:
                break
            previous_state = state
            continue

        return _solution_result(
            q=q,
            weights=weights,
            status="optimal_capped_active_set_lstsq" if used_lstsq else "optimal_capped_active_set",
            iterations=iteration,
            params=params,
        )

    fallback = _projected_gradient_minimum_variance(q, params)
    fallback["status"] = f"fallback_projected_gradient_after_capped_active_set:{fallback['status']}"
    return fallback


def _restricted_capped_minimum_variance(
    q: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    cap: float,
) -> tuple[np.ndarray, float, bool]:
    n_assets = q.shape[0]
    weights = np.zeros(n_assets, dtype=np.float64)
    weights[upper] = cap
    free = ~(lower | upper)
    free_idx = np.where(free)[0]
    target_sum = 1.0 - float(np.sum(weights))
    used_lstsq = False

    if len(free_idx) == 0:
        if abs(target_sum) <= 1e-10:
            grad = np.dot(q, weights)
            lambda_ = float(np.mean(grad))
            return weights, lambda_, used_lstsq
        return np.full(n_assets, np.nan), float("nan"), used_lstsq

    sub = q[np.ix_(free_idx, free_idx)]
    ones = np.ones(len(free_idx), dtype=np.float64)
    fixed_term = np.dot(q[np.ix_(free_idx, np.where(~free)[0])], weights[~free]) if np.any(~free) else 0.0
    try:
        inv_ones = np.linalg.solve(sub, ones)
        inv_fixed = np.linalg.solve(sub, fixed_term) if np.any(~free) else np.zeros(len(free_idx), dtype=np.float64)
    except np.linalg.LinAlgError:
        inv_ones = np.linalg.lstsq(sub, ones, rcond=None)[0]
        inv_fixed = (
            np.linalg.lstsq(sub, fixed_term, rcond=None)[0]
            if np.any(~free)
            else np.zeros(len(free_idx), dtype=np.float64)
        )
        used_lstsq = True

    denom = float(np.dot(ones, inv_ones))
    if not np.isfinite(denom) or denom <= 0:
        return np.full(n_assets, np.nan), float("nan"), used_lstsq
    lambda_ = (target_sum + float(np.dot(ones, inv_fixed))) / denom
    weights[free_idx] = lambda_ * inv_ones - inv_fixed
    return weights, float(lambda_), used_lstsq


def _bound_state(lower: np.ndarray, upper: np.ndarray) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return tuple(np.where(lower)[0]), tuple(np.where(upper)[0])


def _project_simplex(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    sorted_values = np.sort(values)[::-1]
    cssv = np.cumsum(sorted_values) - 1.0
    ind = np.arange(1, len(values) + 1)
    cond = sorted_values - cssv / ind > 0
    if not np.any(cond):
        return np.full_like(values, 1.0 / len(values))
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / rho
    return np.maximum(values - theta, 0.0)


def _project_capped_simplex(values: np.ndarray, cap: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if cap is None:
        return _project_simplex(values)
    if cap * len(values) < 1.0 - 1e-12:
        raise ValueError("Capped simplex is infeasible")
    low = float(np.min(values - cap))
    high = float(np.max(values))
    for _ in range(100):
        mid = (low + high) / 2.0
        projected = np.clip(values - mid, 0.0, cap)
        if float(np.sum(projected)) > 1.0:
            low = mid
        else:
            high = mid
    projected = np.clip(values - high, 0.0, cap)
    total = float(np.sum(projected))
    if total <= 0:
        return np.full_like(values, 1.0 / len(values))
    return projected / total


def _project_bounded_simplex(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("values must be a non-empty 1D array")
    if lower < 0 or upper <= 0 or lower > upper:
        raise ValueError("Invalid bounded simplex limits")
    n_values = len(values)
    lower_sum = lower * n_values
    upper_sum = upper * n_values
    if lower_sum > 1.0 + 1e-12 or upper_sum < 1.0 - 1e-12:
        raise ValueError("Bounded simplex is infeasible")
    if abs(lower_sum - 1.0) <= 1e-12:
        return np.full_like(values, lower)
    if abs(upper_sum - 1.0) <= 1e-12:
        return np.full_like(values, upper)

    low = float(np.min(values - upper))
    high = float(np.max(values - lower))
    for _ in range(100):
        mid = (low + high) / 2.0
        projected = np.clip(values - mid, lower, upper)
        if float(np.sum(projected)) > 1.0:
            low = mid
        else:
            high = mid

    projected = np.clip(values - high, lower, upper)
    residual = 1.0 - float(np.sum(projected))
    if abs(residual) > 1e-12:
        if residual > 0:
            room = upper - projected
            recipients = room > 1e-12
            if np.any(recipients):
                projected[recipients] += residual * room[recipients] / float(np.sum(room[recipients]))
        else:
            room = projected - lower
            recipients = room > 1e-12
            if np.any(recipients):
                projected[recipients] += residual * room[recipients] / float(np.sum(room[recipients]))
    return projected


def _projected_gradient_minimum_variance(q: np.ndarray, params: MinimumVarianceParams) -> dict[str, Any]:
    n_assets = q.shape[0]
    cap = _resolve_single_asset_cap(params.single_asset_cap, n_assets)
    eig_max = float(np.max(np.linalg.eigvalsh(q)))
    step = 1.0 / max(eig_max, params.psd_floor)
    weights = np.full(n_assets, 1.0 / n_assets, dtype=np.float64)
    status = "max_iter"

    for iteration in range(1, params.max_projected_gradient_iter + 1):
        grad = np.dot(q, weights)
        next_weights = _project_capped_simplex(weights - step * grad, cap) if cap is not None else _project_simplex(weights - step * grad)
        if np.linalg.norm(next_weights - weights, ord=1) <= params.kkt_tol:
            weights = next_weights
            status = "converged"
            break
        weights = next_weights

    return _solution_result(q=q, weights=weights, status=status, iterations=iteration, params=params)


def _normalize_simplex(weights: np.ndarray) -> np.ndarray:
    weights = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0:
        return np.full_like(weights, 1.0 / len(weights))
    return weights / total


def _enforce_min_asset_weight(
    weights: np.ndarray,
    params: MinimumVarianceParams,
    cap: float | None,
) -> np.ndarray:
    min_weight = _resolve_min_asset_weight(params.min_asset_weight, cap)
    if min_weight <= 0:
        return weights

    weights = np.asarray(weights, dtype=np.float64)
    n_assets = len(weights)
    upper = cap if cap is not None else 1.0
    positive_floor = float(np.nextafter(min_weight, np.inf))
    order = np.argsort(weights)[::-1]
    active = weights > min_weight
    if not np.any(active):
        active[order[0]] = True

    min_active_count = 1 if cap is None else int(np.ceil((1.0 - 1e-12) / cap))
    for candidate in order:
        if int(np.sum(active)) >= min_active_count:
            break
        active[int(candidate)] = True

    while float(np.sum(active)) * positive_floor > 1.0 + 1e-12:
        removable = [int(idx) for idx in order[::-1] if active[int(idx)]]
        removed = False
        for candidate in removable:
            next_count = int(np.sum(active)) - 1
            if next_count < min_active_count:
                continue
            if cap is not None and next_count * cap < 1.0 - 1e-12:
                continue
            active[candidate] = False
            removed = True
            break
        if not removed:
            raise ValueError("min_asset_weight is infeasible with the selected active set")

    while cap is not None and float(np.sum(active)) * cap < 1.0 - 1e-12:
        added = False
        for candidate in order:
            candidate = int(candidate)
            if not active[candidate]:
                active[candidate] = True
                added = True
                break
        if not added:
            raise ValueError("single_asset_cap is infeasible after min-weight pruning")

    active_idx = np.where(active)[0]
    projected_active = _project_bounded_simplex(
        values=weights[active_idx],
        lower=positive_floor,
        upper=upper,
    )
    pruned = np.zeros(n_assets, dtype=np.float64)
    pruned[active_idx] = projected_active
    total = float(np.sum(pruned))
    if not np.isfinite(total) or total <= 0:
        raise ValueError("min-weight pruning produced invalid weights")
    if abs(total - 1.0) > 1e-10:
        pruned[active_idx] = _project_bounded_simplex(
            values=pruned[active_idx],
            lower=positive_floor,
            upper=upper,
        )
    return pruned


def _min_positive_weight(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=np.float64)
    positive = weights[weights > 0.0]
    if positive.size == 0:
        return 0.0
    return float(np.min(positive))


def _min_asset_weight_violation(weights: np.ndarray, params: MinimumVarianceParams) -> float:
    cap = _resolve_single_asset_cap(params.single_asset_cap, len(weights))
    min_weight = _resolve_min_asset_weight(params.min_asset_weight, cap)
    if min_weight <= 0:
        return 0.0
    min_positive = _min_positive_weight(weights)
    if min_positive <= 0:
        return float("inf")
    return float(max(0.0, min_weight - min_positive))


def _kkt_violation(q: np.ndarray, weights: np.ndarray, params: MinimumVarianceParams) -> float:
    grad = np.dot(q, weights)
    cap = _resolve_single_asset_cap(params.single_asset_cap, len(weights))
    if cap is None:
        active = weights > params.active_tol
        if not np.any(active):
            return float("inf")
        lambda_ = float(np.mean(grad[active]))
        active_violation = float(np.max(np.abs(grad[active] - lambda_))) if np.any(active) else 0.0
        inactive = ~active
        inactive_violation = float(np.max(np.maximum(lambda_ - grad[inactive], 0.0))) if np.any(inactive) else 0.0
        return max(active_violation, inactive_violation)

    free = (weights > params.active_tol) & (weights < cap - params.active_tol)
    if not np.any(free):
        return float("inf")
    lambda_ = float(np.mean(grad[free]))
    free_violation = float(np.max(np.abs(grad[free] - lambda_)))
    lower = weights <= params.active_tol
    upper = weights >= cap - params.active_tol
    lower_violation = float(np.max(np.maximum(lambda_ - grad[lower], 0.0))) if np.any(lower) else 0.0
    upper_violation = float(np.max(np.maximum(grad[upper] - lambda_, 0.0))) if np.any(upper) else 0.0
    return max(free_violation, lower_violation, upper_violation)


def _solution_result(
    q: np.ndarray,
    weights: np.ndarray,
    status: str,
    iterations: int,
    params: MinimumVarianceParams,
) -> dict[str, Any]:
    cap = _resolve_single_asset_cap(params.single_asset_cap, len(weights))
    weights = _project_capped_simplex(weights, cap) if cap is not None else _normalize_simplex(weights)
    weights = _enforce_min_asset_weight(weights, params, cap)
    eigvals = np.linalg.eigvalsh((q + q.T) / 2.0)
    min_eig = float(np.min(eigvals))
    max_eig = float(np.max(eigvals))
    condition_number = max_eig / min_eig if min_eig > 0 else float("inf")
    return {
        "weights": weights,
        "status": status,
        "iterations": int(iterations),
        "objective_variance": float(np.dot(weights, np.dot(q, weights))),
        "kkt_violation": _kkt_violation(q, weights, params),
        "weight_sum_error": float(abs(np.sum(weights) - 1.0)),
        "min_weight": float(np.min(weights)),
        "min_positive_weight": _min_positive_weight(weights),
        "max_weight": float(np.max(weights)),
        "single_asset_cap_violation": float(max(0.0, np.max(weights) - cap)) if cap is not None else 0.0,
        "min_asset_weight_violation": _min_asset_weight_violation(weights, params),
        "active_count": int(np.sum(weights > params.active_tol)),
        "condition_number": float(condition_number),
        "min_eig_objective": min_eig,
        "max_eig_objective": max_eig,
    }


def _load_inputs(forecast_path: Path, prvm_path: Path) -> dict[str, np.ndarray]:
    if not forecast_path.exists():
        raise FileNotFoundError(forecast_path)
    if not prvm_path.exists():
        raise FileNotFoundError(prvm_path)

    forecast = np.load(forecast_path, allow_pickle=False)
    prvm = np.load(prvm_path, allow_pickle=False)
    forecast_required = {"target_dates", "origin_dates", "tickers", "forecasts"}
    prvm_required = {"dates", "tickers", "jv"}
    missing_forecast = forecast_required.difference(forecast.files)
    missing_prvm = prvm_required.difference(prvm.files)
    if missing_forecast:
        raise ValueError(f"Forecast npz missing required arrays: {sorted(missing_forecast)}")
    if missing_prvm:
        raise ValueError(f"PRVM npz missing required arrays: {sorted(missing_prvm)}")

    tickers = np.asarray(forecast["tickers"], dtype="U")
    prvm_tickers = np.asarray(prvm["tickers"], dtype="U")
    if tickers.shape != prvm_tickers.shape or np.any(tickers != prvm_tickers):
        raise ValueError("Ticker mismatch between Phase 4 forecasts and Phase 3 PRVM results")

    forecasts = np.asarray(forecast["forecasts"], dtype=np.float64)
    jv = np.asarray(prvm["jv"], dtype=np.float64)
    if forecasts.ndim != 3 or forecasts.shape[1] != forecasts.shape[2]:
        raise ValueError(f"Expected forecasts shape (days, assets, assets), got {forecasts.shape}")
    if jv.ndim != 3 or jv.shape[1:] != forecasts.shape[1:]:
        raise ValueError(f"Expected jv shape compatible with forecasts, got {jv.shape}")
    if len(forecast["target_dates"]) != forecasts.shape[0]:
        raise ValueError("target_dates length must match forecasts day dimension")
    if not (np.isfinite(forecasts).all() and np.isfinite(jv).all()):
        raise ValueError("Input matrices must be finite")

    return {
        "target_dates": np.asarray(forecast["target_dates"], dtype="U10"),
        "origin_dates": np.asarray(forecast["origin_dates"], dtype="U10"),
        "tickers": tickers,
        "forecasts": forecasts,
        "prvm_dates": np.asarray(prvm["dates"], dtype="U10"),
        "jv": jv,
    }


def _build_objective_matrices(
    forecasts: np.ndarray,
    target_dates: np.ndarray,
    origin_dates: np.ndarray,
    prvm_dates: np.ndarray,
    jv: np.ndarray,
    params: MinimumVarianceParams,
) -> tuple[np.ndarray, np.ndarray]:
    if len(target_dates) != len(origin_dates):
        raise ValueError("target_dates and origin_dates must have the same length")
    date_to_idx = {str(date): idx for idx, date in enumerate(prvm_dates)}
    objective_matrices = np.empty_like(forecasts, dtype=np.float64)
    lagged_jv_dates: list[str] = []

    for idx, origin_date in enumerate(origin_dates):
        origin = str(origin_date)
        if origin not in date_to_idx:
            raise ValueError(f"Missing JV matrix for origin_date={origin}, target_date={target_dates[idx]}")
        lagged_jv_dates.append(origin)
        objective_matrices[idx] = project_psd(forecasts[idx] + jv[date_to_idx[origin]], floor=params.psd_floor)

    return objective_matrices, np.asarray(lagged_jv_dates, dtype="U10")


def _optimize_cycle(
    cycle_days: int,
    target_dates: np.ndarray,
    origin_dates: np.ndarray,
    lagged_jv_dates: np.ndarray,
    tickers: np.ndarray,
    objective_matrices: np.ndarray,
    params: MinimumVarianceParams,
    limit_rebalances: int | None,
) -> dict[str, Any]:
    if cycle_days < 1:
        raise ValueError("cycle_days must be >= 1")
    rebalance_indices = _rebalance_indices(target_dates, cycle_days, params.rebalance_frequency)
    if limit_rebalances is not None:
        rebalance_indices = rebalance_indices[:limit_rebalances]
    if len(rebalance_indices) == 0:
        raise ValueError(f"No rebalance dates produced for cycle={cycle_days}")

    weights = np.empty((len(rebalance_indices), len(tickers)), dtype=np.float64)
    rows: list[dict[str, Any]] = []
    previous_weights: np.ndarray | None = None
    total_rebalances = len(rebalance_indices)
    print(
        f"Phase 5 progress: cycle={cycle_days}d optimizing {total_rebalances} rebalance(s)",
        flush=True,
    )

    for out_idx, forecast_idx in enumerate(rebalance_indices):
        solved = solve_long_only_minimum_variance(objective_matrices[forecast_idx], params=params)
        weight = solved["weights"]
        weights[out_idx] = weight
        top_idx = int(np.argmax(weight))
        active_tickers = tickers[weight > params.active_tol]
        hold_end_idx = min(forecast_idx + cycle_days, len(target_dates)) - 1
        turnover = float(np.sum(np.abs(weight - previous_weights))) if previous_weights is not None else 0.0
        previous_weights = weight
        completed = out_idx + 1
        if completed == 1 or completed % 10 == 0 or completed == total_rebalances:
            pct = completed / total_rebalances * 100.0
            print(
                f"Phase 5 progress: cycle={cycle_days}d {completed}/{total_rebalances} "
                f"({pct:.1f}%), rebalance={target_dates[forecast_idx]}, "
                f"top={tickers[top_idx]} {weight[top_idx]:.2%}",
                flush=True,
            )

        rows.append(
            {
                "cycle_days": cycle_days,
                "rebalance_date": str(target_dates[forecast_idx]),
                "origin_date": str(origin_dates[forecast_idx]),
                "lagged_jv_date": str(lagged_jv_dates[forecast_idx]),
                "hold_start_date": str(target_dates[forecast_idx]),
                "hold_end_date": str(target_dates[hold_end_idx]),
                "active_count": int(solved["active_count"]),
                "active_tickers": "|".join(map(str, active_tickers)),
                "top_ticker": str(tickers[top_idx]),
                "top_weight": float(weight[top_idx]),
                "max_weight": float(solved["max_weight"]),
                "min_weight": float(solved["min_weight"]),
                "min_positive_weight": float(solved["min_positive_weight"]),
                "objective_variance": float(solved["objective_variance"]),
                "condition_number_objective": float(solved["condition_number"]),
                "min_eig_objective": float(solved["min_eig_objective"]),
                "max_eig_objective": float(solved["max_eig_objective"]),
                "turnover_from_prev": turnover,
                "solver_status": str(solved["status"]),
                "solver_iterations": int(solved["iterations"]),
                "kkt_violation": float(solved["kkt_violation"]),
                "weight_sum_error": float(solved["weight_sum_error"]),
                "single_asset_cap_violation": float(solved["single_asset_cap_violation"]),
                "min_asset_weight_violation": float(solved["min_asset_weight_violation"]),
            }
        )

    scheduled_dates, scheduled_weights, scheduled_rebalance_dates = _expand_scheduled_weights(
        cycle_days=cycle_days,
        target_dates=target_dates,
        rebalance_indices=rebalance_indices,
        weights=weights,
    )
    return {
        "cycle_days": cycle_days,
        "rebalance_indices": rebalance_indices,
        "rebalance_dates": target_dates[rebalance_indices],
        "weights": weights,
        "summary": pd.DataFrame(rows),
        "scheduled_dates": scheduled_dates,
        "scheduled_weights": scheduled_weights,
        "scheduled_rebalance_dates": scheduled_rebalance_dates,
    }


def _rebalance_indices(target_dates: np.ndarray, cycle_days: int, rebalance_frequency: str) -> np.ndarray:
    if rebalance_frequency == "cycle":
        return np.arange(0, len(target_dates), cycle_days, dtype=int)
    if rebalance_frequency == "monthly":
        months = pd.Series(pd.to_datetime(target_dates).to_period("M").astype(str))
        first_indices = months.drop_duplicates().index.to_numpy(dtype=int)
        return first_indices
    raise ValueError(f"Unsupported rebalance_frequency={rebalance_frequency}")


def _expand_scheduled_weights(
    cycle_days: int,
    target_dates: np.ndarray,
    rebalance_indices: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scheduled_parts: list[np.ndarray] = []
    date_parts: list[np.ndarray] = []
    rebalance_date_parts: list[np.ndarray] = []

    for idx, start_idx in enumerate(rebalance_indices):
        end_idx = min(start_idx + cycle_days, len(target_dates))
        block_dates = target_dates[start_idx:end_idx]
        date_parts.append(block_dates)
        scheduled_parts.append(np.repeat(weights[idx : idx + 1], len(block_dates), axis=0))
        rebalance_date_parts.append(np.repeat(target_dates[start_idx], len(block_dates)))

    return (
        np.concatenate(date_parts).astype("U10"),
        np.vstack(scheduled_parts).astype(np.float64),
        np.concatenate(rebalance_date_parts).astype("U10"),
    )


def _weights_long_frame(cycle_results: dict[int, dict[str, Any]], tickers: np.ndarray) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for cycle, result in cycle_results.items():
        repeated_dates = np.repeat(result["rebalance_dates"], len(tickers))
        frames.append(
            pd.DataFrame(
                {
                    "cycle_days": cycle,
                    "rebalance_date": repeated_dates,
                    "ticker": np.tile(tickers, len(result["rebalance_dates"])),
                    "weight": result["weights"].reshape(-1),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _scheduled_weights_long_frame(cycle_results: dict[int, dict[str, Any]], tickers: np.ndarray) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for cycle, result in cycle_results.items():
        frames.append(
            pd.DataFrame(
                {
                    "cycle_days": cycle,
                    "date": np.repeat(result["scheduled_dates"], len(tickers)),
                    "rebalance_date": np.repeat(result["scheduled_rebalance_dates"], len(tickers)),
                    "ticker": np.tile(tickers, len(result["scheduled_dates"])),
                    "weight": result["scheduled_weights"].reshape(-1),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _weights_wide_frame(cycle_results: dict[int, dict[str, Any]], tickers: np.ndarray) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for cycle, result in cycle_results.items():
        frame = pd.DataFrame(result["weights"], columns=tickers)
        frame.insert(0, "rebalance_date", result["rebalance_dates"])
        frame.insert(0, "cycle_days", cycle)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _cycle_summary(summary_df: pd.DataFrame) -> dict[str, Any]:
    status_counts = summary_df["solver_status"].value_counts().to_dict()
    return {
        "n_rebalances": int(len(summary_df)),
        "first_rebalance_date": str(summary_df["rebalance_date"].iloc[0]),
        "last_rebalance_date": str(summary_df["rebalance_date"].iloc[-1]),
        "active_count_mean": float(summary_df["active_count"].mean()),
        "active_count_min": int(summary_df["active_count"].min()),
        "active_count_max": int(summary_df["active_count"].max()),
        "top_weight_mean": float(summary_df["top_weight"].mean()),
        "top_weight_max": float(summary_df["top_weight"].max()),
        "turnover_mean": float(summary_df["turnover_from_prev"].iloc[1:].mean())
        if len(summary_df) > 1
        else 0.0,
        "kkt_violation_max": float(summary_df["kkt_violation"].max()),
        "weight_sum_error_max": float(summary_df["weight_sum_error"].max()),
        "single_asset_cap_violation_max": float(summary_df["single_asset_cap_violation"].max()),
        "min_weight_min": float(summary_df["min_weight"].min()),
        "min_positive_weight_min": float(summary_df["min_positive_weight"].min()),
        "min_asset_weight_violation_max": float(summary_df["min_asset_weight_violation"].max()),
        "solver_status_counts": {str(k): int(v) for k, v in status_counts.items()},
    }


def _product_interpretation(
    cycle_summary: dict[str, dict[str, Any]],
    single_asset_cap: float | None,
    min_asset_weight: float,
) -> dict[str, Any]:
    top_weight_max = max(summary["top_weight_max"] for summary in cycle_summary.values())
    top_weight_mean_max = max(summary["top_weight_mean"] for summary in cycle_summary.values())
    if single_asset_cap is None:
        concentration_note = (
            "The unconstrained long-only minimum variance setup can concentrate heavily in the asset "
            "with the lowest forecast variance/correlation contribution. This is expected behavior, but "
            "it should be disclosed and tested against candidate single-asset caps."
        )
        next_steps = [
            "Complete Phase 6 backtest against equal-weight on the same 50-asset universe.",
            "Evaluate drawdown, turnover, active asset count, and top-weight time series.",
            "Test single-asset caps such as 20% or 30% if concentration materially worsens realized risk.",
            "Use research wording such as monthly virtual asset model portfolio rather than public fund wording.",
        ]
    else:
        concentration_note = (
            f"A {single_asset_cap:.0%} single-asset cap is active. This directly limits concentration, "
            "but the capped strategy still needs Phase 6 comparison against equal-weight and the uncapped variant."
        )
        next_steps = [
            "Complete Phase 6 backtest against equal-weight on the same 50-asset universe.",
            "Compare capped versus uncapped minimum variance on return, volatility, drawdown, and turnover.",
            "Use BTC HODL as market context rather than the primary benchmark.",
            "Use research wording such as monthly virtual asset model portfolio rather than public fund wording.",
        ]
    return {
        "positioning": "monthly_virtual_asset_model_portfolio_after_phase6_validation",
        "not_ready_as_public_fund_product": True,
        "reason": (
            "Phase 5 only produces optimized weights. Backtested returns, drawdowns, benchmark comparison, "
            "turnover impact, and execution assumptions must be evaluated in Phase 6 before any product-like framing."
        ),
        "concentration_risk": {
            "top_weight_mean_max_across_cycles": float(top_weight_mean_max),
            "top_weight_max_across_cycles": float(top_weight_max),
            "requires_phase6_cap_review": bool(top_weight_max >= 0.5),
            "note": concentration_note,
        },
        "candidate_next_steps": next_steps,
        "minimum_position_size": {
            "min_asset_weight": float(min_asset_weight),
            "note": (
                f"Positive portfolio weights are pruned/projected to stay at or above {min_asset_weight:.3%}; "
                "smaller numerical positions are set to zero."
            )
            if min_asset_weight > 0
            else "No minimum positive position size is applied.",
        },
    }


def _write_outputs(
    output_dir: Path,
    forecast_path: Path,
    prvm_path: Path,
    params: MinimumVarianceParams,
    tickers: np.ndarray,
    n_forecast_days: int,
    cycle_results: dict[int, dict[str, Any]],
    limit_rebalances: int | None,
) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    npz_path = output_dir / "minimum_variance_portfolios.npz"
    weights_long_path = output_dir / "minimum_variance_weights_long.csv"
    weights_wide_path = output_dir / "minimum_variance_weights_wide.csv"
    scheduled_weights_long_path = output_dir / "minimum_variance_scheduled_weights_long.csv"
    summary_path = output_dir / "phase5_portfolio_summary.csv"
    report_path = output_dir / "phase5_portfolio_report.json"

    weights_long = _weights_long_frame(cycle_results, tickers)
    weights_wide = _weights_wide_frame(cycle_results, tickers)
    scheduled_weights_long = _scheduled_weights_long_frame(cycle_results, tickers)
    summary_df = pd.concat([result["summary"] for result in cycle_results.values()], ignore_index=True)

    weights_long.to_csv(weights_long_path, index=False)
    weights_wide.to_csv(weights_wide_path, index=False)
    scheduled_weights_long.to_csv(scheduled_weights_long_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    npz_payload: dict[str, np.ndarray] = {"tickers": tickers.astype("U")}
    for cycle, result in cycle_results.items():
        prefix = f"cycle_{cycle}"
        npz_payload[f"{prefix}_rebalance_dates"] = result["rebalance_dates"].astype("U10")
        npz_payload[f"{prefix}_weights"] = result["weights"].astype(np.float64)
        npz_payload[f"{prefix}_scheduled_dates"] = result["scheduled_dates"].astype("U10")
        npz_payload[f"{prefix}_scheduled_rebalance_dates"] = result["scheduled_rebalance_dates"].astype("U10")
        npz_payload[f"{prefix}_scheduled_weights"] = result["scheduled_weights"].astype(np.float64)
    np.savez_compressed(npz_path, **npz_payload)

    cycle_summary = {str(cycle): _cycle_summary(result["summary"]) for cycle, result in cycle_results.items()}
    product_interpretation = _product_interpretation(cycle_summary, params.single_asset_cap, params.min_asset_weight)
    report = {
        "forecast_path": repo_relative_path(forecast_path, ROOT),
        "prvm_path": repo_relative_path(prvm_path, ROOT),
        "output_dir": repo_relative_path(output_dir, ROOT),
        "params": params.to_dict(),
        "limit_rebalances": limit_rebalances,
        "n_assets": int(len(tickers)),
        "n_forecast_days": int(n_forecast_days),
        "cycle_summary": cycle_summary,
        "product_interpretation": product_interpretation,
        "sanity": {
            "all_weight_sums_close_to_one": bool(summary_df["weight_sum_error"].max() <= 1e-10),
            "all_weights_long_only": bool(summary_df["min_weight"].min() >= -1e-12),
            "all_weights_within_single_asset_cap": bool(summary_df["single_asset_cap_violation"].max() <= 1e-10),
            "all_positive_weights_at_or_above_min_asset_weight": bool(
                summary_df["min_asset_weight_violation"].max() <= 1e-12
            ),
            "min_positive_weight": float(summary_df["min_positive_weight"].min()),
            "max_kkt_violation": float(summary_df["kkt_violation"].max()),
        },
        "interpretation_note": (
            "Scheduled weights repeat each rebalance weight across its hold window for date alignment. "
            "They are not drifted realized portfolio weights; Phase 6 should compute hold-period drift from returns."
        ),
        "outputs": {
            "npz": repo_relative_path(npz_path, ROOT),
            "weights_long_csv": repo_relative_path(weights_long_path, ROOT),
            "weights_wide_csv": repo_relative_path(weights_wide_path, ROOT),
            "scheduled_weights_long_csv": repo_relative_path(scheduled_weights_long_path, ROOT),
            "summary_csv": repo_relative_path(summary_path, ROOT),
            "report_json": repo_relative_path(report_path, ROOT),
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_phase5_markdown_report(output_dir / "phase5_results_note.md", report)
    return report


def _write_phase5_markdown_report(path: Path, report: dict[str, Any]) -> None:
    cycle_7 = report["cycle_summary"].get("7", {})
    cycle_14 = report["cycle_summary"].get("14", {})
    interp = report["product_interpretation"]
    single_asset_cap = report["params"]["single_asset_cap"]
    min_asset_weight = report["params"]["min_asset_weight"]
    if single_asset_cap is None:
        concentration_lines = [
            "- Concentration risk is material: top weights can exceed 90% in the unconstrained long-only setup.",
            "- This behavior is consistent with minimum variance optimization, but it creates a clear Phase 6 cap-review item.",
        ]
    else:
        concentration_lines = [
            f"- A single-asset cap of {single_asset_cap:.0%} is active; top weights are mechanically bounded at that level.",
            "- This capped variant should be compared against the uncapped strategy and equal-weight in Phase 6.",
        ]
    lines = [
        "# Phase 5 Results Note",
        "",
        "## Optimization Output",
        "",
        f"- Rebalance frequency: {report['params']['rebalance_frequency']}",
        "",
        "| Cycle | Rebalances | Active Mean | Active Min-Max | Min Positive | Top Weight Mean | Top Weight Max | Turnover Mean |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        _phase5_cycle_markdown_row(7, cycle_7),
        _phase5_cycle_markdown_row(14, cycle_14),
        "",
        "## Interpretation",
        "",
        "- Phase 5 is a weight-generation step, not a completed investment product validation.",
        "- The strategy is better framed as a monthly virtual asset model portfolio after Phase 6 backtesting.",
    ]
    if min_asset_weight > 0:
        lines.append(
            f"- Positive weights at or below {min_asset_weight:.2%} are pruned to zero before artifacts are written."
        )
    lines.extend(concentration_lines)
    lines.extend([
        "",
        "## Phase 6 Implications",
        "",
    ])
    lines.extend(f"- {step}" for step in interp["candidate_next_steps"])
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _phase5_cycle_markdown_row(cycle: int, summary: dict[str, Any]) -> str:
    if not summary:
        return f"| {cycle} | n/a | n/a | n/a | n/a | n/a | n/a | n/a |"
    active_range = f"{summary['active_count_min']}-{summary['active_count_max']}"
    return (
        f"| {cycle} | {summary['n_rebalances']} | {summary['active_count_mean']:.2f} | "
        f"{active_range} | {summary['min_positive_weight_min']:.4f} | "
        f"{summary['top_weight_mean']:.4f} | {summary['top_weight_max']:.4f} | "
        f"{summary['turnover_mean']:.4f} |"
    )
