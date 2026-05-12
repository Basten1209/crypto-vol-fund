"""Recursive seeded matrix EWMA forecast for Phase 4.

This module consumes the compact Phase 3 ``prvm_results.npz`` artifact. It does
not read the large 1-minute price panel.
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
from src.phase4_ewma.forecast_evaluator import calculate_daily_mspe, calculate_daily_qlike  # noqa: E402
from src.utils import ensure_dir, project_psd  # noqa: E402


@dataclass(frozen=True)
class EWMAParams:
    lambda_: float = config.LAMBDA_
    init_days: int = config.EWMA_INIT_DAYS
    analysis_start: str = config.ANALYSIS_START
    psd_floor: float = config.PSD_FLOOR

    def to_dict(self) -> dict[str, Any]:
        return {
            "lambda": self.lambda_,
            "init_days": self.init_days,
            "analysis_start": self.analysis_start,
            "psd_floor": self.psd_floor,
            "method": "recursive_seeded_riskmetrics_ewma",
        }


def compute_ewma_forecasts(
    dates: np.ndarray,
    tickers: np.ndarray,
    input_matrices: np.ndarray,
    ground_truth_matrices: np.ndarray | None = None,
    params: EWMAParams | None = None,
    input_matrix_key: str = "prvm",
    ground_truth_matrix_key: str = "prvm",
) -> dict[str, Any]:
    """Compute daily EWMA forecasts from Phase 3 realized-volatility arrays."""
    params = params or EWMAParams()
    ground_truth_matrices = input_matrices if ground_truth_matrices is None else ground_truth_matrices
    _validate_inputs(dates, tickers, input_matrices, params, label=input_matrix_key)
    _validate_inputs(dates, tickers, ground_truth_matrices, params, label=ground_truth_matrix_key)

    date_strings = np.asarray(dates, dtype="U10")
    target_start_idx = int(np.where(date_strings == params.analysis_start)[0][0])
    init_start_idx = target_start_idx - params.init_days
    init_slice = slice(init_start_idx, target_start_idx)

    init_dates = date_strings[init_slice]
    target_dates = date_strings[target_start_idx:]
    origin_dates = date_strings[target_start_idx - 1 : -1]
    ground_truth = np.asarray(ground_truth_matrices[target_start_idx:], dtype=np.float64)

    current_forecast = project_psd(np.mean(input_matrices[init_slice], axis=0), floor=params.psd_floor)
    forecasts = np.empty_like(ground_truth, dtype=np.float64)

    for out_idx, source_idx in enumerate(range(target_start_idx, len(date_strings))):
        forecasts[out_idx] = current_forecast
        current_forecast = project_psd(
            (1.0 - params.lambda_) * input_matrices[source_idx] + params.lambda_ * current_forecast,
            floor=params.psd_floor,
        )

    mspe = calculate_daily_mspe(forecasts, ground_truth)
    qlike = calculate_daily_qlike(forecasts, ground_truth, floor=params.psd_floor)

    return {
        "dates": date_strings,
        "target_dates": target_dates,
        "origin_dates": origin_dates,
        "init_dates": init_dates,
        "tickers": np.asarray(tickers, dtype="U"),
        "forecasts": forecasts,
        "ground_truth_matrix": ground_truth,
        "ground_truth_prvm": ground_truth,
        "mspe": mspe,
        "qlike": qlike,
        "params": params,
        "input_matrix_key": input_matrix_key,
        "ground_truth_matrix_key": ground_truth_matrix_key,
    }


def compute_phase4_ewma(
    input_path: Path | str,
    output_dir: Path | str,
    lambda_: float | None = None,
    init_days: int | None = None,
    analysis_start: str | None = None,
    matrix_key: str = "prvm",
    ground_truth_key: str | None = None,
) -> dict[str, Any]:
    """Run Phase 4 EWMA forecasting and write artifacts to disk."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    params = EWMAParams(
        lambda_=config.LAMBDA_ if lambda_ is None else float(lambda_),
        init_days=config.EWMA_INIT_DAYS if init_days is None else int(init_days),
        analysis_start=config.ANALYSIS_START if analysis_start is None else str(analysis_start),
        psd_floor=config.PSD_FLOOR,
    )

    ground_truth_key = matrix_key if ground_truth_key is None else ground_truth_key
    loaded = _load_phase3_npz(input_path, required_keys=("dates", "tickers", matrix_key, ground_truth_key))
    result = compute_ewma_forecasts(
        dates=loaded["dates"],
        tickers=loaded["tickers"],
        input_matrices=loaded[matrix_key],
        ground_truth_matrices=loaded[ground_truth_key],
        params=params,
        input_matrix_key=matrix_key,
        ground_truth_matrix_key=ground_truth_key,
    )
    report = _write_outputs(output_dir=output_dir, input_path=input_path, result=result)

    print("=== Phase 4 EWMA forecast ===")
    print(f"input: {input_path}")
    print(f"output_dir: {output_dir}")
    print(f"method: {params.to_dict()['method']}")
    print(f"input_matrix: {matrix_key}")
    print(f"ground_truth_matrix: {ground_truth_key}")
    print(f"lambda: {params.lambda_}")
    print(f"init_dates: {result['init_dates'][0]} to {result['init_dates'][-1]}")
    print(f"target_dates: {result['target_dates'][0]} to {result['target_dates'][-1]}")
    print(f"forecast_shape: {tuple(result['forecasts'].shape)}")
    print(f"mean_mspe_x1e4: {report['metrics_summary']['mean_mspe_x1e4']:.6g}")
    print(f"mean_qlike_x1e_minus3: {report['metrics_summary']['mean_qlike_x1e_minus3']:.6g}")
    print(f"saved npz: {report['outputs']['npz']}")
    print(f"saved report: {report['outputs']['report_json']}")
    return report


def _load_phase3_npz(input_path: Path, required_keys: tuple[str, ...] = ("dates", "tickers", "prvm")) -> dict[str, np.ndarray]:
    data = np.load(input_path, allow_pickle=False)
    missing = set(required_keys).difference(data.files)
    if missing:
        raise ValueError(f"Phase 3 npz is missing required arrays: {sorted(missing)}")
    return {key: data[key] for key in required_keys}


def _validate_inputs(
    dates: np.ndarray,
    tickers: np.ndarray,
    matrices: np.ndarray,
    params: EWMAParams,
    label: str,
) -> None:
    if not (0.0 < params.lambda_ < 1.0):
        raise ValueError("lambda must be between 0 and 1")
    if params.init_days < 1:
        raise ValueError("init_days must be >= 1")
    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
        raise ValueError(f"Expected {label} shape (days, assets, assets), got {matrices.shape}")
    if len(dates) != matrices.shape[0]:
        raise ValueError(f"dates length must match {label} day dimension")
    if len(tickers) != matrices.shape[1]:
        raise ValueError(f"tickers length must match {label} asset dimension")
    if not np.isfinite(matrices).all():
        raise ValueError(f"{label} array contains NaN or Inf")

    date_strings = np.asarray(dates, dtype="U10")
    matches = np.where(date_strings == params.analysis_start)[0]
    if len(matches) != 1:
        raise ValueError(f"analysis_start={params.analysis_start} must appear exactly once in dates")
    target_start_idx = int(matches[0])
    if target_start_idx < params.init_days:
        raise ValueError(
            f"Need {params.init_days} initialization days before {params.analysis_start}, "
            f"got {target_start_idx}"
        )


def _write_outputs(output_dir: Path, input_path: Path, result: dict[str, Any]) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    npz_path = output_dir / "ewma_forecasts.npz"
    metrics_path = output_dir / "ewma_metrics.csv"
    report_path = output_dir / "phase4_ewma_report.json"
    params: EWMAParams = result["params"]

    np.savez_compressed(
        npz_path,
        dates=result["dates"],
        target_dates=result["target_dates"],
        origin_dates=result["origin_dates"],
        tickers=result["tickers"],
        forecasts=result["forecasts"],
        ground_truth_matrix=result["ground_truth_matrix"],
        ground_truth_prvm=result["ground_truth_prvm"],
        init_dates=result["init_dates"],
        mspe=result["mspe"],
        qlike=result["qlike"],
        **{"lambda": np.asarray(params.lambda_, dtype=np.float64)},
    )

    metrics_df = pd.DataFrame(
        {
            "target_date": result["target_dates"],
            "origin_date": result["origin_dates"],
            "mspe": result["mspe"],
            "mspe_x1e4": result["mspe"] * 1e4,
            "qlike": result["qlike"],
            "qlike_x1e_minus3": result["qlike"] * 1e-3,
        }
    )
    metrics_df.to_csv(metrics_path, index=False)

    forecasts = result["forecasts"]
    symmetry_error = float(np.max(np.abs(forecasts - np.swapaxes(forecasts, 1, 2))))
    eigvals = np.linalg.eigvalsh((forecasts + np.swapaxes(forecasts, 1, 2)) / 2.0)
    sanity = {
        "all_finite": bool(np.isfinite(forecasts).all() and np.isfinite(result["ground_truth_prvm"]).all()),
        "symmetry_error_forecast_max": symmetry_error,
        "min_eig_forecast_min": float(np.min(eigvals)),
        "forecast_shape": list(forecasts.shape),
        "metrics_all_finite": bool(np.isfinite(result["mspe"]).all() and np.isfinite(result["qlike"]).all()),
    }
    metrics_summary = {
        "mean_mspe": float(np.mean(result["mspe"])),
        "mean_mspe_x1e4": float(np.mean(result["mspe"]) * 1e4),
        "mean_qlike": float(np.mean(result["qlike"])),
        "mean_qlike_x1e_minus3": float(np.mean(result["qlike"]) * 1e-3),
    }
    report = {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "params": params.to_dict(),
        "input_matrix_key": result["input_matrix_key"],
        "ground_truth_matrix_key": result["ground_truth_matrix_key"],
        "n_assets": int(len(result["tickers"])),
        "n_phase3_days": int(len(result["dates"])),
        "n_forecast_days": int(len(result["target_dates"])),
        "first_init_date": str(result["init_dates"][0]),
        "last_init_date": str(result["init_dates"][-1]),
        "first_target_date": str(result["target_dates"][0]),
        "last_target_date": str(result["target_dates"][-1]),
        "sanity": sanity,
        "metrics_summary": metrics_summary,
        "outputs": {
            "npz": str(npz_path),
            "metrics_csv": str(metrics_path),
            "report_json": str(report_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
