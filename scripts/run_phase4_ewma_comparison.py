#!/usr/bin/env python3
"""Compare jump-adjusted EWMA against raw-PRVM EWMA on a common target."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.phase4_ewma.matrix_ewma import EWMAParams, compute_ewma_forecasts  # noqa: E402
from src.utils import ensure_dir  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Phase 4 EWMA forecasts from jump-adjusted PRVM and raw PRVM"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "processed" / "phase3" / "prvm_results.npz",
        help="Input Phase 3 prvm_results.npz.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "phase4_comparison",
        help="Output directory for comparison artifacts.",
    )
    parser.add_argument(
        "--lambda",
        dest="lambda_",
        type=float,
        default=config.LAMBDA_,
        help=f"EWMA lambda. Default: {config.LAMBDA_}",
    )
    parser.add_argument(
        "--init-days",
        type=int,
        default=config.EWMA_INIT_DAYS,
        help=f"Number of initial PRVM days to average. Default: {config.EWMA_INIT_DAYS}",
    )
    parser.add_argument(
        "--analysis-start",
        default=config.ANALYSIS_START,
        help=f"First forecast target date. Default: {config.ANALYSIS_START}",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_comparison(
        input_path=args.input,
        output_dir=args.output_dir,
        lambda_=args.lambda_,
        init_days=args.init_days,
        analysis_start=args.analysis_start,
    )
    print("=== Phase 4 EWMA jump comparison ===")
    print(f"input: {args.input}")
    print(f"output_dir: {args.output_dir}")
    print(f"target: jump-adjusted prvm")
    for row in report["summary"]:
        print(
            f"{row['model']}: MSPE x1e4={row['mean_mspe_x1e4']:.6g}, "
            f"QLIKE x1e-3={row['mean_qlike_x1e_minus3']:.6g}"
        )
    print(f"MSPE delta raw-adjusted x1e4: {report['comparison']['mspe_delta_raw_minus_adjusted_x1e4']:.6g}")
    print(f"QLIKE delta raw-adjusted x1e-3: {report['comparison']['qlike_delta_raw_minus_adjusted_x1e_minus3']:.6g}")
    print(f"saved summary: {report['outputs']['summary_csv']}")
    print(f"saved report: {report['outputs']['report_json']}")
    return 0


def run_comparison(
    input_path: Path,
    output_dir: Path,
    lambda_: float,
    init_days: int,
    analysis_start: str,
) -> dict:
    input_path = Path(input_path)
    output_dir = ensure_dir(output_dir)
    data = np.load(input_path, allow_pickle=False)
    required = {"dates", "tickers", "prvm", "raw_prvm"}
    missing = required.difference(data.files)
    if missing:
        raise ValueError(f"Phase 3 npz is missing required arrays: {sorted(missing)}")

    params = EWMAParams(lambda_=lambda_, init_days=init_days, analysis_start=analysis_start)
    adjusted = compute_ewma_forecasts(
        dates=data["dates"],
        tickers=data["tickers"],
        input_matrices=data["prvm"],
        ground_truth_matrices=data["prvm"],
        params=params,
        input_matrix_key="prvm",
        ground_truth_matrix_key="prvm",
    )
    raw = compute_ewma_forecasts(
        dates=data["dates"],
        tickers=data["tickers"],
        input_matrices=data["raw_prvm"],
        ground_truth_matrices=data["prvm"],
        params=params,
        input_matrix_key="raw_prvm",
        ground_truth_matrix_key="prvm",
    )

    summary_rows = [_summary_row("jump_adjusted_prvm_ewma", adjusted), _summary_row("raw_prvm_ewma", raw)]
    summary_df = pd.DataFrame(summary_rows)
    daily_df = pd.DataFrame(
        {
            "target_date": adjusted["target_dates"],
            "origin_date": adjusted["origin_dates"],
            "adjusted_mspe": adjusted["mspe"],
            "raw_mspe": raw["mspe"],
            "mspe_delta_raw_minus_adjusted": raw["mspe"] - adjusted["mspe"],
            "adjusted_qlike": adjusted["qlike"],
            "raw_qlike": raw["qlike"],
            "qlike_delta_raw_minus_adjusted": raw["qlike"] - adjusted["qlike"],
        }
    )

    summary_path = output_dir / "phase4_ewma_comparison_summary.csv"
    daily_path = output_dir / "phase4_ewma_comparison_daily.csv"
    npz_path = output_dir / "phase4_ewma_comparison.npz"
    report_path = output_dir / "phase4_ewma_comparison_report.json"
    summary_df.to_csv(summary_path, index=False)
    daily_df.to_csv(daily_path, index=False)
    np.savez_compressed(
        npz_path,
        target_dates=adjusted["target_dates"],
        origin_dates=adjusted["origin_dates"],
        tickers=adjusted["tickers"],
        forecasts_adjusted=adjusted["forecasts"],
        forecasts_raw=raw["forecasts"],
        ground_truth_prvm=adjusted["ground_truth_prvm"],
        adjusted_mspe=adjusted["mspe"],
        raw_mspe=raw["mspe"],
        adjusted_qlike=adjusted["qlike"],
        raw_qlike=raw["qlike"],
        **{"lambda": np.asarray(lambda_, dtype=np.float64)},
    )

    comparison = {
        "mspe_delta_raw_minus_adjusted": float(np.mean(raw["mspe"]) - np.mean(adjusted["mspe"])),
        "mspe_delta_raw_minus_adjusted_x1e4": float((np.mean(raw["mspe"]) - np.mean(adjusted["mspe"])) * 1e4),
        "qlike_delta_raw_minus_adjusted": float(np.mean(raw["qlike"]) - np.mean(adjusted["qlike"])),
        "qlike_delta_raw_minus_adjusted_x1e_minus3": float(
            (np.mean(raw["qlike"]) - np.mean(adjusted["qlike"])) * 1e-3
        ),
        "days_adjusted_lower_mspe": int(np.sum(adjusted["mspe"] < raw["mspe"])),
        "days_adjusted_lower_qlike": int(np.sum(adjusted["qlike"] < raw["qlike"])),
        "n_forecast_days": int(len(adjusted["target_dates"])),
    }
    report = {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "target_matrix_key": "prvm",
        "params": params.to_dict(),
        "summary": summary_rows,
        "comparison": comparison,
        "outputs": {
            "summary_csv": str(summary_path),
            "daily_csv": str(daily_path),
            "npz": str(npz_path),
            "report_json": str(report_path),
        },
        "interpretation_note": (
            "Both models are evaluated against the same next-day jump-adjusted PRVM target. "
            "Positive raw-minus-adjusted deltas mean the jump-adjusted EWMA has lower loss."
        ),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _summary_row(model: str, result: dict) -> dict:
    return {
        "model": model,
        "input_matrix_key": result["input_matrix_key"],
        "ground_truth_matrix_key": result["ground_truth_matrix_key"],
        "n_forecast_days": int(len(result["target_dates"])),
        "first_target_date": str(result["target_dates"][0]),
        "last_target_date": str(result["target_dates"][-1]),
        "mean_mspe": float(np.mean(result["mspe"])),
        "mean_mspe_x1e4": float(np.mean(result["mspe"]) * 1e4),
        "mean_qlike": float(np.mean(result["qlike"])),
        "mean_qlike_x1e_minus3": float(np.mean(result["qlike"]) * 1e-3),
    }


if __name__ == "__main__":
    raise SystemExit(main())
