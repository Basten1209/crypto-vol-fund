#!/usr/bin/env python3
"""Run Phase 4 recursive seeded EWMA forecasts from Phase 3 PRVM output."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.phase4_ewma.matrix_ewma import compute_phase4_ewma  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate Phase 4 matrix EWMA forecasts")
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "processed" / "phase3" / "prvm_results.npz",
        help="Input Phase 3 prvm_results.npz.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "phase4",
        help="Output directory for Phase 4 artifacts.",
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
    parser.add_argument(
        "--matrix-key",
        default="prvm",
        help="Phase 3 matrix key used as EWMA input. Default: prvm",
    )
    parser.add_argument(
        "--ground-truth-key",
        default=None,
        help="Phase 3 matrix key used as evaluation target. Default: same as --matrix-key",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    compute_phase4_ewma(
        input_path=args.input,
        output_dir=args.output_dir,
        lambda_=args.lambda_,
        init_days=args.init_days,
        analysis_start=args.analysis_start,
        matrix_key=args.matrix_key,
        ground_truth_key=args.ground_truth_key,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
