#!/usr/bin/env python3
"""Run Phase 5 long-only minimum variance portfolio optimization."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.phase5_portfolio.minimum_variance_optimizer import compute_phase5_portfolios  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate Phase 5 minimum variance portfolio weights")
    parser.add_argument(
        "--forecast-input",
        type=Path,
        default=ROOT / "data" / "processed" / "phase4" / "ewma_forecasts.npz",
        help="Input Phase 4 EWMA forecast npz.",
    )
    parser.add_argument(
        "--prvm-input",
        type=Path,
        default=ROOT / "data" / "processed" / "phase3" / "prvm_results.npz",
        help="Input Phase 3 PRVM/JV npz.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "phase5",
        help="Output directory for Phase 5 artifacts.",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        nargs="+",
        default=config.CYCLES,
        help=f"Rebalance cycles in days. Default: {config.CYCLES}",
    )
    parser.add_argument(
        "--limit-rebalances",
        type=int,
        default=None,
        help="Optimize only the first N rebalances per cycle. Useful for smoke tests.",
    )
    parser.add_argument(
        "--single-asset-cap",
        type=float,
        default=None,
        help="Optional single-asset weight cap, e.g. 0.25 for 25%. Default: no cap.",
    )
    parser.add_argument(
        "--min-asset-weight",
        type=float,
        default=None,
        help=f"Minimum positive asset weight after pruning. Default: {config.MIN_ASSET_WEIGHT}",
    )
    parser.add_argument(
        "--rebalance-frequency",
        choices=["cycle", "monthly"],
        default="cycle",
        help="Use rolling cycle rebalances or first available date of each month. Default: cycle.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    compute_phase5_portfolios(
        forecast_path=args.forecast_input,
        prvm_path=args.prvm_input,
        output_dir=args.output_dir,
        cycles=args.cycles,
        limit_rebalances=args.limit_rebalances,
        single_asset_cap=args.single_asset_cap,
        min_asset_weight=args.min_asset_weight,
        rebalance_frequency=args.rebalance_frequency,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
