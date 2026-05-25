#!/usr/bin/env python3
"""Run Phase 6 walk-forward backtest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.phase6_backtest.walk_forward import run_phase6_backtest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 6 backtest from price panel and Phase 5 weights")
    parser.add_argument(
        "--price-panel",
        type=Path,
        default=ROOT / "price_panel.csv",
        help="Wide price panel CSV from Phase 1. Default: price_panel.csv",
    )
    parser.add_argument(
        "--portfolio-input",
        type=Path,
        default=ROOT / "data" / "processed" / "phase5" / "minimum_variance_portfolios.npz",
        help="Phase 5 minimum variance portfolio npz.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "phase6",
        help="Output directory for Phase 6 artifacts.",
    )
    parser.add_argument(
        "--eval-freq-min",
        type=int,
        default=config.EVAL_FREQ_MIN,
        help=f"Evaluation interval in minutes. Default: {config.EVAL_FREQ_MIN}",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_phase6_backtest(
        price_panel_path=args.price_panel,
        portfolio_path=args.portfolio_input,
        output_dir=args.output_dir,
        eval_freq_min=args.eval_freq_min,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
