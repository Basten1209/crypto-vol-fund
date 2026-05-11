#!/usr/bin/env python3
"""Run Phase 3 PRVM calculation from price_panel.csv."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.phase3_prvm.prvm_calculator import compute_phase3_prvm  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate daily PRVM/JV matrices from price_panel.csv")
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "price_panel.csv",
        help="Input wide price panel CSV. Default: price_panel.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "phase3",
        help="Output directory for Phase 3 artifacts.",
    )
    parser.add_argument(
        "--workers",
        default="auto",
        help="Number of worker processes, or 'auto'. Default: auto",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="CSV rows per chunk. Default: 100000",
    )
    parser.add_argument(
        "--limit-days",
        type=int,
        default=None,
        help="Compute only the first N full trading days. Useful for smoke tests.",
    )
    parser.add_argument(
        "--no-long-csv",
        action="store_true",
        help="Skip writing prvm_long.csv and jv_long.csv.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    compute_phase3_prvm(
        input_path=args.input,
        output_dir=args.output_dir,
        workers=args.workers,
        chunk_size=args.chunk_size,
        limit_days=args.limit_days,
        write_long_csv=not args.no_long_csv,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

