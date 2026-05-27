#!/usr/bin/env python3
"""Generate report-ready Phase 2 EDA tables and figures.

This script is intentionally separate from the interactive Phase 2 Plotly
reports. It creates static PNG figures that can be inserted into the term paper
draft and a compact CSV summary for appendix tables.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRICE_PANEL = ROOT / "price_panel.csv"
DEFAULT_PROCESSED_DIR = ROOT / "data" / "processed" / "phase2_eda"
DEFAULT_FIGURE_DIR = ROOT / "docs" / "phase2_eda" / "figures"
DEFAULT_PREVIEW = ROOT / "docs" / "phase2_eda" / "report_eda_preview.html"
REPRESENTATIVE_TICKERS = ["BTC", "ETH", "XRP", "SOL"]
HIGHLIGHT_TICKERS = set(REPRESENTATIVE_TICKERS)
T5_ORDINARY_KURTOSIS = 9.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate report-ready Phase 2 EDA static artifacts."
    )
    parser.add_argument(
        "--price-panel",
        type=Path,
        default=DEFAULT_PRICE_PANEL,
        help="Wide price panel CSV. Default: price_panel.csv",
    )
    parser.add_argument(
        "--start",
        default="2025-03-01",
        help="Inclusive trading_day start date. Default: 2025-03-01",
    )
    parser.add_argument(
        "--end",
        default="2026-03-30",
        help="Inclusive trading_day end date. Default: 2026-03-30",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Directory for CSV outputs.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=DEFAULT_FIGURE_DIR,
        help="Directory for PNG figures.",
    )
    parser.add_argument(
        "--preview",
        type=Path,
        default=DEFAULT_PREVIEW,
        help="HTML preview output path.",
    )
    return parser.parse_args()


def load_price_panel(path: Path) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Copy the gitignored price_panel.csv first."
        )

    frame = pd.read_csv(path, parse_dates=["timestamp", "trading_day"])
    if "timestamp" not in frame.columns:
        raise ValueError("price panel must include a timestamp column")
    if "trading_day" not in frame.columns:
        raise ValueError("price panel must include a trading_day column")

    frame = frame.set_index("timestamp")
    tickers = [c for c in frame.columns if c != "trading_day"]
    prices = frame[tickers].astype(float)
    trading_day = pd.to_datetime(frame["trading_day"]).dt.normalize()
    trading_day.index = frame.index
    return prices, trading_day, tickers


def filter_minute_returns(
    prices: pd.DataFrame,
    trading_day: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    log_returns = np.log(prices).diff()
    mask = (trading_day >= start) & (trading_day <= end)
    return log_returns.loc[mask].dropna(how="all")


def compute_daily_returns(
    prices: pd.DataFrame,
    trading_day: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    daily_prices = prices.copy()
    daily_prices["_trading_day"] = trading_day.to_numpy()
    daily_last = daily_prices.groupby("_trading_day").last()
    daily_returns = np.log(daily_last).diff()
    daily_returns.index = pd.to_datetime(daily_returns.index).normalize()
    return daily_returns.loc[(daily_returns.index >= start) & (daily_returns.index <= end)]


def ordinary_kurtosis(values: pd.Series) -> float:
    arr = values.dropna().to_numpy(dtype=float)
    if arr.size < 4:
        return float("nan")
    centered = arr - arr.mean()
    second = np.mean(centered**2)
    if not np.isfinite(second) or second <= 0:
        return float("nan")
    fourth = np.mean(centered**4)
    return float(fourth / (second**2))


def build_summary(
    minute_returns: pd.DataFrame,
    daily_returns: pd.DataFrame,
    tickers: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for ticker in tickers:
        r_1m = minute_returns[ticker].dropna()
        r_day = daily_returns[ticker].dropna()
        ok = ordinary_kurtosis(r_1m)
        rows.append(
            {
                "ticker": ticker,
                "n_obs_1m": int(r_1m.size),
                "n_obs_daily": int(r_day.size),
                "mean_log_return_1m": float(r_1m.mean()),
                "std_log_return_1m": float(r_1m.std(ddof=1)),
                "min_log_return_1m": float(r_1m.min()),
                "max_log_return_1m": float(r_1m.max()),
                "mean_log_return_1m_bps": float(r_1m.mean() * 10_000),
                "std_log_return_1m_bps": float(r_1m.std(ddof=1) * 10_000),
                "min_log_return_1m_pct": float(r_1m.min() * 100),
                "max_log_return_1m_pct": float(r_1m.max() * 100),
                "mean_log_return_daily": float(r_day.mean()),
                "std_log_return_daily": float(r_day.std(ddof=1)),
                "mean_log_return_daily_pct": float(r_day.mean() * 100),
                "std_log_return_daily_pct": float(r_day.std(ddof=1) * 100),
                "ordinary_kurtosis_1m": ok,
                "excess_kurtosis_1m": ok - 3 if np.isfinite(ok) else float("nan"),
                "log_ordinary_kurtosis_1m": math.log(ok) if ok > 0 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def normal_pdf(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    if std <= 0 or not np.isfinite(std):
        return np.zeros_like(x)
    z = (x - mean) / std
    return np.exp(-0.5 * z**2) / (std * math.sqrt(2 * math.pi))


def save_log_return_histograms(
    minute_returns: pd.DataFrame,
    output_path: Path,
) -> None:
    tickers = [t for t in REPRESENTATIVE_TICKERS if t in minute_returns.columns]
    if not tickers:
        raise ValueError("None of the representative tickers are present in the panel.")

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), dpi=180)
    axes_flat = axes.ravel()
    colors = {"BTC": "#2f5597", "ETH": "#595959", "XRP": "#1f7a8c", "SOL": "#7b3294"}

    for ax, ticker in zip(axes_flat, tickers):
        values = minute_returns[ticker].dropna().to_numpy(dtype=float) * 100
        low, high = np.quantile(values, [0.01, 0.99])
        clipped = values[(values >= low) & (values <= high)]
        mean = float(values.mean())
        std = float(values.std(ddof=0))

        ax.hist(
            clipped,
            bins=80,
            density=True,
            color=colors.get(ticker, "#4c78a8"),
            alpha=0.72,
            edgecolor="white",
            linewidth=0.25,
        )
        x = np.linspace(low, high, 400)
        ax.plot(
            x,
            normal_pdf(x, mean, std),
            color="#c00000",
            linewidth=1.8,
            label="Normal density",
        )
        ax.axvline(0, color="#303030", linewidth=0.8, alpha=0.55)
        ax.set_title(f"{ticker} 1-minute log returns", fontsize=11, weight="bold")
        ax.set_xlabel("Log return (%)")
        ax.set_ylabel("Density")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="upper right", fontsize=8, frameon=False)

    for ax in axes_flat[len(tickers) :]:
        ax.axis("off")

    fig.suptitle("Representative Digital Asset Log-Return Distributions", fontsize=14)
    fig.text(
        0.5,
        0.015,
        "Display range is clipped to the 1st-99th percentiles; the normal curve uses full-sample mean and standard deviation.",
        ha="center",
        fontsize=9,
        color="#4d4d4d",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_mean_return_bar(summary: pd.DataFrame, output_path: Path) -> None:
    plot_df = summary.sort_values("mean_log_return_1m_bps").reset_index(drop=True)
    colors = [
        "#c00000" if ticker in HIGHLIGHT_TICKERS else "#4c78a8"
        for ticker in plot_df["ticker"]
    ]

    fig, ax = plt.subplots(figsize=(15, 6.8), dpi=180)
    ax.bar(plot_df["ticker"], plot_df["mean_log_return_1m_bps"], color=colors, width=0.78)
    ax.axhline(0, color="#303030", linewidth=0.9)
    ax.set_title("Mean 1-minute Log Return by Asset", fontsize=14, weight="bold")
    ax.set_ylabel("Mean log return (bps)")
    ax.set_xlabel("Ticker, sorted by mean return")
    ax.tick_params(axis="x", rotation=90, labelsize=8)
    ax.grid(axis="y", alpha=0.25)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#c00000", label="BTC/ETH/XRP/SOL"),
        plt.Rectangle((0, 0), 1, 1, color="#4c78a8", label="Other assets"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_log_kurtosis_boxplot(summary: pd.DataFrame, output_path: Path) -> None:
    values = summary["log_ordinary_kurtosis_1m"].dropna().to_numpy(dtype=float)
    tickers = summary.loc[
        summary["log_ordinary_kurtosis_1m"].notna(), "ticker"
    ].to_numpy(dtype=str)
    reference = math.log(T5_ORDINARY_KURTOSIS)
    above_reference = int(np.sum(values > reference))

    rng = np.random.default_rng(42)
    jitter = rng.normal(loc=0.0, scale=0.028, size=values.size)

    fig, ax = plt.subplots(figsize=(10.8, 4.8), dpi=180)
    ax.boxplot(
        values,
        vert=False,
        widths=0.38,
        patch_artist=True,
        boxprops={"facecolor": "#d9eaf7", "edgecolor": "#2f5597", "linewidth": 1.3},
        medianprops={"color": "#2f5597", "linewidth": 2},
        whiskerprops={"color": "#2f5597", "linewidth": 1.1},
        capprops={"color": "#2f5597", "linewidth": 1.1},
        flierprops={"marker": "", "markersize": 0},
    )
    ax.scatter(
        values,
        1 + jitter,
        s=28,
        color="#4c78a8",
        alpha=0.82,
        edgecolor="white",
        linewidth=0.35,
        zorder=3,
    )
    ax.axvline(
        reference,
        color="#c00000",
        linewidth=1.6,
        linestyle="--",
        label="Student-t(5): log(9)",
    )

    top_idx = np.argsort(values)[-5:]
    for idx in top_idx:
        ax.annotate(
            tickers[idx],
            (values[idx], 1 + jitter[idx]),
            xytext=(4, 5),
            textcoords="offset points",
            fontsize=8,
            color="#303030",
        )

    ax.set_title("Log Ordinary Kurtosis of 1-minute Log Returns", fontsize=14, weight="bold")
    ax.set_xlabel("log(ordinary kurtosis)")
    ax.set_yticks([])
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False, loc="upper left")
    ax.text(
        0.99,
        0.08,
        f"{above_reference}/{values.size} assets exceed the Student-t(5) reference",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#4d4d4d",
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_preview(
    summary: pd.DataFrame,
    preview_path: Path,
    figure_dir: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> None:
    rel = lambda path: path.relative_to(preview_path.parent).as_posix()
    figures = [
        figure_dir / "fig05_log_return_histograms.png",
        figure_dir / "fig06_asset_mean_return_bar.png",
        figure_dir / "fig07_log_kurtosis_boxplot.png",
    ]
    selected_cols = [
        "ticker",
        "mean_log_return_1m_bps",
        "std_log_return_1m_bps",
        "min_log_return_1m_pct",
        "max_log_return_1m_pct",
        "mean_log_return_daily_pct",
        "std_log_return_daily_pct",
        "ordinary_kurtosis_1m",
        "log_ordinary_kurtosis_1m",
    ]
    table = summary[selected_cols].round(4).to_html(index=False, border=0)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Phase 2 Report EDA Preview</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 28px; color: #202124; }}
h1 {{ margin-bottom: 4px; }}
.muted {{ color: #5f6368; }}
img {{ display: block; max-width: 1100px; width: 100%; margin: 22px 0 34px; border: 1px solid #ddd; }}
table {{ border-collapse: collapse; font-size: 13px; }}
th, td {{ border-bottom: 1px solid #e5e5e5; padding: 6px 8px; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
</style>
</head>
<body>
<h1>Phase 2 Report EDA Preview</h1>
<p class="muted">Trading-day period: {start.date()} to {end.date()} (inclusive). Partial trading day 2026-03-31 is excluded.</p>
<h2>Figure 5. Representative log-return histograms</h2>
<img src="{rel(figures[0])}" alt="Representative log-return histograms">
<h2>Figure 6. Mean 1-minute log return by asset</h2>
<img src="{rel(figures[1])}" alt="Mean 1-minute log return by asset">
<h2>Figure 7. Log ordinary kurtosis box plot</h2>
<img src="{rel(figures[2])}" alt="Log ordinary kurtosis box plot">
<h2>Summary table</h2>
{table}
</body>
</html>
"""
    preview_path.write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    start = pd.Timestamp(args.start).normalize()
    end = pd.Timestamp(args.end).normalize()
    if end < start:
        raise ValueError("--end must be greater than or equal to --start")

    args.processed_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    args.preview.parent.mkdir(parents=True, exist_ok=True)

    print(f"[phase2_report_eda] loading {args.price_panel}")
    prices, trading_day, tickers = load_price_panel(args.price_panel)

    print(f"[phase2_report_eda] filtering trading days {start.date()} to {end.date()}")
    minute_returns = filter_minute_returns(prices, trading_day, start, end)
    daily_returns = compute_daily_returns(prices, trading_day, start, end)

    print("[phase2_report_eda] computing summary")
    summary = build_summary(minute_returns, daily_returns, tickers)
    summary_path = args.processed_dir / "report_log_return_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    fig05 = args.figure_dir / "fig05_log_return_histograms.png"
    fig06 = args.figure_dir / "fig06_asset_mean_return_bar.png"
    fig07 = args.figure_dir / "fig07_log_kurtosis_boxplot.png"

    print(f"[phase2_report_eda] writing {fig05}")
    save_log_return_histograms(minute_returns, fig05)
    print(f"[phase2_report_eda] writing {fig06}")
    save_mean_return_bar(summary, fig06)
    print(f"[phase2_report_eda] writing {fig07}")
    save_log_kurtosis_boxplot(summary, fig07)
    print(f"[phase2_report_eda] writing {args.preview}")
    write_preview(summary, args.preview, args.figure_dir, start, end)

    reference = math.log(T5_ORDINARY_KURTOSIS)
    above = int((summary["log_ordinary_kurtosis_1m"] > reference).sum())
    print("[phase2_report_eda] complete")
    print(f"  summary: {summary_path}")
    print(f"  rows: {len(summary)}")
    print(f"  median log ordinary kurtosis: {summary['log_ordinary_kurtosis_1m'].median():.3f}")
    print(f"  Student-t(5) reference log(9): {reference:.3f}")
    print(f"  assets above reference: {above}/{len(summary)}")


if __name__ == "__main__":
    main()
