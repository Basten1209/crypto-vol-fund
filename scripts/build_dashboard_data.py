#!/usr/bin/env python3
"""Build static dashboard data from Phase 5/6 monthly capped outputs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_AUM_KRW = 100_000_000
DEFAULT_DATE = "2026-01-01"
CAP_WEIGHT = 0.25
POLICY_LABELS = {
    "enter_once_then_drift": "Simple Mode",
    "daily_rebalance_to_target": "Managed Mode",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build static JSON/JS data for the portfolio dashboard.")
    parser.add_argument("--price-panel", type=Path, default=ROOT / "price_panel.csv")
    parser.add_argument(
        "--weights",
        type=Path,
        default=ROOT / "data" / "processed" / "phase5_monthly_cap25" / "minimum_variance_weights_wide.csv",
    )
    parser.add_argument(
        "--daily-returns",
        type=Path,
        default=ROOT / "data" / "processed" / "phase6_monthly_cap25" / "phase6_daily_returns.csv",
    )
    parser.add_argument(
        "--performance",
        type=Path,
        default=ROOT / "data" / "processed" / "phase6_monthly_cap25" / "phase6_performance_table.csv",
    )
    parser.add_argument(
        "--monthly-returns",
        type=Path,
        default=ROOT / "data" / "processed" / "phase6_monthly_cap25" / "monthly_hold_window_returns.csv",
    )
    parser.add_argument(
        "--monthly-metrics",
        type=Path,
        default=ROOT
        / "data"
        / "processed"
        / "phase6_monthly_cap25"
        / "monthly_equal_weight_vs_minvar_metrics_long.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dashboard" / "data")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path in [args.price_panel, args.weights, args.daily_returns, args.performance, args.monthly_returns, args.monthly_metrics]:
        if not path.exists():
            raise FileNotFoundError(path)

    weights_wide = pd.read_csv(args.weights)
    tickers = [col for col in weights_wide.columns if col not in {"cycle_days", "rebalance_date"}]
    daily_returns = pd.read_csv(args.daily_returns)
    performance = pd.read_csv(args.performance)
    monthly_returns = pd.read_csv(args.monthly_returns)
    monthly_metrics = pd.read_csv(args.monthly_metrics)
    daily_prices = load_daily_prices(args.price_panel, tickers)

    dates = sorted(daily_returns["date"].astype(str).unique())
    weights = build_weight_map(weights_wide, tickers)
    snapshots = build_snapshots(dates, tickers, weights, daily_prices)
    series = build_daily_series(daily_returns)

    data = {
        "metadata": {
            "title": "월간 가상자산 변동성 모델 포트폴리오",
            "subtitle": "Monthly virtual asset volatility model portfolio",
            "generated_from": {
                "weights": repo_path(args.weights),
                "daily_returns": repo_path(args.daily_returns),
                "performance": repo_path(args.performance),
                "monthly_returns": repo_path(args.monthly_returns),
                "monthly_metrics": repo_path(args.monthly_metrics),
            },
            "date_start": dates[0],
            "date_end": dates[-1],
            "default_date": DEFAULT_DATE if DEFAULT_DATE in dates else dates[0],
            "default_cycle_days": 14,
            "default_policy": "daily_rebalance_to_target",
            "default_aum_krw": DEFAULT_AUM_KRW,
            "single_asset_cap": CAP_WEIGHT,
            "transaction_costs": "ignored",
            "aum_policy": "reset_to_default_aum_at_each_monthly_entry",
            "policies": POLICY_LABELS,
            "cycles": sorted(int(cycle) for cycle in weights),
            "tickers": tickers,
        },
        "dates": dates,
        "performance": clean_records(performance),
        "monthly_returns": clean_records(monthly_returns),
        "monthly_metrics": clean_records(monthly_metrics),
        "daily_series": series,
        "snapshots": snapshots,
    }

    validate_dashboard_data(data)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "dashboard_snapshots.json"
    js_path = args.output_dir / "dashboard_snapshots.js"
    payload = json.dumps(json_ready(data), ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    json_path.write_text(payload + "\n", encoding="utf-8")
    js_path.write_text("window.DASHBOARD_DATA = " + payload + ";\n", encoding="utf-8")
    print(f"Saved {repo_path(json_path)}")
    print(f"Saved {repo_path(js_path)}")
    print(f"Snapshots: {len(dates)} dates x {len(weights)} cycles x {len(POLICY_LABELS)} policies")
    return 0


def load_daily_prices(price_panel_path: Path, tickers: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(price_panel_path, usecols=["trading_day", *tickers], encoding="utf-8-sig")
    frame["trading_day"] = frame["trading_day"].astype(str)
    daily = frame.groupby("trading_day", sort=True)[tickers].last()
    if daily.isna().any().any():
        raise ValueError("Daily price panel contains missing values after grouping by trading_day")
    return daily.astype(float)


def build_weight_map(weights_wide: pd.DataFrame, tickers: list[str]) -> dict[int, dict[str, np.ndarray]]:
    output: dict[int, dict[str, np.ndarray]] = {}
    for cycle_days, group in weights_wide.groupby("cycle_days", sort=True):
        by_date: dict[str, np.ndarray] = {}
        for _, row in group.sort_values("rebalance_date").iterrows():
            weights = row[tickers].to_numpy(dtype=np.float64)
            weights[np.abs(weights) < 1e-12] = 0.0
            total = float(weights.sum())
            if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-8):
                raise ValueError(f"Weight sum for cycle {cycle_days} on {row['rebalance_date']} is {total}")
            if float(weights.max()) > CAP_WEIGHT + 1e-8:
                raise ValueError(f"Weight cap exceeded for cycle {cycle_days} on {row['rebalance_date']}")
            by_date[str(row["rebalance_date"])] = weights
        output[int(cycle_days)] = by_date
    return output


def build_snapshots(
    dates: list[str],
    tickers: list[str],
    weights: dict[int, dict[str, np.ndarray]],
    daily_prices: pd.DataFrame,
) -> dict[str, dict[str, dict[str, Any]]]:
    snapshots: dict[str, dict[str, dict[str, Any]]] = {}
    date_index = {date: idx for idx, date in enumerate(dates)}
    all_dates = pd.to_datetime(pd.Series(dates))

    for date in dates:
        date_ts = pd.Timestamp(date)
        snapshots[date] = {}
        for cycle, by_rebalance in weights.items():
            rebalance_dates = sorted(by_rebalance)
            rebalance_date = previous_rebalance(date, rebalance_dates)
            next_rebalance_date = next_rebalance(date, rebalance_dates)
            cycle_payload: dict[str, Any] = {
                "rebalance_date": rebalance_date,
                "next_rebalance_date": next_rebalance_date,
                "policies": {},
            }
            if rebalance_date is None:
                snapshots[date][str(cycle)] = cycle_payload
                continue

            start_ts = pd.Timestamp(rebalance_date)
            hold_end_ts = start_ts + pd.Timedelta(days=cycle - 1)
            exit_ts = start_ts + pd.Timedelta(days=cycle)
            in_hold = start_ts <= date_ts <= hold_end_ts
            is_exit = date_ts == exit_ts
            target = by_rebalance[rebalance_date]
            previous_date = dates[date_index[date] - 1] if date_index[date] > 0 else None

            for policy in POLICY_LABELS:
                current_before = np.zeros(len(tickers), dtype=np.float64)
                target_after = np.zeros(len(tickers), dtype=np.float64)
                holdings_after = np.zeros(len(tickers), dtype=np.float64)
                current_aum_multiplier = 1.0
                action = "Cash"
                status = "Off-window cash"

                if in_hold:
                    if policy == "enter_once_then_drift":
                        if date == rebalance_date:
                            current_before = np.zeros(len(tickers), dtype=np.float64)
                            target_after = target.copy()
                            holdings_after = target.copy()
                            current_aum_multiplier = 1.0
                            action = "Enter monthly target"
                        else:
                            current_before = drift_weights(target, daily_prices, tickers, rebalance_date, date)
                            target_after = current_before.copy()
                            holdings_after = current_before.copy()
                            current_aum_multiplier = portfolio_value_between(
                                target,
                                daily_prices,
                                tickers,
                                rebalance_date,
                                date,
                            )
                            action = "Hold and drift"
                    else:
                        target_after = target.copy()
                        if date == rebalance_date or previous_date is None:
                            current_before = np.zeros(len(tickers), dtype=np.float64)
                            current_aum_multiplier = 1.0
                            action = "Enter monthly target"
                        else:
                            current_before = drift_weights(target, daily_prices, tickers, previous_date, date)
                            current_aum_multiplier = managed_window_value(
                                target,
                                daily_prices,
                                tickers,
                                dates,
                                rebalance_date,
                                date,
                            )
                            action = "Rebalance to target"
                        holdings_after = target.copy()
                    status = f"Active hold day {(date_ts - start_ts).days + 1} / {cycle}"
                elif is_exit:
                    if policy == "enter_once_then_drift":
                        current_before = drift_weights(target, daily_prices, tickers, rebalance_date, date)
                        current_aum_multiplier = portfolio_value_between(
                            target,
                            daily_prices,
                            tickers,
                            rebalance_date,
                            date,
                        )
                    elif previous_date is not None:
                        current_before = drift_weights(target, daily_prices, tickers, previous_date, date)
                        current_aum_multiplier = managed_window_value(
                            target,
                            daily_prices,
                            tickers,
                            dates,
                            rebalance_date,
                            date,
                        )
                    action = "Exit to cash"
                    status = "Hold window finished"

                order_delta = target_after - current_before
                cycle_payload["policies"][policy] = {
                    "label": POLICY_LABELS[policy],
                    "status": status,
                    "action": action,
                    "hold_start": rebalance_date,
                    "hold_end": hold_end_ts.date().isoformat(),
                    "exit_date": exit_ts.date().isoformat(),
                    "in_hold_window": in_hold,
                    "is_exit_day": is_exit,
                    "target_sum": float(target_after.sum()),
                    "current_sum": float(current_before.sum()),
                    "order_abs_sum": float(np.abs(order_delta).sum()),
                    "entry_aum_multiplier": 1.0,
                    "current_aum_multiplier": float(current_aum_multiplier),
                    "order_aum_multiplier": float(current_aum_multiplier),
                    "window_return_to_date": float(current_aum_multiplier - 1.0),
                    "top_weight": float(holdings_after.max()) if holdings_after.size else 0.0,
                    "active_count": int(np.sum(holdings_after > 1e-6)),
                    "holdings": vector_rows(tickers, holdings_after, limit=20),
                    "target": vector_rows(tickers, target, limit=20),
                    "orders": order_rows(tickers, target_after, current_before, order_delta),
                }
            snapshots[date][str(cycle)] = cycle_payload
    return snapshots


def drift_weights(
    target: np.ndarray,
    daily_prices: pd.DataFrame,
    tickers: list[str],
    start_date: str,
    end_date: str,
) -> np.ndarray:
    if start_date not in daily_prices.index or end_date not in daily_prices.index:
        return target.copy()
    start_prices = daily_prices.loc[start_date, tickers].to_numpy(dtype=np.float64)
    end_prices = daily_prices.loc[end_date, tickers].to_numpy(dtype=np.float64)
    relatives = end_prices / start_prices
    drifted = target * relatives
    total = float(drifted.sum())
    if total <= 0 or not np.isfinite(total):
        return target.copy()
    return drifted / total


def portfolio_value_between(
    target: np.ndarray,
    daily_prices: pd.DataFrame,
    tickers: list[str],
    start_date: str,
    end_date: str,
) -> float:
    if start_date not in daily_prices.index or end_date not in daily_prices.index:
        return 1.0
    start_prices = daily_prices.loc[start_date, tickers].to_numpy(dtype=np.float64)
    end_prices = daily_prices.loc[end_date, tickers].to_numpy(dtype=np.float64)
    relatives = end_prices / start_prices
    value = float(np.dot(target, relatives))
    return value if np.isfinite(value) and value > 0 else 1.0


def managed_window_value(
    target: np.ndarray,
    daily_prices: pd.DataFrame,
    tickers: list[str],
    dates: list[str],
    start_date: str,
    end_date: str,
) -> float:
    if start_date not in daily_prices.index or end_date not in daily_prices.index:
        return 1.0
    try:
        start_idx = dates.index(start_date)
        end_idx = dates.index(end_date)
    except ValueError:
        return 1.0
    if end_idx <= start_idx:
        return 1.0

    value = 1.0
    for idx in range(start_idx + 1, end_idx + 1):
        value *= portfolio_value_between(target, daily_prices, tickers, dates[idx - 1], dates[idx])
    return float(value) if np.isfinite(value) and value > 0 else 1.0


def previous_rebalance(date: str, rebalance_dates: list[str]) -> str | None:
    candidates = [item for item in rebalance_dates if item <= date]
    return candidates[-1] if candidates else None


def next_rebalance(date: str, rebalance_dates: list[str]) -> str | None:
    candidates = [item for item in rebalance_dates if item > date]
    return candidates[0] if candidates else None


def vector_rows(tickers: list[str], values: np.ndarray, limit: int) -> list[dict[str, Any]]:
    rows = [
        {"ticker": ticker, "weight": float(weight)}
        for ticker, weight in zip(tickers, values)
        if abs(float(weight)) > 1e-8
    ]
    rows.sort(key=lambda row: abs(row["weight"]), reverse=True)
    return rows[:limit]


def order_rows(
    tickers: list[str],
    target_after: np.ndarray,
    current_before: np.ndarray,
    order_delta: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for ticker, target_weight, current_weight, delta_weight in zip(
        tickers,
        target_after,
        current_before,
        order_delta,
    ):
        if max(abs(float(target_weight)), abs(float(current_weight)), abs(float(delta_weight))) <= 1e-8:
            continue
        rows.append(
            {
                "ticker": ticker,
                "target_weight": float(target_weight),
                "current_weight": float(current_weight),
                "delta_weight": float(delta_weight),
                "side": order_side(float(delta_weight)),
            }
        )
    rows.sort(key=lambda row: abs(row["delta_weight"]), reverse=True)
    return rows


def order_side(delta_weight: float) -> str:
    if delta_weight > 1e-8:
        return "BUY"
    if delta_weight < -1e-8:
        return "SELL"
    return "HOLD"


def build_daily_series(daily_returns: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for (strategy, policy, cycle), group in daily_returns.groupby(
        ["strategy", "rebalance_policy", "cycle_days"],
        dropna=False,
        sort=True,
    ):
        cycle_label = "btc" if pd.isna(cycle) else str(int(cycle))
        key = f"{strategy}|{policy}|{cycle_label}"
        ordered = group.sort_values("date")
        equity = ordered["equity"].to_numpy(dtype=np.float64)
        peak = np.maximum.accumulate(equity)
        drawdown = equity / peak - 1.0
        rows = []
        for idx, row in enumerate(ordered.to_dict(orient="records")):
            rows.append(
                {
                    "date": str(row["date"]),
                    "daily_return": float(row["daily_return"]),
                    "equity": float(row["equity"]),
                    "drawdown": float(drawdown[idx]),
                    "realized_risk_annualized": float(row["realized_risk_annualized"]),
                    "ending_top_ticker": str(row["ending_top_ticker"]),
                    "ending_top_weight": float(row["ending_top_weight"]),
                    "ending_active_count": int(row["ending_active_count"]),
                }
            )
        output[key] = rows
    return output


def validate_dashboard_data(data: dict[str, Any]) -> None:
    date = data["metadata"]["default_date"]
    default = data["snapshots"][date]["14"]["policies"]["daily_rebalance_to_target"]
    btc = next((row for row in default["target"] if row["ticker"] == "BTC"), None)
    tokamak = next((row for row in default["target"] if row["ticker"] == "TOKAMAK"), None)
    ardr = next((row for row in default["target"] if row["ticker"] == "ARDR"), None)
    if btc is None or not math.isclose(btc["weight"], 0.25, abs_tol=1e-8):
        raise ValueError("Default snapshot does not contain BTC at 25%")
    if tokamak is None or not math.isclose(tokamak["weight"], 0.16391955008744796, abs_tol=1e-8):
        raise ValueError("Default snapshot does not contain expected TOKAMAK weight")
    if ardr is None or not math.isclose(ardr["weight"], 0.09621251582192208, abs_tol=1e-8):
        raise ValueError("Default snapshot does not contain expected ARDR weight")
    if not math.isclose(default["entry_aum_multiplier"], 1.0, abs_tol=1e-12):
        raise ValueError("Default snapshot must reset entry AUM to 1.0")
    if not math.isclose(default["current_aum_multiplier"], 1.0, abs_tol=1e-12):
        raise ValueError("Default rebalance date must start from identical AUM")

    perf = data["performance"]
    managed_14 = [
        row
        for row in perf
        if row["strategy"] == "minimum_variance"
        and row["rebalance_policy"] == "daily_rebalance_to_target"
        and int(float(row["cycle_days"])) == 14
    ]
    if not managed_14:
        raise ValueError("Missing 14D Managed Mode performance row")
    row = managed_14[0]
    for field in ["total_return", "sharpe_ratio", "max_drawdown"]:
        if not isinstance(row[field], float) or not math.isfinite(row[field]):
            raise ValueError(f"Invalid 14D Managed Mode {field}")


def clean_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [json_ready(row) for row in frame.to_dict(orient="records")]


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if pd.isna(value):
        return None
    return value


def repo_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
