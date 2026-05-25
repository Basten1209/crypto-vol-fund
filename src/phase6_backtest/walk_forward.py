"""Walk-forward Phase 6 backtesting from price panel and Phase 5 weights."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.phase6_backtest.benchmarks import equal_weight_targets, single_asset_targets  # noqa: E402
from src.phase6_backtest.dm_test import diebold_mariano_test  # noqa: E402
from src.phase6_backtest.metrics import (  # noqa: E402
    annualized_return,
    annualized_volatility,
    max_drawdown,
    performance_row,
    sharpe_ratio,
)
from src.utils import ensure_dir, repo_relative_path  # noqa: E402


def run_phase6_backtest(
    price_panel_path: Path | str,
    portfolio_path: Path | str,
    output_dir: Path | str,
    eval_freq_min: int = config.EVAL_FREQ_MIN,
) -> dict[str, Any]:
    """Run Phase 6 backtest and write performance artifacts."""
    price_panel_path = Path(price_panel_path)
    portfolio_path = Path(portfolio_path)
    output_dir = Path(output_dir)
    if not price_panel_path.exists():
        raise FileNotFoundError(
            f"{price_panel_path} does not exist. Phase 6 requires the gitignored price_panel.csv from Phase 1."
        )
    if not portfolio_path.exists():
        raise FileNotFoundError(portfolio_path)

    portfolios = _load_phase5_portfolios(portfolio_path)
    returns = _load_eval_returns(
        price_panel_path=price_panel_path,
        tickers=portfolios["tickers"],
        eval_freq_min=eval_freq_min,
        required_dates=_required_dates(portfolios),
    )

    backtests: list[pd.DataFrame] = []
    performance_rows: list[dict[str, Any]] = []
    interval_frames: list[pd.DataFrame] = []

    btc_result = _run_strategy(
        strategy="btc_hodl",
        rebalance_policy="buy_and_hold",
        cycle_days=None,
        tickers=portfolios["tickers"],
        dates=returns["dates"],
        interval_counts=returns["interval_counts"],
        simple_returns=returns["simple_returns"],
        rebalance_dates=np.asarray([returns["dates"][0]], dtype="U10"),
        target_weights=single_asset_targets(np.asarray([returns["dates"][0]], dtype="U10"), portfolios["tickers"], "BTC"),
    )
    btc_daily_returns = btc_result["daily"]["daily_return"].to_numpy(dtype=np.float64)
    backtests.append(btc_result["daily"])
    interval_frames.append(btc_result["interval"])
    performance_rows.append(
        _with_policy(
            performance_row(
            strategy="btc_hodl",
            cycle_days=None,
            daily_returns=btc_daily_returns,
            equity=btc_result["daily"]["equity"].to_numpy(dtype=np.float64),
            btc_daily_returns=btc_daily_returns,
            annualization=config.ANNUALIZATION,
            risk_free_rate=config.RISK_FREE_RATE,
            ),
            "buy_and_hold",
            btc_result,
        )
    )

    dm_rows: list[dict[str, Any]] = []
    for cycle in portfolios["cycles"]:
        cycle_data = portfolios["cycles"][cycle]
        equal_weight_scheduled = _equal_weight_scheduled_weights(cycle_data, len(portfolios["tickers"]))
        for rebalance_policy in ["enter_once_then_drift", "daily_rebalance_to_target"]:
            min_var = _run_strategy(
                strategy="minimum_variance",
                rebalance_policy=rebalance_policy,
                cycle_days=cycle,
                tickers=portfolios["tickers"],
                dates=returns["dates"],
                interval_counts=returns["interval_counts"],
                simple_returns=returns["simple_returns"],
                rebalance_dates=cycle_data["rebalance_dates"],
                target_weights=cycle_data["weights"],
                scheduled_dates=cycle_data.get("scheduled_dates"),
                scheduled_rebalance_dates=cycle_data.get("scheduled_rebalance_dates"),
                scheduled_weights=cycle_data.get("scheduled_weights"),
            )
            equal_weight = _run_strategy(
                strategy="equal_weight",
                rebalance_policy=rebalance_policy,
                cycle_days=cycle,
                tickers=portfolios["tickers"],
                dates=returns["dates"],
                interval_counts=returns["interval_counts"],
                simple_returns=returns["simple_returns"],
                rebalance_dates=cycle_data["rebalance_dates"],
                target_weights=equal_weight_targets(cycle_data["rebalance_dates"], len(portfolios["tickers"])),
                scheduled_dates=cycle_data.get("scheduled_dates"),
                scheduled_rebalance_dates=cycle_data.get("scheduled_rebalance_dates"),
                scheduled_weights=equal_weight_scheduled,
            )

            for result in [min_var, equal_weight]:
                daily = result["daily"]
                backtests.append(daily)
                interval_frames.append(result["interval"])
                performance_rows.append(
                    _with_policy(
                        performance_row(
                            strategy=str(daily["strategy"].iloc[0]),
                            cycle_days=cycle,
                            daily_returns=daily["daily_return"].to_numpy(dtype=np.float64),
                            equity=daily["equity"].to_numpy(dtype=np.float64),
                            btc_daily_returns=btc_daily_returns,
                            turnover_mean=result["turnover_mean"],
                            realized_risk_mean=float(daily["realized_risk_annualized"].mean()),
                            annualization=config.ANNUALIZATION,
                            risk_free_rate=config.RISK_FREE_RATE,
                        ),
                        rebalance_policy,
                        result,
                    )
                )

            min_var_loss = np.square(min_var["daily"]["daily_return"].to_numpy(dtype=np.float64))
            ew_loss = np.square(equal_weight["daily"]["daily_return"].to_numpy(dtype=np.float64))
            dm = diebold_mariano_test(min_var_loss, ew_loss, lag=max(0, cycle - 1))
            dm_rows.append(
                {
                    "comparison": "minimum_variance_vs_equal_weight",
                    "rebalance_policy": rebalance_policy,
                    "cycle_days": cycle,
                    "loss": "squared_daily_return",
                    **dm,
                }
            )

    output_dir = ensure_dir(output_dir)
    daily_df = pd.concat(backtests, ignore_index=True)
    interval_df = pd.concat(interval_frames, ignore_index=True)
    performance_df = pd.DataFrame(performance_rows)
    dm_df = pd.DataFrame(dm_rows)

    daily_path = output_dir / "phase6_daily_returns.csv"
    interval_path = output_dir / "phase6_interval_returns.csv"
    performance_path = output_dir / "phase6_performance_table.csv"
    dm_path = output_dir / "phase6_dm_test.csv"
    report_path = output_dir / "phase6_backtest_report.json"

    daily_df.to_csv(daily_path, index=False)
    interval_df.to_csv(interval_path, index=False)
    performance_df.to_csv(performance_path, index=False)
    dm_df.to_csv(dm_path, index=False)
    monthly_outputs = _write_monthly_comparison_outputs(output_dir, daily_df)

    report = {
        "price_panel_path": repo_relative_path(price_panel_path, ROOT),
        "portfolio_path": repo_relative_path(portfolio_path, ROOT),
        "output_dir": repo_relative_path(output_dir, ROOT),
        "params": {
            "eval_freq_min": eval_freq_min,
            "annualization": config.ANNUALIZATION,
            "risk_free_rate": config.RISK_FREE_RATE,
            "transaction_costs": "ignored",
            "weight_drift": "enabled_between_rebalances",
            "scheduled_hold_windows": "enabled_when_phase5_scheduled_dates_do_not_cover_all_days",
            "off_window_return": "cash_0_percent",
            "rebalance_policies": ["enter_once_then_drift", "daily_rebalance_to_target"],
        },
        "n_assets": int(len(portfolios["tickers"])),
        "n_backtest_days": int(len(returns["dates"])),
        "first_date": str(returns["dates"][0]),
        "last_date": str(returns["dates"][-1]),
        "performance": performance_df.to_dict(orient="records"),
        "dm_tests": dm_df.to_dict(orient="records"),
        "outputs": {
            "daily_returns_csv": repo_relative_path(daily_path, ROOT),
            "interval_returns_csv": repo_relative_path(interval_path, ROOT),
            "performance_table_csv": repo_relative_path(performance_path, ROOT),
            "dm_test_csv": repo_relative_path(dm_path, ROOT),
            "report_json": repo_relative_path(report_path, ROOT),
            **{name: repo_relative_path(path, ROOT) for name, path in monthly_outputs.items()},
        },
    }
    report_path.write_text(json.dumps(_json_ready(report), indent=2, allow_nan=False), encoding="utf-8")
    _write_phase6_markdown_report(output_dir / "phase6_results_note.md", report)

    print("=== Phase 6 backtest ===")
    print(f"price_panel: {price_panel_path}")
    print(f"portfolio_input: {portfolio_path}")
    print(f"output_dir: {output_dir}")
    print(f"backtest_dates: {returns['dates'][0]} to {returns['dates'][-1]} ({len(returns['dates'])} days)")
    print(f"saved performance: {repo_relative_path(performance_path, ROOT)}")
    print(f"saved report: {repo_relative_path(report_path, ROOT)}")
    return report


def _write_monthly_comparison_outputs(output_dir: Path, daily_df: pd.DataFrame) -> dict[str, Path]:
    metrics_df = _monthly_strategy_metrics(daily_df)
    if metrics_df.empty:
        return {}

    metrics_long_path = output_dir / "monthly_equal_weight_vs_minvar_metrics_long.csv"
    metrics_wide_path = output_dir / "monthly_equal_weight_vs_minvar_metrics_wide.csv"
    metrics_sheet_path = output_dir / "monthly_equal_weight_vs_minvar_metrics_sheet.md"
    returns_long_path = output_dir / "monthly_equal_weight_vs_minvar_long.csv"
    returns_wide_path = output_dir / "monthly_equal_weight_vs_minvar_wide.csv"
    returns_sheet_path = output_dir / "monthly_equal_weight_vs_minvar_sheet.md"

    metrics_df.to_csv(metrics_long_path, index=False)
    _monthly_metrics_wide(metrics_df).to_csv(metrics_wide_path, index=False)
    _write_monthly_metrics_markdown(metrics_sheet_path, metrics_df)

    returns_df = _monthly_return_comparison(daily_df, metrics_df)
    if not returns_df.empty:
        returns_df.to_csv(returns_long_path, index=False)
        _monthly_returns_wide(returns_df).to_csv(returns_wide_path, index=False)
        _write_monthly_returns_markdown(returns_sheet_path, returns_df)

    outputs = {
        "monthly_metrics_long_csv": metrics_long_path,
        "monthly_metrics_wide_csv": metrics_wide_path,
        "monthly_metrics_sheet_md": metrics_sheet_path,
    }
    if not returns_df.empty:
        outputs.update(
            {
                "monthly_returns_long_csv": returns_long_path,
                "monthly_returns_wide_csv": returns_wide_path,
                "monthly_returns_sheet_md": returns_sheet_path,
            }
        )
    return outputs


def _monthly_strategy_metrics(daily_df: pd.DataFrame) -> pd.DataFrame:
    compare = daily_df[daily_df["strategy"].isin(["minimum_variance", "equal_weight"])].copy()
    compare = compare[compare["cycle_days"].notna()]
    compare = compare[compare["ending_active_count"] > 0]
    if compare.empty:
        return pd.DataFrame()

    compare["date"] = pd.to_datetime(compare["date"])
    compare["month"] = compare["date"].dt.to_period("M").astype(str)

    rows: list[dict[str, Any]] = []
    group_cols = ["month", "strategy", "rebalance_policy", "cycle_days"]
    for (month, strategy, policy, cycle_days), group in compare.groupby(group_cols, sort=True):
        ordered = group.sort_values("date")
        daily_returns = ordered["daily_return"].to_numpy(dtype=np.float64)
        equity = np.cumprod(1.0 + daily_returns)
        rows.append(
            {
                "month": str(month),
                "strategy": str(strategy),
                "rebalance_policy": str(policy),
                "cycle_days": int(cycle_days),
                "hold_start": ordered["date"].iloc[0].date().isoformat(),
                "hold_end": ordered["date"].iloc[-1].date().isoformat(),
                "hold_days": int(len(ordered)),
                "total_return": float(equity[-1] - 1.0),
                "annualized_return": annualized_return(daily_returns, annualization=config.ANNUALIZATION),
                "annualized_volatility": annualized_volatility(daily_returns, annualization=config.ANNUALIZATION),
                "sharpe_ratio": sharpe_ratio(
                    daily_returns,
                    risk_free_rate=config.RISK_FREE_RATE,
                    annualization=config.ANNUALIZATION,
                ),
                "max_drawdown": max_drawdown(equity),
                "realized_risk_annualized_mean": float(ordered["realized_risk_annualized"].mean()),
                "ending_top_weight_max": float(ordered["ending_top_weight"].max()),
                "ending_active_count_mean": float(ordered["ending_active_count"].mean()),
            }
        )

    metrics_df = pd.DataFrame(rows)
    for column in [
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "max_drawdown",
        "realized_risk_annualized_mean",
        "ending_top_weight_max",
    ]:
        metrics_df[f"{column}_pct"] = metrics_df[column] * 100.0
    return metrics_df.sort_values(["rebalance_policy", "cycle_days", "month", "strategy"]).reset_index(drop=True)


def _monthly_metrics_wide(metrics_df: pd.DataFrame) -> pd.DataFrame:
    policies = _ordered_monthly_policies(metrics_df)
    months = sorted(metrics_df["month"].unique())
    cycles = sorted(metrics_df["cycle_days"].unique())
    strategy_alias = {"equal_weight": "ew", "minimum_variance": "mv"}
    metric_cols = [
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "realized_risk_annualized_mean",
        "ending_top_weight_max",
        "ending_active_count_mean",
    ]
    rows: list[dict[str, Any]] = []
    for policy in policies:
        policy_df = metrics_df[metrics_df["rebalance_policy"] == policy]
        for month in months:
            row: dict[str, Any] = {"rebalance_policy": policy, "month": month}
            for cycle in cycles:
                cycle_month = policy_df[(policy_df["month"] == month) & (policy_df["cycle_days"] == cycle)]
                by_strategy = {item["strategy"]: item for item in cycle_month.to_dict(orient="records")}
                for metric in metric_cols:
                    for strategy in ["equal_weight", "minimum_variance"]:
                        values = by_strategy.get(strategy)
                        row[f"{metric}_{int(cycle)}d_{strategy_alias[strategy]}"] = values.get(metric) if values else np.nan
                ew = by_strategy.get("equal_weight", {})
                mv = by_strategy.get("minimum_variance", {})
                for metric in ["total_return", "annualized_volatility", "sharpe_ratio", "max_drawdown"]:
                    row[f"{metric}_{int(cycle)}d_mv_minus_ew"] = _safe_subtract(mv.get(metric), ew.get(metric))
            rows.append(row)
    return pd.DataFrame(rows)


def _monthly_return_comparison(daily_df: pd.DataFrame, metrics_df: pd.DataFrame) -> pd.DataFrame:
    policy = _default_monthly_policy(metrics_df)
    filtered = metrics_df[metrics_df["rebalance_policy"] == policy]
    btc = daily_df[daily_df["strategy"] == "btc_hodl"].copy()
    if not btc.empty:
        btc["date"] = pd.to_datetime(btc["date"])

    rows: list[dict[str, Any]] = []
    for (month, cycle_days), group in filtered.groupby(["month", "cycle_days"], sort=True):
        by_strategy = {item["strategy"]: item for item in group.to_dict(orient="records")}
        minimum_variance = by_strategy.get("minimum_variance")
        equal_weight = by_strategy.get("equal_weight")
        if not minimum_variance or not equal_weight:
            continue
        hold_start = str(minimum_variance["hold_start"])
        hold_end = str(minimum_variance["hold_end"])
        btc_return = np.nan
        if not btc.empty:
            btc_window = btc[(btc["date"] >= pd.Timestamp(hold_start)) & (btc["date"] <= pd.Timestamp(hold_end))]
            if not btc_window.empty:
                btc_return = float(np.prod(1.0 + btc_window["daily_return"].to_numpy(dtype=np.float64)) - 1.0)
        mv_return = float(minimum_variance["total_return"])
        ew_return = float(equal_weight["total_return"])
        rows.append(
            {
                "month": str(month),
                "cycle_days": int(cycle_days),
                "hold_start": hold_start,
                "hold_end": hold_end,
                "hold_days": int(minimum_variance["hold_days"]),
                "minimum_variance_return": mv_return,
                "equal_weight_return": ew_return,
                "btc_same_window_return": btc_return,
                "mv_minus_ew": mv_return - ew_return,
                "mv_minus_btc_same_window": _safe_subtract(mv_return, btc_return),
            }
        )
    return pd.DataFrame(rows)


def _monthly_returns_wide(returns_df: pd.DataFrame) -> pd.DataFrame:
    months = sorted(returns_df["month"].unique())
    cycles = sorted(returns_df["cycle_days"].unique())
    value_cols = ["minimum_variance_return", "equal_weight_return", "mv_minus_ew"]
    rows: list[dict[str, Any]] = []
    for month in months:
        row: dict[str, Any] = {"month": month}
        month_df = returns_df[returns_df["month"] == month]
        for value_col in value_cols:
            for cycle in cycles:
                values = month_df[month_df["cycle_days"] == cycle]
                row[f"{value_col}_{int(cycle)}d"] = float(values[value_col].iloc[0]) if not values.empty else np.nan
        for value_col in value_cols:
            for cycle in cycles:
                raw_value = row.get(f"{value_col}_{int(cycle)}d")
                row[f"{value_col}_{int(cycle)}d_pct"] = raw_value * 100.0 if np.isfinite(raw_value) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _write_monthly_metrics_markdown(path: Path, metrics_df: pd.DataFrame) -> None:
    lines = [
        "# Monthly Equal-Weight vs Minimum-Variance Metrics Sheet",
        "",
        "- Portfolio: monthly cap25 minimum variance versus equal-weight.",
        "- Policies shown: `enter_once_then_drift` and `daily_rebalance_to_target` when both are available.",
        "- Metrics are computed on active daily rows inside each month.",
        "- Annualized metrics use 365-day crypto annualization, so very short windows can annualize aggressively.",
        "",
    ]
    for policy in _ordered_monthly_policies(metrics_df):
        policy_df = metrics_df[metrics_df["rebalance_policy"] == policy]
        for cycle in sorted(policy_df["cycle_days"].unique()):
            lines.extend(
                [
                    f"## {policy} - {int(cycle)}-Day Hold Window",
                    "",
                    "| Month | MV Return | EW Return | MV Ann.Vol | EW Ann.Vol | MV Sharpe | EW Sharpe | MV MDD | EW MDD |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            cycle_df = policy_df[policy_df["cycle_days"] == cycle]
            for month in sorted(cycle_df["month"].unique()):
                month_df = cycle_df[cycle_df["month"] == month]
                by_strategy = {item["strategy"]: item for item in month_df.to_dict(orient="records")}
                mv = by_strategy.get("minimum_variance")
                ew = by_strategy.get("equal_weight")
                if not mv or not ew:
                    continue
                lines.append(
                    "| {month} | {mv_return} | {ew_return} | {mv_vol} | {ew_vol} | {mv_sharpe} | {ew_sharpe} | {mv_mdd} | {ew_mdd} |".format(
                        month=month,
                        mv_return=_fmt_pct(mv["total_return"]),
                        ew_return=_fmt_pct(ew["total_return"]),
                        mv_vol=_fmt_pct(mv["annualized_volatility"]),
                        ew_vol=_fmt_pct(ew["annualized_volatility"]),
                        mv_sharpe=_fmt_num(mv["sharpe_ratio"]),
                        ew_sharpe=_fmt_num(ew["sharpe_ratio"]),
                        mv_mdd=_fmt_pct(mv["max_drawdown"]),
                        ew_mdd=_fmt_pct(ew["max_drawdown"]),
                    )
                )
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_monthly_returns_markdown(path: Path, returns_df: pd.DataFrame) -> None:
    lines = [
        "# Monthly Equal-Weight vs Minimum-Variance Sheet",
        "",
        "- Universe: selected 50 Upbit KRW assets.",
        "- Portfolio: monthly cap25 minimum variance versus equal-weight.",
        "- Rebalance date: first available date of each month.",
        "- Holding windows: first 7 days and first 14 days of each month.",
        "- Values below are hold-window returns, not full-month invested returns.",
        "",
        "| Month | MV 7D | EW 7D | MV-EW 7D | MV 14D | EW 14D | MV-EW 14D |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for month in sorted(returns_df["month"].unique()):
        month_df = returns_df[returns_df["month"] == month]
        by_cycle = {int(row["cycle_days"]): row for row in month_df.to_dict(orient="records")}
        cycle_7 = by_cycle.get(7, {})
        cycle_14 = by_cycle.get(14, {})
        lines.append(
            "| {month} | {mv7} | {ew7} | {diff7} | {mv14} | {ew14} | {diff14} |".format(
                month=month,
                mv7=_fmt_pct(cycle_7.get("minimum_variance_return")),
                ew7=_fmt_pct(cycle_7.get("equal_weight_return")),
                diff7=_fmt_pct(cycle_7.get("mv_minus_ew")),
                mv14=_fmt_pct(cycle_14.get("minimum_variance_return")),
                ew14=_fmt_pct(cycle_14.get("equal_weight_return")),
                diff14=_fmt_pct(cycle_14.get("mv_minus_ew")),
            )
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _default_monthly_policy(metrics_df: pd.DataFrame) -> str:
    policies = set(metrics_df["rebalance_policy"].dropna().astype(str))
    if "enter_once_then_drift" in policies:
        return "enter_once_then_drift"
    return sorted(policies)[0]


def _ordered_monthly_policies(metrics_df: pd.DataFrame) -> list[str]:
    preferred = ["enter_once_then_drift", "daily_rebalance_to_target"]
    policies = set(metrics_df["rebalance_policy"].dropna().astype(str))
    ordered = [policy for policy in preferred if policy in policies]
    ordered.extend(sorted(policies.difference(ordered)))
    return ordered


def _safe_subtract(left: float | None, right: float | None) -> float:
    if left is None or right is None or not np.isfinite(left) or not np.isfinite(right):
        return float("nan")
    return float(left - right)


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _with_policy(row: dict[str, Any], rebalance_policy: str, result: dict[str, Any]) -> dict[str, Any]:
    row["rebalance_policy"] = rebalance_policy
    row["turnover_sum"] = result["turnover_sum"]
    row["turnover_action_count"] = result["turnover_action_count"]
    return row


def _write_phase6_markdown_report(path: Path, report: dict[str, Any]) -> None:
    performance = report["performance"]
    dm_tests = report["dm_tests"]
    lines = [
        "# Phase 6 Backtest Results Note",
        "",
        f"- Period: {report['first_date']} to {report['last_date']} ({report['n_backtest_days']} days)",
        f"- Evaluation frequency: {report['params']['eval_freq_min']} minutes",
        "- Transaction costs: ignored",
        "- Hold-period weight drift: enabled",
        "- Primary comparison: minimum variance versus equal-weight on the same 50-asset universe.",
        "- BTC HODL is retained only as a market reference, not the main benchmark for this strategy design.",
        "",
        "## Performance",
        "",
        "| Strategy | Policy | Cycle | Total Return | Ann. Return | Ann. Vol | Sharpe | MDD | IR vs BTC | Turnover Mean | Turnover Sum | Realized Risk |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in performance:
        lines.append(_performance_markdown_row(row))
    lines.extend(
        [
            "",
            "## DM Test",
            "",
            "| Comparison | Policy | Cycle | Loss | Mean Loss Diff | DM Stat | p-value |",
            "|---|---|---:|---|---:|---:|---:|",
        ]
    )
    for row in dm_tests:
        lines.append(
            "| {comparison} | {rebalance_policy} | {cycle_days} | {loss} | {mean_loss_diff:.6g} | {dm_stat:.4g} | {p_value:.4g} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The defensible benchmark framing is minimum variance versus equal-weight, because both use the same selected 50-asset universe and rebalance cycle.",
            "- BTC HODL is useful context, but it answers a different question: whether the active multi-asset strategy beats passive BTC exposure.",
            "- Minimum variance portfolios should be judged first on realized risk, drawdown, and loss reduction versus equal-weight.",
            "- Concentration remains a key diagnostic; uncapped runs reached above 90% top weight, so capped variants should be compared before final model-portfolio framing.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _performance_markdown_row(row: dict[str, Any]) -> str:
    cycle = "" if row["cycle_days"] is None or not np.isfinite(row["cycle_days"]) else int(row["cycle_days"])
    return (
        f"| {row['strategy']} | {row['rebalance_policy']} | {cycle} | {_fmt_pct(row['total_return'])} | "
        f"{_fmt_pct(row['annualized_return'])} | {_fmt_pct(row['annualized_volatility'])} | "
        f"{_fmt_num(row['sharpe_ratio'])} | {_fmt_pct(row['max_drawdown'])} | "
        f"{_fmt_num(row['information_ratio_vs_btc'])} | {_fmt_num(row['turnover_mean'])} | "
        f"{_fmt_num(row['turnover_sum'])} | "
        f"{_fmt_pct(row['realized_risk_annualized_mean'])} |"
    )


def _fmt_pct(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.2%}"


def _fmt_num(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.4f}"


def _load_phase5_portfolios(portfolio_path: Path) -> dict[str, Any]:
    data = np.load(portfolio_path, allow_pickle=False)
    if "tickers" not in data.files:
        raise ValueError("Phase 5 portfolio npz must include tickers")
    tickers = np.asarray(data["tickers"], dtype="U")
    cycles: dict[int, dict[str, np.ndarray]] = {}
    for key in data.files:
        if key.startswith("cycle_") and key.endswith("_weights") and "scheduled" not in key:
            cycle = int(key.split("_")[1])
            dates_key = f"cycle_{cycle}_rebalance_dates"
            if dates_key not in data.files:
                raise ValueError(f"Missing {dates_key} in Phase 5 portfolio npz")
            weights = np.asarray(data[key], dtype=np.float64)
            if weights.ndim != 2 or weights.shape[1] != len(tickers):
                raise ValueError(f"Invalid weights shape for cycle={cycle}: {weights.shape}")
            cycles[cycle] = {
                "rebalance_dates": np.asarray(data[dates_key], dtype="U10"),
                "weights": weights,
            }
            scheduled_dates_key = f"cycle_{cycle}_scheduled_dates"
            scheduled_rebalance_dates_key = f"cycle_{cycle}_scheduled_rebalance_dates"
            scheduled_weights_key = f"cycle_{cycle}_scheduled_weights"
            if {scheduled_dates_key, scheduled_rebalance_dates_key, scheduled_weights_key}.issubset(data.files):
                cycles[cycle]["scheduled_dates"] = np.asarray(data[scheduled_dates_key], dtype="U10")
                cycles[cycle]["scheduled_rebalance_dates"] = np.asarray(
                    data[scheduled_rebalance_dates_key],
                    dtype="U10",
                )
                cycles[cycle]["scheduled_weights"] = np.asarray(data[scheduled_weights_key], dtype=np.float64)
    if not cycles:
        raise ValueError("No cycle weights found in Phase 5 portfolio npz")
    return {"tickers": tickers, "cycles": dict(sorted(cycles.items()))}


def _required_dates(portfolios: dict[str, Any]) -> np.ndarray:
    dates: list[str] = []
    for cycle_data in portfolios["cycles"].values():
        dates.extend(map(str, cycle_data["rebalance_dates"]))
    return np.asarray(sorted(set(dates)), dtype="U10")


def _load_eval_returns(
    price_panel_path: Path,
    tickers: np.ndarray,
    eval_freq_min: int,
    required_dates: np.ndarray,
) -> dict[str, Any]:
    if eval_freq_min < 1 or 1440 % eval_freq_min != 0:
        raise ValueError("eval_freq_min must divide 1440")
    usecols = ["timestamp", *map(str, tickers)]
    frame = pd.read_csv(price_panel_path, usecols=usecols, encoding="utf-8-sig")
    timestamps = pd.to_datetime(frame["timestamp"])
    prices = frame[list(tickers)].to_numpy(dtype=np.float64)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Price panel contains non-finite or non-positive prices")

    minute_of_day = timestamps.dt.hour.to_numpy() * 60 + timestamps.dt.minute.to_numpy()
    cut_minute = 9 * 60
    aligned = ((minute_of_day - cut_minute) % eval_freq_min) == 0
    aligned_timestamps = timestamps[aligned].reset_index(drop=True)
    aligned_log_prices = np.log(prices[aligned])
    if len(aligned_timestamps) < 2:
        raise ValueError("Not enough aligned price rows to compute returns")

    start_timestamps = aligned_timestamps.iloc[:-1]
    end_timestamps = aligned_timestamps.iloc[1:]
    deltas = (end_timestamps.to_numpy() - start_timestamps.to_numpy()).astype("timedelta64[m]").astype(int)
    valid_interval = deltas == eval_freq_min
    log_returns = np.diff(aligned_log_prices, axis=0)[valid_interval]
    simple_returns = np.expm1(log_returns)
    interval_end_timestamps = end_timestamps[valid_interval].reset_index(drop=True)
    trading_dates = (start_timestamps[valid_interval] - pd.Timedelta(hours=9)).dt.date.astype(str).to_numpy(dtype="U10")

    min_date = str(sorted(map(str, required_dates))[0]) if len(required_dates) else config.ANALYSIS_START
    unique_dates = np.asarray(sorted(set(map(str, trading_dates))), dtype="U10")
    dates = unique_dates[unique_dates >= config.ANALYSIS_START]
    if len(required_dates):
        dates = dates[dates >= min_date]
    if len(dates) == 0:
        raise ValueError("No backtest dates found in price panel")

    intervals_per_day = 1440 // eval_freq_min
    selected_returns: list[np.ndarray] = []
    selected_end_timestamps: list[np.ndarray] = []
    selected_dates: list[str] = []
    interval_counts: list[int] = []
    for date in dates:
        mask = trading_dates == date
        count = int(np.sum(mask))
        if count != intervals_per_day:
            continue
        selected_dates.append(str(date))
        selected_returns.append(simple_returns[mask])
        selected_end_timestamps.append(interval_end_timestamps[mask].astype(str).to_numpy())
        interval_counts.append(count)

    if not selected_dates:
        raise ValueError("No full backtest days with the expected interval count")
    return {
        "dates": np.asarray(selected_dates, dtype="U10"),
        "simple_returns": np.vstack(selected_returns),
        "interval_end_timestamps": np.concatenate(selected_end_timestamps),
        "interval_counts": np.asarray(interval_counts, dtype=int),
    }


def _run_strategy(
    strategy: str,
    rebalance_policy: str,
    cycle_days: int | None,
    tickers: np.ndarray,
    dates: np.ndarray,
    interval_counts: np.ndarray,
    simple_returns: np.ndarray,
    rebalance_dates: np.ndarray,
    target_weights: np.ndarray,
    scheduled_dates: np.ndarray | None = None,
    scheduled_rebalance_dates: np.ndarray | None = None,
    scheduled_weights: np.ndarray | None = None,
) -> dict[str, Any]:
    if rebalance_policy not in {"buy_and_hold", "enter_once_then_drift", "daily_rebalance_to_target"}:
        raise ValueError(f"Unsupported rebalance_policy={rebalance_policy}")
    date_to_weight = {str(date): target_weights[idx] for idx, date in enumerate(rebalance_dates)}
    scheduled_by_date: dict[str, tuple[str, np.ndarray]] | None = None
    if scheduled_dates is not None and scheduled_rebalance_dates is not None and scheduled_weights is not None:
        scheduled_by_date = {
            str(date): (str(scheduled_rebalance_dates[idx]), np.asarray(scheduled_weights[idx], dtype=np.float64))
            for idx, date in enumerate(scheduled_dates)
        }
    current_weights: np.ndarray | None = None
    equity = 1.0
    daily_rows: list[dict[str, Any]] = []
    interval_rows: list[dict[str, Any]] = []
    turnovers: list[float] = []
    offset = 0

    for date_idx, date in enumerate(dates):
        date_str = str(date)
        is_scheduled = scheduled_by_date is None or date_str in scheduled_by_date
        if not is_scheduled:
            interval_count = int(interval_counts[date_idx])
            offset += interval_count
            current_weights = None
            daily_rows.append(
                {
                    "strategy": strategy,
                    "rebalance_policy": rebalance_policy,
                    "cycle_days": cycle_days,
                    "date": date_str,
                    "daily_return": 0.0,
                    "equity": equity,
                    "realized_risk_annualized": 0.0,
                    "ending_top_ticker": "CASH",
                    "ending_top_weight": 1.0,
                    "ending_active_count": 0,
                }
            )
            continue

        if scheduled_by_date is not None:
            scheduled_rebalance_date, scheduled_weight = scheduled_by_date[date_str]
            should_rebalance = (
                current_weights is None
                or date_str == scheduled_rebalance_date
                or rebalance_policy == "daily_rebalance_to_target"
            )
            if should_rebalance:
                turnover = float(np.sum(np.abs(scheduled_weight - current_weights))) if current_weights is not None else 1.0
                current_weights = scheduled_weight.copy()
                turnovers.append(turnover)
        elif current_weights is None:
            current_weights = _initial_weights(date_str, rebalance_dates, target_weights)
            turnovers.append(0.0)
        if scheduled_by_date is None and date_str in date_to_weight:
            next_weights = np.asarray(date_to_weight[date_str], dtype=np.float64)
            turnover = float(np.sum(np.abs(next_weights - current_weights)))
            current_weights = next_weights.copy()
            turnovers.append(turnover)

        start_equity = equity
        daily_log_returns: list[float] = []
        interval_count = int(interval_counts[date_idx])
        day_returns = simple_returns[offset : offset + interval_count]
        offset += interval_count

        if scheduled_by_date is None and rebalance_policy == "daily_rebalance_to_target" and current_weights is not None:
            next_weights = _initial_weights(date_str, rebalance_dates, target_weights)
            turnover = float(np.sum(np.abs(next_weights - current_weights)))
            current_weights = next_weights.copy()
            turnovers.append(turnover)

        for interval_idx, asset_return in enumerate(day_returns, start=1):
            portfolio_return = float(np.dot(current_weights, asset_return))
            equity *= 1.0 + portfolio_return
            daily_log_returns.append(float(np.log1p(portfolio_return)))
            denominator = 1.0 + portfolio_return
            if denominator <= 0:
                raise FloatingPointError(f"{strategy} portfolio equity became non-positive on {date_str}")
            current_weights = current_weights * (1.0 + asset_return) / denominator
            interval_rows.append(
                {
                    "strategy": strategy,
                    "rebalance_policy": rebalance_policy,
                    "cycle_days": cycle_days,
                    "date": date_str,
                    "interval": interval_idx,
                    "return": portfolio_return,
                    "equity": equity,
                }
            )

        daily_return = equity / start_equity - 1.0
        realized_risk = float(np.sqrt(np.sum(np.square(daily_log_returns))) * np.sqrt(config.ANNUALIZATION))
        top_idx = int(np.argmax(current_weights))
        daily_rows.append(
            {
                "strategy": strategy,
                "rebalance_policy": rebalance_policy,
                "cycle_days": cycle_days,
                "date": date_str,
                "daily_return": daily_return,
                "equity": equity,
                "realized_risk_annualized": realized_risk,
                "ending_top_ticker": str(tickers[top_idx]),
                "ending_top_weight": float(current_weights[top_idx]),
                "ending_active_count": int(np.sum(current_weights > 1e-6)),
            }
        )

    daily_df = pd.DataFrame(daily_rows)
    interval_df = pd.DataFrame(interval_rows)
    return {
        "daily": daily_df,
        "interval": interval_df,
        "turnover_mean": float(np.mean(turnovers[1:])) if len(turnovers) > 1 else 0.0,
        "turnover_sum": float(np.sum(turnovers)),
        "turnover_action_count": int(len(turnovers)),
    }


def _initial_weights(date: str, rebalance_dates: np.ndarray, target_weights: np.ndarray) -> np.ndarray:
    matches = np.where(rebalance_dates <= date)[0]
    if len(matches):
        return target_weights[int(matches[-1])].copy()
    return target_weights[0].copy()


def _equal_weight_scheduled_weights(cycle_data: dict[str, np.ndarray], n_assets: int) -> np.ndarray | None:
    if "scheduled_dates" not in cycle_data:
        return None
    return np.full((len(cycle_data["scheduled_dates"]), n_assets), 1.0 / n_assets, dtype=np.float64)
