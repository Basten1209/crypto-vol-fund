#!/usr/bin/env python3
"""Run the Phase 3-6 demo pipeline with visible terminal progress."""

from __future__ import annotations

import argparse
import html
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "processed" / "demo_phase3_to_6"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Phase 3 PRVM, Phase 4 EWMA, Phase 5 portfolio optimization, "
            "and Phase 6 backtesting in one demo command."
        )
    )
    parser.add_argument(
        "--price-panel",
        type=Path,
        default=ROOT / "price_panel.csv",
        help="Input wide price panel CSV. Default: ./price_panel.csv",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Root directory for demo artifacts. Default: {DEFAULT_OUTPUT_ROOT.relative_to(ROOT)}",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run directory name. Default: timestamp such as 20260527_213000.",
    )
    parser.add_argument(
        "--variant",
        choices=["all", "baseline", "monthly-cap25"],
        default="all",
        help="Portfolio/backtest variants to run. Default: all. Baseline is monthly uncapped.",
    )
    parser.add_argument(
        "--workers",
        default="auto",
        help="Phase 3 worker count, or 'auto'. Default: auto.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="Phase 3 CSV rows per chunk. Default: 100000.",
    )
    parser.add_argument(
        "--smoke-days",
        type=int,
        default=None,
        help="Limit Phase 3 to the first N full days for a fast rehearsal.",
    )
    parser.add_argument(
        "--write-long-csv",
        action="store_true",
        help="Write Phase 3 prvm_long.csv and jv_long.csv. Default: skip for faster demos.",
    )
    parser.add_argument(
        "--no-open-artifacts",
        action="store_true",
        help="Do not open generated HTML artifacts in the browser/Finder.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    price_panel = args.price_panel.expanduser().resolve()
    if not price_panel.exists():
        raise FileNotFoundError(
            f"{price_panel} does not exist. Pass --price-panel with the gitignored Phase 1 price_panel.csv."
        )
    if args.smoke_days is not None and args.smoke_days < 29:
        raise ValueError("--smoke-days must be at least 29 so Phase 4 has a 28-day EWMA rolling window")

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_root.expanduser()
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    run_dir = run_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    phase3_dir = run_dir / "phase3"
    phase4_dir = run_dir / "phase4"
    phase4_comparison_dir = run_dir / "phase4_comparison"
    variants = _selected_variants(args.variant)
    open_artifacts = not args.no_open_artifacts

    print("=== Demo Phase 3-6 pipeline ===", flush=True)
    print(f"repo: {ROOT}", flush=True)
    print(f"price_panel: {price_panel}", flush=True)
    print(f"output_run_dir: {_display_path(run_dir)}", flush=True)
    print(f"variant: {args.variant}", flush=True)
    if args.smoke_days is not None:
        print(f"smoke_days: {args.smoke_days}", flush=True)

    phase3_cmd = [
        sys.executable,
        "-u",
        "scripts/run_phase3_prvm.py",
        "--input",
        str(price_panel),
        "--output-dir",
        str(phase3_dir),
        "--workers",
        str(args.workers),
        "--chunk-size",
        str(args.chunk_size),
    ]
    if args.smoke_days is not None:
        phase3_cmd.extend(["--limit-days", str(args.smoke_days)])
    if not args.write_long_csv:
        phase3_cmd.append("--no-long-csv")
    _run_step("Phase 3 PRVM calculation", phase3_cmd)
    phase3_artifact = _build_phase3_artifacts(phase3_dir)
    _artifact_event("Phase 3 PRVM/JV matrices", phase3_artifact, open_artifacts)

    phase4_cmd = [
        sys.executable,
        "-u",
        "scripts/run_phase4_ewma.py",
        "--input",
        str(phase3_dir / "prvm_results.npz"),
        "--output-dir",
        str(phase4_dir),
    ]
    _run_step("Phase 4 EWMA forecasting", phase4_cmd)
    phase4_comparison_cmd = [
        sys.executable,
        "-u",
        "scripts/run_phase4_ewma_comparison.py",
        "--input",
        str(phase3_dir / "prvm_results.npz"),
        "--output-dir",
        str(phase4_comparison_dir),
    ]
    _run_step("Phase 4 EWMA jump-adjusted vs raw comparison", phase4_comparison_cmd)
    phase4_artifact = _build_phase4_artifacts(phase4_dir, phase4_comparison_dir)
    _artifact_event("Phase 4 EWMA forecast comparison", phase4_artifact, open_artifacts)

    for variant in variants:
        phase5_dir = run_dir / f"phase5_{variant['name']}"
        phase6_dir = run_dir / f"phase6_{variant['name']}"

        phase5_cmd = [
            sys.executable,
            "-u",
            "scripts/run_phase5_portfolio.py",
            "--forecast-input",
            str(phase4_dir / "ewma_forecasts.npz"),
            "--prvm-input",
            str(phase3_dir / "prvm_results.npz"),
            "--output-dir",
            str(phase5_dir),
            "--rebalance-frequency",
            variant["rebalance_frequency"],
        ]
        if variant["single_asset_cap"] is not None:
            phase5_cmd.extend(["--single-asset-cap", str(variant["single_asset_cap"])])
        _run_step(f"Phase 5 portfolio optimization ({variant['label']})", phase5_cmd)
        phase5_artifact = _build_phase5_artifacts(phase5_dir, variant["label"])
        _artifact_event(f"Phase 5 portfolio optimization ({variant['label']})", phase5_artifact, open_artifacts)

        phase6_cmd = [
            sys.executable,
            "-u",
            "scripts/run_phase6_backtest.py",
            "--price-panel",
            str(price_panel),
            "--portfolio-input",
            str(phase5_dir / "minimum_variance_portfolios.npz"),
            "--output-dir",
            str(phase6_dir),
        ]
        _run_step(f"Phase 6 backtest ({variant['label']})", phase6_cmd)
        phase6_artifact = _build_phase6_artifacts(phase6_dir, variant["label"])
        _artifact_event(f"Phase 6 backtest ({variant['label']})", phase6_artifact, open_artifacts)

    index_path = _build_demo_index(run_dir, variants)
    print("=== Demo pipeline complete ===", flush=True)
    print(f"Artifacts saved under: {_display_path(run_dir)}", flush=True)
    print(f"Demo index: {_display_path(index_path)}", flush=True)
    if open_artifacts:
        _open_path(index_path)
    for variant in variants:
        perf = run_dir / f"phase6_{variant['name']}" / "phase6_performance_table.csv"
        print(f"{variant['label']} performance table: {_display_path(perf)}", flush=True)
    return 0


def _selected_variants(selection: str) -> list[dict[str, str | float | None]]:
    all_variants: list[dict[str, str | float | None]] = [
        {
            "name": "baseline",
            "label": "baseline uncapped monthly",
            "rebalance_frequency": "monthly",
            "single_asset_cap": None,
        },
        {
            "name": "monthly_cap25",
            "label": "monthly 25% cap",
            "rebalance_frequency": "monthly",
            "single_asset_cap": 0.25,
        },
    ]
    if selection == "all":
        return all_variants
    if selection == "baseline":
        return [all_variants[0]]
    return [all_variants[1]]


def _run_step(label: str, cmd: list[str]) -> None:
    print("", flush=True)
    print(f"--- {label} ---", flush=True)
    print("$ " + " ".join(_quote_for_display(part) for part in cmd), flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    started = datetime.now()
    completed = subprocess.run(cmd, cwd=ROOT, env=env, check=False)
    elapsed = datetime.now() - started
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, cmd)
    print(f"--- completed {label} in {elapsed} ---", flush=True)


def _artifact_event(label: str, path: Path, open_artifacts: bool) -> None:
    print("", flush=True)
    print(f">>> Result event: {label}", flush=True)
    print(f"    artifact: {_display_path(path)}", flush=True)
    if open_artifacts:
        _open_path(path)
        print("    opened in macOS default viewer", flush=True)


def _open_path(path: Path) -> None:
    try:
        subprocess.run(["open", str(path)], cwd=ROOT, check=False)
    except OSError as exc:
        print(f"    open skipped: {exc}", flush=True)


def _build_phase3_artifacts(phase3_dir: Path) -> Path:
    npz_path = phase3_dir / "prvm_results.npz"
    report_path = phase3_dir / "phase3_prvm_report.json"
    summary_path = phase3_dir / "phase3_daily_summary.csv"
    data = np.load(npz_path, allow_pickle=False)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    summary = pd.read_csv(summary_path)
    dates = data["dates"].astype(str)
    tickers = data["tickers"].astype(str)
    prvm = data["prvm"]
    jv = data["jv"]
    latest_idx = len(dates) - 1
    latest_date = str(dates[latest_idx])

    latest_prvm_csv = phase3_dir / "demo_latest_prvm_matrix.csv"
    latest_jv_csv = phase3_dir / "demo_latest_jv_matrix.csv"
    pd.DataFrame(prvm[latest_idx], index=tickers, columns=tickers).to_csv(latest_prvm_csv)
    pd.DataFrame(jv[latest_idx], index=tickers, columns=tickers).to_csv(latest_jv_csv)

    top_jump = summary.sort_values("trace_jv", ascending=False).head(12)
    top_jump_csv = phase3_dir / "demo_top_jump_days.csv"
    top_jump.to_csv(top_jump_csv, index=False)

    html_path = phase3_dir / "demo_phase3_prvm_jv.html"
    body = f"""
    <h1>Phase 3: PRVM / Jump Volatility</h1>
    <p class="note">This phase calculates daily jump-adjusted PRVM, raw PRVM, and JV matrices from 1-minute prices. It is a vectorized matrix calculation, not model training.</p>
    <section class="cards">
      {_card("Full days", report["n_days"])}
      {_card("Date range", f"{report['first_date']} to {report['last_date']}")}
      {_card("Assets", report["n_assets"])}
      {_card("K / num_k", f"{report['params']['k']} / {report['params']['num_k']}")}
    </section>
    <h2>Latest Daily PRVM Matrix: {html.escape(latest_date)}</h2>
    <p class="section-desc">가장 마지막 계산일의 jump-adjusted PRVM 행렬입니다. 각 셀은 두 자산 사이의 일별 고빈도 공분산 추정값을 의미하며, 포트폴리오 최적화에서 위험 구조를 설명하는 핵심 입력입니다.</p>
    <p>CSV: <code>{_display_path(latest_prvm_csv)}</code></p>
    {_matrix_heatmap(prvm[latest_idx], tickers, max_size=18)}
    <h2>Latest JV Matrix: {html.escape(latest_date)}</h2>
    <p class="section-desc">같은 날짜의 Jump Volatility 행렬입니다. raw PRVM에서 jump-adjusted PRVM을 뺀 구성요소로, 갑작스러운 가격 점프가 자산별/자산간 위험에 얼마나 반영되는지 보여줍니다.</p>
    <p>CSV: <code>{_display_path(latest_jv_csv)}</code></p>
    {_matrix_heatmap(jv[latest_idx], tickers, max_size=18)}
    <h2>Largest Jump-Volatility Days</h2>
    <p class="section-desc">전체 기간 중 JV trace가 큰 날짜들입니다. 시장 충격이나 급격한 변동이 컸던 날을 찾고, jump truncation이 실제로 어떤 날짜에서 중요했는지 확인하기 위한 진단 표입니다.</p>
    <p>CSV: <code>{_display_path(top_jump_csv)}</code></p>
    {_df_table(top_jump[["date", "trace_jv", "jump_trace_ratio", "trace_prvm", "trace_raw_prvm"]])}
    """
    _write_html(html_path, "Phase 3 PRVM / JV", body)
    return html_path


def _build_phase4_artifacts(phase4_dir: Path, comparison_dir: Path) -> Path:
    report = json.loads((phase4_dir / "phase4_ewma_report.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(phase4_dir / "ewma_metrics.csv")
    comparison = json.loads((comparison_dir / "phase4_ewma_comparison_report.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(comparison_dir / "phase4_ewma_comparison_summary.csv")
    daily = pd.read_csv(comparison_dir / "phase4_ewma_comparison_daily.csv")
    top_mspe_delta = daily.reindex(daily["mspe_delta_raw_minus_adjusted"].abs().sort_values(ascending=False).index).head(12)

    html_path = phase4_dir / "demo_phase4_ewma_comparison.html"
    body = f"""
    <h1>Phase 4: EWMA Forecast / Jump Adjustment Comparison</h1>
    <p class="note">Both models are evaluated against the same next-day jump-adjusted PRVM target. Positive raw-minus-adjusted deltas mean jump-adjusted EWMA has lower loss.</p>
    <section class="cards">
      {_card("Forecast days", report["n_forecast_days"])}
      {_card("Target range", f"{report['first_target_date']} to {report['last_target_date']}")}
      {_card("Lambda", report["params"]["lambda"])}
      {_card("Rolling window", f"{report['params']['window_days']} days")}
    </section>
    <h2>Mean Loss Table</h2>
    <p class="section-desc">직전 28일 rolling window에 EWMA 가중치를 적용해 만든 예측값을 비교합니다. jump-adjusted PRVM을 입력으로 쓴 모델과 raw PRVM을 입력으로 쓴 모델의 평균 예측 손실을 보여주며, MSPE는 행렬 예측 오차 크기, QLIKE는 공분산 예측의 likelihood 기반 손실입니다.</p>
    <p>CSV: <code>{_display_path(comparison_dir / "phase4_ewma_comparison_summary.csv")}</code></p>
    {_df_table(summary)}
    <h2>Raw minus Adjusted Delta</h2>
    <p class="section-desc">raw PRVM EWMA 손실에서 jump-adjusted PRVM EWMA 손실을 뺀 값입니다. 양수이면 jump-adjusted 방식이 더 낮은 손실을 냈다는 뜻이고, 음수이면 raw 방식이 해당 지표에서 더 낮은 손실을 냈다는 뜻입니다.</p>
    <section class="cards">
      {_card("MSPE delta x1e4", f"{comparison['comparison']['mspe_delta_raw_minus_adjusted_x1e4']:.6g}")}
      {_card("QLIKE delta x1e-3", f"{comparison['comparison']['qlike_delta_raw_minus_adjusted_x1e_minus3']:.6g}")}
      {_card("Adjusted lower MSPE days", comparison["comparison"]["days_adjusted_lower_mspe"])}
      {_card("Adjusted lower QLIKE days", comparison["comparison"]["days_adjusted_lower_qlike"])}
    </section>
    <h2>Largest Daily MSPE Differences</h2>
    <p class="section-desc">날짜별 MSPE 차이가 크게 나타난 사례입니다. jump adjustment가 특정 날짜의 forecast 품질을 얼마나 바꾸는지 보여주며, 극단적인 변동일에서 두 입력 행렬의 차이를 확인하는 데 사용합니다.</p>
    <p>Daily CSV: <code>{_display_path(comparison_dir / "phase4_ewma_comparison_daily.csv")}</code></p>
    {_df_table(top_mspe_delta[["target_date", "adjusted_mspe", "raw_mspe", "mspe_delta_raw_minus_adjusted", "adjusted_qlike", "raw_qlike"]])}
    <h2>Baseline EWMA Metric Sample</h2>
    <p class="section-desc">기본 EWMA 실행에서 생성된 날짜별 MSPE/QLIKE 일부입니다. 전체 시계열은 CSV에 저장되며, 예측 성능이 시간에 따라 어떻게 변하는지 확인하는 원자료입니다.</p>
    {_df_table(metrics.head(12))}
    """
    _write_html(html_path, "Phase 4 EWMA Comparison", body)
    return html_path


def _build_phase5_artifacts(phase5_dir: Path, label: str) -> Path:
    report = json.loads((phase5_dir / "phase5_portfolio_report.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(phase5_dir / "phase5_portfolio_summary.csv")
    weights = pd.read_csv(phase5_dir / "minimum_variance_weights_wide.csv")
    latest = weights.sort_values(["cycle_days", "rebalance_date"]).groupby("cycle_days", as_index=False).tail(1)
    rows = []
    ticker_cols = [col for col in weights.columns if col not in {"cycle_days", "rebalance_date"}]
    for _, row in latest.iterrows():
        top = row[ticker_cols].sort_values(ascending=False).head(10)
        for ticker, weight in top.items():
            rows.append(
                {
                    "cycle_days": int(row["cycle_days"]),
                    "rebalance_date": row["rebalance_date"],
                    "ticker": ticker,
                    "weight": float(weight),
                }
            )
    top_weights = pd.DataFrame(rows)
    top_weights_csv = phase5_dir / "demo_latest_top_weights.csv"
    top_weights.to_csv(top_weights_csv, index=False)

    cycle_summary = pd.DataFrame(
        [{"cycle_days": cycle, **values} for cycle, values in report["cycle_summary"].items()]
    )
    html_path = phase5_dir / "demo_phase5_portfolio.html"
    body = f"""
    <h1>Phase 5: Portfolio Optimization - {html.escape(str(label))}</h1>
    <p class="note">This phase solves long-only minimum variance weights using EWMA covariance plus lagged JV.</p>
    <section class="cards">
      {_card("Rebalance schedule", report["params"]["rebalance_frequency"])}
      {_card("Single asset cap", "none" if report["params"]["single_asset_cap"] is None else f"{report['params']['single_asset_cap']:.0%}")}
      {_card("Gross exposure", report["params"]["gross_exposure"])}
      {_card("Min asset weight", f"{report['params']['min_asset_weight']:.3%}")}
    </section>
    <p class="section-desc">Phase 5는 매월 첫 available date의 목표 비중(target weight)을 계산하는 단계입니다. 이 목표 비중을 보유기간 중 어떻게 운용할지는 Phase 6의 Simple Mode와 Managed Mode에서 평가합니다.</p>
    <h2>Cycle Summary</h2>
    <p class="section-desc">7일/14일 hold window별 최적화 요약입니다. 리밸런싱 날짜는 매월 첫 available date이며, 각 cycle은 같은 월간 목표 비중을 7일 또는 14일 동안 운용했을 때의 후보 포트폴리오를 뜻합니다. 리밸런싱 횟수, 평균 편입 종목 수, 최고 비중, turnover, KKT violation을 통해 포트폴리오가 얼마나 집중되었고 최적화가 안정적으로 풀렸는지 확인합니다.</p>
    {_df_table(cycle_summary[["cycle_days", "n_rebalances", "active_count_mean", "top_weight_mean", "top_weight_max", "turnover_mean", "kkt_violation_max"]])}
    <h2>Latest Rebalance Top Weights</h2>
    <p class="section-desc">각 hold window에서 가장 최근 월간 리밸런싱 시점의 상위 편입 비중입니다. 데모에서는 모델이 실제로 어떤 코인에 얼마만큼 배분했는지를 직관적으로 보여주는 핵심 화면입니다.</p>
    <p>CSV: <code>{_display_path(top_weights_csv)}</code></p>
    {_df_table(top_weights)}
    <h2>Rebalance Diagnostics Sample</h2>
    <p class="section-desc">월간 리밸런싱 날짜별 최적화 진단 샘플입니다. active_count, top_ticker, top_weight, turnover, solver_status를 통해 포트폴리오 생성 과정이 정상적으로 진행되었는지 확인합니다. Simple/Managed의 일별 운용 차이는 이 단계가 아니라 Phase 6 backtest에서 적용됩니다.</p>
    {_df_table(summary.head(12)[["cycle_days", "rebalance_date", "active_count", "top_ticker", "top_weight", "turnover_from_prev", "solver_status"]])}
    """
    _write_html(html_path, "Phase 5 Portfolio", body)
    return html_path


def _build_phase6_artifacts(phase6_dir: Path, label: str) -> Path:
    report = json.loads((phase6_dir / "phase6_backtest_report.json").read_text(encoding="utf-8"))
    performance = pd.read_csv(phase6_dir / "phase6_performance_table.csv")
    dm = pd.read_csv(phase6_dir / "phase6_dm_test.csv")
    monthly_path = phase6_dir / "monthly_hold_window_returns.csv"
    monthly = pd.read_csv(monthly_path) if monthly_path.exists() else pd.DataFrame()
    policy_table = pd.DataFrame(
        [
            {
                "mode": "Simple Mode",
                "rebalance_policy": "enter_once_then_drift",
                "meaning": "월초 목표 비중으로 한 번 진입한 뒤, 보유기간 동안 추가 리밸런싱 없이 가격 변화에 따른 weight drift를 허용합니다.",
            },
            {
                "mode": "Managed Mode",
                "rebalance_policy": "daily_rebalance_to_target",
                "meaning": "보유기간 동안 매일 목표 비중으로 다시 맞춥니다. 즉, weight drift를 줄이기 위해 일 단위 리밸런싱을 반복하는 운용 방식입니다.",
            },
        ]
    )

    html_path = phase6_dir / "demo_phase6_backtest.html"
    body = f"""
    <h1>Phase 6: Backtest - {html.escape(str(label))}</h1>
    <p class="note">This phase compares minimum variance, equal-weight, and BTC HODL on 10-minute evaluation returns.</p>
    <section class="cards">
      {_card("Backtest days", report["n_backtest_days"])}
      {_card("Date range", f"{report['first_date']} to {report['last_date']}")}
      {_card("Eval freq", f"{report['params']['eval_freq_min']} min")}
      {_card("Annualization", report["params"]["annualization"])}
    </section>
    <h2>Simple vs Managed Mode</h2>
    <p class="section-desc">Phase 5에서 계산한 월간 목표 비중을 실제 보유기간에 적용하는 두 가지 운용 방식입니다. Simple Mode는 최초 진입 후 비중 변화를 그대로 두고, Managed Mode는 매일 목표 비중으로 되돌리는 방식입니다. 따라서 같은 Phase 5 weight라도 Phase 6 성과표에는 정책별 결과가 따로 나타납니다.</p>
    {_df_table(policy_table)}
    <h2>Performance Table</h2>
    <p class="section-desc">minimum variance, equal-weight, BTC HODL의 성과 비교표입니다. 누적수익률, 연환산 수익률/변동성, Sharpe, MDD, turnover 등을 통해 최적화 포트폴리오가 benchmark 대비 어떤 특성을 보였는지 평가합니다.</p>
    <p>CSV: <code>{_display_path(phase6_dir / "phase6_performance_table.csv")}</code></p>
    {_df_table(performance)}
    <h2>Diebold-Mariano Test</h2>
    <p class="section-desc">minimum variance와 equal-weight의 손실 차이가 통계적으로 의미 있는지 보는 검정 결과입니다. 여기서는 squared daily return loss를 사용해 위험 감소 관점에서 두 전략을 비교합니다.</p>
    {_df_table(dm)}
    <h2>Monthly Hold Window Returns Sample</h2>
    <p class="section-desc">월별 hold window 안에서 전략별 수익률을 비교한 샘플입니다. 전체 기간을 한 번에 보는 성과표와 달리, 특정 월/보유기간에서 전략이 어떻게 작동했는지 설명할 때 사용합니다.</p>
    {_df_table(monthly.head(16)) if not monthly.empty else "<p>No monthly hold-window output.</p>"}
    """
    _write_html(html_path, "Phase 6 Backtest", body)
    return html_path


def _build_demo_index(run_dir: Path, variants: list[dict[str, str | float | None]]) -> Path:
    links = [
        ("Phase 3 PRVM / JV", run_dir / "phase3" / "demo_phase3_prvm_jv.html"),
        ("Phase 4 EWMA Comparison", run_dir / "phase4" / "demo_phase4_ewma_comparison.html"),
    ]
    for variant in variants:
        links.append(
            (
                f"Phase 5 Portfolio - {variant['label']}",
                run_dir / f"phase5_{variant['name']}" / "demo_phase5_portfolio.html",
            )
        )
        links.append(
            (
                f"Phase 6 Backtest - {variant['label']}",
                run_dir / f"phase6_{variant['name']}" / "demo_phase6_backtest.html",
            )
        )
    link_html = "\n".join(
        f"<li><a href='{html.escape(path.relative_to(run_dir).as_posix())}'>{html.escape(label)}</a></li>"
        for label, path in links
    )
    body = f"""
    <h1>Phase 3-6 Demo Result Index</h1>
    <p class="note">The pipeline does not train a machine-learning model. It computes PRVM/JV matrices, EWMA covariance forecasts, deterministic minimum-variance weights, and walk-forward backtests.</p>
    <h2>Result Pages</h2>
    <p class="section-desc">Phase별 결과 HTML로 이동하는 목차입니다. 데모 녹화 중에는 이 페이지에서 각 산출물을 순서대로 열어 계산 흐름과 결과를 설명할 수 있습니다.</p>
    <ul class="links">{link_html}</ul>
    <h2>Why the Run Can Finish Quickly</h2>
    <p class="section-desc">실행 시간이 짧게 나오는 이유를 설명하는 섹션입니다. 이 프로젝트는 머신러닝 학습 루프가 아니라 벡터화된 행렬 계산과 결정론적 최적화가 중심이므로, 전체 파이프라인이 생각보다 빠르게 끝날 수 있습니다.</p>
    <p>Most heavy work is vectorized NumPy linear algebra on 50 x 50 daily matrices. The wrapper also skips Phase 3 long-format matrix CSVs by default because those are large and are not needed downstream.</p>
    """
    index_path = run_dir / "demo_index.html"
    _write_html(index_path, "Phase 3-6 Demo Index", body)
    return index_path


def _write_html(path: Path, title: str, body: str) -> None:
    path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 28px; color: #172026; }}
    h1 {{ margin-bottom: 8px; }}
    h2 {{ margin-top: 28px; border-bottom: 1px solid #d8dee4; padding-bottom: 6px; }}
    code {{ background: #f6f8fa; padding: 2px 5px; border-radius: 4px; }}
    table {{ border-collapse: collapse; font-size: 12px; margin-top: 10px; max-width: 100%; }}
    th, td {{ border: 1px solid #d8dee4; padding: 5px 7px; text-align: right; white-space: nowrap; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #f6f8fa; }}
    .note {{ color: #46515c; max-width: 920px; line-height: 1.45; }}
    .section-desc {{ color: #36414c; max-width: 980px; line-height: 1.55; margin: 8px 0 12px; }}
    .cards {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 16px 0; }}
    .card {{ border: 1px solid #d8dee4; border-radius: 6px; padding: 10px 12px; min-width: 140px; }}
    .card-label {{ color: #66717d; font-size: 12px; }}
    .card-value {{ font-size: 18px; font-weight: 650; margin-top: 4px; }}
    .heatmap td {{ min-width: 38px; }}
    .links li {{ margin: 8px 0; }}
  </style>
</head>
<body>
{body}
</body>
</html>
""",
        encoding="utf-8",
    )


def _card(label: str, value: object) -> str:
    return (
        "<div class='card'>"
        f"<div class='card-label'>{html.escape(str(label))}</div>"
        f"<div class='card-value'>{html.escape(str(value))}</div>"
        "</div>"
    )


def _df_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "<p>No rows.</p>"
    display = frame.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda value: f"{value:.6g}")
    return display.to_html(index=False, escape=True)


def _matrix_heatmap(matrix: np.ndarray, labels: np.ndarray, max_size: int) -> str:
    size = min(max_size, matrix.shape[0])
    sub = np.asarray(matrix[:size, :size], dtype=float)
    finite = sub[np.isfinite(sub)]
    if finite.size == 0:
        return "<p>Matrix has no finite values.</p>"
    min_value = float(np.min(finite))
    max_value = float(np.max(finite))
    span = max(max_value - min_value, 1e-18)
    header = "".join(f"<th>{html.escape(str(label))}</th>" for label in labels[:size])
    rows = [f"<tr><th></th>{header}</tr>"]
    for idx in range(size):
        cells = []
        for value in sub[idx]:
            intensity = int(255 - 175 * ((float(value) - min_value) / span))
            cells.append(
                f"<td style='background: rgb({intensity}, {intensity + 10}, 255)'>{value:.3g}</td>"
            )
        rows.append(f"<tr><th>{html.escape(str(labels[idx]))}</th>{''.join(cells)}</tr>")
    return "<table class='heatmap'>" + "".join(rows) + "</table>"


def _quote_for_display(value: str) -> str:
    if not value or any(char.isspace() for char in value):
        return repr(value)
    return value


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
