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
        help="Do not open the generated demo index in the browser/Finder.",
    )
    parser.add_argument(
        "--open-phase-artifacts",
        action="store_true",
        help="Open every phase HTML page as soon as it is generated. Default: open only the final index.",
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
    open_index = not args.no_open_artifacts
    open_phase_artifacts = open_index and args.open_phase_artifacts

    print("=== Demo Phase 3-6 pipeline ===", flush=True)
    print(f"repo: {ROOT}", flush=True)
    print(f"price_panel: {price_panel}", flush=True)
    print(f"output_run_dir: {_display_path(run_dir)}", flush=True)
    print(f"variant: {args.variant}", flush=True)
    print(
        "open_mode: "
        + ("none" if not open_index else "phase pages + final index" if open_phase_artifacts else "final index only"),
        flush=True,
    )
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
    _artifact_event("Phase 3 PRVM/JV matrices", phase3_artifact, open_phase_artifacts)

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
    _artifact_event("Phase 4 EWMA forecast comparison", phase4_artifact, open_phase_artifacts)

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
        _artifact_event(f"Phase 5 portfolio optimization ({variant['label']})", phase5_artifact, open_phase_artifacts)

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
        _artifact_event(f"Phase 6 backtest ({variant['label']})", phase6_artifact, open_phase_artifacts)

    index_path = _build_demo_index(run_dir, variants)
    print("=== Demo pipeline complete ===", flush=True)
    print(f"Artifacts saved under: {_display_path(run_dir)}", flush=True)
    print(f"Demo index: {_display_path(index_path)}", flush=True)
    if open_index:
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
    <p class="section-desc">가장 마지막 계산일의 jump-adjusted PRVM 행렬입니다. 각 셀은 두 자산 사이의 일별 고빈도 공분산 추정값을 의미하며, 포트폴리오 최적화에서 변동성/공분산 구조를 설명하는 핵심 입력입니다. 아래 표는 50개 자산 전체의 50x50 행렬이며, 화면이 좁으면 가로로 스크롤해서 볼 수 있습니다.</p>
    <p>CSV: <code>{_display_path(latest_prvm_csv)}</code></p>
    {_matrix_heatmap(prvm[latest_idx], tickers)}
    <h2>Latest JV Matrix: {html.escape(latest_date)}</h2>
    <p class="section-desc">같은 날짜의 Jump Volatility 행렬입니다. raw PRVM에서 jump-adjusted PRVM을 뺀 구성요소로, 갑작스러운 가격 점프가 자산별/자산간 위험에 얼마나 반영되는지 보여줍니다. 이 표도 50개 자산 전체의 50x50 행렬입니다.</p>
    <p>CSV: <code>{_display_path(latest_jv_csv)}</code></p>
    {_matrix_heatmap(jv[latest_idx], tickers)}
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
    daily_display = daily.copy()
    daily_display["adjusted_mspe_x1e4"] = daily_display["adjusted_mspe"] * 1e4
    daily_display["raw_mspe_x1e4"] = daily_display["raw_mspe"] * 1e4
    daily_display["mspe_delta_raw_minus_adjusted_x1e4"] = daily_display["mspe_delta_raw_minus_adjusted"] * 1e4
    daily_display["adjusted_qlike_x1e_minus3"] = daily_display["adjusted_qlike"] * 1e-3
    daily_display["raw_qlike_x1e_minus3"] = daily_display["raw_qlike"] * 1e-3
    daily_display["qlike_delta_raw_minus_adjusted_x1e_minus3"] = (
        daily_display["qlike_delta_raw_minus_adjusted"] * 1e-3
    )
    adjusted_better_mspe = daily_display[daily_display["mspe_delta_raw_minus_adjusted"] > 0].sort_values(
        "mspe_delta_raw_minus_adjusted",
        ascending=False,
    )
    raw_better_mspe = daily_display[daily_display["mspe_delta_raw_minus_adjusted"] < 0].sort_values(
        "mspe_delta_raw_minus_adjusted",
        ascending=True,
    )

    html_path = phase4_dir / "demo_phase4_ewma_comparison.html"
    body = f"""
    <h1>Phase 4: EWMA Forecast / Jump Adjustment Comparison</h1>
    <p class="note">Both models are evaluated against the same next-day jump-adjusted PRVM target. Positive raw-minus-adjusted deltas mean jump-adjusted EWMA has lower loss.</p>
    <section class="cards">
      {_card("Forecast days", report["n_forecast_days"])}
      {_card("Target range", f"{report['first_target_date']} to {report['last_target_date']}")}
      {_card("EWMA decay lambda", report["params"]["lambda"])}
      {_card("Rolling window", f"{report['params']['window_days']} days")}
    </section>
    <p class="section-desc">여기서 <code>lambda=0.94</code>는 과거 전체 기간을 계속 누적하는 recursive EWMA 설정이 아니라, 직전 28일 rolling window 내부에서 최근 PRVM에 더 큰 가중치를 주는 decay 계수입니다. 즉 예측은 항상 최근 28일 데이터만 사용하고, 그 28일 안에서 날짜가 오래될수록 가중치가 <code>0.94</code> 비율로 감소합니다.</p>
    <h2>Mean Loss Table</h2>
    <p class="section-desc">직전 28일 rolling window에 EWMA 가중치를 적용해 만든 예측값을 비교합니다. jump-adjusted PRVM을 입력으로 쓴 모델과 raw PRVM을 입력으로 쓴 모델의 평균 예측 손실을 보여주며, MSPE와 QLIKE는 모두 낮을수록 예측 성능이 좋습니다. 표의 <code>x1e4</code>는 원래 MSPE 값에 10,000을 곱해 보기 쉽게 표시했다는 뜻이고, <code>x1e_minus3</code>는 QLIKE 값에 0.001을 곱해 표시했다는 뜻입니다.</p>
    <p>CSV: <code>{_display_path(comparison_dir / "phase4_ewma_comparison_summary.csv")}</code></p>
    {_df_table(summary)}
    <h2>Raw minus Adjusted Delta</h2>
    <p class="section-desc">raw PRVM EWMA 손실에서 jump-adjusted PRVM EWMA 손실을 뺀 값입니다. 양수이면 raw 손실이 더 크므로 jump-adjusted 방식이 유리하고, 음수이면 raw 방식이 해당 지표에서 유리합니다. MSPE delta의 <code>x1e4</code>와 QLIKE delta의 <code>x1e-3</code> 역시 표시 단위만 바꾼 값이며, 원본 값은 CSV/JSON에 그대로 저장됩니다.</p>
    <section class="cards">
      {_card("MSPE delta x1e4", f"{comparison['comparison']['mspe_delta_raw_minus_adjusted_x1e4']:.6g}")}
      {_card("QLIKE delta x1e-3", f"{comparison['comparison']['qlike_delta_raw_minus_adjusted_x1e_minus3']:.6g}")}
      {_card("Adjusted lower MSPE days", comparison["comparison"]["days_adjusted_lower_mspe"])}
      {_card("Adjusted lower QLIKE days", comparison["comparison"]["days_adjusted_lower_qlike"])}
    </section>
    <h2>Days Where Jump-Adjusted MSPE Is Better</h2>
    <p class="section-desc">MSPE 기준으로 jump-adjusted 입력이 raw 입력보다 더 좋은 날짜입니다. <code>mspe_delta_raw_minus_adjusted_x1e4</code>가 클수록 raw MSPE가 adjusted MSPE보다 더 컸다는 뜻이므로, jump adjustment의 개선 폭이 큰 사례입니다.</p>
    <p>Daily CSV: <code>{_display_path(comparison_dir / "phase4_ewma_comparison_daily.csv")}</code></p>
    {_df_table(adjusted_better_mspe.head(10)[["target_date", "adjusted_mspe_x1e4", "raw_mspe_x1e4", "mspe_delta_raw_minus_adjusted_x1e4", "adjusted_qlike_x1e_minus3", "raw_qlike_x1e_minus3"]])}
    <h2>Days Where Raw MSPE Is Better</h2>
    <p class="section-desc">MSPE 기준으로 raw 입력이 jump-adjusted 입력보다 더 좋은 날짜입니다. delta가 음수이면 adjusted MSPE가 raw MSPE보다 컸다는 뜻이므로, 모든 날짜에서 jump adjustment가 항상 유리하지는 않다는 점을 보여주는 진단 표입니다.</p>
    {_df_table(raw_better_mspe.head(10)[["target_date", "adjusted_mspe_x1e4", "raw_mspe_x1e4", "mspe_delta_raw_minus_adjusted_x1e4", "adjusted_qlike_x1e_minus3", "raw_qlike_x1e_minus3"]])}
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
    diagnostics_legend = _phase5_diagnostics_legend()
    phase5_summary_html = _phase5_summary_sections(
        report=report,
        cycle_summary=cycle_summary,
        top_weights=top_weights,
        top_weights_csv=top_weights_csv,
        summary=summary,
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
    <h2>Diagnostics Glossary</h2>
    <p class="section-desc">아래 용어들은 최적화 결과가 정상적으로 생성되었는지 확인하기 위한 진단 지표입니다. 수익률 지표가 아니라, 포트폴리오 비중 계산의 안정성/집중도/변화량을 설명하는 값입니다.</p>
    {_df_table(diagnostics_legend)}
    {phase5_summary_html}
    """
    _write_html(html_path, "Phase 5 Portfolio", body)
    return html_path


def _phase5_summary_sections(
    report: dict[str, object],
    cycle_summary: pd.DataFrame,
    top_weights: pd.DataFrame,
    top_weights_csv: Path,
    summary: pd.DataFrame,
) -> str:
    summary_cols = [
        "n_rebalances",
        "active_count_mean",
        "top_weight_mean",
        "top_weight_max",
        "turnover_mean",
        "kkt_violation_max",
    ]
    if report["params"]["rebalance_frequency"] == "monthly":
        representative_cycle = int(cycle_summary["cycle_days"].astype(int).min())
        target_summary = cycle_summary[cycle_summary["cycle_days"].astype(int) == representative_cycle][summary_cols]
        target_summary = target_summary.assign(hold_windows=", ".join(cycle_summary["cycle_days"].astype(str) + "D"))
        target_summary = target_summary[["hold_windows", *summary_cols]]
        hold_windows = cycle_summary[
            ["cycle_days", "n_rebalances", "first_rebalance_date", "last_rebalance_date"]
        ].copy()
        top_weights_display = top_weights[top_weights["cycle_days"].astype(int) == representative_cycle].drop(
            columns=["cycle_days"]
        )
        diagnostics_sample = summary[summary["cycle_days"].astype(int) == representative_cycle].head(12)
        diagnostics_sample = diagnostics_sample[
            ["rebalance_date", "active_count", "top_ticker", "top_weight", "turnover_from_prev", "solver_status"]
        ]
        return f"""
    <h2>Monthly Target Summary</h2>
    <p class="section-desc">월간 리밸런싱 구조에서는 7일/14일 window가 같은 월초 target weight를 공유합니다. 따라서 Phase 5의 최적화 요약값은 두 window에서 동일한 것이 정상이며, 중복 표시하지 않고 한 번만 보여줍니다. 7일/14일의 차이는 Phase 6에서 보유기간 길이와 Simple/Managed 운용 방식으로 평가됩니다.</p>
    {_df_table(target_summary)}
    <h2>Hold Window Configuration</h2>
    <p class="section-desc">Phase 5에서 계산된 같은 월간 target weight를 각각 7일 또는 14일 동안 운용하는 설정입니다. 이 표는 최적화 결과가 아니라 Phase 6 backtest에서 사용할 보유기간 구분입니다.</p>
    {_df_table(hold_windows)}
    <h2>Latest Monthly Target Top Weights</h2>
    <p class="section-desc">가장 최근 월간 리밸런싱 시점의 상위 편입 비중입니다. 7일/14일 window가 같은 target weight를 쓰기 때문에 한 번만 표시합니다.</p>
    <p>CSV: <code>{_display_path(top_weights_csv)}</code></p>
    {_df_table(top_weights_display)}
    <h2>Monthly Target Diagnostics Sample</h2>
    <p class="section-desc">월간 리밸런싱 날짜별 최적화 진단 샘플입니다. Simple/Managed의 일별 운용 차이는 이 단계가 아니라 Phase 6 backtest에서 적용됩니다.</p>
    {_df_table(diagnostics_sample)}
    """

    return f"""
    <h2>Cycle Summary</h2>
    <p class="section-desc">cycle별 최적화 요약입니다. 리밸런싱 횟수, 평균 편입 종목 수, 최고 비중, turnover, KKT violation을 통해 포트폴리오가 얼마나 집중되었고 최적화가 안정적으로 풀렸는지 확인합니다.</p>
    {_df_table(cycle_summary[["cycle_days", *summary_cols]])}
    <h2>Latest Rebalance Top Weights</h2>
    <p class="section-desc">각 cycle에서 가장 최근 리밸런싱 시점의 상위 편입 비중입니다. 데모에서는 모델이 실제로 어떤 코인에 얼마만큼 배분했는지를 직관적으로 보여주는 핵심 화면입니다.</p>
    <p>CSV: <code>{_display_path(top_weights_csv)}</code></p>
    {_df_table(top_weights)}
    <h2>Rebalance Diagnostics Sample</h2>
    <p class="section-desc">리밸런싱 날짜별 최적화 진단 샘플입니다. active_count, top_ticker, top_weight, turnover, solver_status를 통해 포트폴리오 생성 과정이 정상적으로 진행되었는지 확인합니다.</p>
    {_df_table(summary.head(12)[["cycle_days", "rebalance_date", "active_count", "top_ticker", "top_weight", "turnover_from_prev", "solver_status"]])}
    """


def _phase5_diagnostics_legend() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "metric": "active_count",
                "meaning": "비중이 사실상 0이 아닌 편입 자산 수입니다.",
                "how_to_read": "클수록 더 분산되어 있고, 작을수록 소수 자산에 집중된 포트폴리오입니다.",
            },
            {
                "metric": "top_weight / top_weight_max",
                "meaning": "가장 큰 단일 자산 비중입니다.",
                "how_to_read": "높을수록 concentration risk가 큽니다. cap25 variant에서는 25%를 넘지 않아야 합니다.",
            },
            {
                "metric": "turnover_from_prev / turnover_mean",
                "meaning": "이전 리밸런싱 목표 비중과 새 목표 비중의 절대 변화량 합입니다.",
                "how_to_read": "0이면 이전과 동일한 비중이고, 값이 클수록 매매해야 할 비중 변화가 큽니다.",
            },
            {
                "metric": "kkt_violation / kkt_violation_max",
                "meaning": "최소분산 최적화의 KKT 조건 위반 정도입니다.",
                "how_to_read": "0에 가까울수록 제약조건을 만족하는 안정적인 최적해입니다. 아주 작은 수는 수치 오차 수준으로 봅니다.",
            },
            {
                "metric": "solver_status",
                "meaning": "최적화 solver가 어떤 방식으로 해를 찾았는지 나타냅니다.",
                "how_to_read": "optimal 계열이면 정상적으로 최적해를 찾은 상태입니다.",
            },
            {
                "metric": "min_positive_weight",
                "meaning": "0이 아닌 편입 자산 중 가장 작은 비중입니다.",
                "how_to_read": "너무 작은 비중은 실거래에서 의미가 작기 때문에 min_asset_weight 기준으로 pruning됩니다.",
            },
        ]
    )


def _build_phase6_artifacts(phase6_dir: Path, label: str) -> Path:
    report = json.loads((phase6_dir / "phase6_backtest_report.json").read_text(encoding="utf-8"))
    performance_path = phase6_dir / "phase6_performance_table.csv"
    performance = pd.read_csv(performance_path)
    dm_path = phase6_dir / "phase6_dm_test.csv"
    dm = pd.read_csv(dm_path)
    dm_display = _phase6_dm_display(dm)
    monthly_path = phase6_dir / "monthly_hold_window_returns.csv"
    monthly = pd.read_csv(monthly_path) if monthly_path.exists() else pd.DataFrame()
    performance_guide = _phase6_performance_guide()
    dm_guide = _phase6_dm_guide()
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
    <h2>Hold-Window Performance Table</h2>
    <p class="section-desc">이 표는 우리가 실제로 설명하려는 운용 정의에 맞춘 메인 성과표입니다. 매월 1일마다 동일한 초기 투자금으로 새 7일/14일 상품을 시작한다고 가정합니다. 직전 28일 PRVM/RVM 정보로 해당 월의 변동성 행렬을 예측하고 target weight를 정한 뒤, 그 weight를 월초부터 7일 또는 14일 동안만 운용합니다. 따라서 아래 성과는 전체 calendar day 성과가 아니라, 각 월의 독립적인 active hold window 성과를 모아 요약한 값입니다. Simple Mode는 최초 진입 후 weight drift를 허용하고, Managed Mode는 보유기간 동안 매일 target weight로 되돌립니다.</p>
    <p>CSV: <code>{_display_path(performance_path)}</code></p>
    {_df_table(performance)}
    <h2>Performance Metric Guide</h2>
    <p class="section-desc">성과표의 주요 컬럼을 해석하는 기준입니다. 모든 값은 active hold window만 대상으로 계산됩니다. 수익률 계열은 높을수록 좋고, 변동성/MDD는 일반적으로 낮을수록 안정적입니다. 다만 minimum variance 전략은 수익률 극대화보다 위험 축소가 1차 목적이므로, equal-weight 대비 변동성·drawdown·DM test를 함께 봐야 합니다.</p>
    {_df_table(performance_guide)}
    <h2>Diebold-Mariano Test</h2>
    <p class="section-desc">Hold-Window Performance Table과 같은 active hold window 일별 수익률만 사용해 minimum variance와 equal-weight의 손실 차이를 검정합니다. 여기서는 <code>squared_daily_return</code>, 즉 일별 수익률의 제곱을 손실로 사용합니다. 이 손실은 수익률 방향이 아니라 수익률의 크기, 즉 realized risk를 보는 지표입니다. 따라서 이 검정은 "어느 전략이 수익률이 더 높았는가"가 아니라 "minimum variance가 equal-weight보다 보유구간의 일별 변동 위험을 유의하게 낮췄는가"에 가깝습니다.</p>
    <p>CSV: <code>{_display_path(dm_path)}</code></p>
    {_df_table(dm_display)}
    <h2>Diebold-Mariano Reading Guide</h2>
    <p class="section-desc">DM test를 읽는 방법입니다. <code>mean_loss_diff</code>는 minimum variance 손실에서 equal-weight 손실을 뺀 값입니다. 음수이면 minimum variance의 평균 손실이 더 낮고, 양수이면 equal-weight의 평균 손실이 더 낮습니다. <code>p_value</code>가 0.05보다 작으면 5% 유의수준에서 차이가 통계적으로 의미 있다고 봅니다.</p>
    {_df_table(dm_guide)}
    <h2>Monthly Hold Window Returns Sample</h2>
    <p class="section-desc">월별 hold window 안에서 전략별 수익률을 비교한 샘플입니다. 이 표의 각 행은 특정 월에 동일한 초기 투자금으로 시작한 새 7일/14일 상품의 성과를 의미합니다. 특정 월/보유기간에서 minimum variance가 equal-weight나 BTC 대비 어떻게 움직였는지 설명할 때 사용합니다.</p>
    {_df_table(monthly.head(16)) if not monthly.empty else "<p>No monthly hold-window output.</p>"}
    """
    _write_html(html_path, "Phase 6 Backtest", body)
    return html_path


def _phase6_performance_guide() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "column": "n_months / invested_days",
                "meaning": "월초 7일/14일 투자 window가 몇 개월, 며칠 포함되었는지 나타냅니다.",
                "how_to_read": "전체 calendar day가 아니라 실제 포트폴리오가 투자된 날짜만 세는 값입니다.",
            },
            {
                "column": "total_return_on_hold_windows",
                "meaning": "각 월의 active hold-window 일별 수익률을 순서대로 이어 붙여 복리로 계산한 누적수익률입니다.",
                "how_to_read": "0.10은 투자된 보유구간들만 연결했을 때 +10%, -0.10은 -10%입니다.",
            },
            {
                "column": "mean_monthly_hold_return",
                "meaning": "각 월의 7일 또는 14일 hold-window 수익률을 계산한 뒤 그 월별 값을 평균한 것입니다.",
                "how_to_read": "월별 window 성과의 평균이 필요할 때 봅니다. 복리 누적값은 total_return_on_hold_windows를 봅니다.",
            },
            {
                "column": "positive_month_rate",
                "meaning": "월별 hold-window 수익률이 0보다 컸던 월의 비율입니다.",
                "how_to_read": "0.60이면 해당 window에서 60%의 월이 플러스 수익률이었다는 뜻입니다.",
            },
            {
                "column": "annualized_return_on_invested_days",
                "meaning": "active hold-window 일별 수익률만 사용해 365일 crypto calendar 기준으로 연환산한 수익률입니다.",
                "how_to_read": "짧은 window 수익률을 연환산하므로 값이 크게 보일 수 있습니다. 방향성 확인용으로 사용합니다.",
            },
            {
                "column": "annualized_volatility_on_invested_days",
                "meaning": "active hold-window 일별 수익률의 표준편차를 연환산한 realized volatility입니다.",
                "how_to_read": "minimum variance 전략에서는 이 값이 equal-weight보다 낮은지가 핵심 진단입니다.",
            },
            {
                "column": "sharpe_on_invested_days",
                "meaning": "active hold-window의 평균 수익률을 변동성으로 나눈 위험조정 성과입니다.",
                "how_to_read": "높을수록 좋지만, 수익률이 음수인 기간에는 변동성이 낮아도 Sharpe가 낮을 수 있습니다.",
            },
            {
                "column": "max_drawdown_on_hold_windows",
                "meaning": "active hold-window 수익률만 연결한 equity curve의 최대 낙폭입니다.",
                "how_to_read": "-0.25는 고점 대비 최대 -25% 하락을 의미하며, 0에 가까울수록 방어적입니다.",
            },
            {
                "column": "realized_risk_annualized_mean",
                "meaning": "active hold-window 중 실제 포트폴리오 비중으로 측정한 평균 연환산 위험입니다.",
                "how_to_read": "ex-ante 최소분산 weight가 실제 운용에서도 낮은 위험으로 이어졌는지 확인합니다.",
            },
        ]
    )


def _phase6_dm_display(dm: pd.DataFrame) -> pd.DataFrame:
    if dm.empty:
        return dm
    display = dm.copy()
    display["direction"] = display["mean_loss_diff"].apply(_dm_direction)
    display["significance_5pct"] = display["p_value"].apply(
        lambda value: "significant" if pd.notna(value) and float(value) < 0.05 else "not significant"
    )
    display["plain_reading"] = display.apply(_dm_plain_reading, axis=1)
    return display


def _dm_direction(mean_loss_diff: float) -> str:
    if not np.isfinite(mean_loss_diff):
        return "not available"
    if mean_loss_diff < 0:
        return "minimum_variance lower risk loss"
    if mean_loss_diff > 0:
        return "equal_weight lower risk loss"
    return "same average loss"


def _dm_plain_reading(row: pd.Series) -> str:
    mean_loss_diff = float(row["mean_loss_diff"])
    p_value = float(row["p_value"])
    if not np.isfinite(mean_loss_diff) or not np.isfinite(p_value):
        return "검정에 필요한 유효 관측치가 부족합니다."
    leader = "minimum variance" if mean_loss_diff < 0 else "equal-weight" if mean_loss_diff > 0 else "neither strategy"
    if p_value < 0.05:
        return f"{leader} has lower squared-return loss, and the difference is statistically significant at 5%."
    return f"{leader} has lower squared-return loss in this sample, but the difference is not statistically significant at 5%."


def _phase6_dm_guide() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "column": "mean_loss_diff",
                "meaning": "minimum_variance의 squared daily return loss에서 equal_weight의 loss를 뺀 평균값입니다.",
                "how_to_read": "음수이면 minimum variance가 평균적으로 더 낮은 위험 손실을 냈다는 뜻입니다.",
            },
            {
                "column": "dm_stat",
                "meaning": "loss 차이를 표준화한 검정통계량입니다.",
                "how_to_read": "부호는 mean_loss_diff와 같고, 절댓값이 클수록 두 전략의 손실 차이가 더 뚜렷합니다.",
            },
            {
                "column": "p_value",
                "meaning": "두 전략의 평균 손실이 같다는 귀무가설에서 현재 정도의 차이가 나올 확률입니다.",
                "how_to_read": "보통 0.05 미만이면 통계적으로 유의하다고 해석합니다. 0.05 이상이면 방향성은 보이더라도 강한 결론은 피합니다.",
            },
            {
                "column": "lag",
                "meaning": "겹치는 보유기간 때문에 생길 수 있는 자기상관을 보정하기 위한 HAC lag입니다.",
                "how_to_read": "7일 cycle은 6, 14일 cycle은 13처럼 cycle_days - 1을 사용합니다.",
            },
            {
                "column": "n_obs",
                "meaning": "검정에 사용된 일별 관측치 수입니다.",
                "how_to_read": "관측치가 많을수록 검정력이 커집니다.",
            },
        ]
    )


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
    <p class="note">This demo shows how 1-minute crypto prices are converted into jump-adjusted volatility matrix estimates, short-horizon minimum-variance fund weights, and active 7-day / 14-day backtest results.</p>
    <h2>End-to-End Workflow</h2>
    <p class="section-desc">전체 연구 흐름입니다. Phase 1-2는 데이터 준비와 문제 진단 단계이고, 이 데모 스크립트는 Phase 3-6의 핵심 산출물을 재현합니다.</p>
    <div class="workflow">
      <div class="phase-box">
        <div class="phase-label">Phase 1</div>
        <div class="phase-title">1-Minute Data Panel</div>
        <p>50개 Upbit KRW 자산의 1분 단위 가격 데이터를 수집하고, 공통 시간축의 price panel로 정리합니다.</p>
      </div>
      <div class="phase-box">
        <div class="phase-label">Phase 2</div>
        <div class="phase-title">EDA / Data Diagnostics</div>
        <p>수집된 고빈도 수익률 데이터의 heavy tailness와 비정규성을 확인합니다. 이 진단을 통해 극단값과 jump를 별도로 handling해야 안정적인 변동성 추정이 가능하다는 문제의식을 도출합니다.</p>
      </div>
      <div class="phase-box active-phase">
        <div class="phase-label">Phase 3</div>
        <div class="phase-title">Jump-Adjusted PRVM</div>
        <p>FIVAR paper의 Shin et al. 방식에 기반해 jump-adjusted pre-averaged realized volatility matrix를 계산하고, JV 행렬로 점프 기여분을 분리합니다.</p>
      </div>
      <div class="phase-box active-phase">
        <div class="phase-label">Phase 4</div>
        <div class="phase-title">Rolling 28-Day EWMA Forecast</div>
        <p>최근 28일 PRVM에 EWMA decay를 적용해 익일 변동성 행렬을 예측합니다. raw PRVM과 jump-adjusted PRVM의 예측 손실을 MSPE / QLIKE로 비교해 jump handling의 효과를 평가합니다.</p>
      </div>
      <div class="phase-box active-phase">
        <div class="phase-label">Phase 5</div>
        <div class="phase-title">Minimum-Variance Fund Weights</div>
        <p>예측된 변동성 행렬을 입력으로 long-only 최소분산 포트폴리오를 계산합니다. 매월 1일 동일 초기금액으로 새 7일 / 14일 단기 상품을 시작하는 구조이며, Simple / Managed 운용 옵션을 둡니다.</p>
      </div>
      <div class="phase-box active-phase">
        <div class="phase-label">Phase 6</div>
        <div class="phase-title">Hold-Window Backtest</div>
        <p>각 월의 7일 / 14일 active hold window 성과를 backtest합니다. Equal Weight 대비 realized risk가 낮은지 성과표와 Diebold-Mariano test로 확인합니다.</p>
      </div>
    </div>
    <h2>Demo Thesis</h2>
    <p class="section-desc">heavy tail과 비정규성이 강한 고빈도 crypto 수익률 데이터에서 jump-adjusted PRVM을 만들고, 최근 28일 rolling EWMA로 단기 변동성 행렬을 예측합니다. 이 예측 행렬로 구성한 최소분산 포트폴리오가 같은 50개 자산 universe의 equal-weight benchmark보다 보유기간 realized risk를 낮출 수 있는지를 검증하는 것이 데모의 핵심 메시지입니다.</p>
    <h2>Result Pages</h2>
    <p class="section-desc">Phase별 결과 HTML로 이동하는 목차입니다. 데모 녹화 중에는 이 페이지에서 각 산출물을 순서대로 열어 계산 흐름과 결과를 설명할 수 있습니다.</p>
    <ul class="links">{link_html}</ul>
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
    .matrix-scroll {{ overflow-x: auto; max-width: 100%; border: 1px solid #d8dee4; border-radius: 6px; padding: 8px; }}
    .heatmap {{ font-size: 10px; }}
    .heatmap th, .heatmap td {{ padding: 3px 5px; min-width: 42px; }}
    .links li {{ margin: 8px 0; }}
    .workflow {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; margin: 16px 0 20px; }}
    .phase-box {{ border: 1px solid #d8dee4; border-radius: 6px; padding: 14px; background: #ffffff; min-height: 170px; }}
    .phase-box.active-phase {{ border-color: #7aa7d9; background: #f7fbff; }}
    .phase-label {{ color: #2563a8; font-size: 12px; font-weight: 700; text-transform: uppercase; }}
    .phase-title {{ font-size: 16px; font-weight: 700; margin-top: 6px; }}
    .phase-box p {{ color: #36414c; line-height: 1.5; margin: 10px 0 0; }}
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


def _matrix_heatmap(matrix: np.ndarray, labels: np.ndarray) -> str:
    size = matrix.shape[0]
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
    return "<div class='matrix-scroll'><table class='heatmap'>" + "".join(rows) + "</table></div>"


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
