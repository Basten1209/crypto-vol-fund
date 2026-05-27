"""
Phase 2 EDA - Noise:Signal Ratio 추정
Aït-Sahalia, Mykland & Zhang (2005) 방법론:
  - noise variance: q² = -E[r_t * r_{t+1}] (lag-1 autocovariance of 1분봉 returns)
  - signal (integrated variance): IV ≈ sum of r_t² + 2*sum(r_t*r_{t+1})
  - noise-to-signal ratio: 2*m*q² / IV
- 산출물: docs/phase2_eda/noise_signal_report.html
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import ANALYSIS_START, RANDOM_SEED, M

np.random.seed(RANDOM_SEED)

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
PRICE_PANEL_PATH = os.path.join(PROJECT_ROOT, 'src', 'phase1_data', 'price_panel.csv')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'docs', 'phase2_eda')


def load_prices():
    df = pd.read_csv(PRICE_PANEL_PATH, index_col=0, parse_dates=True)
    price_cols = [c for c in df.columns if c != 'trading_day']
    trading_day = df['trading_day'] if 'trading_day' in df.columns else None
    return df[price_cols], trading_day, price_cols


def estimate_noise_signal_per_ticker(log_ret_1d: np.ndarray) -> dict:
    """
    Aït-Sahalia et al. (2005) noise variance estimator (일별 데이터 기반)
    log_ret_1d: shape (T,) — 하루의 1분봉 log-return
    반환: noise_var, signal_iv, ratio
    """
    r = log_ret_1d
    n = len(r)
    if n < 4:
        return {'noise_var': np.nan, 'signal_iv': np.nan, 'ratio': np.nan}

    # noise variance: q² = -mean(r_t * r_{t+1}) (AMZ 2005 Proposition 1)
    lag1_cov = np.mean(r[:-1] * r[1:])
    noise_var = max(-lag1_cov, 0.0)  # q² ≥ 0

    # bias-corrected IV (Zhang, Mykland & Aït-Sahalia 2005 TSRV 분자)
    rv = np.sum(r ** 2)
    # simple RV - 2*n*noise_var as IV estimator
    signal_iv = max(rv - 2 * n * noise_var, 1e-20)

    # noise-to-signal ratio: 2*n*q² / IV
    ratio = 2 * n * noise_var / signal_iv if signal_iv > 0 else np.nan

    return {
        'noise_var': noise_var,
        'signal_iv': signal_iv,
        'ratio': ratio,
    }


def compute_daily_noise_signal(prices: pd.DataFrame, trading_day: pd.Series,
                                ticker: str) -> pd.DataFrame:
    """종목의 일별 noise-signal ratio 계산"""
    log_p = np.log(prices[ticker].values)
    log_ret = np.diff(log_p)
    ts = prices.index[1:]

    if trading_day is not None:
        td = trading_day.reindex(ts, method='nearest').values
    else:
        td = pd.to_datetime(ts).date

    # 분석 시작일 필터
    mask = ts >= pd.Timestamp(ANALYSIS_START)
    log_ret = log_ret[mask]
    td = td[mask]

    results = []
    unique_days = pd.unique(td)
    for day in unique_days:
        day_mask = td == day
        r = log_ret[day_mask]
        est = estimate_noise_signal_per_ticker(r)
        est['date'] = day
        results.append(est)

    return pd.DataFrame(results).set_index('date')


def compute_cross_asset_noise_signal(prices, trading_day, price_cols):
    """전 종목 평균 noise-signal ratio 계산"""
    print(f"  [noise_signal] {len(price_cols)}개 종목 처리 중...")
    all_ratios = {}
    all_noise = {}
    for ticker in price_cols:
        df = compute_daily_noise_signal(prices, trading_day, ticker)
        all_ratios[ticker] = df['ratio']
        all_noise[ticker] = df['noise_var']

    ratio_panel = pd.DataFrame(all_ratios)
    noise_panel = pd.DataFrame(all_noise)
    return ratio_panel, noise_panel


def plot_noise_signal_distribution(ratio_panel: pd.DataFrame) -> go.Figure:
    """종목별 noise-signal ratio 분포"""
    mean_ratios = ratio_panel.mean(axis=0).sort_values(ascending=False)

    fig = make_subplots(rows=1, cols=2,
                         subplot_titles=['종목별 평균 Noise-Signal Ratio',
                                         'Cross-Asset 일별 Noise-Signal Ratio 시계열'])

    fig.add_trace(go.Bar(
        x=mean_ratios.index.tolist(),
        y=mean_ratios.values,
        marker_color='steelblue',
        name='평균 ratio',
    ), row=1, col=1)

    # 시계열: cross-asset 중앙값
    ts_median = ratio_panel.median(axis=1)
    ts_q25 = ratio_panel.quantile(0.25, axis=1)
    ts_q75 = ratio_panel.quantile(0.75, axis=1)

    fig.add_trace(go.Scatter(
        x=ts_median.index.astype(str),
        y=ts_q75.values,
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        name='Q75',
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=ts_median.index.astype(str),
        y=ts_q25.values,
        fill='tonexty',
        fillcolor='rgba(70,130,180,0.2)',
        line=dict(width=0),
        showlegend=False,
        name='Q25',
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=ts_median.index.astype(str),
        y=ts_median.values,
        mode='lines',
        line=dict(color='steelblue', width=2),
        name='Cross-asset 중앙값',
    ), row=1, col=2)

    fig.update_layout(
        title='Noise-Signal Ratio (Aït-Sahalia et al. 2005)',
        height=450,
        template='plotly_white',
    )
    return fig


def generate_report(fig, ratio_panel, noise_panel):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

    mean_ratio = ratio_panel.mean(axis=0)
    median_ratio = ratio_panel.median(axis=0)
    summary_df = pd.DataFrame({
        'mean_ratio': mean_ratio.round(4),
        'median_ratio': median_ratio.round(4),
        'mean_noise_var': noise_panel.mean(axis=0).round(8),
    }).sort_values('mean_ratio', ascending=False)
    summary_html = summary_df.to_html(classes='table table-striped table-sm', border=0)

    overall_mean = ratio_panel.values[~np.isnan(ratio_panel.values)].mean()
    interpretation = (
        f"전체 평균 noise-signal ratio: <strong>{overall_mean:.4f}</strong>. "
        "ratio가 낮을수록 microstructure noise가 signal 대비 작아 1분봉 데이터 품질이 양호합니다."
    )

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>Noise-Signal Ratio Report</title>
<link rel="stylesheet"
  href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
<style>body {{ padding: 20px; }} h2 {{ margin-top: 40px; }}</style>
</head>
<body>
<div class="container-fluid">
<h1>Phase 2 EDA — Noise:Signal Ratio 추정 리포트</h1>
<p class="text-muted">분석 기간: {ANALYSIS_START} ~ | 방법론: Aït-Sahalia, Mykland & Zhang (2005)</p>

<div class="alert alert-info">{interpretation}</div>

<div class="mb-5">
<h2>1. Noise-Signal Ratio 분포</h2>
{fig_html}
</div>

<div class="mb-5">
<h2>2. 종목별 요약</h2>
{summary_html}
</div>

</div>
</body>
</html>"""

    out_path = os.path.join(OUTPUT_DIR, 'noise_signal_report.html')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"[noise_signal] 저장 완료: {out_path}")
    return out_path


if __name__ == '__main__':
    print("[noise_signal] 데이터 로드 중...")
    prices, trading_day, price_cols = load_prices()

    print("[noise_signal] 일별 noise-signal ratio 계산 중...")
    ratio_panel, noise_panel = compute_cross_asset_noise_signal(prices, trading_day, price_cols)

    print("[noise_signal] 차트 생성 중...")
    fig = plot_noise_signal_distribution(ratio_panel)

    print("[noise_signal] HTML 리포트 생성 중...")
    out_path = generate_report(fig, ratio_panel, noise_panel)
    print(f"[noise_signal] 완료: {out_path}")
