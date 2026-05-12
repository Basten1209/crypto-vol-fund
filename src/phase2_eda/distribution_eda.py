"""
Phase 2 EDA - 수익률 분포 분석
- 1분봉 / 일별 log-return kurtosis, skewness
- Volatility clustering (squared-return ACF)
- 산출물: docs/phase2_eda/distribution_report.html
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from statsmodels.tsa.stattools import acf

from src.config import ANALYSIS_START, DAILY_CUT_KST, RANDOM_SEED

np.random.seed(RANDOM_SEED)

# 경로 설정
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
PRICE_PANEL_PATH = os.path.join(PROJECT_ROOT, 'src', 'phase1_data', 'price_panel.csv')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'docs', 'phase2_eda')


def load_price_panel() -> pd.DataFrame:
    """price_panel.csv 로드 및 전처리"""
    df = pd.read_csv(PRICE_PANEL_PATH, index_col=0, parse_dates=True)
    # trading_day 컬럼 분리
    price_cols = [c for c in df.columns if c != 'trading_day']
    prices = df[price_cols].copy()
    trading_day = df['trading_day'] if 'trading_day' in df.columns else None
    return prices, trading_day, price_cols


def compute_minute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """1분봉 log-return 계산"""
    log_ret = np.log(prices).diff()
    log_ret = log_ret.iloc[1:]  # 첫 행 NaN 제거
    # 분석 시작일 이후만
    log_ret = log_ret[log_ret.index >= ANALYSIS_START]
    return log_ret


def compute_daily_log_returns(prices: pd.DataFrame, trading_day: pd.Series) -> pd.DataFrame:
    """일별 log-return 계산 (trading_day 기준 첫/마지막 가격 사용)"""
    if trading_day is None:
        # fallback: calendar date
        prices_copy = prices.copy()
        prices_copy['_day'] = prices_copy.index.date
        daily_last = prices_copy.groupby('_day').last().drop(columns=['_day'])
    else:
        prices_copy = prices.copy()
        prices_copy['_day'] = trading_day.values
        daily_last = prices_copy.groupby('_day').last()

    daily_log_ret = np.log(daily_last).diff().iloc[1:]
    daily_log_ret.index = pd.to_datetime(daily_log_ret.index)
    daily_log_ret = daily_log_ret[daily_log_ret.index >= ANALYSIS_START]
    return daily_log_ret


def compute_stats(log_ret: pd.DataFrame) -> pd.DataFrame:
    """종목별 기초 통계 계산"""
    results = []
    for col in log_ret.columns:
        r = log_ret[col].dropna()
        results.append({
            'ticker': col,
            'mean': r.mean(),
            'std': r.std(),
            'skewness': float(stats.skew(r)),
            'kurtosis': float(stats.kurtosis(r)),  # excess kurtosis
            'min': r.min(),
            'max': r.max(),
            'n': len(r),
        })
    return pd.DataFrame(results).set_index('ticker')


def plot_kurtosis_distribution(min_stats: pd.DataFrame, day_stats: pd.DataFrame) -> go.Figure:
    """1분봉 kurtosis 분포 boxplot + 일별 kurtosis bar chart"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['1분봉 Kurtosis 분포 (종목별)', '1분봉 vs 일별 Kurtosis 비교'],
    )

    # boxplot: 1분봉 kurtosis
    fig.add_trace(
        go.Box(
            y=min_stats['kurtosis'].values,
            name='1분봉 Kurtosis',
            marker_color='steelblue',
            boxpoints='all',
            jitter=0.3,
            text=min_stats.index.tolist(),
            hovertemplate='%{text}: %{y:.2f}<extra></extra>',
        ),
        row=1, col=1,
    )

    # bar chart: 1분봉 vs 일별
    tickers = min_stats.index.tolist()
    fig.add_trace(
        go.Bar(
            x=tickers,
            y=min_stats['kurtosis'].values,
            name='1분봉',
            marker_color='steelblue',
            opacity=0.8,
        ),
        row=1, col=2,
    )
    fig.add_trace(
        go.Bar(
            x=tickers,
            y=day_stats.reindex(tickers)['kurtosis'].values,
            name='일별',
            marker_color='tomato',
            opacity=0.8,
        ),
        row=1, col=2,
    )
    fig.update_layout(
        title_text='Log-Return Kurtosis 분석 (Heavy-Tail 확인)',
        height=500,
        barmode='group',
        showlegend=True,
        template='plotly_white',
    )
    fig.update_yaxes(title_text='Excess Kurtosis', row=1, col=1)
    fig.update_yaxes(title_text='Excess Kurtosis', row=1, col=2)
    # normal distribution 기준선 (kurtosis = 0)
    fig.add_hline(y=0, line_dash='dash', line_color='gray', row=1, col=1)
    fig.add_hline(y=0, line_dash='dash', line_color='gray', row=1, col=2)
    return fig


def plot_acf_squared_returns(log_ret: pd.DataFrame, rep_tickers: list, nlags: int = 40) -> go.Figure:
    """대표 종목 squared-return ACF plot (volatility clustering 확인)"""
    fig = make_subplots(
        rows=len(rep_tickers), cols=1,
        subplot_titles=[f'{t} Squared-Return ACF' for t in rep_tickers],
        vertical_spacing=0.05,
    )
    for i, ticker in enumerate(rep_tickers, 1):
        r = log_ret[ticker].dropna()
        r2 = r ** 2
        acf_vals, confint = acf(r2, nlags=nlags, alpha=0.05, fft=True)
        lags = np.arange(len(acf_vals))

        # ACF bar
        colors = ['steelblue' if v >= 0 else 'tomato' for v in acf_vals[1:]]
        fig.add_trace(
            go.Bar(
                x=lags[1:],
                y=acf_vals[1:],
                marker_color=colors,
                name=ticker,
                showlegend=True,
            ),
            row=i, col=1,
        )
        # 신뢰구간
        ci_upper = confint[1:, 1] - acf_vals[1:]
        ci_lower = acf_vals[1:] - confint[1:, 0]
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([lags[1:], lags[1:][::-1]]),
                y=np.concatenate([ci_upper, -ci_lower[::-1]]),
                fill='toself',
                fillcolor='rgba(0,100,255,0.1)',
                line=dict(color='rgba(0,0,0,0)'),
                showlegend=False,
            ),
            row=i, col=1,
        )
        fig.add_hline(y=0, line_dash='dash', line_color='gray', row=i, col=1)

    fig.update_layout(
        title_text='Squared-Return ACF (Volatility Clustering 확인)',
        height=200 * len(rep_tickers),
        template='plotly_white',
        showlegend=True,
    )
    return fig


def plot_stats_scatter(min_stats: pd.DataFrame) -> go.Figure:
    """kurtosis vs std scatter (heavy-tail 구조 확인)"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=min_stats['std'] * 1e4,  # bps 단위
        y=min_stats['kurtosis'],
        mode='markers+text',
        text=min_stats.index.tolist(),
        textposition='top center',
        marker=dict(
            color=min_stats['kurtosis'],
            colorscale='RdBu_r',
            size=8,
            showscale=True,
            colorbar=dict(title='Excess Kurtosis'),
        ),
        hovertemplate='%{text}<br>Std: %{x:.2f} bps<br>Kurtosis: %{y:.2f}<extra></extra>',
    ))
    fig.update_layout(
        title='1분봉 Log-Return: 변동성(std) vs Kurtosis',
        xaxis_title='Std (bps, ×10⁻⁴)',
        yaxis_title='Excess Kurtosis',
        template='plotly_white',
        height=500,
    )
    return fig


def generate_report(fig_kurtosis, fig_acf, fig_scatter, min_stats, day_stats):
    """HTML 리포트 생성"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 각 figure를 HTML div로 변환
    kurtosis_html = fig_kurtosis.to_html(full_html=False, include_plotlyjs='cdn')
    acf_html = fig_acf.to_html(full_html=False, include_plotlyjs=False)
    scatter_html = fig_scatter.to_html(full_html=False, include_plotlyjs=False)

    # 요약 통계 테이블
    summary_html = min_stats.round(4).to_html(classes='table table-striped table-sm', border=0)

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>Phase 2 EDA - Distribution Report</title>
<link rel="stylesheet"
  href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
<style>
  body {{ font-family: 'Noto Sans KR', sans-serif; padding: 20px; }}
  h2 {{ margin-top: 40px; color: #333; }}
  .section {{ margin-bottom: 60px; }}
</style>
</head>
<body>
<div class="container-fluid">
<h1>Phase 2 EDA — 수익률 분포 분석 리포트</h1>
<p class="text-muted">분석 기간: {ANALYSIS_START} ~ | 종목 수: {len(min_stats)}</p>

<div class="section">
<h2>1. Kurtosis 분포 (Heavy-Tail 확인)</h2>
<p>1분봉 log-return의 excess kurtosis > 0이면 정규분포보다 heavy-tail임을 의미합니다.
암호화폐는 일반적으로 kurtosis가 매우 높아 극단적 수익률이 자주 발생합니다.</p>
{kurtosis_html}
</div>

<div class="section">
<h2>2. 분산 vs Kurtosis 산포도</h2>
{scatter_html}
</div>

<div class="section">
<h2>3. Squared-Return ACF (Volatility Clustering)</h2>
<p>제곱 수익률의 자기상관이 양수이고 천천히 감소하면 변동성 군집(volatility clustering)이 존재함을 의미합니다.
이는 EWMA 모델의 이론적 근거입니다.</p>
{acf_html}
</div>

<div class="section">
<h2>4. 1분봉 기초 통계 테이블</h2>
{summary_html}
</div>

</div>
</body>
</html>"""

    out_path = os.path.join(OUTPUT_DIR, 'distribution_report.html')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"[distribution_eda] 저장 완료: {out_path}")
    return out_path


if __name__ == '__main__':
    print("[distribution_eda] 데이터 로드 중...")
    prices, trading_day, price_cols = load_price_panel()

    print("[distribution_eda] 1분봉 log-return 계산 중...")
    min_log_ret = compute_minute_log_returns(prices)

    print("[distribution_eda] 일별 log-return 계산 중...")
    day_log_ret = compute_daily_log_returns(prices, trading_day)

    print("[distribution_eda] 기초 통계 계산 중...")
    min_stats = compute_stats(min_log_ret)
    day_stats = compute_stats(day_log_ret)

    print(f"  1분봉 평균 kurtosis: {min_stats['kurtosis'].mean():.2f}")
    print(f"  일별 평균 kurtosis:  {day_stats['kurtosis'].mean():.2f}")

    # 대표 종목 선정 (BTC, ETH, XRP, SOL, DOGE) — 존재하는 것만
    rep_candidates = ['BTC', 'ETH', 'XRP', 'SOL', 'DOGE']
    rep_tickers = [t for t in rep_candidates if t in min_log_ret.columns][:5]

    print("[distribution_eda] 차트 생성 중...")
    fig_kurtosis = plot_kurtosis_distribution(min_stats, day_stats)
    fig_acf = plot_acf_squared_returns(min_log_ret, rep_tickers)
    fig_scatter = plot_stats_scatter(min_stats)

    print("[distribution_eda] HTML 리포트 생성 중...")
    out_path = generate_report(fig_kurtosis, fig_acf, fig_scatter, min_stats, day_stats)
    print(f"[distribution_eda] 완료: {out_path}")
