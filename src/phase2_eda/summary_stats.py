"""
Phase 2 EDA - 50개 종목 요약 통계
- ticker별: 일평균 거래 횟수, log-return 통계 (일/연환산), kurtosis
- 산출물: docs/phase2_eda/summary_stats.csv + summary_stats.html
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go

from src.config import ANALYSIS_START, ANNUALIZATION, RANDOM_SEED

np.random.seed(RANDOM_SEED)

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
PRICE_PANEL_PATH = os.path.join(PROJECT_ROOT, 'src', 'phase1_data', 'price_panel.csv')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'docs', 'phase2_eda')


def load_prices():
    df = pd.read_csv(PRICE_PANEL_PATH, index_col=0, parse_dates=True)
    price_cols = [c for c in df.columns if c != 'trading_day']
    trading_day = df['trading_day'] if 'trading_day' in df.columns else None
    return df[price_cols], trading_day, price_cols


def build_summary_table(prices: pd.DataFrame, trading_day: pd.Series,
                          price_cols: list) -> pd.DataFrame:
    """50개 종목 요약 통계 테이블 생성"""
    # 1분봉 log-return
    log_ret_1m = np.log(prices).diff().iloc[1:]
    ts = prices.index[1:]

    if trading_day is not None:
        td = trading_day.reindex(ts, method='nearest').values
    else:
        td = pd.to_datetime(ts).date

    # 분석 시작일 필터
    mask = ts >= pd.Timestamp(ANALYSIS_START)
    log_ret_1m = log_ret_1m[mask]
    td_filt = td[mask]
    ts_filt = ts[mask]

    # 일별 log-return (일봉 close-to-close)
    prices_copy = prices.copy()
    prices_copy['_day'] = trading_day.values if trading_day is not None else prices_copy.index.date
    daily_last = prices_copy.groupby('_day').last()
    daily_log_ret = np.log(daily_last).diff().iloc[1:]
    daily_log_ret.index = pd.to_datetime(daily_log_ret.index)
    daily_log_ret = daily_log_ret[daily_log_ret.index >= ANALYSIS_START]

    # 일평균 거래 횟수 (비결측 1분봉 수)
    unique_days = pd.unique(td_filt)
    n_days = len(unique_days)

    rows = []
    for ticker in price_cols:
        r1m = log_ret_1m[ticker].dropna()
        r_daily = daily_log_ret[ticker].dropna() if ticker in daily_log_ret.columns else pd.Series()

        # 일평균 유효 분봉 수
        valid_per_day = []
        for day in unique_days:
            day_mask = td_filt == day
            r_day = log_ret_1m[ticker].values[day_mask]
            valid_per_day.append(np.sum(~np.isnan(r_day)))
        avg_ticks_per_day = float(np.mean(valid_per_day))

        # 일별 수익률 통계
        daily_mean = float(r_daily.mean()) if len(r_daily) > 0 else np.nan
        daily_std = float(r_daily.std()) if len(r_daily) > 0 else np.nan
        ann_mean = daily_mean * ANNUALIZATION
        ann_std = daily_std * np.sqrt(ANNUALIZATION)

        # 전체 기간 수익률
        if ticker in prices.columns:
            p_start = prices[ticker].dropna().iloc[0]
            p_end = prices[ticker].dropna().iloc[-1]
            total_return = float(np.log(p_end / p_start))
        else:
            total_return = np.nan

        rows.append({
            'ticker': ticker,
            'avg_ticks_per_day': round(avg_ticks_per_day, 1),
            'daily_mean': round(daily_mean * 100, 4) if not np.isnan(daily_mean) else np.nan,   # %
            'daily_std': round(daily_std * 100, 4) if not np.isnan(daily_std) else np.nan,     # %
            'ann_mean_pct': round(ann_mean * 100, 2) if not np.isnan(ann_mean) else np.nan,    # %
            'ann_std_pct': round(ann_std * 100, 2) if not np.isnan(ann_std) else np.nan,       # %
            'min_1m': round(float(r1m.min()) * 100, 4),  # %
            'max_1m': round(float(r1m.max()) * 100, 4),  # %
            'total_return_pct': round(total_return * 100, 2),
            'kurtosis_1m': round(float(stats.kurtosis(r1m)), 2) if len(r1m) > 3 else np.nan,
            'skewness_1m': round(float(stats.skew(r1m)), 3) if len(r1m) > 3 else np.nan,
            'n_obs_daily': len(r_daily),
        })

    df = pd.DataFrame(rows).set_index('ticker')
    return df


def plot_summary_interactive(summary_df: pd.DataFrame) -> go.Figure:
    """요약 통계 인터랙티브 scatter (연환산 수익률 vs 변동성)"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=summary_df['ann_std_pct'].values,
        y=summary_df['ann_mean_pct'].values,
        mode='markers+text',
        text=summary_df.index.tolist(),
        textposition='top center',
        marker=dict(
            color=summary_df['kurtosis_1m'].values,
            colorscale='RdBu_r',
            size=10,
            showscale=True,
            colorbar=dict(title='1분봉 Kurtosis'),
        ),
        hovertemplate=(
            '<b>%{text}</b><br>'
            'Ann. Std: %{x:.1f}%<br>'
            'Ann. Return: %{y:.1f}%<br>'
            '<extra></extra>'
        ),
    ))
    # 원점 기준 참조선
    fig.add_hline(y=0, line_dash='dash', line_color='gray', opacity=0.5)

    fig.update_layout(
        title='연환산 수익률 vs 변동성 (색상: 1분봉 Kurtosis)',
        xaxis_title='연환산 변동성 (%)',
        yaxis_title='연환산 수익률 (%)',
        template='plotly_white',
        height=550,
    )
    return fig


def generate_report(summary_df: pd.DataFrame, fig: go.Figure):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # CSV 저장
    csv_path = os.path.join(OUTPUT_DIR, 'summary_stats.csv')
    summary_df.to_csv(csv_path, encoding='utf-8-sig')
    print(f"[summary_stats] CSV 저장: {csv_path}")

    # HTML 저장
    fig_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
    table_html = summary_df.to_html(classes='table table-striped table-sm table-hover', border=0)

    col_desc = """
<ul>
<li><b>avg_ticks_per_day</b>: 일평균 유효 1분봉 수 (결측 제외)</li>
<li><b>daily_mean / daily_std</b>: 일별 log-return 평균 / 표준편차 (%)</li>
<li><b>ann_mean_pct / ann_std_pct</b>: 연환산 수익률 / 변동성 (×√365, %)</li>
<li><b>min_1m / max_1m</b>: 1분봉 최솟/최댓 log-return (%)</li>
<li><b>total_return_pct</b>: 전체 기간 누적 log-return (%)</li>
<li><b>kurtosis_1m / skewness_1m</b>: 1분봉 excess kurtosis / skewness</li>
<li><b>n_obs_daily</b>: 유효 일별 관측 수</li>
</ul>"""

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>Summary Stats Report</title>
<link rel="stylesheet"
  href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
<style>
  body {{ padding: 20px; }}
  h2 {{ margin-top: 40px; }}
  table {{ font-size: 0.85em; }}
</style>
</head>
<body>
<div class="container-fluid">
<h1>Phase 2 EDA — 50개 종목 요약 통계</h1>
<p class="text-muted">분석 기간: {ANALYSIS_START} ~ | 연환산 기준: √{ANNUALIZATION}</p>

<div class="mb-5">
<h2>1. 연환산 수익률 vs 변동성</h2>
{fig_html}
</div>

<div class="mb-5">
<h2>2. 종목별 요약 통계표</h2>
{col_desc}
{table_html}
</div>

</div>
</body>
</html>"""

    html_path = os.path.join(OUTPUT_DIR, 'summary_stats.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"[summary_stats] HTML 저장: {html_path}")
    return csv_path, html_path


if __name__ == '__main__':
    print("[summary_stats] 데이터 로드 중...")
    prices, trading_day, price_cols = load_prices()

    print("[summary_stats] 요약 통계 계산 중...")
    summary_df = build_summary_table(prices, trading_day, price_cols)

    print(f"  처리 종목 수: {len(summary_df)}")
    print(f"  평균 연환산 변동성: {summary_df['ann_std_pct'].mean():.1f}%")
    print(f"  평균 kurtosis(1분): {summary_df['kurtosis_1m'].mean():.2f}")

    print("[summary_stats] 차트 생성 중...")
    fig = plot_summary_interactive(summary_df)

    print("[summary_stats] 산출물 저장 중...")
    csv_path, html_path = generate_report(summary_df, fig)
    print(f"[summary_stats] 완료: {csv_path}, {html_path}")
