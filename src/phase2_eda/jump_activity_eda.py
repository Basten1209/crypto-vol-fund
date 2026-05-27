"""
Phase 2 EDA - Jump Activity 분석
- jump 기준: |1분 log-return| > c0 * sigma_hat * m^alpha_u
- 일별 jump 발생 횟수, 크기 분포 분석
- jump component 시계열 plot (BTC, ETH, XRP)
- c0=4 적절성 판단
- 산출물: docs/phase2_eda/jump_activity_report.html
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import (ANALYSIS_START, RANDOM_SEED, JUMP_C0, JUMP_ALPHA_U,
                         M, DAILY_CUT_KST)

np.random.seed(RANDOM_SEED)

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
PRICE_PANEL_PATH = os.path.join(PROJECT_ROOT, 'src', 'phase1_data', 'price_panel.csv')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'docs', 'phase2_eda')

REP_TICKERS = ['BTC', 'ETH', 'XRP']


def load_prices():
    df = pd.read_csv(PRICE_PANEL_PATH, index_col=0, parse_dates=True)
    price_cols = [c for c in df.columns if c != 'trading_day']
    trading_day = df['trading_day'] if 'trading_day' in df.columns else None
    return df[price_cols], trading_day, price_cols


def compute_jump_threshold(r: np.ndarray, c0: float = JUMP_C0) -> float:
    """
    Jump truncation threshold (EDA용 단순 4-sigma rule):
      u = c0 * sigma_hat
    sigma_hat: bipower variation 기반 per-minute 분산 추정 (robust to jumps)
    참고: PRVM에서는 pre-averaged return에 별도 스케일링을 추가하지만,
          EDA에서는 raw 1분봉 기준 4-sigma를 사용한다.
    """
    n = len(r)
    if n < 4:
        return np.inf
    # Bipower variation: per-minute 분산 추정 (jump에 robust)
    bv = np.sum(np.abs(r[:-1]) * np.abs(r[1:])) * np.pi / 2 / (n - 1)
    sigma_hat = np.sqrt(bv)
    u = c0 * sigma_hat
    return u


def detect_jumps_day(r: np.ndarray, c0: float = JUMP_C0) -> dict:
    """
    하루 1분봉 log-return에서 jump 감지 (EDA용)
    반환: jump mask, jump count, jump sizes, threshold
    """
    u = compute_jump_threshold(r, c0=c0)
    jump_mask = np.abs(r) > u

    return {
        'jump_mask': jump_mask,
        'jump_count': int(jump_mask.sum()),
        'jump_sizes': np.abs(r[jump_mask]),
        'threshold': u,
        'jump_fraction': jump_mask.mean(),
    }


def compute_daily_jump_stats(prices: pd.DataFrame, trading_day: pd.Series,
                               price_cols: list) -> pd.DataFrame:
    """전 종목 일별 jump 통계 계산"""
    # log-return 계산
    log_ret = np.log(prices).diff().iloc[1:]
    ts = prices.index[1:]

    if trading_day is not None:
        td = trading_day.reindex(ts, method='nearest').values
    else:
        td = pd.to_datetime(ts).date

    mask = ts >= pd.Timestamp(ANALYSIS_START)
    log_ret_filt = log_ret.values[mask]
    td_filt = td[mask]

    results = []
    unique_days = pd.unique(td_filt)

    for day in unique_days:
        day_mask = td_filt == day
        r_day = log_ret_filt[day_mask]  # (T_d, N)
        n_tickers = r_day.shape[1]

        jump_counts = []
        jump_fracs = []
        max_jump_sizes = []

        for j in range(n_tickers):
            r_j = r_day[:, j]
            r_j = r_j[~np.isnan(r_j)]
            if len(r_j) < 4:
                jump_counts.append(0)
                jump_fracs.append(0.0)
                max_jump_sizes.append(np.nan)
                continue
            info = detect_jumps_day(r_j)
            jump_counts.append(info['jump_count'])
            jump_fracs.append(info['jump_fraction'])
            max_jump_sizes.append(info['jump_sizes'].max() if len(info['jump_sizes']) > 0 else 0.0)

        results.append({
            'date': day,
            'mean_jump_count': np.mean(jump_counts),
            'total_jump_count': int(np.sum(jump_counts)),
            'mean_jump_frac': np.mean(jump_fracs),
            'max_jump_size': np.nanmax(max_jump_sizes) if max_jump_sizes else np.nan,
        })

    return pd.DataFrame(results).set_index('date')


def compute_ticker_jump_series(prices: pd.DataFrame, trading_day: pd.Series,
                                ticker: str) -> dict:
    """단일 종목의 일별 jump series 및 jump component 계산"""
    log_p = np.log(prices[ticker].values)
    r = np.diff(log_p)
    ts = prices.index[1:]

    if trading_day is not None:
        td = trading_day.reindex(ts, method='nearest').values
    else:
        td = pd.to_datetime(ts).date

    mask = ts >= pd.Timestamp(ANALYSIS_START)
    r = r[mask]
    td = td[mask]
    ts_filt = ts[mask]

    jump_days = []
    jv_series = {}
    c0_sensitivity = {c: [] for c in [2, 3, 4, 5, 6]}

    unique_days = pd.unique(td)
    for day in unique_days:
        day_mask = td == day
        r_day = r[day_mask]
        r_day = r_day[~np.isnan(r_day)]
        if len(r_day) < 4:
            continue

        info = detect_jumps_day(r_day)
        jv = float((r_day[info['jump_mask']] ** 2).sum())  # jump variance
        jv_series[day] = jv

        if info['jump_count'] > 0:
            jump_days.append({'date': day,
                               'jump_count': info['jump_count'],
                               'max_size': info['jump_sizes'].max()})

        # c0 민감도: c0 값에 따라 jump fraction이 어떻게 변하는지
        for c in c0_sensitivity:
            info_c = detect_jumps_day(r_day, c0=c)
            c0_sensitivity[c].append(info_c['jump_fraction'])

    return {
        'jv_series': pd.Series(jv_series),
        'jump_days': pd.DataFrame(jump_days) if jump_days else pd.DataFrame(),
        'c0_sensitivity': {c: np.mean(v) for c, v in c0_sensitivity.items()},
    }


def plot_jump_time_series(ticker_data: dict, tickers: list) -> go.Figure:
    """종목별 일별 jump variance 시계열"""
    n = len(tickers)
    fig = make_subplots(rows=n, cols=1,
                         subplot_titles=[f'{t} 일별 Jump Variance' for t in tickers],
                         vertical_spacing=0.08)
    colors = ['steelblue', 'tomato', 'green']
    for i, ticker in enumerate(tickers, 1):
        jv = ticker_data[ticker]['jv_series']
        fig.add_trace(go.Scatter(
            x=jv.index.astype(str),
            y=jv.values * 1e6,
            mode='lines',
            name=ticker,
            line=dict(color=colors[i - 1], width=1.5),
            fill='tozeroy',
            fillcolor=colors[i - 1].replace(')', ',0.2)').replace('rgb', 'rgba')
                if 'rgb' in colors[i-1] else f'rgba(70,130,180,0.2)',
        ), row=i, col=1)
        fig.update_yaxes(title_text='JV (×10⁻⁶)', row=i, col=1)

    fig.update_layout(
        title='대표 종목 Jump Variance 시계열',
        height=250 * n,
        template='plotly_white',
        showlegend=True,
    )
    return fig


def plot_c0_sensitivity(ticker_data: dict, tickers: list) -> go.Figure:
    """c0 파라미터에 따른 jump fraction 변화"""
    fig = go.Figure()
    colors = ['steelblue', 'tomato', 'green']
    for i, ticker in enumerate(tickers):
        sensitivity = ticker_data[ticker]['c0_sensitivity']
        c_vals = sorted(sensitivity.keys())
        fracs = [sensitivity[c] * 100 for c in c_vals]  # % 단위
        fig.add_trace(go.Scatter(
            x=c_vals,
            y=fracs,
            mode='lines+markers',
            name=ticker,
            line=dict(color=colors[i], width=2),
            marker=dict(size=8),
        ))

    # c0=4 강조
    fig.add_vline(x=JUMP_C0, line_dash='dash', line_color='gray',
                  annotation_text=f'c0={JUMP_C0} (기준)', annotation_position='top right')

    fig.update_layout(
        title=f'c0 파라미터 민감도 분석 (Jump Fraction)',
        xaxis_title='c0',
        yaxis_title='Jump Fraction (%)',
        template='plotly_white',
        height=400,
    )
    return fig


def plot_jump_count_distribution(daily_stats: pd.DataFrame) -> go.Figure:
    """일별 jump 발생 횟수 분포"""
    fig = make_subplots(rows=1, cols=2,
                         subplot_titles=['일별 평균 Jump 횟수 시계열', 'Jump 횟수 히스토그램'])

    fig.add_trace(go.Scatter(
        x=daily_stats.index.astype(str),
        y=daily_stats['mean_jump_count'].values,
        mode='lines',
        name='평균 jump 횟수',
        line=dict(color='steelblue', width=1.5),
        fill='tozeroy',
        fillcolor='rgba(70,130,180,0.2)',
    ), row=1, col=1)

    fig.add_trace(go.Histogram(
        x=daily_stats['mean_jump_count'].values,
        nbinsx=30,
        name='분포',
        marker_color='steelblue',
        opacity=0.7,
    ), row=1, col=2)

    fig.update_layout(
        title='일별 Jump 발생 횟수 분포',
        height=400,
        template='plotly_white',
    )
    return fig


def generate_report(fig_ts, fig_c0, fig_dist, daily_stats, ticker_data, tickers):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ts_html = fig_ts.to_html(full_html=False, include_plotlyjs='cdn')
    c0_html = fig_c0.to_html(full_html=False, include_plotlyjs=False)
    dist_html = fig_dist.to_html(full_html=False, include_plotlyjs=False)

    # c0 적절성 판단
    sensitivity_rows = []
    for c in sorted([2, 3, 4, 5, 6]):
        fracs = []
        for t in tickers:
            sens = ticker_data[t]['c0_sensitivity']
            if c in sens:
                fracs.append(sens[c] * 100)
        mean_frac = np.mean(fracs) if fracs else np.nan
        flag = '← <strong>현재 설정</strong>' if c == JUMP_C0 else ''
        sensitivity_rows.append(
            f"<tr><td>{c}</td><td>{mean_frac:.3f}%</td><td>{flag}</td></tr>"
        )

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>Jump Activity Report</title>
<link rel="stylesheet"
  href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
<style>body {{ padding: 20px; }} h2 {{ margin-top: 40px; }}</style>
</head>
<body>
<div class="container-fluid">
<h1>Phase 2 EDA — Jump Activity 분석 리포트</h1>
<p class="text-muted">분석 기간: {ANALYSIS_START} ~ | c0={JUMP_C0}, alpha_u={JUMP_ALPHA_U}, M={M}</p>

<div class="mb-5">
<h2>1. Jump Variance 시계열 (대표 종목: {', '.join(tickers)})</h2>
{ts_html}
</div>

<div class="mb-5">
<h2>2. c0 파라미터 민감도 분석</h2>
<p>c0가 낮을수록 더 많은 수익률을 jump로 분류합니다.
학계 권장치는 c0=4~6 범위이며, 본 프로젝트는 c0={JUMP_C0}을 사용합니다.</p>
{c0_html}
<table class="table table-sm table-striped mt-3" style="max-width:400px">
<thead><tr><th>c0</th><th>평균 Jump Fraction</th><th>비고</th></tr></thead>
<tbody>{''.join(sensitivity_rows)}</tbody>
</table>
</div>

<div class="mb-5">
<h2>3. 일별 Jump 발생 횟수 분포</h2>
{dist_html}
</div>

</div>
</body>
</html>"""

    out_path = os.path.join(OUTPUT_DIR, 'jump_activity_report.html')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"[jump_activity] 저장 완료: {out_path}")
    return out_path


if __name__ == '__main__':
    print("[jump_activity] 데이터 로드 중...")
    prices, trading_day, price_cols = load_prices()

    print("[jump_activity] 일별 jump 통계 계산 중...")
    daily_stats = compute_daily_jump_stats(prices, trading_day, price_cols)

    print(f"  평균 일별 jump 횟수: {daily_stats['mean_jump_count'].mean():.2f}")
    print(f"  평균 jump fraction: {daily_stats['mean_jump_frac'].mean()*100:.3f}%")

    # 대표 종목 (존재하는 것만)
    rep_tickers = [t for t in REP_TICKERS if t in price_cols]

    print("[jump_activity] 종목별 jump 시계열 계산 중...")
    ticker_data = {}
    for ticker in rep_tickers:
        print(f"  {ticker} 처리 중...")
        ticker_data[ticker] = compute_ticker_jump_series(prices, trading_day, ticker)

    print("[jump_activity] 차트 생성 중...")
    fig_ts = plot_jump_time_series(ticker_data, rep_tickers)
    fig_c0 = plot_c0_sensitivity(ticker_data, rep_tickers)
    fig_dist = plot_jump_count_distribution(daily_stats)

    print("[jump_activity] HTML 리포트 생성 중...")
    out_path = generate_report(fig_ts, fig_c0, fig_dist, daily_stats, ticker_data, rep_tickers)
    print(f"[jump_activity] 완료: {out_path}")
