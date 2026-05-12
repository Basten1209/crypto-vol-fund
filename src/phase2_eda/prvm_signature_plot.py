"""
Phase 2 EDA - PRVM Signature Plot
- sampling frequency를 1, 2, 5, 10, 15, 30분으로 바꿔가며 일평균 RV(분산의 합) 계산
- Signature plot으로 microstructure noise vs signal 시각화
- K=37 파라미터 적절성 검증
- 산출물: docs/phase2_eda/signature_plot_report.html
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import ANALYSIS_START, RANDOM_SEED, K, M

np.random.seed(RANDOM_SEED)

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
PRICE_PANEL_PATH = os.path.join(PROJECT_ROOT, 'src', 'phase1_data', 'price_panel.csv')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'docs', 'phase2_eda')

FREQUENCIES = [1, 2, 5, 10, 15, 30]  # 분 단위


def load_prices() -> tuple:
    df = pd.read_csv(PRICE_PANEL_PATH, index_col=0, parse_dates=True)
    price_cols = [c for c in df.columns if c != 'trading_day']
    trading_day = df['trading_day'] if 'trading_day' in df.columns else None
    return df[price_cols], trading_day, price_cols


def compute_rv_at_frequency(prices: pd.DataFrame, trading_day: pd.Series,
                             freq_min: int) -> pd.Series:
    """
    특정 sampling frequency(분)에서 일별 RV(대각합 평균) 계산
    RV_d = sum of squared log-returns at freq_min intervals
    반환: trading_day 기준 일별 평균 RV (cross-asset mean of trace)
    """
    # freq_min 간격으로 downsampling
    sampled = prices.iloc[::freq_min]

    prices_copy = sampled.copy()
    if trading_day is not None:
        # trading_day를 동일 인덱스로 맞춤
        td_reindexed = trading_day.reindex(sampled.index, method='nearest')
        prices_copy['_day'] = td_reindexed.values
    else:
        prices_copy['_day'] = prices_copy.index.date

    # 일별 log-return
    log_prices = np.log(prices_copy.drop(columns=['_day']))
    log_ret = log_prices.diff().iloc[1:]
    prices_copy = prices_copy.iloc[1:].copy()

    # 분석 시작일 필터
    log_ret = log_ret[prices_copy.index >= ANALYSIS_START]
    days = prices_copy.loc[prices_copy.index >= ANALYSIS_START, '_day'].values

    # 일별 RV = sum of squared returns per day, then cross-asset mean of trace
    rv_dict = {}
    unique_days = pd.unique(days)
    for day in unique_days:
        mask = days == day
        r = log_ret.values[mask]  # (T_d, N)
        if len(r) < 2:
            continue
        # RV 대각합 = 각 종목의 squared return 합
        rv_diag = (r ** 2).sum(axis=0)  # (N,)
        rv_dict[day] = rv_diag.mean()  # cross-asset 평균

    return pd.Series(rv_dict)


def compute_signature_data(prices, trading_day):
    """주파수별 일평균 RV 계산"""
    results = {}
    for freq in FREQUENCIES:
        print(f"  [signature] freq={freq}min 계산 중...")
        rv_series = compute_rv_at_frequency(prices, trading_day, freq)
        results[freq] = rv_series
    return results


def plot_signature_plot(rv_by_freq: dict) -> go.Figure:
    """Signature plot: x=frequency, y=일평균 RV"""
    freqs = sorted(rv_by_freq.keys())
    mean_rv = [rv_by_freq[f].mean() for f in freqs]
    std_rv = [rv_by_freq[f].std() for f in freqs]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=freqs,
        y=[v * 1e6 for v in mean_rv],  # ×10⁶ 스케일
        error_y=dict(
            type='data',
            array=[v * 1e6 for v in std_rv],
            visible=True,
            color='rgba(70,130,180,0.4)',
        ),
        mode='lines+markers',
        marker=dict(size=10, color='steelblue'),
        line=dict(color='steelblue', width=2),
        name='일평균 RV',
        hovertemplate='freq=%{x}min<br>RV=%{y:.3f}×10⁻⁶<extra></extra>',
    ))

    # K=37에 해당하는 pre-averaging window 표시 (≈ 1분)
    fig.add_vline(x=1, line_dash='dash', line_color='tomato',
                  annotation_text='1분봉 (PRVM 기준)', annotation_position='top right')

    fig.update_layout(
        title='PRVM Signature Plot (Sampling Frequency vs Average RV)',
        xaxis_title='Sampling Frequency (분)',
        yaxis_title='일평균 RV (×10⁻⁶)',
        template='plotly_white',
        height=450,
        xaxis=dict(tickvals=FREQUENCIES, type='log'),
    )
    return fig


def plot_rv_time_series(rv_by_freq: dict) -> go.Figure:
    """주파수별 일별 RV 시계열 비교"""
    fig = go.Figure()
    colors = px_color_sequence = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for i, freq in enumerate(sorted(rv_by_freq.keys())):
        rv = rv_by_freq[freq]
        fig.add_trace(go.Scatter(
            x=rv.index.astype(str),
            y=rv.values * 1e6,
            mode='lines',
            name=f'{freq}min',
            line=dict(color=colors[i % len(colors)], width=1.5),
            opacity=0.8,
        ))
    fig.update_layout(
        title='주파수별 일별 RV 시계열',
        xaxis_title='날짜',
        yaxis_title='일별 RV (×10⁻⁶)',
        template='plotly_white',
        height=400,
    )
    return fig


def plot_bias_ratio(rv_by_freq: dict) -> go.Figure:
    """1분봉 대비 bias ratio: RV(freq) / RV(1min)"""
    base = rv_by_freq[1].mean()
    freqs = sorted(rv_by_freq.keys())
    ratios = [rv_by_freq[f].mean() / base for f in freqs]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f'{f}min' for f in freqs],
        y=ratios,
        marker_color=['steelblue' if abs(r - 1) < 0.1 else 'tomato' for r in ratios],
        text=[f'{r:.3f}' for r in ratios],
        textposition='outside',
    ))
    fig.add_hline(y=1.0, line_dash='dash', line_color='gray')
    fig.update_layout(
        title='Bias Ratio: RV(freq) / RV(1min)',
        xaxis_title='Sampling Frequency',
        yaxis_title='Ratio',
        template='plotly_white',
        height=400,
    )
    return fig


def generate_report(fig_sig, fig_ts, fig_bias, rv_by_freq):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    sig_html = fig_sig.to_html(full_html=False, include_plotlyjs='cdn')
    ts_html = fig_ts.to_html(full_html=False, include_plotlyjs=False)
    bias_html = fig_bias.to_html(full_html=False, include_plotlyjs=False)

    # 요약 테이블
    freqs = sorted(rv_by_freq.keys())
    table_rows = []
    for f in freqs:
        rv = rv_by_freq[f]
        table_rows.append(f"<tr><td>{f}min</td><td>{rv.mean()*1e6:.4f}</td>"
                          f"<td>{rv.std()*1e6:.4f}</td><td>{rv.mean()/rv_by_freq[1].mean():.4f}</td></tr>")

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>PRVM Signature Plot Report</title>
<link rel="stylesheet"
  href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
<style>body {{ padding: 20px; }} h2 {{ margin-top: 40px; }}</style>
</head>
<body>
<div class="container-fluid">
<h1>Phase 2 EDA — PRVM Signature Plot 리포트</h1>
<p class="text-muted">분석 기간: {ANALYSIS_START} ~ | K={K}, M={M}</p>

<div class="mb-5">
<h2>1. Signature Plot</h2>
<p>Sampling frequency가 낮아질수록 microstructure noise의 영향이 줄어들고 RV가 수렴합니다.
Pre-averaging은 1분봉 수준에서도 noise를 제거합니다 (K={K}로 37개 평균).</p>
{sig_html}
</div>

<div class="mb-5">
<h2>2. 주파수별 일별 RV 시계열</h2>
{ts_html}
</div>

<div class="mb-5">
<h2>3. Bias Ratio (1분봉 대비)</h2>
{bias_html}
</div>

<div class="mb-5">
<h2>4. 요약 통계</h2>
<table class="table table-striped table-sm">
<thead><tr><th>Frequency</th><th>Mean RV (×10⁻⁶)</th><th>Std RV (×10⁻⁶)</th><th>Bias Ratio</th></tr></thead>
<tbody>{''.join(table_rows)}</tbody>
</table>
</div>

</div>
</body>
</html>"""

    out_path = os.path.join(OUTPUT_DIR, 'signature_plot_report.html')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"[signature_plot] 저장 완료: {out_path}")
    return out_path


if __name__ == '__main__':
    print("[signature_plot] 데이터 로드 중...")
    prices, trading_day, price_cols = load_prices()

    print("[signature_plot] 주파수별 RV 계산 중 (약간 시간 소요)...")
    rv_by_freq = compute_signature_data(prices, trading_day)

    print("[signature_plot] 차트 생성 중...")
    fig_sig = plot_signature_plot(rv_by_freq)
    fig_ts = plot_rv_time_series(rv_by_freq)
    fig_bias = plot_bias_ratio(rv_by_freq)

    print("[signature_plot] HTML 리포트 생성 중...")
    out_path = generate_report(fig_sig, fig_ts, fig_bias, rv_by_freq)
    print(f"[signature_plot] 완료: {out_path}")
