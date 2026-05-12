# Phase 2 EDA 산출물 설명

분석 기간: 2025-03-01 ~ 2026-03-31  
대상 종목: 50개 (Upbit KRW 마켓, stablecoin 제외)  
입력 데이터: `src/phase1_data/price_panel.csv` (1분봉 가격 패널)

---

## 산출물 목록

### 1. `distribution_report.html`
**생성 모듈**: `src/phase2_eda/distribution_eda.py`

| 차트 | 설명 |
|------|------|
| Kurtosis 분포 boxplot | 종목별 1분봉 excess kurtosis 분포. 정규분포 기준선(0) 대비 heavy-tail 여부 확인 |
| 1분봉 vs 일별 kurtosis bar chart | 동일 종목의 frequency에 따른 kurtosis 비교 (일별 집계 시 중심극한정리로 감소) |
| Kurtosis vs 변동성 scatter | std 대비 kurtosis 분포 구조 (색상=kurtosis) |
| Squared-return ACF | 대표 5종목(BTC, ETH, XRP, SOL, DOGE)의 제곱 수익률 자기상관. 양의 ACF → volatility clustering 존재 |
| 기초 통계 테이블 | 종목별 mean, std, skewness, kurtosis, min, max |

**주요 발견**: 1분봉 평균 kurtosis ≈ 179 (정규분포 0 대비 극단적 heavy-tail), 일별 기준 ≈ 14.

---

### 2. `signature_plot_report.html`
**생성 모듈**: `src/phase2_eda/prvm_signature_plot.py`

| 차트 | 설명 |
|------|------|
| Signature Plot | sampling frequency(1/2/5/10/15/30분) vs 일평균 RV. 고주파일수록 microstructure noise 영향 증가 |
| 주파수별 RV 시계열 | 각 frequency에서 일별 RV 추이 비교 |
| Bias Ratio | 1분봉 대비 각 frequency의 RV 비율 (1에 가까울수록 noise 없음) |
| 요약 테이블 | frequency별 mean/std RV, bias ratio |

**목적**: K=37 pre-averaging 윈도우의 적절성 검증. Signature plot에서 고주파(1분)에서 RV가 높고 저주파로 갈수록 수렴하면 microstructure noise가 존재함을 의미.

---

### 3. `noise_signal_report.html`
**생성 모듈**: `src/phase2_eda/noise_signal_estimator.py`

| 차트 | 설명 |
|------|------|
| 종목별 평균 noise-signal ratio | Aït-Sahalia et al. (2005) 추정치. 낮을수록 데이터 품질 양호 |
| 일별 ratio 시계열 | cross-asset 중앙값 및 IQR (shaded band) |
| 종목별 요약 테이블 | mean/median ratio, noise variance |

**방법론**: `q² = -E[r_t * r_{t+1}]` (lag-1 autocovariance, AMZ 2005 Proposition 1)  
Noise-to-signal ratio = 2·m·q² / IV

---

### 4. `jump_activity_report.html`
**생성 모듈**: `src/phase2_eda/jump_activity_eda.py`

| 차트 | 설명 |
|------|------|
| Jump Variance 시계열 | BTC/ETH/XRP의 일별 jump component variance |
| c0 파라미터 민감도 | c0 = 2, 3, 4, 5, 6에 따른 jump fraction 변화. c0=4 적절성 판단 |
| Jump 횟수 시계열 | 일별 평균 jump 발생 횟수 |
| Jump 횟수 히스토그램 | jump 빈도 분포 |

**Jump 기준**: `|r_t| > c0 × σ̂` (EDA용 4-sigma rule, σ̂은 bipower variation 기반 per-minute 추정치)  
**주요 발견**: c0=4 기준 일평균 jump fraction ≈ 2.7% (약 38회/일·종목). c0 증가 시 단조감소.

---

### 5. `summary_stats.csv` / `summary_stats.html`
**생성 모듈**: `src/phase2_eda/summary_stats.py`

| 컬럼 | 설명 |
|------|------|
| `avg_ticks_per_day` | 일평균 유효 1분봉 수 (결측 제외) |
| `daily_mean` / `daily_std` | 일별 log-return 평균 / 표준편차 (%) |
| `ann_mean_pct` / `ann_std_pct` | 연환산 수익률 / 변동성 (×√365, %) |
| `min_1m` / `max_1m` | 1분봉 최소/최대 log-return (%) |
| `total_return_pct` | 전체 기간 누적 log-return (%) |
| `kurtosis_1m` | 1분봉 excess kurtosis |
| `skewness_1m` | 1분봉 skewness |
| `n_obs_daily` | 유효 일별 관측 수 |

HTML 리포트에는 연환산 수익률 vs 변동성 인터랙티브 scatter (색상=kurtosis) 포함.

---

## 재실행 방법

```bash
# 프로젝트 루트에서 실행
python3 -m src.phase2_eda.distribution_eda
python3 -m src.phase2_eda.prvm_signature_plot
python3 -m src.phase2_eda.noise_signal_estimator
python3 -m src.phase2_eda.jump_activity_eda
python3 -m src.phase2_eda.summary_stats
```

---

## EDA 결과 요약 (Sanity Check)

| 항목 | 결과 | 판정 |
|------|------|------|
| 1분봉 평균 kurtosis | ≈ 179 | ✅ Heavy-tail 확인 (정규분포 = 0) |
| 일별 평균 kurtosis | ≈ 14 | ✅ Heavy-tail 완화되나 여전히 높음 |
| Volatility clustering | ACF(r²) 양수, 천천히 감소 | ✅ EWMA 모델 적합 |
| Jump fraction (c0=4) | ≈ 2.7% | ✅ 합리적 수준 (c0=4 적절) |
| Signature plot | 1분 > 30분 RV | ✅ Pre-averaging K=37 필요성 확인 |
