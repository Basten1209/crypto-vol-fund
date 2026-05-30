# PPPD-Crypto: 변동성 기반 단기 공모펀드 구현 계획서

암호화폐 시장에서 변동성 행렬 기반 short-term portfolio fund를 구현하고 백테스팅한다.

---

## 0. 핵심 파이프라인

```
1분봉 OHLCV
   ↓
일별 Jump-adjusted PRVM (Pre-averaging Realized Volatility Matrix)
   ↓
Matrix-valued EWMA (직전일 jump volatility 가산)
   ↓
Long-only Minimum Variance Portfolio (LO-MVP)
   ↓
Monthly 7D / 14D hold-window 백테스팅
```

논문(`docs/references/FIVAR_Revision.pdf`, Shin et al. 2025)의 PRVM + portfolio 부분을 따르되, FIVAR / POET / Clustering은 제외한 단순화 버전.

---

## 1. 데이터 수집 (Phase 1)

### 거래소 / 빈도
- **Upbit 원화(KRW) 마켓 상품만** 대상
- **1분봉(1m)** 사용
- 다운로드 소스: <https://www.upbit.com/historical_data/download?prefix=candle> 의 자산별 1m monthly CSV

### 기간
- 다운로드: 2025-02 ~ 2026-03 monthly CSV
- 폴더 존재 확인용 범위: 2025-01 ~ 2026-04
- 분석 시작 일자: **2025-03-01**

### 종목 선정 기준 (모두 만족)
1. **Upbit 사이트에서 2025-02-01 ~ 2026-03-31 monthly CSV 폴더가 모두 존재** (즉, 2025-01 ~ 2026-04 폴더 모두 확인 후 2025-02 ~ 2026-03 다운로드)
2. **24시간 이상 연속 volume = 0인 적이 없음**
3. **Stablecoin 제외** (USDT, USDC, DAI 등 KRW 페그 자산)

### 종목 수
- 위 조건 만족 종목 중 **상위 50개** (거래대금 또는 시가총액 기준 — Phase 1에서 결정)

### 일봉 정의
- **Upbit 기준**: KST 09:00 ~ 다음날 09:00
- 모든 일별 통계(PRVM, daily return)는 이 cut을 따른다.

### 산출물
- **최종 가격 패널**: index = 1분 timestamp (KST), columns = 50개 종목명, value = price (close 단일값)
  - 즉, 시간 × 종목 2축의 wide-format DataFrame, price만 보유 (OHLCV 중 close 또는 합의된 단일 가격 컬럼)
  - log 변환은 후속 phase에서 수행
- 종목 메타데이터 (선정 사유, 거래대금 통계)
- 결측/이상치 처리 로그

---

## 2. EDA (Phase 2)

### 자산별 기초 통계
- **자산별 일평균 거래횟수**: 50종목 평균 / 분포 (유동성 sanity check)
- **단일 자산 요약 표**: 종목별 log-return mean, std (일/연환산), min/max, 기간 수익률
- **단일 자산 변동성 시계열**: 종목별 rolling 변동성 추이

### 분포 / 의존성 분석
- **log price return kurtosis plot**: 자산별 1분 / 일별 log-return kurtosis (heavy-tail 시각화)
- Volatility clustering: squared-return ACF
- Cross-asset correlation의 시간 변화 (월별)
- 거래량 / 유동성 분포 (종목 선정 정당화)

### 산출물
- EDA 노트북 / HTML 리포트
- 자산별 통계 요약 테이블 (CSV)

---

## 3. PRVM 추정 (Phase 3)

### 식
Jump-adjusted Pre-averaging Realized Volatility Matrix (Aït-Sahalia & Xiu 2016; FIVAR p.25):

$$\widehat{\Gamma}_d = \frac{1}{\psi K} \sum_{k=0}^{m-K} \overline{Y}_{d,k} \overline{Y}_{d,k}^\top \cdot \mathbf{1}\{|\overline{Y}_{d,k}| \le u_d\} - (\text{bias correction})$$

### 파라미터

| 파라미터 | 값 | 근거 |
|---------|-----|------|
| m | 1440 | 24h × 60min |
| K | floor(√m) = **37** | Pre-averaging optimal rate m^{−1/4} (Jacod et al. 2009) |
| g(x) | min(x, 1−x) | Weight function |
| ψ | 1/12 | ∫₀¹ g(x)² dx (수학 상수) |
| c_0 | 4 | Jump truncation 상수 (sample std × 4) |
| α_u | 0.235 | Truncation exponent (Mancini 계열, β ≤ 0.53 robust) |

### PSD projection
- eigenvalue < 1e-10 → 1e-10으로 floor (0으로 자르면 QLIKE의 log det에서 −∞)
- (M + Mᵀ) / 2로 symmetrize
- Reference: `src/reference_code/20250812_refactored.py:75`

### Jump volatility 분리
- raw PRVM (truncation 없음) − jump-adjusted PRVM = **JV_d** (jump component)
- 이후 portfolio 최적화 단계에서 사용

### 주의사항
- 참조 코드 `cal_prvm_final.py`의 hardcoded `371`은 m=390용. **m=1440 적용 시 `num_k = m − K + 1 = 1404`로 수정**.

### Crypto-specific 파라미터 검증
1. **PRVM signature plot**: sampling frequency를 1, 2, 5, 10, 15, 30분으로 바꿔가며 일평균 RV(대각합) 추이 확인 → K = 37의 적절성 검증
2. **Noise:signal ratio 추정**: Aït-Sahalia, Mykland & Zhang (2005) noise variance estimator로 종목별 추정
3. **Jump activity 검증**: jump component 시계열, 빈도/크기 분포 → c_0 = 4 적절성 판정

### 산출물
- 일별 PRVM dictionary `{date: np.ndarray(50, 50)}`
- 일별 JV dictionary `{date: np.ndarray(50, 50)}`
- Long-format CSV: `(date, ticker_i, ticker_j, value)`
- K, c_0 등 PRVM 파라미터의 crypto-specific 적절성 판정 리포트

---

## 4. EWMA 변동성 예측 (Phase 4)

### Matrix-valued EWMA
For target day \(d+1\), use only the previous 28 daily PRVM matrices:

$$\hat{\Sigma}_{d+1|d} =
\frac{\sum_{\ell=0}^{27} \lambda^\ell \cdot \text{PRVM}_{d-\ell}}
{\sum_{\ell=0}^{27} \lambda^\ell}$$

### 파라미터
- **λ = 0.94** (RiskMetrics 표준, 고정. 최근 PRVM에 더 큰 가중치 부여)
- **Rolling window = 28일**: 각 target date마다 직전 28일 PRVM만 사용
- **첫 forecast target**: 2025-03-01, window = 2025-02-01 ~ 2025-02-28
- 29일 이전 데이터는 해당 target forecast에서 제외하고, 28일 window 안의 EWMA weights는 합이 1이 되도록 정규화
- 매트릭스 EWMA는 PSD 보존 (PSD 행렬의 convex combination)

### 평가 메트릭
- **MSPE**: $\frac{1}{T}\sum \|\hat{\Sigma}_{d+1|d} - \text{PRVM}_{d+1}\|_F^2$ (×10⁴ 보고)
- **QLIKE**: $\frac{1}{T}\sum [\log\det(\hat{\Sigma}_{d+1|d}) + \mathrm{tr}(\hat{\Sigma}_{d+1|d}^{-1} \cdot \text{PRVM}_{d+1})]$ (×10⁻³ 보고)
- Ground truth proxy: **PRVM_{d+1}** 자체 (논문 방식)

### 산출물
- 일별 forecast Σ̂_{d+1|d} dictionary
- MSPE / QLIKE 시계열

---

## 5. 포트폴리오 최적화 (Phase 5)

### Long-only Minimum Variance Portfolio
논문 식(p.31) 그대로:

$$\min_\omega \omega^\top (\hat{\Sigma}_{d+1|d} + \widehat{JV}_{d-1}) \omega \quad \text{s.t.} \quad \omega^\top \mathbf{1} = 1, \quad \|\omega\|_1 \le c_0$$

- **c_0 = 1**: long-only 자동 강제 (ω ≥ 0)
- 잔여 가중치 합 = 1 (현금 보유 X)
- **JV_{d-1}**: 직전일 jump volatility를 다음날 jump 예측으로 사용 (논문 그대로)

### 단일종목 cap
- 논문 일관성 확인을 위한 uncapped variant와 상품화 가능성을 위한 **25% cap variant**를 모두 산출한다.
- Dashboard와 최종 데모는 concentration risk를 통제한 monthly cap25 variant를 기본으로 사용한다.

### 월초 단기 상품 운용 구조
| Window | Forecast 기준 | Hold | 운용 가정 |
|-------|----------|------|----------|
| **7D** | 매월 첫 available date의 직전 28일 PRVM | 월초 7일 | 동일 초기 AUM으로 새 상품 시작 |
| **14D** | 매월 첫 available date의 직전 28일 PRVM | 월초 14일 | 동일 초기 AUM으로 새 상품 시작 |

- 7D / 14D 상품을 동시에 평가한다.
- **Simple Mode**: 월초 target weight로 진입한 뒤 hold window 동안 weight drift를 허용한다.
- **Managed Mode**: hold window 동안 매일 target weight로 되돌리는 일 단위 리밸런싱을 수행한다.
- 각 월은 동일한 초기 AUM으로 시작하는 독립 상품으로 해석하며, off-window 날짜는 성과표에 포함하지 않는다.

### 거래비용
- **무시** (펀드 컨셉 시연 단계)

### 산출물
- Cycle별 weight 시계열 `{date: ω(50,)}`
- Cycle별 active asset 수 / top-weight 시계열

---

## 6. 백테스팅 / 평가 (Phase 6)

### 평가 메트릭

| 항목 | 정의 |
|------|------|
| Realized portfolio risk | $\sqrt{\sum_{k=1}^{144} (\omega^\top \Delta Y_{d,k}^{10\text{-min}})^2}$, 연환산 × √365 |
| Total return on hold windows | 각 월 7D / 14D active hold-window 일별 수익률을 연결한 복리 수익률 |
| Mean monthly hold return | 월별 독립 상품 수익률의 단순 평균 |
| Annualized return / volatility | active hold-window 일별 수익률만 사용, 365일 기준 연환산 |
| Sharpe ratio | active hold-window 일별 수익률, r_f = 0 가정 |
| Max drawdown | active hold-window equity curve 기준 |
| Positive month rate | 월별 상품 수익률이 양수인 월 비율 |

- **연환산**: √365 (24/7 거래)
- **평가 frequency**: 10분 portfolio return (1분은 microstructure noise 큼, 논문 p.32 그대로)
- 하루 144 = 1440 / 10 구간
- **Dashboard 표시 원칙**: drawdown chart와 volatility KPI를 분리한다. Sharpe KPI는 선택 날짜의 짧은 partial window가 아니라 Phase 6 active hold-window 전체 성과표를 사용해 짧은 샘플의 연율화 왜곡을 피한다.

### Benchmark
1. **Equal-weight (EW)**: minimum variance portfolio와 동일한 50종목, 동일 cycle/리밸런싱 규칙
2. **BTC same-window return**: 월별 hold-window 수익률 표에서 시장 reference로만 사용

### 통계적 검정
- **Diebold-Mariano test**: active hold-window의 squared daily return loss를 사용해 minimum variance와 equal-weight의 realized risk 차이를 검정한다.

### Sanity check 자동화
- PRVM_d trace 시계열 (총 분산 spike 검출)
- Σ̂ EWMA의 condition number / top-k eigenvalue 시계열
- Minimum variance portfolio ω의 max(|ω|) 시계열 → 단일종목 cap 도입 여부 판단 근거

### 산출물
- Hold-window equity curve (cycle / mode별)
- Hold-window drawdown chart
- Risk vs benchmark 비교 차트
- active hold-window 평가 테이블
- active hold-window DM test 결과
- Final report

---

## 7. 핵심 결정사항 / 미결정 사항

### 확정
| 항목 | 결정 |
|------|------|
| 데이터 기간 | 2025-02-01 ~ 2026-02-28 (수집), 2025-03-01 시작 |
| 자산 수 | 50개 |
| Stablecoin | 제외 |
| 일봉 cut | KST 09:00 |
| PSD projection | eigenvalue floor 1e-10 |
| 거래비용 | 무시 |
| Cycle | 1주 + 2주 동시 |
| EWMA λ | 0.94 (고정) |
| Minimum variance 제약 | c_0 = 1 (long-only) |
| 단일종목 cap | 없음 (재고 가능) |
| Benchmark | BTC HODL + Equal-weight |
| 연환산 | √365 |
| 평가 frequency | 10분 |

### Phase 별 검증 후 결정
- Phase 3 검증 결과: K = 37 적절성 → 필요시 grid search
- Phase 3 검증 결과: c_0 = 4 (jump) 적절성 → 필요시 5~6 상향
- Phase 6 결과: 단일종목 cap 도입 여부 (max weight 분포 보고)
- Phase 6 결과: subsample 분리 평가 여부 (구조적 break 발견 시)

---

## 8. 역할 분담

| 담당자 | 담당 Phase | 책임 |
|--------|-----------|------|
| **@iron-4842** | Phase 1, 2, 6 (보조) | 데이터 수집/전처리, EDA, 결과 해석 보조 |
| **@basten1209** | Phase 3, 4, 5, 6 | PRVM, EWMA, 포트폴리오 최적화, 백테스팅 |

---

## 9. 참고 문헌

### 핵심
- **Shin, M., Kim, D., Wang, Y., & Fan, J. (2025)**, *Factor and Idiosyncratic VAR Volatility Matrix Models for Heavy-Tailed High-Frequency Financial Observations* — `docs/references/FIVAR_Revision.pdf`

### PRVM
- Jacod, J., Li, Y., Mykland, P. A., Podolskij, M., & Vetter, M. (2009)
- Christensen, K., Kinnebrock, S., & Podolskij, M. (2010)
- Aït-Sahalia, Y., & Xiu, D. (2016, 2017)

### Jump truncation
- Mancini, C. (2009)

### Long-only minimum variance sparsity
- Jagannathan, R., & Ma, T. (2003), *Risk Reduction in Large Portfolios: Why Imposing the Wrong Constraints Helps*

### 평가 메트릭
- Patton, A. J. (2011)
- Hansen, P. R., & Lunde, A. (2006a)
- Diebold, F. X., & Mariano, R. S. (1995)

### EWMA
- RiskMetrics Technical Document (1996)
