# CLAUDE.md

이 파일은 본 repository에서 Claude Code (또는 다른 AI agent)가 작업할 때 참조하는 가이드입니다. 협업자에게도 빠른 navigation 자료로 활용됩니다.

---

## 프로젝트 한 줄 요약

**Upbit 1분봉 → Jump-adjusted PRVM → Matrix EWMA → Long-only Minimum Variance Portfolio** 파이프라인으로 암호화폐 단기 포트폴리오 펀드를 백테스팅한다. 메인 reference는 Shin et al. (2025) FIVAR 논문이며, 본 프로젝트는 FIVAR/POET/Clustering을 제외한 단순화 버전이다.

상세 내용은 [`PROJECT_PLAN.md`](PROJECT_PLAN.md) 참조.

---

## Repository 구조

```
.
├── CLAUDE.md                   # 이 파일 — AI agent 및 협업자 가이드
├── PROJECT_PLAN.md             # 프로젝트 진행 계획서 (메인 문서)
├── README.md                   # (선택) 일반 소개
│
├── src/
│   ├── reference_code/         # 기존 참조 구현 (수정 금지, 읽기 전용)
│   ├── config.py               # 전역 파라미터 (이하 모든 모듈이 import)
│   ├── utils.py                # 공통 유틸 (PSD projection, I/O 등)
│   │
│   ├── phase1_data/            # 데이터 수집/전처리 — @iron-4842
│   │   ├── data_collector.py
│   │   ├── asset_filter.py
│   │   └── data_preprocessor.py
│   │
│   ├── phase2_eda/             # EDA — @iron-4842
│   │   ├── distribution_eda.py
│   │   ├── prvm_signature_plot.py
│   │   ├── noise_signal_estimator.py
│   │   └── jump_activity_eda.py
│   │
│   ├── phase3_prvm/            # PRVM 추정 — @basten1209
│   │   ├── prvm_calculator.py
│   │   └── jump_separator.py
│   │
│   ├── phase4_ewma/            # EWMA 예측 — @basten1209
│   │   ├── matrix_ewma.py
│   │   └── forecast_evaluator.py
│   │
│   ├── phase5_portfolio/       # GMV 최적화 — @basten1209
│   │   └── lo_mvp_optimizer.py
│   │
│   └── phase6_backtest/        # 백테스팅 / 평가 — @basten1209 (해석 보조 @iron-4842)
│       ├── walk_forward.py
│       ├── benchmarks.py
│       ├── metrics.py
│       ├── dm_test.py
│       └── visualization.py
│
└── docs/
    └── references/             # 참고 논문
        └── FIVAR_Revision.pdf
```

각 Phase 디렉토리는 본격 구현 시점에 생성. 기본 구조만 잡혀있고 디테일은 담당자가 채움.

---

## 핵심 파라미터 (`src/config.py`에서 중앙 관리)

```python
# === 데이터 ===
START_DATE        = "2025-02-01"
END_DATE          = "2026-02-28"
ANALYSIS_START    = "2025-03-01"
DAILY_CUT_KST     = "09:00"
N_ASSETS          = 50
EXCLUDE_STABLE    = True           # USDT, USDC, DAI 등 제외

# === PRVM ===
M                 = 1440           # 24h × 60min
K                 = 37             # floor(sqrt(M))
PSI               = 1/12           # weight integral
JUMP_C0           = 4              # jump truncation 상수
JUMP_ALPHA_U      = 0.235          # jump truncation exponent

# === PSD projection ===
PSD_FLOOR         = 1e-10

# === EWMA ===
LAMBDA            = 0.94           # RiskMetrics 표준
WINDOW            = 28             # EWMA 초기값 산출 기간
EWMA_INIT_DAYS    = 28             # 초기값 sample mean 일수

# === Portfolio ===
GMV_C0            = 1              # gross exposure (long-only)
SINGLE_ASSET_CAP  = None           # 결과 보고 재검토

# === Backtest ===
CYCLES            = [7, 14]        # 1주, 2주 동시 운용
EVAL_FREQ_MIN     = 10             # 평가용 frequency
EVAL_INTERVALS    = 144            # 1440 / 10
ANNUALIZATION     = 365            # 24/7 거래
RISK_FREE_RATE    = 0.0
```

---

## 작업 순서 / Phase 의존성

```
Phase 1 (데이터)
   ↓ log-price DataFrame, 종목 메타데이터
Phase 2 (EDA)
   ↓ K, c_0 검증 결과 (필요시 config 업데이트)
Phase 3 (PRVM)
   ↓ daily PRVM dict, daily JV dict
Phase 4 (EWMA)
   ↓ daily forecast Σ̂_{d+1|d}
Phase 5 (Portfolio)
   ↓ cycle별 ω weight 시계열
Phase 6 (Backtest)
   ↓ equity curve, metrics 테이블, DM test
```

각 Phase는 **독립 실행 가능**해야 한다 (산출물은 디스크에 저장하고 다음 Phase가 read).

---

## 역할 분담

| 담당자 | Phase | 작업 영역 |
|--------|-------|----------|
| **@iron-4842** | Phase 1, 2, 6 (보조) | 데이터 수집/전처리, EDA, 결과 해석 |
| **@basten1209** | Phase 3, 4, 5, 6 | PRVM, EWMA, 포트폴리오, 백테스팅 |

Phase 6은 두 명이 협업: @basten1209가 메트릭 산출, @iron-4842가 도메인 해석.

---

## Reference Code 사용 규칙

1. **`src/reference_code/`는 읽기 전용**. 직접 수정 금지. 로직만 참고하여 새 모듈 작성.
2. 핵심 참조 파일:
   | 파일 | 역할 |
   |------|------|
   | `cal_prvm_final.py` | PRVM 핵심 계산 (`_calculate_prvm_for_day_optimized`) |
   | `20250812_refactored.py` | PSD projection (`project_psd`, line 75) |
   | `poet-prvm_final.py` | MSPE / QLIKE 계산 |
3. **주의**: `cal_prvm_final.py`의 hardcoded `371`은 m=390용. m=1440 적용 시 `num_k = m − K + 1 = 1404`로 수정.

---

## 핵심 결정사항 (FAQ)

**Q: 왜 1분봉인가?**
A: 거래자산 평균 거래 횟수가 분당 다회 → sync 단위로 적절. 논문 PRVM의 권장 frequency.

**Q: 왜 m=1440 / K=37인가?**
A: 24h × 60min = 1440. K = floor(√1440) ≈ 37 (Jacod et al. 2009 theoretical optimal rate, m^{−1/4} 수렴).

**Q: PSD projection 시 eigenvalue를 0이 아닌 1e-10으로 floor하는 이유?**
A: QLIKE 계산 시 log det(Σ) 필요. eigenvalue가 0이면 −∞ → numerical error.

**Q: λ = 0.94 고정 이유?**
A: RiskMetrics 표준. 첫 28일 PRVM 평균으로 초기화한 뒤 recursive EWMA로 매일 갱신한다. Effective sample length는 약 16.7일이다.

**Q: 왜 stablecoin 제외?**
A: USDT 등은 KRW 변동성이 BTC 대비 1/20 수준 → GMV가 stablecoin에 50%+ 몰아주어 "변동성 기반 펀드" 콘셉트 훼손.

**Q: 단일종목 cap 없는 이유?**
A: 논문(Shin et al. 2025) 일관. Phase 6 결과 (max weight 분포)를 보고 재검토.

**Q: 평가에서 10분 frequency 사용 이유?**
A: 1분 portfolio return은 microstructure noise 큼. 논문(p.32) 그대로 따름.

**Q: 연환산 √252가 아니라 √365인 이유?**
A: 24/7 거래라 영업일 개념 없음.

**Q: 거래비용은?**
A: 무시 (펀드 컨셉 시연 단계).

**Q: GMV가 50개 모두에 투자하는가?**
A: 아니다. Long-only 제약 + 암호화폐의 높은 cross-correlation 때문에 자연스럽게 sparse해진다 (보통 active asset 5~15개). Jagannathan & Ma (2003) 참조.

**Q: 포트폴리오 최적화에 들어가는 Σ는?**
A: EWMA forecast (jump-adjusted PRVM 기반) + 직전일 jump volatility (JV_{d-1}). 논문 그대로.

**Q: cycle별 forecast는 매번 새로 계산하는가?**
A: EWMA recursion은 매일 update, forecast Σ̂는 매일 산출. 단, GMV weight ω는 cycle 시작일에만 산출하고 hold 기간 내 고정 (리밸런싱 없음).

---

## Coding Convention

- 파라미터 하드코딩 금지. 모두 `config.py`에서 import.
- 변동성 행렬은 항상 PSD projection (`utils.project_psd`) 적용 후 사용.
- Random seed 고정 (`np.random.seed(42)`).
- 새 모듈 작성 시 small data (5종목, 10일)로 sanity test 먼저 실행.
- 각 Phase 모듈은 CLI entry point 권장 (`python -m src.phaseN_xxx.module_name`).
- 산출물은 `data/processed/phaseN/`에 저장 (디렉토리 명명 규칙).
- 시각화는 `figures/phaseN/`에 저장.
- 함수 docstring은 한국어 OK, 변수명은 영어.

---

## Sanity Check 체크리스트 (Phase별)

| Phase | 체크 항목 |
|-------|----------|
| 1 | 종목 수 = 50, 결측 분봉 비율 분포, KST 09:00 cut 정확성 |
| 2 | log-return kurtosis > 3 (heavy-tail), volatility clustering 존재 |
| 3 | PRVM symmetric, eigenvalue ≥ 0, trace 시계열에 비현실적 spike 없음 |
| 4 | EWMA forecast PSD, MSPE/QLIKE 시계열 안정 |
| 5 | ω.sum() = 1 (within tol), ω ≥ 0, active asset 수 5~20 |
| 6 | Realized risk가 forecast risk와 동일 order, equity curve 연속성 |

---

## 참고 문헌

| 분야 | 문헌 |
|------|------|
| 메인 (FIVAR / PRVM / portfolio) | Shin, Kim, Wang & Fan (2025) — `docs/references/FIVAR_Revision.pdf` |
| Pre-averaging | Jacod et al. (2009), Christensen et al. (2010), Aït-Sahalia & Xiu (2016) |
| Jump truncation | Mancini (2009) |
| Long-only GMV sparsity | Jagannathan & Ma (2003) |
| 평가 메트릭 | Patton (2011), Hansen & Lunde (2006a) |
| Forecast 비교 검정 | Diebold & Mariano (1995) |
| EWMA | RiskMetrics Technical Document (1996) |
