"""
전역 파라미터 설정 - 모든 모듈에서 이 파일을 import하여 사용
하드코딩 금지: 모든 파라미터는 여기서 중앙 관리
"""
import numpy as np

# === 데이터 ===
START_DATE        = "2025-02-01"       # 데이터 다운로드 시작일
END_DATE          = "2026-03-31"       # 데이터 다운로드 종료일
ANALYSIS_START    = "2025-03-01"       # 분석 시작일 (EWMA 초기화 후)
DAILY_CUT_KST     = "09:00"           # 일봉 구분 기준 (KST)
N_ASSETS          = 50                 # 선정 종목 수
EXCLUDE_STABLE    = True               # 스테이블코인 제외 여부

# 수집 대상 월 범위
DOWNLOAD_START    = "2025-02"
DOWNLOAD_END      = "2026-03"

# === PRVM ===
M                 = 1440               # 24h × 60min
K                 = 37                 # floor(sqrt(M))
PSI               = 1 / 12            # ∫₀¹ g(x)² dx
JUMP_C0           = 4                  # jump truncation 상수
JUMP_ALPHA_U      = 0.235             # jump truncation exponent

# === PSD projection ===
PSD_FLOOR         = 1e-10             # eigenvalue floor (QLIKE log det 용)

# === EWMA ===
LAMBDA_           = 0.94              # RiskMetrics 표준 (lambda는 Python 예약어)
WINDOW            = 28                # EWMA 초기값 산출 기간 (days)
EWMA_INIT_DAYS    = 28               # 초기값 계산용 일수

# === Portfolio ===
GMV_C0            = 1                 # gross exposure (long-only)
SINGLE_ASSET_CAP  = None             # 단일종목 cap (없음, Phase 6 후 재검토)

# === Backtest ===
CYCLES            = [7, 14]           # 1주, 2주 동시 운용
EVAL_FREQ_MIN     = 10               # 평가용 frequency (분)
EVAL_INTERVALS    = 144              # 1440 / 10
ANNUALIZATION     = 365              # 24/7 거래
RISK_FREE_RATE    = 0.0

# === 재현성 ===
RANDOM_SEED       = 42
np.random.seed(RANDOM_SEED)

# === 스테이블코인 제외 키워드 ===
STABLE_KEYWORDS   = ['USDT', 'USDC', 'DAI', 'BUSD', 'TUSD', 'USDP', 'FDUSD', 'KRW', 'USD']
