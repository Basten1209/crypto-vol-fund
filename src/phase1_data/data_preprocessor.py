"""
Phase 1 전처리 모듈
실행 순서: asset_filter → data_collector → data_preprocessor

동작:
1. raw/ 폴더의 월별 CSV를 종목별로 연결
   (CSV 컬럼: date_time_utc, open, high, low, close, acc_trade_price, acc_trade_volume)
2. 24시간(1440분) 이상 연속 acc_trade_volume=0 구간 종목 제거
3. 평균 거래대금(acc_trade_price) 기준 상위 50개 선정
4. wide-format close 가격 패널 생성 (KST 1분봉)
5. trading_day 컬럼 추가 (KST 09:00 기준 일봉 구분)
6. 결과 저장
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import N_ASSETS, DAILY_CUT_KST

# ──────────────────────────────────────────────
# 경로
# ──────────────────────────────────────────────
PHASE1_DIR     = Path(__file__).parent
RAW_DIR        = PHASE1_DIR / "raw"
CANDIDATE_FILE = PHASE1_DIR / "candidate_tickers.json"
PRICE_PANEL    = PHASE1_DIR / "price_panel.csv"
SELECTED_JSON  = PHASE1_DIR / "selected_tickers.json"
PREP_LOG       = PHASE1_DIR / "preprocessing_log.txt"

# 전체 기간 (KST 기준, UTC로 저장된 데이터를 변환 후 이 범위로 reindex)
FULL_START_KST = "2025-02-01 00:00:00"
FULL_END_KST   = "2026-03-31 23:59:00"

ZERO_VOL_THRESHOLD = 1440  # 24시간 연속 volume=0 제거 임계값

# KST = UTC + 9h
UTC_TO_KST_OFFSET = pd.Timedelta(hours=9)
KST_CUT_HOUR      = int(DAILY_CUT_KST.split(":")[0])  # 9


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("data_preprocessor")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    PREP_LOG.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(PREP_LOG, encoding="utf-8", mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ──────────────────────────────────────────────
# CSV 파싱
# ──────────────────────────────────────────────

def load_ticker_csv(ticker: str, logger: logging.Logger) -> pd.DataFrame | None:
    """
    raw/{TICKER}/ 디렉토리의 월별 CSV를 로드하여 하나의 DataFrame으로 합치기
    CSV 형식: date_time_utc, open, high, low, close, acc_trade_price, acc_trade_volume
    timestamp는 UTC → KST로 변환 (+ 9h) 후 인덱스로 설정
    """
    ticker_dir = RAW_DIR / ticker
    if not ticker_dir.exists():
        logger.warning(f"{ticker}: raw 디렉토리 없음")
        return None

    csv_files = sorted(ticker_dir.glob("*.csv"))
    if not csv_files:
        logger.warning(f"{ticker}: CSV 파일 없음")
        return None

    frames = []
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path, encoding="utf-8", low_memory=False)
            if df.empty:
                continue

            # date_time_utc 컬럼 파싱 (또는 첫 번째 컬럼을 타임스탬프로 사용)
            ts_col = None
            for col in df.columns:
                if "date" in col.lower() or "time" in col.lower() or "timestamp" in col.lower():
                    ts_col = col
                    break
            if ts_col is None:
                ts_col = df.columns[0]

            df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
            df = df.dropna(subset=[ts_col])

            # UTC → KST 변환
            df[ts_col] = df[ts_col] + UTC_TO_KST_OFFSET
            df = df.set_index(ts_col)
            df.index.name = "timestamp"

            # 컬럼 정규화
            rename_map = {}
            col_lower = {c.lower(): c for c in df.columns}
            for std, aliases in [
                ("close",  ["close", "종가"]),
                ("volume", ["acc_trade_volume", "volume", "거래량"]),
                ("value",  ["acc_trade_price", "value", "거래대금"]),
            ]:
                for a in aliases:
                    if a in col_lower:
                        rename_map[col_lower[a]] = std
                        break

            df = df.rename(columns=rename_map)
            needed = [c for c in ["close", "volume", "value"] if c in df.columns]
            if "close" not in needed:
                logger.warning(f"{ticker}: {csv_path.name}에 close 컬럼 없음")
                continue

            frames.append(df[needed])

        except Exception as e:
            logger.warning(f"{ticker}: {csv_path.name} 읽기 실패 — {e}")
            continue

    if not frames:
        return None

    combined = pd.concat(frames)
    combined = combined[~combined.index.duplicated(keep="first")]
    combined = combined.sort_index()
    return combined


# ──────────────────────────────────────────────
# 필터: 연속 volume=0 체크
# ──────────────────────────────────────────────

def has_long_zero_volume(df: pd.DataFrame, ticker: str, logger: logging.Logger) -> bool:
    """
    전체 기간에서 1440분 이상 연속 volume=0 구간이 있으면 True
    volume 컬럼 없으면 False (제거 안 함)
    """
    if "volume" not in df.columns:
        return False

    full_idx = pd.date_range(FULL_START_KST, FULL_END_KST, freq="1min")
    vol_full = df["volume"].reindex(full_idx, fill_value=0).fillna(0)

    is_zero = (vol_full == 0).astype(int)
    group = (is_zero != is_zero.shift()).cumsum()
    max_run = is_zero.groupby(group).sum().max()

    if max_run >= ZERO_VOL_THRESHOLD:
        logger.info(f"{ticker}: 연속 volume=0 최대 {max_run}분 → 제외")
        return True
    return False


# ──────────────────────────────────────────────
# 주요 전처리 함수
# ──────────────────────────────────────────────

def compute_avg_value(df: pd.DataFrame) -> float:
    """평균 거래대금 계산"""
    if "value" in df.columns:
        return float(df["value"].fillna(0).mean())
    elif "close" in df.columns and "volume" in df.columns:
        return float((df["close"] * df["volume"]).fillna(0).mean())
    return 0.0


def preprocess(logger: logging.Logger) -> tuple[pd.DataFrame, list[str], dict]:
    """
    전체 전처리 파이프라인 실행
    반환: (panel, top_tickers, avg_values)
    """
    if not CANDIDATE_FILE.exists():
        raise FileNotFoundError(f"{CANDIDATE_FILE} 없음. asset_filter.py를 먼저 실행하세요.")

    with open(CANDIDATE_FILE, "r", encoding="utf-8") as f:
        candidates_data = json.load(f)
    candidates = [c["ticker"] for c in candidates_data["candidates"]]
    logger.info(f"후보 종목 {len(candidates)}개 로드")

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **kw): return x

    # ── Step 1: CSV 로드 + 연속 volume=0 필터 ──────────────────────────
    logger.info("Step 1: CSV 로드 + 연속 volume=0 필터")
    ticker_data: dict[str, pd.DataFrame] = {}
    excluded = []

    for ticker in tqdm(candidates, desc="CSV 로드", unit="ticker"):
        df = load_ticker_csv(ticker, logger)
        if df is None:
            logger.warning(f"{ticker}: 데이터 없음 → 제외")
            excluded.append(ticker)
            continue

        if has_long_zero_volume(df, ticker, logger):
            excluded.append(ticker)
            continue

        ticker_data[ticker] = df

    logger.info(f"  {len(excluded)}개 제외, 남은 종목: {len(ticker_data)}개")

    if len(ticker_data) < N_ASSETS:
        logger.warning(f"남은 종목({len(ticker_data)}) < 목표({N_ASSETS}). 가능한 모든 종목 사용.")

    # ── Step 2: 평균 거래대금 → 상위 N_ASSETS 선정 ──────────────────────
    logger.info("Step 2: 평균 거래대금 기준 상위 N_ASSETS 선정")
    avg_values = {t: compute_avg_value(df) for t, df in ticker_data.items()}
    sorted_tickers = sorted(avg_values, key=avg_values.get, reverse=True)
    top_tickers = sorted_tickers[:N_ASSETS]

    logger.info(f"  상위 {len(top_tickers)}개 선정:")
    for rank, t in enumerate(top_tickers[:10], 1):
        logger.info(f"    {rank:2d}. {t:10s} 평균 거래대금: {avg_values[t]:,.0f} KRW")

    # ── Step 3: wide-format close 패널 구성 ──────────────────────────────
    logger.info("Step 3: wide-format close 패널 구성")
    full_idx = pd.date_range(FULL_START_KST, FULL_END_KST, freq="1min", name="timestamp")

    close_frames = {}
    for ticker in tqdm(top_tickers, desc="패널 구성", unit="ticker"):
        close_frames[ticker] = ticker_data[ticker]["close"].reindex(full_idx)

    panel = pd.DataFrame(close_frames, index=full_idx)

    # ── Step 4: 결측치 처리 ────────────────────────────────────────────
    logger.info("Step 4: 결측치 처리 (ffill → bfill)")
    missing_before = int(panel.isna().sum().sum())
    panel = panel.ffill().bfill()
    missing_after = int(panel.isna().sum().sum())
    logger.info(f"  결측치: {missing_before} → {missing_after}")

    # ── Step 5: trading_day 컬럼 추가 ────────────────────────────────────
    logger.info("Step 5: trading_day 컬럼 추가 (KST 09:00 기준)")
    # KST 09:00이 하루의 시작이므로 index - 9h → date
    panel["trading_day"] = (panel.index - pd.Timedelta(hours=KST_CUT_HOUR)).normalize()
    logger.info(f"  패널 크기: {panel.shape}, 기간: {panel.index[0]} ~ {panel.index[-1]}")

    return panel, top_tickers, avg_values


def save_results(
    panel: pd.DataFrame,
    top_tickers: list[str],
    avg_values: dict,
    logger: logging.Logger,
) -> None:
    """결과 파일 저장"""
    # price_panel.csv
    logger.info(f"price_panel.csv 저장 중 ({panel.shape[0]:,}행)...")
    panel.to_csv(PRICE_PANEL, encoding="utf-8-sig")
    logger.info(f"  저장 완료: {PRICE_PANEL}")

    # 후보 메타데이터 로드
    with open(CANDIDATE_FILE, "r", encoding="utf-8") as f:
        candidates_data = json.load(f)
    meta_map = {c["ticker"]: c for c in candidates_data["candidates"]}

    selected = []
    for rank, ticker in enumerate(top_tickers, 1):
        meta = meta_map.get(ticker, {})
        selected.append({
            "rank":             rank,
            "ticker":           ticker,
            "market":           meta.get("market", f"KRW-{ticker}"),
            "korean_name":      meta.get("korean_name", ""),
            "english_name":     meta.get("english_name", ""),
            "avg_trade_value":  round(avg_values.get(ticker, 0), 2),
            "selection_reason": "평균 거래대금 상위 50 (연속 volume=0 없음)",
        })

    output = {
        "generated_at":    datetime.now().isoformat(),
        "n_assets":        len(selected),
        "period":          f"{FULL_START_KST} ~ {FULL_END_KST} (KST)",
        "selection_basis": "avg_acc_trade_price (KRW)",
        "assets":          selected,
    }

    with open(SELECTED_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info(f"  저장 완료: {SELECTED_JSON}")


def main() -> pd.DataFrame:
    print("=" * 60)
    print("Phase 1 Step 3: 데이터 전처리")
    print("=" * 60)

    logger = setup_logger()

    print("\n[1/2] 전처리 실행...")
    panel, top_tickers, avg_values = preprocess(logger)

    print(f"\n[2/2] 결과 저장...")
    save_results(panel, top_tickers, avg_values, logger)

    print("\n전처리 완료!")
    print(f"  price_panel.csv: {panel.shape[0]:,}행 × {panel.shape[1]}컬럼")
    print(f"  선정 종목 ({len(top_tickers)}개): {top_tickers[:10]}{'...' if len(top_tickers)>10 else ''}")
    print(f"  로그: {PREP_LOG}")

    return panel


if __name__ == "__main__":
    main()
