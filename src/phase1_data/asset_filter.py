"""
Upbit KRW 마켓 종목 필터링 모듈
실행 순서: asset_filter → data_collector → data_preprocessor

동작:
1. Upbit REST API로 전체 KRW 마켓 목록 수집
2. 스테이블코인 키워드 기반 제외
3. crix-data-api 파일 리스팅으로 2025-02 ~ 2026-03 (14개월)
   월별 ZIP 파일 존재 여부 확인 (티커당 2번 API 호출)
4. 통과 종목을 candidate_tickers.json으로 저장

발견한 API:
  리스팅: https://crix-data-api.upbit.com/api/v1/market-data/listing?prefix=candle/{MARKET}/monthly/1m/{YEAR}
  다운로드: https://crix-data.upbit.com/candle/{MARKET}/monthly/1m/{YEAR}/{MARKET}_candle-1m_{YYYYMM}.zip
"""

import sys
import json
import time
import requests
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import STABLE_KEYWORDS

# ──────────────────────────────────────────────
# 상수
# ──────────────────────────────────────────────
UPBIT_MARKET_API = "https://api.upbit.com/v1/market/all"
LISTING_API_BASE = "https://crix-data-api.upbit.com/api/v1/market-data/listing"

# 필수 보유 기간: 2025-02 ~ 2026-03 (14개월)
# 2025년 필요 월: 02~12 (11개)
# 2026년 필요 월: 01~03 (3개)
REQUIRED_2025 = [f"20250{m}" if m < 10 else f"2025{m}" for m in range(2, 13)]
REQUIRED_2026 = ["202601", "202602", "202603"]
REQUIRED_MONTHS = REQUIRED_2025 + REQUIRED_2026  # 14개

OUTPUT_PATH = Path(__file__).parent / "candidate_tickers.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Referer": "https://www.upbit.com/historical_data",
}


def get_krw_markets() -> list[dict]:
    """Upbit REST API에서 KRW 마켓 전체 목록 반환"""
    resp = requests.get(UPBIT_MARKET_API, timeout=30)
    resp.raise_for_status()
    all_markets = resp.json()
    return [m for m in all_markets if m["market"].startswith("KRW-")]


def is_stablecoin(ticker: str) -> bool:
    """스테이블코인 여부 확인 (코인 심볼에 키워드 포함 시 True)"""
    t = ticker.upper()
    return any(kw in t for kw in STABLE_KEYWORDS)


def list_monthly_files(session: requests.Session, market: str, year: int) -> set[str]:
    """
    crix-data-api 리스팅으로 특정 종목의 특정 연도 파일 목록 반환
    반환: 파일명 집합 (예: {"KRW-BTC_candle-1m_202502", ...})
    """
    url = f"{LISTING_API_BASE}?prefix=candle/{market}/monthly/1m/{year}"
    try:
        resp = session.get(url, timeout=15)
        if resp.status_code != 200:
            return set()
        items = resp.json()
        # key 예: "candle/KRW-BTC/monthly/1m/2025/KRW-BTC_candle-1m_202502.zip"
        names = set()
        for item in items:
            key = item.get("key", "")
            if key.endswith(".zip"):
                basename = key.split("/")[-1]  # KRW-BTC_candle-1m_202502.zip
                name_no_ext = basename.replace(".zip", "")  # KRW-BTC_candle-1m_202502
                # yyyymm 부분만 추출
                yyyymm = name_no_ext.split("_")[-1]  # 202502
                names.add(yyyymm)
        return names
    except Exception:
        return set()


def check_data_availability(session: requests.Session, market: str) -> tuple[bool, list[str]]:
    """
    2025-02 ~ 2026-03 모든 월 데이터 존재 여부 확인
    반환: (통과 여부, 누락된 월 목록)
    """
    files_2025 = list_monthly_files(session, market, 2025)
    files_2026 = list_monthly_files(session, market, 2026)

    missing = []
    for yyyymm in REQUIRED_MONTHS:
        year = int(yyyymm[:4])
        if year == 2025 and yyyymm not in files_2025:
            missing.append(yyyymm)
        elif year == 2026 and yyyymm not in files_2026:
            missing.append(yyyymm)

    return len(missing) == 0, missing


def filter_tickers(markets: list[dict]) -> list[dict]:
    """
    스테이블코인 제외 + 데이터 존재 확인으로 종목 필터링
    반환: 통과 종목 정보 리스트
    """
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **kw): return x

    # 1단계: 스테이블코인 제외
    candidates = []
    excluded_stable = []
    for m in markets:
        ticker = m["market"].replace("KRW-", "")
        if is_stablecoin(ticker):
            excluded_stable.append(ticker)
        else:
            candidates.append({
                "market":       m["market"],
                "ticker":       ticker,
                "korean_name":  m.get("korean_name", ""),
                "english_name": m.get("english_name", ""),
            })

    print(f"  스테이블코인 {len(excluded_stable)}개 제외: {excluded_stable}")
    print(f"  필터링 대상: {len(candidates)}개 종목")
    print(f"  확인 기간: {REQUIRED_MONTHS[0]} ~ {REQUIRED_MONTHS[-1]} ({len(REQUIRED_MONTHS)}개월)")

    # 2단계: 데이터 존재 확인
    session = requests.Session()
    session.headers.update(HEADERS)

    valid = []
    missing_info = {}
    incomplete_count = 0

    for item in tqdm(candidates, desc="데이터 존재 확인", unit="ticker"):
        market = item["market"]
        ok, missing = check_data_availability(session, market)
        if ok:
            valid.append(item)
        else:
            missing_info[market] = missing
            incomplete_count += 1
        time.sleep(0.05)  # API 부하 방지 (연도당 2회 호출)

    if missing_info:
        print(f"\n  데이터 불완전 {incomplete_count}개 제외")
        for mkt, months in list(missing_info.items())[:5]:
            print(f"    {mkt}: {months[:3]}{'...' if len(months) > 3 else ''} 누락")

    return valid


def main() -> list[dict]:
    print("=" * 60)
    print("Phase 1 Step 1: Upbit KRW 마켓 종목 필터링")
    print("=" * 60)

    # Upbit API 호출
    print("\n[1/3] Upbit API에서 KRW 마켓 목록 수집 중...")
    markets = get_krw_markets()
    print(f"  총 {len(markets)}개 KRW 마켓 발견")

    # 필터링
    print("\n[2/3] 스테이블코인 제외 + 데이터 존재 확인...")
    valid_tickers = filter_tickers(markets)
    print(f"\n  최종 통과 종목: {len(valid_tickers)}개")

    # 결과 저장
    print("\n[3/3] 결과 저장...")
    output = {
        "generated_at":    datetime.now().isoformat(),
        "total_krw_markets": len(markets),
        "required_period": f"{REQUIRED_MONTHS[0]} ~ {REQUIRED_MONTHS[-1]}",
        "required_months": REQUIRED_MONTHS,
        "candidate_count": len(valid_tickers),
        "candidates":      valid_tickers,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    tickers_only = [t["ticker"] for t in valid_tickers]
    print(f"  저장 완료: {OUTPUT_PATH}")
    print(f"  통과 종목: {tickers_only}")

    return valid_tickers


if __name__ == "__main__":
    main()
