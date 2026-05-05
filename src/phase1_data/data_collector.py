"""
Upbit 1분봉 월별 ZIP 다운로드 + CSV 추출 모듈
실행 순서: asset_filter → data_collector → data_preprocessor

동작:
1. candidate_tickers.json 로드
2. crix-data.upbit.com에서 2025-02 ~ 2026-03 월별 ZIP 다운로드
3. ZIP 압축 해제 → CSV 저장: raw/{TICKER}/{YYYY-MM}.csv
4. 이미 다운로드된 파일 스킵 (resume 가능)
5. 실패 항목 로그 기록

API:
  다운로드: https://crix-data.upbit.com/candle/{MARKET}/monthly/1m/{YEAR}/{MARKET}_candle-1m_{YYYYMM}.zip
  CSV 컬럼: date_time_utc, open, high, low, close, acc_trade_price, acc_trade_volume
"""

import sys
import io
import json
import time
import logging
import zipfile
import threading
import requests
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DOWNLOAD_START, DOWNLOAD_END

# ──────────────────────────────────────────────
# 상수
# ──────────────────────────────────────────────
DATA_BASE      = "https://crix-data.upbit.com"
CANDIDATE_FILE = Path(__file__).parent / "candidate_tickers.json"
RAW_DIR        = Path(__file__).parent / "raw"
LOG_FILE       = Path(__file__).parent / "download_log.txt"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Referer": "https://www.upbit.com/historical_data",
}

MAX_WORKERS = 5   # 동시 다운로드 종목 수
RETRY_COUNT = 3   # 재시도 횟수

# 다운로드 대상 월 목록 (YYYYMM 형식)
def _make_month_list(start: str, end: str) -> list[str]:
    """YYYY-MM 형식의 start/end를 받아 YYYYMM 목록 반환"""
    months = []
    y, m = int(start[:4]), int(start[5:7])
    ey, em = int(end[:4]), int(end[5:7])
    while (y, m) <= (ey, em):
        months.append(f"{y:04d}{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months

DOWNLOAD_MONTHS = _make_month_list(DOWNLOAD_START, DOWNLOAD_END)

# 로거는 스레드 안전 핸들러로 설정
_log_lock = threading.Lock()


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("data_collector")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def load_candidates() -> list[dict]:
    """candidate_tickers.json 로드"""
    if not CANDIDATE_FILE.exists():
        raise FileNotFoundError(
            f"{CANDIDATE_FILE} 없음. asset_filter.py를 먼저 실행하세요."
        )
    with open(CANDIDATE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["candidates"]


def build_zip_url(market: str, yyyymm: str) -> str:
    """다운로드 URL 생성"""
    year = yyyymm[:4]
    filename = f"{market}_candle-1m_{yyyymm}.zip"
    return f"{DATA_BASE}/candle/{market}/monthly/1m/{year}/{filename}"


def download_and_extract(
    session: requests.Session,
    market: str,
    ticker: str,
    yyyymm: str,
    logger: logging.Logger,
) -> bool:
    """
    단일 종목 단일 월 ZIP 다운로드 후 CSV 저장
    반환: 성공 여부
    """
    yyyy_mm = f"{yyyymm[:4]}-{yyyymm[4:6]}"
    dest_csv = RAW_DIR / ticker / f"{yyyy_mm}.csv"

    # 이미 존재하면 스킵
    if dest_csv.exists() and dest_csv.stat().st_size > 100:
        return True  # 스킵 (성공으로 처리)

    url = build_zip_url(market, yyyymm)

    for attempt in range(RETRY_COUNT):
        try:
            resp = session.get(url, timeout=60)
            if resp.status_code == 404:
                # 데이터 없음 (정상 케이스: 신규 상장 종목 등)
                logger.debug(f"404 (데이터 없음): {market} {yyyymm}")
                return False
            if resp.status_code != 200:
                logger.warning(f"HTTP {resp.status_code}: {market} {yyyymm}")
                if attempt < RETRY_COUNT - 1:
                    time.sleep(2 ** attempt)
                continue

            # ZIP 검증 및 추출
            try:
                with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                    csv_names = [n for n in z.namelist() if n.endswith(".csv")]
                    if not csv_names:
                        logger.warning(f"ZIP에 CSV 없음: {market} {yyyymm}")
                        return False

                    csv_content = z.read(csv_names[0])

                dest_csv.parent.mkdir(parents=True, exist_ok=True)
                dest_csv.write_bytes(csv_content)
                return True

            except zipfile.BadZipFile:
                logger.warning(f"손상된 ZIP: {market} {yyyymm} (시도 {attempt+1})")
                if attempt < RETRY_COUNT - 1:
                    time.sleep(2 ** attempt)
                continue

        except requests.RequestException as e:
            logger.warning(f"요청 오류 (시도 {attempt+1}): {market} {yyyymm} - {e}")
            if attempt < RETRY_COUNT - 1:
                time.sleep(2 ** attempt)

    logger.error(f"최종 실패: {market} {yyyymm}")
    return False


def download_ticker(
    item: dict,
    logger: logging.Logger,
    pbar,
    results_lock: threading.Lock,
    results: dict,
) -> None:
    """단일 종목 전체 월 다운로드 (스레드 함수)"""
    market = item["market"]
    ticker = item["ticker"]

    session = requests.Session()
    session.headers.update(HEADERS)

    ticker_results = {}
    for yyyymm in DOWNLOAD_MONTHS:
        yyyy_mm = f"{yyyymm[:4]}-{yyyymm[4:6]}"
        dest_csv = RAW_DIR / ticker / f"{yyyy_mm}.csv"

        if dest_csv.exists() and dest_csv.stat().st_size > 100:
            ticker_results[yyyymm] = "skip"
        else:
            ok = download_and_extract(session, market, ticker, yyyymm, logger)
            ticker_results[yyyymm] = "ok" if ok else "fail"

        if pbar is not None:
            pbar.update(1)

    with results_lock:
        results[ticker] = ticker_results


def collect_all(candidates: list[dict], logger: logging.Logger) -> dict:
    """
    모든 종목 × 14개월 ZIP 다운로드 (병렬)
    반환: {ticker: {yyyymm: 'ok'|'skip'|'fail'}}
    """
    try:
        from tqdm import tqdm
        pbar = tqdm(
            total=len(candidates) * len(DOWNLOAD_MONTHS),
            desc="다운로드",
            unit="파일",
        )
    except ImportError:
        pbar = None

    results = {}
    results_lock = threading.Lock()
    total_files = len(candidates) * len(DOWNLOAD_MONTHS)

    logger.info(
        f"다운로드 시작 | {len(candidates)}개 종목 × {len(DOWNLOAD_MONTHS)}개월 "
        f"= {total_files}파일 | 병렬 {MAX_WORKERS}개"
    )

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(download_ticker, item, logger, pbar, results_lock, results)
            for item in candidates
        ]
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                logger.error(f"스레드 오류: {e}")

    if pbar is not None:
        pbar.close()

    # 통계
    downloaded = sum(1 for r in results.values() for s in r.values() if s == "ok")
    skipped    = sum(1 for r in results.values() for s in r.values() if s == "skip")
    failed     = sum(1 for r in results.values() for s in r.values() if s == "fail")
    logger.info(f"완료 | 신규: {downloaded} | 스킵: {skipped} | 실패: {failed}")

    return results


def save_summary(results: dict, logger: logging.Logger) -> None:
    """다운로드 결과 요약 저장"""
    summary_path = Path(__file__).parent / "download_summary.json"
    failed_list = [
        {"ticker": t, "month": yyyymm}
        for t, months in results.items()
        for yyyymm, status in months.items()
        if status == "fail"
    ]
    summary = {
        "generated_at":   datetime.now().isoformat(),
        "total_tickers":  len(results),
        "download_months": DOWNLOAD_MONTHS,
        "failed_count":   len(failed_list),
        "failed_items":   failed_list,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if failed_list:
        logger.warning(f"실패 항목 {len(failed_list)}개 → {summary_path}")
    else:
        logger.info(f"모든 파일 다운로드 성공 → {summary_path}")


def main() -> dict:
    print("=" * 60)
    print("Phase 1 Step 2: 1분봉 월별 ZIP 다운로드")
    print("=" * 60)

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logger()

    print(f"\n[1/3] candidate_tickers.json 로드...")
    candidates = load_candidates()
    print(f"  {len(candidates)}개 종목 로드 완료")

    print(f"\n[2/3] ZIP 다운로드 시작 ({DOWNLOAD_MONTHS[0]} ~ {DOWNLOAD_MONTHS[-1]})...")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    results = collect_all(candidates, logger)

    print(f"\n[3/3] 결과 요약 저장...")
    save_summary(results, logger)

    return results


if __name__ == "__main__":
    main()
