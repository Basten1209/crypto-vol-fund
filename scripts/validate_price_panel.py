"""
Phase 1 산출물 price_panel.csv 검증 스크립트.

Phase 3 (PRVM) 진행 전, 데이터가 다음을 만족하는지 streaming 방식으로 검증한다:
  - schema (50 ticker + trading_day, stablecoin 제외)
  - 시간 인덱스 (2025-02-01 ~ 2026-03-31 KST, 1분 균일, 중복/gap 없음)
  - 결측/이상치 (NaN / Inf / 가격 <= 0)
  - trading_day 정합성 (KST 09:00 cut)
  - Phase 3 입력 적합성 (float64, log-return finite)

실행: python3 scripts/validate_price_panel.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402

CSV_PATH = ROOT / "price_panel.csv"
SELECTED_TICKERS_JSON = ROOT / "src" / "phase1_data" / "selected_tickers.json"
REPORT_PATH = ROOT / "data" / "validation" / "phase1_validation_report.json"
CHUNK_SIZE = 100_000

EXPECTED_START = pd.Timestamp("2025-02-01 00:00:00")
EXPECTED_END = pd.Timestamp("2026-03-31 23:59:00")
EXPECTED_ROWS = 610_560  # 14개월 × 30~31일 × 1440min — phase1_report.md 기준
ONE_MIN = pd.Timedelta(minutes=1)
KST_OFFSET = pd.Timedelta(hours=9)


class Check:
    def __init__(self) -> None:
        self.results: list[dict[str, Any]] = []

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.results.append({"name": name, "ok": ok, "detail": detail})
        tag = "PASS" if ok else "FAIL"
        line = f"[{tag}] {name}"
        if detail:
            line += f" — {detail}"
        print(line)

    def all_ok(self) -> bool:
        return all(r["ok"] for r in self.results)


def stage_a_schema(check: Check) -> tuple[list[str], list[str]]:
    """헤더만 읽어 schema 검증. ticker 리스트 반환."""
    header_df = pd.read_csv(CSV_PATH, nrows=0)
    cols = list(header_df.columns)

    check.add("Header column count", len(cols) == 52,
              detail=f"got {len(cols)} (expected 52: timestamp + 50 tickers + trading_day)")

    check.add("'timestamp' column present", cols[0] == "timestamp",
              detail=f"first column = '{cols[0]}'")
    check.add("'trading_day' column present", "trading_day" in cols,
              detail="trading_day in columns" if "trading_day" in cols else "MISSING")

    tickers = [c for c in cols if c not in {"timestamp", "trading_day"}]
    check.add("Ticker count", len(tickers) == config.N_ASSETS,
              detail=f"got {len(tickers)} tickers (expected {config.N_ASSETS})")

    # Stablecoin filter
    stable_hits = [t for t in tickers
                   if any(kw in t.upper() for kw in config.STABLE_KEYWORDS)]
    check.add("No stablecoin tickers", len(stable_hits) == 0,
              detail=f"matches: {stable_hits}" if stable_hits else "clean")

    # selected_tickers.json 정합성
    json_tickers: list[str] = []
    if SELECTED_TICKERS_JSON.exists():
        meta = json.loads(SELECTED_TICKERS_JSON.read_text())
        json_tickers = [a["ticker"] for a in meta.get("assets", [])]
        set_csv = set(tickers)
        set_json = set(json_tickers)
        missing_in_csv = set_json - set_csv
        extra_in_csv = set_csv - set_json
        ok = (not missing_in_csv) and (not extra_in_csv)
        detail = "match"
        if not ok:
            detail = f"missing_in_csv={sorted(missing_in_csv)}, extra_in_csv={sorted(extra_in_csv)}"
        check.add("CSV tickers == selected_tickers.json", ok, detail=detail)
    else:
        check.add("selected_tickers.json present", False,
                  detail=f"not found at {SELECTED_TICKERS_JSON}")

    return tickers, json_tickers


def stage_b_streaming(check: Check, tickers: list[str]) -> None:
    """전수 chunked scan."""
    total_rows = 0
    nan_count = 0
    inf_count = 0
    nonpositive_count = 0

    first_ts: pd.Timestamp | None = None
    last_ts: pd.Timestamp | None = None
    prev_last_ts: pd.Timestamp | None = None

    gap_violations = 0
    duplicate_count = 0
    dtype_issues: list[str] = []
    dtype_checked = False

    # trading_day별 카운터 & KST 09:00 cut 검증
    td_counts: dict[str, int] = {}
    # KST 09:00 마다 새 trading_day가 시작되는지: 최소-시각 별 trading_day 매핑 점검
    bad_cut_examples: list[tuple[str, str]] = []
    expected_td_by_ts: dict[pd.Timestamp, str] = {}  # not used; computed on the fly

    # 일별 std=0 감지를 위해 일별 first/last price 추적 (rough)
    per_day_minmax: dict[str, dict[str, float]] = {}

    # 종목별 전구간 min/max/sum
    ticker_min = pd.Series(np.inf, index=tickers, dtype=float)
    ticker_max = pd.Series(-np.inf, index=tickers, dtype=float)
    ticker_sum = pd.Series(0.0, index=tickers, dtype=float)
    ticker_count = pd.Series(0, index=tickers, dtype=int)

    reader = pd.read_csv(
        CSV_PATH,
        parse_dates=["timestamp"],
        chunksize=CHUNK_SIZE,
        dtype={t: "float64" for t in tickers},
    )

    for chunk_idx, chunk in enumerate(reader):
        n = len(chunk)
        total_rows += n

        # dtype check (first chunk only)
        if not dtype_checked:
            for t in tickers:
                if chunk[t].dtype != np.float64:
                    dtype_issues.append(f"{t}={chunk[t].dtype}")
            # trading_day는 string/date 둘 다 OK
            dtype_checked = True

        ts = chunk["timestamp"]
        if first_ts is None:
            first_ts = ts.iloc[0]
        last_ts = ts.iloc[-1]

        # 인접 timestamp diff
        if prev_last_ts is not None:
            if (ts.iloc[0] - prev_last_ts) != ONE_MIN:
                gap_violations += 1
        diffs = ts.diff().dropna()
        gap_violations += int((diffs != ONE_MIN).sum())
        duplicate_count += int((diffs == pd.Timedelta(0)).sum())
        prev_last_ts = ts.iloc[-1]

        # NaN / Inf / <=0 in ticker columns
        ticker_block = chunk[tickers]
        nan_count += int(ticker_block.isna().to_numpy().sum())
        inf_count += int(np.isinf(ticker_block.to_numpy()).sum())
        nonpositive_count += int((ticker_block.to_numpy() <= 0).sum())

        # ticker stats (NaN/inf 영향 줄이기 위해 finite mask)
        arr = ticker_block.to_numpy()
        finite = np.isfinite(arr)
        # min/max
        masked = np.where(finite, arr, np.nan)
        chunk_min = np.nanmin(masked, axis=0)
        chunk_max = np.nanmax(masked, axis=0)
        ticker_min = np.minimum(ticker_min.values, chunk_min)
        ticker_max = np.maximum(ticker_max.values, chunk_max)
        ticker_min = pd.Series(ticker_min, index=tickers)
        ticker_max = pd.Series(ticker_max, index=tickers)
        ticker_sum = ticker_sum.add(pd.Series(np.nansum(masked, axis=0), index=tickers))
        ticker_count = ticker_count.add(pd.Series(finite.sum(axis=0), index=tickers))

        # trading_day counts + KST 09:00 cut 검증
        td_col = chunk["trading_day"].astype(str)
        vc = td_col.value_counts()
        for k, v in vc.items():
            td_counts[k] = td_counts.get(k, 0) + int(v)

        # 검증: (timestamp - 9h).normalize() == trading_day
        expected_td = (ts - KST_OFFSET).dt.normalize().dt.strftime("%Y-%m-%d")
        actual_td = td_col.str.slice(0, 10)
        mism = expected_td != actual_td
        n_mism = int(mism.sum())
        if n_mism > 0 and len(bad_cut_examples) < 5:
            idxs = mism[mism].index[: 5 - len(bad_cut_examples)]
            for i in idxs:
                bad_cut_examples.append((str(ts.loc[i]), str(td_col.loc[i])))

    # ----- summary checks -----
    check.add("Row count", total_rows == EXPECTED_ROWS,
              detail=f"got {total_rows:,} (expected {EXPECTED_ROWS:,})")
    check.add("Start timestamp", first_ts == EXPECTED_START,
              detail=f"first = {first_ts}")
    check.add("End timestamp", last_ts == EXPECTED_END,
              detail=f"last  = {last_ts}")
    check.add("1-min uniform spacing", gap_violations == 0,
              detail=f"non-1min diffs: {gap_violations}")
    check.add("No duplicate timestamps", duplicate_count == 0,
              detail=f"duplicates: {duplicate_count}")
    check.add("No NaN in tickers", nan_count == 0, detail=f"NaN count: {nan_count}")
    check.add("No Inf in tickers", inf_count == 0, detail=f"Inf count: {inf_count}")
    check.add("All prices > 0", nonpositive_count == 0,
              detail=f"<=0 count: {nonpositive_count}")
    check.add("All ticker columns float64", not dtype_issues,
              detail="; ".join(dtype_issues) if dtype_issues else "OK")

    # trading_day 검증
    n_days = len(td_counts)
    full_day_count = sum(1 for v in td_counts.values() if v == 1440)
    partial_days = [(k, v) for k, v in td_counts.items() if v != 1440]
    check.add("trading_day cut == (timestamp - 9h).normalize()",
              len(bad_cut_examples) == 0,
              detail=f"examples: {bad_cut_examples[:3]}" if bad_cut_examples else "OK")
    # 첫/마지막 날은 partial 가능 (start 2025-02-01 00:00 = 2025-01-31 trading_day의 끝 09:00 전, end 2026-03-31 23:59 등)
    check.add("Most days have 1440 rows",
              len(partial_days) <= 2,
              detail=f"n_days={n_days}, full={full_day_count}, partial={partial_days[:5]}")

    # ticker stat 요약 출력 (정보 제공)
    means = (ticker_sum / ticker_count.replace(0, np.nan)).round(2)
    summary = pd.DataFrame({
        "min": ticker_min.round(2),
        "max": ticker_max.round(2),
        "mean": means,
    })
    print("\n--- Ticker price summary (top 10 by mean) ---")
    print(summary.sort_values("mean", ascending=False).head(10).to_string())
    print("--- Ticker price summary (bottom 5 by mean) ---")
    print(summary.sort_values("mean").head(5).to_string())

    # log-return sanity: 첫 번째 chunk만 다시 읽어 첫 day log-return finite 확인
    sample = pd.read_csv(CSV_PATH, parse_dates=["timestamp"], nrows=1440 * 2)
    log_returns = np.log(sample[tickers].astype(float)).diff().iloc[1:]
    n_bad = int((~np.isfinite(log_returns.to_numpy())).sum())
    check.add("Log-returns finite (first 2 days sample)", n_bad == 0,
              detail=f"non-finite log-return count: {n_bad}")

    # save report
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps({
        "summary": {
            "row_count": total_rows,
            "start": str(first_ts),
            "end": str(last_ts),
            "n_trading_days": n_days,
            "full_1440_days": full_day_count,
            "partial_days": partial_days[:10],
            "nan_count": nan_count,
            "inf_count": inf_count,
            "nonpositive_count": nonpositive_count,
            "gap_violations": gap_violations,
            "duplicate_count": duplicate_count,
        },
        "checks": check.results,
    }, indent=2, default=str))
    print(f"\nReport saved to: {REPORT_PATH.relative_to(ROOT)}")


def main() -> int:
    print(f"=== Phase 1 price_panel.csv Validation ===")
    print(f"file: {CSV_PATH}")
    size_mb = CSV_PATH.stat().st_size / (1024 * 1024)
    print(f"size: {size_mb:.1f} MB\n")

    check = Check()

    print("--- Stage A: schema ---")
    tickers, _ = stage_a_schema(check)
    print()

    print("--- Stage B: streaming scan ---")
    stage_b_streaming(check, tickers)
    print()

    if check.all_ok():
        print(">>> RESULT: ALL CHECKS PASSED — Phase 3 (PRVM) 진행 가능")
        return 0
    n_fail = sum(1 for r in check.results if not r["ok"])
    print(f">>> RESULT: {n_fail} CHECK(S) FAILED — 상세 메시지 확인 후 데이터 재처리 검토")
    return 1


if __name__ == "__main__":
    sys.exit(main())
