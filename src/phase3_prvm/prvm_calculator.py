"""Chunked Phase 3 PRVM calculator.

This module reads the 1-minute price panel without loading the CSV text into the
conversation or keeping the full CSV in memory. It builds daily log-return
blocks, computes jump-adjusted PRVM, raw PRVM, and jump volatility matrices, and
writes downstream artifacts for Phase 4+.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import as_strided

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.utils import ensure_dir, project_psd  # noqa: E402


@dataclass(frozen=True)
class PRVMParams:
    m: int = config.M
    k: int = config.K
    psi: float = config.PSI
    jump_c0: float = config.JUMP_C0
    jump_alpha_u: float = config.JUMP_ALPHA_U
    psd_floor: float = config.PSD_FLOOR

    @property
    def num_k(self) -> int:
        return self.m - self.k + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "m": self.m,
            "k": self.k,
            "num_k": self.num_k,
            "psi": self.psi,
            "jump_c0": self.jump_c0,
            "jump_alpha_u": self.jump_alpha_u,
            "psd_floor": self.psd_floor,
        }


def _g(x: np.ndarray) -> np.ndarray:
    return np.minimum(x, 1.0 - x)


def _safe_condition_number(eigvals: np.ndarray) -> float:
    max_eig = float(np.max(eigvals))
    min_eig = float(np.min(eigvals))
    if min_eig <= 0:
        return float("inf")
    return max_eig / min_eig


def _matrix_summary(date: str, prvm: np.ndarray, raw_prvm: np.ndarray, jv: np.ndarray) -> dict[str, Any]:
    eigvals = np.linalg.eigvalsh((prvm + prvm.T) / 2.0)
    trace_prvm = float(np.trace(prvm))
    trace_raw = float(np.trace(raw_prvm))
    trace_jv = float(np.trace(jv))
    symmetry_error = float(np.max(np.abs(prvm - prvm.T)))
    jump_trace_ratio = trace_jv / trace_raw if trace_raw > 0 else float("nan")
    return {
        "date": date,
        "trace_prvm": trace_prvm,
        "trace_raw_prvm": trace_raw,
        "trace_jv": trace_jv,
        "jump_trace_ratio": jump_trace_ratio,
        "min_eig_prvm": float(np.min(eigvals)),
        "max_eig_prvm": float(np.max(eigvals)),
        "condition_number_prvm": _safe_condition_number(eigvals),
        "diag_min_prvm": float(np.min(np.diag(prvm))),
        "diag_max_prvm": float(np.max(np.diag(prvm))),
        "symmetry_error_prvm": symmetry_error,
    }


def calculate_prvm_for_day(
    date: str,
    daily_log_returns: np.ndarray,
    params: PRVMParams | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Calculate raw PRVM, jump-adjusted PRVM, and JV for one full day."""
    if params is None:
        params = PRVMParams()
    elif isinstance(params, dict):
        params = PRVMParams(
            m=int(params["m"]),
            k=int(params["k"]),
            psi=float(params["psi"]),
            jump_c0=float(params["jump_c0"]),
            jump_alpha_u=float(params["jump_alpha_u"]),
            psd_floor=float(params["psd_floor"]),
        )

    y = np.ascontiguousarray(daily_log_returns, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError(f"{date}: expected 2D return block, got shape {y.shape}")
    if y.shape[0] != params.m:
        raise ValueError(f"{date}: expected {params.m} returns, got {y.shape[0]}")
    if params.k < 2 or params.num_k <= 0:
        raise ValueError(f"Invalid PRVM params: {params.to_dict()}")
    if not np.isfinite(y).all():
        raise ValueError(f"{date}: non-finite log returns")

    num_assets = y.shape[1]
    row_stride, col_stride = y.strides

    bar_shape = (params.num_k, params.k - 1, num_assets)
    bar_strides = (row_stride, row_stride, col_stride)
    y_window_bar = as_strided(y[:-1], shape=bar_shape, strides=bar_strides, writeable=False)

    weights_bar = _g(np.arange(1, params.k, dtype=np.float64) / params.k)
    y_bar = np.einsum("j,kji->ki", weights_bar, y_window_bar, optimize=True)

    scaled_y_bar_sq = np.square((params.m ** 0.25) * y_bar)
    std_devs = np.sqrt(np.sum(scaled_y_bar_sq, axis=0) / params.num_k)
    thresholds = (params.jump_c0 * std_devs) * (params.m ** (-params.jump_alpha_u))

    hat_shape = (params.num_k, params.k, num_assets)
    hat_strides = (row_stride, row_stride, col_stride)
    y_window_hat = as_strided(y, shape=hat_shape, strides=hat_strides, writeable=False)

    g_right = _g(np.arange(1, params.k + 1, dtype=np.float64) / params.k)
    g_left = _g(np.arange(0, params.k, dtype=np.float64) / params.k)
    weights_hat_sq = np.square(g_right - g_left)
    y_hat = np.einsum("l,kli,klj->kij", weights_hat_sq, y_window_hat, y_window_hat, optimize=True)

    y_outer = np.einsum("ki,kj->kij", y_bar, y_bar, optimize=True)
    raw_terms = y_outer - 0.5 * y_hat
    raw_prvm_est = (1.0 / (params.psi * params.k)) * np.sum(raw_terms, axis=0)

    non_jump = np.abs(y_bar) <= thresholds
    non_jump_outer = np.einsum(
        "ki,kj->kij",
        non_jump.astype(np.float64),
        non_jump.astype(np.float64),
        optimize=True,
    )
    prvm_est = (1.0 / (params.psi * params.k)) * np.sum(raw_terms * non_jump_outer, axis=0)
    jv_est = raw_prvm_est - prvm_est

    raw_prvm = project_psd(raw_prvm_est, floor=params.psd_floor)
    prvm = project_psd(prvm_est, floor=params.psd_floor)
    jv = project_psd(jv_est, floor=params.psd_floor)
    if not (np.isfinite(raw_prvm).all() and np.isfinite(prvm).all() and np.isfinite(jv).all()):
        raise FloatingPointError(f"{date}: non-finite matrix after PSD projection")

    return {
        "date": date,
        "prvm": prvm,
        "raw_prvm": raw_prvm,
        "jv": jv,
        "summary": _matrix_summary(date, prvm, raw_prvm, jv),
    }


def _calculate_prvm_for_day_worker(args: tuple[str, np.ndarray, dict[str, Any]]) -> dict[str, Any]:
    date, daily_log_returns, params = args
    return calculate_prvm_for_day(date, daily_log_returns, params)


def _resolve_workers(workers: str | int | None) -> int:
    if workers is None or workers == "auto":
        cpu_count = mp.cpu_count() or 1
        return max(1, min(4, cpu_count - 1))
    resolved = int(workers)
    if resolved < 1:
        raise ValueError("--workers must be >= 1")
    return resolved


def _read_header(input_path: Path) -> list[str]:
    header = pd.read_csv(input_path, nrows=0, encoding="utf-8-sig")
    return list(header.columns)


def _validate_header(input_path: Path) -> tuple[list[str], list[str]]:
    cols = _read_header(input_path)
    if "timestamp" not in cols:
        raise ValueError("Input CSV must include a timestamp column")
    if "trading_day" not in cols:
        raise ValueError("Input CSV must include a trading_day column")

    tickers = [c for c in cols if c not in {"timestamp", "trading_day"}]
    if len(tickers) != config.N_ASSETS:
        raise ValueError(f"Expected {config.N_ASSETS} tickers, got {len(tickers)}")
    return cols, tickers


def _append_return_segments(
    returns: np.ndarray,
    trading_days: np.ndarray,
    state: dict[str, Any],
    params: PRVMParams,
    submit_day,
) -> bool:
    """Append run-length encoded return slices. Returns True when caller should stop."""
    if len(returns) == 0:
        return False

    change_points = np.flatnonzero(trading_days[1:] != trading_days[:-1]) + 1
    starts = np.r_[0, change_points]
    ends = np.r_[change_points, len(trading_days)]

    for start, end in zip(starts, ends):
        day = str(trading_days[start])[:10]
        segment = returns[start:end]

        if state["current_day"] is None:
            state["current_day"] = day

        if day != state["current_day"]:
            should_stop = _finalize_current_day(state, params, submit_day)
            if should_stop:
                return True
            state["current_day"] = day
            state["parts"] = []
            state["count"] = 0

        state["parts"].append(segment.copy())
        state["count"] += int(end - start)

    return False


def _finalize_current_day(state: dict[str, Any], params: PRVMParams, submit_day) -> bool:
    day = state["current_day"]
    if day is None:
        return False

    count = int(state["count"])
    if count == params.m:
        daily_returns = np.vstack(state["parts"])
        return bool(submit_day(day, daily_returns))

    state["skipped_days"].append({"date": day, "return_count": count, "reason": "not_full_day"})
    return False


def _stream_and_submit_days(
    input_path: Path,
    tickers: list[str],
    params: PRVMParams,
    chunk_size: int,
    limit_days: int | None,
    submit_day,
) -> tuple[int, list[dict[str, Any]]]:
    state: dict[str, Any] = {
        "current_day": None,
        "parts": [],
        "count": 0,
        "skipped_days": [],
    }
    prev_log_prices: np.ndarray | None = None
    submitted_full_days = 0

    def counted_submit(day: str, daily_returns: np.ndarray) -> bool:
        nonlocal submitted_full_days
        if limit_days is not None and submitted_full_days >= limit_days:
            return True
        submitted_full_days += 1
        should_stop = bool(submit_day(day, daily_returns))
        if limit_days is not None and submitted_full_days >= limit_days:
            should_stop = True
        return should_stop

    dtype_map = {ticker: "float64" for ticker in tickers}
    usecols = ["timestamp", *tickers, "trading_day"]
    reader = pd.read_csv(
        input_path,
        chunksize=chunk_size,
        usecols=usecols,
        dtype=dtype_map,
        encoding="utf-8-sig",
    )

    stop = False
    for chunk_idx, chunk in enumerate(reader, start=1):
        prices = chunk[tickers].to_numpy(dtype=np.float64, copy=False)
        if not np.isfinite(prices).all():
            raise ValueError(f"Chunk {chunk_idx}: price panel contains NaN or Inf")
        if (prices <= 0).any():
            raise ValueError(f"Chunk {chunk_idx}: price panel contains non-positive prices")

        log_prices = np.log(prices)
        trading_days = chunk["trading_day"].astype(str).str.slice(0, 10).to_numpy()

        if prev_log_prices is None:
            if len(log_prices) < 2:
                prev_log_prices = log_prices[-1].copy()
                continue
            returns = np.diff(log_prices, axis=0)
            return_days = trading_days[1:]
        else:
            returns = np.empty_like(log_prices)
            returns[0] = log_prices[0] - prev_log_prices
            if len(log_prices) > 1:
                returns[1:] = log_prices[1:] - log_prices[:-1]
            return_days = trading_days

        prev_log_prices = log_prices[-1].copy()
        stop = _append_return_segments(returns, return_days, state, params, counted_submit)
        if stop:
            break

    if not stop:
        _finalize_current_day(state, params, counted_submit)

    return submitted_full_days, state["skipped_days"]


def _write_long_csv(path: Path, dates: np.ndarray, tickers: list[str], matrices: np.ndarray) -> None:
    row_tickers = np.repeat(np.asarray(tickers, dtype=object), len(tickers))
    col_tickers = np.tile(np.asarray(tickers, dtype=object), len(tickers))
    header = True
    for date, matrix in zip(dates, matrices):
        frame = pd.DataFrame(
            {
                "date": str(date),
                "ticker_i": row_tickers,
                "ticker_j": col_tickers,
                "value": matrix.reshape(-1),
            }
        )
        frame.to_csv(path, mode="w" if header else "a", index=False, header=header)
        header = False


def _write_outputs(
    output_dir: Path,
    input_path: Path,
    tickers: list[str],
    params: PRVMParams,
    results: list[dict[str, Any]],
    skipped_days: list[dict[str, Any]],
    write_long_csv: bool,
    chunk_size: int,
    workers: int,
    limit_days: int | None,
) -> dict[str, Any]:
    if not results:
        raise ValueError("No full-day PRVM results were produced")

    results = sorted(results, key=lambda item: item["date"])
    dates = np.asarray([item["date"] for item in results], dtype="U10")
    tickers_arr = np.asarray(tickers, dtype="U")
    prvm = np.stack([item["prvm"] for item in results]).astype(np.float64)
    raw_prvm = np.stack([item["raw_prvm"] for item in results]).astype(np.float64)
    jv = np.stack([item["jv"] for item in results]).astype(np.float64)
    summary_df = pd.DataFrame([item["summary"] for item in results])

    output_dir = ensure_dir(output_dir)
    npz_path = output_dir / "prvm_results.npz"
    summary_path = output_dir / "phase3_daily_summary.csv"
    report_path = output_dir / "phase3_prvm_report.json"
    prvm_long_path = output_dir / "prvm_long.csv"
    jv_long_path = output_dir / "jv_long.csv"

    params_json = json.dumps(params.to_dict(), sort_keys=True)
    np.savez_compressed(
        npz_path,
        dates=dates,
        tickers=tickers_arr,
        prvm=prvm,
        raw_prvm=raw_prvm,
        jv=jv,
        params_json=np.asarray(params_json),
    )
    summary_df.to_csv(summary_path, index=False)

    if write_long_csv:
        _write_long_csv(prvm_long_path, dates, tickers, prvm)
        _write_long_csv(jv_long_path, dates, tickers, jv)

    sanity = {
        "all_finite": bool(np.isfinite(prvm).all() and np.isfinite(raw_prvm).all() and np.isfinite(jv).all()),
        "min_eig_prvm_min": float(summary_df["min_eig_prvm"].min()),
        "symmetry_error_prvm_max": float(summary_df["symmetry_error_prvm"].max()),
        "trace_prvm_min": float(summary_df["trace_prvm"].min()),
        "trace_prvm_max": float(summary_df["trace_prvm"].max()),
        "jump_trace_ratio_mean": float(summary_df["jump_trace_ratio"].mean()),
    }
    report = {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "params": params.to_dict(),
        "chunk_size": chunk_size,
        "workers": workers,
        "limit_days": limit_days,
        "n_days": int(len(dates)),
        "first_date": str(dates[0]),
        "last_date": str(dates[-1]),
        "n_assets": int(len(tickers)),
        "tickers": tickers,
        "arrays": {
            "prvm_shape": list(prvm.shape),
            "raw_prvm_shape": list(raw_prvm.shape),
            "jv_shape": list(jv.shape),
        },
        "skipped_days": skipped_days,
        "sanity": sanity,
        "outputs": {
            "npz": str(npz_path),
            "daily_summary_csv": str(summary_path),
            "prvm_long_csv": str(prvm_long_path) if write_long_csv else None,
            "jv_long_csv": str(jv_long_path) if write_long_csv else None,
            "report_json": str(report_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def compute_phase3_prvm(
    input_path: Path | str,
    output_dir: Path | str,
    workers: str | int | None = "auto",
    chunk_size: int = 100_000,
    limit_days: int | None = None,
    write_long_csv: bool = True,
) -> dict[str, Any]:
    """Run the full Phase 3 PRVM pipeline and write artifacts to disk."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if chunk_size < 2:
        raise ValueError("chunk_size must be >= 2")
    if limit_days is not None and limit_days < 1:
        raise ValueError("limit_days must be >= 1")

    params = PRVMParams()
    _, tickers = _validate_header(input_path)
    resolved_workers = _resolve_workers(workers)

    print("=== Phase 3 PRVM calculation ===")
    print(f"input: {input_path}")
    print(f"output_dir: {output_dir}")
    print(f"assets: {len(tickers)}")
    print(f"params: {params.to_dict()}")
    print(f"workers: {resolved_workers}")
    if limit_days is not None:
        print(f"limit_days: {limit_days}")

    params_dict = params.to_dict()
    results: list[dict[str, Any]] = []
    handles = []

    if resolved_workers == 1:
        pool = None

        def submit_day(day: str, daily_returns: np.ndarray) -> bool:
            result = calculate_prvm_for_day(day, daily_returns, params)
            results.append(result)
            if len(results) == 1 or len(results) % 10 == 0:
                print(f"computed {len(results)} day(s), latest={day}")
            return False

    else:
        pool = mp.Pool(processes=resolved_workers)

        def submit_day(day: str, daily_returns: np.ndarray) -> bool:
            handles.append(pool.apply_async(_calculate_prvm_for_day_worker, ((day, daily_returns, params_dict),)))
            submitted = len(handles)
            if submitted == 1 or submitted % 25 == 0:
                print(f"submitted {submitted} day(s), latest={day}")
            return False

    try:
        submitted_days, skipped_days = _stream_and_submit_days(
            input_path=input_path,
            tickers=tickers,
            params=params,
            chunk_size=chunk_size,
            limit_days=limit_days,
            submit_day=submit_day,
        )

        if pool is not None:
            pool.close()
            for idx, handle in enumerate(handles, start=1):
                results.append(handle.get())
                if idx == 1 or idx % 25 == 0 or idx == len(handles):
                    latest = results[-1]["date"]
                    print(f"collected {idx}/{len(handles)} result(s), latest={latest}")
            pool.join()

        print(f"full days submitted: {submitted_days}")
        print(f"skipped partial days: {len(skipped_days)}")
        report = _write_outputs(
            output_dir=output_dir,
            input_path=input_path,
            tickers=tickers,
            params=params,
            results=results,
            skipped_days=skipped_days,
            write_long_csv=write_long_csv,
            chunk_size=chunk_size,
            workers=resolved_workers,
            limit_days=limit_days,
        )
        print(f"saved npz: {report['outputs']['npz']}")
        print(f"saved report: {report['outputs']['report_json']}")
        return report
    except Exception:
        if pool is not None:
            pool.terminate()
            pool.join()
        raise


def load_phase3_results(npz_path: Path | str) -> dict[str, Any]:
    """Load the compressed Phase 3 artifact produced by this module."""
    data = np.load(npz_path, allow_pickle=False)
    params_json = str(data["params_json"].item())
    return {
        "dates": data["dates"],
        "tickers": data["tickers"],
        "prvm": data["prvm"],
        "raw_prvm": data["raw_prvm"],
        "jv": data["jv"],
        "params": json.loads(params_json),
    }
