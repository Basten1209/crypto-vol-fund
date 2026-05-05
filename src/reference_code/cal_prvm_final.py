import pandas as pd
import numpy as np
from math import floor, sqrt
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import contextlib
from numpy.lib.stride_tricks import as_strided

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """
    joblib의 진행 상황을 tqdm progress bar에 업데이트하기 위한 컨텍스트 관리자입니다.
    """
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    새로운 형식의 CSV 파일을 로드하고 데이터를 전처리합니다.
    첫 번째 열을 타임스탬프 인덱스로 사용하고, 첫 번째 행을 헤더(티커)로 사용합니다.
    """
    print(f"'{filepath}' 파일 로딩 및 전처리를 시작합니다...")
    try:
        df = pd.read_csv(filepath, header=0, index_col=0, encoding='latin1', low_memory=False)
    except Exception as e:
        print(f"오류: '{filepath}' 파일을 읽는 중 문제가 발생했습니다. 파일 인코딩이나 형식을 확인해주세요. 오류 메시지: {e}")
        raise

    df.index = pd.to_datetime(df.index, errors='coerce')

    original_rows = len(df)
    df = df[df.index.notna()]
    new_rows = len(df)
    if original_rows > new_rows:
        print(f"{original_rows - new_rows}개의 행이 유효하지 않은 타임스탬프로 인해 제거되었습니다.")

    price_tickers = df.columns
    for ticker in tqdm(price_tickers, desc="데이터 클리닝 중"):
        df[ticker] = pd.to_numeric(df[ticker], errors='coerce')
        df[ticker].fillna(method='ffill', inplace=True)
        df[ticker].fillna(method='bfill', inplace=True)

    if df.isnull().values.any():
       warnings.warn("경고: 데이터 클리닝 후에도 여전히 NaN 값이 남아있습니다.")
    else:
       print("결측치 처리가 완료되었습니다.")

    print("데이터 로딩 및 전처리가 완료되었습니다.")
    return df


def remove_half_trading_days(df: pd.DataFrame, full_day_minutes: int = 390) -> pd.DataFrame:
    """
    정규 거래 시간(분)보다 데이터 포인트가 적은 날(단축 거래일)을 제거합니다.
    """
    print(f"\n거래 시간이 부족한 날(기준: {full_day_minutes}분)을 제거합니다...")
    daily_counts = df.groupby(df.index.date).size()
    full_trading_dates = daily_counts[daily_counts >= full_day_minutes].index

    original_days = len(daily_counts)
    removed_days = original_days - len(full_trading_dates)

    if removed_days > 0:
        print(f"총 {original_days}일 중 거래 시간이 부족한 {removed_days}일이 분석에서 제외되었습니다.")
    else:
        print("분석에서 제외할 단축 거래일이 없습니다.")

    filtered_df = df[df.index.to_series().dt.date.isin(full_trading_dates)]
    return filtered_df

def _calculate_prvm_for_day_optimized(date, daily_log_returns, num_tickers):
    """
    단일 거래일에 대한 PRVM을 계산하는 헬퍼 함수 (NumPy 벡터화로 최적화).
    """
    def g(x):
        return np.minimum(x, 1 - x)

    m = len(daily_log_returns)
    if m < 4: return date, None

    K = floor(m**0.5)
    if K < 2: return date, None

    Y_day_np = daily_log_returns.values
    num_k = m - K + 1

    # --- 1. Y_bar 행렬 계산 (벡터화 최적화) ---
    shape = (num_k, K - 1, num_tickers)
    strides = (Y_day_np.strides[0], Y_day_np.strides[0], Y_day_np.strides[1])
    Y_day_window_bar = as_strided(Y_day_np[:-1], shape=shape, strides=strides)
    
    weights_y_bar = g(np.arange(1, K) / K)
    Y_bar_matrix = np.einsum('j,kji->ki', weights_y_bar, Y_day_window_bar)

    # --- 2. 절삭 파라미터(u) 계산 (사용자 정의 제약조건 적용) ---
    alpha_u, c0 = 0.235, 4.0
    
    # ★★★ 수정된 부분 ★★★
    # 기존: np.std(m**(1/4) * Y_bar_matrix, axis=0, ddof=1)
    # 변경: 평균=0 가정, 분모=371 고정 조건에 맞춰 표준편차를 직접 계산
    scaled_Y_bar_sq = np.square(m**(1/4) * Y_bar_matrix)
    sum_of_squares = np.sum(scaled_Y_bar_sq, axis=0)
    std_devs = np.sqrt(sum_of_squares / 371)
    
    u_thresholds = (c0 * std_devs) * (m**(-alpha_u))

    # --- 3. Y_hat 텐서 계산 (벡터화 최적화) ---
    shape = (num_k, K, num_tickers)
    strides = (Y_day_np.strides[0], Y_day_np.strides[0], Y_day_np.strides[1])
    Y_day_window_hat = as_strided(Y_day_np, shape=shape, strides=strides)

    weights_y_hat_sq = (g(np.arange(1, K + 1) / K) - g(np.arange(0, K) / K))**2
    Y_hat_tensor = np.einsum('l,kli,klj->kij', weights_y_hat_sq, Y_day_window_hat, Y_day_window_hat)

    # --- 4. PRVM 합산 및 점프 조정 (벡터화 연산) ---
    is_not_jump_matrix = np.abs(Y_bar_matrix) <= u_thresholds 
    outer_valid_mask_tensor = np.einsum('ki,kj->kij',
                                        is_not_jump_matrix.astype(float),
                                        is_not_jump_matrix.astype(float))
    y_outer_tensor = np.einsum('ki,kj->kij', Y_bar_matrix, Y_bar_matrix)
    terms_to_add_tensor = (y_outer_tensor - 0.5 * Y_hat_tensor) * outer_valid_mask_tensor
    prvm_sum = np.sum(terms_to_add_tensor, axis=0)

    # --- 5. 최종 PRVM 계산 ---
    psi = 1 / 12
    final_prvm = (1 / (psi * K)) * prvm_sum
    np.fill_diagonal(final_prvm, np.maximum(np.diag(final_prvm), 0))
    
    return date, final_prvm

def calculate_prvm_parallel_optimized(data: pd.DataFrame, start_year: int = 2016, end_year: int = 2019) -> tuple[dict, list]:
    """
    최적화된 벡터화 함수와 병렬 처리를 사용하여 PRVM을 계산합니다.
    """
    print(f"PRVM 계산을 시작합니다 (최적화 벡터화 및 병렬 처리 버전). 대상 기간: {start_year}-{end_year}")

    tickers = data.columns.tolist()
    num_tickers = len(tickers)

    log_prices = data
    log_returns = log_prices.diff().dropna()

    print(f"{start_year}년부터 {end_year}년까지의 데이터로 필터링합니다...")
    filtered_log_returns = log_returns[
        (log_returns.index.year >= start_year) & (log_returns.index.year <= end_year)
    ]

    if filtered_log_returns.empty:
        warnings.warn("경고: 지정된 기간에 해당하는 데이터가 없습니다.")
        return {}, []

    daily_groups = filtered_log_returns.groupby(filtered_log_returns.index.date)
    
    num_days = len(daily_groups)
    print(f"\n분석 기간({start_year}-{end_year})에 포함된 영업일 수: {num_days}일")
    if start_year == 2016 and end_year == 2019:
        if num_days == 997:
            print("-> 목표 영업일 수(997일)와 일치하는 것을 확인했습니다.")
        else:
            print(f"-> 경고: 목표 영업일 수(997일)와 다릅니다. 현재 {num_days}일 입니다. 'remove_half_trading_days' 함수 실행 결과를 확인해주세요.")

    tasks = [(date, daily_data) for date, daily_data in daily_groups]

    print(f"\n{len(tasks)}일의 데이터에 대해 병렬 PRVM 계산을 시작합니다 (joblib 사용)...")
    
    with tqdm_joblib(tqdm(total=len(tasks), desc="일별 PRVM 계산 중")) as progress_bar:
        results = joblib.Parallel(n_jobs=-1, backend="multiprocessing")(
            joblib.delayed(_calculate_prvm_for_day_optimized)(date, daily_data, num_tickers) for date, daily_data in tasks
        )

    daily_prvms = {date: prvm for date, prvm in results if prvm is not None}

    print("PRVM 계산이 완료되었습니다.")
    return daily_prvms, tickers

def plot_log_kurtosis(daily_prvms: dict, tickers: list):
    """
    계산된 PRVM의 분산으로부터 로그 첨도를 계산하고 박스플롯을 그립니다.
    """
    if not daily_prvms:
        print("계산된 PRVM 데이터가 없어 플롯을 생성할 수 없습니다.")
        return

    print("\n로그 첨도 박스플롯 생성을 시작합니다...")
    dates = sorted(daily_prvms.keys())
    variances_list = [np.diag(daily_prvms[date]) for date in dates]
    daily_variances = pd.DataFrame(variances_list, index=dates, columns=tickers, dtype=float)

    if len(daily_variances) < 4:
        print(f"오류: 첨도를 계산하려면 최소 4일의 데이터가 필요하지만, 현재 {len(daily_variances)}일의 데이터만 있습니다.")
        return

    kurtosis_values = daily_variances.kurtosis() + 3
    positive_kurtosis = kurtosis_values[kurtosis_values > 0]

    if positive_kurtosis.empty:
        print("경고: 유효한 양수 첨도 값이 없어 플롯을 생성할 수 없습니다.")
        return

    log_kurtosis = np.log(positive_kurtosis)

    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(x=log_kurtosis, ax=ax, color='lightgray', width=0.4, showfliers=True)

    nu = 5
    t_kurtosis = 6 / (nu - 4) + 3
    log_t_kurtosis = np.log(t_kurtosis)
    ax.axvline(x=log_t_kurtosis, color='red', linestyle='--', linewidth=1, label=f't-distribution with dof={nu}')

    ax.set_title('Box plot of Log Kurtosis')
    ax.set_xlabel('Log Kurtosis')
    ax.legend()
    plt.show()

def save_as_single_csv(daily_prvms: dict, tickers: list, filepath: str):
    """
    모든 일별 PRVM 결과를 단일 CSV 파일(Long Format)으로 저장합니다.
    """
    print(f"모든 결과를 단일 CSV 파일 '{filepath}'로 저장합니다...")
    all_data = []
    for date, prvm_matrix in tqdm(daily_prvms.items(), desc="데이터 변환 중"):
        prvm_df = pd.DataFrame(prvm_matrix, index=tickers, columns=tickers)
        long_format = prvm_df.unstack().reset_index()
        long_format.columns = ['ticker_i', 'ticker_j', 'value']
        long_format['date'] = date
        all_data.append(long_format)

    final_df = pd.concat(all_data, ignore_index=True)
    final_df = final_df[['date', 'ticker_i', 'ticker_j', 'value']]
    final_df.to_csv(filepath, index=False)
    print(f"'{filepath}'에 모든 데이터 저장을 완료했습니다.")


if __name__ == '__main__':
    # 새로운 데이터 파일 경로를 지정해주세요.
    real_data_filepath = 'DataY.csv'

    try:
        cleaned_data = load_and_clean_data(real_data_filepath)
        cleaned_data_full_days = remove_half_trading_days(cleaned_data)

        # 최적화된 버전의 함수를 호출합니다.
        prvm_results, tickers_list = calculate_prvm_parallel_optimized(cleaned_data_full_days, start_year=2016, end_year=2019)

        if prvm_results:
            plot_log_kurtosis(prvm_results, tickers_list)
            save_as_single_csv(prvm_results, tickers_list, "prvm_0731_final_corrected.csv")
        else:
            print("\n지정된 기간에 대해 계산된 PRVM 결과가 없습니다.")

    except FileNotFoundError:
        print(f"\n[오류] 파일을 찾을 수 없습니다.")
        print(f"지정한 경로 '{real_data_filepath}'에 파일이 있는지, 파일명이 올바른지 확인해주세요.")
    except Exception as e:
        print(f"\n[오류] 스크립트 실행 중 문제가 발생했습니다: {e}")
