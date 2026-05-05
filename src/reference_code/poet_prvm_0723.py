import numpy as np
import pandas as pd
from numpy.linalg import eigh, inv, norm
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- 데이터 저장을 위한 헬퍼 함수 ---
def save_matrices_to_csv(matrices, dates, tickers, filename):
    """
    3D 행렬 배열(T, p, p)을 long-format의 DataFrame으로 변환하여 CSV로 저장합니다.
    
    Args:
        matrices (np.array): 저장할 3D 행렬 배열 (T, p, p)
        dates (pd.DatetimeIndex): 날짜 인덱스 (길이 T)
        tickers (list): 자산 티커 목록 (길이 p)
        filename (str): 저장할 CSV 파일 이름
    """
    print(f"'{filename}' 파일로 데이터를 저장합니다...")
    num_days, num_assets, _ = matrices.shape
    
    # 데이터를 저장할 리스트 초기화
    records = []
    
    # 모든 날짜와 행렬에 대해 반복
    for t in tqdm(range(num_days), desc=f"'{filename}' 변환 중"):
        date = dates[t]
        matrix = matrices[t]
        for i in range(num_assets):
            for j in range(num_assets):
                # 대칭 행렬이므로 하삼각행렬(lower triangle)만 저장하여 파일 크기 최적화
                if i >= j:
                    records.append({
                        "date": date.strftime('%Y-%m-%d'),
                        "ticker_i": tickers[i],
                        "ticker_j": tickers[j],
                        "value": matrix[i, j]
                    })
    
    # 리스트를 DataFrame으로 변환
    df_to_save = pd.DataFrame(records)
    
    # CSV 파일로 저장
    df_to_save.to_csv(filename, index=False)
    print(f"저장 완료: {filename}")


# --- 1. 데이터 준비 (기존 코드와 동일) ---
def load_csv_data(file_path):
    """
    long-format의 .csv 파일에서 PRVM 데이터를 불러와 3D 텐서로 변환합니다.
    """
    print(f"'{file_path}' 파일에서 CSV 데이터를 불러옵니다...")
    try:
        df = pd.read_csv(file_path, parse_dates=['date'])
        unique_dates = sorted(df['date'].unique())
        tickers = sorted(pd.unique(df[['ticker_i', 'ticker_j']].values.ravel('K')))
        num_days = len(unique_dates)
        num_assets = len(tickers)
        
        print(f"총 {num_days}일, {num_assets}개의 자산 데이터를 처리합니다.")

        prvm_data = np.zeros((num_days, num_assets, num_assets))
        grouped = df.groupby('date')
        
        for i, date in enumerate(tqdm(unique_dates, desc="데이터 변환 중")):
            daily_data = grouped.get_group(date)
            matrix_df = daily_data.pivot_table(index='ticker_i', columns='ticker_j', values='value')
            matrix_df = matrix_df.reindex(index=tickers, columns=tickers, fill_value=0)
            matrix = matrix_df.to_numpy()
            # 상삼각행렬을 하삼각행렬로 복사하여 대칭 행렬 완성
            lower_triangle = np.tril(matrix)
            matrix = lower_triangle + lower_triangle.T - np.diag(np.diag(matrix))
            prvm_data[i] = matrix

    except Exception as e:
        print(f"파일을 읽는 중 오류가 발생했습니다: {e}")
        return None, None, None

    dates = pd.to_datetime(unique_dates)
    print(f"데이터 로딩 및 변환 완료: {prvm_data.shape} 형태")
    print(f"날짜 범위: {dates.min().date()} ~ {dates.max().date()}")
    return prvm_data, dates, tickers

def load_gics_data(gics_file_path, num_assets):
    """GICS 분류가 담긴 CSV 파일을 불러옵니다."""
    print(f"'{gics_file_path}' 파일에서 GICS 데이터를 불러옵니다...")
    try:
        gics_df = pd.read_csv(gics_file_path, header=None)
        gics_sectors = gics_df.iloc[1:, 0].to_numpy()
        if len(gics_sectors) != num_assets:
            print(f"GICS 데이터의 개수({len(gics_sectors)})와 자산의 수({num_assets})가 일치하지 않습니다.")
            return None
        return gics_sectors
    except Exception as e:
        print(f"GICS 파일을 읽는 중 오류가 발생했습니다: {e}")
        return None

def project_psd(matrix):
    """행렬을 양의 준정부호(Positive Semi-Definite) 행렬로 변환합니다."""
    eigenvalues, eigenvectors = eigh(matrix)
    eigenvalues[eigenvalues < 0] = 0
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

# --- 2. POET 기법 구현 (기존 코드와 동일) ---
def apply_poet(prvm, gics_sectors, num_factors=3):
    """
    주어진 PRVM 행렬에 GICS 기반 hard-thresholding을 적용합니다.
    """
    p = prvm.shape[0]
    prvm_psd = project_psd(prvm)
    eigenvalues, eigenvectors = eigh(prvm_psd)
    
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    factor_part = np.zeros_like(prvm)
    for i in range(num_factors):
        factor_part += eigenvalues[i] * np.outer(eigenvectors[:, i], eigenvectors[:, i])
        
    idiosyncratic_part_raw = prvm_psd - factor_part
    same_sector_mask = (gics_sectors.reshape(-1, 1) == gics_sectors)
    idiosyncratic_part_thresholded = idiosyncratic_part_raw * same_sector_mask
    
    poet_prvm = factor_part + idiosyncratic_part_thresholded
    final_poet_prvm_psd = project_psd(poet_prvm)
    
    return final_poet_prvm_psd, idiosyncratic_part_thresholded, factor_part

# --- 3. 성능 평가 및 플롯 생성 (기존 코드와 동일) ---
def calculate_mspe(forecasts, ground_truths):
    """MSPE (Mean Squared Prediction Error)를 계산합니다."""
    T = len(forecasts)
    if T == 0: return np.nan
    errors = [norm(forecasts[i] - ground_truths[i], 'fro')**2 for i in range(T)]
    return np.mean(errors)

def calculate_qlike(forecasts, ground_truths, reg=1e-8):
    """QLIKE 손실 함수를 계산합니다."""
    T = len(forecasts)
    if T == 0: return np.nan
    qlike_vals = []
    for i in tqdm(range(T), desc="QLIKE 계산 중"):
        forecast_reg = forecasts[i] + np.eye(forecasts[i].shape[0]) * reg
        try:
            eigenvalues = eigh(forecast_reg)[0]
            positive_eigenvalues = eigenvalues[eigenvalues > 1e-12]
            if len(positive_eigenvalues) == 0: continue
            log_det_val = np.sum(np.log(positive_eigenvalues))
            trace_val = np.trace(inv(forecast_reg) @ ground_truths[i])
            qlike_vals.append(log_det_val + trace_val)
        except np.linalg.LinAlgError:
            print(f"경고: {i}번째 데이터 계산에 실패하여 건너뜁니다.")
            continue
    return np.mean(qlike_vals) if qlike_vals else np.nan

def plot_eigen_gaps(idio_matrices, num_factors=3, num_gaps_to_plot=100):
    """Eigen Gap 플롯을 생성합니다."""
    print("\nEigen Gap 플롯을 생성합니다...")
    avg_idio_matrix = np.mean(idio_matrices, axis=0)
    eigenvalues = eigh(avg_idio_matrix)[0]
    sorted_eigenvalues = np.sort(eigenvalues)[::-1]
    eigen_gaps = -np.diff(sorted_eigenvalues)
    gaps_to_plot = eigen_gaps[num_factors : num_factors + num_gaps_to_plot]
    x_axis = range(1, len(gaps_to_plot) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(x_axis, gaps_to_plot, facecolors='none', edgecolors='b')
    plt.title("Idiosyncratic Volatility Matrix Eigen Gaps", fontsize=16)
    plt.xlabel("Index", fontsize=12)
    plt.ylabel("Eigen Gap", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

# --- 4. 메인 실행 로직 (수정됨) ---
def main():
    prvm_file_path = 'prvm_0731_final_corrected.csv' 
    gics_file_path = 'gicslist.csv'
    
    prvm_data_all, dates, tickers = load_csv_data(prvm_file_path)
    if prvm_data_all is None: return

    gics_data = load_gics_data(gics_file_path, len(tickers))
    if gics_data is None: return

    num_factors = 3

    print("\n모든 날짜에 대해 POET-PRVM, Idio Matrix 및 Factor Part를 계산합니다...")
    results_list = [apply_poet(prvm, gics_data, num_factors=num_factors) for prvm in tqdm(prvm_data_all, desc="POET 적용 중")]
    poet_prvm_all = np.array([res[0] for res in results_list])
    idio_matrices_all = np.array([res[1] for res in results_list])
    factor_parts_all = np.array([res[2] for res in results_list])

    plot_eigen_gaps(idio_matrices_all, num_factors=num_factors)

    try:
        start_2018 = np.where(dates >= '2018-01-01')[0][0]
        end_2018 = np.where(dates <= '2018-12-31')[0][-1]
        start_2019 = np.where(dates >= '2019-01-01')[0][0]
        end_2019 = np.where(dates <= '2019-12-31')[0][-1]

        periods = {
            "Period 1 (2018-2019)": (start_2018, end_2019),
            "Period 2 (2018)": (start_2018, end_2018),
            "Period 3 (2019)": (start_2019, end_2019),
        }
    except IndexError:
        print("평가 기간을 정의하는 중 오류 발생: 데이터가 2018-2019년 기간을 포함하지 않습니다.")
        return
    
    results = []

    for period_name, (start_idx, end_idx) in periods.items():
        print(f"\n--- {period_name} 평가 시작 ---")
        
        # 1. MSPE 계산: 순수 요인 부분(Factor Part)의 예측력을 평가
        mspe_forecasts = factor_parts_all[start_idx-1 : end_idx]
        mspe_truths = factor_parts_all[start_idx : end_idx+1]
        
        # 2. QLIKE 계산: 전체 POET 모델의 예측력을 평가
        # 예측값(Forecast): t-1 시점의 POET 모델 결과
        qlike_forecasts = poet_prvm_all[start_idx-1 : end_idx]
        # 실제값(Ground Truth): t 시점의 가공되지 않은 실제 데이터
        qlike_truths = prvm_data_all[start_idx : end_idx+1]
        
        # 평가 수행
        print("순수 요인 부분(Factor Part)에 대한 MSPE를 계산합니다...")
        mspe_val = calculate_mspe(mspe_forecasts, mspe_truths)
        
        print("전체 POET-PRVM 모델에 대한 QLIKE를 계산합니다...")
        qlike_val = calculate_qlike(qlike_forecasts, qlike_truths)
        
        results.append({
            "Period": period_name,
            "MSPE (x10^4)": mspe_val * 10**4 if not np.isnan(mspe_val) else 'N/A',
            "QLIKE (x10^-3)": qlike_val * 10**-3 if not np.isnan(qlike_val) else 'N/A',
        })

    results_df = pd.DataFrame(results)
    print("\n--- 최종 평가 결과 ---")
    print(results_df.to_string(index=False))

    # --- FIVAR 분석을 위한 데이터 저장 ---
    print("\n--- FIVAR 분석을 위한 데이터 저장 시작 ---")
    save_matrices_to_csv(prvm_data_all, dates, tickers, "fivar_input_prvm.csv")
    save_matrices_to_csv(idio_matrices_all, dates, tickers, "fivar_input_idio.csv")
    # 선택적으로 factor_parts_all도 저장할 수 있습니다.
    # save_matrices_to_csv(factor_parts_all, dates, tickers, "fivar_input_factor.csv")
    print("--- 모든 데이터 저장 완료 ---")


if __name__ == '__main__':
    main()
