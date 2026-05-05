import numpy as np
import pandas as pd
# import h5py # .h5 파일을 더 이상 사용하지 않으므로 주석 처리 또는 삭제
from numpy.linalg import eigh, inv, det, norm
from tqdm import tqdm
import matplotlib.pyplot as plt # 플롯 생성을 위해 matplotlib 라이브러리를 추가합니다.

# --- 1. 데이터 준비 ---
# .csv 파일에서 데이터를 불러오도록 수정된 함수입니다.
def load_csv_data(file_path):
    """
    long-format의 .csv 파일에서 PRVM 데이터를 불러와 3D 텐서로 변환합니다.
    
    Args:
        file_path (str): .csv 파일의 전체 경로

    Returns:
        tuple: (PRVM 데이터 배열, 날짜 인덱스, 티커 목록)
    """
    print(f"'{file_path}' 파일에서 CSV 데이터를 불러옵니다...")
    try:
        # CSV 파일을 pandas DataFrame으로 읽습니다.
        df = pd.read_csv(file_path, parse_dates=['date'])

        # 고유한 날짜와 티커 목록을 추출하고 정렬하여 일관성을 유지합니다.
        unique_dates = sorted(df['date'].unique())
        tickers = sorted(pd.unique(df[['ticker_i', 'ticker_j']].values.ravel('K')))
        num_days = len(unique_dates)
        num_assets = len(tickers)
        
        print(f"총 {num_days}일, {num_assets}개의 자산 데이터를 처리합니다.")

        # 최종 3D 배열을 0으로 초기화합니다.
        prvm_data = np.zeros((num_days, num_assets, num_assets))
        
        # 날짜별로 그룹화하여 처리 속도를 높입니다.
        grouped = df.groupby('date')
        
        # 각 날짜에 대해 2D 행렬을 생성합니다.
        for i, date in enumerate(tqdm(unique_dates, desc="데이터 변환 중")):
            daily_data = grouped.get_group(date)
            # pivot_table을 사용하여 long-format 데이터를 2D 행렬(매트릭스)로 변환합니다.
            matrix_df = daily_data.pivot_table(index='ticker_i', columns='ticker_j', values='value')
            # 모든 티커가 포함되도록 reindex하고, 누락된 값은 0으로 채웁니다.
            matrix_df = matrix_df.reindex(index=tickers, columns=tickers, fill_value=0)
            # 대칭 행렬이 아닐 경우, 전치 행렬을 더해 대칭으로 만듭니다.
            matrix = matrix_df.to_numpy()
            matrix = (matrix + matrix.T) / 2
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
        # 설명에 따라 첫 번째 행(0)을 제외하고 데이터를 가져옵니다.
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
    eigenvalues[eigenvalues < 0] = 0  # 음수 고유값을 0으로 만듭니다.
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

# --- 2. POET 기법 구현 ---
def apply_poet(prvm, gics_sectors, num_factors=3):
    """
    주어진 PRVM 행렬에 GICS 기반 hard-thresholding을 적용합니다.
    
    Args:
        prvm (np.array): (p, p) 크기의 단일 PRVM 행렬
        gics_sectors (np.array): 자산별 GICS 분류 번호가 담긴 배열
        num_factors (int): 사용할 공통 요인의 수 (논문에서는 r=3 사용)

    Returns:
        tuple: (최종 POET-PRVM 행렬, 특이 변동성 행렬)
    """
    p = prvm.shape[0]
    
    # Step 1 & 2: 고유값 분해 및 요인(Factor) 부분 추출
    prvm_psd = project_psd(prvm)
    eigenvalues, eigenvectors = eigh(prvm_psd)
    
    # 고유값을 내림차순으로 정렬
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    factor_part = np.zeros_like(prvm)
    for i in range(num_factors):
        factor_part += eigenvalues[i] * np.outer(eigenvectors[:, i], eigenvectors[:, i])
        
    # Step 3: 특이 변동성(Idiosyncratic) 부분 계산
    idiosyncratic_part_raw = prvm_psd - factor_part
    
    # Step 4: GICS 기반 hard-thresholding 적용
    same_sector_mask = (gics_sectors.reshape(-1, 1) == gics_sectors)
    idiosyncratic_part_thresholded = idiosyncratic_part_raw * same_sector_mask
    
    
    # 최종 POET-PRVM 행렬
    poet_prvm = factor_part + idiosyncratic_part_thresholded
    final_poet_prvm_psd = project_psd(poet_prvm)
    
    # PSD 처리된 최종 행렬과 특이 변동성 행렬을 반환
    return final_poet_prvm_psd, idiosyncratic_part_thresholded


# --- 3. 성능 평가 및 플롯 생성 ---
def calculate_mspe(forecasts, ground_truths):
    """MSPE (Mean Squared Prediction Error)를 계산합니다."""
    T = len(forecasts)
    errors = [norm(forecasts[i] - ground_truths[i], 'fro')**2 for i in range(T)]
    return np.mean(errors)

def calculate_qlike(forecasts, ground_truths, reg=1e-8):
    """QLIKE 손실 함수를 계산합니다."""
    T = len(forecasts)
    qlike_vals = []
    for i in tqdm(range(T), desc="QLIKE 계산 중"):
        forecast_reg = forecasts[i] + np.eye(forecasts[i].shape[0]) * reg
        try:
            eigenvalues = eigh(forecast_reg)[0]
            positive_eigenvalues = eigenvalues[eigenvalues > 1e-12]
            log_det_val = np.sum(np.log(positive_eigenvalues))
            trace_val = np.trace(inv(forecast_reg) @ ground_truths[i])
            qlike_vals.append(log_det_val + trace_val)
        except np.linalg.LinAlgError:
            print(f"경고: {i}번째 데이터 계산에 실패하여 건너뜁니다.")
            continue
    return np.mean(qlike_vals)

def plot_eigen_gaps(idio_matrices, num_factors=3, num_gaps_to_plot=100):
    """
    논문 62페이지의 Figure 7과 같이 Eigen Gap 플롯을 생성합니다.
    잔여 요인 효과를 제거하기 위해 처음 num_factors개의 gap은 제외합니다.
    """
    print("\nEigen Gap 플롯을 생성합니다...")
    # 1. 모든 날짜의 특이 변동성 행렬의 평균을 계산합니다.
    avg_idio_matrix = np.mean(idio_matrices, axis=0)
    
    # 2. 평균 행렬의 고유값을 계산하고 내림차순으로 정렬합니다.
    eigenvalues = eigh(avg_idio_matrix)[0]
    sorted_eigenvalues = np.sort(eigenvalues)[::-1]
    
    # 3. 연속된 고유값 간의 차이(gap)를 계산합니다.
    eigen_gaps = -np.diff(sorted_eigenvalues)
    
    # 4. 잔여 요인에 해당하는 처음 num_factors개의 gap을 제외하고 플롯할 데이터를 준비합니다.
    gaps_to_plot = eigen_gaps[num_factors : num_factors + num_gaps_to_plot]
    x_axis = range(1, len(gaps_to_plot) + 1)
    
    # 5. 플롯을 생성합니다.
    plt.figure(figsize=(10, 6))
    plt.scatter(x_axis, gaps_to_plot, facecolors='none', edgecolors='b')
    plt.title("idio volatility matrix", fontsize=16)
    plt.xlabel("Index", fontsize=12)
    plt.ylabel("Eigen Gap", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

# --- 4. 메인 실행 로직 ---
def main():
    # 1. 데이터 로드 및 준비
    prvm_file_path = 'prvm_aaaa.csv' 
    gics_file_path = 'gicslist.csv'
    
    prvm_data_all, dates, tickers = load_csv_data(prvm_file_path)
    if prvm_data_all is None: return

    gics_data = load_gics_data(gics_file_path, len(tickers))
    if gics_data is None: return

    # 사용할 요인의 수를 변수로 정의합니다.
    num_factors = 3

    # 2. 모든 날짜에 대해 POET-PRVM 및 Idio Matrix 계산
    print("\n모든 날짜에 대해 POET-PRVM 및 Idio Matrix를 계산합니다...")
    results_list = [apply_poet(prvm, gics_data, num_factors=num_factors) for prvm in tqdm(prvm_data_all, desc="POET 적용 중")]
    poet_prvm_all = np.array([res[0] for res in results_list])
    idio_matrices_all = np.array([res[1] for res in results_list])

    # 3. Eigen Gap 플롯 생성
    plot_eigen_gaps(idio_matrices_all, num_factors=num_factors)

    # 4. 평가 기간 정의
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

    # 5. 각 기간에 대해 평가 실행
    for period_name, (start_idx, end_idx) in periods.items():
        print(f"\n--- {period_name} 평가 시작 ---")
        
        forecasts = poet_prvm_all[start_idx-1 : end_idx]
        ground_truths = poet_prvm_all[start_idx : end_idx+1]
        
        mspe_val = calculate_mspe(forecasts, ground_truths)
        qlike_val = calculate_qlike(forecasts, ground_truths)
        
        results.append({
            "Period": period_name,
            "MSPE (x10^4)": mspe_val * 10**4,
            "QLIKE (x10^-3)": qlike_val * 10**-3
        })

    # 6. 결과 출력
    results_df = pd.DataFrame(results)
    print("\n--- 최종 평가 결과 (POET-PRVM) ---")
    print(results_df.to_string(index=False))
    print("\n참고: 논문의 Table 2와 유사한 형식으로 결과를 출력했습니다.")
    print("MSPE와 QLIKE의 스케일은 논문과 동일하게 맞추었습니다.")

if __name__ == '__main__':
    main()
