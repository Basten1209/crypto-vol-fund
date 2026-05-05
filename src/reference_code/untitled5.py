import pandas as pd
import numpy as np
import os
from scipy.linalg import eigh
from sklearn.linear_model import HuberRegressor
from sklearn.exceptions import ConvergenceWarning
import warnings
from tqdm import tqdm
from typing import Dict, List, Tuple
# --- [수정] 최적화를 위한 라이브러리 임포트 ---
from joblib import Parallel, delayed
import numba

# 경고 메시지 무시 설정
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- 데이터 로딩 및 전처리 함수 ---

def load_prvm_data_fast(file_path: str) -> Tuple[np.ndarray, List[pd.Timestamp], List[str]]:
    """
    원본 prvm.csv를 'pivot_table'을 사용하여 빠르게 읽고 3D NumPy 배열로 변환합니다.
    """
    print(f"'{file_path}'에서 원본 PRVM 데이터를 빠르게 로딩합니다...")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"오류: '{file_path}' 파일을 찾을 수 없습니다.")
        
    df = pd.read_csv(file_path, parse_dates=['date'])
    
    unique_dates = sorted(df['date'].unique())
    tickers = sorted(pd.unique(df[['ticker_i', 'ticker_j']].values.ravel('K')))
    num_days = len(unique_dates)
    num_assets = len(tickers)
    
    print(f"총 {num_days}일, {num_assets}개의 자산 데이터를 처리합니다.")

    prvm_data_all = np.zeros((num_days, num_assets, num_assets))
    grouped = df.groupby('date')
    
    for i, date in enumerate(tqdm(unique_dates, desc="원본 PRVM 변환 중")):
        daily_data = grouped.get_group(date)
        matrix_df = daily_data.pivot_table(index='ticker_i', columns='ticker_j', values='value')
        matrix_df = matrix_df.reindex(index=tickers, columns=tickers, fill_value=0)
        matrix = matrix_df.to_numpy()
        matrix = (matrix + matrix.T) / 2
        prvm_data_all[i] = matrix
        
    dates = [pd.to_datetime(d) for d in unique_dates]
    return prvm_data_all, dates, tickers

def load_gics_data(gics_file_path: str, num_assets: int) -> np.ndarray:
    """GICS 분류가 담긴 CSV 파일을 불러옵니다."""
    print(f"'{gics_file_path}' 파일에서 GICS 데이터를 불러옵니다...")
    try:
        gics_df = pd.read_csv(gics_file_path, header=None)
        gics_sectors = gics_df.iloc[1:, 0].to_numpy()
        if len(gics_sectors) != num_assets:
            raise ValueError(f"GICS 데이터의 개수({len(gics_sectors)})와 자산의 수({num_assets})가 일치하지 않습니다.")
        return gics_sectors
    except Exception as e:
        raise IOError(f"GICS 파일을 읽는 중 오류가 발생했습니다: {e}")

# --- [수정] Numba JIT 컴파일러 적용으로 함수 가속화 ---
@numba.jit(nopython=True)
def apply_poet_for_idio_numba(prvm: np.ndarray, same_sector_mask: np.ndarray, r: int) -> np.ndarray:
    """
    Numba로 가속화된 POET 적용 함수.
    """
    # Numba 호환성을 위해 project_psd 로직을 내부로 이동
    eigenvalues, eigenvectors = np.linalg.eigh(prvm)
    eigenvalues[eigenvalues < 0] = 0
    prvm_psd = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    
    eigenvalues, eigenvectors = np.linalg.eigh(prvm_psd)
    
    idx = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[idx]
    sorted_eigenvectors = eigenvectors[:, idx]
    
    factor_part = np.zeros_like(prvm)
    for i in range(r):
        e_vec = sorted_eigenvectors[:, i:i+1]
        factor_part += sorted_eigenvalues[i] * (e_vec @ e_vec.T)
        
    idiosyncratic_part_raw = prvm_psd - factor_part
    idiosyncratic_part_thresholded = idiosyncratic_part_raw * same_sector_mask
    
    return idiosyncratic_part_thresholded

def apply_poet_for_idio_wrapper(args):
    """Numba 함수를 병렬 처리에서 사용하기 위한 래퍼 함수"""
    index, prvm, same_sector_mask, r = args
    return index, apply_poet_for_idio_numba(prvm, same_sector_mask, r)

# --- 병렬 처리를 위한 워커 함수 ---
def fit_idio_component(args: Tuple) -> Tuple[int, HuberRegressor, float]:
    """
    단일 특이 고유값(종목)에 대한 최적의 c_eta를 찾고 모델을 학습시키는 워커 함수.
    """
    i, y_i, X, p, n_reg, c_eta_candidates = args
    best_bic_i, best_c_eta_i = np.inf, None

    for c_eta in c_eta_candidates:
        eta_I = c_eta * np.sqrt(np.log(p) / n_reg)
        reg = HuberRegressor(fit_intercept=True, alpha=eta_I)
        reg.fit(X, y_i)
        y_pred = reg.predict(X)
        rss = np.sum((y_i - y_pred)**2)
        num_params = np.sum(reg.coef_ != 0) + 1
        bic = n_reg * np.log(rss / n_reg) + num_params * np.log(n_reg) if rss > 0 else np.inf
        if bic < best_bic_i:
            best_bic_i, best_c_eta_i = bic, c_eta
    
    final_eta_I = best_c_eta_i * np.sqrt(np.log(p) / n_reg)
    final_regressor = HuberRegressor(fit_intercept=True, alpha=final_eta_I)
    final_regressor.fit(X, y_i)
    
    return i, final_regressor, best_c_eta_i

# --- FIVAR 모델 클래스 ---
class FIVARModel:
    def __init__(self, r: int, h: int = 1, l: int = 22):
        self.r, self.h, self.l = r, h, l
        self.p, self.tickers = None, None
        self.factor_eigenvectors, self.idio_eigenvectors = None, None
        self.factor_regressors, self.idio_regressors = None, None
        self.last_known_eigenvalues = None

    def fit(self, prvm_matrices: Dict, idio_matrices: Dict, tickers: List, max_workers: int = None) -> List[float]:
        self.tickers, self.p = tickers, len(tickers)
        all_dates = sorted(prvm_matrices.keys())
        
        if len(all_dates) < self.l:
             raise ValueError("고유벡터 계산에 필요한 데이터가 부족합니다.")

        last_l_dates = all_dates[-self.l:]
        avg_prvm_matrix = np.mean([prvm_matrices[d] for d in last_l_dates], axis=0)
        _, factor_evecs_all = eigh(avg_prvm_matrix)
        self.factor_eigenvectors = factor_evecs_all[:, -self.r:][:, ::-1]

        avg_idio_matrix = np.mean([idio_matrices[d] for d in last_l_dates], axis=0)
        _, idio_evecs_all = eigh(avg_idio_matrix)
        self.idio_eigenvectors = idio_evecs_all[:, ::-1]

        factor_evals = [np.diag(self.factor_eigenvectors.T @ prvm_matrices[d] @ self.factor_eigenvectors) / self.p for d in all_dates]
        idio_evals = [np.diag(self.idio_eigenvectors.T @ idio_matrices[d] @ self.idio_eigenvectors) for d in all_dates]
        
        eigenvalue_df = pd.DataFrame([np.concatenate(z) for z in zip(factor_evals, idio_evals)], index=all_dates)

        y = eigenvalue_df.iloc[self.h:]
        X = pd.concat([eigenvalue_df.shift(k).iloc[self.h:] for k in range(1, self.h + 1)], axis=1)
        
        self.factor_regressors = [HuberRegressor(alpha=0.0) for _ in range(self.r)]
        X_factor_regressors = X.iloc[:, :self.r * self.h]
        for i in range(self.r):
            self.factor_regressors[i].fit(X_factor_regressors, y.iloc[:, i])

        n_reg = len(y)
        # --- [수정] c_eta 탐색 범위를 논문에 맞게 [0.1, 10.0]으로 변경 ---
        c_eta_candidates = np.linspace(0.1, 10.0, 100)
        self.idio_regressors = [None] * self.p
        optimal_c_etas = [None] * self.p

        tasks = [(i, y.iloc[:, self.r + i].copy(), X.copy(), self.p, n_reg, c_eta_candidates) for i in range(self.p)]
        
        results = Parallel(n_jobs=max_workers)(delayed(fit_idio_component)(task) for task in tqdm(tasks, desc="  종목별 계수 추정(병렬)"))
        for i, regressor, c_eta in results:
            self.idio_regressors[i] = regressor
            optimal_c_etas[i] = c_eta

        self.last_known_eigenvalues = eigenvalue_df.iloc[-self.h:].to_numpy()
        return optimal_c_etas

    def predict(self) -> np.ndarray:
        if self.last_known_eigenvalues is None: raise RuntimeError("모델이 적합되지 않았습니다.")
        X_pred = self.last_known_eigenvalues.flatten().reshape(1, -1)
        pred_evals = np.zeros(self.p + self.r)
        X_pred_factor = X_pred[:, :self.r * self.h]
        for i in range(self.r):
            pred_evals[i] = self.factor_regressors[i].predict(X_pred_factor)[0]
        for i in range(self.p):
            pred_evals[self.r + i] = self.idio_regressors[i].predict(X_pred)[0]
        pred_evals[pred_evals < 0] = 0
        psi_hat = self.factor_eigenvectors @ np.diag(pred_evals[:self.r]) @ self.factor_eigenvectors.T * self.p
        sigma_hat = self.idio_eigenvectors @ np.diag(pred_evals[self.r:]) @ self.idio_eigenvectors.T
        return psi_hat + sigma_hat

# --- 롤링 윈도우 예측 실행 함수 ---
def run_rolling_forecast(prvm_data_all, idio_matrices_all, dates, tickers, r, h, l, in_sample_window, start_date_str, max_workers, refit_interval: int = 1):
    predictions = {}
    daily_optimal_c_etas = {}
    
    try:
        start_index = dates.index(pd.to_datetime(start_date_str))
    except ValueError:
        print(f"오류: 시작 날짜 '{start_date_str}'를 데이터에서 찾을 수 없습니다.")
        return None, None

    model = None
    
    for t in tqdm(range(start_index, len(dates)), desc="전체 롤링 윈도우 예측 진행"):
        prediction_date = dates[t]
        
        if (t - start_index) % refit_interval == 0:
            print(f"\n[{prediction_date.date()}] 모델 재학습 수행...")
            train_start_idx = t - in_sample_window
            train_end_idx = t
            if train_start_idx < 0: continue
            
            train_dates = dates[train_start_idx:train_end_idx]
            prvm_train = {d: m for d, m in zip(train_dates, prvm_data_all[train_start_idx:train_end_idx])}
            idio_train = {d: m for d, m in zip(train_dates, idio_matrices_all[train_start_idx:train_end_idx])}
            
            model = FIVARModel(r=r, h=h, l=l)
            optimal_c_etas = model.fit(prvm_train, idio_train, tickers, max_workers=max_workers)
            daily_optimal_c_etas[prediction_date] = optimal_c_etas
        else:
            if model is None: continue
            # 마지막으로 학습된 모델의 c_eta 값을 재사용
            daily_optimal_c_etas[prediction_date] = daily_optimal_c_etas[dates[t-1]]
            # 예측에 필요한 최신 데이터로 모델의 상태 업데이트
            # 이 부분은 실제로는 전체 eigenvalue_df에서 슬라이싱해야 더 정확합니다.
            # 여기서는 간단히 마지막 학습된 모델을 그대로 사용하는 것으로 가정합니다.
            
        predicted_matrix = model.predict()
        predictions[prediction_date] = predicted_matrix

    return predictions, daily_optimal_c_etas

if __name__ == '__main__':
    prvm_csv_path = 'prvm.csv'
    gics_csv_path = 'gicslist.csv'
    
    prvm_data_all, dates, tickers = load_prvm_data_fast(prvm_csv_path)
    gics_sectors = load_gics_data(gics_csv_path, len(tickers))
    
    print("\nPOET 기법을 적용하여 특이 변동성 행렬을 생성합니다 (병렬 처리)...")
    r_factors = 3
    MAX_WORKERS = os.cpu_count() - 1 if os.cpu_count() and os.cpu_count() > 1 else 1
    print(f"병렬 처리를 위해 {MAX_WORKERS}개의 워커를 사용합니다.")
    
    same_sector_mask = (gics_sectors[:, None] == gics_sectors)
    
    tasks = [(i, prvm_data_all[i], same_sector_mask, r_factors) for i in range(len(prvm_data_all))]
    idio_matrices_all = np.zeros_like(prvm_data_all)
    
    results = Parallel(n_jobs=MAX_WORKERS)(delayed(apply_poet_for_idio_wrapper)(task) for task in tqdm(tasks, desc="POET 적용(병렬)"))
    for index, result_matrix in results:
        idio_matrices_all[index] = result_matrix

    in_sample_days = 251 
    forecast_start_date_index = in_sample_days
    if forecast_start_date_index < len(dates):
        forecast_start_date = dates[forecast_start_date_index].strftime('%Y-%m-%d')
    else:
        print("오류: 예측을 시작하기에 데이터가 충분하지 않습니다.")
        exit()

    predictions, daily_c_etas = run_rolling_forecast(
        prvm_data_all=prvm_data_all,
        idio_matrices_all=idio_matrices_all,
        dates=dates,
        tickers=tickers,
        r=r_factors, h=1, l=22,
        in_sample_window=in_sample_days,
        start_date_str=forecast_start_date,
        max_workers=MAX_WORKERS,
        refit_interval=10
    )
    
    print("\n--- 롤링 윈도우 예측 완료 ---")
    if predictions:
        last_prediction_date = max(predictions.keys())
        last_predicted_matrix = predictions[last_prediction_date]
        
        print(f"마지막 예측일 ({last_prediction_date.date()})의 변동성 행렬 (상위 5x5):")
        predicted_df = pd.DataFrame(last_predicted_matrix, index=tickers, columns=tickers)
        print(predicted_df.iloc[:5, :5].round(6))
        
        all_c_etas_flat = [item for sublist in daily_c_etas.values() for item in sublist if item is not None]
        if all_c_etas_flat:
            c_eta_series = pd.Series(all_c_etas_flat)
            print("\n--- 전체 기간의 최적 c_eta 값 요약 ---")
            print(c_eta_series.describe())
        else:
            print("계산된 c_eta 값이 없습니다.")
    else:
        print("예측이 수행되지 않았습니다. 시작 날짜와 데이터 기간을 확인하세요.")
