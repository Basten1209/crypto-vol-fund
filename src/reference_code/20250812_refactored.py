#%% --- [셀 1] 초기 설정: 라이브러리 임포트, 함수 및 클래스 정의 ---
# 이 셀은 스크립트 실행 시 가장 먼저 한 번만 실행하면 됩니다.
# 모든 필요한 함수와 클래스를 메모리에 로드합니다.

import pandas as pd
import numpy as np
import os
from scipy.linalg import eigh, inv
from numpy.linalg import norm
from sklearn.linear_model import HuberRegressor, Lasso, LinearRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
import warnings
from tqdm import tqdm
from typing import Dict, List, Tuple
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns

# --- 경고 메시지 무시 설정 ---
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# --- 1. 데이터 로딩 및 기본 헬퍼 함수 ---

def load_csv_data(file_path: str) -> Tuple[np.ndarray, pd.DatetimeIndex, List[str]]:
    """Long-format의 CSV 파일에서 PRVM 데이터를 불러와 3D 텐서로 변환합니다."""
    print(f"'{file_path}' 파일에서 PRVM 데이터를 불러옵니다...")
    try:
        df = pd.read_csv(file_path, parse_dates=['date'])
        unique_dates = sorted(df['date'].unique())
        tickers = sorted(pd.unique(df[['ticker_i', 'ticker_j']].values.ravel('K')))
        num_days = len(unique_dates)
        num_assets = len(tickers)
        
        print(f"총 {num_days}일, {num_assets}개의 자산 데이터를 처리합니다.")

        prvm_data = np.zeros((num_days, num_assets, num_assets))
        ticker_map = {ticker: i for i, ticker in enumerate(tickers)}
        
        df['i'] = df['ticker_i'].map(ticker_map)
        df['j'] = df['ticker_j'].map(ticker_map)

        for t, date in enumerate(tqdm(unique_dates, desc="데이터 텐서 변환 중")):
            daily_data = df[df['date'] == date]
            matrix = np.zeros((num_assets, num_assets))
            matrix[daily_data['i'], daily_data['j']] = daily_data['value']
            matrix[daily_data['j'], daily_data['i']] = daily_data['value']
            prvm_data[t] = matrix

    except Exception as e:
        print(f"파일을 읽는 중 오류가 발생했습니다: {e}")
        return None, None, None

    dates = pd.to_datetime(unique_dates)
    print(f"데이터 로딩 및 변환 완료. Shape: {prvm_data.shape}")
    print(f"날짜 범위: {dates.min().date()} ~ {dates.max().date()}")
    return prvm_data, dates, tickers

def load_gics_data(gics_file_path: str, tickers: List[str]) -> np.ndarray:
    """gicslist.csv 파일의 특정 형식에 맞게 GICS 데이터를 로드합니다."""
    print(f"'{gics_file_path}' 파일에서 GICS 데이터를 불러옵니다...")
    try:
        gics_df = pd.read_csv(gics_file_path, header=None)
        gics_sectors = gics_df.iloc[1:, 0].astype(str).to_numpy()
        if len(gics_sectors) != len(tickers):
            raise ValueError(f"GICS 데이터의 수({len(gics_sectors)})와 자산의 수({len(tickers)})가 일치하지 않습니다.")
        print("GICS 데이터 로딩 완료.")
        return gics_sectors
    except Exception as e:
        raise IOError(f"GICS 파일을 읽는 중 오류가 발생했습니다: {e}")

def project_psd(matrix: np.ndarray) -> np.ndarray:
    """행렬을 양의 준정부호(Positive Semi-Definite) 행렬로 변환합니다."""
    eigenvalues, eigenvectors = eigh(matrix)
    eigenvalues[eigenvalues < 1e-10] = 1e-10
    psd_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    return (psd_matrix + psd_matrix.T) / 2

# --- 2. 성능 평가 함수 ---

def calculate_mspe(forecasts: List[np.ndarray], ground_truths: List[np.ndarray]) -> float:
    """MSPE (Mean Squared Prediction Error)를 계산합니다."""
    if not forecasts: return np.nan
    errors = [norm(forecasts[i] - ground_truths[i], 'fro')**2 for i in range(len(forecasts))]
    return np.mean(errors)

def calculate_qlike(forecasts: List[np.ndarray], ground_truths: List[np.ndarray]) -> float:
    """QLIKE 손실 함수를 계산합니다."""
    if not forecasts: return np.nan
    qlike_vals = []
    for i in range(len(forecasts)):
        try:
            forecast_psd = project_psd(forecasts[i])
            forecast_reg = forecast_psd + np.eye(forecast_psd.shape[0]) * 1e-10
            sign, log_det_val = np.linalg.slogdet(forecast_reg)
            if sign <= 0: continue
            trace_val = np.trace(inv(forecast_reg) @ ground_truths[i])
            qlike_vals.append(log_det_val + trace_val)
        except np.linalg.LinAlgError:
            continue
    return np.mean(qlike_vals) if qlike_vals else np.nan


# --- 3. 변동성 예측 모델 구현 ---

def calculate_poet_prvm(prvm_matrix: np.ndarray, gics_sectors: np.ndarray, num_factors: int) -> np.ndarray:
    """단일 PRVM 행렬로부터 POET-PRVM을 계산합니다."""
    p = prvm_matrix.shape[0]
    prvm_psd = project_psd(prvm_matrix)
    
    eigenvalues, eigenvectors = eigh(prvm_psd)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:, idx]
    
    factor_part = np.zeros_like(prvm_psd)
    for i in range(num_factors):
        factor_part += eigenvalues[i] * np.outer(eigenvectors[:, i], eigenvectors[:, i])
        
    idiosyncratic_part_raw = prvm_psd - factor_part
    
    same_sector_mask = (gics_sectors.reshape(-1, 1) == gics_sectors)
    np.fill_diagonal(same_sector_mask, True)
    idiosyncratic_part_thresholded = idiosyncratic_part_raw * same_sector_mask
    
    poet_prvm = factor_part + idiosyncratic_part_thresholded
    return project_psd(poet_prvm)

def get_idio_matrices(prvm_data: np.ndarray, gics_sectors: np.ndarray, r: int) -> np.ndarray:
    """POET 기법을 적용하여 전체 기간에 대한 특이 변동성 행렬을 계산합니다."""
    print("POET 기법을 적용하여 특이 변동성 행렬을 생성합니다 (병렬 처리)...")
    p = prvm_data.shape[1]
    same_sector_mask = (gics_sectors[:, None] == gics_sectors)
    np.fill_diagonal(same_sector_mask, True)
    
    def process_day(prvm_matrix):
        prvm_psd = project_psd(prvm_matrix)
        eigenvalues, eigenvectors = eigh(prvm_psd)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:, idx]
        
        factor_part = np.zeros_like(prvm_psd)
        for i in range(r):
            factor_part += eigenvalues[i] * np.outer(eigenvectors[:, i], eigenvectors[:, i])
            
        idiosyncratic_part_raw = prvm_psd - factor_part
        return idiosyncratic_part_raw * same_sector_mask

    idio_matrices = Parallel(n_jobs=-1)(delayed(process_day)(prvm) for prvm in tqdm(prvm_data, desc="POET 적용 중"))
    return np.array(idio_matrices)

def fit_idio_component(args: Tuple) -> Tuple[int, HuberRegressor, float]:
    """[FIVARModel 헬퍼] 단일 특이 고유값에 대한 최적의 c_eta를 찾고 모델을 학습."""
    i, y_i, X, p, n_reg, c_eta_candidates, model_type = args
    best_bic_i, best_c_eta_i = np.inf, None

    if model_type in ['h-lasso', 'lasso']:
        for c_eta in c_eta_candidates:
            eta_I = c_eta * np.sqrt(np.log(p) / n_reg)
            
            if model_type == 'h-lasso':
                reg = HuberRegressor(fit_intercept=False, alpha=eta_I, epsilon=1.345)
            else:
                reg = Lasso(alpha=eta_I, fit_intercept=False, max_iter=2000)
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                reg.fit(X, y_i)

            y_pred = reg.predict(X)
            rss = np.sum((y_i - y_pred)**2)
            num_params = np.sum(np.abs(reg.coef_) > 1e-10)
            
            bic = n_reg * np.log(rss / n_reg) + num_params * np.log(n_reg) if rss > 0 else np.inf
            if bic < best_bic_i:
                best_bic_i, best_c_eta_i = bic, c_eta
        
        final_eta_I = best_c_eta_i * np.sqrt(np.log(p) / n_reg)
    else: # OLS
        final_eta_I = 0.0
        best_c_eta_i = 0.0

    if model_type == 'h-lasso':
        final_regressor = HuberRegressor(fit_intercept=False, alpha=final_eta_I)
    elif model_type == 'lasso':
        final_regressor = Lasso(alpha=final_eta_I, fit_intercept=False, max_iter=2000)
    else: # OLS
        final_regressor = LinearRegression(fit_intercept=False)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        final_regressor.fit(X, y_i)
    
    return i, final_regressor, best_c_eta_i

class FIVARModel:
    def __init__(self, r: int, model_type: str, h: int = 1, l: int = 22):
        self.r, self.h, self.l = r, h, l
        self.model_type = model_type.lower()
        self.p, self.tickers = None, None
        self.factor_eigenvectors, self.idio_eigenvectors = None, None
        self.factor_regressors, self.idio_regressors = None, None
        self.last_known_eigenvalues = None
        self.idio_mean_forecast = None
        self.scaler_X = None
        self.scalers_y = None

    def fit(self, prvm_matrices: Dict, idio_matrices: Dict, tickers: List[str]):
        self.tickers, self.p = tickers, len(tickers)
        all_dates = sorted(prvm_matrices.keys())
        
        if len(all_dates) < self.l:
             raise ValueError(f"고유벡터 계산에 필요한 데이터가 부족합니다 (필요: {self.l}, 보유: {len(all_dates)}).")

        last_l_dates = all_dates[-self.l:]
        avg_prvm_matrix = project_psd(np.mean([prvm_matrices[d] for d in last_l_dates], axis=0))
        _, factor_evecs_all = eigh(avg_prvm_matrix)
        self.factor_eigenvectors = factor_evecs_all[:, -self.r:][:, ::-1]

        avg_idio_matrix = project_psd(np.mean([idio_matrices[d] for d in last_l_dates], axis=0))
        _, idio_evecs_all = eigh(avg_idio_matrix)
        self.idio_eigenvectors = idio_evecs_all[:, ::-1]

        factor_evals = [np.diag(self.factor_eigenvectors.T @ prvm_matrices[d] @ self.factor_eigenvectors) / self.p for d in all_dates]
        idio_evals = [np.diag(self.idio_eigenvectors.T @ idio_matrices[d] @ self.idio_eigenvectors) for d in all_dates]
        
        eigenvalue_df = pd.DataFrame([np.concatenate(z) for z in zip(factor_evals, idio_evals)], index=all_dates)

        y_df = eigenvalue_df.iloc[self.h:]
        X_list = [eigenvalue_df.shift(k).iloc[self.h:] for k in range(1, self.h + 1)]
        X_df = pd.concat(X_list, axis=1)
        
        self.scaler_X = StandardScaler()
        self.scalers_y = [StandardScaler() for _ in range(y_df.shape[1])]
        
        X_scaled = self.scaler_X.fit_transform(X_df)
        y_scaled = np.zeros_like(y_df)
        for i in range(y_df.shape[1]):
            y_scaled[:, i] = self.scalers_y[i].fit_transform(y_df.iloc[:, i].values.reshape(-1, 1)).flatten()
        
        n_reg = len(y_df)
        
        self.factor_regressors = [None] * self.r
        X_factor_scaled = X_scaled[:, :self.r * self.h]
        for i in range(self.r):
            if self.model_type == 'h-lasso':
                self.factor_regressors[i] = HuberRegressor(fit_intercept=False, alpha=0.0)
            else:
                self.factor_regressors[i] = LinearRegression(fit_intercept=False)
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                self.factor_regressors[i].fit(X_factor_scaled, y_scaled[:, i])

        self.idio_regressors = [None] * self.p
        if self.model_type == 'ols':
            self.idio_mean_forecast = eigenvalue_df.iloc[-self.l:, self.r:].mean().to_numpy()
        else:
            c_eta_candidates = np.linspace(0.1, 10.0, 10)
            tasks = [(i, y_scaled[:, self.r + i].copy(), X_scaled, self.p, n_reg, c_eta_candidates, self.model_type) for i in range(self.p)]
            results = Parallel(n_jobs=-1)(delayed(fit_idio_component)(task) for task in tasks)
            for i, regressor, _ in results:
                self.idio_regressors[i] = regressor

        self.last_known_eigenvalues = eigenvalue_df.iloc[-self.h:].to_numpy()

    def predict(self) -> np.ndarray:
        if self.last_known_eigenvalues is None: raise RuntimeError("모델이 학습되지 않았습니다.")
        
        X_pred_raw = self.last_known_eigenvalues.flatten().reshape(1, -1)
        X_pred_scaled = self.scaler_X.transform(X_pred_raw)
        
        pred_evals_scaled = np.zeros(self.p + self.r)
        
        X_pred_factor_scaled = X_pred_scaled[:, :self.r * self.h]
        for i in range(self.r):
            pred_evals_scaled[i] = self.factor_regressors[i].predict(X_pred_factor_scaled)[0]
            
        if self.model_type == 'ols':
            pred_evals = np.zeros(self.p + self.r)
            pred_evals[self.r:] = self.idio_mean_forecast
        else:
            for i in range(self.p):
                pred_evals_scaled[self.r + i] = self.idio_regressors[i].predict(X_pred_scaled)[0]
        
        pred_evals = np.zeros(self.p + self.r)
        if self.model_type != 'ols':
            for i in range(self.p + self.r):
                pred_evals[i] = self.scalers_y[i].inverse_transform(pred_evals_scaled[i].reshape(1, -1))[0, 0]
        else:
             for i in range(self.r):
                pred_evals[i] = self.scalers_y[i].inverse_transform(pred_evals_scaled[i].reshape(1, -1))[0, 0]
        
        pred_evals[pred_evals < 1e-10] = 1e-10
        
        factor_forecast = self.p * (self.factor_eigenvectors @ np.diag(pred_evals[:self.r]) @ self.factor_eigenvectors.T)
        idio_forecast = self.idio_eigenvectors @ np.diag(pred_evals[self.r:]) @ self.idio_eigenvectors.T
        
        return project_psd(factor_forecast + idio_forecast)

print("✅ [셀 1] 초기 설정 완료: 모든 함수와 클래스가 로드되었습니다.")

#%% --- [셀 2] 파라미터 설정 ---
# 분석에 필요한 주요 파라미터들을 여기서 설정합니다.
# 이 값을 변경하면서 다른 조건으로 실험해볼 수 있습니다.

prvm_file_path = 'prvm_0731_final_corrected.csv' 
gics_file_path = 'gicslist.csv'
num_factors = 3
in_sample_size = 251
h_lag = 1
l_window = 22

# 분석 기간 정의
periods = {
    "Period 1 (2018-2019)": ('2018-01-01', '2019-12-31'),
    "Period 2 (2018)": ('2018-01-01', '2018-12-31'),
    "Period 3 (2019)": ('2019-01-01', '2019-12-31'),
}

# 분석할 모델 리스트
models = ['POET-PRVM', 'OLS', 'LASSO', 'H-LASSO']

print("✅ [셀 2] 파라미터 설정 완료.")


#%% --- [셀 3] 데이터 로딩 ---
# CSV 파일에서 PRVM 데이터와 GICS 데이터를 불러옵니다.
# 이 셀 실행 후, Variable Explorer에서 prvm_data_all, dates, tickers, gics_data 변수를 확인할 수 있습니다.

prvm_data_all, dates, tickers = load_csv_data(prvm_file_path)
if prvm_data_all is not None:
    gics_data = load_gics_data(gics_file_path, tickers)
    print("\n데이터 로딩이 성공적으로 완료되었습니다.")
    print("Variable Explorer에서 'prvm_data_all', 'dates', 'tickers', 'gics_data'를 확인하세요.")
else:
    print("데이터 로딩에 실패하여 이후 과정을 진행할 수 없습니다.")


#%% --- [셀 4] 특이 변동성 행렬 계산 (Idiosyncratic Matrices) ---
# 로드된 전체 PRVM 데이터에 POET 기법을 적용하여 특이 변동성 행렬을 미리 계산합니다.
# 이 과정은 다소 시간이 소요될 수 있습니다.
# 완료 후 'idio_data_all' 변수를 확인할 수 있습니다.

if 'prvm_data_all' in locals() and prvm_data_all is not None:
    idio_data_all = get_idio_matrices(prvm_data_all, gics_data, num_factors)
    print(f"✅ [셀 4] 특이 변동성 행렬 계산 완료. Shape: {idio_data_all.shape}")
    print("Variable Explorer에서 'idio_data_all'을 확인하세요.")
else:
    print("데이터가 로드되지 않아 [셀 4]를 실행할 수 없습니다. [셀 3]을 먼저 실행해주세요.")


#%% --- [셀 5] 롤링 윈도우 예측 실행 (메인 루프) ---
# ⚠️ 이 셀은 실행에 약 5시간이 소요될 수 있습니다. ⚠️
# 전체 기간에 대해 롤링 윈도우 방식으로 각 모델의 예측을 수행합니다.
# 실행이 완료되면 'forecasts', 'ground_truths', 'forecast_dates' 변수가 생성됩니다.

# 예측 결과를 저장할 변수 초기화
forecasts = {model: [] for model in models}
ground_truths, forecast_dates = [], []

if 'idio_data_all' in locals():
    start_idx = dates.searchsorted(pd.to_datetime('2018-01-01'))
    if start_idx < in_sample_size:
        start_idx = in_sample_size
        print(f"데이터가 충분하지 않아 {dates[start_idx].date()}부터 예측을 시작합니다.")

    # 메인 루프 실행
    for t in tqdm(range(start_idx, len(dates)), desc="전체 롤링 윈도우 예측"):
        train_start_idx = t - in_sample_size
        train_end_idx = t
        in_sample_prvm = prvm_data_all[train_start_idx:train_end_idx]
        in_sample_idio = idio_data_all[train_start_idx:train_end_idx]
        in_sample_dates = dates[train_start_idx:train_end_idx]
        
        prvm_train_dict = {d: m for d, m in zip(in_sample_dates, in_sample_prvm)}
        idio_train_dict = {d: m for d, m in zip(in_sample_dates, in_sample_idio)}
        
        ground_truth_matrix = prvm_data_all[t]
        ground_truths.append(calculate_poet_prvm(ground_truth_matrix, gics_data, num_factors))
        forecast_dates.append(dates[t])

        prvm_t_minus_1 = in_sample_prvm[-1]
        forecasts['POET-PRVM'].append(calculate_poet_prvm(prvm_t_minus_1, gics_data, num_factors))
        
        for model_type in ['OLS', 'LASSO', 'H-LASSO']:
            model = FIVARModel(r=num_factors, model_type=model_type, h=h_lag, l=l_window)
            model.fit(prvm_train_dict, idio_train_dict, tickers)
            prediction = model.predict()
            forecasts[model_type].append(prediction)
    
    print("✅ [셀 5] 롤링 윈도우 예측 완료!")
    print("Variable Explorer에서 'forecasts', 'ground_truths'를 확인하세요.")
else:
    print("[셀 4]가 실행되지 않아 예측을 진행할 수 없습니다.")


#%% --- [셀 6] 결과 집계 및 출력 ---
# [셀 5]에서 생성된 예측 결과를 바탕으로 성능(MSPE, QLIKE)을 계산하고 표로 정리합니다.
# 이 셀은 매우 빠르게 실행되므로, 표의 형식을 바꾸거나 다른 기간으로 재계산할 때 유용합니다.

if 'forecasts' in locals() and forecasts['POET-PRVM']:
    print("\n--- 최종 평가 결과 (논문 Table 2 형식) ---")
    final_results = []
    forecast_dates_pd = pd.to_datetime(forecast_dates)
    
    for period_name, (start_date, end_date) in periods.items():
        period_mask = (forecast_dates_pd >= start_date) & (forecast_dates_pd <= end_date)
        period_truths = [ground_truths[i] for i, val in enumerate(period_mask) if val]
        
        for model in models:
            period_forecasts = [forecasts[model][i] for i, val in enumerate(period_mask) if val]
            mspe_val = calculate_mspe(period_forecasts, period_truths)
            qlike_val = calculate_qlike(period_forecasts, period_truths)
            
            final_results.append({
                "Period": period_name, "Model": model,
                "MSPE (x10^4)": mspe_val * 10**4,
                "QLIKE x 10^-3": qlike_val * 10**-3,
            })

    results_df = pd.DataFrame(final_results).pivot_table(index='Model', columns='Period', values=['MSPE (x10^4)', 'QLIKE x 10^-3'])
    
    model_order = ['POET-PRVM', 'OLS', 'LASSO', 'H-LASSO']
    summary_table = results_df.swaplevel(0, 1, axis=1).sort_index(axis=1)
    
    print(summary_table.loc[model_order].to_string(float_format="%.3f"))
    print("\n✅ [셀 6] 결과 집계 완료!")
    print("Variable Explorer에서 'results_df', 'summary_table' 등을 확인하세요.")
else:
    print("[셀 5]가 실행되지 않아 결과를 집계할 수 없습니다.")