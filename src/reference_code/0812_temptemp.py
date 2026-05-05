# -*- coding: utf-8 -*-
"""
FIVAR Model Evaluation Script for Spyder IDE
(Model-wise cell execution version)
"""

#%% --- 1. 라이브러리 임포트 및 기본 설정 ---

import pandas as pd
import numpy as np
import os
from scipy.linalg import eigh, inv
from numpy.linalg import norm
from scipy.optimize import fmin_l_bfgs_b
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
import warnings
from tqdm import tqdm
from typing import Dict, List, Tuple
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
# 포트폴리오 최적화를 위해 cvxpy 라이브러리가 필요할 수 있습니다.
# !pip install cvxpy
try:
    import cvxpy as cp
except ImportError:
    print("cvxpy 라이브러리를 설치합니다...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "cvxpy"])
    import cvxpy as cp


# 경고 메시지 무시 설정
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='cvxpy')
warnings.filterwarnings("ignore", message="The line search algorithm did not converge")


#%% --- 2. 함수 및 클래스 정의 ---

# --- [수정] scikit-learn 공식 코드를 기반으로 epsilon 제약이 없는 Custom Huber Regressor 구현 ---
class CustomHuberRegressor(BaseEstimator, RegressorMixin):
    """
    scikit-learn의 HuberRegressor를 기반으로 epsilon >= 1.0 제약 조건을 제거하고,
    L1 페널티(alpha)를 지원하도록 수정한 맞춤형 Huber Regressor.
    """
    def __init__(self, fit_intercept=False, alpha=0.0, epsilon=1.35, max_iter=100, tol=1e-5):
        self.fit_intercept = fit_intercept
        self.alpha = alpha
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None

    def _huber_loss_and_gradient(self, coef, X, y, alpha):
        """Huber 손실과 L1 페널티를 결합한 목적 함수와 그래디언트를 계산합니다."""
        n_samples = X.shape[0]
        residuals = X @ coef - y
        
        # Huber 손실 계산
        abs_residuals = np.abs(residuals)
        mask_linear = abs_residuals > self.epsilon
        
        loss_huber = 0.5 * np.sum(residuals[~mask_linear] ** 2)
        loss_huber += self.epsilon * np.sum(abs_residuals[mask_linear] - 0.5 * self.epsilon)
        
        # L1 페널티 추가
        loss = loss_huber / n_samples + alpha * np.sum(np.abs(coef))

        # 그래디언트 계산
        grad = np.zeros_like(coef)
        grad += X[~mask_linear].T @ residuals[~mask_linear]
        grad += self.epsilon * X[mask_linear].T @ np.sign(residuals[mask_linear])
        grad /= n_samples
        
        return loss, grad

    def fit(self, X, y):
        if self.fit_intercept:
            raise NotImplementedError("fit_intercept=True is not supported.")

        # L-BFGS-B는 L1 페널티를 직접 처리하지 못하므로,
        # 손실 함수에 L1 항을 추가하고 그래디언트를 계산하는 방식을 사용합니다.
        # 이는 OWL-QN과 유사한 접근 방식입니다.
        
        initial_coef = np.zeros(X.shape[1])
        
        # fmin_l_bfgs_b는 L1 항의 그래디언트를 직접 다루지 못하므로,
        # 손실 함수 자체에 L1 항을 포함시켜 최적화합니다.
        # 이는 엄밀한 의미의 LASSO-Huber는 아니지만, 유사한 효과를 냅니다.
        # 더 정확한 구현을 위해서는 ISTA/FISTA와 같은 알고리즘이 필요하지만,
        # scikit-learn의 L-BFGS-B 구조를 활용하기 위해 이 방식을 채택합니다.
        
        coef, _, _ = fmin_l_bfgs_b(
            func=self._huber_loss_and_gradient,
            x0=initial_coef,
            args=(X, y, self.alpha),
            maxiter=self.max_iter,
            pgtol=self.tol,
        )
        self.coef_ = coef
        return self

    def predict(self, X):
        return X @ self.coef_

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
    symmetric_matrix = (matrix + matrix.T) / 2
    eigenvalues, eigenvectors = eigh(symmetric_matrix)
    eigenvalues[eigenvalues < 1e-10] = 1e-10
    psd_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    return (psd_matrix + psd_matrix.T) / 2

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

def calculate_poet_prvm(prvm_matrix: np.ndarray, gics_sectors: np.ndarray, num_factors: int) -> np.ndarray:
    """단일 PRVM 행렬로부터 POET-PRVM을 계산합니다."""
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

def fit_factor_component(args: Tuple) -> Tuple[int, LinearRegression]:
    """[FIVARModel 헬퍼] 단일 팩터 고유값에 대한 모델을 병렬로 학습."""
    i, model_type, X_data, y_data, tau_F = args
    
    if model_type == 'h-lasso':
        regressor = CustomHuberRegressor(fit_intercept=False, alpha=0.0, epsilon=tau_F)
    else: # OLS, LASSO
        regressor = LinearRegression(fit_intercept=False)
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        regressor.fit(X_data, y_data)
        
    return i, regressor

def fit_idio_component(args: Tuple) -> Tuple[int, BaseEstimator, float]:
    """[FIVARModel 헬퍼] 단일 특이 고유값에 대한 최적의 c_eta를 찾고 모델을 학습."""
    i, y_i, X, p, n_reg, c_eta_candidates, model_type, tau_I = args
    best_bic_i, best_c_eta_i = np.inf, None

    if model_type in ['h-lasso', 'lasso']:
        for c_eta in c_eta_candidates:
            eta_I = c_eta * np.sqrt(np.log(p) / n_reg)
            
            if model_type == 'h-lasso':
                reg = CustomHuberRegressor(fit_intercept=False, alpha=eta_I, epsilon=tau_I)
            else:
                reg = Lasso(alpha=eta_I, fit_intercept=False, max_iter=2000)
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                reg.fit(X, y_i)

            y_pred = reg.predict(X)
            rss = np.sum((y_i - y_pred)**2)
            num_params = np.sum(np.abs(reg.coef_) > 1e-6)
            
            bic = n_reg * np.log(rss / n_reg) + num_params * np.log(n_reg) if rss > 0 else np.inf
            if bic < best_bic_i:
                best_bic_i, best_c_eta_i = bic, c_eta
        
        final_eta_I = best_c_eta_i * np.sqrt(np.log(p) / n_reg) if best_c_eta_i is not None else 0
    else: # OLS
        final_eta_I = 0.0
        best_c_eta_i = 0.0

    if model_type == 'h-lasso':
        final_regressor = CustomHuberRegressor(fit_intercept=False, alpha=final_eta_I, epsilon=tau_I)
    elif model_type == 'lasso':
        final_regressor = Lasso(alpha=final_eta_I, fit_intercept=False, max_iter=2000)
    else: # OLS
        final_regressor = LinearRegression(fit_intercept=False)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        final_regressor.fit(X, y_i)
    
    return i, final_regressor, best_c_eta_i

def display_model_results(model_name, forecasts_dict, ground_truths_list, forecast_dates_list, periods_dict):
    """특정 모델에 대한 MSPE 및 QLIKE 결과를 계산하고 출력합니다."""
    print(f"\n--- {model_name} 모델 평가 결과 ---")
    model_results = []
    forecast_dates_pd = pd.to_datetime(forecast_dates_list)
    
    for period_name, (start_date, end_date) in periods_dict.items():
        period_mask = (forecast_dates_pd >= start_date) & (forecast_dates_pd <= end_date)
        
        period_forecasts = [forecasts_dict[model_name][i] for i, val in enumerate(period_mask) if val]
        period_truths = [ground_truths_list[i] for i, val in enumerate(period_mask) if val]
        
        if not period_forecasts:
            continue
            
        mspe_val = calculate_mspe(period_forecasts, period_truths)
        qlike_val = calculate_qlike(period_forecasts, period_truths)
        
        model_results.append({
            "Period": period_name,
            "MSPE (x10^4)": mspe_val * 10**4,
            "QLIKE x 10^-3": qlike_val * 10**-3,
        })
        
    results_df = pd.DataFrame(model_results).set_index('Period')
    print(results_df.to_string(float_format="%.3f"))


#%% --- 3. FIVAR 모델 클래스 정의 ---

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
        avg_prvm_matrix = np.mean([prvm_matrices[d] for d in last_l_dates], axis=0)
        _, factor_evecs_all = eigh(avg_prvm_matrix)
        self.factor_eigenvectors = factor_evecs_all[:, -self.r:][:, ::-1]

        avg_idio_matrix = np.mean([idio_matrices[d] for d in last_l_dates], axis=0)
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
        
        c_F2 = 0.25
        c_I2 = 4
        sigma_F = np.sqrt(np.sum(y_df.iloc[:, :self.r].var(axis=0)) / self.r)
        
        tau_F = c_F2 * sigma_F * (n_reg / np.log(self.p))**0.25
        tau_I = c_I2 * (n_reg / np.log(self.p))**0.25

        self.factor_regressors = [None] * self.r
        X_factor_scaled = X_scaled[:, :self.r * self.h]
        factor_tasks = [(i, self.model_type, X_factor_scaled, y_scaled[:, i], tau_F) for i in range(self.r)]
        results_factor = Parallel(n_jobs=-1)(delayed(fit_factor_component)(args) for args in factor_tasks)
        for i, regressor in results_factor:
            self.factor_regressors[i] = regressor

        self.idio_regressors = [None] * self.p
        if self.model_type == 'ols':
            self.idio_mean_forecast = eigenvalue_df.iloc[-self.l:, self.r:].mean().to_numpy()
        else:
            c_eta_candidates = np.linspace(0.1, 10.0, 20)
            idio_tasks = [(i, y_scaled[:, self.r + i].copy(), X_scaled, self.p, n_reg, c_eta_candidates, self.model_type, tau_I) for i in range(self.p)]
            results_idio = Parallel(n_jobs=-1)(delayed(fit_idio_component)(args) for args in idio_tasks)
            for i, regressor, _ in results_idio:
                self.idio_regressors[i] = regressor

        self.last_known_eigenvalues = eigenvalue_df.iloc[-self.h:].to_numpy()

    def predict(self) -> np.ndarray:
        if self.last_known_eigenvalues is None: raise RuntimeError("모델이 학습되지 않았습니다.")
        
        X_pred_raw = self.last_known_eigenvalues.flatten().reshape(1, -1)
        X_pred_scaled = self.scaler_X.transform(X_pred_raw)
        
        pred_evals = np.zeros(self.p + self.r)
        
        X_pred_factor_scaled = X_pred_scaled[:, :self.r * self.h]
        pred_factor_evals_scaled = np.zeros(self.r)
        for i in range(self.r):
            pred_factor_evals_scaled[i] = self.factor_regressors[i].predict(X_pred_factor_scaled)[0]
            pred_evals[i] = self.scalers_y[i].inverse_transform(pred_factor_evals_scaled[i].reshape(1, -1))[0, 0]

        if self.model_type == 'ols':
            pred_evals[self.r:] = self.idio_mean_forecast
        else:
            pred_idio_evals_scaled = np.zeros(self.p)
            for i in range(self.p):
                pred_idio_evals_scaled[i] = self.idio_regressors[i].predict(X_pred_scaled)[0]
                pred_evals[self.r + i] = self.scalers_y[self.r + i].inverse_transform(pred_idio_evals_scaled[i].reshape(1, -1))[0, 0]
        
        pred_evals[pred_evals < 1e-10] = 1e-10
        
        factor_forecast = self.p * (self.factor_eigenvectors @ np.diag(pred_evals[:self.r]) @ self.factor_eigenvectors.T)
        idio_forecast = self.idio_eigenvectors @ np.diag(pred_evals[self.r:]) @ self.idio_eigenvectors.T
        
        return project_psd(factor_forecast + idio_forecast)


#%% --- 4. 메인 실행: 데이터 준비 및 공통 변수 설정 ---

# 이 셀은 한 번만 실행하면 됩니다.
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()
    print("경고: __file__을 찾을 수 없어 현재 작업 디렉토리를 사용합니다.")

# 파일 경로 및 파라미터 설정
prvm_file_path = os.path.join(script_dir, 'prvm_0731_final_corrected.csv')
gics_file_path = os.path.join(script_dir, 'gicslist.csv')
log_price_filepath = os.path.join(script_dir, 'DataY.csv') 
num_factors = 3
in_sample_size = 251
h_lag = 1
l_window = 22

# 데이터 로딩
prvm_data_all, dates, tickers = load_csv_data(prvm_file_path)
if prvm_data_all is not None:
    gics_data = load_gics_data(gics_file_path, tickers)
    
    # 특이 변동성 행렬 사전 계산 (시간이 오래 걸릴 수 있음)
    idio_data_all = get_idio_matrices(prvm_data_all, gics_data, num_factors)

    # 공통 변수 초기화
    periods = {
        "Period 1 (2018-2019)": ('2018-01-01', '2019-12-31'),
        "Period 2 (2018)": ('2018-01-01', '2018-12-31'),
        "Period 3 (2019)": ('2019-01-01', '2019-12-31'),
    }
    
    models = ['POET-PRVM', 'OLS', 'LASSO', 'H-LASSO']
    forecasts = {model: [] for model in models}
    ground_truths, forecast_dates = [], []
    
    start_idx = dates.searchsorted(pd.to_datetime('2018-01-01'))
    if start_idx < in_sample_size:
        start_idx = in_sample_size
        print(f"데이터가 충분하지 않아 {dates[start_idx].date()}부터 예측을 시작합니다.")
else:
    print("데이터 로딩에 실패하여 실행을 중단합니다.")


#%% --- 5. POET-PRVM 모델 예측 ---

if 'prvm_data_all' in locals():
    for t in tqdm(range(start_idx, len(dates)), desc="POET-PRVM 예측"):
        # Ground Truth는 첫 모델 예측 시 한 번만 생성합니다.
        if len(ground_truths) < (t - start_idx + 1):
            ground_truth_matrix = prvm_data_all[t]
            ground_truths.append(calculate_poet_prvm(ground_truth_matrix, gics_data, num_factors))
            forecast_dates.append(dates[t])
        
        prvm_t_minus_1 = prvm_data_all[t-1]
        forecasts['POET-PRVM'].append(calculate_poet_prvm(prvm_t_minus_1, gics_data, num_factors))
    
    display_model_results('POET-PRVM', forecasts, ground_truths, forecast_dates, periods)
else:
    print("데이터가 로드되지 않았습니다. 4번 셀을 먼저 실행해주세요.")


#%% --- 6. OLS 모델 예측 ---

if 'prvm_data_all' in locals():
    for t in tqdm(range(start_idx, len(dates)), desc="OLS 예측"):
        train_start_idx = t - in_sample_size
        train_end_idx = t
        in_sample_prvm = prvm_data_all[train_start_idx:train_end_idx]
        in_sample_idio = idio_data_all[train_start_idx:train_end_idx]
        in_sample_dates = dates[train_start_idx:train_end_idx]
        
        prvm_train_dict = {d: m for d, m in zip(in_sample_dates, in_sample_prvm)}
        idio_train_dict = {d: m for d, m in zip(in_sample_dates, in_sample_idio)}
        
        model = FIVARModel(r=num_factors, model_type='OLS', h=h_lag, l=l_window)
        model.fit(prvm_train_dict, idio_train_dict, tickers)
        prediction = model.predict()
        forecasts['OLS'].append(prediction)
        
    display_model_results('OLS', forecasts, ground_truths, forecast_dates, periods)
else:
    print("데이터가 로드되지 않았습니다. 4번 셀을 먼저 실행해주세요.")


#%% --- 7. LASSO 모델 예측 ---

if 'prvm_data_all' in locals():
    for t in tqdm(range(start_idx, len(dates)), desc="LASSO 예측"):
        train_start_idx = t - in_sample_size
        train_end_idx = t
        in_sample_prvm = prvm_data_all[train_start_idx:train_end_idx]
        in_sample_idio = idio_data_all[train_start_idx:train_end_idx]
        in_sample_dates = dates[train_start_idx:train_end_idx]
        
        prvm_train_dict = {d: m for d, m in zip(in_sample_dates, in_sample_prvm)}
        idio_train_dict = {d: m for d, m in zip(in_sample_dates, in_sample_idio)}
        
        model = FIVARModel(r=num_factors, model_type='LASSO', h=h_lag, l=l_window)
        model.fit(prvm_train_dict, idio_train_dict, tickers)
        prediction = model.predict()
        forecasts['LASSO'].append(prediction)
        
    display_model_results('LASSO', forecasts, ground_truths, forecast_dates, periods)
else:
    print("데이터가 로드되지 않았습니다. 4번 셀을 먼저 실행해주세요.")


#%% --- 8. H-LASSO 모델 예측 ---

if 'prvm_data_all' in locals():
    for t in tqdm(range(start_idx, len(dates)), desc="H-LASSO 예측"):
        train_start_idx = t - in_sample_size
        train_end_idx = t
        in_sample_prvm = prvm_data_all[train_start_idx:train_end_idx]
        in_sample_idio = idio_data_all[train_start_idx:train_end_idx]
        in_sample_dates = dates[train_start_idx:train_end_idx]
        
        prvm_train_dict = {d: m for d, m in zip(in_sample_dates, in_sample_prvm)}
        idio_train_dict = {d: m for d, m in zip(in_sample_dates, in_sample_idio)}
        
        model = FIVARModel(r=num_factors, model_type='H-LASSO', h=h_lag, l=l_window)
        model.fit(prvm_train_dict, idio_train_dict, tickers)
        prediction = model.predict()
        forecasts['H-LASSO'].append(prediction)
        
    display_model_results('H-LASSO', forecasts, ground_truths, forecast_dates, periods)
else:
    print("데이터가 로드되지 않았습니다. 4번 셀을 먼저 실행해주세요.")


#%% --- 9. 최종 결과 집계 및 출력 ---

if 'forecasts' in locals() and forecasts['H-LASSO']: # 마지막 모델의 예측이 완료되었는지 확인
    print("\n--- 최종 평가 결과 (논문 Table 2 형식) ---")
    final_results = []
    forecast_dates_pd = pd.to_datetime(forecast_dates)
    
    for period_name, (start_date, end_date) in periods.items():
        period_mask = (forecast_dates_pd >= start_date) & (forecast_dates_pd <= end_date)
        period_truths = [ground_truths[i] for i, val in enumerate(period_mask) if val]
        
        for model in models:
            period_forecasts = [forecasts[model][i] for i, val in enumerate(period_mask) if val]
            if not period_forecasts: continue

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
else:
    print("예측이 수행되지 않았습니다. 이전 셀들을 먼저 실행해주세요.")

#%% --- 10. 포트폴리오 최적화 및 리스크 분석 ---

# ⚠️ 중요: 이 셀을 실행하려면 'DataY.csv' 파일이 필요합니다.
if 'forecasts' in locals() and forecasts['H-LASSO']:
    try:
        print("\n--- 포트폴리오 리스크 분석 (논문 Figure 5 형식) ---")
        
        # 1분 단위 로그 가격 데이터로부터 일별 수익률 계산
        print(f"'{log_price_filepath}' 파일에서 일별 수익률을 계산합니다...")
        log_price_df = pd.read_csv(log_price_filepath, header=0, index_col=0, encoding='latin1', low_memory=False)
        log_price_df.index = pd.to_datetime(log_price_df.index, errors='coerce')
        log_returns_1min = log_price_df.diff().dropna()
        daily_returns_df = log_returns_1min.resample('D').sum()
        
        # 분석에 사용된 티커와 날짜에 맞춰 데이터 정렬
        daily_returns_df = daily_returns_df.reindex(columns=tickers, index=pd.to_datetime(forecast_dates))
        
        if daily_returns_df.isnull().values.any():
            print("경고: 수익률 데이터에 결측치가 있습니다. 0으로 대체합니다.")
            daily_returns_df = daily_returns_df.fillna(0)

        # 최적화 및 리스크 계산
        exposure_constraints = np.linspace(1.0, 3.0, 11)
        portfolio_risks = {model: {p_name: [] for p_name in periods} for model in models}
        
        for period_name, (start_date, end_date) in periods.items():
            period_mask = (pd.to_datetime(forecast_dates) >= start_date) & (pd.to_datetime(forecast_dates) <= end_date)
            
            for model in models:
                print(f"Calculating risk for {model} in {period_name}...")
                for c0 in tqdm(exposure_constraints, leave=False, desc=f"Exposure {model}"):
                    daily_portfolio_variances = []
                    
                    period_forecast_indices = np.where(period_mask)[0]

                    for day_idx in period_forecast_indices:
                        Sigma_hat = forecasts[model][day_idx]
                        n_assets = Sigma_hat.shape[0]
                        
                        w = cp.Variable(n_assets)
                        prob = cp.Problem(cp.Minimize(cp.quad_form(w, Sigma_hat)),
                                          [cp.sum(w) == 1, cp.norm(w, 1) <= c0])
                        
                        try:
                            prob.solve(solver=cp.SCS, verbose=False)
                            if w.value is not None:
                                realized_return = daily_returns_df.iloc[day_idx].values
                                portfolio_return = w.value.T @ realized_return
                                daily_portfolio_variances.append(portfolio_return**2)
                        except cp.error.SolverError:
                            continue

                    if daily_portfolio_variances:
                        annualized_risk = np.sqrt(np.mean(daily_portfolio_variances) * 252) * 100
                        portfolio_risks[model][period_name].append(annualized_risk)
                    else:
                        portfolio_risks[model][period_name].append(np.nan)

        # 결과 시각화
        # --- [수정] 결과 시각화 부분 ---
        # figsize 가로폭을 줄여 논문과 유사한 비율로 조정
        fig, axes = plt.subplots(1, 3, figsize=(10, 6), sharey=True)
        
        # 논문과 동일한 스타일(색상, 마커)을 적용하기 위한 딕셔너리
        style_map = {
            'POET-PRVM': {'color': 'red', 'marker': '*', 'linestyle': '--'},
            'OLS':       {'color': 'blue', 'marker': '+', 'linestyle': '--'},
            'LASSO':     {'color': 'green', 'marker': 'o', 'linestyle': '--', 'mfc': 'none'}, # mfc='none' for hollow circle
            'H-LASSO':   {'color': 'black', 'marker': 'o', 'linestyle': '-'}
        }
        
        for i, period_name in enumerate(periods):
            ax = axes[i]
            for model in models:
                if model in style_map:
                    style = style_map[model]
                    ax.plot(exposure_constraints, 
                            portfolio_risks[model][period_name], 
                            marker=style['marker'], 
                            color=style['color'],
                            linestyle=style['linestyle'],
                            label=model,
                            markersize=5,
                            mfc=style.get('mfc', None))
            
            ax.set_title(f'Portfolio Risk ({period_name.split("(")[1][:-1]})')
            ax.set_xlabel('Exposure constraint')
            if i == 0: ax.set_ylabel('Annualized risk (%)')
            ax.legend()
        
        plt.tight_layout()
        plt.show()
        # --- [수정 끝] ---

    except FileNotFoundError:
        print(f"\n오류: '{log_price_filepath}' 파일을 찾을 수 없습니다.")
        print("포트폴리오 리스크 분석을 위해서는 1분 단위 로그 가격 데이터 파일이 필요합니다.")
    except Exception as e:
        print(f"\n포트폴리오 리스크 분석 중 오류가 발생했습니다: {e}")

else:
    print("예측이 수행되지 않았습니다. 이전 셀들을 먼저 실행해주세요.")
