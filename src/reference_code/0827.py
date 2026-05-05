# -*- coding: utf-8 -*-
"""
FIVAR Model Implementation and Evaluation (Corrected Truncation Logic)

이 스크립트는 "Factor and Idiosyncratic VAR Volatility Matrix Models for
Heavy-Tailed High-Frequency Financial Observations" 논문에 제시된 FIVAR 모델을
구현하고 평가합니다.

주요 기능:
1.  고빈도 로그 가격(DataY.csv) 데이터로부터 직접 Jump-Adjusted PRVM을 계산하는 기능 내장.
2.  4가지 Estimator 구현 (POET-PRVM, OLS, LASSO, H-LASSO).
3.  롤링 윈도우(Rolling Window) 방식을 사용한 시계열 예측 수행.
4.  MSPE, QLIKE 지표를 사용한 모델 성능 평가 (논문 Table 2 재현).
5.  포트폴리오 최적화를 통한 리스크 분석 및 시각화 (논문 Figure 5 재현).

주요 수정 사항 (v10):
- [핵심 수정] 논문의 방법론을 엄밀하게 따르도록 Truncation 전처리 로직을 수정했습니다.
  이제 Truncation은 H-LASSO 모델에만 적용되며, OLS와 LASSO 모델은 원본 데이터를
  그대로 사용합니다. 이것이 변동성이 높은 기간에 MSPE가 낮게 나오던 문제의
  직접적인 원인이었습니다.
"""

# %% --- 1. 라이브러리 임포트 및 기본 설정 ---

import pandas as pd
import numpy as np
import os
import warnings
from tqdm import tqdm
from typing import Dict, List, Tuple, Union
from math import floor
from numpy.lib.stride_tricks import as_strided


from scipy.linalg import eigh, inv
from scipy.optimize import fmin_l_bfgs_b
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning


import matplotlib.pyplot as plt
import seaborn as sns

try:
    import cvxpy as cp
except ImportError:
    print("cvxpy 라이브러리가 설치되어 있지 않습니다. 설치를 시작합니다...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "cvxpy"])
    import cvxpy as cp

sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore")


# %% --- 2. 유틸리티 함수 및 커스텀 클래스 정의 ---

def project_psd(matrix: np.ndarray) -> np.ndarray:
    """주어진 행렬을 가장 가까운 양의 준정부호(Positive Semi-Definite, PSD) 행렬로 변환합니다."""
    symmetric_matrix = (matrix + matrix.T) / 2
    eigenvalues, eigenvectors = eigh(symmetric_matrix)
    eigenvalues[eigenvalues < 1e-10] = 1e-10
    psd_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    return (psd_matrix + psd_matrix.T) / 2

def truncate_data(data: np.ndarray, omega: float) -> np.ndarray:
    """데이터를 특정 임계값(omega)으로 잘라내는 Truncation(Winsorization)을 수행합니다."""
    return np.clip(data, -omega, omega)

class CustomHuberRegressor(BaseEstimator, RegressorMixin):
    """논문의 H-LASSO 모델 구현을 위한 커스텀 Huber 회귀 모델."""
    def __init__(self, alpha: float = 0.0, epsilon: float = 1.35, max_iter: int = 100, tol: float = 1e-5):
        self.alpha = alpha
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None

    def _objective_function(self, coef: np.ndarray, X: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
        """Huber 손실과 L1 페널티를 결합한 목적 함수와 그래디언트를 계산합니다."""
        n_samples = X.shape[0]
        residuals = X @ coef - y
        abs_residuals = np.abs(residuals)
        
        mask_linear = abs_residuals > self.epsilon
        loss_huber = 0.5 * np.sum(residuals[~mask_linear] ** 2)
        loss_huber += self.epsilon * np.sum(abs_residuals[mask_linear] - 0.5 * self.epsilon)
        
        loss = loss_huber / n_samples + self.alpha * np.sum(np.abs(coef))

        grad = np.zeros_like(coef)
        grad += X[~mask_linear].T @ residuals[~mask_linear]
        grad += self.epsilon * X[mask_linear].T @ np.sign(residuals[mask_linear])
        grad /= n_samples
        
        return loss, grad

    def fit(self, X: np.ndarray, y: np.ndarray):
        initial_coef = np.zeros(X.shape[1])
        coef, _, _ = fmin_l_bfgs_b(
            func=self._objective_function, x0=initial_coef, args=(X, y),
            maxiter=self.max_iter, pgtol=self.tol,
        )
        self.coef_ = coef
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_


# %% --- 3. 데이터 로더 및 전처리기 ---

class DataLoader:
    """데이터 로딩 및 전처리를 담당하는 클래스."""
    def __init__(self, gics_path: str, log_price_path: str):
        self.gics_path = gics_path
        self.log_price_path = log_price_path

    def load_all_data(self) -> Dict[str, Union[np.ndarray, pd.DataFrame, pd.DatetimeIndex, List[str]]]:
        """모든 필요한 데이터를 로드하고 기본 전처리를 수행합니다."""
        print("데이터 로딩 및 전처리를 시작합니다...")
        
        log_price_data = self._load_and_clean_log_prices()
        tickers = log_price_data.columns.tolist()
        
        prvm_data, dates = self._calculate_prvm_from_log_prices(log_price_data)
        
        print("-> 모든 PRVM 행렬을 PSD로 변환합니다...")
        prvm_data = np.array([project_psd(m) for m in tqdm(prvm_data, desc="   PSD 변환 중")])

        gics_data = self._load_gics_data(tickers)
        jump_vol_data = self._calculate_jump_volatility(prvm_data, dates, log_price_data)
        
        print("\n모든 데이터 로딩 및 전처리가 완료되었습니다.")
        return {
            "prvm_data": prvm_data,
            "dates": dates,
            "tickers": tickers,
            "gics_data": gics_data,
            "jump_vol_data": jump_vol_data,
            "log_price_data": log_price_data
        }

    def _load_and_clean_log_prices(self) -> pd.DataFrame:
        """1분 단위 로그 가격 데이터를 로드하고 클리닝합니다."""
        print(f"-> 1분 단위 로그 가격 데이터 로딩 및 클리닝: '{os.path.basename(self.log_price_path)}'")
        df = pd.read_csv(self.log_price_path, header=0, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, errors='coerce')
        df = df.dropna(how='all', axis=0)
        df = df.loc[:, df.notna().any(axis=0)]
        
        for ticker in tqdm(df.columns, desc="   결측치 처리"):
            df[ticker] = pd.to_numeric(df[ticker], errors='coerce')
            df[ticker].fillna(method='ffill', inplace=True)
            df[ticker].fillna(method='bfill', inplace=True)
        
        daily_counts = df.groupby(df.index.date).size()
        full_trading_dates = daily_counts[daily_counts >= 390].index
        df = df[df.index.to_series().dt.date.isin(full_trading_dates)]
        
        return df

    def _calculate_prvm_from_log_prices(self, log_price_data: pd.DataFrame) -> Tuple[np.ndarray, pd.DatetimeIndex]:
        """로그 가격 데이터로부터 Jump-Adjusted PRVM을 병렬로 계산합니다."""
        print("-> 로그 가격으로부터 Jump-Adjusted PRVM을 계산합니다 (병렬 처리)...")
        log_returns = log_price_data.diff().dropna()
        daily_groups = log_returns.groupby(log_returns.index.date)
        
        tasks = [(date, group.values) for date, group in daily_groups]

        results = Parallel(n_jobs=-1)(
            delayed(self._prvm_day_calculator)(task) for task in tqdm(tasks, desc="   일별 PRVM 계산")
        )

        valid_results = [res for res in results if res is not None]
        dates = pd.to_datetime([res[0] for res in valid_results])
        prvm_matrices = np.array([res[1] for res in valid_results])

        return prvm_matrices, dates

    @staticmethod
    def _prvm_day_calculator(task: Tuple) -> Union[Tuple, None]:
        """단일 거래일에 대한 PRVM을 계산하는 최적화된 헬퍼 함수."""
        date, daily_log_returns = task
        num_tickers = daily_log_returns.shape[1]

        def g(x): return np.minimum(x, 1 - x)

        m = len(daily_log_returns)
        if m < 4: return None
        K = 19
        if K < 2: return None

        shape_bar = (m - K + 1, K - 1, num_tickers)
        strides_bar = (daily_log_returns.strides[0], daily_log_returns.strides[0], daily_log_returns.strides[1])
        Y_window_bar = as_strided(daily_log_returns[:-1], shape=shape_bar, strides=strides_bar)
        weights_y_bar = g(np.arange(1, K) / K)
        Y_bar_matrix = np.einsum('j,kji->ki', weights_y_bar, Y_window_bar)

        scaled_Y_bar_sq = np.square(m**(1/4) * Y_bar_matrix)
        std_devs = np.sqrt(np.sum(scaled_Y_bar_sq, axis=0) / (m - K))
        u_thresholds = (4.0 * std_devs) * (m**(-0.235))

        shape_hat = (m - K + 1, K, num_tickers)
        strides_hat = (daily_log_returns.strides[0], daily_log_returns.strides[0], daily_log_returns.strides[1])
        Y_window_hat = as_strided(daily_log_returns, shape=shape_hat, strides=strides_hat)
        weights_y_hat_sq = (g(np.arange(1, K + 1) / K) - g(np.arange(0, K) / K))**2
        Y_hat_tensor = np.einsum('l,kli,klj->kij', weights_y_hat_sq, Y_window_hat, Y_window_hat)

        is_not_jump = np.abs(Y_bar_matrix) <= u_thresholds
        valid_mask = np.einsum('ki,kj->kij', is_not_jump, is_not_jump)
        y_outer = np.einsum('ki,kj->kij', Y_bar_matrix, Y_bar_matrix)
        terms_to_add = (y_outer - 0.5 * Y_hat_tensor) * valid_mask
        prvm_sum = np.sum(terms_to_add, axis=0)

        psi = 1/12 
        final_prvm = (1 / (psi * K)) * prvm_sum
        np.fill_diagonal(final_prvm, np.maximum(np.diag(final_prvm), 0))
        
        return date, final_prvm

    def _load_gics_data(self, tickers: List[str]) -> np.ndarray:
        """GICS 섹터 정보를 로드합니다."""
        print(f"-> GICS 데이터 로딩: '{os.path.basename(self.gics_path)}'")
        gics_df = pd.read_csv(self.gics_path, header=None)
        gics_sectors = gics_df.iloc[1:, 0].astype(str).to_numpy()
        if len(gics_sectors) != len(tickers):
            raise ValueError("GICS 데이터와 자산 수가 일치하지 않습니다.")
        return gics_sectors

    def _calculate_jump_volatility(self, prvm_continuous: np.ndarray, dates: pd.DatetimeIndex, log_price_data: pd.DataFrame) -> np.ndarray:
        """점프 변동성(총 변동성 - 연속 변동성)을 추정하고 PSD로 변환합니다."""
        print(f"-> 점프 변동성 계산...")
        daily_groups = log_price_data.groupby(log_price_data.index.date)
        
        jump_vol_all = np.zeros_like(prvm_continuous)
        m = 390
        K = 19

        for t, date in enumerate(tqdm(dates, desc="   점프 변동성 계산")):
            if date.date() in daily_groups.groups:
                daily_log_prices = daily_groups.get_group(date.date()).values.T
                prvm_total = self._prvm_total_from_log_prices(daily_log_prices, K)
                
                jump_vol_raw = prvm_total - prvm_continuous[t]
                jump_vol_all[t] = project_psd(jump_vol_raw)
        
        return jump_vol_all

    @staticmethod
    def _prvm_total_from_log_prices(daily_log_prices: np.ndarray, K: int) -> np.ndarray:
        """고빈도 로그 가격으로부터 총 PRVM을 계산합니다."""
        p, m = daily_log_prices.shape
        daily_returns = np.diff(daily_log_prices, axis=1)
        
        g = np.array([min(i, K - i) / K for i in range(1, K)])
        psi = np.sum(g**2)

        pre_averaged_returns = np.zeros((p, m - K + 1))
        for k in range(m - K + 1):
            pre_averaged_returns[:, k] = np.sum(daily_returns[:, k:k+K-1] * g, axis=1)

        return (1 / (psi * K)) * (pre_averaged_returns @ pre_averaged_returns.T)

class Preprocessor:
    """POET 기법을 적용하여 특이/전체 변동성 행렬을 계산하는 클래스."""
    def __init__(self, num_factors: int, gics_data: np.ndarray):
        self.r = num_factors
        self.gics_data = gics_data
        self.p = len(gics_data)
        self.same_sector_mask = (self.gics_data[:, None] == self.gics_data)
        np.fill_diagonal(self.same_sector_mask, True)

    def get_idiosyncratic_matrices(self, prvm_data: np.ndarray) -> np.ndarray:
        """전체 기간의 PRVM 데이터로부터 특이 변동성 행렬을 계산합니다."""
        print("\nPOET 기법을 적용하여 특이 변동성 행렬을 사전 계산합니다 (병렬 처리)...")
        
        idio_matrices = Parallel(n_jobs=-1)(
            delayed(self._calculate_idio_part)(prvm) for prvm in tqdm(prvm_data, desc="특이 행렬 계산 중")
        )
        return np.array(idio_matrices)

    def get_full_poet_matrix(self, prvm_matrix: np.ndarray) -> np.ndarray:
        """단일 PRVM 행렬로부터 완전한 POET 행렬을 계산합니다."""
        eigenvalues, eigenvectors = eigh(prvm_matrix)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:, idx]
        
        factor_part = sum(
            eigenvalues[i] * np.outer(eigenvectors[:, i], eigenvectors[:, i])
            for i in range(self.r)
        )
        
        idiosyncratic_raw = prvm_matrix - factor_part
        idiosyncratic_thresholded = idiosyncratic_raw * self.same_sector_mask
        
        return factor_part + idiosyncratic_thresholded

    def _calculate_idio_part(self, prvm_matrix: np.ndarray) -> np.ndarray:
        """특이 변동성 행렬만 계산하는 내부 헬퍼 함수."""
        eigenvalues, eigenvectors = eigh(prvm_matrix)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:, idx]
        
        factor_part = sum(
            eigenvalues[i] * np.outer(eigenvectors[:, i], eigenvectors[:, i])
            for i in range(self.r)
        )
        
        idiosyncratic_raw = prvm_matrix - factor_part
        return idiosyncratic_raw * self.same_sector_mask


# %% --- 4. 모델 정의 ---

class BaseEstimatorModel:
    def __init__(self, model_name: str): self.model_name = model_name
    def fit(self, **kwargs): raise NotImplementedError
    def predict(self, **kwargs) -> np.ndarray: raise NotImplementedError

class POETPRVMEstimator(BaseEstimatorModel):
    def __init__(self, num_factors: int, gics_data: np.ndarray):
        super().__init__("POET-PRVM")
        self.preprocessor = Preprocessor(num_factors, gics_data)
    def fit(self, **kwargs): pass
    def predict(self, prvm_matrix: np.ndarray, **kwargs) -> np.ndarray:
        return project_psd(self.preprocessor.get_full_poet_matrix(prvm_matrix))

class BaseFIVAREstimator(BaseEstimatorModel):
    def __init__(self, model_name: str, num_factors: int, h_lag: int, l_window: int):
        super().__init__(model_name)
        self.r, self.h, self.l = num_factors, h_lag, l_window
        self.p=None; self.factor_evecs=None; self.idio_evecs=None; self.factor_regressors=None; self.idio_regressors=None
        self.last_known_evals_full=None; self.last_known_evals_factor=None
        self.idio_mean_forecast=None; self.scaler_X_full=None; self.scaler_X_factor=None; self.scalers_y=None
        self.omega_F=None; self.omega_I=None; self.tau_F=None; self.tau_I=None

    def fit(self, prvm_train: np.ndarray, idio_train: np.ndarray):
        self.p = prvm_train.shape[1]; n_train = len(prvm_train)
        
        avg_prvm = np.mean(prvm_train[-self.l:], axis=0); _, factor_evecs_all = eigh(avg_prvm); self.factor_evecs = factor_evecs_all[:, -self.r:][:, ::-1]
        avg_idio = np.mean(idio_train[-self.l:], axis=0); _, idio_evecs_all = eigh(avg_idio); self.idio_evecs = idio_evecs_all[:, ::-1]
        
        factor_evals = np.array([np.diag(self.factor_evecs.T @ prvm @ self.factor_evecs) / self.p for prvm in prvm_train])
        idio_evals = np.array([np.diag(self.idio_evecs.T @ idio @ self.idio_evecs) for idio in idio_train])
        
        y_data = np.hstack([factor_evals[self.h:], idio_evals[self.h:]])
        
        X_full_data = np.hstack([np.hstack([factor_evals[i:n_train-self.h+i], idio_evals[i:n_train-self.h+i]]) for i in range(self.h)])
        X_factor_data = np.hstack([factor_evals[i:n_train-self.h+i] for i in range(self.h)])
        
        n_reg = len(y_data)
        
        sigma_F = np.sqrt(np.sum(np.var(y_data[:, :self.r], axis=0)) / self.r); self.omega_F = 4.0 * sigma_F * (n_reg / np.log(self.p))**0.25
        self.omega_I = 4.0 * (n_reg / np.log(self.p))**0.25; self.tau_F = 0.25 * sigma_F * (n_reg / np.log(self.p))**0.25; self.tau_I = 4.0 * (n_reg / np.log(self.p))**0.25

        # [핵심 수정] 모델 타입에 따라 Truncation 적용 여부 결정
        X_full_to_scale = X_full_data
        X_factor_to_scale = X_factor_data
        if self.model_name == 'H-LASSO':
            X_full_to_scale = self._apply_truncation_to_X(X_full_data, is_factor_only=False)
            X_factor_to_scale = self._apply_truncation_to_X(X_factor_data, is_factor_only=True)

        self.scaler_X_full = StandardScaler().fit(X_full_to_scale)
        self.scaler_X_factor = StandardScaler().fit(X_factor_to_scale)
        X_full_scaled = self.scaler_X_full.transform(X_full_to_scale)
        X_factor_scaled = self.scaler_X_factor.transform(X_factor_to_scale)
        
        self.scalers_y = [StandardScaler().fit(y_data[:, i].reshape(-1, 1)) for i in range(y_data.shape[1])]
        y_scaled = np.array([s.transform(y_data[:, i].reshape(-1, 1)).flatten() for i, s in enumerate(self.scalers_y)]).T
        
        self._fit_factor_models(X_factor_scaled, y_scaled)
        self._fit_idio_models(X_full_scaled, y_scaled, idio_evals)
        
        self.last_known_evals_full = np.hstack([factor_evals[-self.h:], idio_evals[-self.h:]])
        self.last_known_evals_factor = factor_evals[-self.h:]

    def predict(self) -> np.ndarray:
        if self.last_known_evals_full is None: raise RuntimeError("Model not fitted.")
        
        X_pred_full_raw = self.last_known_evals_full.flatten().reshape(1, -1)
        X_pred_factor_raw = self.last_known_evals_factor.flatten().reshape(1, -1)
        
        # [핵심 수정] 예측 시에도 모델 타입에 따라 Truncation 적용 여부 결정
        X_pred_full_to_scale = X_pred_full_raw
        X_pred_factor_to_scale = X_pred_factor_raw
        if self.model_name == 'H-LASSO':
            X_pred_full_to_scale = self._apply_truncation_to_X(X_pred_full_raw, is_factor_only=False)
            X_pred_factor_to_scale = self._apply_truncation_to_X(X_pred_factor_raw, is_factor_only=True)

        X_pred_full_scaled = self.scaler_X_full.transform(X_pred_full_to_scale)
        X_pred_factor_scaled = self.scaler_X_factor.transform(X_pred_factor_to_scale)
        
        pred_factor_evals_scaled = np.array([reg.predict(X_pred_factor_scaled)[0] for reg in self.factor_regressors])
        pred_factor_evals = np.array([self.scalers_y[i].inverse_transform(val.reshape(1, -1))[0, 0] for i, val in enumerate(pred_factor_evals_scaled)])

        if self.model_name == 'OLS':
            pred_idio_evals = self.idio_mean_forecast
        else:
            pred_idio_evals_scaled = np.array([reg.predict(X_pred_full_scaled)[0] for reg in self.idio_regressors])
            pred_idio_evals = np.array([self.scalers_y[self.r + i].inverse_transform(val.reshape(1, -1))[0, 0] for i, val in enumerate(pred_idio_evals_scaled)])
        
        pred_evals = np.concatenate([pred_factor_evals, pred_idio_evals]); pred_evals[pred_evals < 1e-10] = 1e-10
        
        factor_forecast = self.p * (self.factor_evecs @ np.diag(pred_evals[:self.r]) @ self.factor_evecs.T)
        idio_forecast = self.idio_evecs @ np.diag(pred_evals[self.r:]) @ self.idio_evecs.T
        
        return project_psd(factor_forecast + idio_forecast)

    def _apply_truncation_to_X(self, X_data: np.ndarray, is_factor_only: bool) -> np.ndarray:
        X_truncated = X_data.copy()
        if is_factor_only:
            for i in range(self.h):
                cols = slice(i * self.r, (i + 1) * self.r)
                X_truncated[:, cols] = truncate_data(X_truncated[:, cols], self.omega_F)
        else:
            for i in range(self.h):
                factor_cols = slice(i * (self.p + self.r), i * (self.p + self.r) + self.r)
                idio_cols = slice(i * (self.p + self.r) + self.r, (i + 1) * (self.p + self.r))
                X_truncated[:, factor_cols] = truncate_data(X_truncated[:, factor_cols], self.omega_F)
                X_truncated[:, idio_cols] = truncate_data(X_truncated[:, idio_cols], self.omega_I)
        return X_truncated

    def _fit_factor_models(self, X_factor_scaled, y_scaled): raise NotImplementedError
    def _fit_idio_models(self, X_full_scaled, y_scaled, idio_evals): raise NotImplementedError

class FIVAR_OLS_Estimator(BaseFIVAREstimator):
    def __init__(self, num_factors: int, h_lag: int, l_window: int):
        super().__init__("OLS", num_factors, h_lag, l_window)
    def _fit_factor_models(self, X_factor_scaled, y_scaled):
        self.factor_regressors = [LinearRegression(fit_intercept=False).fit(X_factor_scaled, y_scaled[:, i]) for i in range(self.r)]
    def _fit_idio_models(self, X_full_scaled, y_scaled, idio_evals):
        self.idio_mean_forecast = np.mean(idio_evals[-self.l:], axis=0)

class FIVAR_LASSO_Estimator(BaseFIVAREstimator):
    def __init__(self, num_factors: int, h_lag: int, l_window: int):
        super().__init__("LASSO", num_factors, h_lag, l_window)
    def _fit_factor_models(self, X_factor_scaled, y_scaled):
        self.factor_regressors = [LinearRegression(fit_intercept=False).fit(X_factor_scaled, y_scaled[:, i]) for i in range(self.r)]
    def _fit_idio_models(self, X_full_scaled, y_scaled, idio_evals):
        self.idio_regressors = [None] * self.p; n_reg = X_full_scaled.shape[0]
        def find_best_lasso(i):
            y_i = y_scaled[:, self.r + i]; best_bic, best_reg = np.inf, None
            for c_eta in np.linspace(0.1, 10.0, 20):
                eta_I = c_eta * np.sqrt(np.log(self.p) / n_reg); reg = Lasso(alpha=eta_I, fit_intercept=False, max_iter=2000).fit(X_full_scaled, y_i)
                rss = np.sum((y_i - reg.predict(X_full_scaled))**2); num_params = np.sum(np.abs(reg.coef_) > 1e-6)
                bic = n_reg * np.log(rss / n_reg) + num_params * np.log(n_reg) if rss > 0 else np.inf
                if bic < best_bic: best_bic, best_reg = bic, reg
            return i, best_reg
        results = Parallel(n_jobs=-1)(delayed(find_best_lasso)(i) for i in range(self.p))
        for i, reg in results: self.idio_regressors[i] = reg

class FIVAR_HLASSO_Estimator(BaseFIVAREstimator):
    def __init__(self, num_factors: int, h_lag: int, l_window: int):
        super().__init__("H-LASSO", num_factors, h_lag, l_window)
    def _fit_factor_models(self, X_factor_scaled, y_scaled):
        self.factor_regressors = [CustomHuberRegressor(alpha=0.0, epsilon=self.tau_F).fit(X_factor_scaled, y_scaled[:, i]) for i in range(self.r)]
    def _fit_idio_models(self, X_full_scaled, y_scaled, idio_evals):
        self.idio_regressors = [None] * self.p; n_reg = X_full_scaled.shape[0]
        def find_best_hlasso(i):
            y_i = y_scaled[:, self.r + i]; best_bic, best_reg = np.inf, None
            for c_eta in np.linspace(0.1, 10.0, 20):
                eta_I = c_eta * np.sqrt(np.log(self.p) / n_reg); reg = CustomHuberRegressor(alpha=eta_I, epsilon=self.tau_I).fit(X_full_scaled, y_i)
                rss = np.sum((y_i - reg.predict(X_full_scaled))**2); num_params = np.sum(np.abs(reg.coef_) > 1e-6)
                bic = n_reg * np.log(rss / n_reg) + num_params * np.log(n_reg) if rss > 0 else np.inf
                if bic < best_bic: best_bic, best_reg = bic, reg
            return i, best_reg
        results = Parallel(n_jobs=-1)(delayed(find_best_hlasso)(i) for i in range(self.p))
        for i, reg in results: self.idio_regressors[i] = reg


# %% --- 5. 평가 및 결과 분석 클래스 ---
# (이전 버전과 동일하므로 생략)
class Evaluator:
    def __init__(self, models: List[BaseEstimatorModel], periods: Dict):
        self.models = {model.model_name: model for model in models}; self.periods = periods; self.forecasts = {name: [] for name in self.models}; self.ground_truths = []; self.forecast_dates = []
    def run_evaluation(self, data: Dict, config: Dict):
        prvm_data = data["prvm_data"]; idio_data = data["idio_data"]; dates = data["dates"]
        start_idx = dates.searchsorted(pd.to_datetime(config['eval_start_date']))
        if start_idx < config['in_sample_size']: start_idx = config['in_sample_size']; print(f"Data insufficient, starting from {dates[start_idx].date()}.")
        for t in tqdm(range(start_idx, len(dates)), desc="Overall Model Prediction"):
            ground_truth_matrix = self.models['POET-PRVM'].predict(prvm_data[t]); self.ground_truths.append(ground_truth_matrix); self.forecast_dates.append(dates[t])
            self.forecasts['POET-PRVM'].append(self.models['POET-PRVM'].predict(prvm_data[t-1]))
            train_start = t - config['in_sample_size']; prvm_train = prvm_data[train_start:t]; idio_train = idio_data[train_start:t]
            for name in ['OLS', 'LASSO', 'H-LASSO']:
                model = self.models[name]; model.fit(prvm_train, idio_train); self.forecasts[name].append(model.predict())
        print("\nAll predictions are complete. Starting result analysis."); self.display_summary_table()
    def display_summary_table(self):
        print("\n--- Final Evaluation Results (Replicating Table 2) ---"); final_results = []
        forecast_dates_pd = pd.to_datetime(self.forecast_dates)
        for period_name, (start, end) in self.periods.items():
            mask = (forecast_dates_pd >= start) & (forecast_dates_pd <= end)
            period_truths = [self.ground_truths[i] for i, val in enumerate(mask) if val]
            for model_name in self.models:
                period_forecasts = [self.forecasts[model_name][i] for i, val in enumerate(mask) if val]
                if not period_forecasts: continue
                mspe = self._calculate_mspe(period_forecasts, period_truths); qlike = self._calculate_qlike(period_forecasts, period_truths)
                final_results.append({"Period": period_name, "Model": model_name, "MSPE (x10^4)": mspe * 1e4, "QLIKE x 10^-3": qlike * 1e-3})
        df = pd.DataFrame(final_results)
        summary = df.pivot_table(index='Model', columns='Period', values=['MSPE (x10^4)', 'QLIKE x 10^-3']).swaplevel(0, 1, axis=1).sort_index(axis=1)
        print(summary.loc[['POET-PRVM', 'OLS', 'LASSO', 'H-LASSO']].to_string(float_format="%.3f"))
    @staticmethod
    def _calculate_mspe(forecasts: List[np.ndarray], truths: List[np.ndarray]) -> float:
        errors = [np.linalg.norm(f - t, 'fro')**2 for f, t in zip(forecasts, truths)]; return np.mean(errors) if errors else np.nan
    @staticmethod
    def _calculate_qlike(forecasts: List[np.ndarray], truths: List[np.ndarray]) -> float:
        qlike_vals = []
        for f, t in zip(forecasts, truths):
            try:
                f_psd = project_psd(f); sign, log_det = np.linalg.slogdet(f_psd)
                if sign <= 0: continue
                trace_val = np.trace(inv(f_psd) @ t); qlike_vals.append(log_det + trace_val)
            except np.linalg.LinAlgError: continue
        return np.mean(qlike_vals) if qlike_vals else np.nan

class PortfolioOptimizer:
    def __init__(self, evaluator: Evaluator, data: Dict, config: Dict):
        self.evaluator = evaluator; self.data = data; self.config = config; self.intraday_returns = self._prepare_intraday_returns()
    def _prepare_intraday_returns(self) -> Dict[str, np.ndarray]:
        print("\nPreparing 10-minute returns for portfolio analysis...");
        if self.data['log_price_data'] is None: print("Warning: Log price data not found, skipping portfolio analysis."); return None
        returns_10min = self.data['log_price_data'].resample('10min').last().diff().dropna()
        daily_groups = returns_10min.groupby(returns_10min.index.date); return {date: group.values for date, group in daily_groups}
    def run_portfolio_analysis(self):
        if self.intraday_returns is None: return
        print("Starting portfolio risk analysis (this may take some time)..."); exposure_constraints = np.linspace(1.0, 3.0, 11)
        portfolio_risks = {name: {p: [] for p in self.evaluator.periods} for name in self.evaluator.models}
        for period_name, (start, end) in self.evaluator.periods.items():
            period_mask = (pd.to_datetime(self.evaluator.forecast_dates) >= start) & (pd.to_datetime(self.evaluator.forecast_dates) <= end)
            for model_name in self.evaluator.models:
                print(f"-> Calculating risk for {model_name} in {period_name}...")
                for c0 in tqdm(exposure_constraints, leave=False, desc=f"   Exposure (c0)"):
                    realized_variances = []
                    for i, is_in_period in enumerate(period_mask):
                        if not is_in_period: continue
                        continuous_vol = self.evaluator.forecasts[model_name][i]
                        jump_vol_idx = self.data['dates'].get_loc(self.evaluator.forecast_dates[i]) - 1
                        jump_vol = self.data['jump_vol_data'][jump_vol_idx] if jump_vol_idx >= 0 else 0
                        total_vol_forecast = continuous_vol + jump_vol
                        w = self._optimize_portfolio(total_vol_forecast, c0)
                        current_date = self.evaluator.forecast_dates[i].date()
                        if w is not None and current_date in self.intraday_returns:
                            returns_10min_today = self.intraday_returns[current_date]
                            portfolio_returns_10min = returns_10min_today @ w
                            realized_variances.append(np.sum(portfolio_returns_10min**2))
                    if realized_variances: portfolio_risks[model_name][period_name].append(np.sqrt(np.mean(realized_variances) * 252) * 100)
                    else: portfolio_risks[model_name][period_name].append(np.nan)
        self._plot_portfolio_risks(portfolio_risks, exposure_constraints)
    def _optimize_portfolio(self, cov_matrix: np.ndarray, c0: float) -> Union[np.ndarray, None]:
        n_assets = cov_matrix.shape[0]; w = cp.Variable(n_assets); objective = cp.Minimize(cp.quad_form(w, cov_matrix))
        constraints = [cp.sum(w) == 1, cp.norm(w, 1) <= c0]; prob = cp.Problem(objective, constraints)
        try:
            prob = cp.Problem(cp.Minimize(cp.quad_form(w, cov_matrix + 1e-8 * np.eye(n_assets))), constraints)
            prob.solve(solver=cp.SCS, verbose=False)
            return w.value if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] else None
        except cp.error.SolverError: return None
    def _plot_portfolio_risks(self, risks: Dict, constraints: np.ndarray):
        print("\nVisualizing portfolio risk analysis results."); fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
        styles = {'POET-PRVM': {'c': 'red', 'marker': '*', 'ls': '--'},'OLS': {'c': 'blue', 'marker': '+', 'ls': '--'},'LASSO': {'c': 'green', 'marker': 'o', 'ls': '--', 'mfc': 'none'},'H-LASSO': {'c': 'black', 'marker': 'o', 'ls': '-'}}
        for i, period_name in enumerate(self.evaluator.periods):
            ax = axes[i]
            for model_name, style in styles.items(): ax.plot(constraints, risks[model_name][period_name], label=model_name, **style)
            title_period = period_name.split('(')[1][:-1]; ax.set_title(f'Portfolio Risk ({title_period})', fontsize=14)
            ax.set_xlabel('Exposure Constraint (c0)', fontsize=12)
            if i == 0: ax.set_ylabel('Annualized Risk (%)', fontsize=12)
            ax.legend(); ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.suptitle('Portfolio Optimization Results (Replicating Figure 5)', fontsize=18, y=1.02); plt.tight_layout(); plt.show()


# %% --- 6. 메인 실행 스크립트 ---

if __name__ == '__main__':
    try: script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError: script_dir = os.getcwd()

    config = {
        "gics_file_path": os.path.join(script_dir, 'gicslist.csv'),
        "log_price_file_path": os.path.join(script_dir, 'DataY.csv'),
        "num_factors": 3, "h_lag": 1, "l_window": 22,
        "in_sample_size": 251, "eval_start_date": '2018-01-01',
        "periods": {
            "Period 1 (2018-2019)": ('2018-01-01', '2019-12-31'),
            "Period 2 (2018)": ('2018-01-01', '2018-12-31'),
            "Period 3 (2019)": ('2019-01-01', '2019-12-31'),
        }
    }

    data_loader = DataLoader(
        gics_path=config['gics_file_path'],
        log_price_path=config['log_price_file_path']
    )
    all_data = data_loader.load_all_data()

    preprocessor = Preprocessor(
        num_factors=config['num_factors'],
        gics_data=all_data['gics_data']
    )
    all_data['idio_data'] = preprocessor.get_idiosyncratic_matrices(all_data['prvm_data'])

    models_to_evaluate = [
        POETPRVMEstimator(config['num_factors'], all_data['gics_data']),
        FIVAR_OLS_Estimator(config['num_factors'], config['h_lag'], config['l_window']),
        FIVAR_LASSO_Estimator(config['num_factors'], config['h_lag'], config['l_window']),
        FIVAR_HLASSO_Estimator(config['num_factors'], config['h_lag'], config['l_window'])
    ]

    evaluator = Evaluator(models_to_evaluate, config['periods'])
    evaluator.run_evaluation(all_data, config)

    portfolio_analyzer = PortfolioOptimizer(evaluator, {
        'log_price_data': all_data['log_price_data'],
        'jump_vol_data': all_data['jump_vol_data'],
        'dates': all_data['dates'],
        'tickers': all_data['tickers']
    }, config)
    portfolio_analyzer.run_portfolio_analysis()
