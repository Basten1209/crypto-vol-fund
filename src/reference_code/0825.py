# -*- coding: utf-8 -*-
"""
FIVAR Model Implementation and Evaluation (Refactored Version)

이 스크립트는 "Factor and Idiosyncratic VAR Volatility Matrix Models for
Heavy-Tailed High-Frequency Financial Observations" 논문에 제시된 FIVAR 모델을
구현하고 평가합니다.

주요 기능:
1.  논문에서 사용된 데이터(PRVM, GICS, 고빈도 로그 가격) 로딩 및 전처리
2.  4가지 Estimator 구현:
    - POETPRVMEstimator: 비모수적 벤치마크 모델
    - FIVAR_OLS_Estimator: 요인(Factor) 동적성만 고려하는 OLS 기반 모델
    - FIVAR_LASSO_Estimator: 요인 및 특이(Idiosyncratic) 동적성을 모두 고려하는 LASSO 기반 모델
    - FIVAR_HLASSO_Estimator: LASSO 모델에 강건성(Robustness)을 추가한 최종 제안 모델
3.  롤링 윈도우(Rolling Window) 방식을 사용한 시계열 예측 수행
4.  MSPE, QLIKE 지표를 사용한 모델 성능 평가 (논문 Table 2 재현)
5.  포트폴리오 최적화를 통한 리스크 분석 및 시각화 (논문 Figure 5 재현)

실행 순서:
Spyder IDE와 같은 환경에서 각 셀(#%%)을 순서대로 실행하는 것을 권장합니다.
"""

# %% --- 1. 라이브러리 임포트 및 기본 설정 ---

import pandas as pd
import numpy as np
import os
import warnings
from tqdm import tqdm
from typing import Dict, List, Tuple, Union

from scipy.linalg import eigh, inv
from scipy.optimize import fmin_l_bfgs_b
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

import matplotlib.pyplot as plt
import seaborn as sns

# 포트폴리오 최적화를 위한 cvxpy 라이브러리 임포트
try:
    import cvxpy as cp
except ImportError:
    print("cvxpy 라이브러리가 설치되어 있지 않습니다. 설치를 시작합니다...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "cvxpy"])
    import cvxpy as cp

# 스타일 설정
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 경고 메시지 무시
warnings.filterwarnings("ignore")


# %% --- 2. 유틸리티 함수 및 커스텀 클래스 정의 ---

def project_psd(matrix: np.ndarray) -> np.ndarray:
    """
    주어진 행렬을 가장 가까운 양의 준정부호(Positive Semi-Definite, PSD) 행렬로 변환합니다.
    대칭성을 보장하고, 음수 고유값을 0에 가까운 작은 양수로 클리핑합니다.

    Args:
        matrix (np.ndarray): 변환할 입력 행렬.

    Returns:
        np.ndarray: PSD로 변환된 행렬.
    """
    symmetric_matrix = (matrix + matrix.T) / 2
    eigenvalues, eigenvectors = eigh(symmetric_matrix)
    eigenvalues[eigenvalues < 1e-10] = 1e-10  # 수치적 안정을 위해 작은 양수로 클리핑
    psd_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    return (psd_matrix + psd_matrix.T) / 2

def truncate_data(data: np.ndarray, omega: float) -> np.ndarray:
    """
    데이터를 특정 임계값(omega)으로 잘라내는 Truncation(Winsorization)을 수행합니다.
    논문의 psi_omega(x) 함수 (식 (3.1) 이전)에 해당합니다.
    이는 데이터의 극단치(outlier) 영향을 완화하여 모델의 강건성을 높입니다.

    Args:
        data (np.ndarray): 처리할 데이터.
        omega (float): 상/하한 임계값.

    Returns:
        np.ndarray: Truncation이 적용된 데이터.
    """
    return np.clip(data, -omega, omega)

class CustomHuberRegressor(BaseEstimator, RegressorMixin):
    """
    논문의 H-LASSO 모델 구현을 위한 커스텀 Huber 회귀 모델.
    scikit-learn의 HuberRegressor와 달리 L1 페널티(alpha)를 지원하며,
    epsilon 제약 조건을 완화하여 논문의 요구사항을 충족시킵니다.

    Args:
        alpha (float): L1 규제 강도 (LASSO 페널티).
        epsilon (float): Huber 손실 함수의 선형-제곱 영역 전환 임계값 (논문의 tau).
        max_iter (int): 최적화 최대 반복 횟수.
        tol (float): 최적화 수렴 허용 오차.
    """
    def __init__(self, alpha: float = 0.0, epsilon: float = 1.35, max_iter: int = 100, tol: float = 1e-5):
        self.alpha = alpha
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None

    def _objective_function(self, coef: np.ndarray, X: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Huber 손실과 L1 페널티를 결합한 목적 함수와 그래디언트를 계산합니다.
        (논문 식 (3.2), (3.4)의 l_tau 부분)
        """
        n_samples = X.shape[0]
        residuals = X @ coef - y
        abs_residuals = np.abs(residuals)
        
        # Huber 손실 계산
        mask_linear = abs_residuals > self.epsilon
        loss_huber = 0.5 * np.sum(residuals[~mask_linear] ** 2)
        loss_huber += self.epsilon * np.sum(abs_residuals[mask_linear] - 0.5 * self.epsilon)
        
        # L1 페널티 추가
        loss = loss_huber / n_samples + self.alpha * np.sum(np.abs(coef))

        # 그래디언트 계산
        grad = np.zeros_like(coef)
        grad += X[~mask_linear].T @ residuals[~mask_linear]
        grad += self.epsilon * X[mask_linear].T @ np.sign(residuals[mask_linear])
        grad /= n_samples
        
        return loss, grad

    def fit(self, X: np.ndarray, y: np.ndarray):
        """ L-BFGS-B 최적화 알고리즘을 사용하여 모델 계수를 학습합니다. """
        initial_coef = np.zeros(X.shape[1])
        
        coef, _, _ = fmin_l_bfgs_b(
            func=self._objective_function,
            x0=initial_coef,
            args=(X, y),
            maxiter=self.max_iter,
            pgtol=self.tol,
        )
        self.coef_ = coef
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """ 학습된 계수를 사용하여 예측값을 반환합니다. """
        return X @ self.coef_


# %% --- 3. 데이터 로더 및 전처리기 ---

class DataLoader:
    """데이터 로딩 및 전처리를 담당하는 클래스."""
    def __init__(self, prvm_path: str, gics_path: str, log_price_path: str):
        self.prvm_path = prvm_path
        self.gics_path = gics_path
        self.log_price_path = log_price_path

    def load_all_data(self) -> Dict[str, Union[np.ndarray, pd.DatetimeIndex, List[str]]]:
        """모든 필요한 데이터를 로드하고 기본 전처리를 수행합니다."""
        print("데이터 로딩을 시작합니다...")
        prvm_data, dates, tickers = self._load_prvm_data()
        gics_data = self._load_gics_data(tickers)
        jump_vol_data = self._calculate_jump_volatility(prvm_data, dates)
        
        print("\n모든 데이터 로딩 및 전처리가 완료되었습니다.")
        return {
            "prvm_data": prvm_data,
            "dates": dates,
            "tickers": tickers,
            "gics_data": gics_data,
            "jump_vol_data": jump_vol_data
        }

    def _load_prvm_data(self) -> Tuple[np.ndarray, pd.DatetimeIndex, List[str]]:
        """Long-format의 PRVM CSV 파일을 불러와 3D 텐서로 변환합니다."""
        print(f"-> PRVM 데이터 로딩: '{os.path.basename(self.prvm_path)}'")
        df = pd.read_csv(self.prvm_path, parse_dates=['date'])
        unique_dates = sorted(df['date'].unique())
        tickers = sorted(pd.unique(df[['ticker_i', 'ticker_j']].values.ravel('K')))
        
        num_days, num_assets = len(unique_dates), len(tickers)
        print(f"   총 {num_days}일, {num_assets}개 자산 데이터 처리 중...")
        
        prvm_data = np.zeros((num_days, num_assets, num_assets))
        ticker_map = {ticker: i for i, ticker in enumerate(tickers)}
        
        df['i'] = df['ticker_i'].map(ticker_map)
        df['j'] = df['ticker_j'].map(ticker_map)

        for t, date in enumerate(tqdm(unique_dates, desc="   PRVM 텐서 변환")):
            daily_data = df[df['date'] == date]
            matrix = np.zeros((num_assets, num_assets))
            matrix[daily_data['i'], daily_data['j']] = daily_data['value']
            matrix[daily_data['j'], daily_data['i']] = daily_data['value']
            prvm_data[t] = matrix
            
        return prvm_data, pd.to_datetime(unique_dates), tickers

    def _load_gics_data(self, tickers: List[str]) -> np.ndarray:
        """GICS 섹터 정보를 로드합니다."""
        print(f"-> GICS 데이터 로딩: '{os.path.basename(self.gics_path)}'")
        gics_df = pd.read_csv(self.gics_path, header=None)
        gics_sectors = gics_df.iloc[1:, 0].astype(str).to_numpy()
        if len(gics_sectors) != len(tickers):
            raise ValueError("GICS 데이터와 자산 수가 일치하지 않습니다.")
        return gics_sectors

    def _calculate_jump_volatility(self, prvm_continuous: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
        """
        고빈도 로그 가격 데이터로부터 총 변동성을 계산하고,
        이를 이용해 점프 변동성(총 변동성 - 연속 변동성)을 추정합니다.
        """
        print(f"-> 점프 변동성 계산: '{os.path.basename(self.log_price_path)}'")
        try:
            log_price_df = pd.read_csv(self.log_price_path, header=0, index_col=0, parse_dates=True)
            daily_groups = log_price_df.groupby(log_price_df.index.date)
            
            jump_vol_all = np.zeros_like(prvm_continuous)
            m = 390  # 1분 데이터 기준 하루 관측치 수
            K = int(np.floor(m**0.5)) # 논문에 따른 K값 설정

            for t, date in enumerate(tqdm(dates, desc="   점프 변동성 계산")):
                if date.date() in daily_groups.groups:
                    daily_log_prices = daily_groups.get_group(date.date()).T.values
                    prvm_total = self._prvm_from_log_prices(daily_log_prices, K)
                    
                    jump_vol = prvm_total - prvm_continuous[t]
                    jump_vol[jump_vol < 0] = 0 # 이론적으로 0 이상이어야 함
                    jump_vol_all[t] = jump_vol
            
            return jump_vol_all
        except FileNotFoundError:
            print(f"   경고: '{self.log_price_path}' 파일을 찾을 수 없어 점프 변동성을 0으로 처리합니다.")
            return np.zeros_like(prvm_continuous)

    @staticmethod
    def _prvm_from_log_prices(daily_log_prices: np.ndarray, K: int) -> np.ndarray:
        """
        고빈도 로그 가격으로부터 점프 조정을 적용하지 않은 총 PRVM을 계산합니다.
        (논문 부록 A.2, 식 (A.2)에서 점프 절단 부분을 제외)
        """
        p, m = daily_log_prices.shape
        daily_returns = np.diff(daily_log_prices, axis=1)
        
        g = np.array([min(i, K - i) / K for i in range(1, K)])
        psi = np.sum(g**2)

        pre_averaged_returns = np.zeros((p, m - K + 1))
        for k in range(m - K + 1):
            pre_averaged_returns[:, k] = np.sum(daily_returns[:, k:k+K-1] * g, axis=1)

        return (1 / (psi * K)) * (pre_averaged_returns @ pre_averaged_returns.T)

class Preprocessor:
    """
    POET 기법을 적용하여 특이(Idiosyncratic) 변동성 행렬을 사전 계산하는 클래스.
    (논문 4.2절 POET 적용 부분)
    """
    def __init__(self, num_factors: int, gics_data: np.ndarray):
        self.r = num_factors
        self.gics_data = gics_data
        self.p = len(gics_data)
        # 같은 섹터에 속하는 자산 쌍을 나타내는 마스크. 대각선은 항상 True.
        self.same_sector_mask = (self.gics_data[:, None] == self.gics_data)
        np.fill_diagonal(self.same_sector_mask, True)

    def get_idiosyncratic_matrices(self, prvm_data: np.ndarray) -> np.ndarray:
        """
        전체 기간의 PRVM 데이터에 POET 기법을 병렬로 적용하여
        특이 변동성 행렬 시계열을 생성합니다.
        """
        print("\nPOET 기법을 적용하여 특이 변동성 행렬을 사전 계산합니다 (병렬 처리)...")
        
        idio_matrices = Parallel(n_jobs=-1)(
            delayed(self._apply_poet)(prvm) for prvm in tqdm(prvm_data, desc="POET 적용 중")
        )
        return np.array(idio_matrices)

    def _apply_poet(self, prvm_matrix: np.ndarray) -> np.ndarray:
        """단일 PRVM 행렬에 POET 기법을 적용합니다."""
        prvm_psd = project_psd(prvm_matrix)
        eigenvalues, eigenvectors = eigh(prvm_psd)
        
        # 고유값을 내림차순으로 정렬
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:, idx]
        
        # 상위 r개의 고유값/고유벡터로 요인(Factor) 부분 재구성
        factor_part = sum(
            eigenvalues[i] * np.outer(eigenvectors[:, i], eigenvectors[:, i])
            for i in range(self.r)
        )
        
        # 원본 행렬에서 요인 부분을 빼서 특이(Idiosyncratic) 부분 계산
        idiosyncratic_raw = prvm_psd - factor_part
        
        # GICS 섹터 정보로 Thresholding 적용 (다른 섹터 간 공분산은 0으로)
        return idiosyncratic_raw * self.same_sector_mask


# %% --- 4. 모델 정의 ---

class BaseEstimatorModel:
    """모든 Estimator 모델의 기반이 되는 추상 클래스."""
    def __init__(self, model_name: str):
        self.model_name = model_name

    def fit(self, **kwargs):
        raise NotImplementedError

    def predict(self, **kwargs) -> np.ndarray:
        raise NotImplementedError

class POETPRVMEstimator(BaseEstimatorModel):
    """
    POET-PRVM 모델. t-1 시점의 PRVM에 POET을 적용하여 t 시점을 예측합니다.
    가장 간단한 비모수적 벤치마크 역할을 합니다.
    """
    def __init__(self, num_factors: int, gics_data: np.ndarray):
        super().__init__("POET-PRVM")
        self.preprocessor = Preprocessor(num_factors, gics_data)

    def fit(self, **kwargs):
        """POET-PRVM은 별도의 학습 과정이 없습니다."""
        pass

    def predict(self, prvm_t_minus_1: np.ndarray, **kwargs) -> np.ndarray:
        """t-1 시점의 PRVM을 POET 처리하여 예측값으로 반환합니다."""
        poet_prvm = self.preprocessor._apply_poet(prvm_t_minus_1)
        return project_psd(poet_prvm)

class BaseFIVAREstimator(BaseEstimatorModel):
    """OLS, LASSO, H-LASSO 모델의 공통 로직을 담는 기반 클래스."""
    def __init__(self, model_name: str, num_factors: int, h_lag: int, l_window: int):
        super().__init__(model_name)
        self.r, self.h, self.l = num_factors, h_lag, l_window
        self.p = None
        self.factor_evecs, self.idio_evecs = None, None
        self.factor_regressors, self.idio_regressors = None, None
        self.last_known_eigenvalues = None
        self.idio_mean_forecast = None # OLS 모델에서만 사용
        self.scaler_X, self.scalers_y = None, None
        self.omega_F, self.omega_I = None, None # Truncation 파라미터
        self.tau_F, self.tau_I = None, None # Huber Loss 파라미터

    def fit(self, prvm_train: np.ndarray, idio_train: np.ndarray):
        """FIVAR 모델의 학습 과정을 수행합니다."""
        self.p = prvm_train.shape[1]
        n_train = len(prvm_train)

        # 1. 고유벡터(Eigenvector) 추정 (논문 4.2절)
        # 최근 l일 간의 평균 공분산 행렬로부터 안정적인 고유벡터를 추정
        avg_prvm = np.mean(prvm_train[-self.l:], axis=0)
        _, factor_evecs_all = eigh(avg_prvm)
        self.factor_evecs = factor_evecs_all[:, -self.r:][:, ::-1]

        avg_idio = np.mean(idio_train[-self.l:], axis=0)
        _, idio_evecs_all = eigh(avg_idio)
        self.idio_evecs = idio_evecs_all[:, ::-1]

        # 2. 고유값(Eigenvalue) 시계열 생성
        factor_evals = np.array([np.diag(self.factor_evecs.T @ prvm @ self.factor_evecs) / self.p for prvm in prvm_train])
        idio_evals = np.array([np.diag(self.idio_evecs.T @ idio @ self.idio_evecs) for idio in idio_train])
        
        # 3. VAR 모델을 위한 데이터(X, y) 준비
        y_data = np.hstack([factor_evals[self.h:], idio_evals[self.h:]])
        X_list = [np.hstack([factor_evals[i:n_train-self.h+i], idio_evals[i:n_train-self.h+i]]) for i in range(self.h)]
        X_data = np.hstack(X_list)
        
        n_reg = len(y_data)

        # 4. 강건성(Robustness)을 위한 파라미터 계산 (논문 4.3절, 식 (4.13), (4.14))
        # Truncation 임계값 (omega)
        sigma_F = np.sqrt(np.sum(np.var(y_data[:, :self.r], axis=0)) / self.r)
        self.omega_F = 4.0 * sigma_F * (n_reg / np.log(self.p))**0.25
        self.omega_I = 4.0 * (n_reg / np.log(self.p))**0.25
        
        # Huber Loss 임계값 (tau)
        self.tau_F = 0.25 * sigma_F * (n_reg / np.log(self.p))**0.25
        self.tau_I = 4.0 * (n_reg / np.log(self.p))**0.25

        # 5. 데이터 전처리: Truncation 및 Standardization
        X_truncated = self._apply_truncation_to_X(X_data)
        
        self.scaler_X = StandardScaler().fit(X_truncated)
        X_scaled = self.scaler_X.transform(X_truncated)
        
        self.scalers_y = [StandardScaler().fit(y_data[:, i].reshape(-1, 1)) for i in range(y_data.shape[1])]
        y_scaled = np.array([self.scalers_y[i].transform(y_data[:, i].reshape(-1, 1)).flatten() for i in range(y_data.shape[1])]).T

        # 6. 모델 학습 (세부 구현은 하위 클래스에서 정의)
        self._fit_factor_models(X_scaled, y_scaled, n_reg)
        self._fit_idio_models(X_scaled, y_scaled, n_reg)

        # 7. 예측에 사용할 마지막 데이터 저장
        self.last_known_eigenvalues = np.hstack([factor_evals[-self.h:], idio_evals[-self.h:]])

    def predict(self) -> np.ndarray:
        """학습된 모델을 사용하여 다음 시점의 변동성 행렬을 예측합니다."""
        if self.last_known_eigenvalues is None:
            raise RuntimeError("모델이 학습되지 않았습니다. fit()을 먼저 호출하세요.")
        
        # 1. 예측을 위한 입력(X) 데이터 준비
        X_pred_raw = self.last_known_eigenvalues.flatten().reshape(1, -1)
        
        # 2. 학습과 동일한 전처리 적용: Truncation 및 Standardization
        X_pred_truncated = self._apply_truncation_to_X(X_pred_raw)
        X_pred_scaled = self.scaler_X.transform(X_pred_truncated)
        
        # 3. 고유값 예측
        pred_evals_scaled = np.zeros(self.p + self.r)
        
        # 요인 고유값 예측
        X_pred_factor_scaled = X_pred_scaled[:, :self.r * self.h]
        for i in range(self.r):
            pred_evals_scaled[i] = self.factor_regressors[i].predict(X_pred_factor_scaled)[0]
        
        # 특이 고유값 예측
        if self.model_name == 'OLS':
            # OLS는 과거 평균으로 예측 (논문 5.1절)
            pred_idio_evals = self.idio_mean_forecast
        else:
            pred_idio_evals_scaled = np.zeros(self.p)
            for i in range(self.p):
                pred_idio_evals_scaled[i] = self.idio_regressors[i].predict(X_pred_scaled)[0]
            
            # 예측된 표준화 값을 원래 스케일로 복원
            pred_idio_evals = np.array([
                self.scalers_y[self.r + i].inverse_transform(pred_idio_evals_scaled[i].reshape(1, -1))[0, 0]
                for i in range(self.p)
            ])

        # 4. 최종 예측 행렬 재구성
        pred_factor_evals = np.array([
            self.scalers_y[i].inverse_transform(pred_evals_scaled[i].reshape(1, -1))[0, 0]
            for i in range(self.r)
        ])
        
        pred_evals = np.concatenate([pred_factor_evals, pred_idio_evals])
        pred_evals[pred_evals < 1e-10] = 1e-10 # PSD 보장을 위해 양수 클리핑
        
        factor_forecast = self.p * (self.factor_evecs @ np.diag(pred_evals[:self.r]) @ self.factor_evecs.T)
        idio_forecast = self.idio_evecs @ np.diag(pred_evals[self.r:]) @ self.idio_evecs.T
        
        return project_psd(factor_forecast + idio_forecast)

    def _apply_truncation_to_X(self, X_data: np.ndarray) -> np.ndarray:
        """VAR 모델의 입력 데이터(X)에 Truncation을 적용합니다."""
        X_truncated = X_data.copy()
        for i in range(self.h):
            start_idx, end_idx = i * (self.p + self.r), (i + 1) * (self.p + self.r)
            X_truncated[:, start_idx : start_idx + self.r] = truncate_data(X_truncated[:, start_idx : start_idx + self.r], self.omega_F)
            X_truncated[:, start_idx + self.r : end_idx] = truncate_data(X_truncated[:, start_idx + self.r : end_idx], self.omega_I)
        return X_truncated

    def _fit_factor_models(self, X_scaled, y_scaled, n_reg):
        """요인(Factor) 고유값에 대한 회귀 모델들을 학습합니다."""
        raise NotImplementedError
        
    def _fit_idio_models(self, X_scaled, y_scaled, n_reg):
        """특이(Idiosyncratic) 고유값에 대한 회귀 모델들을 학습합니다."""
        raise NotImplementedError

class FIVAR_OLS_Estimator(BaseFIVAREstimator):
    """FIVAR-OLS 모델. 요인 부분만 동적 모델링하고, 특이 부분은 과거 평균으로 예측."""
    def __init__(self, num_factors: int, h_lag: int, l_window: int):
        super().__init__("OLS", num_factors, h_lag, l_window)

    def _fit_factor_models(self, X_scaled, y_scaled, n_reg):
        self.factor_regressors = [None] * self.r
        X_factor_scaled = X_scaled[:, :self.r * self.h]
        
        for i in range(self.r):
            reg = LinearRegression(fit_intercept=False)
            reg.fit(X_factor_scaled, y_scaled[:, i])
            self.factor_regressors[i] = reg

    def _fit_idio_models(self, X_scaled, y_scaled, n_reg):
        """OLS 모델은 특이 고유값을 모델링하지 않고, 마지막 l일의 평균을 사용합니다."""
        self.idio_mean_forecast = np.mean(y_scaled[-self.l:, self.r:], axis=0)
        # 스케일 복원을 위해 원본 데이터의 평균을 사용
        original_y_data = self.scalers_y[self.r].inverse_transform(y_scaled[:, self.r].reshape(-1,1))
        for i in range(1, self.p):
             original_y_data = np.hstack([original_y_data, self.scalers_y[self.r+i].inverse_transform(y_scaled[:, self.r+i].reshape(-1,1))])
        self.idio_mean_forecast = np.mean(original_y_data[-self.l:], axis=0)


class FIVAR_LASSO_Estimator(BaseFIVAREstimator):
    """FIVAR-LASSO 모델. 요인과 특이 부분 모두 동적 모델링하며, 특이 부분에 LASSO 규제 적용."""
    def __init__(self, num_factors: int, h_lag: int, l_window: int):
        super().__init__("LASSO", num_factors, h_lag, l_window)

    def _fit_factor_models(self, X_scaled, y_scaled, n_reg):
        # LASSO 모델의 요인 부분은 OLS와 동일하게 학습
        self.factor_regressors = [None] * self.r
        X_factor_scaled = X_scaled[:, :self.r * self.h]
        for i in range(self.r):
            reg = LinearRegression(fit_intercept=False)
            reg.fit(X_factor_scaled, y_scaled[:, i])
            self.factor_regressors[i] = reg

    def _fit_idio_models(self, X_scaled, y_scaled, n_reg):
        """BIC를 사용하여 최적의 LASSO 규제 파라미터를 찾고 모델을 병렬로 학습합니다."""
        self.idio_regressors = [None] * self.p
        
        def find_best_lasso(i):
            y_i = y_scaled[:, self.r + i]
            best_bic, best_reg = np.inf, None
            
            # 논문에 따라 BIC를 최소화하는 규제 파라미터 탐색
            for c_eta in np.linspace(0.1, 10.0, 20):
                eta_I = c_eta * np.sqrt(np.log(self.p) / n_reg)
                reg = Lasso(alpha=eta_I, fit_intercept=False, max_iter=2000)
                reg.fit(X_scaled, y_i)
                
                y_pred = reg.predict(X_scaled)
                rss = np.sum((y_i - y_pred)**2)
                num_params = np.sum(np.abs(reg.coef_) > 1e-6)
                
                bic = n_reg * np.log(rss / n_reg) + num_params * np.log(n_reg) if rss > 0 else np.inf
                if bic < best_bic:
                    best_bic, best_reg = bic, reg
            return i, best_reg

        results = Parallel(n_jobs=-1)(delayed(find_best_lasso)(i) for i in range(self.p))
        for i, reg in results:
            self.idio_regressors[i] = reg

class FIVAR_HLASSO_Estimator(BaseFIVAREstimator):
    """FIVAR-H-LASSO 모델. LASSO 모델에 Huber 손실을 추가하여 heavy-tail 데이터에 대한 강건성을 확보."""
    def __init__(self, num_factors: int, h_lag: int, l_window: int):
        super().__init__("H-LASSO", num_factors, h_lag, l_window)

    def _fit_factor_models(self, X_scaled, y_scaled, n_reg):
        self.factor_regressors = [None] * self.r
        X_factor_scaled = X_scaled[:, :self.r * self.h]
        for i in range(self.r):
            # 요인 부분은 L1 규제 없이 Huber 회귀만 적용 (논문 식 (3.3))
            reg = CustomHuberRegressor(alpha=0.0, epsilon=self.tau_F)
            reg.fit(X_factor_scaled, y_scaled[:, i])
            self.factor_regressors[i] = reg

    def _fit_idio_models(self, X_scaled, y_scaled, n_reg):
        """BIC를 사용하여 최적의 H-LASSO 규제 파라미터를 찾고 모델을 병렬로 학습합니다."""
        self.idio_regressors = [None] * self.p

        def find_best_hlasso(i):
            y_i = y_scaled[:, self.r + i]
            best_bic, best_reg = np.inf, None

            for c_eta in np.linspace(0.1, 10.0, 20):
                eta_I = c_eta * np.sqrt(np.log(self.p) / n_reg)
                reg = CustomHuberRegressor(alpha=eta_I, epsilon=self.tau_I)
                reg.fit(X_scaled, y_i)
                
                y_pred = reg.predict(X_scaled)
                rss = np.sum((y_i - y_pred)**2)
                num_params = np.sum(np.abs(reg.coef_) > 1e-6)
                
                bic = n_reg * np.log(rss / n_reg) + num_params * np.log(n_reg) if rss > 0 else np.inf
                if bic < best_bic:
                    best_bic, best_reg = bic, reg
            return i, best_reg

        results = Parallel(n_jobs=-1)(delayed(find_best_hlasso)(i) for i in range(self.p))
        for i, reg in results:
            self.idio_regressors[i] = reg


# %% --- 5. 평가 및 결과 분석 클래스 ---

class Evaluator:
    """모델 평가 및 결과 시각화를 담당하는 클래스."""
    def __init__(self, models: List[BaseEstimatorModel], periods: Dict):
        self.models = {model.model_name: model for model in models}
        self.periods = periods
        self.forecasts = {name: [] for name in self.models}
        self.ground_truths = []
        self.forecast_dates = []

    def run_evaluation(self, data: Dict, config: Dict):
        """롤링 윈도우 방식으로 전체 평가 파이프라인을 실행합니다."""
        prvm_data = data["prvm_data"]
        idio_data = data["idio_data"]
        dates = data["dates"]
        
        start_idx = dates.searchsorted(pd.to_datetime(config['eval_start_date']))
        if start_idx < config['in_sample_size']:
            start_idx = config['in_sample_size']
            print(f"데이터가 충분하지 않아 {dates[start_idx].date()}부터 예측을 시작합니다.")

        for t in tqdm(range(start_idx, len(dates)), desc="전체 모델 예측 진행"):
            # 1. Ground Truth 생성 (POET-PRVM(t))
            ground_truth_matrix = self.models['POET-PRVM'].predict(prvm_data[t])
            self.ground_truths.append(ground_truth_matrix)
            self.forecast_dates.append(dates[t])

            # 2. POET-PRVM 예측 (POET-PRVM(t-1))
            self.forecasts['POET-PRVM'].append(self.models['POET-PRVM'].predict(prvm_data[t-1]))
            
            # 3. FIVAR 모델들 예측
            train_start = t - config['in_sample_size']
            prvm_train = prvm_data[train_start:t]
            idio_train = idio_data[train_start:t]
            
            for name in ['OLS', 'LASSO', 'H-LASSO']:
                model = self.models[name]
                model.fit(prvm_train, idio_train)
                self.forecasts[name].append(model.predict())
        
        print("\n모든 예측이 완료되었습니다. 결과 분석을 시작합니다.")
        self.display_summary_table()

    def display_summary_table(self):
        """MSPE와 QLIKE 결과를 논문 Table 2 형식으로 출력합니다."""
        print("\n--- 최종 평가 결과 (논문 Table 2 형식) ---")
        final_results = []
        forecast_dates_pd = pd.to_datetime(self.forecast_dates)
        
        for period_name, (start, end) in self.periods.items():
            mask = (forecast_dates_pd >= start) & (forecast_dates_pd <= end)
            period_truths = [self.ground_truths[i] for i, val in enumerate(mask) if val]
            
            for model_name in self.models:
                period_forecasts = [self.forecasts[model_name][i] for i, val in enumerate(mask) if val]
                if not period_forecasts: continue

                mspe = self._calculate_mspe(period_forecasts, period_truths)
                qlike = self._calculate_qlike(period_forecasts, period_truths)
                
                final_results.append({
                    "Period": period_name, "Model": model_name,
                    "MSPE (x10^4)": mspe * 1e4,
                    "QLIKE x 10^-3": qlike * 1e-3,
                })

        df = pd.DataFrame(final_results)
        summary = df.pivot_table(
            index='Model', columns='Period', values=['MSPE (x10^4)', 'QLIKE x 10^-3']
        ).swaplevel(0, 1, axis=1).sort_index(axis=1)
        
        model_order = ['POET-PRVM', 'OLS', 'LASSO', 'H-LASSO']
        print(summary.loc[model_order].to_string(float_format="%.3f"))

    @staticmethod
    def _calculate_mspe(forecasts: List[np.ndarray], truths: List[np.ndarray]) -> float:
        """Mean Squared Prediction Error (Frobenius norm 기반)를 계산합니다."""
        errors = [np.linalg.norm(f - t, 'fro')**2 for f, t in zip(forecasts, truths)]
        return np.mean(errors) if errors else np.nan

    @staticmethod
    def _calculate_qlike(forecasts: List[np.ndarray], truths: List[np.ndarray]) -> float:
        """QLIKE 손실 함수를 계산합니다."""
        qlike_vals = []
        for f, t in zip(forecasts, truths):
            try:
                f_psd = project_psd(f)
                sign, log_det = np.linalg.slogdet(f_psd)
                if sign <= 0: continue
                trace_val = np.trace(inv(f_psd) @ t)
                qlike_vals.append(log_det + trace_val)
            except np.linalg.LinAlgError:
                continue
        return np.mean(qlike_vals) if qlike_vals else np.nan

class PortfolioOptimizer:
    """포트폴리오 최적화 및 리스크 분석을 수행하는 클래스."""
    def __init__(self, evaluator: Evaluator, data: Dict, config: Dict):
        self.evaluator = evaluator
        self.data = data
        self.config = config
        self.daily_returns = self._prepare_daily_returns()

    def _prepare_daily_returns(self) -> pd.DataFrame:
        """고빈도 로그 가격 데이터로부터 일별 수익률을 계산합니다."""
        print("\n포트폴리오 분석을 위해 일별 수익률을 계산합니다...")
        try:
            log_price_df = pd.read_csv(self.data['log_price_path'], index_col=0, parse_dates=True)
            # 일별 수익률은 그날의 마지막 가격과 전날 마지막 가격의 차이로 계산
            daily_returns_df = log_price_df.resample('D').last().diff().dropna()
            
            # 분석 기간과 자산에 맞춰 데이터 정렬 및 결측치 처리
            forecast_dates_pd = pd.to_datetime(self.evaluator.forecast_dates)
            aligned_returns = daily_returns_df.reindex(index=forecast_dates_pd, columns=self.data['tickers']).fillna(0)
            return aligned_returns
        except FileNotFoundError:
            print(f"경고: '{self.data['log_price_path']}' 파일이 없어 포트폴리오 분석을 건너뜁니다.")
            return None

    def run_portfolio_analysis(self):
        """모든 모델과 기간에 대해 포트폴리오 리스크를 계산하고 결과를 시각화합니다."""
        if self.daily_returns is None: return

        print("포트폴리오 리스크 분석을 시작합니다 (시간이 다소 소요될 수 있습니다)...")
        exposure_constraints = np.linspace(1.0, 3.0, 11)
        portfolio_risks = {name: {p: [] for p in self.evaluator.periods} for name in self.evaluator.models}

        for period_name, (start, end) in self.evaluator.periods.items():
            period_mask = (pd.to_datetime(self.evaluator.forecast_dates) >= start) & \
                          (pd.to_datetime(self.evaluator.forecast_dates) <= end)
            
            for model_name in self.evaluator.models:
                print(f"-> {period_name} 기간, {model_name} 모델 리스크 계산 중...")
                for c0 in tqdm(exposure_constraints, leave=False, desc=f"   Exposure (c0)"):
                    daily_portfolio_variances = []
                    
                    for i, is_in_period in enumerate(period_mask):
                        if not is_in_period: continue
                        
                        # 예측된 연속 변동성 + 전날의 점프 변동성 = 총 변동성
                        continuous_vol = self.evaluator.forecasts[model_name][i]
                        jump_vol = self.data['jump_vol_data'][self.data['dates'].get_loc(self.evaluator.forecast_dates[i]) - 1]
                        total_vol_forecast = project_psd(continuous_vol + jump_vol)
                        
                        # 최적 포트폴리오 가중치 계산 (논문 5.2절)
                        w = self._optimize_portfolio(total_vol_forecast, c0)
                        
                        if w is not None:
                            realized_return_vec = self.daily_returns.iloc[i].values
                            portfolio_return = w @ realized_return_vec
                            daily_portfolio_variances.append(portfolio_return**2)
                    
                    # 연율화된 리스크 계산
                    if daily_portfolio_variances:
                        annualized_risk = np.sqrt(np.mean(daily_portfolio_variances) * 252) * 100
                        portfolio_risks[model_name][period_name].append(annualized_risk)
                    else:
                        portfolio_risks[model_name][period_name].append(np.nan)
        
        self._plot_portfolio_risks(portfolio_risks, exposure_constraints)

    def _optimize_portfolio(self, cov_matrix: np.ndarray, c0: float) -> Union[np.ndarray, None]:
        """cvxpy를 사용하여 최소 분산 포트폴리오 문제를 해결합니다."""
        n_assets = cov_matrix.shape[0]
        w = cp.Variable(n_assets)
        objective = cp.Minimize(cp.quad_form(w, cov_matrix))
        constraints = [cp.sum(w) == 1, cp.norm(w, 1) <= c0]
        prob = cp.Problem(objective, constraints)
        
        try:
            prob.solve(solver=cp.SCS, verbose=False)
            return w.value if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] else None
        except cp.error.SolverError:
            return None

    def _plot_portfolio_risks(self, risks: Dict, constraints: np.ndarray):
        """포트폴리오 리스크 분석 결과를 논문 Figure 5와 같이 시각화합니다."""
        print("\n포트폴리오 리스크 분석 결과를 시각화합니다.")
        fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
        
        styles = {
            'POET-PRVM': {'c': 'red', 'marker': '*', 'ls': '--'},
            'OLS':       {'c': 'blue', 'marker': '+', 'ls': '--'},
            'LASSO':     {'c': 'green', 'marker': 'o', 'ls': '--', 'mfc': 'none'},
            'H-LASSO':   {'c': 'black', 'marker': 'o', 'ls': '-'}
        }
        
        for i, period_name in enumerate(self.evaluator.periods):
            ax = axes[i]
            for model_name, style in styles.items():
                ax.plot(constraints, risks[model_name][period_name], label=model_name, **style)
            
            title_period = period_name.split('(')[1][:-1]
            ax.set_title(f'Portfolio Risk ({title_period})', fontsize=14)
            ax.set_xlabel('Exposure Constraint (c0)', fontsize=12)
            if i == 0: ax.set_ylabel('Annualized Risk (%)', fontsize=12)
            ax.legend()
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        plt.suptitle('포트폴리오 최적화 결과 (논문 Figure 5 재현)', fontsize=18, y=1.02)
        plt.tight_layout()
        plt.show()

# %% --- 6. 메인 실행 스크립트 ---

if __name__ == '__main__':
    # --- 설정 (Configuration) ---
    try:
        # 스크립트 실행 위치를 기준으로 파일 경로 설정
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # Spyder 등에서 __file__ 변수가 없을 경우 현재 작업 디렉토리 사용
        script_dir = os.getcwd()

    config = {
        # 파일 경로
        "prvm_file_path": os.path.join(script_dir, 'prvm_0731_final_corrected.csv'),
        "gics_file_path": os.path.join(script_dir, 'gicslist.csv'),
        "log_price_file_path": os.path.join(script_dir, 'DataY.csv'),
        
        # 모델 파라미터 (논문 5.2절과 동일하게 설정)
        "num_factors": 3,
        "h_lag": 1,
        "l_window": 22,
        
        # 평가 파라미터
        "in_sample_size": 251, # 1년 (trading days)
        "eval_start_date": '2018-01-01',
        "periods": {
            "Period 1 (2018-2019)": ('2018-01-01', '2019-12-31'),
            "Period 2 (2018)": ('2018-01-01', '2018-12-31'),
            "Period 3 (2019)": ('2019-01-01', '2019-12-31'),
        }
    }

    # --- 데이터 준비 (Data Preparation) ---
    data_loader = DataLoader(
        prvm_path=config['prvm_file_path'],
        gics_path=config['gics_file_path'],
        log_price_path=config['log_price_file_path']
    )
    all_data = data_loader.load_all_data()

    preprocessor = Preprocessor(
        num_factors=config['num_factors'],
        gics_data=all_data['gics_data']
    )
    all_data['idio_data'] = preprocessor.get_idiosyncratic_matrices(all_data['prvm_data'])

    # --- 모델 초기화 (Model Initialization) ---
    models_to_evaluate = [
        POETPRVMEstimator(config['num_factors'], all_data['gics_data']),
        FIVAR_OLS_Estimator(config['num_factors'], config['h_lag'], config['l_window']),
        FIVAR_LASSO_Estimator(config['num_factors'], config['h_lag'], config['l_window']),
        FIVAR_HLASSO_Estimator(config['num_factors'], config['h_lag'], config['l_window'])
    ]

    # --- 평가 실행 (Run Evaluation) ---
    evaluator = Evaluator(models_to_evaluate, config['periods'])
    evaluator.run_evaluation(all_data, config)

    # --- 포트폴리오 분석 (Portfolio Analysis) ---
    portfolio_analyzer = PortfolioOptimizer(evaluator, {
        'log_price_path': config['log_price_file_path'],
        'jump_vol_data': all_data['jump_vol_data'],
        'dates': all_data['dates'],
        'tickers': all_data['tickers']
    }, config)
    portfolio_analyzer.run_portfolio_analysis()
