import pandas as pd
import numpy as np
import os
from scipy.linalg import eigh
from scipy.stats import kurtosis
from sklearn.linear_model import HuberRegressor
from sklearn.exceptions import ConvergenceWarning
import warnings
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from typing import Dict, List, Tuple

# --- 시각화를 위한 한글 폰트 설정 ---
try:
    font_path = None
    if os.name == 'nt': font_path = fm.findfont(fm.FontProperties(family='Malgun Gothic'))
    elif os.name == 'posix': font_path = fm.findfont(fm.FontProperties(family='AppleGothic')) or fm.findfont(fm.FontProperties(family='NanumGothic'))
    if font_path: plt.rc('font', family=fm.FontProperties(fname=font_path).get_name())
    else: print("경고: 한글 폰트를 찾을 수 없습니다. 그래프의 한글이 깨질 수 있습니다.")
except Exception as e:
    print(f"폰트 설정 중 오류 발생: {e}")

warnings.filterwarnings("ignore", category=ConvergenceWarning)

def load_prvm_from_csv(file_path: str) -> Tuple[Dict[object, np.ndarray], List[str]]:
    """
    'date', 'ticker_i', 'ticker_j', 'value' 형식의 CSV 파일을 읽어,
    {date: matrix} 딕셔너리와 티커 리스트를 반환합니다.
    """
    print(f"'{file_path}'에서 미리 계산된 PRVM 데이터를 로딩합니다...")
    df = pd.read_csv(file_path, parse_dates=['date'])
    tickers = sorted(df['ticker_i'].unique())
    p = len(tickers)
    print(f"총 {p}개의 고유 티커를 발견했습니다.")
    prvm_matrices = {}
    for date, group in df.groupby('date'):
        matrix_df = group.pivot_table(index='ticker_i', columns='ticker_j', values='value')
        reindexed_df = matrix_df.reindex(index=tickers, columns=tickers)
        prvm_matrices[date.date()] = reindexed_df.fillna(0).to_numpy()
    print(f"총 {len(prvm_matrices)}일치의 PRVM 데이터를 로딩했습니다.")
    return prvm_matrices, tickers


class FIVARModel:
    """
    논문 'Factor and Idiosyncratic VAR Volatility Matrix Models...'의
    Algorithm 1을 충실하게 구현한 클래스.
    """
    def __init__(self, r: int, h: int = 1, l: int = 22):
        self.r = r; self.h = h; self.l = l
        self.p = None; self.daily_prvm_matrices = None; self.factor_eigenvectors = None
        self.idio_eigenvectors = None; self.all_idio_matrices = None; self.eigenvalue_df = None
        self.factor_regressors = None; self.idio_regressors = None
        self.last_eigenvalues = None; self.predicted_vol_matrix = None
        # --- [추가] 각 자산별 최적 c_eta 저장 ---
        self.best_c_etas_for_assets = None

    def fit(self, prvm_matrices_window: dict, sector_data: pd.DataFrame):
        """
        [수정] 이제 이 메소드는 전체 데이터가 아닌, 특정 기간(window)의 데이터로 학습합니다.
        """
        self.daily_prvm_matrices = prvm_matrices_window
        self.p = next(iter(self.daily_prvm_matrices.values())).shape[0]
        
        valid_dates_for_fit = sorted(list(self.daily_prvm_matrices.keys()))
        if len(valid_dates_for_fit) < self.l + self.h:
             raise ValueError(f"모델 적합에 필요한 유효 데이터 일수가 부족합니다. (필요: {self.l + self.h}, 실제: {len(valid_dates_for_fit)})")
        dates = valid_dates_for_fit
        
        # Step 2-5: Eigen-decomposition and eigenvalue calculation
        last_l_prvm = [self.daily_prvm_matrices[d] for d in dates[-self.l:]]
        avg_prvm_matrix = np.mean(last_l_prvm, axis=0)
        _, self.factor_eigenvectors = eigh(avg_prvm_matrix, subset_by_index=[self.p - self.r, self.p - 1])
        self.factor_eigenvectors = self.factor_eigenvectors[:, ::-1]
        
        factor_eigenvalues_list = []
        for d in dates:
            prvm = self.daily_prvm_matrices[d]
            factor_eigenvalues = np.diag(self.factor_eigenvectors.T @ prvm @ self.factor_eigenvectors)
            factor_eigenvalues_list.append(factor_eigenvalues)

        idio_matrices_input = []
        for i, d in enumerate(dates):
            prvm = self.daily_prvm_matrices[d]
            factor_component = self.factor_eigenvectors @ np.diag(factor_eigenvalues_list[i]) @ self.factor_eigenvectors.T
            idio_matrix_input = prvm - factor_component
            idio_matrices_input.append(idio_matrix_input)
            
        sectors = sector_data['Sector'].values
        gics_mask = (sectors[:, None] == sectors)
        self.all_idio_matrices = []
        for idio_matrix_input in idio_matrices_input:
            diag_idio = np.diag(np.diag(idio_matrix_input))
            off_diag_idio = idio_matrix_input - diag_idio
            idio_matrix_final = diag_idio + (off_diag_idio * gics_mask)
            self.all_idio_matrices.append(idio_matrix_final)

        last_l_idio = self.all_idio_matrices[-self.l:]
        avg_idio_matrix = np.mean(last_l_idio, axis=0)
        _, self.idio_eigenvectors = eigh(avg_idio_matrix)
        self.idio_eigenvectors = self.idio_eigenvectors[:, ::-1]
        
        idio_eigenvalues_list = []
        for idio_matrix in self.all_idio_matrices:
            idio_eigenvalues = np.diag(self.idio_eigenvectors.T @ idio_matrix @ self.idio_eigenvectors)
            idio_eigenvalues_list.append(idio_eigenvalues)
        
        all_eigenvalues = [np.concatenate([f, i]) for f, i in zip(factor_eigenvalues_list, idio_eigenvalues_list)]
        self.eigenvalue_df = pd.DataFrame(all_eigenvalues, index=dates)

        # Regression part
        y = self.eigenvalue_df.iloc[self.h:]; n_regression = len(y)
        X_list = [self.eigenvalue_df.shift(i + 1).iloc[self.h:] for i in range(self.h)]
        X = pd.concat(X_list, axis=1)
        params = {'c_F1': 4.0, 'c_F2': 0.25, 'c_I1': 4.0, 'c_I2': 4.0}
        sigma_F = np.std(self.eigenvalue_df.iloc[:, :self.r].values)
        omega_F_cap = params['c_F1'] * sigma_F * (n_regression / np.log(self.p))**0.25
        tau_F_cap = params['c_F2'] * sigma_F * (n_regression / np.log(self.p))**0.25
        omega_I_cap = params['c_I1'] * (n_regression / np.log(self.p))**0.25
        tau_I_cap = params['c_I2'] * (n_regression / np.log(self.p))**0.25
        truncate_factor = lambda data: np.clip(data, -omega_F_cap, omega_F_cap)
        truncate_idio = lambda data: np.clip(data, -omega_I_cap, omega_I_cap)
        
        # Step 6: Factor regression
        self.factor_regressors = [HuberRegressor(fit_intercept=True) for _ in range(self.r)]
        for i in range(self.r):
            y_i = y.iloc[:, i]; X_factor = X.iloc[:, :self.r * self.h]; X_trunc = truncate_factor(X_factor)
            tau = tau_F_cap; epsilon_skl = self.factor_regressors[i].epsilon
            if tau < epsilon_skl:
                scale_factor = epsilon_skl / tau; y_scaled = y_i * scale_factor; X_scaled = X_trunc * scale_factor
                self.factor_regressors[i].fit(X_scaled, y_scaled)
                self.factor_regressors[i].intercept_ /= scale_factor
            else:
                self.factor_regressors[i].epsilon = tau
                self.factor_regressors[i].fit(X_trunc, y_i)
        
        # Step 7: Idiosyncratic regression with asset-specific c_eta optimization
        self.idio_regressors = [None] * self.p
        self.best_c_etas_for_assets = [None] * self.p # 결과 저장을 위해 초기화
        X_trunc_idio = truncate_idio(X)
        
        for i in range(self.p):
            y_i = y.iloc[:, self.r + i]
            c_eta_candidates = np.linspace(0.1, 10.0, 100)
            best_bic = np.inf
            best_c_eta_for_asset = None
            best_regressor_for_asset = None
            
            for c_eta in c_eta_candidates:
                final_eta_I = c_eta * np.sqrt(np.log(self.p) / n_regression)
                temp_regressor = HuberRegressor(fit_intercept=True, alpha=final_eta_I)
                tau = tau_I_cap
                epsilon_skl = temp_regressor.epsilon
                
                if tau < epsilon_skl:
                    scale_factor = epsilon_skl / tau
                    y_scaled = y_i * scale_factor; X_scaled = X_trunc_idio * scale_factor
                    temp_regressor.fit(X_scaled, y_scaled)
                else:
                    temp_regressor.epsilon = tau
                    temp_regressor.fit(X_trunc_idio, y_i)

                y_pred = temp_regressor.predict(X_trunc_idio)
                rss = np.sum((y_i - y_pred)**2)
                num_params = np.sum(temp_regressor.coef_ != 0)
                if temp_regressor.fit_intercept: num_params += 1
                
                n_obs = n_regression
                if rss <= 0: bic = np.inf
                else: bic = num_params * np.log(n_obs) + n_obs * np.log(rss / n_obs)

                if bic < best_bic:
                    best_bic = bic
                    best_c_eta_for_asset = c_eta
                    best_regressor_for_asset = temp_regressor

            self.idio_regressors[i] = best_regressor_for_asset
            self.best_c_etas_for_assets[i] = best_c_eta_for_asset
        
        self.last_eigenvalues = self.eigenvalue_df.iloc[-self.h:].to_numpy()

    def predict(self) -> np.ndarray:
        if self.last_eigenvalues is None: raise RuntimeError("모델이 적합되지 않았습니다.")
        
        n_regression = len(self.eigenvalue_df.iloc[self.h:])
        sigma_F = np.std(self.eigenvalue_df.iloc[:, :self.r].values)
        params = {'c_F1': 4.0, 'c_F2': 0.25, 'c_I1': 4.0, 'c_I2': 4.0}
        omega_F_cap = params['c_F1'] * sigma_F * (n_regression / np.log(self.p))**0.25
        omega_I_cap = params['c_I1'] * (n_regression / np.log(self.p))**0.25
        truncate_factor = lambda data: np.clip(data, -omega_F_cap, omega_F_cap)
        truncate_idio = lambda data: np.clip(data, -omega_I_cap, omega_I_cap)

        X_pred = self.last_eigenvalues.flatten().reshape(1, -1)
        X_pred_factor_trunc = truncate_factor(X_pred[:, :self.r * self.h])
        X_pred_idio_trunc = truncate_idio(X_pred)
        
        predicted_eigenvalues = np.zeros(self.r + self.p)
        for i in range(self.r): predicted_eigenvalues[i] = self.factor_regressors[i].predict(X_pred_factor_trunc)[0]
        for i in range(self.p): predicted_eigenvalues[i+self.r] = self.idio_regressors[i].predict(X_pred_idio_trunc)[0]
        predicted_eigenvalues[predicted_eigenvalues < 0] = 0
        
        pred_factor_evals, pred_idio_evals = predicted_eigenvalues[:self.r], predicted_eigenvalues[self.r:]
        psi_hat = self.factor_eigenvectors @ np.diag(pred_factor_evals) @ self.factor_eigenvectors.T
        sigma_hat = self.idio_eigenvectors @ np.diag(pred_idio_evals) @ self.idio_eigenvectors.T
        self.predicted_vol_matrix = psi_hat + sigma_hat
        return self.predicted_vol_matrix


if __name__ == '__main__':
    # --- 1. 초기 설정 ---
    prvm_csv_path = 'prvm.csv'
    sector_csv_path = 'gicslist.csv'
    
    # 롤링 윈도우 파라미터
    IN_SAMPLE_WINDOW_SIZE = 251 # 논문에서 사용된 인-샘플 기간 n
    R = 3  # Factor 수
    H = 1  # 예측 시차
    L = 22 # 평균 행렬 계산에 사용될 기간

    # --- 2. 데이터 로딩 ---
    if not os.path.exists(prvm_csv_path):
        print(f"오류: '{prvm_csv_path}' 파일을 찾을 수 없습니다."); exit()
    daily_prvm_matrices, tickers = load_prvm_from_csv(prvm_csv_path)
    
    if not os.path.exists(sector_csv_path):
        print(f"오류: '{sector_csv_path}' 파일을 찾을 수 없습니다."); exit()
    gics_raw_data = pd.read_csv(sector_csv_path, header=None)
    sector_codes = gics_raw_data.iloc[1:, 0].values
    if len(tickers) != len(sector_codes):
        raise ValueError(f"PRVM의 고유 티커 수({len(tickers)})와 GICS 데이터 수({len(sector_codes)})가 일치하지 않습니다.")
    sector_data = pd.DataFrame({'Ticker': tickers, 'Sector': sector_codes})
    print("GICS 섹터 데이터 준비 완료.")

    # --- 3. 롤링 예측 실행 ---
    all_sorted_dates = sorted(daily_prvm_matrices.keys())
    
    # 결과를 저장할 딕셔너리
    all_predicted_matrices = {}
    all_optimal_c_etas = {}

    print(f"\n--- 롤링 예측을 시작합니다 (총 {len(all_sorted_dates) - IN_SAMPLE_WINDOW_SIZE}일 예측) ---")
    
    # 롤링 윈도우 루프
    for i in range(IN_SAMPLE_WINDOW_SIZE, len(all_sorted_dates)):
        prediction_date = all_sorted_dates[i]
        window_start_date = all_sorted_dates[i - IN_SAMPLE_WINDOW_SIZE]
        window_end_date = all_sorted_dates[i - 1]

        print(f"\n[{prediction_date}] 예측 중...")
        print(f"  - 학습 기간: {window_start_date} ~ {window_end_date} ({IN_SAMPLE_WINDOW_SIZE}일)")

        # 현재 윈도우에 해당하는 데이터 슬라이싱
        window_prvm_matrices = {
            date: matrix for date, matrix in daily_prvm_matrices.items() 
            if window_start_date <= date <= window_end_date
        }

        # 현재 윈도우에 대한 모델 생성 및 학습
        fivar_model_window = FIVARModel(r=R, h=H, l=L)
        fivar_model_window.fit(window_prvm_matrices, sector_data)
        
        # 예측 수행
        predicted_matrix = fivar_model_window.predict()
        
        # 결과 저장
        all_predicted_matrices[prediction_date] = predicted_matrix
        all_optimal_c_etas[prediction_date] = fivar_model_window.best_c_etas_for_assets
        
        print(f"  - [{prediction_date}] 예측 완료. 자산별 최적 c_eta (상위 5개): {np.round(fivar_model_window.best_c_etas_for_assets[:5], 2)}")

    # --- 4. 최종 결과 확인 ---
    print("\n--- 롤링 예측 완료 ---")
    if all_predicted_matrices:
        last_prediction_date = list(all_predicted_matrices.keys())[-1]
        last_predicted_matrix = all_predicted_matrices[last_prediction_date]
        last_c_etas = all_optimal_c_etas[last_prediction_date]

        print(f"\n마지막 예측일: {last_prediction_date}")
        print("마지막 예측 변동성 행렬 (상위 5x5):")
        print(pd.DataFrame(last_predicted_matrix, index=tickers, columns=tickers).iloc[:5, :5])
        
        print("\n마지막 예측에 사용된 자산별 최적 c_eta 값:")
        c_eta_df = pd.DataFrame({'Ticker': tickers, 'Optimal_c_eta': last_c_etas})
        print(c_eta_df.head(10))
    else:
        print("예측이 수행되지 않았습니다. 데이터 기간을 확인해주세요.")
