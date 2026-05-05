import numpy as np
import pandas as pd
from numpy.linalg import eigh, inv, norm
from tqdm import tqdm
from sklearn.linear_model import LinearRegression, Lasso, HuberRegressor
from joblib import Parallel, delayed
import warnings

# 경고 메시지 무시
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)


# --- 1. 데이터 준비 및 헬퍼 함수 ---

def load_csv_data(file_path):
    """long-format의 .csv 파일에서 PRVM 데이터를 불러와 3D 텐서로 변환합니다."""
    print(f"'{file_path}' 파일에서 CSV 데이터를 불러옵니다...")
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

        for t, date in enumerate(tqdm(unique_dates, desc="데이터 변환 중")):
            daily_data = df[df['date'] == date]
            matrix = np.zeros((num_assets, num_assets))
            matrix[daily_data['i'], daily_data['j']] = daily_data['value']
            matrix[daily_data['j'], daily_data['i']] = daily_data['value']
            prvm_data[t] = matrix

    except Exception as e:
        print(f"파일을 읽는 중 오류가 발생했습니다: {e}")
        return None, None, None

    dates = pd.to_datetime(unique_dates)
    print(f"데이터 로딩 및 변환 완료: {prvm_data.shape} 형태")
    print(f"날짜 범위: {dates.min().date()} ~ {dates.max().date()}")
    return prvm_data, dates, tickers

def load_gics_data(gics_file_path, tickers):
    """GICS 분류가 담긴 CSV 파일을 불러옵니다."""
    print(f"'{gics_file_path}' 파일에서 GICS 데이터를 불러옵니다...")
    try:
        gics_df = pd.read_csv(gics_file_path, header=None)
        
        if gics_df.shape[1] < 2:
            print(f"경고: '{gics_file_path}'에 1개의 열만 있습니다. 자산 순서와 일치하는 섹터 목록으로 간주합니다.")
            if len(gics_df) == len(tickers) + 1:
                gics_sectors = gics_df.iloc[1:, 0].astype(str).to_numpy()
            else:
                gics_sectors = gics_df.iloc[:, 0].astype(str).to_numpy()

            if len(gics_sectors) != len(tickers):
                print(f"오류: 처리 후 GICS 데이터의 수({len(gics_sectors)})와 자산의 수({len(tickers)})가 일치하지 않습니다.")
                return None
            return gics_sectors

        print("파일에서 티커와 GICS 섹터를 매핑합니다 (첫 번째 열: 티커, 두 번째 열: 섹터).")
        gics_map = pd.Series(gics_df.iloc[:, 1].astype(str).values, index=gics_df.iloc[:, 0].astype(str).values)
        gics_sectors = gics_map.reindex(tickers).to_numpy()
        
        if pd.isna(gics_sectors).any():
            missing_tickers_count = pd.isna(gics_sectors).sum()
            print(f"경고: {missing_tickers_count}개의 티커에 대한 GICS 정보를 찾을 수 없습니다. 'Unknown'으로 대체합니다.")
            gics_sectors = pd.Series(gics_sectors).fillna('Unknown').to_numpy()
            
        return gics_sectors
        
    except Exception as e:
        print(f"GICS 파일을 읽는 중 오류가 발생했습니다: {e}")
        return None

def project_psd(matrix):
    """행렬을 양의 준정부호(Positive Semi-Definite) 행렬로 변환합니다."""
    eigenvalues, eigenvectors = eigh(matrix)
    eigenvalues[eigenvalues < 1e-7] = 1e-7
    psd_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    return (psd_matrix + psd_matrix.T) / 2

def vech(matrix):
    """행렬의 하삼각행렬 부분을 벡터로 변환합니다."""
    return matrix[np.tril_indices(matrix.shape[0], k=-1)]

def unvech(vector, p):
    """벡터를 대칭 행렬로 변환합니다."""
    matrix = np.zeros((p, p))
    matrix[np.tril_indices(p, k=-1)] = vector
    matrix += matrix.T
    return matrix

# --- 2. 성능 평가 함수 ---

def calculate_mspe(forecasts, ground_truths):
    """MSPE (Mean Squared Prediction Error)를 계산합니다."""
    if not forecasts: return np.nan
    errors = [norm(forecasts[i] - ground_truths[i], 'fro')**2 for i in range(len(forecasts))]
    return np.mean(errors)

def calculate_qlike(forecasts, ground_truths):
    """QLIKE 손실 함수를 계산합니다."""
    if not forecasts: return np.nan
    qlike_vals = []
    for i in range(len(forecasts)):
        try:
            forecast_psd = project_psd(forecasts[i])
            forecast_reg = forecast_psd + np.eye(forecast_psd.shape[0]) * 1e-7
            sign, log_det_val = np.linalg.slogdet(forecast_reg)
            if sign <= 0: continue
            trace_val = np.trace(inv(forecast_reg) @ ground_truths[i])
            qlike_vals.append(log_det_val + trace_val)
        except np.linalg.LinAlgError:
            continue
    return np.mean(qlike_vals) if qlike_vals else np.nan

# --- 3. 변동성 예측 모델 구현 ---

def calculate_poet_prvm(prvm_matrix, gics_sectors, num_factors=3):
    """단일 PRVM 행렬로부터 POET-PRVM을 계산합니다. (Ground Truth 및 벤치마크 예측용)"""
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
    np.fill_diagonal(same_sector_mask, 1)
    idiosyncratic_part_thresholded = idiosyncratic_part_raw * same_sector_mask
    
    poet_prvm = factor_part + idiosyncratic_part_thresholded
    return project_psd(poet_prvm)

def get_eigencomponents(prvm_window, gics_sectors, num_factors=3):
    """주어진 기간의 PRVM 데이터로부터 요인/특이 고유벡터를 추정합니다."""
    p = prvm_window[0].shape[0]
    avg_prvm = np.mean(prvm_window, axis=0)
    
    _, q_F_all = eigh(avg_prvm)
    q_F = q_F_all[:, -num_factors:]
    
    factor_part = (q_F @ q_F.T) @ avg_prvm @ (q_F @ q_F.T)
    idio_part_raw = avg_prvm - factor_part
    
    same_sector_mask = (gics_sectors.reshape(-1, 1) == gics_sectors)
    np.fill_diagonal(same_sector_mask, 1)
    avg_idio_thresholded = idio_part_raw * same_sector_mask
    
    _, q_I = eigh(avg_idio_thresholded)
    return q_F, q_I

def get_fivar_eigenvalues(prvm_all, thresholded_idio_matrices, q_F, q_I):
    """FIVAR 모델 학습을 위한 일별 요인/특이 고유값을 계산합니다."""
    num_days, p, _ = prvm_all.shape
    num_factors = q_F.shape[1]
    xi_all = np.zeros((num_days, p + num_factors))

    for t in range(num_days):
        xi_all[t, :num_factors] = np.diag(q_F.T @ prvm_all[t] @ q_F) / p
        xi_all[t, num_factors:] = np.diag(q_I.T @ thresholded_idio_matrices[t] @ q_I)
        
    return xi_all

# *** NEW: 병렬 처리를 위한 헬퍼 함수 ***
def _process_day_for_thresholding(prvm_matrix, q_F, same_sector_mask):
    """단일 날짜의 PRVM에 대해 임계값 처리된 특이 공분산 행렬을 계산합니다."""
    factor_part = (q_F @ q_F.T) @ prvm_matrix @ (q_F @ q_F.T)
    idio_part_raw = prvm_matrix - factor_part
    return idio_part_raw * same_sector_mask

def _huber_lasso_regression(X, y, alpha=0.01, tau_scale=1.345, max_iter=100, tol=1e-4):
    """IRLS를 사용하여 Huber 손실을 가지는 LASSO 회귀를 풉니다."""
    lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=2000, tol=tol)
    lasso.fit(X, y)
    beta = lasso.coef_
    
    for _ in range(max_iter):
        beta_old = beta.copy()
        residuals = y - X @ beta
        median_res = np.median(residuals)
        scale = np.median(np.abs(residuals - median_res)) / 0.6749
        if scale < 1e-7: scale = 1e-7
        tau = tau_scale * scale
        weights = np.ones_like(residuals)
        abs_residuals = np.abs(residuals)
        mask = abs_residuals > tau
        weights[mask] = tau / abs_residuals[mask]
        sqrt_weights = np.sqrt(weights)
        X_w = X * sqrt_weights[:, np.newaxis]
        y_w = y * sqrt_weights
        lasso_w = Lasso(alpha=alpha, fit_intercept=False, max_iter=2000, tol=tol)
        lasso_w.fit(X_w, y_w)
        beta = lasso_w.coef_
        if norm(beta - beta_old) / (norm(beta_old) + 1e-7) < tol:
            break
    return beta

def forecast_var_model(xi_in_sample, q_F, q_I, model_type='lasso', h=1):
    """OLS, LASSO, H-LASSO 예측을 위한 VAR 모델"""
    num_days, p_plus_r = xi_in_sample.shape
    p = q_I.shape[0]
    num_factors = q_F.shape[1]

    xi_mean = np.mean(xi_in_sample, axis=0)
    xi_std = np.std(xi_in_sample, axis=0)
    xi_std[xi_std < 1e-8] = 1.0
    xi_in_sample_std = (xi_in_sample - xi_mean) / xi_std

    X_list, y_list = [], []
    for t in range(h, num_days):
        X_list.append(np.concatenate(([1.0], xi_in_sample_std[t-h:t, :].flatten())))
        y_list.append(xi_iqwen_sample_stdn_sample_std[t, :])
    X_train, y_train = np.array(X_list), np.array(y_list)
    X_pred_input = np.concatenate(([1.0], xi_in_sample_std[-h:, :].flatten())).reshape(1, -1)
    
    xi_forecast = np.zeros(p_plus_r)
    
    factor_cols = np.concatenate(([0], np.arange(1, h * num_factors + 1))).astype(int)
    X_train_factor = X_train[:, factor_cols]
    X_pred_input_factor = X_pred_input[:, factor_cols]

    if model_type == 'h-lasso':
        X_train_factor_trunc = X_train_factor.copy()
        for col in range(1, X_train_factor_trunc.shape[1]):
            lower, upper = np.percentile(X_train_factor_trunc[:, col], [2.5, 97.5])
            X_train_factor_trunc[:, col] = np.clip(X_train_factor_trunc[:, col], lower, upper)
        X_pred_input_factor_trunc = X_pred_input_factor.copy()
        for col in range(1, X_pred_input_factor.shape[1]):
            lower, upper = np.percentile(X_train_factor[:, col], [2.5, 97.5])
            X_pred_input_factor_trunc[0, col] = np.clip(X_pred_input_factor_trunc[0, col], lower, upper)
        huber = HuberRegressor(fit_intercept=False)
        for i in range(num_factors):
            huber.fit(X_train_factor_trunc, y_train[:, i])
            pred_std = huber.predict(X_pred_input_factor_trunc)[0]
            xi_forecast[i] = (pred_std * xi_std[i]) + xi_mean[i]
    else:
        ols = LinearRegression(fit_intercept=False)
        for i in range(num_factors):
            ols.fit(X_train_factor, y_train[:, i])
            pred_std = ols.predict(X_pred_input_factor)[0]
            xi_forecast[i] = (pred_std * xi_std[i]) + xi_mean[i]

    if model_type == 'ols':
        xi_forecast[num_factors:] = np.mean(xi_in_sample[-22:, num_factors:], axis=0)
    elif model_type == 'lasso':
        lasso = Lasso(alpha=0.01, max_iter=2000, fit_intercept=False)
        for i in range(num_factors, p_plus_r):
            lasso.fit(X_train, y_train[:, i])
            pred_std = lasso.predict(X_pred_input)[0]
            xi_forecast[i] = (pred_std * xi_std[i]) + xi_mean[i]
    elif model_type == 'h-lasso':
        X_train_trunc = X_train.copy()
        for col in range(1, X_train.shape[1]):
            lower, upper = np.percentile(X_train_trunc[:, col], [2.5, 97.5])
            X_train_trunc[:, col] = np.clip(X_train_trunc[:, col], lower, upper)
        X_pred_input_trunc = X_pred_input.copy()
        for col in range(1, X_pred_input.shape[1]):
            lower, upper = np.percentile(X_train[:, col], [2.5, 97.5])
            X_pred_input_trunc[0, col] = np.clip(X_pred_input_trunc[0, col], lower, upper)
        for i in range(num_factors, p_plus_r):
            beta_i = _huber_lasso_regression(X_train_trunc, y_train[:, i], alpha=0.01)
            pred_std = (X_pred_input_trunc @ beta_i)[0]
            xi_forecast[i] = (pred_std * xi_std[i]) + xi_mean[i]

    xi_forecast[xi_forecast < 1e-7] = 1e-7
    factor_forecast = p * (q_F @ np.diag(xi_forecast[:num_factors]) @ q_F.T)
    idio_forecast = q_I @ np.diag(xi_forecast[num_factors:]) @ q_I.T
    return project_psd(factor_forecast + idio_forecast)

def forecast_har_drd(prvm_in_sample):
    """HAR-DRD 모델 예측"""
    num_days, p, _ = prvm_in_sample.shape
    D_list, R_vech_list = [], []
    for t in range(num_days):
        prvm_t = project_psd(prvm_in_sample[t])
        D_t_diag = np.diag(prvm_t).copy()
        D_t_diag[D_t_diag <= 0] = 1e-7
        D_list.append(D_t_diag)
        inv_sqrt_D = np.diag(1.0 / np.sqrt(D_t_diag))
        R_t = inv_sqrt_D @ prvm_t @ inv_sqrt_D
        R_vech_list.append(vech(R_t))
    D_ts, R_vech_ts = np.array(D_list), np.array(R_vech_list)
    
    def get_har_predictors(ts_data, t):
        day = ts_data[t-1]
        week = np.mean(ts_data[t-5:t], axis=0) if t >= 5 else ts_data[t-1]
        month = np.mean(ts_data[t-22:t], axis=0) if t >= 22 else ts_data[t-1]
        return np.concatenate([day, week, month])

    if num_days < 23:
        D_forecast = np.mean(D_ts, axis=0)
        D_forecast[D_forecast < 1e-7] = 1e-7
        R_forecast_raw = unvech(R_vech_ts[-1], p)
        np.fill_diagonal(R_forecast_raw, 1)
        R_forecast = project_psd(R_forecast_raw)
        sqrt_D_forecast = np.diag(np.sqrt(D_forecast))
        return sqrt_D_forecast @ R_forecast @ sqrt_D_forecast

    X_list, y_D_list, y_R_list = [], [], []
    for t in range(22, num_days):
        X_list.append(get_har_predictors(D_ts, t))
        y_D_list.append(D_ts[t])
        y_R_list.append(R_vech_ts[t])
    X_train, y_D_train, y_R_train = np.array(X_list), np.array(y_D_list), np.array(y_R_list)
    
    D_forecast = np.zeros(p)
    X_pred_input_har = get_har_predictors(D_ts, num_days).reshape(1, -1)
    for i in range(p):
        ols_d = LinearRegression(fit_intercept=True)
        ols_d.fit(X_train[:, [i, p+i, 2*p+i]], y_D_train[:, i])
        D_forecast[i] = ols_d.predict(X_pred_input_har[:, [i, p+i, 2*p+i]])[0]
    
    D_forecast[D_forecast < 1e-7] = 1e-7

    ols_r = LinearRegression(fit_intercept=True)
    ols_r.fit(X_train, y_R_train)
    R_vech_forecast = ols_r.predict(X_pred_input_har)[0]
    R_forecast_raw = unvech(R_vech_forecast, p)
    np.fill_diagonal(R_forecast_raw, 1)
    
    R_forecast_psd = project_psd(R_forecast_raw)
    diag_vals = np.diag(R_forecast_psd).copy()
    diag_vals[diag_vals <= 0] = 1e-7
    inv_diag_sqrt = np.diag(1.0 / np.sqrt(diag_vals))
    R_forecast = inv_diag_sqrt @ R_forecast_psd @ inv_diag_sqrt
    R_forecast = project_psd(R_forecast)
    
    sqrt_D_forecast = np.diag(np.sqrt(D_forecast))
    return sqrt_D_forecast @ R_forecast @ sqrt_D_forecast

def forecast_dcc_nl(daily_returns_in_sample):
    """DCC-NL 모델 예측 (논문에서는 벤치마크로 사용되었으나, 여기서는 구현되지 않음)"""
    return None

# --- 4. 메인 실행 로직 ---
def main():
    prvm_file_path = 'prvm_0731_final_corrected.csv' 
    gics_file_path = 'gicslist.csv'
    num_factors = 3
    in_sample_size = 251
    
    prvm_data_all, dates, tickers = load_csv_data(prvm_file_path)
    if prvm_data_all is None: return
    gics_data = load_gics_data(gics_file_path, tickers)
    if gics_data is None: return
    
    periods = {
        "Period 1 (2018-2019)": ('2018-01-01', '2019-12-31'),
        "Period 2 (2018)": ('2018-01-01', '2018-12-31'),
        "Period 3 (2019)": ('2019-01-01', '2019-12-31'),
    }
    
    models = ['POET-PRVM', 'OLS', 'LASSO', 'H-LASSO', 'HAR-DRD', 'DCC-NL']
    forecasts = {model: [] for model in models}
    ground_truths, forecast_dates = [], []

    start_date = pd.to_datetime('2018-01-01')
    start_idx = dates.searchsorted(start_date)
    if start_idx < in_sample_size:
        print(f"데이터가 충분하지 않아 {dates[in_sample_size]}부터 예측을 시작합니다.")
        start_idx = in_sample_size

    for t in tqdm(range(start_idx, len(dates)), desc="롤링 윈도우 예측 중"):
        in_sample_prvm = prvm_data_all[t - in_sample_size : t]
        
        ground_truth_matrix = prvm_data_all[t]
        ground_truths.append(calculate_poet_prvm(ground_truth_matrix, gics_data, num_factors))
        forecast_dates.append(dates[t])

        prvm_t_minus_1 = in_sample_prvm[-1]
        forecasts['POET-PRVM'].append(calculate_poet_prvm(prvm_t_minus_1, gics_data, num_factors))
        
        # --- FIVAR 모델들(OLS, LASSO, H-LASSO)을 위한 입력 데이터 준비 ---
        
        # 1. 최근 22일 데이터로 고유벡터(q_F, q_I) 추정
        q_F, q_I = get_eigencomponents(in_sample_prvm[-22:], gics_data, num_factors)
        
        # *** FIX: 병렬 처리를 사용하여 속도 향상 ***
        # 2. In-sample 기간 전체에 대해 일별 특이 공분산 행렬에 임계값 처리 적용
        same_sector_mask = (gics_data.reshape(-1, 1) == gics_data)
        np.fill_diagonal(same_sector_mask, 1)
        
        # joblib을 사용하여 병렬로 계산 (n_jobs=-1은 모든 CPU 코어 사용)
        thresholded_idio_matrices = Parallel(n_jobs=-1)(
            delayed(_process_day_for_thresholding)(in_sample_prvm[i], q_F, same_sector_mask)
            for i in range(len(in_sample_prvm))
        )
        
        # 3. 안정화된 고유값 시계열 생성
        xi_all_in_sample = get_fivar_eigenvalues(in_sample_prvm, thresholded_idio_matrices, q_F, q_I)
        
        # --- 모델별 예측 수행 ---
        forecasts['OLS'].append(forecast_var_model(xi_all_in_sample, q_F, q_I, model_type='ols'))
        forecasts['LASSO'].append(forecast_var_model(xi_all_in_sample, q_F, q_I, model_type='lasso'))
        forecasts['H-LASSO'].append(forecast_var_model(xi_all_in_sample, q_F, q_I, model_type='h-lasso'))
        
        forecasts['HAR-DRD'].append(forecast_har_drd(in_sample_prvm))
        
        forecasts['DCC-NL'].append(forecast_dcc_nl(None))

    # --- 결과 집계 ---
    final_results = []
    forecast_dates = pd.to_datetime(forecast_dates)
    for period_name, (start_date, end_date) in periods.items():
        period_mask = (forecast_dates >= start_date) & (forecast_dates <= end_date)
        period_truths = [ground_truths[i] for i, val in enumerate(period_mask) if val]
        
        for model in models:
            period_forecasts = [forecasts[model][i] for i, val in enumerate(period_mask) if val]
            valid_indices = [i for i, f in enumerate(period_forecasts) if f is not None]
            
            if not valid_indices:
                mspe_val, qlike_val = np.nan, np.nan
            else:
                valid_forecasts = [period_forecasts[i] for i in valid_indices]
                valid_truths = [period_truths[i] for i in valid_indices]
                mspe_val = calculate_mspe(valid_forecasts, valid_truths)
                qlike_val = calculate_qlike(valid_forecasts, valid_truths)
            
            final_results.append({
                "Period": period_name, "Model": model,
                "MSPE (x10^4)": mspe_val * 10**4 if not np.isnan(mspe_val) else np.nan,
                "QLIKE x 10^-3": qlike_val * 10**-3 if not np.isnan(qlike_val) else np.nan,
            })

    results_df = pd.DataFrame(final_results)
    pivot_df = results_df.pivot_table(index='Model', columns='Period', values=['MSPE (x10^4)', 'QLIKE x 10^-3'])
    
    print("\n--- 최종 평가 결과 (논문 표 형식) ---")
    model_order = ['POET-PRVM', 'OLS', 'LASSO', 'H-LASSO', 'DCC-NL', 'HAR-DRD']
    
    output_rows = []
    for model in model_order:
        row_mspe = {'Model': model, 'Metric': 'MSPE x 10^4'}
        row_qlike = {'Model': '', 'Metric': 'QLIKE x 10^-3'}
        for period in periods:
            try:
                row_mspe[period] = pivot_df.loc[model, ('MSPE (x10^4)', period)]
                row_qlike[period] = pivot_df.loc[model, ('QLIKE x 10^-3', period)]
            except KeyError:
                row_mspe[period] = np.nan
                row_qlike[period] = np.nan
        output_rows.append(row_mspe)
        output_rows.append(row_qlike)

    summary_table = pd.DataFrame(output_rows).set_index(['Model', 'Metric'])
    print(summary_table.to_string(float_format="%.3f"))

if __name__ == '__main__':
    main()
