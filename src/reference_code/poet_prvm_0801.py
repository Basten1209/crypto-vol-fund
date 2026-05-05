import numpy as np
import pandas as pd
from numpy.linalg import eigh, inv, norm
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- Helper function for saving data ---
def save_matrices_to_csv(matrices, dates, tickers, filename):
    """
    Converts a 3D matrix array (T, p, p) to a long-format DataFrame and saves it as a CSV.
    
    Args:
        matrices (np.array): The 3D matrix array to save (T, p, p).
        dates (pd.DatetimeIndex): The date index (length T).
        tickers (list): The list of asset tickers (length p).
        filename (str): The name of the CSV file to save.
    """
    print(f"Saving data to '{filename}'...")
    num_days, num_assets, _ = matrices.shape
    
    records = []
    
    for t in tqdm(range(num_days), desc=f"Converting '{filename}'"):
        date = dates[t]
        matrix = matrices[t]
        for i in range(num_assets):
            for j in range(num_assets):
                # Optimize file size by saving only the lower triangle for symmetric matrices
                if i >= j:
                    records.append({
                        "date": date.strftime('%Y-%m-%d'),
                        "ticker_i": tickers[i],
                        "ticker_j": tickers[j],
                        "value": matrix[i, j]
                    })
    
    df_to_save = pd.DataFrame(records)
    df_to_save.to_csv(filename, index=False)
    print(f"Save complete: {filename}")


# --- 1. Data Preparation ---
def load_csv_data(file_path):
    """
    Loads PRVM data from a long-format .csv file and converts it to a 3D tensor.
    """
    print(f"Loading CSV data from '{file_path}'...")
    try:
        df = pd.read_csv(file_path, parse_dates=['date'])
        unique_dates = sorted(df['date'].unique())
        tickers = sorted(pd.unique(df[['ticker_i', 'ticker_j']].values.ravel('K')))
        num_days = len(unique_dates)
        num_assets = len(tickers)
        
        print(f"Processing data for {num_days} days and {num_assets} assets.")

        prvm_data = np.zeros((num_days, num_assets, num_assets))
        grouped = df.groupby('date')
        
        for i, date in enumerate(tqdm(unique_dates, desc="Converting data")):
            daily_data = grouped.get_group(date)
            matrix_df = daily_data.pivot_table(index='ticker_i', columns='ticker_j', values='value')
            matrix_df = matrix_df.reindex(index=tickers, columns=tickers, fill_value=0)
            matrix = matrix_df.to_numpy()
            # Complete the symmetric matrix by copying the lower triangle to the upper triangle
            lower_triangle = np.tril(matrix)
            matrix = lower_triangle + lower_triangle.T - np.diag(np.diag(matrix))
            prvm_data[i] = matrix

    except Exception as e:
        print(f"Error reading file: {e}")
        return None, None, None

    dates = pd.to_datetime(unique_dates)
    print(f"Data loading and conversion complete. Shape: {prvm_data.shape}")
    print(f"Date range: {dates.min().date()} to {dates.max().date()}")
    return prvm_data, dates, tickers

def load_gics_data(gics_file_path, num_assets):
    """Loads GICS classification from a CSV file."""
    print(f"Loading GICS data from '{gics_file_path}'...")
    try:
        gics_df = pd.read_csv(gics_file_path, header=None)
        gics_sectors = gics_df.iloc[1:, 0].to_numpy()
        if len(gics_sectors) != num_assets:
            print(f"Mismatch between number of GICS entries ({len(gics_sectors)}) and number of assets ({num_assets}).")
            return None
        return gics_sectors
    except Exception as e:
        print(f"Error reading GICS file: {e}")
        return None

def project_psd(matrix):
    """Projects a matrix onto the cone of positive semi-definite matrices."""
    eigenvalues, eigenvectors = eigh(matrix)
    eigenvalues[eigenvalues < 0] = 0
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

# --- 2. POET Implementation ---
def apply_poet(prvm, gics_sectors, num_factors=3):
    """
    Applies GICS-based hard-thresholding to a given PRVM matrix.
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

# --- 3. Performance Evaluation ---
def calculate_mspe(forecasts, ground_truths):
    """Calculates Mean Squared Prediction Error (MSPE)."""
    T = len(forecasts)
    if T == 0: return np.nan
    errors = [norm(forecasts[i] - ground_truths[i], 'fro')**2 for i in range(T)]
    return np.mean(errors)

def calculate_qlike(forecasts, ground_truths, reg=1e-8):
    """Calculates the QLIKE loss function."""
    T = len(forecasts)
    if T == 0: return np.nan
    qlike_vals = []
    for i in tqdm(range(T), desc="Calculating QLIKE"):
        forecast_reg = forecasts[i] + np.eye(forecasts[i].shape[0]) * reg
        try:
            eigenvalues, _ = eigh(forecast_reg)
            positive_eigenvalues = eigenvalues[eigenvalues > 1e-12]
            if len(positive_eigenvalues) == 0: continue
            log_det_val = np.sum(np.log(positive_eigenvalues))
            trace_val = np.trace(inv(forecast_reg) @ ground_truths[i])
            qlike_vals.append(log_det_val + trace_val)
        except np.linalg.LinAlgError:
            print(f"Warning: Failed to compute for data point {i}, skipping.")
            continue
    return np.mean(qlike_vals) if qlike_vals else np.nan

def plot_eigen_gaps(idio_matrices, num_factors=3, num_gaps_to_plot=100):
    """Generates an Eigen Gap plot."""
    print("\nGenerating Eigen Gap plot...")
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

# --- 4. Main Execution Logic (Corrected) ---
def main():
    prvm_file_path = 'prvm_0731_final_corrected.csv' 
    gics_file_path = 'gicslist.csv'
    
    prvm_data_all, dates, tickers = load_csv_data(prvm_file_path)
    if prvm_data_all is None: return

    gics_data = load_gics_data(gics_file_path, len(tickers))
    if gics_data is None: return

    num_factors = 3

    print("\nCalculating POET-PRVM, Idio Matrix, and Factor Part for all dates...")
    results_list = [apply_poet(prvm, gics_data, num_factors=num_factors) for prvm in tqdm(prvm_data_all, desc="Applying POET")]
    poet_prvm_all = np.array([res[0] for res in results_list])
    idio_matrices_all = np.array([res[1] for res in results_list])
    # factor_parts_all is not used in the corrected evaluation but kept for potential analysis
    # factor_parts_all = np.array([res[2] for res in results_list])

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
        print("Error defining evaluation periods: Data does not cover the 2018-2019 range.")
        return
    
    results = []

    for period_name, (start_idx, end_idx) in periods.items():
        print(f"\n--- Evaluating {period_name} ---")
        
        # The POET-PRVM benchmark uses the t-1 matrix to forecast the t matrix.
        # Both forecast and ground truth are from the POET-processed data.
        
        # 1. Forecasts: POET-PRVM results from the previous day (t-1)
        forecasts = poet_prvm_all[start_idx-1 : end_idx]
        
        # 2. Ground Truths: POET-PRVM results for the current day (t), used as a proxy.
        truths = poet_prvm_all[start_idx : end_idx+1]
        
        # Perform evaluation
        print("Calculating MSPE for the POET-PRVM benchmark...")
        mspe_val = calculate_mspe(forecasts, truths)
        
        print("Calculating QLIKE for the POET-PRVM benchmark...")
        qlike_val = calculate_qlike(forecasts, truths)
        
        results.append({
            "Period": period_name,
            "MSPE (x10^4)": mspe_val * 10**4 if not np.isnan(mspe_val) else 'N/A',
            "QLIKE (x10^-3)": qlike_val * 10**-3 if not np.isnan(qlike_val) else 'N/A',
        })

    results_df = pd.DataFrame(results)
    print("\n--- Final Evaluation Results ---")
    print(results_df.to_string(index=False))

    # --- Save data for FIVAR analysis ---


if __name__ == '__main__':
    main()
