import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_degradation_trends(diag_csv, year_str):
    print(f"Loading Diagnostic CSV file: {diag_csv}...")
    df_diag = pd.read_csv(diag_csv)
    
    # 1. Reconstruct Datetime from individual time columns
    df_diag['Datetime'] = pd.to_datetime(df_diag[['TYRS', 'TMON', 'TDAY', 'THRS', 'TMIN', 'TSEC']].rename(columns={
        'TYRS': 'year', 'TMON': 'month', 'TDAY': 'day', 'THRS': 'hour', 'TMIN': 'minute', 'TSEC': 'second'
    }))
    
    # Sort chronologically to ensure clean lines
    df_diag = df_diag.sort_values('Datetime').reset_index(drop=True)
    
    # 2. Extract Data
    times = df_diag['Datetime']
    amp1 = df_diag['AMP1']
    amp2 = df_diag['AMP2']
    ssn3 = df_diag['SSN3']

    # 3. Generate a 3-Panel Plot
    print("Generating continuous degradation plot...")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # --- Panel 1: Loop 1 Amplitude ---
    ax1.plot(times, amp1, color='blue', alpha=0.7, linewidth=1.5)
    # ax1.axhline(y=0.01, color='red', linestyle='--', linewidth=2, label='Failure Threshold (0.01)')
    # Shade the "Dead Zone" below the threshold
    ax1.fill_between(times, 0, 0.01, color='red', alpha=0.1)
    
    ax1.set_title(f"Hardware Degradation Trajectories ({year_str})", fontsize=16)
    ax1.set_ylabel("Loop 1 Amplitude", fontsize=12)
    ax1.set_ylim(-0.05, df_diag['AMP1'].quantile(0.99)) # Cap outliers for better viewing
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle=':', alpha=0.6)

    # --- Panel 2: Loop 2 Amplitude ---
    ax2.plot(times, amp2, color='green', alpha=0.7, linewidth=1.5)
    # ax2.axhline(y=0.01, color='red', linestyle='--', linewidth=2, label='Failure /Threshold (0.01)')
    # ax2.fill_between(times, 0, 0.01, color='red', alpha=0.1)
    
    ax2.set_ylabel("Loop 2 Amplitude", fontsize=12)
    ax2.set_ylim(-0.05, df_diag['AMP2'].quantile(0.99))
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle=':', alpha=0.6)

    # --- Panel 3: Monopole Signal-to-Noise (SSN3) ---
    ax3.plot(times, ssn3, color='purple', alpha=0.7, linewidth=1.5)
    # ax3.axhline(y=5.0, color='red', linestyle='--', linewidth=2, label='Failure Threshold (5.0 dB)')
    # ax3.fill_between(times, -10, 5.0, color='red', alpha=0.1)
    
    ax3.set_ylabel("Monopole SNR (dB)", fontsize=12)
    ax3.set_xlabel("Date", fontsize=12)
    ax3.set_ylim(-5, df_diag['SSN3'].quantile(0.99))
    ax3.legend(loc='upper right')
    ax3.grid(True, linestyle=':', alpha=0.6)
    
    # 4. Format the shared X-axis
    year_start = pd.to_datetime(f"{year_str}-01-01")
    year_end = pd.to_datetime(f"{year_str}-12-31")
    ax1.set_xlim(year_start, year_end)
    
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=45)

    plt.tight_layout()
    output_filename = f"HATY_Diagnostic_Degradation_{year_str}.png"
    plt.savefig(output_filename, dpi=300)
    print(f"Plot saved successfully to: {output_filename}")
    plt.close(fig)

if __name__ == "__main__":
    diag_csv_path = "cross_spectra_samples/HATY_2010_Diagnostics_Amplitudes.csv"
    plot_degradation_trends(diag_csv_path, "2010")