import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_outage_blocks(ax, dates, color):
    """
    Groups continuous outages together and draws a single shaded block.
    """
    if dates.empty:
        return
    
    dates = dates.sort_values().reset_index(drop=True)
    start = dates.iloc[0]
    prev = dates.iloc[0]
    
    for d in dates.iloc[1:]:
        if (d - prev).total_seconds() > 7200: # 2 hour gap means the outage ended
            ax.axvspan(start, prev, color=color, alpha=0.15, lw=0)
            start = d
        prev = d
        
    ax.axvspan(start, prev, color=color, alpha=0.15, lw=0)

def compare_all_qc_methods(nc_file, cs_csv, diag_csv, year_str):
    print(f"Loading NetCDF Data Availability: {nc_file}...")
    
    # 1. Process NetCDF
    ds = xr.open_dataset(nc_file)
    ds['time'] = pd.to_datetime(ds['time'].values)
    ds_subset = ds.sel(time=slice(f'{year_str}-01-01', f'{year_str}-12-31'))
    
    if len(ds_subset.time) > 0:
        total_cells = len(ds_subset.range) * len(ds_subset.bearing)
        valid_counts = ds_subset['velocity'].notnull().sum(dim=['range', 'bearing'])
        availability_pct = (valid_counts / total_cells) * 100
        times_nc = pd.to_datetime(ds_subset['time'].values)
        pct_values = availability_pct.values
    else:
        print("Warning: No NetCDF data found.")
        times_nc, pct_values = [], []

    # 2. Process Our Custom Cross-Spectra QC
    print(f"Loading Custom CS QC: {cs_csv}...")
    df_cs = pd.read_csv(cs_csv)
    df_cs['Datetime'] = pd.to_datetime(df_cs['Datetime'])
    
    for col in ['Loop1_Dead', 'Loop2_Dead', 'Monopole_Dead']:
        if df_cs[col].dtype == object:
            df_cs[col] = df_cs[col].astype(str).str.lower() == 'true'
            
    cs_l1_fail = df_cs[df_cs['Loop1_Dead']]['Datetime']
    cs_l2_fail = df_cs[df_cs['Loop2_Dead']]['Datetime']
    cs_mono_fail = df_cs[df_cs['Monopole_Dead']]['Datetime']

    # 3. Process the New Diagnostics Data (.rdt derived)
    print(f"Loading STAT/Diagnostic QC: {diag_csv}...")
    df_diag = pd.read_csv(diag_csv)
    
    # Reconstruct Datetime from individual time columns
    df_diag['Datetime'] = pd.to_datetime(df_diag[['TYRS', 'TMON', 'TDAY', 'THRS', 'TMIN', 'TSEC']].rename(columns={
        'TYRS': 'year', 'TMON': 'month', 'TDAY': 'day', 'THRS': 'hour', 'TMIN': 'minute', 'TSEC': 'second'
    }))
    
    # Diagnostic QC Logic: Zeroed Amplitude = Dead Loop. Low SSN3 = Dead Monopole.
    diag_l1_fail = df_diag[df_diag['AMP1'] < 0.01]['Datetime']
    diag_l2_fail = df_diag[df_diag['AMP2'] < 0.01]['Datetime']
    diag_mono_fail = df_diag[df_diag['SSN3'] < 5.0]['Datetime']

    # 4. Generate the 3-Panel Plot
    print("Generating 3-panel comparison plot...")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]})
    
    # --- Top Panel: Data Availability ---
    if len(times_nc) > 0:
        # We plot the scatter data, but skip plt.colorbar() entirely
        ax1.scatter(times_nc, pct_values, c=pct_values, cmap='viridis', vmin=0, vmax=100, s=10, alpha=0.8, edgecolors='none')
        
    ax1.set_title(f"HATY Data Availability vs QC Methods ({year_str})", fontsize=16)
    ax1.set_ylabel("Availability (%)", fontsize=12)
    ax1.set_ylim(-5, 105)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # X-Ray Overlay: Draw shading on the top panel using OUR custom QC dates
    plot_outage_blocks(ax1, cs_mono_fail, 'black')
    plot_outage_blocks(ax1, cs_l1_fail, 'red')
    plot_outage_blocks(ax1, cs_l2_fail, 'orange')

    # --- Middle Panel: Custom Cross-Spectra QC ---
    ax2.scatter(cs_l1_fail, [1]*len(cs_l1_fail), color='red', marker='|', s=80)
    ax2.scatter(cs_l2_fail, [2]*len(cs_l2_fail), color='orange', marker='|', s=80)
    ax2.scatter(cs_mono_fail, [3]*len(cs_mono_fail), color='black', marker='X', s=80)
    
    ax2.set_ylabel("Custom CS QC", fontsize=12)
    ax2.set_yticks([1, 2, 3])
    ax2.set_yticklabels(['L1 Out', 'L2 Out', 'Mono Out'])
    ax2.set_ylim(0.5, 3.5)
    ax2.grid(True, axis='x', linestyle=':', alpha=0.6)

    # --- Bottom Panel: STAT/Diagnostic Data QC ---
    ax3.scatter(diag_l1_fail, [1]*len(diag_l1_fail), color='red', marker='|', s=80)
    ax3.scatter(diag_l2_fail, [2]*len(diag_l2_fail), color='orange', marker='|', s=80)
    ax3.scatter(diag_mono_fail, [3]*len(diag_mono_fail), color='black', marker='X', s=80)
    
    ax3.set_ylabel("Diagnostic .rdt QC", fontsize=12)
    ax3.set_xlabel("Date", fontsize=12)
    ax3.set_yticks([1, 2, 3])
    ax3.set_yticklabels(['L1 Out', 'L2 Out', 'Mono Out'])
    ax3.set_ylim(0.5, 3.5)
    ax3.grid(True, axis='x', linestyle=':', alpha=0.6)
    
    # Format the shared X-axis
    year_start = pd.to_datetime(f"{year_str}-01-01")
    year_end = pd.to_datetime(f"{year_str}-12-31")
    ax1.set_xlim(year_start, year_end)
    
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=45)

    plt.tight_layout()
    output_filename = f"HATY_3Panel_QC_Comparison_{year_str}.png"
    plt.savefig(output_filename, dpi=300)
    print(f"Plot saved successfully to: {output_filename}")
    plt.close(fig)

if __name__ == "__main__":
    netcdf_path = "cross_spectra_samples/HATY_R23_MQ_PFS_2004-2024.nc"
    cs_csv_path = "cross_spectra_samples/qc_failures_2010.csv"
    diag_csv_path = "cross_spectra_samples/HATY_2010_Diagnostics_Amplitudes.csv"
    
    compare_all_qc_methods(netcdf_path, cs_csv_path, diag_csv_path, "2010")