import os
import re
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm
import concurrent.futures

# Import your existing modules
from read_cs_file import read_cs_file
from normalize_background import normalize_background
from apply_mcws import apply_mcws
from calc_radar_physics import calculate_radar_physics
from extra_FOL import extract_first_order_limits

def parse_codar_datetime(filename):
    """
    Extracts the datetime from a standard CODAR filename.
    Format expected: CSS_XXXX_YY_MM_DD_HHMM.cs
    """
    match = re.search(r'\_(\d{2})\_(\d{2})\_(\d{2})\_(\d{4})', filename)
    if match:
        yy, mm, dd, hhmm = match.groups()
        year = 2000 + int(yy)
        try:
            return datetime(year, int(mm), int(dd), int(hhmm[:2]), int(hhmm[2:]))
        except ValueError:
            return None
    return None

def check_antenna_health(spectra_dbm_list, alims, threshold_db=25.0):
    """
    Returns the degradation (difference in dBm) and boolean QC flags.
    """
    loop1, loop2, mono = spectra_dbm_list
    valid_powers_l1, valid_powers_l2, valid_powers_mono = [], [], []
    
    for r in range(alims.shape[0]):
        for col_offset in [0, 2]:
            start, end = alims[r, col_offset], alims[r, col_offset + 1]
            if end > start: 
                valid_powers_mono.append(np.mean(mono[r, start:end+1]))
                valid_powers_l1.append(np.mean(loop1[r, start:end+1]))
                valid_powers_l2.append(np.mean(loop2[r, start:end+1]))
    
    if not valid_powers_mono:
        return None, None, None, None 
        
    mean_mono = np.mean(valid_powers_mono)
    mean_l1 = np.mean(valid_powers_l1)
    mean_l2 = np.mean(valid_powers_l2)
    
    l1_diff = mean_mono - mean_l1
    l2_diff = mean_mono - mean_l2
    
    l1_degraded = l1_diff > threshold_db
    l2_degraded = l2_diff > threshold_db
    
    return l1_diff, l2_diff, l1_degraded, l2_degraded

def process_single_file(file_path):
    """
    Worker function to process a single cross-spectra file.
    Returns a dictionary with the results or error status.
    """
    # Hardcoded parameters for the worker
    vel_scale = 40.0
    max_vel = 200.0
    snr_min = 5.0
    DEGRADATION_THRESHOLD_DB = 25.0
    
    file_dt = parse_codar_datetime(file_path.name)
    if not file_dt:
        return {'status': 'skip', 'msg': f"Skipping {file_path.name} - could not parse datetime."}

    try:
        spectra_list, metadata = read_cs_file(str(file_path))
        if spectra_list is None or len(spectra_list) < 3:
            return {'status': 'skip', 'msg': "Invalid or incomplete spectra."}
            
        true_rep_freq = metadata.get('rep_freq_hz', 2.0)
        iFBragg, v_incr = calculate_radar_physics(
            metadata['freq_mhz'], 
            metadata['doppler_cells'], 
            true_rep_freq
        )
        
        spectra_dbm_list = spectra_list[0:3]
        monopole_dbm = spectra_list[2]
        
        h2_norm, DN_tuple, N = normalize_background(
            monopole_dbm, vel_scale, max_vel, v_incr, iFBragg
        )
        
        center = h2_norm.shape[1] // 2
        left_half = h2_norm[:, :center]
        right_half = h2_norm[:, center:]

        left_labels, _ = apply_mcws(left_half, DN_tuple[0], N)
        right_labels, _ = apply_mcws(right_half, DN_tuple[1], N)
        
        monopole_labels = np.zeros_like(monopole_dbm)
        monopole_labels[:, :center] = left_labels
        monopole_labels[:, center:] = right_labels + (right_labels > 0) * np.max(left_labels)
        
        monopole_alims = extract_first_order_limits(
            monopole_dbm, monopole_labels, iFBragg, N, max_vel, v_incr, snr_min
        )
        
        l1_diff, l2_diff, l1_bad, l2_bad = check_antenna_health(
            spectra_dbm_list, monopole_alims, DEGRADATION_THRESHOLD_DB
        )
        
        if l1_diff is not None:
            return {
                'status': 'success',
                'data': {
                    'datetime': file_dt,
                    'filename': file_path.name,
                    'l1_diff': l1_diff,
                    'l2_diff': l2_diff,
                    'l1_bad': l1_bad,
                    'l2_bad': l2_bad
                }
            }
        else:
            return {'status': 'skip', 'msg': "No valid FOL regions found."}
            
    except Exception as e:
        return {'status': 'error', 'msg': f"Error on {file_path.name}: {e}"}

def plot_qc_timeseries(results, output_filename="qc_timeseries_plot.png"):
    """
    Generates a time-series plot of the loop degradation and QC flags.
    """
    import matplotlib.pyplot as plt # Import here to avoid multiprocess collision
    
    if not results:
        print("No valid results to plot.")
        return

    results.sort(key=lambda x: x['datetime'])
    
    dates = [r['datetime'] for r in results]
    l1_diffs = [r['l1_diff'] for r in results]
    l2_diffs = [r['l2_diff'] for r in results]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(dates, l1_diffs, marker='o', linestyle='-', label='Loop 1 Drop (dB)', color='blue', alpha=0.7)
    ax.plot(dates, l2_diffs, marker='s', linestyle='-', label='Loop 2 Drop (dB)', color='green', alpha=0.7)
    
    threshold = 25.0
    ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'Failure Threshold ({threshold} dB)')
    
    ax.set_title("CODAR Antenna Loop Diagnostics Over Time", fontsize=14)
    ax.set_ylabel("Power Difference vs Monopole (dB)", fontsize=12)
    ax.set_xlabel("Date/Time", fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.6)
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    print(f"\nTime-series QC plot saved to: {output_filename}")
    plt.close(fig)

# ==========================================
# MAIN EXECUTION PIPELINE
# ==========================================
if __name__ == "__main__":
    
    target_dir = Path("/media/jude/Extreme Pro/CHATTS/HATY/css")
    
    # Gather files and strictly filter out .csr before counting
    all_files = list(target_dir.rglob("*.cs")) + list(target_dir.rglob("*.cs4"))
    files_to_process = [f for f in all_files if f.suffix.lower() != '.csr']
    
    print(f"Found {len(files_to_process)} valid cross spectra files to process.\n")
    
    qc_results = []
    
    # Fire up the multiprocessing pool
    # By default, ProcessPoolExecutor uses all available CPU cores.
    with concurrent.futures.ProcessPoolExecutor() as executor:
        
        # Submit all tasks to the pool
        futures = {executor.submit(process_single_file, fp): fp for fp in files_to_process}
        
        # Use as_completed to update the progress bar the moment a core finishes a file
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="QC Processing", unit="file"):
            result = future.result()
            
            if result['status'] == 'success':
                data = result['data']
                qc_results.append(data)
                
                # Still output warnings if an antenna goes down
                if data['l1_bad'] or data['l2_bad']:
                    tqdm.write(f"🚨 ALARM: {data['filename']} | L1 Drop: {data['l1_diff']:.1f}dB | L2 Drop: {data['l2_diff']:.1f}dB")
            
            elif result['status'] == 'error':
                tqdm.write(f"❌ {result['msg']}")

    print("\nBatch processing complete. Generating plot...")
    plot_qc_timeseries(qc_results)