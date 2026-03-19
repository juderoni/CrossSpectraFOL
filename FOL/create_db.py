import os
import re
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm
import concurrent.futures

# Import your custom modules
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
    
    return l1_diff, l2_diff, l1_diff > threshold_db, l2_diff > threshold_db

def process_codar_file(file_path):
    """
    Worker function: Opens the file, runs the FOL extraction, and checks health.
    """
    filename = file_path.name
    file_dt = parse_codar_datetime(filename)
    
    # Strict year enforcement
    if not file_dt or file_dt.year != 2010:
        return {'status': 'skip', 'filename': filename, 'msg': 'Not 2010 or invalid date'}
        
    vel_scale = 40.0
    max_vel = 200.0
    snr_min = 5.0
    DEGRADATION_THRESHOLD_DB = 25.0
    
    try:
        spectra_list, metadata = read_cs_file(str(file_path))
        if spectra_list is None or len(spectra_list) < 3:
            return {'status': 'skip', 'filename': filename, 'msg': 'Invalid or missing spectra'}
            
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
                    'filename': filename,
                    'l1_diff': l1_diff,
                    'l2_diff': l2_diff,
                    'l1_bad': l1_bad,
                    'l2_bad': l2_bad
                }
            }
        else:
            return {'status': 'skip', 'filename': filename, 'msg': 'No valid FOL regions'}
            
    except Exception as e:
        return {'status': 'error', 'filename': filename, 'msg': str(e)}

def plot_qc_timeseries(results, output_filename="qc_timeseries_2010.png"):
    """
    Generates a time-series plot of the loop degradation and QC flags.
    """
    if not results:
        print("No valid results to plot.")
        return

    results.sort(key=lambda x: x['datetime'])
    dates = [r['datetime'] for r in results]
    l1_diffs = [r['l1_diff'] for r in results]
    l2_diffs = [r['l2_diff'] for r in results]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates, l1_diffs, marker='o', linestyle='-', markersize=3, label='Loop 1 Drop (dB)', color='blue', alpha=0.7)
    ax.plot(dates, l2_diffs, marker='s', linestyle='-', markersize=3, label='Loop 2 Drop (dB)', color='green', alpha=0.7)
    
    threshold = 25.0
    ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'Failure Threshold ({threshold} dB)')
    
    ax.set_title("CODAR Antenna Loop Diagnostics - 2010", fontsize=14)
    ax.set_ylabel("Power Difference vs Monopole (dB)", fontsize=12)
    ax.set_xlabel("Date/Time", fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.6)
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    print(f"\nTime-series QC plot saved to: {output_filename}")
    plt.close(fig)

if __name__ == "__main__":
    # Point directly to the 2010 subdirectory to save massive globbing overhead
    target_dir = Path("/media/jude/Extreme Pro/CHATTS/HATY/css/2010")
    
    print(f"Scanning {target_dir} for files...")
    all_files = list(target_dir.rglob("*.cs")) + list(target_dir.rglob("*.cs4"))
    
    # Filter out .csr files immediately
    files_to_process = [f for f in all_files if f.suffix.lower() != '.csr']
    print(f"Found {len(files_to_process)} valid cross spectra files for 2010.\n")
    
    if not files_to_process:
        print("No files found. Check your directory path!")
        exit(0)

    qc_results = []
    
    # Spin up the multiprocess pool
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Map the worker function to the files
        futures = {executor.submit(process_codar_file, fp): fp for fp in files_to_process}
        
        # Process them exactly as they finish
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing 2010 Data"):
            result = future.result()
            
            if result['status'] == 'success':
                data = result['data']
                qc_results.append(data)
                
                # Still output warnings if an antenna goes down
                if data['l1_bad'] or data['l2_bad']:
                    tqdm.write(f"🚨 ALARM: {data['filename']} | L1 Drop: {data['l1_diff']:.1f}dB | L2 Drop: {data['l2_diff']:.1f}dB")
            
            elif result['status'] == 'error':
                tqdm.write(f"❌ Error on {result['filename']}: {result['msg']}")

    print("\nBatch processing complete. Generating plot...")
    plot_qc_timeseries(qc_results)