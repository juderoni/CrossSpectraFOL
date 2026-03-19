import os
import re
import csv
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

def independent_fol_qc(spectra_dbm_list, vel_scale, max_vel, v_incr, iFBragg, snr_min=5.0):
    """
    Evaluates antenna health by attempting to extract First Order Limits (FOL) 
    independently on Loop 1, Loop 2, and the Monopole. 
    """
    # Indices: 0 = Loop 1, 1 = Loop 2, 2 = Monopole
    failures = [True, True, True] 
    
    for i, antenna_dbm in enumerate(spectra_dbm_list):
        try:
            h2_norm, DN_tuple, N = normalize_background(
                antenna_dbm, vel_scale, max_vel, v_incr, iFBragg
            )
            
            center = h2_norm.shape[1] // 2
            left_half = h2_norm[:, :center]
            right_half = h2_norm[:, center:]

            left_labels, _ = apply_mcws(left_half, DN_tuple[0], N)
            right_labels, _ = apply_mcws(right_half, DN_tuple[1], N)
            
            labels = np.zeros_like(antenna_dbm)
            labels[:, :center] = left_labels
            labels[:, center:] = right_labels + (right_labels > 0) * np.max(left_labels)
            
            alims = extract_first_order_limits(
                antenna_dbm, labels, iFBragg, N, max_vel, v_incr, snr_min
            )
            
            has_valid_peak = False
            for r in range(alims.shape[0]):
                if (alims[r, 1] > alims[r, 0]) or (alims[r, 3] > alims[r, 2]):
                    has_valid_peak = True
                    break 
            
            failures[i] = not has_valid_peak

        except Exception:
            # Math blew up (e.g., dead static arrays), flag as failed
            failures[i] = True
            
    return failures[0], failures[1], failures[2]

def process_single_file(file_path):
    """
    Worker function to process a single cross-spectra file.
    """
    vel_scale = 40.0
    max_vel = 200.0
    snr_min = 5.0
    
    filename = file_path.name
    file_dt = parse_codar_datetime(filename)
    
    # Strictly enforce 2010 files only
    if not file_dt or file_dt.year != 2010:
        return {'status': 'skip', 'msg': f"Skipping {filename} - not 2010 or invalid date."}

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
        
        # Pass the first three arrays directly to the new independent QC function
        spectra_dbm_list = spectra_list[0:3]
        
        l1_bad, l2_bad, mono_bad = independent_fol_qc(
            spectra_dbm_list, vel_scale, max_vel, v_incr, iFBragg, snr_min
        )
        
        return {
            'status': 'success',
            'data': {
                'datetime': file_dt,
                'filename': filename,
                'l1_bad': l1_bad,
                'l2_bad': l2_bad,
                'mono_bad': mono_bad
            }
        }
            
    except Exception as e:
        return {'status': 'error', 'msg': f"Error on {filename}: {e}"}

def plot_qc_timeseries(results, output_filename="qc_timeseries_2010.png"):
    """
    Generates a timeline scatter plot showing exactly when each antenna failed.
    """
    if not results:
        print("No valid results to plot.")
        return

    results.sort(key=lambda x: x['datetime'])
    
    # Isolate failure dates for each specific antenna
    l1_fail_dates = [r['datetime'] for r in results if r['l1_bad']]
    l2_fail_dates = [r['datetime'] for r in results if r['l2_bad']]
    mono_fail_dates = [r['datetime'] for r in results if r['mono_bad']]
    
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Plot dots only when an outage occurs. 
    ax.scatter(l1_fail_dates, [1]*len(l1_fail_dates), color='red', marker='|', s=100, label='Loop 1 Dead')
    ax.scatter(l2_fail_dates, [2]*len(l2_fail_dates), color='orange', marker='|', s=100, label='Loop 2 Dead')
    ax.scatter(mono_fail_dates, [3]*len(mono_fail_dates), color='black', marker='X', s=100, label='Monopole Dead')
    
    ax.set_title("CODAR Hardware Outage Timeline - 2010", fontsize=14)
    ax.set_xlabel("Date", fontsize=12)
    
    # Format the Y-axis to clearly label the hardware
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(['Loop 1', 'Loop 2', 'Monopole'])
    ax.set_ylim(0.5, 3.5)
    
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    ax.grid(True, axis='x', linestyle=':', alpha=0.6)
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    print(f"\nTime-series QC plot saved to: {output_filename}")
    plt.close(fig)

def export_qc_to_csv(results, output_filename="qc_failures_2010.csv"):
    """
    Exports the complete QC results timeline to a CSV file.
    """
    if not results:
        print("No valid results to export to CSV.")
        return
        
    # Sort chronologically to match the plot
    results.sort(key=lambda x: x['datetime'])
    
    # Write to CSV
    with open(output_filename, mode='w', newline='') as csv_file:
        fieldnames = ['Datetime', 'Filename', 'Loop1_Dead', 'Loop2_Dead', 'Monopole_Dead']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        writer.writeheader()
        for r in results:
            writer.writerow({
                'Datetime': r['datetime'].strftime('%Y-%m-%d %H:%M:%S'),
                'Filename': r['filename'],
                'Loop1_Dead': r['l1_bad'],
                'Loop2_Dead': r['l2_bad'],
                'Monopole_Dead': r['mono_bad']
            })
            
    print(f"CSV data export saved to: {output_filename}")

# ==========================================
# MAIN EXECUTION PIPELINE
# ==========================================
if __name__ == "__main__":
    
    # Point directly to the 2010 folder to skip searching through other years
    target_dir = Path("/media/jude/Extreme Pro/CHATTS/HATY/css/2010")
    
    if not target_dir.exists():
        print(f"Error: Directory {target_dir} does not exist. Check your path.")
        exit(1)
        
    all_files = list(target_dir.rglob("*.cs")) + list(target_dir.rglob("*.cs4"))
    files_to_process = [f for f in all_files if f.suffix.lower() != '.csr']
    
    print(f"Found {len(files_to_process)} valid cross spectra files in 2010 to process.\n")
    
    qc_results = []
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_single_file, fp): fp for fp in files_to_process}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing 2010 Data", unit="file"):
            result = future.result()
            
            if result['status'] == 'success':
                data = result['data']
                qc_results.append(data)
                
                # Output a warning if ANY antenna is dead
                if data['l1_bad'] or data['l2_bad'] or data['mono_bad']:
                    status = []
                    if data['mono_bad']: status.append("MONO DEAD")
                    if data['l1_bad']: status.append("L1 DEAD")
                    if data['l2_bad']: status.append("L2 DEAD")
                    tqdm.write(f"🚨 ALARM: {data['filename']} | {' | '.join(status)}")
            
            elif result['status'] == 'error':
                tqdm.write(f"❌ {result['msg']}")

    print("\nBatch processing complete. Generating outputs...")
    plot_qc_timeseries(qc_results)
    export_qc_to_csv(qc_results)