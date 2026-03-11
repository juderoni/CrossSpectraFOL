from read_cs_file import read_cs_file
from normalize_background import normalize_background
from calc_radar_physics import calculate_radar_physics
import numpy as np
import matplotlib.pyplot as plt

def debug_loop2_clipping():
    test_file = "/home/jude/Repositories/CrossSpectraFOL/cross_spectra_samples/CSS_OCRA_25_09_01_0000.cs"
    print(f"Loading {test_file}...")
    spectra_list, metadata = read_cs_file(test_file)
    
    # Get Antenna 2 (Loop 2)
    loop2_dbm = spectra_list[1] 
    
    # Physics & Normalization
    true_rep_freq = metadata.get('rep_freq_hz', 1.0)
    iFBragg, v_incr = calculate_radar_physics(metadata['freq_mhz'], metadata['doppler_cells'], true_rep_freq)
    
    h2_norm, DN_tuple, N = normalize_background(loop2_dbm, 40.0, 200.0, v_incr, iFBragg)
    
    # Isolate Left Half
    center = h2_norm.shape[1] // 2
    left_half = h2_norm[:, :center]
    dn = DN_tuple[0]
    
    # Pick a specific range cell in the middle of the array
    test_range = 15
    raw_row = left_half[test_range, :]
    
    # Replicate Step 1 of apply_mcws.py
    p = max(10, 100 - round(2 * dn + test_range / (2 * dn)))
    p2 = np.percentile(raw_row, p)
    if p2 <= 0: p2 = 0.01
    
    clipped_row = np.clip(raw_row / p2, 0, 1)
    
    # Plot the transformation
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    axes[0].plot(loop2_dbm[test_range, :center], color='blue')
    axes[0].set_title(f"1. Raw Power (dBm) - Range Cell {test_range}")
    axes[0].grid(True)
    
    axes[1].plot(raw_row, color='green')
    axes[1].axhline(p2, color='red', linestyle='--', label=f'p2 Percentile Threshold ({p2:.4f})')
    axes[1].set_title("2. Normalized (H array) before clipping")
    axes[1].legend()
    axes[1].grid(True)
    
    axes[2].plot(clipped_row, color='purple')
    axes[2].set_title("3. Final Clipped Array (I array) fed into Watershed")
    axes[2].set_xlabel("Doppler Bins (Left Half)")
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig("debug_loop2_1d.png", dpi=300)
    print("Saved debug plot to debug_loop2_1d.png")

if __name__ == "__main__":
    debug_loop2_clipping()