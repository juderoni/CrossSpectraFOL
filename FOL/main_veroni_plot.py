from read_cs_file import read_cs_file
from normalize_background import normalize_background
from apply_mcws import apply_mcws
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from calc_radar_physics import calculate_radar_physics
from skimage.segmentation import find_boundaries

# Assuming you have loaded your data

def plot_watershed_results(monopole_dbm, left_labels, right_labels, meta):
    """
    Overlays the watershed labels on the raw spectrum to verify performance.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    im = ax.imshow(monopole_dbm, aspect='auto', origin='lower', cmap='viridis', vmin=-160, vmax=-80)
    
    full_labels = np.zeros_like(monopole_dbm)
    center = monopole_dbm.shape[1] // 2
    full_labels[:, :center] = left_labels
    full_labels[:, center:] = right_labels

    # FIX: Explicitly find the boundaries between the integer segments
    boundaries = find_boundaries(full_labels, mode='inner')

    # Contour the boundaries array
    ax.contour(boundaries, levels=[0.5], colors='red', linewidths=1.0, alpha=0.8)

    ax.set_title(f"Monopole Watershed Segmentation\n{meta['filename']}")
    ax.set_ylabel('Range Cell Index')
    ax.set_xlabel('Doppler Cell Index')
    fig.colorbar(im, ax=ax, label='Power (dBm)')
    
    output_filename = f"mcws_result_{Path(meta['filename']).stem}.png"
    plt.savefig(output_filename, dpi=300)
    print(f"Watershed plot saved to: {output_filename}")
    plt.close(fig)
# ==========================================
# 6. MAIN EXECUTION PIPELINE
# ==========================================
if __name__ == "__main__":
    # Parameters exactly as defined in the MATLAB user_param list
    vel_scale = 40.0
    max_vel = 200.0
    snr_min = 5.0
    
    # Hardware specific (Modify if your system uses a different sweep rate)
    rep_freq_hz = 2.0 

    # Replace with your local file path
    test_file = "/home/jude/Repositories/OpenCSS/cross_spectra_samples/CSS_OCRA_25_09_01_0000.cs"

    
    try:
        print(f"Processing: {test_file}")
        spectra_list, metadata = read_cs_file(test_file)
        
        if spectra_list is not None:
            monopole_dbm = spectra_list[2] # Grab Antenna 3
            
            # 1. Physics
            iFBragg, v_incr = calculate_radar_physics(
                metadata['freq_mhz'], 
                metadata['doppler_cells'], 
                rep_freq_hz
            )
            print(f"Calculated Bragg Indices: {iFBragg}")
            
            # 2. Normalize Background
            h2_norm, DN_tuple, N = normalize_background(
                monopole_dbm, vel_scale, max_vel, v_incr, iFBragg
            )
            
            # 3. Apply Watershed
            center = h2_norm.shape[1] // 2
            left_half = h2_norm[:, :center]
            right_half = h2_norm[:, center:]
            
            # Change these lines in your main.py:
            left_labels, _ = apply_mcws(left_half, DN_tuple[0], N)
            right_labels, _ = apply_mcws(right_half, DN_tuple[1], N)
            
            # 4. Save visualization
            plot_watershed_results(monopole_dbm, left_labels, right_labels, metadata)
            print("Pipeline completed successfully.")
            
    except Exception as e:
        print(f"An error occurred: {e}")