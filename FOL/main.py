from read_cs_file import read_cs_file
from normalize_background import normalize_background
from apply_mcws import apply_mcws
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from calc_radar_physics import calculate_radar_physics
from skimage.segmentation import find_boundaries
from extra_FOL import extract_first_order_limits
def plot_watershed_results(monopole_dbm, full_labels, alims, meta):
    """
    Overlays the watershed boundaries (red) and the final Alims extraction limits (white).
    """
    from skimage.segmentation import find_boundaries
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot original Monopole data
    im = ax.imshow(monopole_dbm, aspect='auto', origin='lower', cmap='viridis', vmin=-160, vmax=-80)
    
    # Plot the raw watershed segment boundaries
    boundaries = find_boundaries(full_labels, mode='inner')
    ax.contour(boundaries, levels=[0.5], colors='red', linewidths=0.5, alpha=0.5)

    # Plot the final extracted Alims boundaries
    n_range = monopole_dbm.shape[0]
    y_vals = np.arange(n_range)
    
    # Plot Left Bragg boundaries
    ax.plot(alims[:, 0], y_vals, color='white', linewidth=2, label='First Order Limits')
    ax.plot(alims[:, 1], y_vals, color='white', linewidth=2)
    # Plot Right Bragg boundaries
    ax.plot(alims[:, 2], y_vals, color='white', linewidth=2)
    ax.plot(alims[:, 3], y_vals, color='white', linewidth=2)

    ax.set_title(f"Monopole FOL Extraction\n{meta['filename']}")
    ax.set_ylabel('Range Cell Index')
    ax.set_xlabel('Doppler Cell Index')
    ax.legend(loc='upper right')
    fig.colorbar(im, ax=ax, label='Power (dBm)')
    
    output_filename = f"fol_extracted_{Path(meta['filename']).stem}.png"
    plt.savefig(output_filename, dpi=300)
    print(f"Extraction plot saved to: {output_filename}")
    plt.close(fig)

# ==========================================
# MAIN EXECUTION PIPELINE
# ==========================================
if __name__ == "__main__":
    vel_scale = 40.0
    max_vel = 200.0
    snr_min = 5.0
    rep_freq_hz = 2.0 

    test_file = "/home/jude/Repositories/OpenCSS/cross_spectra_samples/CSS_OCRA_25_09_01_0000.cs"
    
    try:
        print(f"Processing: {test_file}")
        spectra_list, metadata = read_cs_file(test_file)
        
        if spectra_list is not None:
            monopole_dbm = spectra_list[2] 
            
            true_rep_freq = metadata.get('rep_freq_hz', 1.0) # Defaults to 1.0 if not found
            
            iFBragg, v_incr = calculate_radar_physics(
                metadata['freq_mhz'], 
                metadata['doppler_cells'], 
                true_rep_freq
            )
            print(f"Sweep Rate: {true_rep_freq} Hz")
            print(f"Calculated Bragg Indices: {iFBragg}")
            
            h2_norm, DN_tuple, N = normalize_background(
                monopole_dbm, vel_scale, max_vel, v_incr, iFBragg
            )
            
            center = h2_norm.shape[1] // 2
            left_half = h2_norm[:, :center]
            right_half = h2_norm[:, center:]
            
            left_labels, _ = apply_mcws(left_half, DN_tuple[0], N)
            right_labels, _ = apply_mcws(right_half, DN_tuple[1], N)
            
            # STITCH THE LABELS TOGETHER
            full_labels = np.zeros_like(monopole_dbm)
            full_labels[:, :center] = left_labels
            # Ensure unique segment IDs between left and right halves
            full_labels[:, center:] = right_labels + (right_labels > 0) * np.max(left_labels)
            
            # THE FINAL EXTRACTION
            alims = extract_first_order_limits(
                monopole_dbm, full_labels, iFBragg, N, max_vel, v_incr, snr_min
            )
            
            plot_watershed_results(monopole_dbm, full_labels, alims, metadata)
            print("Pipeline completed successfully. Alims array generated.")
            
    except Exception as e:
        print(f"An error occurred: {e}")