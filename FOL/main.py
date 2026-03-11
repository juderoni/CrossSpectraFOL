from read_cs_file import read_cs_file
from normalize_background import normalize_background
from apply_mcws import apply_mcws
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from calc_radar_physics import calculate_radar_physics
from skimage.segmentation import find_boundaries
from extra_FOL import extract_first_order_limits
from build_covariance_matrix import build_covariance_matrix, calculate_music_doa

def plot_watershed_results(spectra_dbm_list, full_labels_list, alims_list, meta):
    """
    Overlays the watershed boundaries (red) and the final Alims extraction limits (white)
    for Loop 1, Loop 2, and the Monopole in a vertically stacked 3-panel subplot.
    """
    fig, axes = plt.subplots(3, 1, figsize=(15, 8), sharex=True, sharey=True)
    antenna_names = ['Loop 1', 'Loop 2', 'Monopole']
    
    for i, ax in enumerate(axes):
        dbm = spectra_dbm_list[i]
        labels = full_labels_list[i]
        alims = alims_list[i]
        
        # Plot original dBm data
        im = ax.imshow(dbm, aspect='auto', origin='lower', cmap='viridis', vmin=-160, vmax=-80)
        
        # Plot the raw watershed segment boundaries
        boundaries = find_boundaries(labels, mode='inner')
        ax.contour(boundaries, levels=[0.5], colors='red', linewidths=0.5, alpha=0.5)

        # Plot the final extracted Alims boundaries
        n_range = dbm.shape[0]
        y_vals = np.arange(n_range)
        
        # Plot Left Bragg boundaries
        ax.plot(alims[:, 0], y_vals, color='white', linewidth=2, label='First Order Limits' if i==0 else "")
        ax.plot(alims[:, 1], y_vals, color='white', linewidth=2)
        # Plot Right Bragg boundaries
        ax.plot(alims[:, 2], y_vals, color='white', linewidth=2)
        ax.plot(alims[:, 3], y_vals, color='white', linewidth=2)

        ax.set_title(f"{antenna_names[i]} FOL Extraction (Monopole Derived)")
        ax.set_ylabel('Range Cell Index')
        
        if i == 2:
            ax.set_xlabel('Doppler Cell Index')
        if i == 0:
            ax.legend(loc='upper right')
            
        fig.colorbar(im, ax=ax, label='Power (dBm)')
    
    plt.suptitle(f"Multi-Antenna FOL Extraction\n{meta['filename']}", fontsize=14, y=0.98)
    plt.tight_layout()
    
    output_filename = f"fol_extracted_all_{Path(meta['filename']).stem}.png"
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

    # test_file = "/home/jude/Repositories/CrossSpectraFOL/cross_spectra_samples/CSS_OCRA_25_09_01_0000.cs"
    test_file = "/home/jude/Repositories/CrossSpectraFOL/cross_spectra_samples/CSS_OCRA_23_01_16_0100.cs"
    try:
        print(f"Processing: {test_file}")
        spectra_list, metadata = read_cs_file(test_file)
        
        if spectra_list is not None:
            true_rep_freq = metadata.get('rep_freq_hz', 1.0) # Defaults to 1.0 if not found
            
            iFBragg, v_incr = calculate_radar_physics(
                metadata['freq_mhz'], 
                metadata['doppler_cells'], 
                true_rep_freq
            )
            print(f"Sweep Rate: {true_rep_freq} Hz")
            print(f"Calculated Bragg Indices: {iFBragg}")
            
            # Grab all three arrays for plotting later
            spectra_dbm_list = spectra_list[0:3]
            
            # 1. Isolate the Monopole (Antenna 3) for FOL processing
            monopole_dbm = spectra_list[2]
            print("Extracting limits using Monopole (Antenna 3) data...")
            
            h2_norm, DN_tuple, N = normalize_background(
                monopole_dbm, vel_scale, max_vel, v_incr, iFBragg
            )
            
            center = h2_norm.shape[1] // 2
            left_half = h2_norm[:, :center]
            right_half = h2_norm[:, center:]

            left_labels, _ = apply_mcws(left_half, DN_tuple[0], N)
            right_labels, _ = apply_mcws(right_half, DN_tuple[1], N)
            
            # STITCH THE LABELS TOGETHER
            monopole_labels = np.zeros_like(monopole_dbm)
            monopole_labels[:, :center] = left_labels
            monopole_labels[:, center:] = right_labels + (right_labels > 0) * np.max(left_labels)
            
            # THE FINAL EXTRACTION
            monopole_alims = extract_first_order_limits(
                monopole_dbm, monopole_labels, iFBragg, N, max_vel, v_incr, snr_min
            )
            
            # 2. Replicate the Monopole limits for Loop 1 and Loop 2 plotting
            full_labels_list = [monopole_labels, monopole_labels, monopole_labels]
            alims_list = [monopole_alims, monopole_alims, monopole_alims]
            
            # 3. Generate Plot
            plot_watershed_results(spectra_dbm_list, full_labels_list, alims_list, metadata)
            print("Pipeline completed successfully. Multi-panel FOL image generated.")

            # Let's test Range Cell 15
            test_range = 15
            
            # Get the Left Bragg limits for this range cell
            left_start = monopole_alims[test_range, 0]
            left_end = monopole_alims[test_range, 1]
            
            print(f"\nRunning MUSIC for Range Cell {test_range} (Doppler Bins {left_start} to {left_end}):")
            
            # Loop exclusively through the bins that fall INSIDE the First Order Limits
            for doppler_bin in range(left_start, left_end + 1):
                # 1. Build the matrix
                R = build_covariance_matrix(spectra_list, test_range, doppler_bin)
                
                # 2. Run MUSIC
                doas, spectrum = calculate_music_doa(R, num_sources=1)
                
                if len(doas) > 0:
                    velocity = (doppler_bin - iFBragg[0]) * v_incr
                    print(f"  Bin {doppler_bin}: Velocity = {velocity:+.2f} cm/s | Bearing = {doas[0]:03d}°")
            
    except Exception as e:
        print(f"An error occurred: {e}")