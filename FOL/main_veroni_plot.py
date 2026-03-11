from read_cs_file import read_cs_file
from normalize_background import normalize_background
from apply_mcws import apply_mcws
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from calc_radar_physics import calculate_radar_physics
from skimage.segmentation import find_boundaries

def plot_watershed_results(spectra_dbm_list, left_labels_list, right_labels_list, meta):
    """
    Overlays the watershed labels on the raw spectrum to verify performance
    for Loop 1, Loop 2, and the Monopole in a vertically stacked 3-panel subplot.
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True, sharey=True)
    antenna_names = ['Loop 1', 'Loop 2', 'Monopole']
    
    for i, ax in enumerate(axes):
        dbm = spectra_dbm_list[i]
        left_labels = left_labels_list[i]
        right_labels = right_labels_list[i]
        
        im = ax.imshow(dbm, aspect='auto', origin='lower', cmap='viridis', vmin=-160, vmax=-80)
        
        full_labels = np.zeros_like(dbm)
        center = dbm.shape[1] // 2
        full_labels[:, :center] = left_labels
        
        # Ensure unique segment IDs between left and right halves 
        # so boundaries draw correctly down the middle
        full_labels[:, center:] = right_labels + (right_labels > 0) * np.max(left_labels)

        # Explicitly find the boundaries between the integer segments
        boundaries = find_boundaries(full_labels, mode='inner')

        # Contour the boundaries array
        ax.contour(boundaries, levels=[0.5], colors='red', linewidths=1.0, alpha=0.8)

        ax.set_title(f"{antenna_names[i]} Watershed Segmentation")
        ax.set_ylabel('Range Cell Index')
        
        if i == 2:
            ax.set_xlabel('Doppler Cell Index')
            
        fig.colorbar(im, ax=ax, label='Power (dBm)')
    
    plt.suptitle(f"Multi-Antenna Watershed Segmentation\n{meta['filename']}", fontsize=14, y=0.98)
    plt.tight_layout()
    
    output_filename = f"mcws_result_all_{Path(meta['filename']).stem}.png"
    plt.savefig(output_filename, dpi=300)
    print(f"Watershed plot saved to: {output_filename}")
    plt.close(fig)

# ==========================================
# MAIN EXECUTION PIPELINE
# ==========================================
if __name__ == "__main__":
    # Parameters exactly as defined in the MATLAB user_param list
    vel_scale = 40.0
    max_vel = 200.0
    snr_min = 5.0
    
    # Hardware specific (Modify if your system uses a different sweep rate)
    rep_freq_hz = 2.0 

    # Replace with your local file path
    test_file = "/home/jude/Repositories/CrossSpectraFOL/cross_spectra_samples/CSS_OCRA_25_09_01_0000.cs"
    
    try:
        print(f"Processing: {test_file}")
        spectra_list, metadata = read_cs_file(test_file)
        
        if spectra_list is not None:
            # Safely grab sweep rate from metadata if available
            true_rep_freq = metadata.get('rep_freq_hz', rep_freq_hz)
            
            # 1. Physics
            iFBragg, v_incr = calculate_radar_physics(
                metadata['freq_mhz'], 
                metadata['doppler_cells'], 
                true_rep_freq
            )
            print(f"Calculated Bragg Indices: {iFBragg}")
            
            # Grab the first three arrays (Loop 1, Loop 2, Monopole in dBm)
            spectra_dbm_list = spectra_list[0:3]
            
            left_labels_list = []
            right_labels_list = []
            
            # Loop through each antenna to isolate the watershed segments
            for idx, dbm_array in enumerate(spectra_dbm_list):
                print(f"Applying watershed for Antenna {idx + 1}...")
                
                # 2. Normalize Background
                h2_norm, DN_tuple, N = normalize_background(
                    dbm_array, vel_scale, max_vel, v_incr, iFBragg
                )
                
                # 3. Apply Watershed
                center = h2_norm.shape[1] // 2
                left_half = h2_norm[:, :center]
                right_half = h2_norm[:, center:]
                
                left_labels, _ = apply_mcws(left_half, DN_tuple[0], N)
                right_labels, _ = apply_mcws(right_half, DN_tuple[1], N)
                
                left_labels_list.append(left_labels)
                right_labels_list.append(right_labels)
                
            # 4. Save visualization
            plot_watershed_results(spectra_dbm_list, left_labels_list, right_labels_list, metadata)
            print("Pipeline completed successfully. Multi-panel watershed image generated.")
            
    except Exception as e:
        print(f"An error occurred: {e}")





