import struct
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from datetime import datetime, timedelta
import os
from pathlib import Path
# Define the masking threshold
THRESHOLD_DBM = -135

def read_cs_file(filepath):
    # (This function remains identical to the previous versions)
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None, None

    with open(filepath, 'rb') as f:
        # 1. Read Base Header
        header_start = f.read(10)
        version, timestamp, v1_extent = struct.unpack('>hIi', header_start)
        
        if version > 32 or version < 1:
            raise ValueError(f"Invalid file version: {version}. Not a valid .cs file.")

        f.seek(10)
        cskind = struct.unpack('>h', f.read(2))[0]

        epoch = datetime(1904, 1, 1)
        record_time = epoch + timedelta(seconds=timestamp)

        metadata = {
            'filename': os.path.basename(filepath),
            'version': version,
            'time': record_time.strftime('%Y-%m-%d %H:%M:%S UTC'),
            'site': 'UNKN',
            'range_cells': 31,
            'doppler_cells': 512,
            'range_dist_km': 0.0,
            'freq_mhz': 0.0
        }

        # 2. Extract Extended V4+ Header Info
        if version >= 4:
            f.seek(16)
            site_code_bytes = struct.unpack('4s', f.read(4))[0]
            metadata['site'] = site_code_bytes.decode('ascii', errors='ignore').strip('\x00')
            
            f.seek(36)
            sweep_data = f.read(32)
            unpacked = struct.unpack('>fffiiiif', sweep_data)
            
            metadata['freq_mhz'] = unpacked[0]
            metadata['doppler_cells'] = unpacked[4]
            metadata['range_cells'] = unpacked[5]
            metadata['range_dist_km'] = unpacked[7]

        # 3. Jump to Data Section
        header_size = v1_extent + 10
        f.seek(header_size)

        # 4. Read the Multi-Dimensional Arrays for Antennas 1, 2, and 3
        n_range = metadata['range_cells']
        n_dopp = metadata['doppler_cells']
        
        # Pre-allocate a list of 3 arrays
        ant_spectra = [np.zeros((n_range, n_dopp), dtype=np.float32) for _ in range(3)]

        # Bytes to skip (3 Cross-Spectra complexes * 8 bytes each = 24 bytes)
        bytes_to_skip = 24 * n_dopp
        if cskind >= 2:
            bytes_to_skip += 4 * n_dopp # Skip Quality data array

        for r in range(n_range):
            for i in range(3):
                # Read each antenna in sequence
                data = np.fromfile(f, dtype='>f4', count=n_dopp)
                data = np.abs(data)
                data[data == 0] = 1e-10 
                ant_spectra[i][r, :] = data
            
            # Skip the cross-spectra to reach the next range cell
            f.seek(bytes_to_skip, 1) 

        # Convert all three arrays to dBm
        ant_spectra_dbm = [10 * np.log10(ant) - 34.2 for ant in ant_spectra]
        
        return ant_spectra_dbm, metadata

def plot_spectra_single_cmap(spectra_list, meta):
    if not spectra_list:
        return

    # Create a vertical stack (3x1) of subplots, sharing the X axis
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    # Adjust layout to leave significant room at the bottom for metadata
    plt.subplots_adjust(bottom=0.22, hspace=0.15) 

    # --- Configure the Colormap with a "Masking" Color ---
    # Get standard viridis colormap
    try:
        current_cmap = cm.get_cmap('viridis').copy() # Copy to avoid modifying global state
    except AttributeError:
        # Fallback for newer matplotlib versions
        current_cmap = plt.colormaps['viridis'].copy()

    # set_under handles any value < vmin explicitly.
    current_cmap.set_under('darkblue')

    # Determine a shared vmax based on the data (keep consistent across subplots)
    overall_vmax = max(np.max(ant) for ant in spectra_list)
    # Ensure vmax is at least higher than our threshold
    if overall_vmax <= THRESHOLD_DBM:
        overall_vmax = -50 

    antenna_names = ['Loop 1', 'Loop 2', 'Monopole']
    im = None # Store reference for colorbar

    for i, ant_data in enumerate(spectra_list):
        ax = axes[i]
        
        # Plot the data using the custom cmap.
        # CRITICAL: vmin=THRESHOLD_DBM triggers the 'set_under' color for noise floor
        im = ax.imshow(ant_data, aspect='auto', origin='lower',
                       cmap=current_cmap, vmin=THRESHOLD_DBM, vmax=overall_vmax)
        
        ax.set_title(f"Antenna {i+1} ({antenna_names[i]}) Spectra", fontsize=14)
        ax.set_ylabel('Range Cell Index', fontsize=12)
        
        if i == 2: # Only label X on the bottom plot
            ax.set_xlabel('Doppler Cell Index', fontsize=12)

    # --- Configure the single Colorbar ---
    # place it to the right of the entire plot stack
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation='vertical', fraction=0.046, pad=0.04, extend='min')
    cbar.set_label('Power (dBm)', fontsize=12)
    
    # Update the colorbar ticks to explicitly label that Dark Blue is our noise floor
    current_ticks = cbar.ax.get_yticks()
    # Replace the lowest tick (which is vmin/-135) with our specific label
    new_tick_labels = [str(int(t)) if t > THRESHOLD_DBM else f'< {THRESHOLD_DBM}' for t in current_ticks]
    cbar.ax.set_yticklabels(new_tick_labels)

    # --- Add the description of the data at the bottom ---
    desc_text = (
        f"File: {meta['filename']}   |   Version: {meta['version']}\n"
        f"Site: {meta['site']}   |   Timestamp: {meta['time']}\n"
        f"Transmit Freq: {meta['freq_mhz']:.3f} MHz   |   Range Resolution: {meta['range_dist_km']:.3f} km\n"
        f"Grid: {meta['range_cells']} Range Cells x {meta['doppler_cells']} Doppler Cells"
    )
    
    # Place text box in the dedicated bottom space
    fig.text(0.5, 0.03, desc_text, ha='center', va='bottom', fontsize=14, 
             bbox=dict(boxstyle='round,pad=1.0', facecolor='whitesmoke', edgecolor='gray', alpha=0.9))
     # Save the output
    path_file_path = Path(file_path)
    temp = path_file_path.stem
    output_filename = f'spectra_{temp}.png'
    plt.savefig(output_filename, dpi=300)
    print(f"Plot successfully saved to: {output_filename}")

# --- Execution ---
if __name__ == "__main__":
    filepaths = ["cross_spectra_samples/CSS_HATY_21_02_08_0230.cs",
                 "cross_spectra_samples/CSS_HATY_03_08_20_0000.cs4",
                 "cross_spectra_samples/CSS_HATY_06_05_03_0030.cs4",
                 "cross_spectra_samples/CSS_OCRA_23_01_16_0100.cs"
    ]
    
    for file_path in filepaths:
        try:
            spectra_list, metadata = read_cs_file(file_path)
            if spectra_list is not None:
                plot_spectra_single_cmap(spectra_list, metadata)
        except Exception as e:
            print(f"An error occurred: {e}")