import struct
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
from pathlib import Path

def read_cs_file(filepath):
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

def plot_spectra(spectra_data_list, meta):
    if not spectra_data_list:
        return

    # Create a 3x1 grid of subplots (Stacked vertically)
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Adjust spacing: 
    # - bottom=0.18 gives space for the text box.
    # - right=0.88 gives space for the colorbar.
    # - hspace=0.25 prevents titles from overlapping the plots above them.
    plt.subplots_adjust(bottom=0.18, right=0.88, hspace=0.25)
    
    # Calculate global min and max for a unified color scale
    vmin = min(np.min(ant) for ant in spectra_data_list)
    vmax = max(np.max(ant) for ant in spectra_data_list)

    im = None
    antenna_names = ['Loop 1', 'Loop 2', 'Monopole'] # Standard SeaSonde layout

    for i, ax in enumerate(axes):
        # Apply the shared vmin/vmax so colors mean the exact same dBm across all plots
        im = ax.imshow(spectra_data_list[i], aspect='auto', origin='lower', 
                       cmap='viridis', vmin=vmin, vmax=vmax)
        
        ax.set_title(f"Antenna {i+1} ({antenna_names[i]})")
        ax.set_ylabel('Range Cell Index') # Y-label on every plot
        
        if i == 2:
            ax.set_xlabel('Doppler Cell Index') # X-label only on the bottom plot

    # Add a single vertical colorbar for the entire figure on the right side
    # [left, bottom, width, height]
    cbar_ax = fig.add_axes([0.90, 0.25, 0.02, 0.5]) 
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Power (dBm)')
    
    # Add the descriptive metadata string at the bottom
    desc_text = (
        f"File: {meta['filename']}  |  Version: {meta['version']}\n"
        f"Site: {meta['site']}  |  Timestamp: {meta['time']}\n"
        f"Transmit Freq: {meta['freq_mhz']:.3f} MHz  |  Range Resolution: {meta['range_dist_km']:.3f} km\n"
        f"Grid: {meta['range_cells']} Range Cells x {meta['doppler_cells']} Doppler Cells"
    )
    
    # Lowered the Y coordinate from 0.05 to 0.02 so it sits perfectly under the plots
    fig.text(0.5, 0.02, desc_text, ha='center', va='bottom', fontsize=12, 
             bbox=dict(boxstyle='round,pad=0.5', facecolor='whitesmoke', edgecolor='gray', alpha=0.9))
    
    # Save the output
    path_file_path = Path(meta['filename'])
    temp = path_file_path.stem
    output_filename = f'spectra_raw_{temp}.png'
    plt.savefig(output_filename, dpi=300)
    print(f"Plot successfully saved to: {output_filename}")
    plt.close(fig) # Prevent memory leaks if processing many files

# --- Execution ---
if __name__ == "__main__":
    filepaths = ["cross_spectra_samples/CSS_HATY_21_02_08_0230.cs",
                 "cross_spectra_samples/CSS_HATY_03_08_20_0000.cs4",
                 "cross_spectra_samples/CSS_HATY_06_05_03_0030.cs4",
                 "cross_spectra_samples/CSS_OCRA_23_01_16_0100.cs",
                 "cross_spectra_samples/CSS_OCRA_25_09_01_0000.cs"
    ]
    
    for file_path in filepaths:
        try:
            spectra_list, metadata = read_cs_file(file_path)
            if spectra_list is not None:
                plot_spectra(spectra_list, metadata)
        except Exception as e:
            print(f"An error occurred: {e}")