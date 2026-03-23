import os
import struct
import numpy as np
from datetime import datetime, timedelta

def read_full_cs_file(filepath):
    """
    Parses a CODAR .cs / .cs4 / .cs6 file completely based on the official spec.
    Returns a comprehensive metadata dictionary and the extracted data arrays.
    """
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None, None

    metadata = {'filename': os.path.basename(filepath)}
    
    with open(filepath, 'rb') as f:
        # --- 1. BASE HEADER (All Versions) ---
        base_header = f.read(10)
        version, timestamp, v1_extent = struct.unpack('>hIi', base_header)
        
        if version > 32 or version < 1:
            raise ValueError(f"Invalid file version: {version}. Not a valid .cs file.")
            
        metadata['version'] = version
        metadata['timestamp_raw'] = timestamp
        metadata['v1_extent'] = v1_extent
        
        # Mac Epoch: Jan 1, 1904
        epoch = datetime(1904, 1, 1)
        metadata['time_utc'] = (epoch + timedelta(seconds=timestamp)).strftime('%Y-%m-%d %H:%M:%S')

        # Default fallback values for older versions
        n_spectra_channels = 3
        n_range = 31 if version <= 3 else 0 
        n_dopp = 512
        ref_gain_db = -34.2 # Default CODAR gain offset
        
        # --- 2. VERSION 2+ HEADER ---
        if version >= 2:
            cskind, v2_extent = struct.unpack('>hi', f.read(6))
            metadata['cskind'] = cskind
            metadata['v2_extent'] = v2_extent
            
        # --- 3. VERSION 3+ HEADER ---
        if version >= 3:
            site_code = f.read(4).decode('ascii', errors='ignore').strip('\x00')
            v3_extent = struct.unpack('>i', f.read(4))[0]
            metadata['site_code'] = site_code
            metadata['v3_extent'] = v3_extent
            
        # --- 4. VERSION 4+ HEADER ---
        # --- 4. VERSION 4+ HEADER ---
        if version >= 4:
            v4_data = f.read(44)
            # FIXED: Removed the extra 'i'. Now exactly 11 fields (44 bytes)
            unpacked_v4 = struct.unpack('>iiifffiiiif', v4_data) 
            
            metadata['cover_minutes'] = unpacked_v4[0]
            metadata['deleted_source'] = unpacked_v4[1]
            metadata['override_src_info'] = unpacked_v4[2]
            metadata['start_freq_mhz'] = unpacked_v4[3]
            metadata['rep_freq_hz'] = unpacked_v4[4]
            metadata['bandwidth_khz'] = unpacked_v4[5]
            metadata['sweep_up'] = unpacked_v4[6]
            
            n_dopp = unpacked_v4[7]
            n_range = unpacked_v4[8]
            
            metadata['doppler_cells'] = n_dopp
            metadata['range_cells'] = n_range
            metadata['first_range_cell'] = unpacked_v4[9]
            metadata['range_cell_dist_km'] = unpacked_v4[10]
            
            v4_extent = struct.unpack('>i', f.read(4))[0]
            metadata['v4_extent'] = v4_extent

        # --- 5. VERSION 5+ HEADER ---
        if version >= 5:
            # FIXED: Increased read size from 20 to 24 bytes to match the 6 extracted fields
            v5_data = f.read(24) 
            unpacked_v5 = struct.unpack('>i4s4siiI', v5_data)
            metadata['output_interval'] = unpacked_v5[0]
            metadata['creator_type'] = unpacked_v5[1].decode('ascii', errors='ignore').strip('\x00')
            metadata['creator_version'] = unpacked_v5[2].decode('ascii', errors='ignore').strip('\x00')
            metadata['active_channels'] = unpacked_v5[3]
            
            n_spectra_channels = unpacked_v5[4]
            metadata['spectra_channels'] = n_spectra_channels
            metadata['active_chan_bits'] = unpacked_v5[5]
            
            v5_extent = struct.unpack('>i', f.read(4))[0]
            metadata['v5_extent'] = v5_extent

        # --- 6. VERSION 6+ HEADER (Blocks) ---
        metadata['v6_blocks'] = {}
        if version >= 6:
            v6_byte_size = struct.unpack('>I', f.read(4))[0]
            bytes_read = 0
            
            while bytes_read < v6_byte_size:
                block_key = f.read(4).decode('ascii', errors='ignore').strip('\x00')
                block_size = struct.unpack('>I', f.read(4))[0]
                block_data = f.read(block_size)
                
                # Check for Receiver Info block to update the Reference Gain
                if block_key == 'RCVI' and block_size >= 16:
                    # nReceiverModel(4), nRxAntennaModel(4), fReferenceGainDB(8)
                    ref_gain_db = struct.unpack('>ii d', block_data[:16])[2]
                    metadata['reference_gain_db'] = ref_gain_db
                    
                metadata['v6_blocks'][block_key] = f"{block_size} bytes"
                bytes_read += (8 + block_size)

        if 'reference_gain_db' not in metadata:
            metadata['reference_gain_db'] = ref_gain_db

        # --- 7. DATA SECTION ---
        # The spec mandates using v1_extent + 10 to jump safely to the data section
        f.seek(v1_extent + 10)
        
        # Pre-allocate arrays
        self_spectra = np.zeros((n_range, n_spectra_channels, n_dopp), dtype=np.float32)
        cross_spectra = np.zeros((n_range, n_spectra_channels, n_dopp), dtype=np.complex64)
        quality_data = np.zeros((n_range, n_dopp), dtype=np.float32)

        for r in range(n_range):
            # Read Self Spectra (Voltage Squared)
            for i in range(n_spectra_channels):
                data = np.fromfile(f, dtype='>f4', count=n_dopp)
                # Ensure no zeros before log10 conversion
                data[data <= 0] = 1e-10 
                # Convert to dBm immediately using the correct reference gain
                self_spectra[r, i, :] = 10 * np.log10(data) + ref_gain_db

            # Read Cross Spectra (Complex pairs)
            for i in range(n_spectra_channels):
                # 2 floats per complex number (Real, Imaginary)
                raw_complex = np.fromfile(f, dtype='>f4', count=n_dopp * 2)
                # Reshape to (n_dopp, 2) and combine into a single complex array
                complex_array = raw_complex[0::2] + 1j * raw_complex[1::2]
                cross_spectra[r, i, :] = complex_array

            # Read Quality Array (if applicable)
            if version >= 2 and metadata.get('cskind', 0) >= 2:
                quality_data[r, :] = np.fromfile(f, dtype='>f4', count=n_dopp)

        data_payload = {
            'self_spectra_dbm': self_spectra,
            'cross_spectra_complex': cross_spectra,
            'quality': quality_data
        }

        return metadata, data_payload

# Example Usage:
meta, data = read_full_cs_file('/home/jude/Repositories/CrossSpectraFOL/cross_spectra_samples/CSS_HATY_21_02_08_0230.cs')
for k, v in meta.items():
    print(f"{k}: {v}")