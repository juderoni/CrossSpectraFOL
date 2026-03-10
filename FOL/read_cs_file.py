import os
import numpy as np
import struct
from datetime import datetime, timedelta
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
            
            start_freq_mhz = unpacked[0]
            bandwidth_khz = unpacked[2]
            sweep_rate_hz = unpacked[1] # This is the true fRepFreqHz
            
            # True Center Frequency = Start Freq + (Bandwidth / 2)
            metadata['freq_mhz'] = start_freq_mhz + (bandwidth_khz / 2000.0)
            metadata['rep_freq_hz'] = sweep_rate_hz
            
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
