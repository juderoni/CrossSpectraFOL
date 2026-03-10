import numpy as np
from scipy.signal import find_peaks

def normalize_background(monopole_dbm, vel_scale, max_vel, v_incr, iFBragg):
    """
    Filters range-dependent background energy, normalizes the spectra, 
    and dynamically calculates the smoothing length scale (DN).
    """
    n_range, n_dopp = monopole_dbm.shape
    center = n_dopp // 2
    
    # 1. Base length scale calculations
    N = int(np.round(vel_scale / (v_incr * 100)))
    if N % 2 == 0: N -= 1
        
    N_max = int(np.round(max_vel / (v_incr * 100)))
    if N_max % 2 == 0: N_max -= 1
        
    min_DN = int(np.ceil(0.25 * vel_scale / (v_incr * 100)))

    # Mask out the center frequencies (DC peak)
    ci_start = center - N
    ci_end = center + N + 1
    gain3 = np.copy(monopole_dbm)
    gain3[:, ci_start:ci_end] = np.nan

    # Split into left and right halves
    left_half = gain3[:, :center]
    right_half = gain3[:, center:]

    # 2. Get Max, Mean, and Outer Edge Background for each half
    def get_stats(half_data, is_left):
        edge_data = half_data[:, :N] if is_left else half_data[:, -N:]
        return np.column_stack((
            np.nanmax(half_data, axis=1),
            np.nanmean(half_data, axis=1),
            np.nanmean(edge_data, axis=1)
        ))

    lmm = get_stats(left_half, is_left=True)
    rmm = get_stats(right_half, is_left=False)

    # 3. Apply Moving Average Filter across range cells
    f_l = int(np.ceil(n_range / N))
    if f_l % 2 == 0: f_l += 1
    filter_shape = np.ones(f_l) / f_l

    def apply_range_filter(stats_array):
        pad_len = f_l // 2
        padded = np.pad(stats_array, ((pad_len, pad_len), (0, 0)), mode='edge')
        filtered = np.zeros_like(stats_array)
        for col in range(stats_array.shape[1]):
            filtered[:, col] = np.convolve(padded[:, col], filter_shape, mode='valid')
        return filtered

    lmml = apply_range_filter(lmm)
    rmml = apply_range_filter(rmm)

    # 4. Create normalized h2 array
    h2 = np.zeros_like(gain3)
    
    # Subtract the mean background (column 1)
    h2[:, :center] = gain3[:, :center] - lmml[:, 1][:, np.newaxis]
    h2[:, center:] = gain3[:, center:] - rmml[:, 1][:, np.newaxis]
    
    # Normalize by (max - mean)
    left_norm = lmml[:, 0] - lmml[:, 1]
    right_norm = rmml[:, 0] - rmml[:, 1]
    
    h2[:, :center] /= left_norm[:, np.newaxis]
    h2[:, center:] /= right_norm[:, np.newaxis]

    # Clean up bounds
    h2[:, ci_start:ci_end] = 0
    h2[h2 < 0] = 0
    h2[np.isnan(h2)] = 0

    # 5. Dynamic DN calculation based on 2nd Order Energy
    # Calculate range-averaged spectrum to find peaks
    mm = np.nanmean(gain3, axis=0)
    peaks, _ = find_peaks(mm, prominence=np.nanstd(mm))
    
    # Simplification of the MATLAB 2nd-order logic:
    # If a prominent peak is found too close to the Bragg peak, we shrink DN to avoid smearing
    DN_left, DN_right = N, N 
    
    left_peaks = peaks[peaks < center]
    if len(left_peaks) > 0 and abs(left_peaks[-1] - iFBragg[0]) < 3 * N:
        DN_left = max(min_DN, int(N * 0.5)) # Shrink window
        
    right_peaks = peaks[peaks > center]
    if len(right_peaks) > 0 and abs(right_peaks[0] - iFBragg[1]) < 3 * N:
        DN_right = max(min_DN, int(N * 0.5)) # Shrink window

    return h2, (DN_left, DN_right), N