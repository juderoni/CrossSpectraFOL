import numpy as np

def extract_first_order_limits(monopole_dbm, full_labels, iFBragg, N, max_vel, v_incr, snr_min=5.0):
    """
    Isolates the watershed segments intersecting the Bragg peaks, applies an SNR threshold,
    and extracts the left/right bounding indices (Alims) for each range cell.
    """
    n_range, n_dopp = monopole_dbm.shape
    
    # Calculate N_max (maximum allowed index distance from the theoretical Bragg peak)
    N_max = int(np.round(max_vel / (v_incr * 100)))
    if N_max % 2 == 0: N_max -= 1
    
    # Calculate a proxy for the noise floor per range cell (bottom 10% of energy)
    # SNR = Pixel Power - Noise Floor
    noise_floor = np.percentile(monopole_dbm, 10, axis=1, keepdims=True)
    snr = monopole_dbm - noise_floor

    # Initialize the Alims array: [Left-Left, Left-Right, Right-Left, Right-Right]
    alims = np.zeros((n_range, 4), dtype=int)
    
    # Window to search for the Bragg segment
    a = max(1, N // 4)
    
    for half_idx, bragg_idx in enumerate(iFBragg):
        # 1. Find all segment IDs that touch the theoretical Bragg peak +/- 'a'
        bragg_window = full_labels[:, max(0, bragg_idx - a) : min(n_dopp, bragg_idx + a + 1)]
        bragg_segment_ids = np.unique(bragg_window)
        bragg_segment_ids = bragg_segment_ids[bragg_segment_ids > 0] # Ignore background (0)
        
        # 2. Create a binary mask of just those segments, filtered by SNR
        fo_mask = np.isin(full_labels, bragg_segment_ids) & (snr >= snr_min)
        
        # 3. Extract the boundaries per range cell
        col_offset = half_idx * 2
        
        for r in range(n_range):
            # Constrain our search to N_max around the Bragg peak
            search_start = max(0, bragg_idx - N_max)
            search_end = min(n_dopp, bragg_idx + N_max + 1)
            
            row_mask = fo_mask[r, search_start:search_end]
            true_indices = np.where(row_mask)[0] + search_start
            
            # If valid first-order pixels exist in this range cell, log their extents
            if len(true_indices) > 0:
                alims[r, col_offset] = true_indices[0]
                alims[r, col_offset + 1] = true_indices[-1]
            else:
                # Fallback: if no valid data, just drop the boundary on the theoretical peak
                alims[r, col_offset] = bragg_idx
                alims[r, col_offset + 1] = bragg_idx
                
    return alims