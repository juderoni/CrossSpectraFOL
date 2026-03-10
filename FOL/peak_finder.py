import numpy as np
from scipy.signal import find_peaks

def get_significant_peaks(spectrum_avg, std_multiplier=1.0):
    """
    Replaces pksfinder. Finds peaks with a prominence threshold.
    """
    # Define the threshold (db_level in MATLAB)
    threshold = np.nanstd(spectrum_avg) * std_multiplier
    
    # Find peaks with a prominence of at least the threshold
    peaks, properties = find_peaks(spectrum_avg, prominence=threshold)
    
    return peaks