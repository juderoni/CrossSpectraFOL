import numpy as np

def calculate_radar_physics(freq_mhz, n_dopp, rep_freq_hz):
    """
    Calculates the Bragg frequencies, indices, and velocity increment.
    Equivalent to the 'calculated constants from the CSS file' block.
    """
    c = 299792458.0  # Speed of light (m/s)
    g = 9.81         # Gravity (m/s^2)
    
    # 1. Frequency and Doppler setup
    fc = freq_mhz * 1e6
    f_dmax = 0.5 * rep_freq_hz
    delta_f = 2 * f_dmax / n_dopp
    
    # Create the doppler frequency array (-fDmax to +fDmax)
    doppler_freq = np.arange(-f_dmax + delta_f, f_dmax + delta_f, delta_f)
    # Ensure it's exactly n_dopp long due to floating point rounding
    doppler_freq = doppler_freq[:n_dopp] 
    
    # 2. Bragg theoretical calculations
    wavelength = (c / fc) / 2
    phase_velocity = np.sqrt(g * wavelength / (2 * np.pi))
    f_bragg = 2 * phase_velocity * fc / c
    
    # Find the indices closest to the +/- theoretical Bragg frequencies
    # We look for the two closest matches to f_bragg (which handles both positive and negative sides)
    diffs = np.abs(np.abs(doppler_freq) - f_bragg)
    sorted_indices = np.argsort(diffs)
    iFBragg = sorted_indices[:2]
    iFBragg.sort() # Ensure left peak is first, right peak is second
    
    # 3. Velocity calculations
    doppler_vel = doppler_freq * c / (2 * fc)
    
    # Create a current velocity array by subtracting the phase velocity 
    # (negative for approaching, positive for receding)
    phase_vel_array = np.concatenate([
        -phase_velocity * np.ones(n_dopp // 2), 
         phase_velocity * np.ones(n_dopp // 2)
    ])
    current_vel = doppler_vel - phase_vel_array
    
    # Get the median difference between velocity bins
    v_incr = np.median(np.diff(current_vel))
    return iFBragg, v_incr