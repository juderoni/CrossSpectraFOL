import numpy as np
from loguru import logger
from read_cs_file import read_cs_file
from peak_finder import find_peaks
from scipy import linalg as la

# Configure Loguru to dump output to a text file
logger.add("logs/music_covariance_test.log", format="{time} | {level} | {message}", level="INFO")

def build_covariance_matrix(spectra_list, r_idx, d_idx):
    """
    Extracts the linear auto and cross spectra from the expanded spectra_list
    and builds the 3x3 Hermitian covariance matrix for a specific cell.
    """
    # 1. Extract Linear Auto-spectra (Indices 3, 4, 5)
    s11 = spectra_list[3][r_idx, d_idx]  # Loop 1 Auto
    s22 = spectra_list[4][r_idx, d_idx]  # Loop 2 Auto
    s33 = spectra_list[5][r_idx, d_idx]  # Monopole Auto
    
    # 2. Extract Linear Complex Cross-spectra (Indices 6, 7, 8)
    s12 = spectra_list[6][r_idx, d_idx]  # Loop 1 x Loop 2
    s13 = spectra_list[7][r_idx, d_idx]  # Loop 1 x Monopole
    s23 = spectra_list[8][r_idx, d_idx]  # Loop 2 x Monopole
    
    # 3. Assemble the 3x3 Hermitian Matrix
    R = np.array([
        [s11, s12, s13],
        [np.conj(s12), s22, s23],
        [np.conj(s13), np.conj(s23), s33]
    ])
    
    return R

def ideal_steering_vector(theta_deg):
    """
    Returns the ideal [Loop 1, Loop 2, Monopole] response for a given compass bearing.
    """
    theta_rad = np.radians(theta_deg)
    # Loop 1 = cos, Loop 2 = sin, Monopole = 1
    return np.array([np.cos(theta_rad), np.sin(theta_rad), 1.0], dtype=np.complex64)

def calculate_music_doa(covariance_matrix, num_sources=1):
    """
    Executes the MUSIC algorithm on a 3x3 covariance matrix and returns the 
    best Direction of Arrival (DOA) angles.
    """
    # 1. Eigenvalue Decomposition
    # la.eigh is optimized for Hermitian matrices and returns sorted eigenvalues (ascending)
    eigenvalues, eigenvectors = la.eigh(covariance_matrix)
    
    # 2. Isolate the Noise Subspace
    # If 1 source, we take the 2 eigenvectors with the smallest eigenvalues
    noise_eigenvectors = eigenvectors[:, :-num_sources] 
    
    # Precompute Un * Un^H to save time in the loop
    Un_UnH = noise_eigenvectors @ noise_eigenvectors.conj().T
    
    # 3. Sweep the Compass
    angles = np.arange(1, 361)
    pseudo_spectrum = np.zeros(len(angles))
    
    for i, theta in enumerate(angles):
        a_theta = ideal_steering_vector(theta)
        
        # Calculate the denominator: a^H * Un * Un^H * a
        denominator = np.abs(a_theta.conj().T @ Un_UnH @ a_theta)
        
        # Add a tiny epsilon to prevent true division by zero
        pseudo_spectrum[i] = 1.0 / (denominator + 1e-10) 
        
    # 4. Find the Peaks
    # We use distance=10 to prevent finding two peaks that are essentially the same broad swell
    peaks, _ = find_peaks(pseudo_spectrum, distance=10) 
    
    if len(peaks) > 0:
        # Sort the peaks so the highest energy DOA is first
        peaks_sorted = peaks[np.argsort(pseudo_spectrum[peaks])][::-1]
        best_angles = angles[peaks_sorted[:num_sources]]
        return best_angles, pseudo_spectrum
    
    return [], pseudo_spectrum

if __name__ == "__main__":
    test_file = "cross_spectra_samples/CSS_OCRA_23_01_16_0100.cs"
    
    # Arbitrary test cell indices 
    test_range = 5
    test_doppler = 256
    
    logger.info(f"Reading file: {test_file}")
    
    try:
        spectra_list, metadata = read_cs_file(test_file)
        
        if spectra_list is not None:
            logger.info(f"Targeting Range: {test_range}, Doppler: {test_doppler}")
            
            covariance_matrix = build_covariance_matrix(spectra_list, test_range, test_doppler)
            
            logger.success("Matrix successfully built.")
            logger.info("\n=== 3x3 Covariance Matrix (Linear Space) ===")
            
            for i, row in enumerate(covariance_matrix):
                formatted_row = ["{:.4e}{:+.4e}j".format(val.real, val.imag) for val in row]
                logger.info(f"Row {i+1}: " + " | ".join(formatted_row))
                
    except Exception as e:
        logger.error(f"Failed during execution: {e}")