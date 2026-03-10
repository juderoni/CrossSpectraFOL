import numpy as np
from skimage.morphology import disk, erosion, dilation, reconstruction, local_maxima, remove_small_objects
from skimage.filters import sobel
from skimage.segmentation import watershed
from skimage.measure import label
from scipy.ndimage import distance_transform_edt


def apply_mcws(H, dn, N):
    """
    Applies Marker-Controlled Watershed Segmentation to a spectrum half.
    H: Normalized spectrum array
    dn: Dynamic length scale (radius of the disk)
    N: Original smoothing length scale (minimum pixel area)
    """
    # 1. Normalize and Saturate (like the MATLAB loop)
    I = np.copy(H)
    n_range, n_dopp = I.shape
    for i in range(n_range):
        p = max(10, 100 - round(2 * dn + i / (2 * dn)))
        p2 = np.percentile(H[i, :], p)
        if p2 <= 0:
            p2 = 0.01
        I[i, :] = np.clip(H[i, :] / p2, 0, 1)

    # 2. Morphological Opening/Closing by Reconstruction
    se = disk(int(dn))
    
    # Erode & Reconstruct
    Ie = erosion(I, se)
    Iobr = reconstruction(Ie, I, method='dilation')
    
    # Dilate & Reconstruct Complement
    Iobrd = dilation(Iobr, se)
    # skimage requires inverting for the closing reconstruction
    Iobrcbr_comp = reconstruction(1 - Iobrd, 1 - Iobr, method='dilation')
    Iobrcbr = 1 - Iobrcbr_comp

    # 3. Sobel Edge Mask
    gradmag = sobel(Iobrcbr)
    
    # Normalize gradmag and subtract to sharpen edges
    gm = (gradmag / np.max(gradmag)) * np.max(Iobrcbr)
    I_sharpened = Iobrcbr - gm

    # Find regional maxima (boolean mask of exact peaks)
    fgm = local_maxima(I_sharpened)
    
    # FIX: Just use the 1-pixel peaks directly as seeds! 
    # Delete or comment out the remove_small_objects line:
    # bw2 = remove_small_objects(fgm, max_size=int(N)) 
    bw2 = fgm

    distance = distance_transform_edt(~bw2)
    
    # Label the markers
    markers = label(bw2)
    
    # Apply watershed
    labels = watershed(distance, markers)
    
    return labels, Iobrcbr