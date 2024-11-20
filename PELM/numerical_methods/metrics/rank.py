import numpy as np
from numpy.typing import NDArray

def Compute_Rank(H: NDArray[np.float64], compute_uv: bool=False):
    """Computes the rank of the dataset matrix H.

    Args:
        H (NDArray[np.complex64]): Dataset matrix
        compute_uv (bool, optional): Whether np.linalg.svd computes the U and V orthogonal matrices. Defaults to False.

    Returns:
        if compute_uv is True returns svd of H else returns (U, svd, V).
    """
    M, w, h = H.shape
    
    if len(H.shape) == 3:
        H = np.reshape(H, (M, w*h))
    
    if compute_uv is True:
        U, svd, V = np.linalg.svd(H, compute_uv=True)
        return U, svd, V
    elif compute_uv is False:
        svd = np.linalg.svd(H, compute_uv=False)
        return svd
    
def WeylsThreshold(H_noisy: NDArray[np.float64], H_clean: NDArray[np.float64]) -> np.float64:
    """Return the maximum noise singular value of the noisy dataset matrix.

    Args:
        H_noisy (NDArray[np.float64]): Noisy dataset matrix.
        H_clean (NDArray[np.float64]): Dataset matrix with reduced or no noise.

    Returns:
        np.float64: Maximum of the singular value of the noise matrix (H_noisy - H_clean).
    """
    M, w, h = H_noisy.shape
    
    noise = np.reshape(np.abs(H_noisy-H_clean), (M, w*h))
    
    noise_svds = np.linalg.svd(noise, compute_uv=False)
    
    return np.max(noise_svds)