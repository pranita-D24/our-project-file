import cv2
import numpy as np
import logging
import math
import time

logger = logging.getLogger(__name__)

def compute_zernike_basis(size=64, order=8):
    """
    Pre-computes Zernike basis functions to accelerate moment calculation.
    """
    x = np.arange(size) - (size - 1) / 2
    y = np.arange(size) - (size - 1) / 2
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2) / (size / 2)
    Theta = np.arctan2(Y, X)
    
    # Mask for unit disk
    mask = R <= 1.0
    
    # We only care about unique magnitude pairs (n, m) where n-|m| is even
    basis = []
    for n in range(order + 1):
        for m in range(n + 1):
            if (n - m) % 2 == 0:
                # Radial polynomial R_nm
                rad = np.zeros_like(R)
                for k in range((n - m) // 2 + 1):
                    num = ((-1)**k * math.factorial(n - k))
                    den = (math.factorial(k) * 
                           math.factorial((n + m) // 2 - k) * 
                           math.factorial((n - m) // 2 - k))
                    rad += (num / den) * R**(n - 2 * k)
                
                # Complex basis element: R_nm * exp(-j*m*theta)
                # But for similarity we often just use magnitude
                basis.append((n, m, rad * np.exp(-1j * m * Theta) * mask))
                
    return basis

# Global cache for basis to avoid redundant re-computation
_ZERNIKE_BASIS = compute_zernike_basis(64, 8)

def get_zernike_moments(patch_64):
    """
    Computes Order-8 Zernike magnitude moments.
    Returns a 25-element vector.
    """
    moments = []
    # Normalize pixel values
    img = patch_64.astype(float) / 255.0
    
    for n, m, b in _ZERNIKE_BASIS:
        # Complex moment A_nm = (n+1)/pi * sum(img * conj(basis))
        # We handle (n+1)/pi via final unit-normalization anyway
        val = np.sum(img * b)
        moments.append(np.abs(val))
        
    return np.array(moments)

def get_hu_moments(patch_64):
    """
    Computes log-scaled Hu moments with epsilon safety.
    """
    m = cv2.moments(patch_64)
    hu = cv2.HuMoments(m).flatten()
    # PRECISE FORMULA: -sign(hu) * log10(abs(hu) + epsilon)
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    return hu_log

def l2_normalize(v):
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return v
    return v / norm

def compute_similarity(patch1, patch2):
    """
    Calculates combined Hu/Zernike cosine similarity.
    Each descriptor is L2-normalized independently before concatenation.
    """
    # 1. Extract raw vectors (Telemetry Instrumented)
    _t = time.time()
    hu1 = get_hu_moments(patch1)
    hu2 = get_hu_moments(patch2)
    print(f"[TEL] Hu={time.time()-_t:.3f}s")
    
    _t = time.time()
    z1 = get_zernike_moments(patch1)
    z2 = get_zernike_moments(patch2)
    print(f"[TEL] Zernike={time.time()-_t:.3f}s")
    
    # 2. Independent L2-normalization (User specific constraint)
    hu1_n, z1_n = l2_normalize(hu1), l2_normalize(z1)
    hu2_n, z2_n = l2_normalize(hu2), l2_normalize(z2)
    
    # 3. Concatenate
    vec1 = np.concatenate([hu1_n, z1_n])
    vec2 = np.concatenate([hu2_n, z2_n])
    
    # 4. Cosine Similarity
    dot = np.dot(vec1, vec2)
    sim = dot / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-10)
    
    return float(sim)