"""Imperceptibility and robustness metrics: PSNR, SSIM, BER, NC, extraction success.

Images can be numpy arrays (BGR or gray uint8, the pipeline's format) or PIL images.
"""
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from .config import BER_SUCCESS_THRESHOLD


def _arr(img):
    """Converts a PIL Image or array-like object into a standard NumPy array.
    
    Args:
        img (np.ndarray or PIL.Image.Image): Input image data.
        
    Returns:
        np.ndarray: Coerced NumPy array.
    """
    return img if isinstance(img, np.ndarray) else np.asarray(img)


def psnr(a, b):
    """Computes Peak Signal-to-Noise Ratio (PSNR) in decibels between two images.
    
    Identical images return 100.0 dB to avoid infinite division anomalies.
    
    Args:
        a (np.ndarray or PIL.Image.Image): Original host image.
        b (np.ndarray or PIL.Image.Image): Watermarked or distorted image.
        
    Returns:
        float: Calculated PSNR value in dB (bounded to 100.0 for identical inputs).
    """
    a, b = _arr(a), _arr(b)
    # Return capped ceiling if images match identically to avoid division by zero
    if np.array_equal(a, b):
        return 100.0
    return float(peak_signal_noise_ratio(a, b, data_range=255))


def ssim(a, b):
    """Computes Structural Similarity Index (SSIM) between two images.
    
    Evaluates luminance, contrast, and structure preservation across images.
    Averages results across color channels if input array is three-dimensional.
    
    Args:
        a (np.ndarray or PIL.Image.Image): Original reference image.
        b (np.ndarray or PIL.Image.Image): Target watermarked/attacked image.
        
    Returns:
        float: Mean SSIM index bounded in [-1.0, 1.0], where 1.0 indicates identity.
    """
    a, b = _arr(a), _arr(b)
    # Multi-channel color images specify channel_axis for per-channel averaging
    if a.ndim == 3:
        return float(structural_similarity(a, b, channel_axis=2, data_range=255))
    return float(structural_similarity(a, b, data_range=255))


def ber(true_bits, got_bits):
    """Computes Bit Error Rate (BER) between ground-truth and extracted binary sequences.
    
    BER is defined as the normalized Hamming distance:
        BER = 0.0 indicates perfect extraction (100% bit accuracy).
        BER ~= 0.5 indicates uncorrelated / random chance extraction.
        
    Args:
        true_bits (array-like): Ground truth binary sequence.
        got_bits (array-like): Recovered binary sequence from extraction pipeline.
        
    Returns:
        float: Fraction of misaligned bits in range [0.0, 1.0].
        
    Raises:
        ValueError: If input bit arrays differ in total element count.
    """
    t = np.asarray(true_bits).ravel()
    g = np.asarray(got_bits).ravel()
    # Enforce strict length equivalence between ground truth and candidate vectors
    if len(t) != len(g):
        raise ValueError(f"bit vectors must have equal length ({len(t)} vs {len(g)})")
    return float(np.mean(t != g))


def nc(true_bits, got_bits):
    """Computes Normalized Correlation (NC) between antipodal (+1/-1) bit vectors.
    
    Maps binary {0, 1} bits to bipolar {-1, +1} values:
        NC = 1.0 indicates perfect alignment.
        NC = 0.0 indicates orthogonal or random noise behavior.
        NC = -1.0 indicates completely inverted bitstreams.
        
    Args:
        true_bits (array-like): Ground truth binary sequence.
        got_bits (array-like): Recovered binary sequence.
        
    Returns:
        float: Normalized cross-correlation score in [-1.0, 1.0].
    """
    # Transform binary domain {0, 1} into zero-mean antipodal representation {-1.0, +1.0}
    t = 2.0 * np.asarray(true_bits).ravel() - 1
    g = 2.0 * np.asarray(got_bits).ravel() - 1
    return float(np.dot(t, g) / len(t))


def is_successful_extraction(ber_value, threshold=BER_SUCCESS_THRESHOLD):
    """Determines lenient extraction success based on raw Bit Error Rate tolerance.
    
    Evaluates whether raw extraction errors fall within the operational error-correction
    budget of downstream forward error correction (e.g., Reed-Solomon codes).
    
    Args:
        ber_value (float): Observed Bit Error Rate.
        threshold (float): Maximum permissible BER cutoff (default from config).
        
    Returns:
        bool: True if BER <= threshold, False otherwise.
    """
    return ber_value <= threshold
