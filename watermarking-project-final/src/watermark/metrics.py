"""Imperceptibility and robustness metrics: PSNR, SSIM, BER, NC, extraction success.

Images can be numpy arrays (BGR or gray uint8, the pipeline's format) or PIL images.
"""
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from .config import BER_SUCCESS_THRESHOLD


def _arr(img):
    """numpy array from a numpy array or a PIL image.
    
    Args:
        img: Input image as a numpy ndarray or PIL Image.
        
    Returns:
        np.ndarray: Image data converted to a numpy array.
    """
    return img if isinstance(img, np.ndarray) else np.asarray(img)


def psnr(a, b):
    """Peak signal-to-noise ratio in dB. Identical images return 100.0 (instead of inf).
    
    Args:
        a: Reference host image array or PIL image.
        b: Distorted or watermarked image array or PIL image.
        
    Returns:
        float: Calculated PSNR value in dB, capped at 100.0 for identical inputs.
    """
    a, b = _arr(a), _arr(b)
    # Avoid infinite values when both images match exactly
    if np.array_equal(a, b):
        return 100.0
    return float(peak_signal_noise_ratio(a, b, data_range=255))


def ssim(a, b):
    """Structural similarity, averaged over colour channels for colour images.
    
    Args:
        a: Reference host image array or PIL image.
        b: Target evaluated image array or PIL image.
        
    Returns:
        float: SSIM index ranging from -1.0 to 1.0 (1.0 = identical).
    """
    a, b = _arr(a), _arr(b)
    # Multi-channel color images require setting channel_axis for channel-wise mean
    if a.ndim == 3:
        return float(structural_similarity(a, b, channel_axis=2, data_range=255))
    return float(structural_similarity(a, b, data_range=255))


def ber(true_bits, got_bits):
    """Bit error rate: 0.0 = perfect extraction, ~0.5 = random guess. Bit accuracy = 1 - BER.
    
    Args:
        true_bits: Ground-truth embedded bitstream.
        got_bits: Extracted candidate bitstream.
        
    Returns:
        float: Proportion of differing bits in range [0.0, 1.0].
        
    Raises:
        ValueError: If true_bits and got_bits differ in length.
    """
    t = np.asarray(true_bits).ravel()
    g = np.asarray(got_bits).ravel()
    # Check length compatibility between target and recovered bits
    if len(t) != len(g):
        raise ValueError(f"bit vectors must have equal length ({len(t)} vs {len(g)})")
    return float(np.mean(t != g))


def nc(true_bits, got_bits):
    """Normalized correlation of the +/-1 versions (1.0 = identical, 0 = random).
    
    Args:
        true_bits: Ground-truth embedded bitstream.
        got_bits: Extracted candidate bitstream.
        
    Returns:
        float: Normalized correlation value between -1.0 and 1.0.
    """
    # Transform binary {0, 1} domain to bipolar {-1, +1}
    t = 2.0 * np.asarray(true_bits).ravel() - 1
    g = 2.0 * np.asarray(got_bits).ravel() - 1
    return float(np.dot(t, g) / len(t))


def is_successful_extraction(ber_value, threshold=BER_SUCCESS_THRESHOLD):
    """Lenient success: raw BER low enough for an error-correcting code to recover.
    
    Args:
        ber_value (float): Observed Bit Error Rate.
        threshold (float): Upper allowable BER bound for valid ECC recovery.
        
    Returns:
        bool: True if BER is within threshold, False otherwise.
    """
    return ber_value <= threshold
