"""Imperceptibility and robustness metrics: PSNR, SSIM, BER, NC, extraction success.

Images can be numpy arrays (BGR or gray uint8, the pipeline's format) or PIL images.
"""
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from .config import BER_SUCCESS_THRESHOLD


def _arr(img):
    """numpy array from a numpy array or a PIL image."""
    return img if isinstance(img, np.ndarray) else np.asarray(img)


def psnr(a, b):
    """Peak signal-to-noise ratio in dB. Identical images return 100.0 (instead of inf)."""
    a, b = _arr(a), _arr(b)
    if np.array_equal(a, b):
        return 100.0
    return float(peak_signal_noise_ratio(a, b, data_range=255))


def ssim(a, b):
    """Structural similarity, averaged over colour channels for colour images."""
    a, b = _arr(a), _arr(b)
    if a.ndim == 3:
        return float(structural_similarity(a, b, channel_axis=2, data_range=255))
    return float(structural_similarity(a, b, data_range=255))


def ber(true_bits, got_bits):
    """Bit error rate: 0.0 = perfect extraction, ~0.5 = random guess. Bit accuracy = 1 - BER."""
    t = np.asarray(true_bits).ravel()
    g = np.asarray(got_bits).ravel()
    if len(t) != len(g):
        raise ValueError(f"bit vectors must have equal length ({len(t)} vs {len(g)})")
    return float(np.mean(t != g))


def nc(true_bits, got_bits):
    """Normalized correlation of the +/-1 versions (1.0 = identical, 0 = random)."""
    t = 2.0 * np.asarray(true_bits).ravel() - 1
    g = 2.0 * np.asarray(got_bits).ravel() - 1
    return float(np.dot(t, g) / len(t))


def is_successful_extraction(ber_value, threshold=BER_SUCCESS_THRESHOLD):
    """Lenient success: raw BER low enough for an error-correcting code to recover."""
    return ber_value <= threshold
