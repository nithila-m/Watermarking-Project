import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr(a, b):
    if np.array_equal(a, b):
        return 100.0
    return float(peak_signal_noise_ratio(a, b, data_range=255))


def ssim(a, b):
    if a.ndim == 3:
        return float(structural_similarity(a, b, channel_axis=2, data_range=255))
    return float(structural_similarity(a, b, data_range=255))


def ber(true_bits, got_bits):
    """Bit error rate. Bit accuracy = 1 - BER."""
    return float(np.mean(np.asarray(true_bits) != np.asarray(got_bits)))


def nc(true_bits, got_bits):
    """Normalized correlation of the +/-1 versions (1.0 = identical, 0 = random)."""
    t = 2.0 * np.asarray(true_bits) - 1
    g = 2.0 * np.asarray(got_bits) - 1
    return float(np.dot(t, g) / len(t))
