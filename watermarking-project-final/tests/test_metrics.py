"""Unit test suite for imperceptibility and robustness metrics (tests/test_metrics.py).

Validates:
1. Boundary condition handling for identical images (PSNR capping at 100 dB, SSIM ~= 1.0).
2. Input polymorphism (proper handling of PIL Image instances alongside NumPy arrays).
3. Exact calculations for Bit Error Rate (BER), Normalized Correlation (NC), and success gating.
4. Exception raising when comparing bit arrays of mismatched lengths.
"""
import numpy as np
import pytest
from PIL import Image

from watermark.metrics import ber, is_successful_extraction, nc, psnr, ssim


def test_identical_images(image):
    """Verifies that evaluating an image against an identical copy yields ceiling metric values.

    Args:
        image (np.ndarray): Pytest fixture supplying a baseline BGR test image.
    """
    # Identical arrays should hit the bounded ceiling of 100.0 dB rather than returning infinity
    assert psnr(image, image.copy()) == 100.0
    # SSIM of identical images must approximate 1.0
    assert ssim(image, image.copy()) == pytest.approx(1.0)


def test_pil_images_accepted(image):
    """Verifies that metric calculation functions accept PIL Image instances directly.

    Args:
        image (np.ndarray): Pytest fixture supplying a baseline BGR test image.
    """
    # Convert array to PIL Image format
    a = Image.fromarray(image)
    # Apply a slight uniform offset of +2 intensity levels
    b = Image.fromarray(np.clip(image.astype(int) + 2, 0, 255).astype(np.uint8))

    # A subtle global shift should produce a high, realistic PSNR between 30 dB and 100 dB
    assert 30 < psnr(a, b) < 100


def test_ber_and_success():
    """Validates the mathematical calculation of BER, NC, and threshold-based success gating."""
    # Define ground truth and an extracted sequence with 1 altered bit out of 8 (12.5% error)
    secret = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    got = np.array([1, 0, 1, 0, 0, 0, 1, 0])          # 1 error in 8 = 12.5 %

    # Confirm BER equals 0.125
    b = ber(secret, got)
    assert b == pytest.approx(0.125)

    # 12.5% BER should pass a 15% threshold
    assert is_successful_extraction(b, threshold=0.15)
    # 20% BER should fail a 15% threshold
    assert not is_successful_extraction(0.2, threshold=0.15)

    # Normalized Correlation of identical sequences must be exactly 1.0
    assert nc(secret, secret) == 1.0


def test_ber_length_mismatch():
    """Verifies that computing BER between vectors of unequal lengths raises a ValueError."""
    # Length mismatch (3 bits vs 2 bits) must raise an exception
    with pytest.raises(ValueError):
        ber([1, 0, 1], [1, 0])
