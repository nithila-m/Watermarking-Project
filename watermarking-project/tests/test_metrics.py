"""Unit test suite for imperceptibility and robustness metrics (tests/test_metrics.py).

Validates:
1. Corner-case handling for identical images (PSNR capping at 100 dB, SSIM ~= 1.0).
2. Input format polymorphism (acceptance of PIL Image objects alongside NumPy arrays).
3. Mathematical precision of Bit Error Rate (BER), Normalized Correlation (NC), and success flags.
4. Error handling when evaluating mismatched vector lengths.
"""
import numpy as np
import pytest
from PIL import Image

from watermark.metrics import ber, is_successful_extraction, nc, psnr, ssim


def test_identical_images(image):
    """Verifies that comparing an image to its exact clone returns ceiling metrics.

    Args:
        image (np.ndarray): Pytest fixture providing a baseline test image.
    """
    # Identical arrays should hit the bounded 100.0 dB PSNR ceiling
    assert psnr(image, image.copy()) == 100.0
    # SSIM between identical images must approximate perfect unity (1.0)
    assert ssim(image, image.copy()) == pytest.approx(1.0)


def test_pil_images_accepted(image):
    """Verifies that metrics seamlessly process PIL Image instances as well as arrays.

    Args:
        image (np.ndarray): Pytest fixture providing a baseline test image.
    """
    # Convert baseline array to PIL Image representation
    a = Image.fromarray(image)
    # Generate an altered PIL Image with a uniform +2 intensity perturbation
    b = Image.fromarray(np.clip(image.astype(int) + 2, 0, 255).astype(np.uint8))

    # A subtle +2 shift should yield high but non-identical PSNR in the range (30 dB, 100 dB)
    assert 30 < psnr(a, b) < 100


def test_ber_and_success():
    """Validates the precision of BER, threshold gating, and Normalized Correlation."""
    # Define ground-truth binary payload and a corrupted candidate bitstream
    secret = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    got = np.array([1, 0, 1, 0, 0, 0, 1, 0])          # Exactly 1 bit error out of 8 bits = 12.5%

    # Calculate Bit Error Rate
    b = ber(secret, got)
    assert b == pytest.approx(0.125)

    # 12.5% BER should pass lenient extraction threshold (<= 15%)
    assert is_successful_extraction(b, threshold=0.15)
    # A 20% BER should fail the 15% threshold cutoff
    assert not is_successful_extraction(0.2, threshold=0.15)

    # Normalized Correlation of identical sequences must equal 1.0
    assert nc(secret, secret) == 1.0


def test_ber_length_mismatch():
    """Verifies that passing vectors of differing lengths raises a ValueError."""
    # Length mismatch (3 bits vs 2 bits) must be rejected
    with pytest.raises(ValueError):
        ber([1, 0, 1], [1, 0])
