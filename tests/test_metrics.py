import numpy as np
import pytest
from PIL import Image

from watermark.metrics import ber, is_successful_extraction, nc, psnr, ssim


def test_identical_images(image):
    assert psnr(image, image.copy()) == 100.0
    assert ssim(image, image.copy()) == pytest.approx(1.0)


def test_pil_images_accepted(image):
    a = Image.fromarray(image)
    b = Image.fromarray(np.clip(image.astype(int) + 2, 0, 255).astype(np.uint8))
    assert 30 < psnr(a, b) < 100


def test_ber_and_success():
    secret = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    got = np.array([1, 0, 1, 0, 0, 0, 1, 0])          # 1 error in 8 = 12.5 %
    b = ber(secret, got)
    assert b == pytest.approx(0.125)
    assert is_successful_extraction(b, threshold=0.15)
    assert not is_successful_extraction(0.2, threshold=0.15)
    assert nc(secret, secret) == 1.0


def test_ber_length_mismatch():
    with pytest.raises(ValueError):
        ber([1, 0, 1], [1, 0])
