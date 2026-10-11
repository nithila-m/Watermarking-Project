"""Transforms shared by embed and extract: colour space, DWT, block DCT, keyed bit assignment."""
import cv2
import numpy as np
import pywt
from scipy.fft import dctn, idctn

from .config import BLOCK, WAVELET


def load_image(path):
    """Read an image as BGR uint8 (grayscale images are returned 2-D)."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def save_image(path, img):
    cv2.imwrite(str(path), img)


def split_luma(img):
    """Return (Y as float64, YCrCb image or None). Watermark lives in Y only."""
    if img.ndim == 2:
        return img.astype(np.float64), None
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb).astype(np.float64)
    return ycc[..., 0].copy(), ycc


def merge_luma(y, ycc):
    y = np.clip(np.round(y), 0, 255)
    if ycc is None:
        return y.astype(np.uint8)
    out = ycc.copy()
    out[..., 0] = y
    out = np.clip(np.round(out), 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_YCrCb2BGR)


def dwt(y):
    ll, details = pywt.dwt2(y, WAVELET)
    return ll, details


def idwt(ll, details):
    return pywt.idwt2((ll, details), WAVELET)


def grid_shape(img_shape):
    """Number of block rows/cols for an image of this shape."""
    h, w = img_shape[:2]
    if h % (2 * BLOCK) or w % (2 * BLOCK):
        raise ValueError(f"image size {h}x{w} must be a multiple of {2 * BLOCK}; run prepare_images first")
    return h // (2 * BLOCK), w // (2 * BLOCK)


def to_blocks(band):
    """(H, W) -> (n_blocks, 8, 8), row-major block order."""
    h, w = band.shape
    return (band.reshape(h // BLOCK, BLOCK, w // BLOCK, BLOCK)
                .swapaxes(1, 2).reshape(-1, BLOCK, BLOCK))


def from_blocks(blocks, shape):
    h, w = shape
    return (blocks.reshape(h // BLOCK, w // BLOCK, BLOCK, BLOCK)
                  .swapaxes(1, 2).reshape(h, w))


def block_dct(blocks):
    return dctn(blocks, axes=(1, 2), norm="ortho")


def block_idct(coefs):
    return idctn(coefs, axes=(1, 2), norm="ortho")


def bit_assignment(n_blocks, n_bits, key):
    """Keyed pseudo-random map block -> payload bit index.

    Every bit is repeated ~n_blocks/n_bits times, scattered over the image so a
    local edit (inpainting, crop) only removes some copies of each bit.
    """
    rng = np.random.default_rng(key)
    order = rng.permutation(n_blocks)
    assign = np.empty(n_blocks, dtype=np.int64)
    assign[order] = np.arange(n_blocks) % n_bits
    return assign
