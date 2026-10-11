"""Per-block image features for the survival predictor.

One row per watermark block (a 16x16 pixel patch = one 8x8 block of the LL band),
in the same row-major order used by embed/extract. Everything is computed from the
ORIGINAL image, because that's all we have at embedding time.
"""
import cv2
import numpy as np
import pandas as pd

from . import core
from .config import BLOCK, COEF_A, COEF_B

FEATURE_NAMES = [
    "mean", "std", "grad_mean", "edge_density", "saliency",
    "dct_low", "dct_mid", "dct_high", "host_margin", "coef_mag",
    "dist_center", "dist_border", "neigh_std",
]

_saliency = None


def _saliency_map(gray):
    """Spectral-residual saliency in [0, 1] (OpenCV contrib)."""
    global _saliency
    if _saliency is None:
        _saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    ok, sal = _saliency.computeSaliency(gray)
    if not ok:
        return np.zeros(gray.shape, np.float32)
    sal = cv2.GaussianBlur(sal.astype(np.float32), (0, 0), 3)
    return (sal - sal.min()) / (np.ptp(sal) + 1e-9)


def _pixel_blocks(arr, rows, cols):
    """(H, W) image -> (n_blocks, 16, 16) patches aligned with the watermark blocks."""
    p = 2 * BLOCK
    return arr[: rows * p, : cols * p].reshape(rows, p, cols, p).swapaxes(1, 2).reshape(-1, p, p)


def block_features(img):
    """Return a DataFrame (n_blocks x len(FEATURE_NAMES))."""
    rows, cols = core.grid_shape(img.shape)
    y, _ = core.split_luma(img)
    gray = np.clip(y, 0, 255).astype(np.uint8)

    pix = _pixel_blocks(y, rows, cols)
    gx = cv2.Sobel(y, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(y, cv2.CV_64F, 0, 1, ksize=3)
    grad = _pixel_blocks(np.hypot(gx, gy), rows, cols)
    edges = _pixel_blocks(cv2.Canny(gray, 80, 160).astype(np.float64) / 255, rows, cols)
    sal = _pixel_blocks(_saliency_map(gray), rows, cols)

    # DCT statistics of the actual LL-band blocks the watermark lives in
    ll, _ = core.dwt(y)
    c = np.abs(core.block_dct(core.to_blocks(ll)))
    u, v = np.meshgrid(np.arange(BLOCK), np.arange(BLOCK), indexing="ij")
    freq = u + v
    low = (freq > 0) & (freq <= 2)
    mid = (freq >= 3) & (freq <= 6)
    high = freq >= 7
    signed = core.block_dct(core.to_blocks(ll))
    host_margin = np.abs(signed[:, COEF_A[0], COEF_A[1]] - signed[:, COEF_B[0], COEF_B[1]])

    # position (crop and inpainting tend to hit edges / centre differently)
    r, cc = np.divmod(np.arange(rows * cols), cols)
    ry, rx = (r + 0.5) / rows, (cc + 0.5) / cols
    dist_center = np.hypot(ry - 0.5, rx - 0.5)
    dist_border = np.minimum.reduce([ry, rx, 1 - ry, 1 - rx])

    std = pix.std(axis=(1, 2))
    std_grid = std.reshape(rows, cols)
    neigh_std = cv2.blur(std_grid.astype(np.float32), (3, 3)).ravel()  # texture of the neighbourhood

    data = {
        "mean": pix.mean(axis=(1, 2)) / 255,
        "std": std,
        "grad_mean": grad.mean(axis=(1, 2)),
        "edge_density": edges.mean(axis=(1, 2)),
        "saliency": sal.mean(axis=(1, 2)),
        "dct_low": np.log1p(c[:, low].mean(axis=1)),
        "dct_mid": np.log1p(c[:, mid].mean(axis=1)),
        "dct_high": np.log1p(c[:, high].mean(axis=1)),
        "host_margin": np.log1p(host_margin),
        "coef_mag": np.log1p(c[:, COEF_A[0], COEF_A[1]] + c[:, COEF_B[0], COEF_B[1]]),
        "dist_center": dist_center,
        "dist_border": dist_border,
        "neigh_std": neigh_std,
    }
    return pd.DataFrame(data, columns=FEATURE_NAMES)
