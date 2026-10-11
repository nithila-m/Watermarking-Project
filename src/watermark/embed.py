"""Blind DWT-DCT watermark embedding with a per-block strength map.

Rule (coefficient-pair comparison, per 8x8 block of the LL band):
    bit 1  ->  force  C[A] - C[B] >= +s
    bit 0  ->  force  C[A] - C[B] <= -s
The extractor only reads the SIGN of C[A] - C[B], so s can differ per block
and the extractor never needs the strength map. That is what makes the
adaptive version possible.
"""
import numpy as np

from . import core
from .config import COEF_A, COEF_B, SKIP_RATIO, TARGET_PSNR
from .metrics import psnr


def embed(img, bits, strength_map, key, skip_ratio=SKIP_RATIO, return_skipped=False):
    """Embed `bits` (array of 0/1) into `img` (BGR or gray uint8).

    strength_map: shape (n_blocks,) or scalar. Returns the watermarked uint8 image
    (and the boolean mask of skipped blocks if return_skipped=True).
    """
    bits = np.asarray(bits, dtype=np.int64)
    y, ycc = core.split_luma(img)
    ll, details = core.dwt(y)
    blocks = core.to_blocks(ll)
    n_blocks = blocks.shape[0]

    s = np.broadcast_to(np.asarray(strength_map, dtype=np.float64), (n_blocks,))
    coefs = core.block_dct(blocks)

    assign = core.bit_assignment(n_blocks, len(bits), key)
    sign = 2.0 * bits[assign] - 1.0                  # +1 for bit 1, -1 for bit 0

    a = coefs[:, COEF_A[0], COEF_A[1]]
    b = coefs[:, COEF_B[0], COEF_B[1]]
    d = a - b
    shortfall = np.maximum(s - sign * d, 0.0)         # how far the block is from the target margin
    skipped = (sign * d) < -skip_ratio * s            # too expensive to flip: leave it alone
    shortfall[skipped] = 0.0
    change = sign * shortfall
    coefs[:, COEF_A[0], COEF_A[1]] = a + change / 2
    coefs[:, COEF_B[0], COEF_B[1]] = b - change / 2

    ll_new = core.from_blocks(core.block_idct(coefs), ll.shape)
    y_new = core.idwt(ll_new, details)[: y.shape[0], : y.shape[1]]
    out = core.merge_luma(y_new, ycc)
    return (out, skipped) if return_skipped else out


def embed_at_psnr(img, bits, base_map, key, target_psnr=TARGET_PSNR, iters=12):
    """Scale `base_map` by one global factor so the result hits `target_psnr`.

    Used for every configuration so uniform and adaptive are compared at the
    same image quality. Returns (watermarked_img, final_strength_map, psnr).
    """
    base = np.asarray(base_map, dtype=np.float64)
    lo, hi = 0.0, 1.0
    while psnr(img, embed(img, bits, base * hi, key)) > target_psnr and hi < 1e3:
        hi *= 2.0
    for _ in range(iters):                            # binary search on the scale factor
        mid = (lo + hi) / 2
        if psnr(img, embed(img, bits, base * mid, key)) > target_psnr:
            lo = mid
        else:
            hi = mid
    s = base * lo
    wm = embed(img, bits, s, key)
    p = psnr(img, wm)
    if p < target_psnr - 0.5:
        import warnings
        warnings.warn(f"could not reach {target_psnr} dB (got {p:.1f}); lower SKIP_RATIO")
    return wm, s, p
