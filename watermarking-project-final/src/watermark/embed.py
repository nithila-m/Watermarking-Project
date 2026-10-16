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

    Args:
        img (np.ndarray): Host image as a BGR or grayscale uint8 array.
        bits (array-like): Binary bit sequence (0 or 1) to be embedded.
        strength_map (array-like or float): Per-block embedding strengths or a global scalar.
        key (int): Pseudo-random seed governing block-to-bit assignment.
        skip_ratio (float, optional): Threshold multiple of strength beyond which heavily
            opposed blocks are left unmodified. Defaults to SKIP_RATIO.
        return_skipped (bool, optional): If True, returns a boolean mask indicating
            which blocks were skipped. Defaults to False.

    Returns:
        np.ndarray or tuple[np.ndarray, np.ndarray]: Watermarked uint8 image array,
        or tuple of (watermarked_image, skipped_mask) if return_skipped is True.
    """
    bits = np.asarray(bits, dtype=np.int64)
    # Isolate luminance channel (Y) to carry the watermark perturbation
    y, ycc = core.split_luma(img)
    # Perform 1-level 2D Haar DWT to obtain the approximation subband
    ll, details = core.dwt(y)
    # Divide LL band into non-overlapping 8x8 blocks
    blocks = core.to_blocks(ll)
    n_blocks = blocks.shape[0]

    # Ensure strength map aligns with total block count
    s = np.broadcast_to(np.asarray(strength_map, dtype=np.float64), (n_blocks,))
    # Transform spatial blocks into orthonormal DCT frequency domain
    coefs = core.block_dct(blocks)

    # Determine keyed pseudo-random block-to-bit mapping
    assign = core.bit_assignment(n_blocks, len(bits), key)
    sign = 2.0 * bits[assign] - 1.0                   # +1 for bit 1, -1 for bit 0

    # Extract target mid-frequency coefficient pairs C[A] and C[B]
    a = coefs[:, COEF_A[0], COEF_A[1]]
    b = coefs[:, COEF_B[0], COEF_B[1]]
    d = a - b
    shortfall = np.maximum(s - sign * d, 0.0)         # how far the block is from the target margin
    # Skip blocks where host difference strongly opposes the target bit
    skipped = (sign * d) < -skip_ratio * s            # too expensive to flip: leave it alone
    shortfall[skipped] = 0.0

    # Distribute the required margin shift equally across both coefficients
    change = sign * shortfall
    coefs[:, COEF_A[0], COEF_A[1]] = a + change / 2
    coefs[:, COEF_B[0], COEF_B[1]] = b - change / 2

    # Invert DCT and reconstruct continuous LL subband
    ll_new = core.from_blocks(core.block_idct(coefs), ll.shape)
    # Synthesize modified luminance through Inverse DWT, matching original spatial dimensions
    y_new = core.idwt(ll_new, details)[: y.shape[0], : y.shape[1]]
    # Merge modified luminance with chrominance channels and clamp to uint8 bounds
    out = core.merge_luma(y_new, ycc)
    return (out, skipped) if return_skipped else out


def embed_at_psnr(img, bits, base_map, key, target_psnr=TARGET_PSNR, iters=12):
    """Scale `base_map` by one global factor so the result hits `target_psnr`.

    Used for every configuration so uniform and adaptive are compared at the
    same image quality. Returns (watermarked_img, final_strength_map, psnr).

    Args:
        img (np.ndarray): Host image array (BGR or grayscale uint8).
        bits (array-like): Codeword bitstream to embed.
        base_map (array-like): Relative strength map profile.
        key (int): Key for pseudo-random block assignment.
        target_psnr (float, optional): Objective PSNR ceiling in dB. Defaults to TARGET_PSNR (42.0).
        iters (int, optional): Binary search convergence iterations. Defaults to 12.

    Returns:
        tuple[np.ndarray, np.ndarray, float]:
            - wm: Calibrated watermarked uint8 image.
            - s: Final calibrated per-block embedding strength vector.
            - p: Measured PSNR (in dB) of the watermarked image relative to the host.
    """
    base = np.asarray(base_map, dtype=np.float64)
    lo, hi = 0.0, 1.0

    # Exponential bracket expansion to identify an upper bound for binary search
    while psnr(img, embed(img, bits, base * hi, key)) > target_psnr and hi < 1e3:
        hi *= 2.0

    # Binary search to find the global scalar matching target_psnr
    for _ in range(iters):                            # binary search on the scale factor
        mid = (lo + hi) / 2
        if psnr(img, embed(img, bits, base * mid, key)) > target_psnr:
            lo = mid
        else:
            hi = mid

    # Apply final calibrated strength map
    s = base * lo
    wm = embed(img, bits, s, key)
    p = psnr(img, wm)

    # Issue diagnostic warning if calibration falls short of the intended quality ceiling
    if p < target_psnr - 0.5:
        import warnings
        warnings.warn(f"could not reach {target_psnr} dB (got {p:.1f}); lower SKIP_RATIO")
    return wm, s, p
