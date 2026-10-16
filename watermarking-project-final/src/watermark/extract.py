"""Blind extraction: needs only the (possibly edited) image, the key and the payload length.

Extracts the embedded watermark payload directly from the frequency domain without
requiring the unwatermarked host image or the adaptive strength map. Operates by:
  1. Extracting the luminance channel (Y) and decomposing it via 1-level 2D Haar DWT.
  2. Transforming 8x8 spatial blocks of the LL subband via orthonormal 2D DCT.
  3. Computing the signed difference margin between coefficient pairs C[A] and C[B].
  4. Accumulating per-bit votes using either unweighted sign majority ("hard") or
     median-scaled tanh soft voting ("soft") across keyed block assignments.
"""
import numpy as np

from . import core
from .config import COEF_A, COEF_B


def block_margins(img):
    """Signed margin C[A]-C[B] for every block. Sign = the block's bit vote.

    Decomposes the image into the wavelet domain, sections the LL approximation
    band into 8x8 blocks, and measures the difference between designated
    mid-frequency DCT coefficients C[A] and C[B].

    Args:
        img (np.ndarray): Input BGR or grayscale image array.

    Returns:
        np.ndarray: 1-D array of signed coefficient differences of length n_blocks.
    """
    # Isolate luminance channel and compute 1-level 2D DWT
    y, _ = core.split_luma(img)
    ll, _ = core.dwt(y)

    # Slice LL band into 8x8 blocks and compute orthonormal 2D DCT
    coefs = core.block_dct(core.to_blocks(ll))

    # Evaluate the differential margin between the paired coefficients
    return coefs[:, COEF_A[0], COEF_A[1]] - coefs[:, COEF_B[0], COEF_B[1]]


def extract(img, n_bits, key, mode="soft", return_details=False):
    """Recover `n_bits` payload bits.

    mode="hard": majority vote of block signs.
    mode="soft": votes weighted by margin, squashed with tanh so one large
                 garbage block (e.g. inpainted) can't outvote many good ones.

    Args:
        img (np.ndarray): Possibly attacked or edited image array.
        n_bits (int): Total expected codeword bits (e.g., 64).
        key (int): Pseudo-random seed used to re-generate the block-to-bit mapping.
        mode (str, optional): Voting aggregation rule, either 'soft' or 'hard'.
            Defaults to 'soft'.
        return_details (bool, optional): If True, returns diagnostic dictionaries
            containing raw margins, block assignments, and per-bit vote totals.
            Defaults to False.

    Returns:
        np.ndarray or tuple[np.ndarray, dict]: Decoded binary bit array (int64),
        or tuple of (bits, details_dict) if return_details is True.
    """
    # Compute signed coefficient differences across all blocks
    d = block_margins(img)

    # Reconstruct the deterministic keyed block-to-bit mapping permutation
    assign = core.bit_assignment(len(d), n_bits, key)

    if mode == "hard":
        # Unweighted sign-majority voting: {-1, 0, +1}
        votes = np.sign(d)
    else:
        # Scale margins by median absolute deviation and bound through hyperbolic tangent
        scale = np.median(np.abs(d)) + 1e-9
        votes = np.tanh(d / scale)

    # Sum votes for each assigned bit index across all redundant observations
    totals = np.bincount(assign, weights=votes, minlength=n_bits)

    # Decode bit as 1 if aggregate consensus is strictly positive; otherwise 0
    bits = (totals > 0).astype(np.int64)

    if return_details:
        return bits, {"margins": d, "assign": assign, "totals": totals}
    return bits
