"""Blind extraction: needs only the (possibly edited) image, the key and the payload length."""
import numpy as np

from . import core
from .config import COEF_A, COEF_B


def block_margins(img):
    """Signed margin C[A]-C[B] for every block. Sign = the block's bit vote."""
    y, _ = core.split_luma(img)
    ll, _ = core.dwt(y)
    coefs = core.block_dct(core.to_blocks(ll))
    return coefs[:, COEF_A[0], COEF_A[1]] - coefs[:, COEF_B[0], COEF_B[1]]


def extract(img, n_bits, key, mode="soft", return_details=False):
    """Recover `n_bits` payload bits.

    mode="hard": majority vote of block signs.
    mode="soft": votes weighted by margin, squashed with tanh so one large
                 garbage block (e.g. inpainted) can't outvote many good ones.
    """
    d = block_margins(img)
    assign = core.bit_assignment(len(d), n_bits, key)
    if mode == "hard":
        votes = np.sign(d)
    else:
        scale = np.median(np.abs(d)) + 1e-9
        votes = np.tanh(d / scale)
    totals = np.bincount(assign, weights=votes, minlength=n_bits)
    bits = (totals > 0).astype(np.int64)
    if return_details:
        return bits, {"margins": d, "assign": assign, "totals": totals}
    return bits
