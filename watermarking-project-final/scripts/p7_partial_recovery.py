r"""Measure how repeated block votes and Reed-Solomon recover a partial payload.

Run with: python scripts\p7_partial_recovery.py
"""
from unittest.mock import patch

import numpy as np

from watermark import ecc
from watermark.config import BLOCK, IMG_SIZE, MESSAGE_BITS, PAYLOAD_BITS
from watermark.core import bit_assignment
from watermark.extract import extract

CORRUPTION_LEVELS = (0, 10, 25, 40, 50, 60, 75)
KEY = 2024
SEED = 2026
N_BLOCKS = (IMG_SIZE // (2 * BLOCK)) ** 2


def run_experiment(seed=SEED):
    """Exercise the real soft-voting extractor with controlled synthetic margins."""
    message = ecc.text_to_bits("TEST")
    coded = ecc.encode(message)
    if len(message) != MESSAGE_BITS or len(coded) != PAYLOAD_BITS:
        raise ValueError("ECC output does not match configured message/payload lengths")

    assignment = bit_assignment(N_BLOCKS, PAYLOAD_BITS, KEY)
    copies = np.bincount(assignment, minlength=PAYLOAD_BITS)
    if not np.all(copies == N_BLOCKS // PAYLOAD_BITS):
        raise ValueError("experiment requires equal repeated observations per coded bit")

    rng = np.random.default_rng(seed)
    results = []
    for corruption_percent in CORRUPTION_LEVELS:
        corrupt_per_bit = int(np.floor(corruption_percent * copies[0] / 100 + 0.5))
        observations = coded[assignment].copy()
        for bit_index in range(PAYLOAD_BITS):
            block_indices = np.flatnonzero(assignment == bit_index)
            corrupted = rng.permutation(block_indices)[:corrupt_per_bit]
            observations[corrupted] ^= 1

        # Equal-strength +/-1 margins make the production soft vote a repeated
        # majority vote, while keeping its tanh weighting and tie handling intact.
        margins = 2 * observations.astype(np.float64) - 1
        with patch("watermark.extract.block_margins", return_value=margins):
            voted = extract(np.empty((0, 0)), PAYLOAD_BITS, KEY, mode="soft")
        decoded, ecc_ok = ecc.decode(voted)
        reconstructed = bool(ecc_ok and np.array_equal(decoded, message))
        results.append({
            "requested_percent": corruption_percent,
            "actual_percent": 100 * corrupt_per_bit / int(copies[0]),
            "coded_accuracy": float(np.mean(voted == coded)),
            "ecc_ok": bool(ecc_ok),
            "reconstructed": reconstructed,
            "decoded_text": ecc.bits_to_text(decoded),
            "observations_per_bit": int(copies[0]),
        })
    return results


def main():
    print(f"Partial-watermark recovery (seed={SEED}, message=TEST, copies/bit={N_BLOCKS // PAYLOAD_BITS})")
    print("corruption  actual  coded accuracy  ECC decode  message recovered")
    for result in run_experiment():
        print(
            f"{result['requested_percent']:>9}%"
            f"  {result['actual_percent']:>5.1f}%"
            f"  {result['coded_accuracy']:>13.3f}"
            f"  {str(result['ecc_ok']):>10}"
            f"  {str(result['reconstructed']):>17}"
        )


if __name__ == "__main__":
    main()
