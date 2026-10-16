"""Deterministic tests for experimental confidence-aware RS erasures.

Validates the Phase 10 erasure recovery pipeline:
  - Normalization of per-bit voting margins into [0, 1] confidence scores.
  - Conservative "weakest-link" aggregation from bit confidence to byte confidence.
  - Deterministic erasure thresholding and byte index identification.
  - Reed-Solomon decoding with known erasures (expanding correction bound to 4 erasures).
  - Reproducibility and schema conformance of image-level erasure analysis sweeps.
"""
import numpy as np

from scripts.p10_erasure_recovery import (
    analyze_image,
    byte_confidence,
    decode_with_erasures,
    erasure_indices,
    normalize_bit_confidence,
)
from watermark import ecc
from watermark.config import MESSAGE_BITS


def test_confidence_normalization_and_conservative_byte_scores():
    """Verify bit confidence normalization and weakest-link byte aggregation.

    Confirms that:
      1. Margin magnitudes are linearly normalized relative to the maximum observed margin.
      2. Normalization is strictly deterministic across identical inputs.
      3. Byte confidence correctly computes the minimum bit confidence within each 8-bit chunk.
      4. Monotonically increasing bit margins produce strictly increasing byte scores.
    """
    totals = np.arange(1, 65, dtype=np.float64)
    first = normalize_bit_confidence(totals)
    second = normalize_bit_confidence(totals.copy())
    assert np.array_equal(first, second)
    assert first[0] == 1 / 64
    assert first[-1] == 1.0

    # Conservative byte score: the byte is only as confident as its least reliable bit
    byte_scores = byte_confidence(first)
    assert np.array_equal(byte_scores, first.reshape(8, 8).min(axis=1))
    assert np.all(byte_scores[1:] > byte_scores[:-1])


def test_thresholding_returns_deterministic_byte_erasure_indices():
    """Verify that erasure thresholding returns deterministic byte position lists.

    Checks boundary conditions across different thresholds:
      - threshold=0.0 flags no bytes as erasures.
      - threshold=0.25 flags bytes strictly below 0.25 confidence.
      - Higher thresholds flag progressively more byte positions.
      - Invocations on copies return identical index sequences.
    """
    confidence = np.array([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0, 0.3])
    assert erasure_indices(confidence, 0.0) == []
    assert erasure_indices(confidence, 0.25) == [0, 1]
    assert erasure_indices(confidence, 0.5) == [0, 1, 2, 7]
    assert erasure_indices(confidence, 1.0) == [0, 1, 2, 3, 4, 5, 7]
    assert erasure_indices(confidence, 0.5) == erasure_indices(confidence.copy(), 0.5)


def test_reed_solomon_erasure_helper_and_existing_ecc_behavior():
    """Verify RS error-and-erasure decoding capacity versus standard hard decoding.

    Demonstrates that 3 corrupted byte symbols defeat standard algebraic decoding
    (which is capped at floor(4 / 2) = 2 errors), but can be successfully recovered
    when their positions are provided as known erasures (erasure capacity: 2t + e <= 4).
    """
    # Generate canonical 32-bit test message and systematic 64-bit codeword
    message = ecc.text_to_bits("TEST")
    coded = ecc.encode(message)
    damaged = coded.copy()

    # Corrupt bits across 3 distinct byte symbols: byte 0 (bit 0), byte 2 (bit 16), byte 4 (bit 32)
    damaged[[0, 16, 32]] ^= 1

    # Standard Reed-Solomon decoding fails on 3 corrupted symbols
    normal_message, normal_ok = ecc.decode(damaged)
    assert normal_ok is False
    assert not np.array_equal(normal_message, message)

    # Erasure-assisted decoding succeeds by marking the 3 known byte indices as erasures
    restored, erasure_ok = decode_with_erasures(damaged, [0, 2, 4])
    assert erasure_ok is True
    assert np.array_equal(restored, message)

    # Verify baseline Reed-Solomon codec behavior on pristine codewords
    existing_message, existing_ok = ecc.decode(coded)
    assert existing_ok is True
    assert len(existing_message) == MESSAGE_BITS
    assert np.array_equal(existing_message, message)


def test_image_erasure_analysis_is_deterministic_and_has_expected_columns(image):
    """Verify that analyze_image produces repeatable, correctly formatted telemetry rows.

    Args:
        image (np.ndarray): Pytest session fixture providing a 512x512 host image.
    """
    cases = [("none", "none")]
    # Execute identical runs to confirm determinism
    first = analyze_image(image, "fixture", cases, key=2024, seed=2026)
    second = analyze_image(image, "fixture", cases, key=2024, seed=2026)

    assert first == second
    # Evaluates 5 standardized confidence thresholds: 0.0, 0.25, 0.5, 0.75, 1.0
    assert len(first) == 5
    assert {row["threshold"] for row in first} == {0.0, 0.25, 0.5, 0.75, 1.0}

    expected_columns = {
        "image", "attack", "severity", "threshold", "bit_accuracy",
        "actual_erroneous_rs_bytes", "predicted_erasure_bytes",
        "predicted_erasure_indices", "oracle_erasure_indices",
        "normal_ecc_success", "erasure_ecc_success", "oracle_erasure_success",
        "byte_confidences",
    }
    assert expected_columns == set(first[0])
    # Under pristine identity conditions, oracle erasure recovery should succeed unconditionally
    assert all(row["oracle_erasure_success"] for row in first)
