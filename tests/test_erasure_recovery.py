"""Deterministic tests for experimental confidence-aware RS erasures."""
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
    totals = np.arange(1, 65, dtype=np.float64)
    first = normalize_bit_confidence(totals)
    second = normalize_bit_confidence(totals.copy())
    assert np.array_equal(first, second)
    assert first[0] == 1 / 64
    assert first[-1] == 1.0

    byte_scores = byte_confidence(first)
    assert np.array_equal(byte_scores, first.reshape(8, 8).min(axis=1))
    assert np.all(byte_scores[1:] > byte_scores[:-1])


def test_thresholding_returns_deterministic_byte_erasure_indices():
    confidence = np.array([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0, 0.3])
    assert erasure_indices(confidence, 0.0) == []
    assert erasure_indices(confidence, 0.25) == [0, 1]
    assert erasure_indices(confidence, 0.5) == [0, 1, 2, 7]
    assert erasure_indices(confidence, 1.0) == [0, 1, 2, 3, 4, 5, 7]
    assert erasure_indices(confidence, 0.5) == erasure_indices(confidence.copy(), 0.5)


def test_reed_solomon_erasure_helper_and_existing_ecc_behavior():
    message = ecc.text_to_bits("TEST")
    coded = ecc.encode(message)
    damaged = coded.copy()
    damaged[[0, 16, 32]] ^= 1

    normal_message, normal_ok = ecc.decode(damaged)
    assert normal_ok is False
    assert not np.array_equal(normal_message, message)

    restored, erasure_ok = decode_with_erasures(damaged, [0, 2, 4])
    assert erasure_ok is True
    assert np.array_equal(restored, message)

    existing_message, existing_ok = ecc.decode(coded)
    assert existing_ok is True
    assert len(existing_message) == MESSAGE_BITS
    assert np.array_equal(existing_message, message)


def test_image_erasure_analysis_is_deterministic_and_has_expected_columns(image):
    cases = [("none", "none")]
    first = analyze_image(image, "fixture", cases, key=2024, seed=2026)
    second = analyze_image(image, "fixture", cases, key=2024, seed=2026)

    assert first == second
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
    assert all(row["oracle_erasure_success"] for row in first)
