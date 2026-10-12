"""Tests for P11 experimental aggregation over existing extraction margins."""
from unittest.mock import patch

import numpy as np

from scripts.p11_robust_voting import METHODS, aggregate_votes
from watermark import core
from watermark.config import PAYLOAD_BITS
from watermark.extract import extract


def test_all_voting_methods_are_deterministic_and_return_payload_length():
    assignment = np.repeat(np.arange(PAYLOAD_BITS), 3)
    margins = np.tile(np.array([-2.0, 1.0, 4.0]), PAYLOAD_BITS)

    first = aggregate_votes(margins, assignment)
    second = aggregate_votes(margins.copy(), assignment.copy())

    assert tuple(first) == METHODS
    assert first.keys() == second.keys()
    for method in METHODS:
        assert np.array_equal(first[method], second[method])
        assert len(first[method]) == PAYLOAD_BITS


def test_median_margin_voting_uses_per_bit_median():
    margins = np.array([1.0, 2.0, 100.0, -1.0, -2.0, -100.0])
    assignment = np.array([0, 0, 0, 1, 1, 1])

    result = aggregate_votes(margins, assignment, n_bits=2)

    assert np.array_equal(result["median_margin"], [1, 0])


def test_trimmed_mean_removes_one_low_and_high_observation():
    margins = np.array([-100.0, 2.0, 3.0, 100.0, -2.0, -3.0])
    assignment = np.array([0, 0, 0, 0, 1, 1])

    result = aggregate_votes(margins, assignment, n_bits=2)

    # Four observations: trimming leaves [2, 3]. Two observations fall back to mean.
    assert np.array_equal(result["trimmed_mean"], [1, 0])


def test_sign_majority_ties_resolve_to_zero():
    margins = np.array([1.0, -1.0, 0.0, -2.0, 1.0, 1.0])
    assignment = np.array([0, 0, 0, 1, 1, 1])

    result = aggregate_votes(margins, assignment, n_bits=2)

    # Bit 0 ties (+1 and -1; zero abstains); bit 1 has two positive signs.
    assert np.array_equal(result["sign_majority"], [0, 1])


def test_confidence_weighted_sign_uses_magnitude_weight():
    margins = np.array([10.0, -2.0, 1.0, -3.0])
    assignment = np.array([0, 0, 1, 1])

    result = aggregate_votes(margins, assignment, n_bits=2)

    # Weighted signed sums are 10 - 2 = 8 and 1 - 3 = -2.
    assert np.array_equal(result["confidence_weighted_sign"], [1, 0])


def test_current_method_matches_unchanged_production_extractor():
    n_blocks = 128
    n_bits = PAYLOAD_BITS
    key = 2024
    assignment = core.bit_assignment(n_blocks, n_bits, key)
    margins = np.linspace(-3.0, 4.0, n_blocks)

    experimental = aggregate_votes(margins, assignment, n_bits)["current_soft"]
    with patch("watermark.extract.block_margins", return_value=margins):
        production = extract(np.empty((0, 0)), n_bits, key)

    assert np.array_equal(experimental, production)
