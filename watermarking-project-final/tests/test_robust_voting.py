"""Tests for P11 experimental aggregation over existing extraction margins.

Validates the Phase 11 robust voting strategies:
  - Determinism and output shape conformance across all candidate aggregation rules.
  - Per-bit median margin aggregation resisting single extreme outlier blocks.
  - Symmetric trimmed mean aggregation removing outlier extremes.
  - Sign majority voting behavior, including zero-margin abstention and tie-breaking to bit 0.
  - Linear confidence/magnitude-weighted sign aggregation.
  - Bit-for-bit equivalence between experimental 'current_soft' voting and the production
    tanh-weighted extractor in watermark.extract.
"""
from unittest.mock import patch

import numpy as np

from scripts.p11_robust_voting import METHODS, aggregate_votes
from watermark import core
from watermark.config import PAYLOAD_BITS
from watermark.extract import extract


def test_all_voting_methods_are_deterministic_and_return_payload_length():
    """Verify that all voting methods execute deterministically and return PAYLOAD_BITS bits.

    Confirms that:
      1. Every registered strategy in METHODS produces an output entry.
      2. Consecutive executions with identical margins and assignments return identical bit arrays.
      3. The decoded bit array length for every method equals PAYLOAD_BITS (64).
    """
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
    """Verify that median margin voting isolates the median observation per bit.

    Demonstrates robustness to extreme outliers (e.g. +100.0 or -100.0) resulting from
    severe spatial edits:
      - Bit 0: margins [1.0, 2.0, 100.0] -> median is 2.0 (> 0) -> decodes to bit 1.
      - Bit 1: margins [-1.0, -2.0, -100.0] -> median is -2.0 (<= 0) -> decodes to bit 0.
    """
    margins = np.array([1.0, 2.0, 100.0, -1.0, -2.0, -100.0])
    assignment = np.array([0, 0, 0, 1, 1, 1])

    result = aggregate_votes(margins, assignment, n_bits=2)

    assert np.array_equal(result["median_margin"], [1, 0])


def test_trimmed_mean_removes_one_low_and_high_observation():
    """Verify symmetric trimming of extreme margins across redundant block copies.

    Confirms that:
      - Bit 0 has 4 observations: [-100.0, 2.0, 3.0, 100.0]. Trimming the lowest (-100.0)
        and highest (100.0) leaves [2.0, 3.0], whose mean is 2.5 (> 0) -> decodes to bit 1.
      - Bit 1 has only 2 observations: [-2.0, -3.0]. Insufficient elements for trimming
        triggers fallback to arithmetic mean (-2.5 <= 0) -> decodes to bit 0.
    """
    margins = np.array([-100.0, 2.0, 3.0, 100.0, -2.0, -3.0])
    assignment = np.array([0, 0, 0, 0, 1, 1])

    result = aggregate_votes(margins, assignment, n_bits=2)

    # Four observations: trimming leaves [2, 3]. Two observations fall back to mean.
    assert np.array_equal(result["trimmed_mean"], [1, 0])


def test_sign_majority_ties_resolve_to_zero():
    """Verify unweighted sign majority voting and strict tie resolution.

    Confirms that:
      - Bit 0: margins [1.0, -1.0, 0.0] yield signs [+1, -1, 0]. The sum is 0 (tie),
        which evaluates to > 0 as False -> decodes to bit 0.
      - Bit 1: margins [-2.0, 1.0, 1.0] yield signs [-1, +1, +1]. The sum is +1 (> 0),
        which evaluates to True -> decodes to bit 1.
    """
    margins = np.array([1.0, -1.0, 0.0, -2.0, 1.0, 1.0])
    assignment = np.array([0, 0, 0, 1, 1, 1])

    result = aggregate_votes(margins, assignment, n_bits=2)

    # Bit 0 ties (+1 and -1; zero abstains); bit 1 has two positive signs.
    assert np.array_equal(result["sign_majority"], [0, 1])


def test_confidence_weighted_sign_uses_magnitude_weight():
    """Verify sign voting weighted directly by margin absolute magnitude.

    Confirms linear weighting:
      - Bit 0: margins [+10.0, -2.0] -> sum is +10.0 - 2.0 = +8.0 (> 0) -> decodes to bit 1.
      - Bit 1: margins [+1.0, -3.0] -> sum is +1.0 - 3.0 = -2.0 (<= 0) -> decodes to bit 0.
    """
    margins = np.array([10.0, -2.0, 1.0, -3.0])
    assignment = np.array([0, 0, 1, 1])

    result = aggregate_votes(margins, assignment, n_bits=2)

    # Weighted signed sums are 10 - 2 = 8 and 1 - 3 = -2.
    assert np.array_equal(result["confidence_weighted_sign"], [1, 0])


def test_current_method_matches_unchanged_production_extractor():
    """Assert exact parity between 'current_soft' experimental voting and production extractor.

    Patches watermark.extract.block_margins with synthetic margins and confirms that
    aggregate_votes(margins, assign)['current_soft'] exactly matches the output of
    watermark.extract.extract(..., mode='soft').
    """
    n_blocks = 128
    n_bits = PAYLOAD_BITS
    key = 2024
    assignment = core.bit_assignment(n_blocks, n_bits, key)
    margins = np.linspace(-3.0, 4.0, n_blocks)

    experimental = aggregate_votes(margins, assignment, n_bits)["current_soft"]
    with patch("watermark.extract.block_margins", return_value=margins):
        production = extract(np.empty((0, 0)), n_bits, key)

    assert np.array_equal(experimental, production)
