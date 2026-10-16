"""Sampling, paired statistics, Colab manifest bits and the parallel runner.

Validates core utility modules supporting the evaluation pipeline:
  - watermark.sampling: Nested prefix preservation and stratified texture balance.
  - watermark.stats: Paired bootstrap confidence intervals and exact McNemar significance tests.
  - watermark.external: Bitstring serialization/deserialization and erasure keyword tagging.
  - watermark.predictor vs. watermark.external: Parity of synthetic disturbance/survival targets.
  - watermark.parallel.pmap: Fault-tolerant multi-worker mapping and error aggregation.
"""
import numpy as np
import pandas as pd
import pytest

from watermark.external import bits_to_str, is_erase, str_to_bits, survival_rows
from watermark.parallel import pmap
from watermark.predictor import build_survival_rows
from watermark.sampling import sample, stratified
from watermark.stats import compare, mcnemar, paired_bootstrap

# Synthetic pool of 300 sequential image identifiers
NAMES = [f"{i:06d}" for i in range(300)]
# Balanced 3-tier texture categorization across the pool
CATS = {n: ["Smooth", "Medium", "Complex"][i % 3] for i, n in enumerate(NAMES)}


def test_samples_are_nested_and_balanced():
    """Verify nested prefix subsets and equal quota allocation in stratified sampling.

    Confirms that:
      1. Slicing smaller n subsets forms a strict subset of larger n draws (nested property).
      2. Stratified draws allocate equal representation across categories (30 each for n=90).
      3. Requesting n=0 falls back gracefully to returning all candidate names sorted.
    """
    # Nested guarantee: smaller draws are strict subsets of larger draws under identical seed
    assert set(sample(NAMES, 20)) <= set(sample(NAMES, 80))

    # Stratified nested guarantee and exact category balance
    small, big = stratified(NAMES, 30, CATS), stratified(NAMES, 90, CATS)
    assert len(big) == 90 and set(small) <= set(big)
    assert pd.Series([CATS[n] for n in big]).value_counts().tolist() == [30, 30, 30]

    # Boundary condition: n=0 returns full sorted sequence
    assert stratified(NAMES, 0, CATS) == sorted(NAMES)


def _results(p_a, p_b, n_images=40, seed=0):
    """Generates synthetic paired experimental trial records for hypothesis testing.

    Args:
        p_a (float): Success probability for configuration A.
        p_b (float): Success probability for configuration B.
        n_images (int, optional): Number of distinct image clusters. Defaults to 40.
        seed (int, optional): Deterministic random seed. Defaults to 0.

    Returns:
        pd.DataFrame: Synthetic paired results table.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_images):
        for k in range(10):
            for cfg, p in (("A", p_a), ("B", p_b)):
                rows.append({
                    "image": f"im{i}",
                    "attack": f"k{k}",
                    "severity": "x",
                    "config": cfg,
                    "success": bool(rng.random() < p),
                })
    return pd.DataFrame(rows)


def test_stats_detect_a_real_difference_and_not_noise():
    """Verify paired bootstrap CI and McNemar tests distinguish real effects from noise.

    Confirms that:
      1. A substantial true difference (95% vs 60%) yields a strictly positive 95% CI lower
         bound (lo > 0) and highly significant McNemar p-value (p < 0.001).
      2. Equal distributions (80% vs 80%) span zero in the CI and are flagged as not significant.
    """
    # Large performance gap should trigger statistical significance
    big = paired_bootstrap(_results(0.95, 0.60), "A", "B")
    assert big["lo"] > 0 and mcnemar(_results(0.95, 0.60), "A", "B")["p_value"] < 0.001

    # Identical performance distributions should fail to reject the null hypothesis
    same = compare(_results(0.8, 0.8), [("A", "B")])
    assert not same.significant.iloc[0]


def test_manifest_bits_round_trip():
    """Verify lossless conversion between bit arrays and manifest strings, plus erase tagging.

    Confirms that:
      1. Binary arrays pack losslessly to binary digit strings and unpack to identical arrays.
      2. Inpainting edits are correctly categorized as erasures via is_erase().
      3. Global distortions (regeneration, VAE encoding) are recognized as noise-like perturbations.
    """
    bits = np.array([0, 1, 1, 0, 0, 0, 1, 0])
    assert bits_to_str(bits) == "01100010"
    assert np.array_equal(str_to_bits(bits_to_str(bits)), bits)

    # Validate classification of edit mechanisms
    assert is_erase("inpaint_severe") and not is_erase("regen_0.15") and not is_erase("vae")


def test_survival_rows_match_original_without_external(image):
    """Verify exact equivalence between survival_rows and build_survival_rows without external edits.

    Ensures that external.survival_rows defaults to the exact synthetic regression
    targets (log_sigma and keep) computed by predictor.build_survival_rows when no
    external real generative AI edits are supplied.

    Args:
        image (np.ndarray): Pytest session fixture providing a 512x512 host image.
    """
    a = build_survival_rows(image, "x")
    b = survival_rows(image, "x")

    # Both target columns must match within numerical floating point precision
    assert np.allclose(a.log_sigma, b.log_sigma) and np.allclose(a.keep, b.keep)
    # Zero external edits recorded in the metadata column
    assert b.n_external.eq(0).all()


def _square(x):
    """Helper job function that raises an exception on specific input for error handling tests."""
    if x == 3:
        raise RuntimeError("bad image")
    return x * x


@pytest.mark.parametrize("workers", [1, 2])
def test_pmap_collects_results_and_errors(workers):
    """Verify that pmap executes across processes, preserves outputs, and handles exceptions safely.

    Tests both single-process (workers=1) and multi-process pool (workers=2) execution modes,
    confirming that an exception raised by a single item does not terminate the batch.

    Args:
        workers (int): Parallel worker concurrency level.
    """
    out = {item: (res, err) for item, res, err in pmap(_square, range(5), workers=workers)}

    # Successful task outcome verification
    assert out[4] == (16, None)

    # Handled failure verification: result is None and traceback contains the raised message
    assert out[3][0] is None and "bad image" in out[3][1]
