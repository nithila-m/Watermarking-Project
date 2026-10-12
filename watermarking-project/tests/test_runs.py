"""Sampling, paired statistics, Colab manifest bits and the parallel runner."""
import numpy as np
import pandas as pd
import pytest

from watermark.external import bits_to_str, is_erase, str_to_bits, survival_rows
from watermark.parallel import pmap
from watermark.predictor import build_survival_rows
from watermark.sampling import sample, stratified
from watermark.stats import compare, mcnemar, paired_bootstrap

NAMES = [f"{i:06d}" for i in range(300)]
CATS = {n: ["Smooth", "Medium", "Complex"][i % 3] for i, n in enumerate(NAMES)}


def test_samples_are_nested_and_balanced():
    assert set(sample(NAMES, 20)) <= set(sample(NAMES, 80))
    small, big = stratified(NAMES, 30, CATS), stratified(NAMES, 90, CATS)
    assert len(big) == 90 and set(small) <= set(big)
    assert pd.Series([CATS[n] for n in big]).value_counts().tolist() == [30, 30, 30]
    assert stratified(NAMES, 0, CATS) == sorted(NAMES)


def _results(p_a, p_b, n_images=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_images):
        for k in range(10):
            for cfg, p in (("A", p_a), ("B", p_b)):
                rows.append({"image": f"im{i}", "attack": f"k{k}", "severity": "x",
                             "config": cfg, "success": bool(rng.random() < p)})
    return pd.DataFrame(rows)


def test_stats_detect_a_real_difference_and_not_noise():
    big = paired_bootstrap(_results(0.95, 0.60), "A", "B")
    assert big["lo"] > 0 and mcnemar(_results(0.95, 0.60), "A", "B")["p_value"] < 0.001
    same = compare(_results(0.8, 0.8), [("A", "B")])
    assert not same.significant.iloc[0]


def test_manifest_bits_round_trip():
    bits = np.array([0, 1, 1, 0, 0, 0, 1, 0])
    assert bits_to_str(bits) == "01100010"
    assert np.array_equal(str_to_bits(bits_to_str(bits)), bits)
    assert is_erase("inpaint_severe") and not is_erase("regen_0.15") and not is_erase("vae")


def test_survival_rows_match_original_without_external(image):
    a = build_survival_rows(image, "x")
    b = survival_rows(image, "x")
    assert np.allclose(a.log_sigma, b.log_sigma) and np.allclose(a.keep, b.keep)
    assert b.n_external.eq(0).all()


def _square(x):
    if x == 3:
        raise RuntimeError("bad image")
    return x * x


@pytest.mark.parametrize("workers", [1, 2])
def test_pmap_collects_results_and_errors(workers):
    out = {item: (res, err) for item, res, err in pmap(_square, range(5), workers=workers)}
    assert out[4] == (16, None)
    assert out[3][0] is None and "bad image" in out[3][1]
