"""Unit test suite for the distortion suite (tests/test_attacks.py).

Verifies three core behavioral guarantees of watermark/attacks.py:
  1. Shape and datatype invariance across all attack kinds and severity levels.
  2. Deterministic reproducibility for stochastic distortions given identical seeds.
  3. Proper segregation of geometric vs. non-geometric attacks in the default suite.
"""
import pytest

from watermark.attacks import PARAMS, SEVERITIES, all_attacks, attack


@pytest.mark.parametrize("kind", list(PARAMS))
@pytest.mark.parametrize("severity", SEVERITIES)
def test_attack_keeps_shape_and_dtype(image, kind, severity):
    """Ensures that every attack preserves the spatial resolution, channels, and uint8 dtype.

    Args:
        image (np.ndarray): Pytest fixture supplying a baseline BGR uint8 test image.
        kind (str): Attack identifier from the PARAMS registry.
        severity (str): Distortion intensity ('light', 'moderate', or 'severe').
    """
    # Execute the attack with a deterministic seed
    out = attack(image, kind, severity, seed=0)

    # Spatial dimensions, channel depth, and data type must remain strictly identical
    assert out.shape == image.shape and out.dtype == image.dtype


def test_attacks_are_reproducible(image):
    """Verifies that randomized attacks produce bit-identical results with matching seeds.

    Args:
        image (np.ndarray): Pytest fixture supplying a baseline BGR uint8 test image.
    """
    # Run the same stochastic attack twice with an identical seed
    a = attack(image, "inpaint_square", "moderate", seed=3)
    b = attack(image, "inpaint_square", "moderate", seed=3)

    # Both resultant arrays must match across every single element
    assert (a == b).all()


def test_default_suite_excludes_geometric():
    """Verifies that geometric attacks requiring resynchronization are omitted by default."""
    # Default attack generator must omit desynchronizing attacks like crop_resize
    kinds = {k for k, _ in all_attacks()}
    assert "crop_resize" not in kinds and {"smooth", "inpaint", "jpeg"} <= kinds

    # Enabling include_geometric must expose crop_resize in the returned suite
    assert "crop_resize" in {k for k, _ in all_attacks(include_geometric=True)}
