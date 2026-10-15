"""Unit test suite for the adversarial distortion engine (tests/test_attacks.py).

Validates that all classical and AI-based attack functions adhere to:
1. Shape and datatype invariance (must preserve dimensions and uint8 format).
2. Deterministic reproducibility across identical pseudo-random seeds.
3. Proper filtering of geometric versus non-geometric attack suites.
"""
import pytest

from watermark.attacks import PARAMS, SEVERITIES, all_attacks, attack


@pytest.mark.parametrize("kind", list(PARAMS))
@pytest.mark.parametrize("severity", SEVERITIES)
def test_attack_keeps_shape_and_dtype(image, kind, severity):
    """Verifies that every distortion maintains the original image shape and uint8 dtype.

    Args:
        image (np.ndarray): Pytest fixture providing a sample BGR image.
        kind (str): Distortion type name from PARAMS registry.
        severity (str): Intensity level ('light', 'moderate', 'severe').
    """
    # Execute attack on input image fixture
    out = attack(image, kind, severity, seed=0)

    # Assert spatial dimensions, channel count, and memory representation remain unaltered
    assert out.shape == image.shape and out.dtype == image.dtype


def test_attacks_are_reproducible(image):
    """Verifies that stochastic attacks yield identical outputs given identical seeds.

    Args:
        image (np.ndarray): Pytest fixture providing a sample BGR image.
    """
    # Generate two independent attack instances with matching seed parameters
    a = attack(image, "inpaint_square", "moderate", seed=3)
    b = attack(image, "inpaint_square", "moderate", seed=3)

    # Verify bit-for-bit equivalence between both outputs
    assert (a == b).all()


def test_default_suite_excludes_geometric():
    """Verifies default evaluation suite excludes desynchronizing geometric attacks.

    Crop-resize requires a synchronization step outside the blind extraction scope
    and must only appear when explicitly requested.
    """
    # Verify standard default attack registry excludes geometric shifts
    kinds = {k for k, _ in all_attacks()}
    assert "crop_resize" not in kinds and {"smooth", "inpaint", "jpeg"} <= kinds

    # Verify geometric distortions are included when explicitly enabled
    assert "crop_resize" in {k for k, _ in all_attacks(include_geometric=True)}
