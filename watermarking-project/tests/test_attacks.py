import pytest

from watermark.attacks import PARAMS, SEVERITIES, all_attacks, attack


@pytest.mark.parametrize("kind", list(PARAMS))
@pytest.mark.parametrize("severity", SEVERITIES)
def test_attack_keeps_shape_and_dtype(image, kind, severity):
    out = attack(image, kind, severity, seed=0)
    assert out.shape == image.shape and out.dtype == image.dtype


def test_attacks_are_reproducible(image):
    a = attack(image, "inpaint_square", "moderate", seed=3)
    b = attack(image, "inpaint_square", "moderate", seed=3)
    assert (a == b).all()


def test_default_suite_excludes_geometric():
    kinds = {k for k, _ in all_attacks()}
    assert "crop_resize" not in kinds and {"smooth", "inpaint", "jpeg"} <= kinds
    assert "crop_resize" in {k for k, _ in all_attacks(include_geometric=True)}
