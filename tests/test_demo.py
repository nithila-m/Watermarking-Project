"""Tests for the final demo pipeline without launching the UI."""
import numpy as np
import pytest

from scripts.demo import (
    decode_uploaded_image,
    load_demo_image,
    run_demo,
)


def test_clean_text_payload_embeds_extracts_and_reconstructs(image):
    result = run_demo(image, "TEST", key=2024, seed=2026)

    assert result["original_text"] == "TEST"
    assert result["recovered_text"] == "TEST"
    assert result["ecc_success"] is True
    assert result["exact_recovery"] is True
    assert result["bit_accuracy"] >= 0.0
    assert result["ber"] >= 0.0
    assert result["psnr"] > 0.0
    assert result["ssim"] > 0.0
    assert result["watermarked_image"].shape == (512, 512, 3)
    assert np.array_equal(result["attacked_image"], result["watermarked_image"])


def test_demo_is_deterministic_for_fixed_key_and_attack_seed(image):
    first = run_demo(image, "TEST", "blur", "moderate", key=2024, seed=2026)
    second = run_demo(image, "TEST", "blur", "moderate", key=2024, seed=2026)

    for key in (
        "recovered_text", "exact_recovery", "ecc_success", "bit_accuracy",
        "ber", "psnr", "ssim", "attack", "severity",
    ):
        assert first[key] == second[key]
    assert np.array_equal(first["watermarked_image"], second["watermarked_image"])
    assert np.array_equal(first["attacked_image"], second["attacked_image"])


@pytest.mark.parametrize("text", ["", "   ", "ABCDE", "ééé"])
def test_demo_rejects_empty_or_oversized_watermarks(image, text):
    with pytest.raises(ValueError, match="(?i)watermark"):
        run_demo(image, text)


def test_demo_rejects_invalid_image_data_and_unsupported_path(image, tmp_path):
    with pytest.raises(ValueError, match="8-bit"):
        run_demo(image.astype(np.float32), "TEST")
    with pytest.raises(ValueError, match="supported or readable image"):
        decode_uploaded_image(b"not an image")
    with pytest.raises(ValueError, match="Unsupported image format"):
        load_demo_image(tmp_path / "photo.gif")
