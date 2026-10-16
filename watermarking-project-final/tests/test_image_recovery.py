"""Checks deterministic real-image Member C recovery measurements.

Validates the Phase 8 image recovery pipeline:
  - Exact repeatability of embedding, distortion, blind extraction, and ECC decoding.
  - Correct schema adherence of returned recovery telemetry records.
  - Zero-error extraction and exact payload ("TEST") recovery under pristine conditions.
  - Accurate serialization and column retention when writing results to CSV.
"""
import pandas as pd

from scripts.p8_image_recovery import run_experiment, write_results


def test_image_recovery_is_deterministic_and_writes_expected_csv(image, tmp_path):
    """Verify that run_experiment produces identical results and valid CSV outputs.

    Args:
        image (np.ndarray): Pytest session fixture providing a 512x512 BGR test image.
        tmp_path (pathlib.Path): Pytest fixture providing a temporary filesystem directory.
    """
    cases = [("none", "none"), ("blur", "light")]
    # Execute two identical runs to verify end-to-end determinism
    first = run_experiment(image, "fixture", cases, key=2024, seed=2026)
    second = run_experiment(image, "fixture", cases, key=2024, seed=2026)

    assert first == second
    assert len(first) == 2
    assert [(row["attack"], row["severity"]) for row in first] == cases

    # Expected telemetry columns emitted per evaluation condition
    expected_fields = {
        "image", "attack", "severity", "coded_bit_accuracy", "ber",
        "ecc_decode_ok", "exact_message_recovered", "decoded_text",
        "mean_vote_strength", "tied_votes", "embed_psnr",
    }
    assert expected_fields == set(first[0])

    # Pristine baseline assertion: 100% bit accuracy and exact string recovery
    assert first[0]["coded_bit_accuracy"] == 1.0
    assert first[0]["ecc_decode_ok"] is True
    assert first[0]["exact_message_recovered"] is True
    assert first[0]["decoded_text"] == "TEST"

    # Verify CSV file output integrity and schema match
    output = write_results(first, tmp_path / "image_recovery.csv")
    frame = pd.read_csv(output)
    assert len(frame) == 2
    assert expected_fields == set(frame.columns)
