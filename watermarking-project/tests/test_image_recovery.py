"""Checks deterministic real-image Member C recovery measurements."""
import pandas as pd

from scripts.p8_image_recovery import run_experiment, write_results


def test_image_recovery_is_deterministic_and_writes_expected_csv(image, tmp_path):
    cases = [("none", "none"), ("blur", "light")]
    first = run_experiment(image, "fixture", cases, key=2024, seed=2026)
    second = run_experiment(image, "fixture", cases, key=2024, seed=2026)

    assert first == second
    assert len(first) == 2
    assert [(row["attack"], row["severity"]) for row in first] == cases
    expected_fields = {
        "image", "attack", "severity", "coded_bit_accuracy", "ber",
        "ecc_decode_ok", "exact_message_recovered", "decoded_text",
        "mean_vote_strength", "tied_votes", "embed_psnr",
    }
    assert expected_fields == set(first[0])
    assert first[0]["coded_bit_accuracy"] == 1.0
    assert first[0]["ecc_decode_ok"] is True
    assert first[0]["exact_message_recovered"] is True
    assert first[0]["decoded_text"] == "TEST"

    output = write_results(first, tmp_path / "image_recovery.csv")
    frame = pd.read_csv(output)
    assert len(frame) == 2
    assert expected_fields == set(frame.columns)
