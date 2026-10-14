"""Byte-symbol analysis tests for Member C's Reed-Solomon recovery path."""
from watermark import ecc
from watermark.config import ECC_PARITY_BYTES, PAYLOAD_BITS
from scripts.p9_symbol_error_analysis import analyze_image, symbol_error_profile


def test_symbol_error_profile_uses_ecc_byte_packing_and_counts_bits():
    original = ecc.bytes_to_bits(bytes((0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00)))
    extracted = ecc.bytes_to_bits(bytes((0x81, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00)))

    profile = symbol_error_profile(original, extracted)

    assert len(ecc.bits_to_bytes(original)) == PAYLOAD_BITS // 8 == 8
    assert ECC_PARITY_BYTES == 4
    assert profile["total_bit_errors"] == 3
    assert profile["erroneous_symbol_count"] == 2
    assert profile["erroneous_symbol_indices"] == "0,1"
    assert profile["bit_errors_per_erroneous_symbol"] == "0:2;1:1"
    assert profile["max_bit_errors_in_symbol"] == 2
    assert profile["symbol_error_distribution"] == '{"1": 1, "2": 1}'
    assert profile["error_pattern"] == "concentrated"
    assert profile["within_rs_symbol_capacity"] is True
    assert profile["original_rs_bytes_hex"] == "0000000000000000"
    assert profile["extracted_rs_bytes_hex"] == "8108000000000000"


def test_symbol_analysis_is_deterministic_and_returns_expected_columns(image):
    cases = [("none", "none"), ("blur", "moderate")]
    first = analyze_image(image, "fixture", cases, key=2024, seed=2026)
    second = analyze_image(image, "fixture", cases, key=2024, seed=2026)

    assert first == second
    assert len(first) == 2
    expected_columns = {
        "image", "attack", "severity", "coded_bit_accuracy", "ber",
        "ecc_decode_ok", "exact_message_recovered", "decoded_text",
        "mean_vote_strength", "tied_votes", "embed_psnr", "total_bit_errors",
        "erroneous_symbol_count", "erroneous_symbol_indices",
        "bit_errors_per_erroneous_symbol", "max_bit_errors_in_symbol",
        "symbol_error_distribution", "error_pattern",
        "within_rs_symbol_capacity", "original_rs_bytes_hex",
        "extracted_rs_bytes_hex",
    }
    assert expected_columns == set(first[0])
    assert first[0]["total_bit_errors"] == 0
    assert first[0]["ecc_decode_ok"] is True
