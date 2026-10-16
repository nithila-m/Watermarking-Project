"""Byte-symbol analysis tests for Member C's Reed-Solomon recovery path.

Validates the Phase 9 symbol error diagnostic module:
  - Byte packing and symbol error distribution profiling under known bit flips.
  - Verification of Reed-Solomon symbol capacity limits (up to 2 corrupted byte symbols).
  - Exact formatting of symbol error metrics, indices, distributions, and hex representations.
  - Determinism, schema conformance, and error-free baseline extraction across full image trials.
"""
from watermark import ecc
from watermark.config import ECC_PARITY_BYTES, PAYLOAD_BITS
from scripts.p9_symbol_error_analysis import analyze_image, symbol_error_profile


def test_symbol_error_profile_uses_ecc_byte_packing_and_counts_bits():
    """Verify byte-symbol error accounting, burst detection, and RS capacity checks.

    Constructs a controlled synthetic 8-byte (64-bit) codeword:
      - Byte 0 is corrupted with 2 bit flips: 0x00 -> 0x81 (bits 0 and 7 flipped).
      - Byte 1 is corrupted with 1 bit flip:  0x00 -> 0x08 (bit 3 flipped).
      - Bytes 2 through 7 remain pristine (0x00).

    Confirms that:
      1. Total bit errors equal 3 across the entire 64-bit payload.
      2. Exactly 2 byte symbols are marked erroneous (indices 0 and 1).
      3. Error pattern is classified as 'concentrated' due to multiple errors within a byte.
      4. The 2 erroneous symbols fall within the RS correction bound (floor(4 / 2) = 2 symbols).
      5. Formatted strings and hex byte representations match the canonical schema.
    """
    original = ecc.bytes_to_bits(bytes((0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00)))
    extracted = ecc.bytes_to_bits(bytes((0x81, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00)))

    profile = symbol_error_profile(original, extracted)

    # Payload consists of 8 bytes (64 bits), with 4 parity bytes allocating t=2 symbol corrections
    assert len(ecc.bits_to_bytes(original)) == PAYLOAD_BITS // 8 == 8
    assert ECC_PARITY_BYTES == 4

    # Bit error counts and erroneous symbol metrics
    assert profile["total_bit_errors"] == 3
    assert profile["erroneous_symbol_count"] == 2
    assert profile["erroneous_symbol_indices"] == "0,1"
    assert profile["bit_errors_per_erroneous_symbol"] == "0:2;1:1"
    assert profile["max_bit_errors_in_symbol"] == 2
    assert profile["symbol_error_distribution"] == '{"1": 1, "2": 1}'
    assert profile["error_pattern"] == "concentrated"

    # 2 corrupted bytes <= floor(ECC_PARITY_BYTES / 2) = 2 -> decodable
    assert profile["within_rs_symbol_capacity"] is True

    # Hex representation of transmitted vs extracted 8-byte codewords
    assert profile["original_rs_bytes_hex"] == "0000000000000000"
    assert profile["extracted_rs_bytes_hex"] == "8108000000000000"


def test_symbol_analysis_is_deterministic_and_returns_expected_columns(image):
    """Verify that image-level symbol error analysis is deterministic and schema-compliant.

    Args:
        image (np.ndarray): Pytest session fixture providing a 512x512 BGR test image.
    """
    cases = [("none", "none"), ("blur", "moderate")]
    # Execute two identical runs to verify deterministic repeatability
    first = analyze_image(image, "fixture", cases, key=2024, seed=2026)
    second = analyze_image(image, "fixture", cases, key=2024, seed=2026)

    assert first == second
    assert len(first) == 2

    # Comprehensive verification of output telemetry columns
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

    # Pristine condition assertion: zero bit errors and successful RS decoding
    assert first[0]["total_bit_errors"] == 0
    assert first[0]["ecc_decode_ok"] is True
